# src/phase6_cache/manager.py
#
# CacheManager: the 5-tier Redis cache for the Enterprise RAG pipeline.
#
# CACHE KEY CONSTRUCTION:
# Every key uses SHA-256 of the content as the identifier.
# This gives us:
#   - Determinism: same question → same key, always
#   - Collision resistance: different questions → different keys
#   - Fixed length: all keys are prefix + 64 hex chars
#   - Privacy: the cache key doesn't expose the question content
#
# KEY FORMAT: "{tier_prefix}:{sha256_hex}"
# Examples:
#   emb:a3f92b1c8d4e...      ← embedding for "What is OOMKilled?"
#   intent:7d9f8b4c2a1e...   ← intent for "Which pods are failing?"
#   ans:5c6d7e8f9a0b...      ← answer for "How do I fix CrashLoopBackOff?"
#
# SERIALISATION:
# Redis stores strings. Different tiers need different serialisation:
#   embeddings:   json.dumps(list[float])    ~8KB per vector (1536 × 4 bytes)
#   intent:       str ("rag", "sql", "hybrid")
#   sql_gen:      str (SQL query text)
#   sql_result:   str (formatted table text)
#   answer:       str (LLM answer text)
#
# CACHE HIT RATE EXPECTATIONS (K8s ops steady state):
#   answer cache:     60-80% — SREs ask the same questions repeatedly
#   intent cache:     70-85% — question phrasing is consistent
#   embedding cache:  75-90% — exact same question text
#   sql_gen cache:    65-80% — same operational questions
#   sql_result cache: 30-50% — cluster state changes frequently

import hashlib
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── TTLs (seconds) ─────────────────────────────────────────────────────────────
TTL_EMBEDDING  = 7  * 24 * 3_600   # 7 days   — embedding model won't change
TTL_INTENT     = 24 * 3_600        # 24 hours  — intent is stable
TTL_SQL_GEN    = 24 * 3_600        # 24 hours  — SQL structure is stable
TTL_SQL_RESULT = 15 * 60           # 15 minutes — live cluster data, tolerate some staleness
TTL_ANSWER     = 1  * 3_600        # 1 hour    — doc answers stable; SQL answers refresh with SQL result


# ── Cache key prefixes ─────────────────────────────────────────────────────────
PREFIX_EMBEDDING  = "emb"
PREFIX_INTENT     = "intent"
PREFIX_SQL_GEN    = "sqlgen"
PREFIX_SQL_RESULT = "sqlres"
PREFIX_ANSWER     = "ans"


def _make_key(prefix: str, content: str) -> str:
    """Build a Redis cache key: prefix:SHA256(content)."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


class CacheManager:
    """
    5-tier Redis cache manager.

    All operations are safe to call even if Redis is unavailable.
    A None client causes all gets to return None and all sets to be no-ops.

    USAGE PATTERN (cache-aside):
        value = cache.get_answer(question)
        if value is not None:
            return value   # cache hit
        value = expensive_compute()
        cache.set_answer(question, value)
        return value

    SINGLETON:
        Use get_cache_manager() to get the shared instance.
        Building it directly is fine for tests.
    """

    def __init__(self, client):
        """
        Args:
            client: redis.Redis client (or None for no-op mode)
        """
        self._r = client
        self._hits  = 0
        self._misses = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get(self, key: str) -> Optional[str]:
        """Get a raw string value from Redis. Returns None on miss or error."""
        if self._r is None:
            return None
        try:
            return self._r.get(key)
        except Exception as e:
            logger.warning(f"Redis GET error for key '{key[:20]}...': {e}")
            return None

    def _set(self, key: str, value: str, ttl: int) -> None:
        """Set a raw string value in Redis with TTL. Silent on error."""
        if self._r is None:
            return
        try:
            self._r.setex(key, ttl, value)
        except Exception as e:
            logger.warning(f"Redis SET error for key '{key[:20]}...': {e}")

    def _log_hit(self, tier: str, key_preview: str) -> None:
        self._hits += 1
        logger.info(f"Cache HIT  [{tier}] key={key_preview}")

    def _log_miss(self, tier: str, key_preview: str) -> None:
        self._misses += 1
        logger.debug(f"Cache MISS [{tier}] key={key_preview}")

    # ── Tier 1: Embedding cache ────────────────────────────────────────────────

    def get_embedding(self, question: str) -> Optional[list[float]]:
        """
        Retrieve a cached embedding vector for a question.

        WHY 7d TTL:
        Embeddings are deterministic given the model and input text.
        They only change if we upgrade the embedding model (text-embedding-3-small).
        7 days is conservative — in practice, cached forever until model change.

        Returns: 1536-dim float list, or None on miss.
        """
        key = _make_key(PREFIX_EMBEDDING, question)
        raw = self._get(key)
        if raw is None:
            self._log_miss("embedding", key[:16])
            return None
        self._log_hit("embedding", key[:16])
        return json.loads(raw)

    def set_embedding(self, question: str, embedding: list[float]) -> None:
        key = _make_key(PREFIX_EMBEDDING, question)
        self._set(key, json.dumps(embedding), TTL_EMBEDDING)

    # ── Tier 2: Intent cache ───────────────────────────────────────────────────

    def get_intent(self, question: str) -> Optional[str]:
        """
        Retrieve cached intent classification for a question.

        WHY 24h TTL:
        "How many pods are failing right now?" will always be "sql".
        "What does OOMKilled mean?" will always be "rag".
        Question intent doesn't change day to day.

        Returns: "rag", "sql", "hybrid", or None on miss.
        """
        key = _make_key(PREFIX_INTENT, question)
        val = self._get(key)
        if val is None:
            self._log_miss("intent", key[:16])
            return None
        self._log_hit("intent", key[:16])
        return val

    def set_intent(self, question: str, intent: str) -> None:
        key = _make_key(PREFIX_INTENT, question)
        self._set(key, intent, TTL_INTENT)

    # ── Tier 3: SQL generation cache ──────────────────────────────────────────

    def get_sql(self, question: str) -> Optional[str]:
        """
        Retrieve cached generated SQL for a question.

        WHY 24h TTL:
        "How many pods are in CrashLoopBackOff?" always generates the same SELECT.
        The SQL structure is stable even if the data changes.
        The SQL result cache (Tier 4) handles data freshness separately.

        Note: SQL questions still require HITL approval even on cache hit.
        We don't bypass human review just because we cached the SQL.

        Returns: SQL query string, or None on miss.
        """
        key = _make_key(PREFIX_SQL_GEN, question)
        val = self._get(key)
        if val is None:
            self._log_miss("sql_gen", key[:16])
            return None
        self._log_hit("sql_gen", key[:16])
        return val

    def set_sql(self, question: str, sql: str) -> None:
        key = _make_key(PREFIX_SQL_GEN, question)
        self._set(key, sql, TTL_SQL_GEN)

    # ── Tier 4: SQL result cache ───────────────────────────────────────────────

    def get_sql_result(self, sql: str) -> Optional[str]:
        """
        Retrieve cached SQL execution results.

        WHY 15m TTL:
        Live cluster data changes frequently (pods crash, recover, restart).
        15 minutes is the acceptable staleness window for ops queries.
        SREs tolerate seeing "as of 15 minutes ago" — it's still useful.
        For P1 incidents: the short TTL means they'll usually get fresh data.

        KEY: SHA256 of the SQL query (not the question).
        Same SQL from different phrasings ("failing pods" vs "broken pods")
        will share the same result cache if they generate identical SQL.

        Returns: formatted table string, or None on miss.
        """
        key = _make_key(PREFIX_SQL_RESULT, sql)
        val = self._get(key)
        if val is None:
            self._log_miss("sql_result", key[:16])
            return None
        self._log_hit("sql_result", key[:16])
        return val

    def set_sql_result(self, sql: str, result: str) -> None:
        key = _make_key(PREFIX_SQL_RESULT, sql)
        self._set(key, result, TTL_SQL_RESULT)

    # ── Tier 5: Answer cache ───────────────────────────────────────────────────

    def get_answer(self, question: str) -> Optional[str]:
        """
        Retrieve cached final LLM answer for a question.

        This is the highest-value cache tier.
        A hit here bypasses: intent routing + embedding + retrieval +
        reranking + CRAG grading + LLM generation.
        Total savings: ~4,500ms and ~$0.01 per query.

        WHY 1h TTL:
        RAG answers (from static runbooks): stable for hours.
        SQL answers (from live data): the underlying SQL result cache
        expires in 15m anyway, so the answer cache for SQL questions
        effectively has a 15m staleness window through the dependency chain.

        In practice: cache the answer for 1 hour. If the underlying
        SQL result changes before 1h, the cached answer becomes slightly
        stale. For ops questions, this is acceptable — SREs can always
        add "right now" to force a fresh SQL query.

        Returns: answer string, or None on miss.
        """
        key = _make_key(PREFIX_ANSWER, question)
        val = self._get(key)
        if val is None:
            self._log_miss("answer", key[:16])
            return None
        self._log_hit("answer", key[:16])
        return val

    def set_answer(self, question: str, answer: str) -> None:
        key = _make_key(PREFIX_ANSWER, question)
        self._set(key, answer, TTL_ANSWER)

    # ── Cache statistics ────────────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits":     self._hits,
            "misses":   self._misses,
            "total":    total,
            "hit_rate": f"{self.hit_rate:.1%}",
            "enabled":  self._r is not None,
        }

    def flush_all(self) -> bool:
        """
        Clear all cached data. USE WITH CAUTION in production.
        Safe to call in development and testing.
        """
        if self._r is None:
            return False
        try:
            self._r.flushdb()
            self._hits = self._misses = 0
            logger.info("Cache flushed")
            return True
        except Exception as e:
            logger.error(f"Cache flush failed: {e}")
            return False

    def flush_tier(self, tier: str) -> int:
        """
        Delete all keys for one cache tier.

        Args:
            tier: "embedding", "intent", "sql_gen", "sql_result", "answer"

        Returns:
            Number of keys deleted
        """
        prefix_map = {
            "embedding":  PREFIX_EMBEDDING,
            "intent":     PREFIX_INTENT,
            "sql_gen":    PREFIX_SQL_GEN,
            "sql_result": PREFIX_SQL_RESULT,
            "answer":     PREFIX_ANSWER,
        }
        prefix = prefix_map.get(tier)
        if not prefix or self._r is None:
            return 0
        try:
            pattern = f"{prefix}:*"
            keys    = self._r.keys(pattern)
            if keys:
                self._r.delete(*keys)
            logger.info(f"Flushed {len(keys)} keys from tier '{tier}'")
            return len(keys)
        except Exception as e:
            logger.error(f"Tier flush failed: {e}")
            return 0


# ── Singleton ────────────────────────────────────────────────────────────────

from functools import lru_cache

@lru_cache(maxsize=1)
def get_cache_manager() -> CacheManager:
    """Return the shared CacheManager singleton."""
    from src.phase6_cache.client import get_redis_client
    client = get_redis_client()
    return CacheManager(client)

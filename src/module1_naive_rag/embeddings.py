# src/module1_naive_rag/embeddings.py
#
# Everything related to converting text -> vector.
#
# RESPONSIBILITIES:
#   - Single text embedding (embed_text)
#   - Batch embedding with automatic batching (embed_texts_batched)
#   - Retry logic for OpenAI rate limits (429) and server errors (500)
#   - Cost and token tracking
#   - Cache key generation (SHA-256) -- used by Phase 6 Redis cache
#
# PHASE EVOLUTION:
# Module 1: Synchronous single/batch embedding with retry
# Phase 3:  Add HyDE -- embed(question + 3 hypothetical answers) -> 4 vectors
# Phase 6:  Add Redis cache check before every API call
# Phase 10: Switch to AsyncOpenAI for FastAPI async endpoints

import hashlib
import logging
import time
from dataclasses import dataclass

from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from src.config import settings

logger = logging.getLogger(__name__)

# Module-level OpenAI client (lazy singleton)
_openai_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """
    Lazy singleton OpenAI client.
    Created on first call, reused on subsequent calls.
    Thread-safe: the OpenAI client handles connection pooling internally.
    """
    global _openai_client
    if _openai_client is None:
        # Reads OPENAI_API_KEY from environment automatically.
        # Raises openai.AuthenticationError if the key is missing or invalid.
        _openai_client = OpenAI()
    return _openai_client


@dataclass
class EmbeddingUsage:
    """
    Token and cost tracking for a single embed API call.
    We log this per batch to monitor spend over time.
    In Phase 9 (Observability) these metrics go to Prometheus.

    Pricing (as of 2024):
        text-embedding-3-small: $0.020 per million tokens
        text-embedding-3-large: $0.130 per million tokens
    """
    model: str
    input_texts: int
    total_tokens: int
    estimated_cost_usd: float

    COST_PER_TOKEN = {
        "text-embedding-3-small": 0.020 / 1_000_000,
        "text-embedding-3-large": 0.130 / 1_000_000,
        "text-embedding-ada-002":  0.100 / 1_000_000,
    }

    @classmethod
    def from_response(cls, response, model: str, n_texts: int) -> "EmbeddingUsage":
        tokens = response.usage.total_tokens
        cost = tokens * cls.COST_PER_TOKEN.get(model, 0.020 / 1_000_000)
        return cls(model=model, input_texts=n_texts,
                   total_tokens=tokens, estimated_cost_usd=cost)


# Retry decorator applied to the raw API call.
# Retries on: 429 (rate limit), 500 (server error), network errors.
# Waits: 1s -> 2s -> 4s (exponential, capped at 10s).
# Gives up after 3 total attempts and re-raises the last exception.
_embed_retry = retry(
    retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@_embed_retry
def _call_embed_api(
    texts: list[str],
    model: str,
) -> tuple[list[list[float]], EmbeddingUsage]:
    """
    Internal: raw batched OpenAI embedding API call with retry.

    Why separate from the public functions?
    The retry decorator only wraps the actual API call, not the
    surrounding logic (batching, logging, cache checks). Keeping them
    separate makes each layer's responsibility explicit.

    Args:
        texts: 1 to 100 strings to embed in one API call
        model: OpenAI model name

    Returns:
        (embeddings, usage)
        embeddings: list[list[float]], one 1536-dim vector per input text
        usage:      cost and token tracking
    """
    client = _get_client()
    response = client.embeddings.create(model=model, input=texts)

    # Sort by .index to guarantee order matches input order.
    # OpenAI guarantees order but we enforce it explicitly.
    # response.data: list[Embedding]
    # Embedding.index: int (position in input list)
    # Embedding.embedding: list[float] (1536 L2-normalized floats)
    sorted_embeddings = sorted(response.data, key=lambda e: e.index)
    vectors = [e.embedding for e in sorted_embeddings]

    usage = EmbeddingUsage.from_response(response, model=model, n_texts=len(texts))
    return vectors, usage


def embed_text(text: str, model: str | None = None) -> list[float]:
    """
    Embed a single text string into a 1536-dimensional float vector.

    Called at query time by retrieval.py to embed the user's question.
    Also called during ingestion for individual chunks (but use
    embed_texts_batched() for large volumes -- it's 100x faster).

    Args:
        text:  String to embed. Max ~8,191 tokens for text-embedding-3-small.
        model: Override model (defaults to settings.embedding_model).

    Returns:
        list[float] of length 1536, L2-normalized (magnitude = 1.0).
        Cosine similarity with this vector = dot product (cheaper to compute).

    LATENCY:  ~80-120ms (OpenAI API round-trip)
    COST:     ~$0.0000003 per call (negligible)

    PHASE 6 UPGRADE:
    Will check Redis cache first (key = SHA-256(text), TTL=7d).
    Cache hit returns in ~1ms instead of ~100ms.
    """
    model = model or settings.embedding_model
    vectors, usage = _call_embed_api([text], model)
    logger.debug(
        f"embed_text | {usage.total_tokens} tokens | ${usage.estimated_cost_usd:.7f}"
    )
    return vectors[0]


def embed_texts_batched(
    texts: list[str],
    model: str | None = None,
    batch_size: int | None = None,
) -> list[list[float]]:
    """
    Embed a large list of texts using batched API calls.

    Performance comparison for 1,000 texts:
        Sequential (1 text/call): 1,000 calls x 100ms = ~100 seconds
        Batched (100 texts/call): 10 calls  x 100ms = ~1 second

    Args:
        texts:      List of strings to embed. Any length.
        model:      Override model.
        batch_size: Texts per API call. Default: settings.embed_batch_size (100).
                    Hard limit: 2048 (OpenAI API maximum).

    Returns:
        list[list[float]] -- same order as input, same length as input.
        Each embedding is 1536-dim, L2-normalized.

    COST EXAMPLE (10,000 K8s runbook chunks):
        10,000 chunks x 300 tokens avg = 3,000,000 tokens
        x $0.020 / 1,000,000 = $0.060 total ingestion embedding cost
    """
    if not texts:
        return []

    model = model or settings.embedding_model
    batch_size = batch_size or settings.embed_batch_size

    all_vectors: list[list[float]] = []
    total_tokens = 0
    total_cost = 0.0
    n_batches = (len(texts) + batch_size - 1) // batch_size

    logger.info(
        f"Embedding {len(texts)} texts | "
        f"{n_batches} batches of {batch_size} | model={model}"
    )

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"  Batch {batch_num}/{n_batches}: {len(batch)} texts...")

        vectors, usage = _call_embed_api(batch, model)
        all_vectors.extend(vectors)
        total_tokens += usage.total_tokens
        total_cost += usage.estimated_cost_usd

        # Brief pause between batches to stay under token-per-minute rate limits.
        # text-embedding-3-small: 1,000,000 tokens/minute limit.
        # 100 texts x 300 tokens = 30,000 tokens per batch.
        # At 0.1s sleep: ~100 batches/10s = 300,000 tokens/10s = well under limit.
        if i + batch_size < len(texts):
            time.sleep(0.1)

    logger.info(
        f"Embedding complete | "
        f"{len(texts)} texts | {total_tokens:,} tokens | ${total_cost:.4f}"
    )
    return all_vectors


def make_cache_key(text: str) -> str:
    """
    Generate a deterministic, compact cache key for any text.

    SHA-256(text) -> 64-character hex string.
    Same text always produces the same key.
    Different texts produce different keys (collision probability: 1/2^256).

    Used as:
    - Redis key prefix for embedding cache (Phase 6): "emb:{key}"
    - Redis key prefix for answer cache (Phase 6):    "ans:{key}"
    - Qdrant payload field for deduplication (Phase 6): "content_hash"
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

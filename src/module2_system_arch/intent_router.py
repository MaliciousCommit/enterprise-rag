# src/module2_system_arch/intent_router.py
#
# Intent Router: classifies every question into one of three pipelines.
#
# CLASSIFICATION TARGETS:
#   "rag"    → needs document knowledge (runbooks, guides, explanations)
#   "sql"    → needs live structured data from the cluster database
#   "hybrid" → needs BOTH (live data + document knowledge)
#
# IMPLEMENTATION CHOICE: LLM-based (not rule-based)
#
# Alternative: rule-based classifier using regex patterns:
#   if "right now" in question or "currently" in question: return "sql"
#   if "how many" in question: return "sql"
#   else: return "rag"
#
# WHY WE CHOSE LLM OVER REGEX:
#   Regex fails on natural language variation:
#   "What pods are having issues?" → should be "sql" (live data)
#   but contains no "right now" or "how many"
#
#   "What is currently the recommended memory limit?" → should be "rag"
#   but contains "currently" which regex would classify as "sql"
#
#   LLM understands intent, not surface patterns.
#   Tradeoff: ~300ms latency, ~$0.0001 cost per classification.
#   Mitigated by Redis cache (TTL=24h) — most questions repeat.
#
# MODEL CHOICE: gpt-4o-mini (not gpt-4o)
#   Classification is simple — gpt-4o-mini is 15x cheaper and
#   nearly identical quality for this binary/ternary task.
#   temperature=0.0 for full determinism.
#   max_tokens=5 — we only need one word back.
#
# PHASE EVOLUTION:
#   Module 2:  basic LLM classification (this file)
#   Phase 3:   add Kubernetes-specific few-shot examples
#   Phase 6:   Redis cache wrapper around classify_intent()
#   Phase 9:   log classification latency + accuracy to Prometheus

import logging
from enum import Enum

from openai import AsyncOpenAI

from src.module2_system_arch.state import RAGState

logger = logging.getLogger(__name__)

# ── Model configuration ────────────────────────────────────────────────────────
CLASSIFICATION_MODEL    = "gpt-4o-mini"
CLASSIFICATION_TEMP     = 0.0     # fully deterministic — same Q always → same intent
CLASSIFICATION_MAX_TOK  = 5       # "rag" = 1 token, "sql" = 1 token, "hybrid" = 2 tokens


# ── Intent enum ───────────────────────────────────────────────────────────────

class Intent(str, Enum):
    """
    The three routing targets in our system.

    Using an Enum (not bare strings) provides:
    - Autocomplete in IDEs
    - Type safety — can't accidentally pass "RAG" instead of "rag"
    - Single source of truth for valid values
    """
    RAG    = "rag"
    SQL    = "sql"
    HYBRID = "hybrid"


# ── Classification prompt ─────────────────────────────────────────────────────
#
# PROMPT ENGINEERING DECISIONS:
#
# 1. Few-shot examples are the most important part.
#    The model generalises from these examples to new questions.
#    Examples are chosen to cover edge cases (ambiguous questions).
#
# 2. Signal descriptions tell the model WHAT TO LOOK FOR.
#    This is more reliable than examples alone.
#
# 3. "Reply with ONLY one word" is critical.
#    Without this, the model adds explanation: "The answer is sql because..."
#    which breaks our response parsing.
#
# 4. We deliberately do NOT say "Kubernetes" in the signal descriptions.
#    The model should classify based on question structure, not domain.
#    This makes the router reusable for other domains.

INTENT_SYSTEM_PROMPT = """
You are a query router for an IT operations question-answering system.

Classify the user's question into exactly ONE of three categories.

RAG — needs document knowledge
  Signals: "what is", "how do I", "why does", "what does X mean",
  "explain", "what should I do", procedures, configurations, concepts
  Examples:
    "What does OOMKilled mean?" → rag
    "How do I fix a CrashLoopBackOff?" → rag
    "What is the difference between requests and limits?" → rag
    "What does the runbook say about node NotReady?" → rag
    "How do I roll back a deployment?" → rag

SQL — needs live structured data from the database
  Signals: "right now", "currently", "today", "how many", "list all",
  "which pods", "what is the status of", specific names, metric values,
  counts, aggregations, time-filtered queries
  Examples:
    "How many pods are in CrashLoopBackOff?" → sql
    "What is the CPU usage of payment-service right now?" → sql
    "List all P1 incidents from last week" → sql
    "Which nodes have memory pressure?" → sql
    "How many restarts has auth-service had today?" → sql

HYBRID — needs BOTH document knowledge AND live data
  Signals: question asks about current state AND what to do about it
  Examples:
    "Which pods are failing and what should I do to fix them?" → hybrid
    "My payment-service is crashing, what does the runbook say?" → hybrid
    "Show me all OOMKilled pods and how to prevent it" → hybrid

Reply with ONLY one word: rag, sql, or hybrid
No explanation. No punctuation. Just the single word.
""".strip()


# ── Core classification function ──────────────────────────────────────────────

async def classify_intent(question: str) -> Intent:
    """
    Classify a natural language question into rag, sql, or hybrid.

    Makes a single LLM API call with:
    - gpt-4o-mini: fast + cheap for classification tasks
    - temperature=0.0: fully deterministic, same question = same result
    - max_tokens=5: we only expect one word back

    LATENCY: ~150-300ms (network RTT to OpenAI + fast mini model)
    COST:    ~$0.0001 per classification
    After Phase 6 Redis cache: ~2ms on cache hit, $0 cost

    FALLBACK STRATEGY:
    If the LLM returns an unexpected value (not rag/sql/hybrid),
    we fall back to "rag" — document retrieval is always safe.
    SQL execution without validation is dangerous.
    Defaulting to "rag" is the conservative choice.

    Args:
        question: The user's natural language question

    Returns:
        Intent enum value: Intent.RAG, Intent.SQL, or Intent.HYBRID
    """
    client = AsyncOpenAI()

    logger.debug(f"Classifying intent: '{question[:60]}...'")

    response = await client.chat.completions.create(
        model=CLASSIFICATION_MODEL,
        temperature=CLASSIFICATION_TEMP,
        max_tokens=CLASSIFICATION_MAX_TOK,
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
    )

    raw = response.choices[0].message.content.strip().lower()

    # Map raw LLM output to Intent enum
    # If LLM returns something unexpected, fall back to RAG (safe default)
    intent_map = {
        "rag":    Intent.RAG,
        "sql":    Intent.SQL,
        "hybrid": Intent.HYBRID,
    }
    intent = intent_map.get(raw, Intent.RAG)

    if raw not in intent_map:
        logger.warning(
            f"Unexpected intent classification: '{raw}' for question "
            f"'{question[:40]}...'. Defaulting to 'rag'."
        )

    # Log token usage for observability (Phase 9 pipes to Prometheus)
    usage = response.usage
    logger.info(
        f"Intent: '{intent.value}' | "
        f"Question: '{question[:40]}...' | "
        f"Tokens: {usage.total_tokens} | "
        f"Model: {CLASSIFICATION_MODEL}"
    )

    return intent


# ── LangGraph node ────────────────────────────────────────────────────────────

async def intent_router_node(state: RAGState) -> dict:
    """
    LangGraph node: classifies the question and updates state["intent"].

    This is the FIRST node the graph executes. Every request passes
    through here before being routed to a pipeline.

    NODE CONTRACT:
      Reads:   state["question"]
      Writes:  {"intent": str, "prompt_tokens": int}
      Returns: partial state update dict

    LangGraph MERGES the returned dict into the current state.
    All other fields (context, scores, answer, etc.) are untouched.

    PHASE EVOLUTION:
    Module 2 (now): LLM classification, no cache
    Phase 6:        Check Redis before LLM call
                    if cached: return cached intent (2ms, free)
                    if miss: classify + cache with TTL=24h
    """
    question = state["question"]

    # ── Phase 6: Check intent cache ───────────────────────────────────────────
    # Cache hit: return immediately, skip the ~300ms LLM classification call.
    # Cache miss: classify with LLM, then store result for 24h.
    try:
        from src.phase6_cache.manager import get_cache_manager
        cache = get_cache_manager()
        cached_intent = cache.get_intent(question)
        if cached_intent:
            return {"intent": cached_intent, "cache_hit": False}  # cache_hit=False: answer not cached yet
    except Exception:
        pass  # Redis down — proceed without cache

    intent = await classify_intent(question)

    # Store result for next 24 hours
    try:
        cache.set_intent(question, intent.value)
    except Exception:
        pass

    return {
        "intent": intent.value,
    }


# ── Routing function (used by graph conditional edges) ────────────────────────

def route_by_intent(state: RAGState) -> str:
    """
    Conditional edge routing function.

    LangGraph calls this after intent_router_node completes.
    The return value is a KEY in the path_map passed to
    add_conditional_edges().

    Returns a descriptive label (not a node name directly).
    The graph maps labels → node names in the path_map.

    Routing logic:
      "rag"    → retrieve first (vector search), then generate
      "sql"    → SQL pipeline (Phase 5, not yet implemented)
                 For now: graceful fallback to RAG with a note
      "hybrid" → retrieve first (need context), then generate
                 In Phase 5: also queries SQL and merges results

    IMPORTANT: The return values here must EXACTLY match the keys
    in the path_map dict passed to graph.add_conditional_edges().
    A typo here causes a silent routing failure (graph hangs).

    Args:
        state: Current RAGState after intent_router_node has run

    Returns:
        str: Routing key ("rag_retrieve", "sql_pipeline", "hybrid_retrieve")
    """
    intent = state.get("intent", "rag")

    routing_map = {
        "rag":    "rag_retrieve",      # → retrieve_node
        "sql":    "sql_pipeline",       # → sql_not_implemented_node (Phase 5)
        "hybrid": "hybrid_retrieve",    # → retrieve_node (same path for now)
    }

    route = routing_map.get(intent, "rag_retrieve")

    logger.info(f"Routing intent='{intent}' → '{route}'")
    return route

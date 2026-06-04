# src/module2_system_arch/state.py
#
# RAGState: the typed state dictionary that flows through every
# node in the LangGraph state machine.
#
# DESIGN PRINCIPLE: Design for the full system, not just today's phase.
#
# Every field used by Phase 4 (CRAG, Self-RAG), Phase 5 (Text2SQL),
# and Phase 6 (caching) is declared HERE, NOW — even if some fields
# are unused in Module 2.
#
# Why? LangGraph checkpoints the FULL state to PostgreSQL after every
# node. If we add a field in Phase 5, old conversation checkpoints
# won't have that field → deserialization fails.
# Designing the schema upfront avoids this breaking change.
#
# HOW STATE FLOWS THROUGH THE GRAPH:
#
# graph.ainvoke(initial_state)
#   ↓
# intent_router_node(state) → returns {"intent": "rag"}
#   ↓ LangGraph MERGES: state["intent"] = "rag"
# retrieve_node(state) → returns {"context": [...], "scores": [...]}
#   ↓ LangGraph MERGES: state["context"] = [...], state["scores"] = [...]
# generate_node(state) → returns {"answer": "...", "iteration": 1}
#   ↓ LangGraph MERGES: state["answer"] = "...", state["iteration"] = 1
# END → final state returned to caller
#
# KEY: Each node returns a PARTIAL dict. LangGraph merges it into
# the current state. Unmentioned fields are preserved unchanged.
# This is why a node that only changes "answer" doesn't need to
# explicitly carry "context" forward — LangGraph handles it.

from typing import TypedDict


class RAGState(TypedDict):
    """
    Complete typed state for the Enterprise RAG state machine.

    All fields must have defaults handled at graph.ainvoke() call time.
    TypedDict fields without Optional still need to be provided in
    the initial state dict when invoking the graph.

    Phase ownership:
      Module 2:  question, session_id, intent, context, scores,
                 sources, answer, iteration
      Phase 4:   iteration (Self-RAG loop counter)
      Phase 5:   sql_query, sql_approved, sql_result
      Phase 6:   cache_hit (tracks whether answer came from cache)
      Phase 9:   prompt_tokens, completion_tokens, latency_ms
    """

    # ── Core input ─────────────────────────────────────────────────────
    question: str
    # The original user question. NEVER modified after initial entry.
    # All nodes read this; none of them should mutate it.
    # Immutable once set = safe to use as Redis cache key.

    session_id: str
    # Links this invocation to a conversation thread.
    # LangGraph uses this as thread_id in checkpoint config:
    #   config = {"configurable": {"thread_id": state["session_id"]}}
    # Same session_id across multiple turns = conversation memory.

    # ── Intent routing ─────────────────────────────────────────────────
    intent: str
    # Classification result from intent_router_node.
    # One of: "rag" | "sql" | "hybrid"
    # Used by conditional edges to route to the correct pipeline.
    # Cached in Redis TTL=24h (same question → same intent tomorrow).

    # ── RAG pipeline outputs ───────────────────────────────────────────
    context: list[str]
    # Retrieved text chunks from Qdrant.
    # Set by retrieve_node. Read by generate_node.
    # In Phase 3: also populated by Tavily fallback (CRAG).
    # In Phase 8: chunks are XML-spotlighted before being stored here.

    scores: list[float]
    # Cosine similarity scores from Qdrant, one per chunk in context.
    # Parallel to context: scores[i] is the score for context[i].
    # Used by CRAG grader (Phase 4):
    #   if max(scores) < 0.7: trigger Tavily web search fallback
    # Used by monitoring (Phase 9): track retrieval quality over time.
    # Module 2: stored but not acted upon.

    sources: list[str]
    # Source document identifiers for the retrieved chunks.
    # Used in: citations in the answer, audit_log, API response.
    # Example: ["runbooks/oomkilled.md", "guides/resource-limits.md"]

    # ── Generation output ──────────────────────────────────────────────
    answer: str
    # The LLM-generated answer to the question.
    # Set by generate_node. May be overwritten multiple times in
    # Self-RAG (Phase 4) if quality score < 0.8.
    # The FINAL value of this field is what gets returned to the user.

    iteration: int
    # How many times generate_node has been called for this question.
    # Starts at 0. Incremented by generate_node on each call.
    # Self-RAG (Phase 4) uses this as a loop guard:
    #   if iteration >= 2: stop regenerating, accept current answer.
    # Module 2: always ends at 1 (single generation, no loop).

    # ── Text2SQL pipeline (Phase 5) ────────────────────────────────────
    sql_query: str | None
    # The LLM-generated SQL query string.
    # Set by sql_generate_node (Phase 5).
    # None for all RAG-only questions.
    # Validated by sql_validate_node before any execution.

    sql_approved: bool | None
    # Whether a human approved the SQL query via the HITL UI.
    # Set by the interrupt() resume mechanism (Phase 5).
    # None until the human acts. True = execute. False = abort.

    sql_result: str | None
    # Formatted string of SQL query results.
    # Set by sql_execute_node (Phase 5).
    # Passed to generate_node as additional context for hybrid queries.

    # ── Observability (Phase 9) ────────────────────────────────────────
    prompt_tokens: int
    # Total input tokens consumed by all LLM calls for this question.
    # Accumulated across multiple calls (intent router + generation).
    # Used for: cost tracking, per-user token budget enforcement.

    completion_tokens: int
    # Total output tokens generated by all LLM calls.
    # Used for cost tracking (output tokens are 3x more expensive).

    latency_ms: float
    # End-to-end latency for this question in milliseconds.
    # Set by the pipeline entry point after full execution.
    # Tracked in Prometheus histogram (Phase 9).

    # ── Cache metadata (Phase 6) ───────────────────────────────────────
    cache_hit: bool
    # True if the answer was served from Redis cache.
    # False if the full pipeline ran.
    # Used for monitoring: track cache hit rate over time.


def initial_state(question: str, session_id: str) -> RAGState:
    """
    Create the initial RAGState for a new question.

    Called at the FastAPI endpoint before invoking the graph.
    All fields must have a defined initial value — TypedDict
    does not support runtime defaults.

    Args:
        question:   The user's question string
        session_id: Conversation session identifier

    Returns:
        RAGState dict with all fields initialised.
    """
    return RAGState(
        # Input
        question=question,
        session_id=session_id,

        # Intent (set by intent_router_node)
        intent="",

        # RAG outputs (set by retrieve_node and generate_node)
        context=[],
        scores=[],
        sources=[],
        answer="",
        iteration=0,

        # Text2SQL (set by Phase 5 nodes)
        sql_query=None,
        sql_approved=None,
        sql_result=None,

        # Observability (accumulated through execution)
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=0.0,

        # Cache
        cache_hit=False,
    )


# ── Module 4 upgrade: add conversation_history to RAGState ──────────────────
#
# We cannot add fields to a TypedDict after the fact without updating the class.
# In production: use Pydantic BaseModel instead (supports field addition more cleanly).
# For our curriculum: we'll use a helper that carries history forward.
#
# The conversation history is NOT part of RAGState TypedDict to avoid
# breaking existing checkpoints. Instead, we store it separately per session
# using MemorySaver's thread state and pass it into the graph via a wrapper.

def update_history(history: list[dict], question: str, answer: str) -> list[dict]:
    """
    Append the latest Q&A pair to the conversation history.
    Returns a new list (immutable pattern — avoids mutation bugs).

    Args:
        history:  Existing conversation history
        question: The user's question
        answer:   The system's answer

    Returns:
        New history list with two new entries appended.
    """
    return history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]


def truncate_history(history: list[dict], max_turns: int = 5) -> list[dict]:
    """
    Keep only the last max_turns conversation turns.
    Each turn = 2 messages (user + assistant).
    Prevents context window overflow on long conversations.

    Args:
        history:   Full conversation history
        max_turns: Number of recent turns to keep (default 5)

    Returns:
        Truncated history keeping the most recent turns.
    """
    max_messages = max_turns * 2
    return history[-max_messages:] if len(history) > max_messages else history

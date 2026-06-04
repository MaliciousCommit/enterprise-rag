# src/module2_system_arch/nodes.py
#
# LangGraph node definitions for Module 2.
#
# These nodes WRAP the Module 1 functions (retrieve, generate) in
# LangGraph's state-machine contract:
#   INPUT:  full RAGState dict
#   OUTPUT: partial dict of ONLY the fields that changed
#
# WHY NOT JUST USE MODULE 1 FUNCTIONS DIRECTLY?
# Module 1 functions (retrieve, generate) are plain Python:
#   chunks = retrieve(client, question, k=5)
#   result = generate(question, chunks)
#
# LangGraph nodes must:
#   1. Read from typed RAGState dict
#   2. Return a PARTIAL dict (not the full state)
#   3. Be async (for non-blocking execution)
#   4. Handle their own errors gracefully (not crash the graph)
#
# These wrappers translate between the two contracts.
#
# PHASE EVOLUTION:
# Module 2: thin wrappers around Module 1 functions (this file)
# Phase 3:  retrieve_node gets HyDE, hybrid search, RRF, reranking
# Phase 4:  add crag_grader_node and self_rag_reflect_node
# Phase 5:  add sql_generate_node, sql_validate_node, sql_execute_node
#
# The wrapper pattern means Phase 3 changes ONLY retrieve_node —
# generate_node and the graph structure stay identical.

import logging
from collections import defaultdict

from qdrant_client import QdrantClient

from src.config import settings
from src.module1_naive_rag.collection import get_qdrant_client
from src.module1_naive_rag.retrieval import retrieve, RetrievedChunk
from src.module1_naive_rag.generation import generate
from src.module2_system_arch.state import RAGState

logger = logging.getLogger(__name__)

# ── In-memory conversation history store (Module 4) ───────────────────────────
# Maps session_id → list of {"role": ..., "content": ...} message dicts.
# This is separate from LangGraph's checkpointing (which stores RAGState).
# Phase 5 will move this into PostgreSQL for persistence across restarts.
#
# defaultdict(list) means: _session_histories["new-id"] → [] automatically.
# No KeyError on first access — clean default.
_session_histories: dict[str, list[dict]] = defaultdict(list)


def _update_session_history(session_id: str, question: str, answer: str) -> None:
    """Append the latest Q&A pair to the in-memory session history."""
    from src.module2_system_arch.state import update_history, truncate_history
    history = _session_histories[session_id]
    updated = update_history(history, question, answer)
    _session_histories[session_id] = truncate_history(updated, max_turns=5)


def get_session_history(session_id: str) -> list[dict]:
    """Retrieve conversation history for a session (for generate_node to use)."""
    return _session_histories.get(session_id, [])

# ── Shared Qdrant client ───────────────────────────────────────────────────────
#
# Created once at module load time. Reused across all invocations.
# QdrantClient manages a connection pool internally.
#
# PRODUCTION NOTE (Phase 3/FastAPI):
# The client is injected via FastAPI's dependency injection instead.
# Each FastAPI worker process has its own client instance.
# This avoids cross-process connection sharing issues.
_qdrant_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    """Lazy-initialise and return the shared Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = get_qdrant_client()
    return _qdrant_client


# ── Node 1: retrieve_node ─────────────────────────────────────────────────────

async def retrieve_node(state: RAGState) -> dict:
    """
    LangGraph node: retrieves relevant chunks from Qdrant.

    Wraps Module 1's synchronous retrieve() in async LangGraph format.

    NODE CONTRACT:
      Reads:   state["question"]
      Writes:  {"context": list[str], "scores": list[float],
                "sources": list[str]}
      Returns: partial state update

    WHY ASYNC IF retrieve() IS SYNCHRONOUS?
    LangGraph nodes must be async to participate in the async event loop.
    We run the sync Module 1 function directly — Python can call sync
    functions from async context without blocking other coroutines as
    long as the sync call is fast (Qdrant search is 2-5ms).

    For the OpenAI embedding call inside retrieve() (100-150ms network):
    This DOES block the event loop briefly. In Phase 3 (FastAPI), we'll
    switch to AsyncOpenAI() and asyncio.run_in_executor() to prevent
    blocking. For now, it's acceptable in single-user dev mode.

    WHAT HAPPENS INSIDE retrieve():
    1. embed_text(question) → 1536-dim vector [~100ms OpenAI call]
    2. client.query_points(query=vector, limit=k) → top-k chunks [~2ms]
    3. Returns list[RetrievedChunk] with text, score, source, etc.

    PHASE EVOLUTION:
    Phase 3: replace retrieve() call with hybrid_retrieve() which adds:
      - HyDE: embed 3 hypothetical answers, use all 4 vectors for search
      - Sparse BM25 search in parallel with dense
      - RRF fusion of dense + sparse results
      - Cross-encoder reranking of top-20 → top-5

    Args:
        state: Current RAGState from LangGraph

    Returns:
        Partial state update with context, scores, sources
    """
    question = state["question"]
    client   = _get_client()

    logger.info(f"retrieve_node: '{question[:50]}...'")

    # Call Module 1's synchronous retrieve()
    # Returns list[RetrievedChunk] with text, score, source, document_id
    chunks = retrieve(client, question, k=settings.retrieval_k)

    if not chunks:
        logger.warning("retrieve_node: No chunks retrieved — knowledge base may be empty")

    # Extract fields from RetrievedChunk objects into plain Python types
    # LangGraph serializes state to JSON for checkpointing —
    # plain types (str, float, list) serialize cleanly.
    # Custom dataclasses do NOT serialize cleanly → extract them here.
    context = [c.text   for c in chunks]
    scores  = [c.score  for c in chunks]
    sources = list({c.source for c in chunks})  # deduplicated set → list

    logger.info(
        f"retrieve_node: {len(chunks)} chunks | "
        f"scores: {[f'{s:.3f}' for s in scores]} | "
        f"sources: {sources}"
    )

    # Return ONLY the fields this node changed
    # LangGraph merges this dict into the existing state
    return {
        "context": context,
        "scores":  scores,
        "sources": sources,
    }


# ── Node 2: generate_node ─────────────────────────────────────────────────────

async def generate_node(state: RAGState) -> dict:
    """
    LangGraph node: generates a grounded answer using GPT-4o.

    Wraps Module 1's synchronous generate() in async LangGraph format.

    NODE CONTRACT:
      Reads:   state["question"], state["context"], state["iteration"]
      Writes:  {"answer": str, "iteration": int,
                "prompt_tokens": int, "completion_tokens": int}
      Returns: partial state update

    SPOTLIGHTING (already active in Module 1's generate()):
    Context chunks are wrapped in XML tags inside generate():
      <doc id="1" source="..." score="...">chunk text</doc>
    This:
      - Gives the LLM explicit chunk boundaries and IDs
      - Enables citations: "According to [doc 1]..."
      - Resists prompt injection from malicious document content

    ITERATION TRACKING:
    We increment state["iteration"] here. Phase 4's Self-RAG loop
    checks this counter — if iteration >= 2, stop regenerating
    and return the current answer regardless of quality score.
    This prevents infinite loops.

    PHASE EVOLUTION:
    Phase 4: after calling generate(), call self_rag_reflect_node:
      quality_score = score_answer(question, context, answer)
      if quality_score < 0.8 and iteration < 2:
          → loop back to retrieve_node with refined query
      else:
          → proceed to END

    Args:
        state: Current RAGState (must have context populated by retrieve_node)

    Returns:
        Partial state update with answer, iteration, token counts
    """
    question  = state["question"]
    context   = state["context"]
    iteration = state["iteration"]

    logger.info(
        f"generate_node: iteration={iteration} | "
        f"{len(context)} context chunks | "
        f"question='{question[:40]}...'"
    )

    if not context:
        # No context retrieved — answer without grounding
        # This happens when: Qdrant empty, score_threshold too high,
        # or all chunks filtered out
        # Safe fallback: honest "I don't know" response
        logger.warning("generate_node: Empty context — generating fallback response")
        return {
            "answer":            "I do not have documentation on this topic in the knowledge base. "
                                 "Please check the official Kubernetes documentation or escalate to the platform team.",
            "iteration":         iteration + 1,
            "prompt_tokens":     0,
            "completion_tokens": 0,
        }

    # ── Adapter: list[str] → list[RetrievedChunk] ────────────────────────────
    #
    # RAGState stores context as list[str] (plain text) because:
    #   1. TypedDict serialises cleanly to JSON for PostgreSQL checkpointing
    #   2. RetrievedChunk is a dataclass — not JSON-serialisable by default
    #
    # Module 1's generate() expects list[RetrievedChunk] because it was
    # designed before LangGraph state was introduced.
    #
    # ADAPTER PATTERN: reconstruct minimal RetrievedChunk objects so
    # generate() can build the spotlighted XML context string.
    # Scores and IDs are not needed for generation — only text + source.
    #
    # Phase 3 will redesign this cleanly: generate() will accept list[str]
    # directly and use a separate context-building step.
    sources = state.get("sources", [])
    chunks = [
        RetrievedChunk(
            text=text,
            score=0.0,           # not needed for generation
            source=sources[i] if i < len(sources) else "unknown",
            document_id="unknown",
            chunk_index=i,
            point_id="unknown",
        )
        for i, text in enumerate(context)
    ]

    # Call Module 1's synchronous generate()
    # Returns GenerationResult with answer + token usage metadata
    result = generate(question=question, chunks=chunks)

    # Cost calculated inline (avoids dependency on estimated_cost_usd property)
    cost_usd = (result.prompt_tokens * 5 + result.completion_tokens * 15) / 1_000_000

    logger.info(
        f"generate_node: answer generated | "
        f"tokens: {result.prompt_tokens}+{result.completion_tokens} | "
        f"cost: ~${cost_usd:.4f}"
    )

    # ── Update conversation history (Module 4) ────────────────────────────
    # Append this Q&A pair to the history stored in the session state.
    # On the NEXT turn with the same session_id, LangGraph loads this
    # checkpoint — the generate_node will see the full history and can
    # pass it to the LLM for context-aware follow-up answers.
    #
    # History is stored in a separate in-memory dict (not in RAGState)
    # to avoid TypedDict compatibility issues with existing checkpoints.
    # Phase 5 will migrate this cleanly into RAGState when we add PostgresSaver.
    _update_session_history(
        session_id=state.get("session_id", ""),
        question=question,
        answer=result.answer,
    )

    return {
        "answer":            result.answer,
        "iteration":         iteration + 1,
        "prompt_tokens":     result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }


# ── Node 3: sql_placeholder_node ─────────────────────────────────────────────

async def sql_placeholder_node(state: RAGState) -> dict:
    """
    Temporary placeholder for the Text2SQL pipeline.

    Phase 5 replaces this with the full pipeline:
      sql_generate_node → sql_validate_node → interrupt() → sql_execute_node

    For now: detect SQL questions and return an informative message
    rather than silently failing or hallucinating.

    This node ALSO serves as a teaching example: in production systems,
    unimplemented features should fail gracefully with clear messaging,
    not silently produce wrong results.

    NODE CONTRACT:
      Reads:   state["question"], state["intent"]
      Writes:  {"answer": str, "iteration": int}
      Returns: partial state update
    """
    question = state["question"]

    logger.info(f"sql_placeholder_node: SQL pipeline not yet implemented")

    return {
        "answer": (
            f"This question appears to need live cluster data: '{question}'\n\n"
            "The Text2SQL pipeline (Phase 5) is not yet implemented. "
            "To answer this, I would need to query the PostgreSQL operational "
            "database with a SQL query like:\n"
            "  SELECT ... FROM pods WHERE namespace = 'prod' ...\n\n"
            "Please use kubectl or your monitoring dashboard for live cluster data."
        ),
        "iteration": 1,
    }

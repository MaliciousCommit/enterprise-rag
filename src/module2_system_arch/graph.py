# src/module2_system_arch/graph.py
#
# Phase 5 graph topology — full Text2SQL pipeline with HITL interrupt().
#
# COMPLETE GRAPH:
#
#   START
#     ↓
#   intent_router
#     ├── rag/hybrid → retrieve → crag_grader → [tavily] → generate → self_rag → END
#     └── sql ──────→ sql_generate → sql_validate
#                                         ├── rejected → generate → self_rag → END
#                                         └── valid   → sql_human (interrupt)
#                                                           ├── rejected → generate → self_rag → END
#                                                           └── approved → sql_execute → sql_format
#                                                                              → generate → self_rag → END
#
# SHARED NODES: generate and self_rag_reflect are shared between RAG and SQL paths.
# Both paths converge at generate_node — it handles both context types.

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.module2_system_arch.state import RAGState
from src.module2_system_arch.intent_router import intent_router_node, route_by_intent
from src.module2_system_arch.nodes import retrieve_node, generate_node

logger = logging.getLogger(__name__)


def build_graph(use_memory_checkpointer: bool = True):
    """Build and compile the Phase 5 LangGraph StateGraph."""

    # Lazy imports to avoid circular dependencies
    from src.phase4_agentic.crag import crag_grader_node, route_by_crag
    from src.phase4_agentic.tavily_search import tavily_search_node
    from src.phase4_agentic.self_rag import self_rag_reflect_node, route_self_rag
    from src.phase5_text2sql.nodes import (
        sql_generate_node, sql_validate_node, sql_human_node,
        sql_execute_node, sql_format_node,
        route_after_validation, route_after_human,
    )

    graph = StateGraph(RAGState)

    # ── Register all nodes ────────────────────────────────────────────────────
    graph.add_node("intent_router",     intent_router_node)
    graph.add_node("retrieve",          retrieve_node)
    graph.add_node("crag_grader",       crag_grader_node)
    graph.add_node("tavily_search",     tavily_search_node)
    graph.add_node("generate",          generate_node)
    graph.add_node("self_rag_reflect",  self_rag_reflect_node)
    # Phase 5: Text2SQL pipeline (replaces sql_placeholder)
    graph.add_node("sql_generate",      sql_generate_node)
    graph.add_node("sql_validate",      sql_validate_node)
    graph.add_node("sql_human",         sql_human_node)         # ← interrupt()
    graph.add_node("sql_execute",       sql_execute_node)
    graph.add_node("sql_format",        sql_format_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.add_edge(START, "intent_router")

    # ── Intent routing ────────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "intent_router", route_by_intent,
        {
            "rag_retrieve":    "retrieve",
            "hybrid_retrieve": "retrieve",
            "sql_pipeline":    "sql_generate",  # Phase 5: real SQL pipeline
        },
    )

    # ── RAG path: retrieve → CRAG → [Tavily] → generate ─────────────────────
    graph.add_edge("retrieve",      "crag_grader")
    graph.add_conditional_edges(
        "crag_grader", route_by_crag,
        {"good_retrieval": "generate", "poor_retrieval": "tavily_search"},
    )
    graph.add_edge("tavily_search", "generate")

    # ── Text2SQL path ─────────────────────────────────────────────────────────
    graph.add_edge("sql_generate",  "sql_validate")

    graph.add_conditional_edges(
        "sql_validate", route_after_validation,
        {
            "rejected":          "generate",   # rejected SQL → explain why → generate
            "needs_human_review": "sql_human",
        },
    )

    graph.add_conditional_edges(
        "sql_human", route_after_human,
        {
            "execute":  "sql_execute",
            "rejected": "generate",            # human rejected → explain → generate
        },
    )

    graph.add_edge("sql_execute", "sql_format")
    graph.add_edge("sql_format",  "generate")

    # ── Shared: generate → Self-RAG → END (or cycle back) ────────────────────
    graph.add_edge("generate", "self_rag_reflect")
    graph.add_conditional_edges(
        "self_rag_reflect", route_self_rag,
        {"accept": END, "regenerate": "retrieve"},
    )

    # ── Compile ───────────────────────────────────────────────────────────────
    checkpointer = MemorySaver() if use_memory_checkpointer else None
    compiled = graph.compile(checkpointer=checkpointer, interrupt_before=["sql_human"])

    logger.info(
        "Phase 5 graph compiled | "
        "nodes: [intent_router, retrieve, crag_grader, tavily_search, generate, "
        "self_rag_reflect, sql_generate, sql_validate, sql_human, sql_execute, sql_format] | "
        "interrupt_before: sql_human"
    )
    return compiled


async def run_graph(question: str, session_id: str, graph=None) -> dict:
    """Run the full pipeline. Handles HITL interrupt detection."""
    from src.module2_system_arch.state import initial_state
    from langgraph.types import Command

    if graph is None:
        graph = build_graph()

    # ── Phase 6: Answer cache check ───────────────────────────────────────────
    # Check BEFORE invoking the graph — a cache hit skips the entire pipeline:
    # intent routing, embedding, retrieval, reranking, LLM generation.
    # ~5,000ms → ~3ms. The highest-value optimisation in the system.
    #
    # NOT cached for SQL questions: live data changes frequently.
    # The SQL result cache (TTL=15m) handles SQL freshness at a lower level.
    # Here we only skip for RAG questions where the answer is truly stable.
    try:
        from src.phase6_cache.manager import get_cache_manager
        cache = get_cache_manager()
        cached_answer = cache.get_answer(question)
        if cached_answer:
            logger.info(f"run_graph: ANSWER CACHE HIT | session='{session_id}'")
            # Return a synthetic state with the cached answer
            state = initial_state(question=question, session_id=session_id)
            state.update({
                "answer":    cached_answer,
                "intent":    cache.get_intent(question) or "rag",
                "cache_hit": True,
            })
            return state
    except Exception:
        pass  # Redis down — run the full pipeline

    state  = initial_state(question=question, session_id=session_id)
    config = {"configurable": {"thread_id": session_id}}

    logger.info(f"run_graph: '{question[:50]}...' | session='{session_id}'")

    final_state = await graph.ainvoke(state, config=config)

    # Detect HITL interrupt: SQL was generated but graph paused for review
    # When interrupt_before=["sql_human"] fires, the graph stops BEFORE sql_human runs.
    # State will have sql_query set, sql_approved=None → pending review.
    if (final_state.get("intent") == "sql" and
        final_state.get("sql_query") and
        final_state.get("sql_approved") is None and
        not final_state.get("answer")):

        final_state["pending_approval"] = True
        logger.info(
            f"Graph interrupted — awaiting SQL approval | "
            f"session='{session_id}' | sql='{final_state['sql_query'][:60]}...'"
        )
    else:
        final_state["pending_approval"] = False
        logger.info(
            f"run_graph complete | intent={final_state.get('intent')} | "
            f"grade={final_state.get('retrieval_grade')} | "
            f"score={final_state.get('self_rag_score', 1.0):.3f} | "
            f"answer_len={len(final_state.get('answer', ''))} chars"
        )

    return final_state


async def resume_graph(session_id: str, approved: bool, graph=None) -> dict:
    """
    Resume a graph that was paused at the HITL interrupt point.

    Called by the FastAPI /api/v1/sql/approve endpoint.

    The Command(resume=...) value is what interrupt() returns inside sql_human_node.
    sql_human_node then sets state["sql_approved"] = True/False based on this value.

    Args:
        session_id: The session that was interrupted (must match original session_id)
        approved:   True = execute the SQL, False = abort
        graph:      The compiled StateGraph (same instance as the original call)

    Returns:
        Final RAGState dict after resumption and completion
    """
    from langgraph.types import Command

    if graph is None:
        graph = build_graph()

    config = {"configurable": {"thread_id": session_id}}

    logger.info(f"Resuming graph | session='{session_id}' | approved={approved}")

    # Resume with the human's decision
    # This is picked up by interrupt() inside sql_human_node
    final_state = await graph.ainvoke(
        Command(resume={"approved": approved}),
        config=config,
    )

    logger.info(
        f"Resume complete | sql_approved={final_state.get('sql_approved')} | "
        f"answer_len={len(final_state.get('answer', ''))} chars"
    )
    return final_state

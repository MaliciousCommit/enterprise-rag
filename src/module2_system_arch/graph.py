# src/module2_system_arch/graph.py
#
# The LangGraph StateGraph definition for Module 2.
#
# This file owns the TOPOLOGY of the system — which nodes exist,
# how they connect, and what conditions control routing.
#
# GRAPH TOPOLOGY (Module 2):
#
#   START
#     │
#     ▼
#   intent_router_node ── classify: rag|sql|hybrid
#     │
#     ├─── rag / hybrid ──► retrieve_node ──► generate_node ──► END
#     │
#     └─── sql ───────────► sql_placeholder_node ─────────────► END
#
# PHASE EVOLUTION (how this topology changes):
#
# Phase 3 (Hybrid Search):
#   retrieve_node internally adds HyDE + hybrid search + reranking
#   Graph topology unchanged — just richer node internals
#
# Phase 4 (CRAG + Self-RAG):
#   retrieve_node → crag_grader_node
#     ├── high confidence → generate_node → self_rag_reflect_node
#     │     ├── quality OK → END
#     │     └── quality low → retrieve_node (CYCLE — loop back)
#     └── low confidence → tavily_search_node → generate_node → ...
#
# Phase 5 (Text2SQL):
#   sql_placeholder_node → sql_generate_node → sql_validate_node
#     → interrupt() → sql_execute_node → generate_node → END
#
# CHECKPOINTING:
#   Every node execution writes state to the checkpointer.
#   Module 2: MemorySaver (in-process dict, no persistence)
#   Phase 5+: PostgresSaver (PostgreSQL, survives restarts + HITL)
#
# WHY STATE MACHINE OVER PLAIN ASYNC PYTHON?
#   Three capabilities that async Python cannot provide:
#   1. CYCLES: Self-RAG requires looping back to retrieve_node
#      if answer quality is low. Async Python has no safe cycle primitive.
#   2. INTERRUPTS: HITL SQL approval requires pausing execution,
#      waiting hours, then resuming exactly where it left off.
#      LangGraph's interrupt() + checkpointing makes this one line.
#   3. BRANCHING: Intent routing to different pipelines is expressed
#      declaratively in add_conditional_edges() — the graph validates
#      it at compile time.

import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.module2_system_arch.state import RAGState
from src.module2_system_arch.intent_router import (
    intent_router_node,
    route_by_intent,
)
from src.module2_system_arch.nodes import (
    retrieve_node,
    generate_node,
    sql_placeholder_node,
)

logger = logging.getLogger(__name__)

# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(use_memory_checkpointer: bool = True):
    """
    Build and compile the Module 2 LangGraph StateGraph.

    COMPILATION:
    graph.compile() does three things:
      1. Validates the graph topology (detects dead ends, missing nodes)
      2. Builds the execution plan (which nodes run in what order)
      3. Attaches the checkpointer (state persistence per session)

    The compiled graph is thread-safe and can handle concurrent invocations.
    In Phase 3 (FastAPI): compile once at startup, share across all requests.

    CHECKPOINTER CHOICE:
      MemorySaver (Module 2):
        - In-process Python dict
        - State survives within a single process run
        - Lost on restart → no conversation memory across sessions
        - No PostgreSQL required → easy setup for dev
        - Use for: development, testing, single-user scripts

      PostgresSaver (Phase 5):
        - Persists to PostgreSQL tables (checkpoints, checkpoint_blobs)
        - Survives restarts → true conversation memory
        - Enables HITL: pause execution, wait hours, resume from exact state
        - Use for: production, multi-user, HITL workflows

    Args:
        use_memory_checkpointer: If True, use MemorySaver.
                                 If False, caller must attach their own.

    Returns:
        Compiled LangGraph StateGraph ready for ainvoke()
    """
    # ── 1. Create the StateGraph typed to RAGState ────────────────────────────
    #
    # StateGraph(RAGState) does two things at compile time:
    #   a. Type-checks node return dicts against RAGState fields
    #      (a node returning {"nonexistent_field": ...} raises TypeError)
    #   b. Validates that all declared nodes have at least one incoming edge
    graph = StateGraph(RAGState)

    # ── 2. Register nodes ─────────────────────────────────────────────────────
    #
    # add_node(name, function) registers a callable as a graph node.
    # The name is what conditional edges use to route to this node.
    # Names must be unique within the graph.
    #
    # Node functions must:
    #   - Accept a single argument: the current RAGState dict
    #   - Return a dict with ONLY the fields they changed
    #   - Be async (for LangGraph's async execution engine)
    graph.add_node("intent_router",     intent_router_node)
    graph.add_node("retrieve",          retrieve_node)
    graph.add_node("generate",          generate_node)
    graph.add_node("sql_placeholder",   sql_placeholder_node)

    # ── 3. Add edges ──────────────────────────────────────────────────────────
    #
    # Edges define the execution flow. Three types:
    #
    # add_edge(A, B):
    #   Unconditional. A always flows to B.
    #   A's return value is merged into state, then B executes.
    #
    # add_conditional_edges(source, routing_fn, path_map):
    #   Conditional. routing_fn(state) returns a KEY.
    #   path_map maps KEY → node name.
    #   This is the branching mechanism.
    #
    # add_edge(START, first_node):
    #   START is a special sentinel. The graph begins here.
    #
    # add_edge(last_node, END):
    #   END is a special sentinel. Execution stops here.
    #   The final state is returned to the caller.

    # Entry point: every request starts at intent_router
    graph.add_edge(START, "intent_router")

    # Conditional routing based on intent classification
    # route_by_intent(state) returns one of:
    #   "rag_retrieve"   → maps to "retrieve" node
    #   "hybrid_retrieve" → maps to "retrieve" node (same path)
    #   "sql_pipeline"   → maps to "sql_placeholder" node
    graph.add_conditional_edges(
        source="intent_router",
        path=route_by_intent,
        path_map={
            "rag_retrieve":    "retrieve",          # rag → retrieve → generate
            "hybrid_retrieve": "retrieve",          # hybrid → retrieve → generate
            "sql_pipeline":    "sql_placeholder",   # sql → placeholder → END
        },
    )

    # RAG path: retrieve → generate → END
    graph.add_edge("retrieve",        "generate")
    graph.add_edge("generate",        END)

    # SQL path (placeholder for Phase 5): sql_placeholder → END
    graph.add_edge("sql_placeholder", END)

    # ── 4. Compile the graph ──────────────────────────────────────────────────
    if use_memory_checkpointer:
        checkpointer = MemorySaver()
        # MemorySaver stores state in a Python dict keyed by thread_id.
        # thread_id = session_id from RAGState.
        # Two calls with the same thread_id will share state →
        # conversation memory within a single process run.
    else:
        checkpointer = None

    compiled = graph.compile(checkpointer=checkpointer)

    logger.info(
        "LangGraph StateGraph compiled | "
        "nodes: [intent_router, retrieve, generate, sql_placeholder] | "
        f"checkpointer: {'MemorySaver' if use_memory_checkpointer else 'None'}"
    )

    return compiled


# ── Graph runner ──────────────────────────────────────────────────────────────

async def run_graph(
    question: str,
    session_id: str,
    graph=None,
) -> dict:
    """
    Run the complete RAG pipeline for a single question.

    High-level wrapper around graph.ainvoke() that:
    1. Builds the initial RAGState
    2. Configures the thread_id for checkpointing
    3. Invokes the graph
    4. Returns the final state as a plain dict

    This is what FastAPI (Phase 3) will call from its endpoint handler.

    THREAD_ID CONFIG:
    The config dict passed to ainvoke() controls checkpointing.
    thread_id = session_id links this invocation to a conversation.
    Same session_id across multiple calls = the checkpointer
    loads the previous state, enabling conversation memory.

    {"configurable": {"thread_id": session_id}}
    This is LangGraph's convention for the checkpointer lookup key.

    Args:
        question:   Natural language question from the user
        session_id: Conversation session ID (unique per user session)
        graph:      Compiled StateGraph (built by build_graph())
                    If None, builds a new graph (dev/test convenience)

    Returns:
        Final RAGState dict with answer, context, scores, etc.
    """
    from src.module2_system_arch.state import initial_state

    if graph is None:
        graph = build_graph()

    # Build initial state with all fields zeroed out
    state = initial_state(question=question, session_id=session_id)

    # Config tells LangGraph which thread to checkpoint to/from
    config = {"configurable": {"thread_id": session_id}}

    logger.info(f"run_graph: question='{question[:50]}...' | session='{session_id}'")

    # ainvoke() runs the graph to completion and returns the final state
    # For interrupt() (Phase 5 HITL), use astream() instead which
    # yields intermediate states and pauses at interrupt points.
    final_state = await graph.ainvoke(state, config=config)

    logger.info(
        f"run_graph complete | "
        f"intent={final_state.get('intent')} | "
        f"answer_length={len(final_state.get('answer', ''))} chars"
    )

    return final_state

# src/phase5_text2sql/nodes.py
#
# LangGraph nodes for the Phase 5 Text2SQL pipeline.
#
# NODE EXECUTION ORDER:
#
# sql_generate_node   → generates SQL, sets state["sql_query"]
#       ↓
# sql_validate_node   → security check; rejects bad SQL immediately
#       ↓
# sql_human_node      → interrupt(): graph PAUSES, human reviews SQL
#       ↓  (after human approves)
# sql_execute_node    → runs the SELECT, stores results in state
#       ↓
# sql_format_node     → converts rows to readable text for generate_node
#       ↓
# generate_node       → answers in English using the formatted SQL results
#
# THE interrupt() MECHANISM:
#
# When sql_human_node calls interrupt(data), LangGraph:
#   1. Saves the FULL state to the checkpointer (MemorySaver)
#   2. Returns the state (as of before sql_human_node ran) to the caller
#   3. Execution is SUSPENDED — the graph is paused in memory
#
# To resume:
#   from langgraph.types import Command
#   result = await graph.ainvoke(
#       Command(resume={"approved": True}),
#       config={"configurable": {"thread_id": session_id}}
#   )
#
# The Command(resume=...) tells LangGraph to:
#   1. Load the saved state from MemorySaver (by thread_id)
#   2. Resume sql_human_node from where it paused
#   3. interrupt() returns the resume value {"approved": True/False}
#   4. The node returns {"sql_approved": True/False}
#   5. Graph continues to sql_execute_node or terminates if rejected

import logging

from langgraph.types import interrupt

from src.module2_system_arch.state import RAGState
from src.phase5_text2sql.database import execute_select, format_results_as_text
from src.phase5_text2sql.generator import generate_sql
from src.phase5_text2sql.validator import validate_sql

logger = logging.getLogger(__name__)


async def sql_generate_node(state: RAGState) -> dict:
    """
    LangGraph node: generate SQL from the user's question.

    NODE CONTRACT:
      Reads:  state["question"]
      Writes: {"sql_query": str}

    WHAT CAN GO WRONG:
    - LLM times out → returns error comment as sql_query
    - LLM hallucinates column name → validation catches it
    - Schema is ambiguous → few-shot examples guide the LLM
    """
    question = state["question"]
    logger.info(f"sql_generate_node: '{question[:50]}...'")

    # ── Phase 6: Check SQL generation cache ──────────────────────────────────
    # Cache hit: skip the ~1.5s GPT-4o SQL generation call.
    # IMPORTANT: SQL questions still require HITL approval even on cache hit.
    # We cache the SQL structure (which is stable), not the execution result
    # (which is fresh data, cached separately in Tier 4).
    try:
        from src.phase6_cache.manager import get_cache_manager
        cache = get_cache_manager()
        cached_sql = cache.get_sql(question)
        if cached_sql:
            logger.info(f"sql_generate_node: SQL cache HIT")
            return {"sql_query": cached_sql}
    except Exception:
        pass

    sql = await generate_sql(question)

    # Store for 24h
    try:
        if not sql.startswith("--"):  # don't cache error comments
            cache.set_sql(question, sql)
    except Exception:
        pass

    return {"sql_query": sql}


async def sql_validate_node(state: RAGState) -> dict:
    """
    LangGraph node: validate the generated SQL for security.

    NODE CONTRACT:
      Reads:  state["sql_query"]
      Writes: {"sql_approved": bool, "answer": str} on failure

    ON VALIDATION FAILURE:
    Sets sql_approved=False and a descriptive answer explaining why.
    The graph must be wired to check sql_approved and terminate
    if False (no human review needed for rejected SQL).

    ON VALIDATION SUCCESS:
    Returns {} (empty dict — no state change, proceed to interrupt).
    """
    sql = state.get("sql_query", "")

    if not sql or sql.startswith("-- SQL generation failed"):
        return {
            "sql_approved": False,
            "answer": "SQL generation failed. Please rephrase your question about cluster data.",
        }

    result = validate_sql(sql)

    if not result.is_valid:
        logger.warning(f"SQL validation failed: {result.reason}")
        return {
            "sql_approved": False,
            "answer": (
                f"Generated SQL was rejected for security reasons: {result.reason}\n\n"
                "The system only permits SELECT queries with LIMIT ≤ 100 "
                "against the operational tables."
            ),
        }

    logger.info("SQL validation passed — proceeding to human review")
    return {}  # No state change — proceed to interrupt


async def sql_human_node(state: RAGState) -> dict:
    """
    LangGraph node: pause execution and wait for human SQL approval.

    THIS IS THE CORE OF HITL.

    The interrupt() call causes LangGraph to:
    1. Checkpoint the full state to MemorySaver (or PostgresSaver in prod)
    2. Return control to the graph.ainvoke() caller
    3. The caller detects the pending state and returns HTTP 200 with
       {"pending_approval": true, "pending_sql": "SELECT ..."}
    4. The SRE reviews the SQL in the UI
    5. The SRE approves or rejects via POST /api/v1/sql/approve
    6. FastAPI calls graph.ainvoke(Command(resume={...}), config)
    7. interrupt() returns the resume value and execution continues

    NODE CONTRACT:
      Reads:  state["sql_query"]
      Writes: {"sql_approved": bool}

    WHY INTERRUPT RATHER THAN POLLING?
    Polling requires the graph to stay alive in memory — not possible if
    the process restarts. interrupt() serialises the graph state so it
    survives restarts (with PostgresSaver) and can be resumed days later.
    """
    sql_query = state.get("sql_query", "")

    logger.info(f"sql_human_node: INTERRUPT — awaiting human approval")
    logger.info(f"  SQL to review: {sql_query[:100]}...")

    # PAUSE HERE: LangGraph suspends, returns state to caller
    # The value passed to interrupt() is available in the interrupt event
    approval = interrupt({
        "message":   "SQL query requires human approval before execution.",
        "sql_query": sql_query,
        "warning":   "This query will run against the production operational database.",
    })

    # RESUMED: approval is the value passed in Command(resume={...})
    approved = approval.get("approved", False) if isinstance(approval, dict) else bool(approval)

    logger.info(f"sql_human_node: RESUMED | approved={approved}")

    return {"sql_approved": approved}


def route_after_validation(state: RAGState) -> str:
    """
    Conditional edge: after sql_validate_node, check if SQL was rejected.
    Rejected SQL → skip human review, go directly to END via generate.
    Valid SQL → proceed to human review.
    """
    # If sql_approved was explicitly set to False by validator: skip human review
    if state.get("sql_approved") is False:
        return "rejected"
    return "needs_human_review"


def route_after_human(state: RAGState) -> str:
    """
    Conditional edge: after sql_human_node, check if human approved.
    Approved → execute SQL
    Rejected → terminate pipeline (generate will explain)
    """
    if state.get("sql_approved"):
        return "execute"
    return "rejected"


async def sql_execute_node(state: RAGState) -> dict:
    """
    LangGraph node: execute the approved SQL against PostgreSQL.

    Only runs after human approves via HITL.
    Enforces: statement timeout (5s), row limit (100 rows).

    NODE CONTRACT:
      Reads:  state["sql_query"]
      Writes: {"sql_result": str}
    """
    sql = state.get("sql_query", "")

    if not sql:
        return {"sql_result": "Error: no SQL query to execute."}

    # ── Phase 6: Check SQL result cache ──────────────────────────────────────
    # Cache hit: skip the PostgreSQL query entirely.
    # TTL=15m means results are at most 15 minutes stale.
    # This is the most impactful cache for repeated SQL questions:
    # "how many pods are failing?" asked every 5 minutes by a monitoring script
    # only hits PostgreSQL once every 15 minutes.
    try:
        from src.phase6_cache.manager import get_cache_manager
        cache = get_cache_manager()
        cached_result = cache.get_sql_result(sql)
        if cached_result:
            logger.info("sql_execute_node: SQL result cache HIT (15m TTL)")
            return {"sql_result": cached_result}
    except Exception:
        pass

    logger.info(f"sql_execute_node: executing approved SQL")

    result = execute_select(sql)
    formatted = format_results_as_text(result)

    logger.info(
        f"sql_execute_node: {result['row_count']} rows | "
        f"error: {result.get('error')}"
    )

    # Cache the result for 15 minutes
    try:
        if not result.get("error"):
            cache.set_sql_result(sql, formatted)
    except Exception:
        pass

    return {"sql_result": formatted}


async def sql_format_node(state: RAGState) -> dict:
    """
    LangGraph node: convert SQL results into LLM-ready context.

    The generate_node needs context as a list of strings.
    We wrap the SQL table in XML spotlighting so the LLM can
    cite "According to the live cluster data..." in its answer.

    NODE CONTRACT:
      Reads:  state["sql_result"], state["question"]
      Writes: {"context": list[str], "sources": list[str]}
    """
    sql_result = state.get("sql_result", "")
    sql_query  = state.get("sql_query",  "")

    if not sql_result or "Error" in sql_result:
        return {
            "context": [f"SQL execution failed or returned no data: {sql_result}"],
            "sources": ["postgresql_error"],
        }

    # Wrap in XML spotlighting for the LLM
    context_chunk = (
        f"[LIVE CLUSTER DATA — SQL query result]\n"
        f"Query: {sql_query[:200]}\n\n"
        f"{sql_result}"
    )

    return {
        "context": [context_chunk],
        "sources": ["postgresql_live_data"],
    }

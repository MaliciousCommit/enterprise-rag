# src/api/routes/query.py
#
# Main query endpoint: POST /api/v1/query
#
# This is the entry point for all RAG questions.
# It orchestrates: auth → validation → LangGraph → response.
#
# REQUEST LIFECYCLE through this endpoint:
#
# 1. Middleware runs first (TimingMiddleware assigns request_id)
# 2. FastAPI validates JSON body against QueryRequest (Pydantic)
# 3. FastAPI resolves all Depends() — calls get_current_user, get_graph
# 4. Our handler runs: session_id assigned, graph invoked
# 5. LangGraph state machine executes (intent → retrieve → generate)
# 6. Final state mapped to ChatResponse
# 7. FastAPI serialises ChatResponse to JSON
# 8. Middleware adds timing headers
# 9. HTTP 200 returned
#
# ASYNC DESIGN:
# The entire path from FastAPI to LangGraph is async.
# While LangGraph awaits OpenAI's API (~1-2 seconds),
# the event loop can handle other incoming requests.
# This is why we're not blocking on each query —
# FastAPI can handle many concurrent requests with a single process.
#
# PRODUCTION NOTE (Phase 10):
# In Kubernetes, we run multiple uvicorn workers per pod.
# Each worker has its own event loop.
# The graph singleton (get_graph) is per-process (per-worker).
# This is correct — no cross-process state sharing is needed.

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status

from src.api.auth import CurrentUser
from src.api.dependencies import GraphDep
from src.api.models import ChatResponse, QueryRequest, RetrievalInfo, UsageInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post(
    "/query",
    response_model=ChatResponse,
    summary="Ask a Kubernetes operations question",
    description=(
        "Submit a natural language question about Kubernetes IT operations. "
        "The system classifies intent (rag/sql/hybrid), retrieves relevant "
        "context from the knowledge base, and generates a grounded answer."
    ),
    responses={
        200: {"description": "Successfully generated answer"},
        401: {"description": "Authentication required"},
        422: {"description": "Invalid request (Pydantic validation failed)"},
        503: {"description": "Knowledge base unavailable"},
    },
)
async def query(
    request_body: QueryRequest,
    request: Request,
    user_id: CurrentUser,
    graph: GraphDep,
) -> ChatResponse:
    """
    Main RAG query endpoint.

    PARAMETERS EXPLAINED:

    request_body: QueryRequest
        Pydantic validates this automatically. If invalid → 422 before we run.

    request: Request
        The raw Starlette request object. We read state.request_id
        (set by TimingMiddleware) for logging and error correlation.

    user_id: CurrentUser
        Injected by get_current_user dependency (from auth.py).
        "dev-user@localhost" in dev mode.
        Actual email/UUID in production mode with JWT.

    graph: GraphDep
        The compiled LangGraph StateGraph singleton.
        Same instance across all requests for this process.
        Thread-safe: LangGraph handles concurrent invocations.

    RESPONSE MAPPING:
    LangGraph returns the final RAGState dict.
    We map it to ChatResponse — extracting only the fields the
    client needs and hiding internal state machine details.
    """
    start = time.perf_counter()
    request_id = getattr(request.state, "request_id", "unknown")

    # Assign session ID (conversation thread)
    # If client provided one: reuse it (conversation memory kicks in)
    # If not: generate a new UUID (fresh stateless query)
    session_id = request_body.session_id or f"sess-{uuid.uuid4().hex[:12]}"

    logger.info(
        f"Query | user='{user_id}' | session='{session_id}' | "
        f"req='{request_id}' | q='{request_body.question[:50]}...'"
    )

    # ── Invoke LangGraph pipeline ─────────────────────────────────────────────
    try:
        from src.module2_system_arch.graph import run_graph

        final_state = await run_graph(
            question=request_body.question,
            session_id=session_id,
            graph=graph,
        )

    except Exception as e:
        logger.error(f"Graph execution failed | req='{request_id}' | error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pipeline failed: {str(e)}. Check server logs with request_id={request_id}",
        )

    # ── Map RAGState → ChatResponse ───────────────────────────────────────────
    #
    # We extract only what the client needs.
    # Internal fields (iteration, sql_approved, checkpoint metadata) are hidden.
    latency_ms = (time.perf_counter() - start) * 1000

    scores  = final_state.get("scores", [])
    sources = final_state.get("sources", [])

    prompt_tokens     = final_state.get("prompt_tokens", 0)
    completion_tokens = final_state.get("completion_tokens", 0)
    total_tokens      = prompt_tokens + completion_tokens

    # Cost estimate: GPT-4o pricing
    # $5/M input tokens, $15/M output tokens
    cost_usd = (prompt_tokens * 5 + completion_tokens * 15) / 1_000_000

    response = ChatResponse(
        answer=final_state.get("answer", ""),
        session_id=session_id,
        intent=final_state.get("intent", "unknown"),

        retrieval=RetrievalInfo(
            num_chunks=len(scores),
            best_score=max(scores) if scores else 0.0,
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            scores=[round(s, 4) for s in scores],
            sources=sources,
        ),

        usage=UsageInfo(
            model="gpt-4o",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost_usd, 6),
        ),

        latency_ms=round(latency_ms, 1),
        cache_hit=final_state.get("cache_hit", False),

        # Phase 5: HITL Text2SQL (not active yet)
        pending_approval=False,
        pending_sql=final_state.get("sql_query"),
    )

    logger.info(
        f"Query complete | user='{user_id}' | intent='{response.intent}' | "
        f"latency={latency_ms:.0f}ms | tokens={total_tokens} | "
        f"cost=${cost_usd:.4f} | req='{request_id}'"
    )

    return response

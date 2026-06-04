# src/api/models.py
#
# Pydantic models defining the API contract.
#
# WHY PYDANTIC MODELS MATTER:
# Every HTTP request passes through these models before any business
# logic runs. If validation fails, FastAPI returns HTTP 422 automatically
# — your route handler never sees the invalid input. This is L1 of our
# security pipeline: schema validation at the HTTP boundary.
#
# SEPARATION FROM RAGState:
# These models are the EXTERNAL contract (what the API exposes).
# RAGState is the INTERNAL contract (what LangGraph uses).
# They are intentionally different — the API hides internal fields
# (iteration count, LangGraph metadata) from the external consumer.
#
# PHASE EVOLUTION:
# Module 3: basic query + response models
# Phase 5:  add SqlApprovalRequest for HITL Text2SQL approval
# Phase 8:  add guardrail_flags field to ChatResponse for audit
# Phase 9:  add trace_id field for LangSmith correlation

import uuid
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Request Models ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    The request body for POST /api/v1/query.

    Every field has explicit constraints. Pydantic enforces these
    before your route handler runs — no manual validation needed.

    FIELD DESIGN DECISIONS:

    question: str (1–2000 chars)
        The user's natural language question.
        Min length 1: empty strings are rejected (not passed to LLM).
        Max length 2000: prevents token budget abuse (~500 tokens max input).
        The tiktoken budget check (L5) further enforces per-request limits.

    session_id: Optional[str]
        Links this request to a conversation thread.
        If None: we generate a fresh UUID → stateless single-turn query.
        If provided: LangGraph loads previous conversation state from
        the MemorySaver (or PostgresSaver in Phase 5).
        This is how multi-turn conversations work in our system.

    namespace: Optional[str]
        Future use — filter Qdrant retrieval to specific K8s namespace.
        Phase 5 will pass this to the Text2SQL WHERE clause.
        Phase 3 will use it as a Qdrant payload filter.
        Included now so clients don't need API version bumps later.
    """
    question: str = Field(
        ...,                    # ... means required (no default)
        min_length=1,
        max_length=2000,
        description="The natural language question about Kubernetes operations",
        examples=["Why is my pod showing OOMKilled?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation memory. Auto-generated if not provided.",
        examples=["sess-alice-2024-001"],
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Kubernetes namespace filter (optional, for future use)",
        examples=["prod", "staging"],
    )

    @field_validator("question")
    @classmethod
    def question_not_whitespace(cls, v: str) -> str:
        """
        Reject questions that are only whitespace.
        '   ' has length > 0 but is not a valid question.
        This validator strips and re-checks after the min_length check.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question cannot be empty or whitespace only")
        return stripped


class SqlApprovalRequest(BaseModel):
    """
    Phase 5 (Text2SQL HITL): approve or reject a generated SQL query.

    When the Text2SQL pipeline fires interrupt(), it returns the SQL
    to the user for review. The user then calls this endpoint to
    approve (execute) or reject (abort) the SQL.

    Included here in Module 3 so the API schema is stable —
    clients built now won't need breaking changes in Phase 5.
    """
    session_id: str = Field(..., description="Session to resume")
    approved: bool = Field(..., description="True = execute SQL, False = abort")
    feedback: Optional[str] = Field(
        default=None,
        description="Optional reason for rejection (helps improve SQL generation)",
    )


# ── Response Models ────────────────────────────────────────────────────────────

class RetrievalInfo(BaseModel):
    """
    Retrieval metadata included in every RAG response.
    Allows clients to display citations and debug retrieval quality.
    """
    num_chunks: int = Field(description="Number of chunks retrieved from Qdrant")
    best_score: float = Field(description="Highest cosine similarity score (0.0-1.0)")
    avg_score: float = Field(description="Average cosine similarity across chunks")
    scores: list[float] = Field(description="Per-chunk similarity scores")
    sources: list[str] = Field(description="Source document identifiers")


class UsageInfo(BaseModel):
    """
    Token usage and cost metadata for observability and billing.
    Phase 9 aggregates these into Prometheus metrics.
    """
    model: str = Field(description="LLM model used for generation")
    prompt_tokens: int = Field(description="Input tokens (LLM pricing: $5/M for GPT-4o)")
    completion_tokens: int = Field(description="Output tokens (LLM pricing: $15/M for GPT-4o)")
    total_tokens: int = Field(description="Combined token count")
    estimated_cost_usd: float = Field(description="Rough cost estimate in USD")


class ChatResponse(BaseModel):
    """
    The response body for POST /api/v1/query.

    DESIGN DECISION — What to expose vs hide:
    Expose:  answer, sources, intent, retrieval scores, latency, tokens
    Hide:    internal LangGraph node execution details
             RAGState fields like sql_query, sql_approved (internal)
             Checkpoint metadata (PostgreSQL implementation detail)

    The exposed fields give clients everything they need for:
    - Displaying the answer to users
    - Showing citation sources
    - Debugging retrieval quality (scores)
    - Cost attribution (tokens)
    - SLA monitoring (latency)

    HITL STATUS:
    pending_approval signals that the Text2SQL pipeline hit interrupt()
    and is waiting for human SQL approval. The client should then
    call POST /api/v1/query/approve with approved=True/False.
    """
    answer: str = Field(description="The generated answer grounded in retrieved context")
    session_id: str = Field(description="Session ID (auto-generated or as provided)")
    intent: str = Field(description="Classified intent: 'rag', 'sql', or 'hybrid'")

    retrieval: RetrievalInfo = Field(description="Retrieval quality metadata")
    usage: UsageInfo = Field(description="Token usage and estimated cost")

    latency_ms: float = Field(description="End-to-end latency in milliseconds")
    cache_hit: bool = Field(description="True if answer was served from Redis cache")

    # Phase 5: HITL Text2SQL
    pending_approval: bool = Field(
        default=False,
        description="True when Text2SQL is awaiting human SQL approval",
    )
    pending_sql: Optional[str] = Field(
        default=None,
        description="The SQL query awaiting approval (Phase 5)",
    )


class HealthResponse(BaseModel):
    """
    Response for GET /health.
    Used by Kubernetes liveness and readiness probes (Phase 10).

    STATUS VALUES:
    "healthy":  All systems operational. Traffic can be served.
    "degraded": Partial functionality. Qdrant up but empty, or
                optional services unavailable.
    "unhealthy": Cannot serve traffic. Qdrant unreachable.

    Kubernetes readiness probe: return 200 only for "healthy"+"degraded"
    Kubernetes liveness probe:  return 200 for all statuses
    (liveness failing → pod restart, readiness failing → no traffic)
    """
    status: str = Field(description="'healthy', 'degraded', or 'unhealthy'")
    collection: dict = Field(description="Qdrant collection info")
    config: dict = Field(description="Active configuration summary")
    version: str = Field(default="1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """
    Standard error response shape. All 4xx and 5xx responses use this.
    Having a consistent error schema allows clients to parse errors reliably.
    """
    error: str = Field(description="Error type identifier")
    message: str = Field(description="Human-readable error description")
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for log correlation (from X-Request-Id header)",
    )

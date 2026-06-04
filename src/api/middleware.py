# src/api/middleware.py
#
# FastAPI middleware functions.
#
# WHAT IS MIDDLEWARE?
# Middleware wraps every HTTP request/response that passes through
# FastAPI. It runs BEFORE your route handler and AFTER it returns.
# Think of it as a pipeline that every request travels through.
#
#   HTTP Request
#       ↓
#   timing_middleware (start timer, assign request ID)
#       ↓
#   Your route handler (query, health, etc.)
#       ↓
#   timing_middleware (calculate elapsed, add headers)
#       ↓
#   HTTP Response
#
# WHY MIDDLEWARE OVER DECORATORS?
# A decorator on one route runs only for that route.
# Middleware runs for EVERY request — ideal for cross-cutting concerns:
#   - Request timing (all endpoints)
#   - Request ID assignment (all endpoints)
#   - CORS headers (all endpoints)
#   - Structured logging (all endpoints)
#
# RELATIONSHIP TO OUR 9-LAYER SECURITY PIPELINE:
# FastAPI middleware handles transport-layer security:
#   L4a: JWT Auth (in auth.py, applied per-route via Depends())
#   L4b: Rate limiting (planned: Redis-based counter in middleware)
# Content-layer security (L1 Pydantic, L2 llm-guard, etc.) happens
# INSIDE the route handler after middleware has already run.
#
# PHASE EVOLUTION:
# Module 3: timing + request IDs (this file)
# Phase 6:  add Redis rate limiting middleware
# Phase 8:  add security headers middleware
# Phase 9:  add OpenTelemetry trace propagation middleware

import logging
import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Measures end-to-end request latency and assigns request IDs.

    RESPONSE HEADERS ADDED:
    X-Request-Id:     A unique ID for this request.
                      Used to correlate logs across services.
                      Format: "req-{8-char-uuid}"
                      Example: "req-a3f92b1c"

    X-Process-Time-Ms: Time spent processing in milliseconds.
                       Excludes network round-trip time.
                       Useful for: SLA monitoring, latency alerts.

    HOW IT WORKS:
    BaseHTTPMiddleware provides the `dispatch` method pattern.
    We record start time before calling `call_next(request)`,
    which runs all inner middleware + the route handler.
    After it returns, we calculate elapsed time and add headers.

    WHY REQUEST IDs?
    When a user reports "my query at 3pm failed", you need to find
    that specific request in your logs across multiple services.
    A unique request ID threaded through all log lines makes this
    possible. Phase 9 (OpenTelemetry) extends this to distributed
    tracing with spans across LangGraph nodes.

    WHY NOT JUST USE `time.time()`?
    `time.time()` can go backwards (NTP adjustments, leap seconds).
    `time.perf_counter()` is monotonic — guaranteed to only increase.
    Use perf_counter for latency measurement, time() for timestamps.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Assign a unique ID to this request
        request_id = f"req-{uuid.uuid4().hex[:8]}"

        # Store on request.state so route handlers can access it
        # request.state is a SimpleNamespace — you can add any attribute
        request.state.request_id = request_id

        # Start the timer
        start = time.perf_counter()

        # Run the route handler (and all inner middleware)
        response = await call_next(request)

        # Calculate elapsed time
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Add response headers
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"

        # Structured log line for every request
        # In Phase 9, this becomes an OpenTelemetry span
        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} "
            f"| {elapsed_ms:.1f}ms "
            f"| {request_id}"
        )

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs incoming requests before they're processed.

    Runs BEFORE the route handler — captures the request details
    even if the handler raises an exception.

    WHY SEPARATE FROM TimingMiddleware?
    Single Responsibility: TimingMiddleware measures time,
    RequestLoggingMiddleware logs incoming details.
    You can disable one without affecting the other.
    In production, you might disable verbose request logging
    on health check endpoints (/health) to reduce log noise.
    """

    # Paths to skip logging (health checks create too much noise)
    SKIP_PATHS = {"/health", "/favicon.ico", "/openapi.json"}

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path not in self.SKIP_PATHS:
            # Get request ID from state (set by TimingMiddleware if it ran first)
            request_id = getattr(request.state, "request_id", "unknown")

            logger.debug(
                f"Incoming: {request.method} {request.url.path} "
                f"| client: {request.client.host if request.client else 'unknown'} "
                f"| {request_id}"
            )

        response = await call_next(request)
        return response

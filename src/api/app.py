# src/api/app.py
#
# FastAPI application factory.
#
# WHY A FACTORY FUNCTION (create_app) INSTEAD OF A MODULE-LEVEL app?
#
# OPTION A — Module-level (common but limited):
#   app = FastAPI()
#   app.include_router(...)
#
# OPTION B — Factory function (our approach):
#   def create_app() -> FastAPI:
#       app = FastAPI(...)
#       app.include_router(...)
#       return app
#   app = create_app()
#
# Factory benefits:
#   1. TESTABILITY: Tests can call create_app() to get a fresh app
#      instance with overridden dependencies — no global state pollution
#   2. CONFIGURABILITY: Pass different settings to create_app() for
#      different environments (test, dev, prod)
#   3. CLARITY: All app configuration is in one place, executed in
#      a clear order, not spread across module-level statements
#
# STARTUP EVENTS:
# The @app.on_event("startup") handler runs when uvicorn starts.
# We use it to validate configuration and pre-warm the graph singleton.
# Without pre-warming, the FIRST request would pay the ~200ms compilation
# cost. With pre-warming, that cost is paid at startup.
#
# PHASE EVOLUTION:
# Module 3: basic FastAPI + CORS + timing middleware
# Phase 6:  add Redis connection to startup
# Phase 8:  add rate limiting middleware
# Phase 9:  add OpenTelemetry instrumentation to startup
# Phase 10: add Prometheus /metrics endpoint

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware import RequestLoggingMiddleware, TimingMiddleware
from src.api.routes import health, query

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    CALLED BY:
    - scripts/05_run_server.py: creates the app for uvicorn to serve
    - Tests: creates a fresh app per test with overridden dependencies

    ASSEMBLY ORDER (matters — middleware wraps in reverse registration order):
    1. Create FastAPI instance with metadata
    2. Add middleware (CORS → TimingMiddleware → RequestLogging)
    3. Register startup/shutdown event handlers
    4. Include routers (health, query)
    5. Register global exception handlers
    6. Return the configured app
    """

    # ── FastAPI instance ──────────────────────────────────────────────────────
    app = FastAPI(
        title="Enterprise RAG — Kubernetes IT Operations",
        description=(
            "Production-grade Retrieval-Augmented Generation system for "
            "Kubernetes platform engineering teams. Answers questions from "
            "operational runbooks, incident history, and live cluster data.\n\n"
            "**Stack:** LangGraph · FastAPI · Qdrant · PostgreSQL · Redis · GPT-4o"
        ),
        version="1.0.0",
        # OpenAPI docs are auto-generated from route decorators + Pydantic models
        docs_url="/docs",           # Swagger UI: http://localhost:8000/docs
        redoc_url="/redoc",         # ReDoc UI:    http://localhost:8000/redoc
        openapi_url="/openapi.json",
    )

    # ── CORS Middleware ───────────────────────────────────────────────────────
    # CORS (Cross-Origin Resource Sharing) lets browsers make requests
    # to our API from a different origin (e.g., our Streamlit UI at
    # localhost:8501 calling our API at localhost:8000).
    #
    # allow_origins=["*"]: any origin can call us (dev only)
    # Production: restrict to specific origins:
    #   allow_origins=["https://your-ui.company.com"]
    #
    # MUST be added BEFORE custom middleware — Starlette applies
    # middleware in reverse registration order (last added = outermost).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],            # dev: allow all | prod: restrict
        allow_credentials=True,
        allow_methods=["GET", "POST"],  # only methods our API uses
        allow_headers=["*"],
    )

    # ── Custom Middleware ─────────────────────────────────────────────────────
    # add_middleware() wraps the app — last added = outermost wrapper.
    # Execution order for a request: TimingMiddleware → RequestLogging → route
    # Execution order for a response: route → RequestLogging → TimingMiddleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(TimingMiddleware)

    # ── Startup event ─────────────────────────────────────────────────────────
    # Runs once when uvicorn starts (before accepting any requests).
    # Pre-warms expensive singletons so the first request isn't slow.
    @app.on_event("startup")
    async def startup():
        logger.info("=" * 60)
        logger.info("Enterprise RAG API starting up...")

        # Validate config (fail fast if OPENAI_API_KEY missing)
        from src.config import settings
        try:
            settings.validate()
            logger.info(f"  Config OK | model={settings.llm_model} | "
                        f"collection={settings.collection_name}")
        except ValueError as e:
            logger.error(f"  Config INVALID: {e}")
            raise  # abort startup

        # Pre-compile LangGraph (pays the ~200ms compilation cost now,
        # not on the first user request)
        from src.api.dependencies import get_graph, get_qdrant
        try:
            get_graph()
            logger.info("  LangGraph graph: compiled")
        except Exception as e:
            logger.error(f"  LangGraph graph: FAILED — {e}")

        # Pre-connect to Qdrant
        try:
            client = get_qdrant()
            from src.module1_naive_rag.collection import collection_exists
            exists = collection_exists(client)
            logger.info(f"  Qdrant: connected | collection exists: {exists}")
            if not exists:
                logger.warning("  Run: python scripts/01_setup.py to ingest documents")
        except Exception as e:
            logger.error(f"  Qdrant: FAILED — {e}")

        logger.info("Enterprise RAG API ready.")
        logger.info("=" * 60)

    # ── Shutdown event ────────────────────────────────────────────────────────
    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Enterprise RAG API shutting down...")

    # ── Routers ───────────────────────────────────────────────────────────────
    # include_router() mounts all routes from a router onto the app.
    # The prefix is prepended to all route paths in the router.
    app.include_router(health.router)    # GET /health
    app.include_router(query.router)     # POST /api/v1/query

    # ── Global exception handlers ─────────────────────────────────────────────
    # Catch unhandled exceptions and return a clean JSON error response
    # rather than an HTML error page or raw traceback.
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            f"Unhandled exception | req='{request_id}' | "
            f"{type(exc).__name__}: {exc}"
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Check server logs.",
                "request_id": request_id,
            },
        )

    return app


# ── Module-level app instance ─────────────────────────────────────────────────
# Created once at module import time.
# uvicorn imports this as: uvicorn.run("src.api.app:app", ...)
# Tests override dependencies via: app.dependency_overrides[get_graph] = mock_fn
app = create_app()

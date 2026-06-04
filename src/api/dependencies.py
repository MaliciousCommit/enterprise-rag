# src/api/dependencies.py
#
# FastAPI dependency injection providers.
#
# WHAT IS DEPENDENCY INJECTION?
# DI is FastAPI's mechanism for sharing expensive resources across requests
# without creating them on every request.
#
# HOW IT WORKS:
# You declare a function that returns a resource.
# Route handlers declare that resource as a parameter using Depends().
# FastAPI calls the function once (or per-request depending on scope)
# and injects the result into your handler.
#
# Example:
#   async def query(graph=Depends(get_graph)):
#       # `graph` is already compiled, ready to use
#
# WHY NOT JUST USE MODULE-LEVEL GLOBALS?
# Globals work but have problems:
#   1. Hard to test: can't swap for a mock in unit tests
#   2. Lifecycle unclear: when does it get initialised?
#   3. Thread safety unclear: are concurrent requests safe?
#
# DI with lru_cache solves all three:
#   1. Tests can override dependencies with app.dependency_overrides
#   2. First request triggers initialisation (lazy, explicit)
#   3. lru_cache is thread-safe (uses a lock internally)
#
# SINGLETON PATTERN:
# @lru_cache(maxsize=1) on a function with no arguments = singleton.
# First call: executes the function, stores result in cache.
# All subsequent calls: return the cached result instantly.
#
# This means:
#   - LangGraph graph compiled ONCE at startup: ~200ms cost paid once
#   - QdrantClient created ONCE: connection pool shared across requests
#   - No per-request overhead for these expensive objects
#
# PHASE EVOLUTION:
# Module 3: MemorySaver graph + QdrantClient
# Phase 5:  PostgresSaver graph (requires DB connection)
# Phase 6:  Redis client for caching
# Phase 9:  LangSmith tracer, Prometheus metrics

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.config import settings

logger = logging.getLogger(__name__)


# ── Graph Singleton ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_graph():
    """
    Build and cache the LangGraph StateGraph (one per process).

    WHY lru_cache AND NOT A MODULE-LEVEL VARIABLE?
    A module-level `graph = build_graph()` runs at import time.
    If the Qdrant connection fails at import, the entire module fails
    to load. Lazy initialisation via lru_cache means the failure
    happens at request time with a clear error, not silently at startup.

    COMPILATION COST:
    build_graph() compiles the StateGraph topology (~50ms),
    creates a MemorySaver checkpointer, and validates all node connections.
    This cost is paid once. All subsequent requests reuse the compiled graph.

    THREAD SAFETY:
    lru_cache uses a lock internally. Concurrent first requests won't
    compile the graph twice — only one wins the lock, compiles,
    stores the result, and all others get the cached version.

    PHASE 5 UPGRADE PATH:
    Replace MemorySaver with PostgresSaver here.
    The rest of the codebase doesn't change — only this function.
    That's the power of the DI pattern.
    """
    logger.info("Building LangGraph StateGraph (first request)...")
    from src.module2_system_arch.graph import build_graph
    graph = build_graph(use_memory_checkpointer=True)
    logger.info("LangGraph StateGraph compiled and cached.")
    return graph


# ── Qdrant Client Singleton ───────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_qdrant():
    """
    Create and cache the Qdrant client (one per process).

    QdrantClient internally manages a connection pool.
    Creating one client and reusing it across all requests
    is significantly more efficient than creating a new client
    per request (avoids TCP handshake overhead).

    In Phase 10 (Kubernetes):
    Each pod has its own QdrantClient instance.
    Qdrant is accessed by Kubernetes Service DNS:
      host="qdrant.qdrant.svc.cluster.local"
    Multiple pods = multiple clients, all hitting the same Qdrant cluster.
    Qdrant handles the load balancing internally.
    """
    logger.info("Creating Qdrant client (first request)...")
    from src.module1_naive_rag.collection import get_qdrant_client
    client = get_qdrant_client()
    logger.info(f"Qdrant client created: {settings.qdrant_host}:{settings.qdrant_port}")
    return client


# ── Settings Singleton ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings():
    """
    Return the validated application settings.

    Validates at first call — fails fast if OPENAI_API_KEY is missing
    rather than letting the first actual LLM call fail.
    """
    settings.validate()
    return settings


# ── Type aliases for cleaner route signatures ─────────────────────────────────
#
# Instead of writing: graph=Depends(get_graph) in every route,
# define typed aliases here and use them directly.
#
# BEFORE (verbose):
#   async def query(graph=Depends(get_graph), client=Depends(get_qdrant)):
#
# AFTER (clean):
#   async def query(graph: GraphDep, client: QdrantDep):
#
# Annotated[T, Depends(f)] is FastAPI's way to attach DI metadata
# to a type hint. The route sees it as type T but FastAPI injects
# the result of f() automatically.

GraphDep    = Annotated[object, Depends(get_graph)]
QdrantDep   = Annotated[object, Depends(get_qdrant)]
SettingsDep = Annotated[object, Depends(get_settings)]

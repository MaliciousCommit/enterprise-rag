# src/api/routes/health.py
#
# Health check endpoint: GET /health
#
# WHY HEALTH CHECKS MATTER:
# In Kubernetes (Phase 10), two probes check pod health:
#
# Liveness probe: "Is the pod still alive?"
#   If it fails → Kubernetes RESTARTS the pod
#   Our liveness check: FastAPI is responding + no deadlock
#
# Readiness probe: "Is the pod ready to receive traffic?"
#   If it fails → Kubernetes STOPS routing traffic to this pod
#   Our readiness check: FastAPI + Qdrant both healthy + collection exists
#
# Example Kubernetes probe config:
#   livenessProbe:
#     httpGet:
#       path: /health
#       port: 8000
#     initialDelaySeconds: 10
#     periodSeconds: 30
#   readinessProbe:
#     httpGet:
#       path: /health
#       port: 8000
#     initialDelaySeconds: 20
#     periodSeconds: 10
#
# The distinction matters:
# A pod that's alive but not ready (Qdrant temporarily unavailable)
# should NOT restart (it would also fail after restart).
# It should stop receiving traffic until Qdrant recovers.

import logging

from fastapi import APIRouter, Request

from src.api.dependencies import QdrantDep, SettingsDep
from src.api.models import HealthResponse

logger = logging.getLogger(__name__)

# APIRouter groups related endpoints.
# The prefix and tags are set in app.py when including the router.
router = APIRouter(tags=["operations"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns system health status. Used by Kubernetes liveness/readiness probes.",
)
async def health_check(
    request: Request,
    client: QdrantDep,
    settings: SettingsDep,
) -> HealthResponse:
    """
    Check system health and return component status.

    STATUS LOGIC:
    "healthy"   → Qdrant reachable + collection exists + has points
    "degraded"  → Qdrant reachable but collection empty or missing
    "unhealthy" → Qdrant unreachable (exception during check)

    HTTP STATUS CODE:
    200 for healthy and degraded (pod stays in rotation, may serve from cache)
    503 for unhealthy (pod removed from rotation)

    This endpoint does NOT auth-guard intentionally.
    Kubernetes probes don't carry JWT tokens.
    The health check exposes only non-sensitive operational metadata.
    """
    from src.module1_naive_rag.collection import (
        collection_exists,
        get_collection_info,
    )

    try:
        if not collection_exists(client):
            logger.warning("Health: collection missing")
            return HealthResponse(
                status="degraded",
                collection={
                    "exists": False,
                    "message": "Collection not found. Run: python scripts/01_setup.py",
                },
                config=_config_summary(settings),
            )

        info = get_collection_info(client)

        if info["points_count"] == 0:
            logger.warning("Health: collection empty")
            return HealthResponse(
                status="degraded",
                collection={**info, "message": "Collection exists but is empty. Run: python scripts/01_setup.py"},
                config=_config_summary(settings),
            )

        logger.debug(f"Health: OK ({info['points_count']} points)")
        return HealthResponse(
            status="healthy",
            collection=info,
            config=_config_summary(settings),
        )

    except Exception as e:
        logger.error(f"Health: Qdrant unreachable: {e}")
        return HealthResponse(
            status="unhealthy",
            collection={"error": str(e)},
            config=_config_summary(settings),
        )


def _config_summary(settings) -> dict:
    """Non-sensitive config summary for health response."""
    return {
        "embedding_model": settings.embedding_model,
        "llm_model":       settings.llm_model,
        "collection":      settings.collection_name,
        "retrieval_k":     settings.retrieval_k,
    }

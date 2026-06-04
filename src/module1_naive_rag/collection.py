# src/module1_naive_rag/collection.py
#
# Qdrant collection lifecycle management.
#
# RESPONSIBILITIES:
#   - Get a configured QdrantClient (the connection to Qdrant)
#   - Create the collection with the right vector configuration
#   - Provide collection introspection (point count, status)
#
# WHY A SEPARATE MODULE FOR THIS?
# Qdrant client creation has several options (host/port, gRPC vs REST,
# API key for cloud, TLS in production). Centralising it here means
# every module that needs Qdrant calls get_qdrant_client() and gets
# a correctly configured instance.
#
# In Phase 10 (Kubernetes), we swap this to use gRPC (port 6334)
# for 30% lower latency — the change happens only in this file.
#
# PHASE EVOLUTION:
# Module 1: Single dense vector collection
# Phase 3:  Recreate with BOTH dense and sparse vectors (hybrid search)
# Phase 10: Add replication_factor=2 and shard_number=2 for HA

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    CollectionStatus,
    OptimizersConfigDiff,
)
from src.config import settings

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """
    Create and return a configured Qdrant client.

    Uses REST API by default (port 6333).
    In Phase 10 we switch to gRPC (port 6334) for production performance.

    QdrantClient is NOT a singleton here — each call creates a new connection.
    In Phase 3 (FastAPI), we create a single client at app startup and inject
    it via FastAPI's dependency injection system (avoids connection churn).

    QDRANT_API_KEY:
    None for local Docker (no auth needed).
    Required for Qdrant Cloud (production).
    We read it from settings.qdrant_api_key which reads from env.
    """
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,  # None = no auth (local dev)
        timeout=30,
        https = False,                                                  # seconds — increase for slow networks
    )
    logger.debug(
        f"Qdrant client created: {settings.qdrant_host}:{settings.qdrant_port}"
    )
    return client


def create_collection(
    client: QdrantClient,
    force_recreate: bool = False,
) -> None:
    """
    Create the Qdrant collection for K8s documentation chunks.

    If the collection already exists and force_recreate=False, this is a no-op.
    If force_recreate=True, the existing collection is deleted first.
    Use force_recreate=True when you want to re-ingest with a different
    embedding model or change vector dimensions.

    WHAT THIS CREATES INTERNALLY IN QDRANT:
    1. A named collection ("k8s_docs_m1")
    2. A 1536-dimensional vector space with cosine distance metric
    3. An HNSW graph index (built incrementally as points are upserted)
    4. A WAL (Write-Ahead Log) for crash recovery
    5. On-disk segment storage for payloads (text + metadata)

    HNSW DEFAULT PARAMETERS (Qdrant defaults, sufficient for our 10k scale):
    m=16            : connections per node (higher = better recall, more memory)
    ef_construct=100: candidates during index build (higher = better recall, slower build)
    Phase 10 will tune these for 1M+ vectors.

    Args:
        client: A QdrantClient instance from get_qdrant_client()
        force_recreate: If True, delete existing collection first

    Raises:
        RuntimeError: If collection creation fails
    """
    collection_name = settings.collection_name

    # Check if collection already exists
    existing_collections = [c.name for c in client.get_collections().collections]
    exists = collection_name in existing_collections

    if exists and not force_recreate:
        # Get info about the existing collection for logging
        info = client.get_collection(collection_name)
        logger.info(
            f"Collection '{collection_name}' already exists "
            f"({info.points_count} points). Skipping creation. "
            f"Use force_recreate=True to rebuild."
        )
        return

    if exists and force_recreate:
        logger.warning(
            f"Deleting existing collection '{collection_name}' "
            f"(force_recreate=True). ALL INDEXED DATA WILL BE LOST."
        )
        client.delete_collection(collection_name)
        logger.info(f"Collection '{collection_name}' deleted.")

    # Create the collection
    # This is the most important configuration decision in the system:
    # we choose the vector dimension and distance metric here.
    # These CANNOT be changed after creation without full recreation.
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=settings.embedding_dim,     # 1536 for text-embedding-3-small
            distance=Distance.COSINE,         # cosine similarity for text embeddings
            # on_disk=False                  # keep vectors in RAM for speed
            #                                # Set on_disk=True for >1M vectors
        ),
        # Optimizer configuration — controls how Qdrant builds/merges segments.
        # Default settings work well up to ~100k points.
        # Phase 10 will add: indexing_threshold, memmap_threshold
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20_000,  # build HNSW index when segment exceeds 20k points
                                        # smaller = faster search but more memory overhead
                                        # larger = slower search startup but less overhead
        ),
    )

    logger.info(
        f"Created collection '{collection_name}': "
        f"{settings.embedding_dim}d, COSINE distance, HNSW index"
    )


def get_collection_info(client: QdrantClient) -> dict:
    """
    Return a summary of the collection's current state.
    Used in scripts to verify ingestion completed correctly.

    Returns a dict with:
        name:          collection name
        status:        "green" (ready), "yellow" (optimizing), "red" (error)
        points_count:  total number of vectors indexed
        vectors_count: should equal points_count for single-vector collections
        segments_count: number of internal storage segments
    """
    info = client.get_collection(settings.collection_name)

    return {
        "name": settings.collection_name,
        "status": info.status.value if info.status else "unknown",
        "points_count": info.points_count or 0,
        "vectors_count": getattr(info, "vectors_count", None) or info.points_count or 0,
        "segments_count": info.segments_count or 0,
        "config": {
            "vector_size": settings.embedding_dim,
            "distance": "COSINE",
        },
    }


def collection_exists(client: QdrantClient) -> bool:
    """Check if the configured collection exists in Qdrant."""
    names = [c.name for c in client.get_collections().collections]
    return settings.collection_name in names


def delete_collection(client: QdrantClient) -> None:
    """
    Delete the collection and all its data.
    Used in tests and when switching embedding models.
    IRREVERSIBLE — all indexed data is lost.
    """
    if collection_exists(client):
        client.delete_collection(settings.collection_name)
        logger.warning(f"Collection '{settings.collection_name}' deleted permanently.")
    else:
        logger.info(f"Collection '{settings.collection_name}' does not exist. Nothing to delete.")

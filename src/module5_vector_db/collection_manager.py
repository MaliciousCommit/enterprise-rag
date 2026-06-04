# src/module5_vector_db/collection_manager.py
#
# Production-grade Qdrant collection management.
#
# This module replaces the Module 1 collection.py with a version
# that properly configures:
#   1. HNSW parameters (m, ef_construct) for the right recall/speed tradeoff
#   2. Payload indexes for fast filtered search
#   3. INT8 scalar quantization for memory efficiency at scale
#   4. On-disk storage option for large collections
#
# WHY THIS MATTERS:
# Module 1's create_collection() used all defaults. Defaults are fine
# for a demo. For production with 10k-1M vectors, tuning matters.
#
# The biggest practical win: PAYLOAD INDEXES.
# Without them, a filter like "only return docs for namespace=prod"
# forces Qdrant to scan ALL payloads then filter. With an index,
# it's an O(1) lookup. At 10k vectors, this is the difference between
# ~2ms and ~0.1ms for filtered queries.
#
# PHASE EVOLUTION:
# Module 1: default collection, no payload indexes
# Module 5: tuned HNSW + payload indexes (this file)
# Phase 3:  add sparse vector config for BM25 hybrid search

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

from src.config import settings

logger = logging.getLogger(__name__)


def create_production_collection(
    client: QdrantClient,
    collection_name: Optional[str] = None,
    force_recreate: bool = False,
    enable_quantization: bool = False,   # enable for >100k vectors
    on_disk: bool = False,               # enable for >500k vectors
) -> None:
    """
    Create a properly-tuned Qdrant collection for production use.

    PARAMETER CHOICES EXPLAINED:

    m=16 (HNSW connections per node):
        The default. Provides 98%+ recall at our scale.
        Increase to 32 for better recall at >1M vectors.
        Each unit costs ~m × 4 bytes × N_vectors in RAM.
        At N=10k: m=16 costs 640KB. Negligible.

    ef_construct=200 (candidates during build):
        Higher than the default (100). We pay this cost once during
        ingestion. The resulting graph quality is permanently better.
        Rule of thumb: ef_construct = 2-4 × m = 32-64 minimum.
        We use 200 for 10k chunks to ensure high-quality graph.

    indexing_threshold=20000:
        Qdrant builds the HNSW index when a segment exceeds this many
        points. Below this threshold, brute force is faster than HNSW.
        At 18 chunks: we're below threshold → Qdrant uses brute force.
        At 10k chunks: we're above → HNSW kicks in.
        This is correct behaviour — HNSW overhead isn't worth it for tiny collections.

    Payload indexes (created separately after collection):
        Without index: filter("team=platform") → full scan
        With index:    filter("team=platform") → direct lookup
        We index: doc_type, team, k8s_version (our most common filters)
    """
    collection_name = collection_name or settings.collection_name

    # Check if collection already exists
    existing = [c.name for c in client.get_collections().collections]
    exists = collection_name in existing

    if exists and not force_recreate:
        logger.info(f"Collection '{collection_name}' exists. Skipping. Use force_recreate=True.")
        return

    if exists and force_recreate:
        client.delete_collection(collection_name)
        logger.info(f"Deleted '{collection_name}' (force_recreate=True)")

    # ── Vector configuration ───────────────────────────────────────────────────
    vectors_config = VectorParams(
        size=settings.embedding_dim,       # 1536 for text-embedding-3-small
        distance=Distance.COSINE,          # cosine similarity for text embeddings

        # on_disk=True: keeps vectors on disk instead of RAM
        # Use for >500k vectors where RAM is limited.
        # Adds disk read latency (~5-20ms per search depending on disk speed)
        # We set to False for our 10k-chunk knowledge base.
        on_disk=on_disk,
    )

    # ── HNSW configuration ─────────────────────────────────────────────────────
    hnsw_config = HnswConfigDiff(
        m=16,                # connections per node (default=16, range: 4-64)
                             # higher = better recall, more RAM, slower build
                             # 16 is optimal for most RAG use cases

        ef_construct=200,    # candidates during index build (default=100)
                             # we use 200 for better graph quality
                             # cost is paid ONCE during ingestion, not at search time

        # full_scan_threshold: below this many vectors, use brute force
        # Qdrant decides automatically based on segment size
        # We don't override this — Qdrant's defaults are excellent
    )

    # ── Optimiser configuration ────────────────────────────────────────────────
    optimizers_config = OptimizersConfigDiff(
        # indexing_threshold: when a segment has >= this many vectors,
        # build the HNSW index. Below this: brute force is faster.
        # At 18 chunks we'll still use brute force (correct for tiny collections).
        # At 10k chunks the HNSW index will be built automatically.
        indexing_threshold=10_000,

        # memmap_threshold: vectors above this count are mmap'd from disk
        # rather than kept in RAM. Only relevant if on_disk=False.
        # Default (None): Qdrant decides based on available memory.
    )

    # ── Quantization (optional, for memory reduction at scale) ─────────────────
    # INT8 scalar quantization: float32 (4 bytes) → int8 (1 byte)
    # Memory reduction: 4x for vector storage
    # Accuracy loss: ~1% recall (usually acceptable for RAG)
    # Enable for: >100k vectors where RAM is constrained
    quantization_config = None
    if enable_quantization:
        quantization_config = ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                # quantile=0.99: vectors beyond the 99th percentile
                # of the value range are clipped. Prevents outliers
                # from compressing the range poorly.
                quantile=0.99,
                # always_ram=True: keep the quantized vectors in RAM
                # even if on_disk=True for the full vectors.
                # Quantized vectors are 4x smaller — cheap to keep in RAM.
                # The full float32 vectors on disk are only read for rescoring.
                always_ram=True,
            )
        )
        logger.info("Quantization: INT8 scalar (4x memory reduction, ~1% recall loss)")

    # ── Create the collection ──────────────────────────────────────────────────
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        hnsw_config=hnsw_config,
        optimizers_config=optimizers_config,
        quantization_config=quantization_config,
    )

    logger.info(
        f"Created collection '{collection_name}': "
        f"{settings.embedding_dim}d COSINE | "
        f"HNSW m=16 ef_construct=200 | "
        f"{'quantized INT8' if enable_quantization else 'float32'} | "
        f"{'on-disk' if on_disk else 'in-RAM'}"
    )


def create_payload_indexes(
    client: QdrantClient,
    collection_name: Optional[str] = None,
) -> None:
    """
    Create payload indexes for fast filtered search.

    WITHOUT A PAYLOAD INDEX:
    filter={"doc_type": "runbook"} requires Qdrant to:
      1. Load ALL 10k payloads from disk
      2. Check each one: does payload["doc_type"] == "runbook"?
      3. Return only the matching vectors for ANN search
    This is O(N) payload reads — slow at scale.

    WITH A PAYLOAD INDEX:
    Qdrant maintains an inverted index for the field.
    filter={"doc_type": "runbook"} becomes an O(1) lookup.
    Qdrant directly fetches the matching vector IDs and searches only those.
    This is the difference between ~2ms and ~0.1ms for filtered queries.

    WHICH FIELDS TO INDEX:
    Only index fields you actually filter on. Each index costs:
    - Disk space: ~20-50 bytes per indexed value per vector
    - Build time: small one-time cost
    - Memory: small ongoing cost for the index structure

    Our indexed fields:
      doc_type:    "runbook" | "guide" | "postmortem"
                   Most common filter: "show only runbooks"
      team:        "platform" | "security" | "networking"
                   Multi-tenant filter: each team sees their own docs
      k8s_version: "1.29" | "1.28" | "1.27"
                   Critical for Phase 5 CRAG version-aware retrieval
                   (fixes the Module 1 Socratic Q5 about stale docs)
    """
    collection_name = collection_name or settings.collection_name

    indexes_to_create = [
        ("doc_type",    PayloadSchemaType.KEYWORD),  # exact string match
        ("team",        PayloadSchemaType.KEYWORD),  # exact string match
        ("k8s_version", PayloadSchemaType.KEYWORD),  # exact string match
        # Future indexes for Phase 3 (time-based filtering):
        # ("created_at", PayloadSchemaType.DATETIME),
    ]

    for field_name, field_type in indexes_to_create:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
            logger.info(f"Created payload index: {field_name} ({field_type.value})")
        except Exception as e:
            # Index may already exist — this is not an error
            if "already exists" in str(e).lower():
                logger.debug(f"Payload index '{field_name}' already exists")
            else:
                logger.warning(f"Failed to create index for '{field_name}': {e}")


def get_collection_stats(client: QdrantClient, collection_name: Optional[str] = None) -> dict:
    """
    Return comprehensive stats about the collection.
    Includes HNSW config, vector counts, and segment details.
    """
    collection_name = collection_name or settings.collection_name
    info = client.get_collection(collection_name)

    # Extract HNSW config from the collection info
    hnsw_config = {}
    try:
        params = info.config.params
        if hasattr(params, 'vectors') and hasattr(params.vectors, 'hnsw_config'):
            hc = params.vectors.hnsw_config
            if hc:
                hnsw_config = {"m": hc.m, "ef_construct": hc.ef_construct}
    except Exception:
        pass

    return {
        "name":          collection_name,
        "status":        info.status.value if info.status else "unknown",
        "points_count":  info.points_count or 0,
        "segments_count": info.segments_count or 0,
        "disk_data_size": getattr(info, "disk_data_size", None),
        "ram_data_size":  getattr(info, "ram_data_size", None),
        "hnsw_config":   hnsw_config,
        "vector_size":   settings.embedding_dim,
        "distance":      "COSINE",
    }

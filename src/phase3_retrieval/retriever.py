# src/phase3_retrieval/retriever.py
#
# Phase3Retriever: the complete advanced retrieval pipeline.
#
# EXECUTION ORDER:
#
# 1. HyDE (optional):
#    question → GPT-4o-mini → 3 hypothetical answers
#    embed each → average → hyde_vector
#    Latency: ~600ms (LLM + 3 embeddings)
#    Benefit: +15-20% recall for vague questions
#
# 2. Dense search (always):
#    hyde_vector (or question_vector) → Qdrant HNSW search
#    Returns top-20 candidates with cosine scores
#    Latency: ~2ms (HNSW is fast)
#
# 3. BM25 sparse search (always):
#    question → BM25 tokenize → rank all docs → top-20
#    Returns top-20 candidates with BM25 scores
#    Latency: ~5ms (in-memory, pure Python)
#
# 4. RRF fusion:
#    Merge dense-20 + sparse-20 lists using RRF(k=60)
#    Returns top-20 by RRF score (may include docs from either or both lists)
#    Latency: ~1ms (Python dict operations)
#
# 5. Fetch texts for reranking:
#    Retrieve texts for top-20 RRF candidates (from BM25 index — no Qdrant call)
#    Latency: ~1ms
#
# 6. Cross-encoder reranking:
#    Score (question, chunk_text) for each of the 20 candidates
#    Sort by relevance, return top-5
#    Latency: ~200ms (20 pairs, CPU inference)
#
# TOTAL LATENCY (cold):
#   HyDE on: ~800ms (HyDE + dense + BM25 + RRF + rerank)
#   HyDE off: ~210ms (dense + BM25 + RRF + rerank)
#   Phase 2 baseline: ~150ms (embed + search only)
#
# LATENCY vs QUALITY TRADEOFF:
#   Phase 3 is 1.4-5x slower than Phase 2 retrieval alone.
#   BUT Phase 3 retrieves significantly better chunks.
#   Better chunks → shorter, more precise answers → fewer tokens → lower cost.
#   And Phase 6 Redis cache eliminates retrieval latency for repeated queries.

import logging
from typing import Optional

from qdrant_client import QdrantClient

from src.config import settings
from src.phase3_retrieval.bm25 import get_bm25_index
from src.phase3_retrieval.reranker import RankedChunk, rerank
from src.phase3_retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

# Retrieval configuration
DENSE_CANDIDATES  = 20   # Dense search returns this many before RRF
SPARSE_CANDIDATES = 20   # BM25 returns this many before RRF
RRF_K             = 60   # RRF constant
RERANK_CANDIDATES = 20   # How many to pass to cross-encoder (top RRF results)
FINAL_TOP_K       = 5    # Final chunks passed to LLM for generation


async def phase3_retrieve(
    question:         str,
    client:           QdrantClient,
    collection_name:  str           = None,
    use_hyde:         bool          = True,
    use_reranking:    bool          = True,
    top_k:            int           = FINAL_TOP_K,
) -> list[RankedChunk]:
    """
    Full Phase 3 retrieval: HyDE + Hybrid Search + RRF + Reranking.

    This is the drop-in replacement for Module 1's retrieve() function.
    Signature is compatible: same inputs, richer outputs.

    Args:
        question:        User's original question
        client:          Qdrant client
        collection_name: Collection to search
        use_hyde:        Whether to run HyDE query expansion
        use_reranking:   Whether to run cross-encoder reranking
        top_k:           Final number of chunks to return

    Returns:
        List of RankedChunk, sorted by rerank_score (or rrf_score if no reranking)
    """
    collection_name = collection_name or settings.collection_name

    # ── Step 1: HyDE or direct embedding ───────────────────────────────────
    if use_hyde:
        logger.info("HyDE: generating hypothetical answers...")
        from src.phase3_retrieval.hyde import generate_hyde_embedding
        query_vector = await generate_hyde_embedding(question)
        logger.info("HyDE: embedding ready")
    else:
        # Direct embedding (Phase 2 approach)
        from src.module1_naive_rag.embeddings import embed_text
        query_vector = embed_text(question)

    # ── Step 2: Dense search ────────────────────────────────────────────────
    logger.info(f"Dense search: top-{DENSE_CANDIDATES}...")
    dense_results = client.query_points(
        collection_name = collection_name,
        query           = query_vector,
        limit           = DENSE_CANDIDATES,
        with_payload    = True,
        with_vectors    = False,
    ).points

    dense_ids    = [str(r.id)    for r in dense_results]
    dense_scores = {str(r.id):   r.score for r in dense_results}
    dense_payloads = {str(r.id): (r.payload or {}) for r in dense_results}

    logger.info(
        f"Dense: {len(dense_ids)} results | "
        f"best={max(dense_scores.values(), default=0):.4f}"
    )

    # ── Step 3: BM25 sparse search ──────────────────────────────────────────
    logger.info("BM25 search...")
    bm25_index = get_bm25_index(
        client          = client,
        collection_name = collection_name,
    )
    bm25_results = bm25_index.search(question, k=SPARSE_CANDIDATES)
    sparse_ids   = [doc_id for doc_id, _ in bm25_results]
    sparse_scores = {doc_id: score for doc_id, score in bm25_results}

    logger.info(
        f"BM25: {len(sparse_ids)} results | "
        f"best={max(sparse_scores.values(), default=0):.4f}"
    )

    # ── Step 4: RRF fusion ──────────────────────────────────────────────────
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids])
    top_rrf_ids = [doc_id for doc_id, _ in fused[:RERANK_CANDIDATES]]
    rrf_scores  = {doc_id: score for doc_id, score in fused}

    logger.info(f"RRF: fused to {len(fused)} unique docs, top-{len(top_rrf_ids)} for reranking")

    # ── Step 5: Build RankedChunk objects for reranking ─────────────────────
    # Text comes from: dense payload (if available) OR BM25 corpus
    chunks: list[RankedChunk] = []

    for doc_id in top_rrf_ids:
        text = None
        source = ""
        heading_path = ""
        doc_type = "runbook"
        chunk_index = 0

        # Try dense results first (already have payload)
        if doc_id in dense_payloads:
            payload      = dense_payloads[doc_id]
            text         = payload.get("text", "")
            source       = payload.get("source", "")
            heading_path = payload.get("heading_path", "")
            doc_type     = payload.get("doc_type", "runbook")
            chunk_index  = payload.get("chunk_index", 0)

        # Fall back to BM25 corpus
        if not text:
            text = bm25_index.get_text(doc_id) or ""

        if not text:
            logger.debug(f"Skipping {doc_id}: no text found")
            continue

        chunks.append(RankedChunk(
            text         = text,
            source       = source,
            point_id     = doc_id,
            dense_score  = dense_scores.get(doc_id, 0.0),
            rrf_score    = rrf_scores.get(doc_id, 0.0),
            rerank_score = rrf_scores.get(doc_id, 0.0),  # will be overwritten by reranker
            heading_path = heading_path,
            doc_type     = doc_type,
            chunk_index  = chunk_index,
        ))

    # ── Step 6: Cross-encoder reranking ────────────────────────────────────
    if use_reranking and chunks:
        logger.info(f"Reranking {len(chunks)} chunks...")
        chunks = rerank(question, chunks, top_k=top_k)
    else:
        # Sort by RRF score if reranking disabled
        chunks = sorted(chunks, key=lambda c: c.rrf_score, reverse=True)[:top_k]

    logger.info(
        f"Phase3 retrieval complete: {len(chunks)} chunks | "
        f"top rerank score: {chunks[0].rerank_score:.4f}" if chunks else "0 chunks"
    )

    return chunks


def phase3_format_context(chunks: list[RankedChunk]) -> str:
    """
    Format Phase 3 chunks as XML-spotlighted context for LLM generation.

    Adds rerank_score to the XML attributes (not present in Phase 2).
    The LLM can see which chunks were scored most relevant.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f'<doc id="{i}" source="{chunk.source}" '
            f'relevance="{chunk.rerank_score:.3f}" '
            f'path="{chunk.heading_path}">\n'
            f'{chunk.text}\n'
            f'</doc>'
        )
    return "\n\n".join(parts)

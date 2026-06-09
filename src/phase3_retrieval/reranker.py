# src/phase3_retrieval/reranker.py
#
# Cross-encoder reranking — the precision filter of Phase 3.
#
# WHY RERANKING IS NECESSARY:
# Vector search (dense or sparse) computes similarity INDEPENDENTLY:
#   embed(query) → similarity(query_vec, doc_vec) for each doc
#
# The query and document are never seen TOGETHER.
# The model doesn't know "given THIS specific question, is THIS specific chunk useful?"
#
# A cross-encoder scores the (query, document) PAIR jointly:
#   cross_encoder(query, document) → relevance score
#
# This is dramatically more accurate but O(K) inference calls instead of O(1).
# That's why we use it AFTER vector search, not instead of it:
#
#   Vector search: fast, retrieves top-20 candidates in ~2ms
#   Cross-encoder: slow but precise, rescores 20 candidates in ~200ms
#   Result: top-5 highly relevant chunks for generation
#
# THE LATENCY TRADEOFF:
#   Phase 2 retrieve: ~150ms (embed + search)
#   Phase 3 retrieve: ~350ms (embed + search + rerank 20 chunks)
#   Phase 6 cache:    ~2ms (Redis hit — pays the 350ms cost once)
#   Net effect: first query slower, repeated queries 0ms
#
# MODEL CHOICE: cross-encoder/ms-marco-MiniLM-L-6-v2
#   - 80MB download (one time, cached locally)
#   - ~200ms for 20 (query, chunk) pairs on CPU
#   - Trained on MS MARCO passage ranking (close to our use case)
#   - Free, runs locally, no API costs
#
# PRODUCTION ALTERNATIVES:
#   Cohere Rerank:    best quality, $0.001/1k requests, API latency ~100ms
#   Voyage AI Rerank: strong quality, similar pricing
#   BGE-reranker-v2:  open source, better than MiniLM, needs GPU for speed
#
# PHASE EVOLUTION:
# Phase 3: local cross-encoder (this file)
# Phase 9: swap to Cohere Rerank API + track latency in Prometheus

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RankedChunk:
    """A document chunk with its cross-encoder relevance score."""
    text:           str
    source:         str
    point_id:       str
    dense_score:    float   # original cosine similarity from vector search
    rerank_score:   float   # cross-encoder relevance score (higher = more relevant)
    rrf_score:      float   # RRF fusion score before reranking
    heading_path:   str     = ""
    doc_type:       str     = "runbook"
    chunk_index:    int     = 0


@lru_cache(maxsize=1)
def _load_cross_encoder():
    """
    Load and cache the cross-encoder model (one per process).

    lru_cache(maxsize=1) means:
    - First call: downloads model if needed, loads into memory (~200MB RAM)
    - All subsequent calls: returns the cached model instantly

    Model loading takes ~2-3 seconds.
    We pay this cost once at startup (or on first reranking request).

    DOWNLOAD:
    First run: model downloads from HuggingFace Hub to ~/.cache/huggingface/
    Subsequent runs: loaded from disk cache (~80MB)
    """
    logger.info(f"Loading cross-encoder: {RERANKER_MODEL} (first load ~2-3s)")
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(RERANKER_MODEL, max_length=512)
        logger.info(f"Cross-encoder loaded: {RERANKER_MODEL}")
        return model
    except Exception as e:
        logger.error(f"Failed to load cross-encoder: {e}")
        raise


def rerank(
    query:      str,
    chunks:     list[RankedChunk],
    top_k:      int = 5,
) -> list[RankedChunk]:
    """
    Rerank chunks using cross-encoder and return top-k.

    HOW IT WORKS:
    1. For each chunk, create a (query, chunk_text) pair
    2. Pass all pairs to the cross-encoder in one batch
    3. Cross-encoder outputs a relevance score for each pair
    4. Sort by score descending, return top_k

    BATCHING:
    CrossEncoder.predict() accepts a list of pairs — it batches internally.
    One model call for all 20 chunks is faster than 20 separate calls.

    MAX_LENGTH=512:
    Cross-encoder truncates input to 512 tokens.
    Our chunks average ~100-300 tokens.
    Query is ~10-20 tokens.
    Total per pair: ~120-320 tokens — well within limit.

    Args:
        query:  The user's original question
        chunks: Candidate chunks from hybrid retrieval (typically 20)
        top_k:  How many to return after reranking

    Returns:
        Top-k chunks sorted by rerank_score descending
    """
    if not chunks:
        return []

    if len(chunks) <= top_k:
        # If fewer candidates than requested, rerank all and return all
        pass

    try:
        model = _load_cross_encoder()

        # Build (query, document) pairs for batch scoring
        # Cross-encoder sees the full query and full chunk text together
        pairs = [(query, chunk.text) for chunk in chunks]

        # Predict relevance scores for all pairs at once
        # Returns array of floats — higher = more relevant
        scores = model.predict(pairs)

        # Attach cross-encoder scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        # Sort by cross-encoder score (descending) and return top-k
        reranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_k]

        logger.info(
            f"Reranked {len(chunks)} → {len(reranked)} chunks | "
            f"best cross-encoder score: {reranked[0].rerank_score:.4f}" if reranked else "none"
        )

        return reranked

    except Exception as e:
        logger.error(f"Reranking failed: {e}. Falling back to RRF order.")
        # Fallback: return top-k in RRF order (still better than Phase 2)
        fallback = sorted(chunks, key=lambda c: c.rrf_score, reverse=True)[:top_k]
        for chunk in fallback:
            chunk.rerank_score = chunk.rrf_score  # use RRF score as proxy
        return fallback

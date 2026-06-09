# src/phase3_retrieval/rrf.py
#
# Reciprocal Rank Fusion (RRF) — merges multiple ranked lists.
#
# WHAT IS RRF?
# You have two ranked lists of documents:
#   Dense:  [chunk_A, chunk_C, chunk_B, chunk_D, ...]
#   Sparse: [chunk_B, chunk_A, chunk_E, chunk_C, ...]
#
# You need one merged list that respects both rankings.
#
# THE FORMULA:
#   RRF_score(d) = Σ over all rankings r: 1 / (k + rank(d in r))
#
# Where:
#   k = 60  (a constant that dampens the impact of high ranks)
#   rank = 1-indexed position in the list (1st place = rank 1)
#
# EXAMPLE:
#   chunk_A: rank 1 in dense, rank 2 in sparse
#   chunk_B: rank 3 in dense, rank 1 in sparse
#   k = 60
#
#   RRF(chunk_A) = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
#   RRF(chunk_B) = 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
#   chunk_A wins because it ranked high in BOTH lists.
#
# WHY k=60?
# Original paper (Cormack et al., 2009) found k=60 optimal across benchmarks.
# k controls sensitivity to rank differences at the top:
#   Small k: top ranks matter much more than lower ranks
#   Large k: more uniform weighting across ranks
#   k=60 gives a good balance — consistent with most production systems.
#
# WHY NOT JUST USE SCORES?
# Dense scores (cosine) and BM25 scores are on different scales:
#   Dense:  [0.91, 0.85, 0.79, ...] — bounded, normalized
#   BM25:   [4.2, 3.1, 1.8, ...]   — unbounded, corpus-dependent
# You can't add or average them directly without normalization.
# RRF avoids this by using only RANKS, which are comparable across systems.
#
# WHY NOT NORMALIZE SCORES?
# Normalization (min-max, z-score) requires knowing the score distribution.
# RRF is parameter-free (only k) and more robust across different queries.

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# The RRF constant. 60 is the standard value from the original paper.
# Decrease for more top-rank sensitivity, increase for more uniform weighting.
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k:            int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked document lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each inner list is a ranked sequence of document IDs.
                      ranked_lists[0] = dense results (ordered best-first)
                      ranked_lists[1] = BM25 results (ordered best-first)
                      Can accept any number of lists (3+ for multi-query).

        k: The RRF constant. Default 60 (standard from original paper).

    Returns:
        List of (doc_id, rrf_score) sorted by score descending.
        Score range: (0, n/k] where n = number of ranked lists.
        Higher score = more consistently highly ranked across all lists.

    EXAMPLE:
        dense_results = ["chunk_A", "chunk_C", "chunk_B", "chunk_D"]
        bm25_results  = ["chunk_B", "chunk_A", "chunk_E", "chunk_C"]

        rrf_scores = reciprocal_rank_fusion([dense_results, bm25_results])
        # Returns: [("chunk_A", 0.0325), ("chunk_B", 0.0323), ("chunk_C", 0.0307), ...]
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            # rank is 1-indexed: first place = 1, second place = 2, etc.
            rrf_contribution = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_contribution

    # Sort by score descending (highest RRF score = best fused rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    logger.debug(
        f"RRF fusion: {len(ranked_lists)} lists → "
        f"{len(fused)} unique docs | "
        f"top score: {fused[0][1]:.4f}" if fused else "0 docs"
    )

    return fused


def explain_rrf(
    dense_list:  list[str],
    sparse_list: list[str],
    doc_id:      str,
    k:           int = RRF_K,
) -> dict:
    """
    Explain why a specific document got its RRF score.
    Useful for debugging retrieval quality.

    Returns dict with rank positions and score contribution from each list.
    """
    dense_rank  = (dense_list.index(doc_id)  + 1) if doc_id in dense_list  else None
    sparse_rank = (sparse_list.index(doc_id) + 1) if doc_id in sparse_list else None

    dense_contribution  = 1.0 / (k + dense_rank)  if dense_rank  else 0.0
    sparse_contribution = 1.0 / (k + sparse_rank) if sparse_rank else 0.0

    return {
        "doc_id":             doc_id,
        "dense_rank":         dense_rank,
        "sparse_rank":        sparse_rank,
        "dense_contribution": round(dense_contribution,  5),
        "sparse_contribution":round(sparse_contribution, 5),
        "total_rrf_score":    round(dense_contribution + sparse_contribution, 5),
        "explanation": (
            f"Dense rank {dense_rank} → +{dense_contribution:.5f} | "
            f"Sparse rank {sparse_rank} → +{sparse_contribution:.5f}"
        )
    }

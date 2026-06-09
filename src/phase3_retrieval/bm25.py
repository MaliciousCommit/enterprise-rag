# src/phase3_retrieval/bm25.py
#
# BM25 (Best Match 25) sparse keyword retrieval.
#
# WHY BM25 ALONGSIDE DENSE VECTORS:
# Dense vectors capture SEMANTIC meaning.
#   "pod memory exhausted" ≈ "container OOM killed" → high cosine similarity
#
# BM25 captures EXACT KEYWORD presence.
#   Query: "payment-svc namespace prod"
#   Dense: weak signal (these strings don't appear in training docs consistently)
#   BM25:  strong signal if "payment-svc" appears verbatim in a chunk
#
# Neither alone is sufficient:
#   Dense alone: misses exact pod names, error codes, namespace identifiers
#   BM25 alone:  misses synonyms, paraphrases, conceptual questions
#   Hybrid:      captures both — the best of both worlds
#
# BM25 FORMULA:
# For a query with terms t1...tN and document d:
#
#   BM25(d, Q) = Σ IDF(ti) × TF(ti, d) × (k1 + 1)
#                              ────────────────────────
#                              TF(ti, d) + k1 × (1 - b + b × |d|/avgdl)
#
# Parameters:
#   k1 = 1.5  controls TF saturation (diminishing returns for repeated terms)
#   b  = 0.75 controls length normalisation (long docs don't dominate)
#   IDF penalises common terms ("the", "is") that appear in many documents
#
# IMPLEMENTATION:
# We use rank-bm25 (pure Python, no ONNX, no model downloads).
# The index is built in-memory from texts loaded out of Qdrant payloads.
# Rebuilt on each process start (~100ms for 10k chunks).
# For production: persist the index to disk, rebuild on collection update.

import logging
import re
from typing import Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """
    Convert text to BM25 tokens.

    TOKENIZATION CHOICES:
    - Lowercase: "OOMKilled" and "oomkilled" are the same term
    - Split on non-alphanumeric: handles camelCase poorly (OOMKilled → oomkilled)
      A production tokenizer would split "OOMKilled" into ["OOM", "Killed"]
      We keep it simple — edge cases handled by dense search via RRF
    - No stemming: "failing" ≠ "fail" in our tokenizer
      Dense vectors handle morphological variation; BM25 handles exact strings

    For Kubernetes ops, this works well because:
    - Pod names are exact strings: "payment-svc" → ["payment", "svc"]
    - Error names: "CrashLoopBackOff" → ["crashloopbackoff"]
    - Namespace names: "kube-system" → ["kube", "system"]
    """
    return re.findall(r'\b[a-z0-9]+\b', text.lower())


class BM25Index:
    """
    In-memory BM25 index built from Qdrant collection payloads.

    LIFECYCLE:
    1. Call build_from_qdrant() once at startup (or after re-ingestion)
    2. Call search() for every retrieval request

    MEMORY:
    For 10k chunks of ~200 tokens each:
    - Tokenized corpus: ~10k × 200 × 8 bytes ≈ 16MB
    - BM25 internal structures: ~20-30MB
    Total: ~50MB RAM for a 10k chunk knowledge base.

    THREAD SAFETY:
    search() is read-only after build — safe for concurrent requests.
    build_from_qdrant() must not be called concurrently with search().
    In FastAPI: build once at startup event, share the built index.
    """

    def __init__(self):
        self._bm25:           Optional[BM25Okapi] = None
        self._corpus_ids:     list[str]           = []   # point_id per corpus entry
        self._corpus_texts:   list[str]           = []   # raw text per entry
        self._is_built:       bool                = False

    def build_from_qdrant(
        self,
        client,
        collection_name: str,
        batch_size:       int = 1000,
    ) -> int:
        """
        Load all point texts from Qdrant and build the BM25 index.

        SCROLL vs QUERY:
        We use scroll() to fetch ALL points in the collection, not a subset.
        The BM25 index needs the full corpus to compute IDF correctly.
        IDF for term t = log((N - df(t) + 0.5) / (df(t) + 0.5))
        where N = total documents, df = document frequency.
        A partial corpus gives wrong IDF scores.

        Returns: number of points indexed
        """
        logger.info(f"Building BM25 index from '{collection_name}'...")

        corpus_ids   = []
        corpus_texts = []

        # Scroll through all points in batches
        offset = None
        while True:
            results, next_offset = client.scroll(
                collection_name = collection_name,
                limit           = batch_size,
                offset          = offset,
                with_payload    = True,
                with_vectors    = False,   # don't need vectors for BM25
            )

            if not results:
                break

            for point in results:
                text = (point.payload or {}).get("text", "")
                if text:
                    corpus_ids.append(str(point.id))
                    corpus_texts.append(text)

            if next_offset is None:
                break
            offset = next_offset

        if not corpus_texts:
            logger.warning("BM25 index: no texts found in collection")
            return 0

        # Tokenize the full corpus
        tokenized_corpus = [_tokenize(text) for text in corpus_texts]

        # Build the BM25Okapi model
        # k1=1.5 (TF saturation), b=0.75 (length normalisation) — default values
        self._bm25         = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
        self._corpus_ids   = corpus_ids
        self._corpus_texts = corpus_texts
        self._is_built     = True

        logger.info(f"BM25 index built: {len(corpus_texts)} documents")
        return len(corpus_texts)

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """
        Search the BM25 index and return top-k (point_id, score) pairs.

        WHY query WITH ORIGINAL QUESTION (not HyDE expanded):
        BM25 works on keyword overlap. The original question has the exact
        keywords the user typed — pod names, namespace names, error codes.
        A HyDE-expanded hypothetical answer may rephrase these and lose
        the exact keyword signal.

        Use original question for BM25.
        Use HyDE embedding for dense search.
        RRF fusion combines both.

        Args:
            query: The user's question (original, not HyDE-expanded)
            k:     Number of results to return

        Returns:
            List of (point_id, bm25_score) pairs, sorted by score descending
        """
        if not self._is_built or self._bm25 is None:
            logger.warning("BM25 index not built — call build_from_qdrant() first")
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        # get_scores returns a score for every document in the corpus
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [
            (self._corpus_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0   # skip zero-score documents
        ]

    def get_text(self, point_id: str) -> Optional[str]:
        """Look up the text for a point_id (used during reranking)."""
        try:
            idx = self._corpus_ids.index(point_id)
            return self._corpus_texts[idx]
        except ValueError:
            return None

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def size(self) -> int:
        return len(self._corpus_ids)


# Module-level singleton — built once, shared across all requests
_bm25_index: Optional[BM25Index] = None


def get_bm25_index(client=None, collection_name: str = "") -> BM25Index:
    """
    Get or build the singleton BM25 index.

    Lazy initialisation: built on first call.
    Requires client and collection_name on first call only.
    """
    global _bm25_index
    if _bm25_index is None or not _bm25_index.is_built:
        if client is None:
            raise RuntimeError("BM25 index not built. Provide client on first call.")
        _bm25_index = BM25Index()
        _bm25_index.build_from_qdrant(client, collection_name)
    return _bm25_index

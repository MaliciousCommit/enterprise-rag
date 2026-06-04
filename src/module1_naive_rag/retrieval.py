# src/module1_naive_rag/retrieval.py
import logging
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from src.config import settings
from src.module1_naive_rag.embeddings import embed_text

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """
    A document chunk returned by the retrieval step.

    Richer than a plain string -- we carry score, source, and metadata
    so downstream nodes (CRAG grader, citation builder) can use them.

    score:       cosine similarity 0.0-1.0 -- used by CRAG in Phase 4:
                 if max(scores) < 0.7 -> trigger Tavily web search fallback.
    source:      for citations: "According to [doc 1, oomkilled.md]..."
    document_id: for tracking which docs contributed to each answer.
    chunk_index: for sentence-window context expansion in Phase 3.
    point_id:    Qdrant UUID -- for user feedback loops ("this answer was wrong").
    """
    text: str
    score: float
    source: str
    document_id: str
    chunk_index: int
    point_id: str


def retrieve(
    client: QdrantClient,
    query: str,
    k: int | None = None,
    score_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k most semantically similar chunks for a query.

    FLOW:
        query -> embed_text() [~100ms OpenAI] -> client.search() [~2ms HNSW] -> chunks

    HNSW SEARCH (simplified):
    The query vector traverses the HNSW graph top-to-bottom, greedily
    moving to the nearest neighbor at each layer. O(log n) complexity.
    At n=10k: ~1ms. At n=1M: ~5ms. Scales to hundreds of millions.

    CRITICAL: Use the SAME embedding model as during ingestion.
    text-embedding-3-small produces a different vector space than
    text-embedding-3-large, even though both output 1536 dims.
    Mixing models = meaningless similarity scores.

    score_threshold:
    None (Module 1): return all k results regardless of score.
    0.7  (Phase 4):  only return results with cosine_sim >= 0.7.
                     If max(scores) < 0.7, CRAG triggers Tavily fallback.

    Args:
        client:          Qdrant client
        query:           Natural language question
        k:               Number of results (default settings.retrieval_k = 5)
        score_threshold: Minimum cosine similarity (None = no filter)

    Returns:
        list[RetrievedChunk] sorted by score descending.
    """
    k = k or settings.retrieval_k

    logger.debug(f"Retrieving: '{query[:60]}...'")
    query_vector = embed_text(query)

    # qdrant-client >= 1.7.0: query_points() replaces search()
    # query=         the embedding vector (was query_vector= in older versions)
    # result.points  the list of ScoredPoint (was the return value directly)
    query_response = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=k,
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=False,  # don't return the 1536-dim vector -- saves bandwidth
    )

    chunks = [
        RetrievedChunk(
            text=r.payload.get("text", ""),
            score=r.score,
            source=r.payload.get("source", "unknown"),
            document_id=r.payload.get("document_id", "unknown"),
            chunk_index=r.payload.get("chunk_index", 0),
            point_id=str(r.id),
        )
        for r in query_response.points
    ]

    score_str = ", ".join(f"{c.score:.3f}" for c in chunks)
    logger.info(f"Retrieved {len(chunks)} chunks | Scores: [{score_str}]")

    return chunks


def format_context(
    chunks: list[RetrievedChunk],
    use_spotlighting: bool = True,
) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.

    use_spotlighting=True (recommended, default):
        Wraps each chunk in XML <doc> tags.
        Benefits:
        (1) LLM cites by ID: "According to [doc 1]..."
        (2) Prompt injection resistance: malicious text inside <doc> is
            labelled as untrusted content, not instructions.
        (3) Cleaner chunk boundaries than plain "---" separators.

    use_spotlighting=False (naive, for benchmarking only):
        Chunks joined with "---". Injection-vulnerable. No citation anchors.

    Lost-in-the-Middle principle: chunks are already sorted by score
    descending from Qdrant, so the most relevant chunk is first -- where
    the LLM pays the most attention.

    Args:
        chunks:          Retrieved chunks, sorted by score descending
        use_spotlighting: True = XML tags, False = plain separators

    Returns:
        str: Context string ready to insert into LLM prompt.
    """
    if not chunks:
        return "No relevant documents found in the knowledge base."

    if use_spotlighting:
        parts = [
            f'<doc id="{i}" source="{c.source}" score="{c.score:.3f}">\n'
            f"{c.text}\n"
            f"</doc>"
            for i, c in enumerate(chunks, start=1)
        ]
        return "\n\n".join(parts)
    else:
        return "\n\n---\n\n".join(c.text for c in chunks)
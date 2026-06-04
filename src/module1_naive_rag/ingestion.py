# src/module1_naive_rag/ingestion.py
# The document ingestion pipeline: raw text -> chunk -> embed -> Qdrant.

import logging
import uuid
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.config import settings
from src.module1_naive_rag.embeddings import embed_texts_batched, make_cache_key

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    A single document chunk -- the atomic unit of our retrieval system.

    FIELDS:
    text:         The actual content the LLM reads. Keep it self-contained (300-500 words).
    source:       Where it came from: file path, URL, S3 URI.
    document_id:  Logical parent document identifier. All chunks from the same doc share this.
    chunk_index:  0-based position within the parent document.
    total_chunks: Total chunks in the parent document.
    metadata:     Arbitrary key-value metadata for Qdrant payload filtering.
                  Example: {"k8s_version": "1.29", "team": "platform", "category": "runbook"}
                  In Phase 3: used for filter={"must": [{"key": "k8s_version", "match": {"value": "1.29"}}]}
                  This is the structural fix for the version-drift problem from Socratic Q5.
    """
    text: str
    source: str
    document_id: str
    chunk_index: int
    total_chunks: int
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA-256 of text -- used in Phase 6 for deduplication."""
        return make_cache_key(self.text)

    def to_payload(self) -> dict:
        """
        Convert to Qdrant point payload dict.

        The payload travels with the vector in Qdrant storage.
        When HNSW search returns a ScoredPoint, we read payload["text"]
        to get the original text back for the LLM prompt.

        WHY CO-LOCATE TEXT AND VECTOR IN QDRANT (not a separate DB)?
        One round-trip to get both match and content vs two round-trips
        (Qdrant for IDs, Postgres for text). At 10k-1M chunks: keep together.
        At 100M+ chunks: consider splitting for storage efficiency.
        """
        return {
            "text":         self.text,
            "source":       self.source,
            "document_id":  self.document_id,
            "chunk_index":  self.chunk_index,
            "total_chunks": self.total_chunks,
            "content_hash": self.content_hash,
            **self.metadata,
        }


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """
    Split raw text into overlapping chunks using paragraph-aware fixed-size strategy.

    STRATEGY:
    1. Split on double newlines (paragraph boundaries).
    2. Accumulate paragraphs until we hit chunk_size words.
    3. Save chunk, carry last `overlap` words forward into the next chunk.

    WHY OVERLAP (default 80 words)?
    A sentence crossing chunk N/N+1 boundary appears in BOTH chunks.
    A query matching that sentence will retrieve both chunks.
    Without overlap: the split sentence matches neither chunk perfectly.

    WHY WORD COUNT, NOT TOKENS?
    Exact token counting needs tiktoken (~100ms overhead per chunk).
    Word count approximation (1 word ~ 1.3 tokens) is sufficient for chunking.
    400 words ~ 520 tokens -- safely under typical 512-token chunk targets.

    PHASE 6 UPGRADE: Semantic chunking -- split when cosine similarity between
    adjacent sentences drops below a threshold. Better chunk coherence = better
    retrieval precision. We compare strategies in Module 6.

    Args:
        text:       Raw document text
        chunk_size: Target word count per chunk (default: settings.chunk_size = 400)
        overlap:    Overlap word count between chunks (default: settings.chunk_overlap = 80)

    Returns:
        list[str]: Non-empty text chunks, each >= 20 words.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_words: list[str] = []
    current_size: int = 0

    for para in paragraphs:
        para_words = para.split()
        para_size = len(para_words)

        if current_size + para_size > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            # Carry forward the last `overlap` words
            overlap_words = current_words[-overlap:] if overlap > 0 else []
            current_words = overlap_words + para_words
            current_size = len(current_words)
        else:
            current_words.extend(para_words)
            current_size += para_size

    if current_words:
        chunks.append(" ".join(current_words))

    # Filter trivially short chunks (headings, stray lines)
    return [c for c in chunks if len(c.split()) >= 20]


def ingest_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    upsert_batch_size: int = 50,
) -> dict:
    """
    Embed a list of Chunk objects and upsert them into Qdrant.

    FLOW:
        chunks -> embed all texts (batched API calls) -> PointStructs -> Qdrant upsert (batched)

    TWO BATCH SIZES:
    - embed_batch_size = 100: OpenAI API input limit
    - upsert_batch_size = 50: Qdrant HTTP request size limit
      (50 vectors x 6KB each = 300KB per request -- comfortable)

    Args:
        client:           Qdrant client
        chunks:           Chunk objects to ingest
        upsert_batch_size: Points per Qdrant upsert call

    Returns:
        dict: {"ingested": int, "errors": int}
    """
    if not chunks:
        return {"ingested": 0, "errors": 0}

    stats = {"ingested": 0, "errors": 0}

    # Step 1: Batch embed all chunk texts
    texts = [c.text for c in chunks]
    logger.info(f"Ingesting {len(chunks)} chunks...")
    vectors = embed_texts_batched(texts)

    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding/chunk count mismatch: {len(vectors)} != {len(chunks)}"
        )

    # Step 2: Build PointStructs
    # PointStruct = one Qdrant record: {id, vector, payload}
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            # uuid4(): random 128-bit UUID. No central coordinator needed.
            # Phase 6 adds: check content_hash before creating, skip if exists.
            vector=vector,
            payload=chunk.to_payload(),
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    # Step 3: Upsert to Qdrant in batches
    n_batches = (len(points) + upsert_batch_size - 1) // upsert_batch_size
    for i in range(0, len(points), upsert_batch_size):
        batch = points[i : i + upsert_batch_size]
        batch_num = i // upsert_batch_size + 1
        try:
            client.upsert(
                collection_name=settings.collection_name,
                points=batch,
                wait=True,  # block until indexed (safe for ingestion scripts)
                            # wait=False for high-throughput async ingestion (Phase 10)
            )
            stats["ingested"] += len(batch)
            logger.info(
                f"  Upserted batch {batch_num}/{n_batches}: "
                f"{len(batch)} points | total: {stats['ingested']}"
            )
        except Exception as e:
            logger.error(f"  Upsert failed batch {batch_num}: {e}")
            stats["errors"] += len(batch)

    logger.info(
        f"Ingestion done: {stats['ingested']} ingested, {stats['errors']} errors"
    )
    return stats


def ingest_document(
    client: QdrantClient,
    text: str,
    source: str,
    document_id: str,
    metadata: dict | None = None,
) -> dict:
    """
    Full ingestion pipeline for one document: raw text -> chunk -> embed -> Qdrant.

    This is the primary entry point for adding documents to the knowledge base.

    Args:
        client:      Qdrant client
        text:        Full document text (any length)
        source:      Source identifier (file path, URL)
        document_id: Unique document ID (for deduplication tracking in Phase 6)
        metadata:    Key-value pairs merged into every chunk's Qdrant payload.
                     Use for: k8s_version, team, category, last_updated, etc.

    Returns:
        dict: {"ingested": int, "errors": int, "chunks_created": int}
    """
    metadata = metadata or {}
    raw_chunks = chunk_text(text)

    if not raw_chunks:
        logger.warning(f"Document '{document_id}' produced 0 chunks.")
        return {"ingested": 0, "errors": 0, "chunks_created": 0}

    logger.info(f"Document '{document_id}': {len(raw_chunks)} chunks from '{source}'")

    chunks = [
        Chunk(
            text=raw,
            source=source,
            document_id=document_id,
            chunk_index=idx,
            total_chunks=len(raw_chunks),
            metadata=metadata,
        )
        for idx, raw in enumerate(raw_chunks)
    ]

    stats = ingest_chunks(client, chunks)
    stats["chunks_created"] = len(chunks)
    return stats

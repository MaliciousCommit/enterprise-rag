# src/module6_ingestion/chunker.py
#
# Production document chunker with multiple strategies.
#
# STRATEGY SELECTION GUIDE (for our K8s knowledge base):
#   Runbooks (structured markdown):  → MarkdownChunker (best)
#   Postmortems (narrative text):    → RecursiveChunker
#   API docs (very structured):      → MarkdownChunker
#   Slack exports (conversational):  → RecursiveChunker with small chunks
#   PDFs with no structure:          → FixedSizeChunker
#
# PHASE EVOLUTION:
# Module 1:  FixedSizeChunker (basic, already implemented)
# Module 6:  All three strategies + metadata enrichment (this file)
# Phase 3:   Add parent-document retrieval (store parent + child chunks)
# Phase 8:   Add PII scrubbing before chunking

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ChunkingStrategy(str, Enum):
    FIXED      = "fixed"       # character count, simple
    RECURSIVE  = "recursive"   # tries paragraph → sentence → word boundaries
    MARKDOWN   = "markdown"    # splits on headings, preserves structure


@dataclass
class DocumentChunk:
    """
    A single chunk produced by any chunking strategy.

    All fields are designed to be stored as Qdrant payload.
    Every field is a primitive type (str, int, list[str]) —
    no nested objects — to ensure clean JSON serialisation.
    """
    text:          str                    # the chunk content to embed
    chunk_index:   int                    # position in the source document
    source:        str                    # file path / URL of source document
    document_id:   str                    # unique ID for the source document
    doc_type:      str = "runbook"        # "runbook" | "guide" | "postmortem" | "api_doc"
    team:          str = "platform"       # owning team for multi-tenant filtering
    k8s_version:   str = "1.29"          # Kubernetes version this doc applies to
    heading_path:  str = ""              # breadcrumb: "Runbook > Symptoms > Diagnosis"
    token_count:   int = 0               # approximate token count (chars / 4)
    char_count:    int = 0               # raw character count
    tags:          list[str] = field(default_factory=list)

    def __post_init__(self):
        """Compute derived fields after construction."""
        self.char_count  = len(self.text)
        self.token_count = len(self.text) // 4  # rough approximation, ~4 chars/token


# ── Strategy 1: Fixed-Size Chunker ────────────────────────────────────────────

class FixedSizeChunker:
    """
    Split text by character count with overlap.
    The Module 1 approach — simple and predictable.

    USE WHEN:
    - Documents have no discernible structure
    - PDFs that have been OCR'd into a single text blob
    - Uniform content where all sections are equally important

    WEAKNESS:
    Cuts mid-sentence. A sentence split across two chunks means
    neither chunk fully captures the idea. Overlap mitigates this
    but doesn't eliminate it.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Args:
            chunk_size: Target character count per chunk.
                        500 chars ≈ 125 tokens ≈ small but precise chunks.
                        1000 chars ≈ 250 tokens ≈ medium, more context.
            overlap:    Characters repeated between adjacent chunks.
                        50 chars (10% of 500) is the minimum useful overlap.
                        100-150 chars (20-30%) for complex technical content.
        """
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def chunk(self, text: str, metadata: dict) -> list[DocumentChunk]:
        chunks   = []
        start    = 0
        idx      = 0

        while start < len(text):
            end  = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(DocumentChunk(
                    text        = chunk_text,
                    chunk_index = idx,
                    **metadata,
                ))
                idx += 1

            start = end - self.overlap  # step back by overlap amount

        return chunks


# ── Strategy 2: Recursive Character Chunker ───────────────────────────────────

class RecursiveChunker:
    """
    Split on the largest available separator first, recursively.
    Industry standard: used by LangChain's RecursiveCharacterTextSplitter.

    SEPARATOR PRIORITY (tries in order, falls back to next):
    1. "\n\n"  — paragraph boundary (best: preserves paragraph units)
    2. "\n"    — line boundary (good: preserves line structure)
    3. ". "    — sentence boundary (ok: preserves sentence units)
    4. " "     — word boundary (fallback: at least no mid-word cuts)
    5. ""      — character (last resort: chunk_size is hard limit)

    USE WHEN:
    - Prose documents (postmortems, incident reports)
    - Documents with paragraph structure but no headings
    - Narrative content where sentence integrity matters

    ADVANTAGE OVER FIXED:
    Chunks end at natural language boundaries.
    "The pod crashed because of memory exhaustion." stays in one chunk.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def chunk(self, text: str, metadata: dict) -> list[DocumentChunk]:
        raw_chunks = self._split_recursive(text, self.SEPARATORS)
        merged     = self._merge_with_overlap(raw_chunks)

        return [
            DocumentChunk(text=c, chunk_index=i, **metadata)
            for i, c in enumerate(merged)
            if c.strip()
        ]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the first separator that works."""
        if not separators:
            # Hard limit: split by character count
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]
        if sep == "":
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts  = text.split(sep)
        result = []
        for part in parts:
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                # This part is still too large — recurse with next separator
                result.extend(self._split_recursive(part, separators[1:]))

        return result

    def _merge_with_overlap(self, parts: list[str]) -> list[str]:
        """
        Merge small parts into chunks, respecting chunk_size.
        Add overlap by repeating the tail of the previous chunk.
        """
        chunks       = []
        current      = ""
        previous_end = ""  # the overlap text from the previous chunk

        for part in parts:
            candidate = (current + " " + part).strip() if current else part.strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(previous_end + current if previous_end else current)
                    # Capture overlap: last self.overlap chars of current chunk
                    previous_end = current[-self.overlap:] + " " if self.overlap > 0 else ""
                current = part.strip()

        if current:
            chunks.append(previous_end + current if previous_end else current)

        return chunks


# ── Strategy 3: Markdown-Aware Chunker ────────────────────────────────────────

class MarkdownChunker:
    """
    Split markdown documents on heading boundaries.
    Each section (heading + content) becomes one chunk.

    BEST STRATEGY FOR OUR K8S KNOWLEDGE BASE because:
    1. All our runbooks and guides are structured markdown
    2. Headings are semantic boundaries: "## Symptoms" vs "## Resolution"
    3. Breadcrumb paths improve retrieval precision:
       Query: "OOMKilled resolution" matches chunk with path
       "OOMKilled Runbook > Resolution" exactly

    HOW HEADING BREADCRUMBS WORK:
    # OOMKilled Runbook
    ## Symptoms           → path: "OOMKilled Runbook > Symptoms"
    ### Exit Code 137     → path: "OOMKilled Runbook > Symptoms > Exit Code 137"
    ## Resolution         → path: "OOMKilled Runbook > Resolution"
    ### Fix Memory Limit  → path: "OOMKilled Runbook > Resolution > Fix Memory Limit"

    The breadcrumb is prepended to the chunk text before embedding.
    This gives the embedding model context about WHERE this chunk lives
    in the document hierarchy — not just what it says.

    CHUNK SIZE HANDLING:
    If a section exceeds max_chunk_chars, it's split with RecursiveChunker.
    This handles very long sections gracefully.
    """

    HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def __init__(self, max_chunk_chars: int = 1500, overlap: int = 100):
        self.max_chunk_chars = max_chunk_chars
        self.overlap         = overlap
        self._recursive      = RecursiveChunker(chunk_size=max_chunk_chars, overlap=overlap)

    def chunk(self, text: str, metadata: dict) -> list[DocumentChunk]:
        sections = self._split_into_sections(text)
        chunks   = []
        idx      = 0

        for section in sections:
            heading_path = section["heading_path"]
            content      = section["content"].strip()

            if not content:
                continue

            # Prepend breadcrumb to chunk text for richer embedding context
            # The LLM also sees this — helps it understand the document structure
            if heading_path:
                chunk_text = f"[{heading_path}]\n\n{content}"
            else:
                chunk_text = content

            if len(chunk_text) <= self.max_chunk_chars:
                # Section fits in one chunk
                chunk_meta = {**metadata, "heading_path": heading_path}
                chunks.append(DocumentChunk(
                    text        = chunk_text,
                    chunk_index = idx,
                    **chunk_meta,
                ))
                idx += 1
            else:
                # Section too large — recursively split while keeping breadcrumb
                chunk_meta   = {**metadata, "heading_path": heading_path}
                sub_chunks   = self._recursive.chunk(content, chunk_meta)
                for sc in sub_chunks:
                    sc.chunk_index  = idx
                    sc.heading_path = heading_path
                    sc.text         = f"[{heading_path}]\n\n{sc.text}" if heading_path else sc.text
                    chunks.append(sc)
                    idx += 1

        return chunks

    def _split_into_sections(self, text: str) -> list[dict]:
        """Split markdown text into sections at heading boundaries."""
        # Find all heading positions
        heading_matches = list(self.HEADING_RE.finditer(text))

        if not heading_matches:
            return [{"heading_path": "", "content": text}]

        sections       = []
        heading_stack  = []  # tracks current hierarchy: [("H1 title", 1), ...]

        for i, match in enumerate(heading_matches):
            level   = len(match.group(1))   # number of # symbols
            title   = match.group(2).strip()
            start   = match.start()
            end     = heading_matches[i+1].start() if i+1 < len(heading_matches) else len(text)

            # Build breadcrumb path
            # Pop headings of same or deeper level (they're completed)
            heading_stack = [(t, l) for t, l in heading_stack if l < level]
            heading_stack.append((title, level))
            breadcrumb = " > ".join(t for t, _ in heading_stack)

            # Content is everything after the heading line until next heading
            content_start = text.index('\n', start) + 1 if '\n' in text[start:] else start
            content       = text[content_start:end].strip()

            sections.append({"heading_path": breadcrumb, "content": content})

        return sections


# ── Strategy comparison utility ────────────────────────────────────────────────

def compare_strategies(text: str, source: str = "test_doc") -> dict:
    """
    Run all three chunking strategies on the same text.
    Returns statistics for comparison.

    Used by the inspection script to show which strategy
    produces the best chunk characteristics for a given document.
    """
    base_metadata = {
        "source":      source,
        "document_id": source,
        "doc_type":    "runbook",
        "team":        "platform",
        "k8s_version": "1.29",
    }

    strategies = {
        "fixed":     FixedSizeChunker(chunk_size=500, overlap=50),
        "recursive": RecursiveChunker(chunk_size=500, overlap=50),
        "markdown":  MarkdownChunker(max_chunk_chars=1500, overlap=100),
    }

    results = {}
    for name, chunker in strategies.items():
        chunks = chunker.chunk(text, base_metadata)
        sizes  = [c.char_count for c in chunks]
        results[name] = {
            "num_chunks":  len(chunks),
            "min_chars":   min(sizes) if sizes else 0,
            "max_chars":   max(sizes) if sizes else 0,
            "avg_chars":   int(sum(sizes) / len(sizes)) if sizes else 0,
            "chunks":      chunks,
        }

    return results

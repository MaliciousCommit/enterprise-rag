# src/module1_naive_rag/pipeline.py
#
# The NaiveRAG class -- ties every module together into one end-to-end pipeline.
#
# This is the object scripts and tests instantiate:
#   rag = NaiveRAG()
#   response = rag.query("Why is my pod OOMKilled?")
#   print(response.answer)
#
# WHAT THIS PHASE DOES:
#   question -> retrieve() -> generate() -> RAGResponse
#
# WHAT IT DOES NOT DO (fixed in later phases):
#   No caching:          every query hits OpenAI twice (Phase 6)
#   No hybrid search:    dense only, misses exact keyword matches (Phase 3)
#   No reranking:        retrieval order not optimized for LLM (Phase 3)
#   No CRAG:             can't detect bad retrieval -> hallucination (Phase 4)
#   No self-RAG:         no quality check on the answer (Phase 4)
#   No SQL pipeline:     can't answer live cluster queries (Phase 5)
#   No security:         no auth, rate limiting, PII protection (Phase 8)
#   No observability:    no latency tracking, cost aggregation (Phase 9)
#   No LangGraph state:  stateless, no conversation memory (Phase 4)
#
# PHASE EVOLUTION:
# Module 1: This 2-step class (retrieve -> generate)
# Phase 4:  Replaced by a LangGraph StateGraph with 5+ nodes
# Phase 10: Invoked via FastAPI async endpoint with SSE streaming

import time
import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient

from src.config import settings
from src.module1_naive_rag.collection import get_qdrant_client
from src.module1_naive_rag.retrieval import retrieve, RetrievedChunk
from src.module1_naive_rag.generation import generate, GenerationResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """
    The complete output of a single NaiveRAG.query() call.

    Contains the answer PLUS full diagnostic metadata:
    - What was retrieved (which chunks, what scores)
    - How the answer was generated (which model, how many tokens)
    - How long it took (latency_ms)

    WHY CARRY ALL THIS IN THE RESPONSE?
    In Phase 9 (Observability), we log every RAGResponse to:
    - LangSmith (full trace: question -> retrieved chunks -> answer)
    - Prometheus (latency histogram, token counter, score gauge)
    - Structured JSON logs (for Grafana dashboards)

    Having this data in the response object makes it trivial to
    instrument without touching the core pipeline code.
    """
    question:         str
    answer:           str
    retrieved_chunks: list[RetrievedChunk]
    generation:       GenerationResult
    latency_ms:       float

    @property
    def retrieval_scores(self) -> list[float]:
        """Cosine similarity scores for each retrieved chunk."""
        return [c.score for c in self.retrieved_chunks]

    @property
    def best_score(self) -> float:
        """Highest cosine similarity among retrieved chunks.
        Phase 4 (CRAG): if best_score < 0.7 -> trigger Tavily fallback."""
        return max(self.retrieval_scores, default=0.0)

    @property
    def avg_score(self) -> float:
        """Average cosine similarity -- a rough retrieval quality signal."""
        scores = self.retrieval_scores
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def estimated_cost_usd(self) -> float:
        """Approximate cost of this query (embedding + generation)."""
        embed_cost = 0.020 / 1_000_000 * 20   # ~20 tokens for query embedding
        gen_input  = 5.00  / 1_000_000 * self.generation.prompt_tokens
        gen_output = 15.00 / 1_000_000 * self.generation.completion_tokens
        return embed_cost + gen_input + gen_output

    def summary(self) -> str:
        """Human-readable one-line summary for logging and CLI output."""
        return (
            f"[{self.latency_ms:.0f}ms] "
            f"scores=[{', '.join(f'{s:.3f}' for s in self.retrieval_scores)}] "
            f"tokens={self.generation.total_tokens} "
            f"cost=${self.estimated_cost_usd:.4f}"
        )


class NaiveRAG:
    """
    The complete Phase 1 (Module 1) naive RAG pipeline.

    Usage:
        from src.module1_naive_rag import NaiveRAG
        rag = NaiveRAG()
        response = rag.query("Why is my pod showing OOMKilled?")
        print(response.answer)

    The client is shared across multiple .query() calls.
    Create one NaiveRAG instance per application and reuse it.
    Do NOT create a new NaiveRAG per query -- it re-opens the Qdrant connection.

    DESIGN DECISION: Class vs. Function
    We use a class even for this simple pipeline because:
    1. We store the Qdrant client as state (reuse the connection)
    2. Later phases will add state: LangGraph runner, Redis client, etc.
    3. The test fixtures can inject a mock client for unit testing
    """

    def __init__(self, client: QdrantClient | None = None):
        """
        Args:
            client: Qdrant client. If None, creates one from settings.
                    Inject a custom client in tests to avoid real network calls.
        """
        self.client = client or get_qdrant_client()
        logger.info(
            f"NaiveRAG initialized | "
            f"collection={settings.collection_name} | "
            f"k={settings.retrieval_k} | "
            f"model={settings.llm_model}"
        )

    def query(
        self,
        question: str,
        k: int | None = None,
    ) -> RAGResponse:
        """
        Execute the complete naive RAG pipeline for a question.

        EXECUTION FLOW:
            question
                |
                v
            retrieve(question, k)          [~102ms: embed + HNSW search]
                |
                v
            list[RetrievedChunk]           [top-k most similar chunks]
                |
                v
            generate(question, chunks)     [~1,500ms: GPT-4o generation]
                |
                v
            RAGResponse                    [answer + full metadata]

        TOTAL LATENCY:
            P50: ~1,600ms (dominated by GPT-4o generation)
            P95: ~3,800ms (OpenAI API tail latency)
            P99: ~8,200ms (rare OpenAI spikes)

        With Phase 6 caching:
            Cache HIT:  ~50ms (Redis lookup, skip embed + search + generate)
            Cache MISS: ~1,600ms (same as now, but result cached for next time)
            Cache hit rate in production: ~40-60%

        Args:
            question: Natural language question about Kubernetes operations
            k:        Override retrieval_k for this query.
                      Use k=20 to retrieve more candidates before reranking (Phase 3).

        Returns:
            RAGResponse with answer, retrieved chunks, generation metadata, latency
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        start = time.perf_counter()
        logger.info(f"NaiveRAG.query: '{question}'")

        # Step 1: Semantic retrieval
        chunks = retrieve(self.client, question, k=k)

        # Step 2: Grounded generation
        generation = generate(question, chunks)

        latency_ms = (time.perf_counter() - start) * 1000

        response = RAGResponse(
            question=question,
            answer=generation.answer,
            retrieved_chunks=chunks,
            generation=generation,
            latency_ms=latency_ms,
        )

        logger.info(f"NaiveRAG.query complete | {response.summary()}")
        return response

    def query_and_print(self, question: str, k: int | None = None) -> RAGResponse:
        """
        Convenience method: query and print results to console.
        Used in scripts/02_query.py for interactive sessions.
        """
        response = self.query(question, k=k)

        print(f"\n{'='*70}")
        print(f"QUESTION: {question}")
        print(f"{'='*70}")
        print(f"\nANSWER:\n{response.answer}")
        print(f"\n{'-'*70}")
        print(f"RETRIEVED CHUNKS ({len(response.retrieved_chunks)} total):")
        for i, chunk in enumerate(response.retrieved_chunks, 1):
            print(f"  [{i}] score={chunk.score:.3f} | {chunk.source}")
            print(f"       {chunk.text[:120].replace(chr(10), ' ')}...")
        print(f"\nMETRICS:")
        print(f"  Latency:    {response.latency_ms:.0f}ms")
        print(f"  Tokens:     {response.generation.total_tokens} "
              f"({response.generation.prompt_tokens} in + "
              f"{response.generation.completion_tokens} out)")
        print(f"  Cost:       ~${response.estimated_cost_usd:.4f}")
        print(f"  Best score: {response.best_score:.3f}")
        print(f"  Avg score:  {response.avg_score:.3f}")
        print(f"{'='*70}\n")

        return response

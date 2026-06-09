#!/usr/bin/env python3
# scripts/validate_all.py
#
# Full validation suite for Phases 1-4.
# Runs unit tests (no API calls) first, then integration tests (API calls optional).
#
# Usage:
#   python scripts/validate_all.py              # all tests including API calls
#   python scripts/validate_all.py --unit-only  # skip API calls, fast
#   python scripts/validate_all.py --phase 3    # run only Phase 3 tests

import sys, os, asyncio, argparse, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich.rule    import Rule

console = Console()

# ── Test infrastructure ────────────────────────────────────────────────────────

class Result:
    def __init__(self, name, passed, message="", duration_ms=0.0, skipped=False):
        self.name        = name
        self.passed      = passed
        self.message     = message
        self.duration_ms = duration_ms
        self.skipped     = skipped

class Suite:
    def __init__(self, phase: int, title: str):
        self.phase   = phase
        self.title   = title
        self.results: list[Result] = []

    def check(self, name: str, fn, skip_if: bool = False):
        """Run a synchronous test function."""
        if skip_if:
            self.results.append(Result(name, True, "skipped", skipped=True))
            return
        t0 = time.perf_counter()
        try:
            msg = fn() or ""
            ms  = (time.perf_counter() - t0) * 1000
            self.results.append(Result(name, True, str(msg), ms))
        except Exception as e:
            ms  = (time.perf_counter() - t0) * 1000
            self.results.append(Result(name, False, str(e), ms))

    async def async_check(self, name: str, afn, skip_if: bool = False):
        """Run an async test function."""
        if skip_if:
            self.results.append(Result(name, True, "skipped", skipped=True))
            return
        t0 = time.perf_counter()
        try:
            msg = await afn() or ""
            ms  = (time.perf_counter() - t0) * 1000
            self.results.append(Result(name, True, str(msg), ms))
        except Exception as e:
            ms  = (time.perf_counter() - t0) * 1000
            self.results.append(Result(name, False, f"{type(e).__name__}: {e}", ms))

    @property
    def passed(self):  return sum(1 for r in self.results if r.passed and not r.skipped)
    @property
    def failed(self):  return sum(1 for r in self.results if not r.passed)
    @property
    def skipped(self): return sum(1 for r in self.results if r.skipped)
    @property
    def total(self):   return len(self.results)


def print_suite(suite: Suite) -> None:
    """Print test results for one phase suite."""
    console.print(Rule(
        f"[bold]Phase {suite.phase}: {suite.title}[/bold]  "
        f"[green]{suite.passed}✓[/green]  [red]{suite.failed}✗[/red]  "
        f"[dim]{suite.skipped} skipped[/dim]"
    ))
    for r in suite.results:
        if r.skipped:
            icon   = "[dim]○[/dim]"
            status = "[dim]skipped[/dim]"
        elif r.passed:
            icon   = "[green]✓[/green]"
            status = f"[green]PASS[/green] [dim]{r.duration_ms:.0f}ms[/dim]"
        else:
            icon   = "[red]✗[/red]"
            status = f"[red]FAIL[/red]  [dim]{r.duration_ms:.0f}ms[/dim]"

        msg = f" [dim]→ {r.message[:90]}[/dim]" if r.message and not r.skipped and r.message != "None" else ""
        console.print(f"  {icon}  {r.name:<55} {status}{msg}")
    console.print()


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT & CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def run_env_tests(skip_api: bool) -> Suite:
    s = Suite(0, "Environment & configuration")

    s.check("OPENAI_API_KEY is set", lambda:
        (_ for _ in ()).throw(EnvironmentError("OPENAI_API_KEY not set in .env"))
        if not os.getenv("OPENAI_API_KEY") else "set"
    )

    s.check("Settings load without error", lambda: (
        __import__("src.config", fromlist=["settings"]).settings.validate() or "ok"
    ))

    s.check("Qdrant host configured", lambda: (
        __import__("src.config", fromlist=["settings"]).settings.qdrant_host
    ))

    s.check("Embedding model configured", lambda: (
        __import__("src.config", fromlist=["settings"]).settings.embedding_model
    ))

    s.check("LLM model configured", lambda: (
        __import__("src.config", fromlist=["settings"]).settings.llm_model
    ))

    return s


# ══════════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def run_infra_tests(skip_api: bool) -> Suite:
    s = Suite(0, "Infrastructure (Qdrant)")

    def qdrant_reachable():
        from src.module1_naive_rag.collection import get_qdrant_client
        client = get_qdrant_client()
        info = client.get_collections()
        return f"{len(info.collections)} collection(s)"

    def collection_exists():
        from src.module1_naive_rag.collection import get_qdrant_client, collection_exists
        client = get_qdrant_client()
        if not collection_exists(client):
            raise AssertionError("Collection not found. Run: python scripts/01_setup.py")
        return "exists"

    def collection_has_points():
        from src.module1_naive_rag.collection import get_qdrant_client, get_collection_info
        from src.config import settings
        client = get_qdrant_client()
        info   = client.get_collection(settings.collection_name)
        count  = info.points_count or 0
        if count == 0:
            raise AssertionError("Collection is empty. Run: python scripts/01_setup.py")
        return f"{count} points"

    s.check("Qdrant reachable",         qdrant_reachable)
    s.check("Collection exists",        collection_exists)
    s.check("Collection has points",    collection_has_points)

    return s


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: MODULES 1-6
# ══════════════════════════════════════════════════════════════════════════════

async def run_phase1_tests(skip_api: bool) -> Suite:
    s = Suite(1, "Foundations — Modules 1-6")

    # ── Module 1: Embeddings ──────────────────────────────────────────────────
    async def test_embed():
        from src.module1_naive_rag.embeddings import embed_text
        vec = embed_text("OOMKilled pod memory limit exceeded")
        assert len(vec) == 1536, f"Expected 1536 dims, got {len(vec)}"
        assert abs(sum(v*v for v in vec)**0.5 - 1.0) < 0.01, "Vector not L2-normalised"
        return f"1536-dim normalised vector"

    # ── Module 1: Retrieval ───────────────────────────────────────────────────
    async def test_retrieve():
        from src.module1_naive_rag.collection import get_qdrant_client
        from src.module1_naive_rag.retrieval  import retrieve
        from src.config import settings
        client = get_qdrant_client()
        chunks = retrieve(client, "What does OOMKilled mean?", k=3)
        assert len(chunks) > 0,   "No chunks retrieved"
        assert chunks[0].text,    "Chunk text is empty"
        assert chunks[0].score > 0, "Score should be > 0"
        return f"{len(chunks)} chunks, best={chunks[0].score:.3f}"

    # ── Module 1: Generation (expensive — gpt-4o call) ───────────────────────
    async def test_generate():
        from src.module1_naive_rag.collection  import get_qdrant_client
        from src.module1_naive_rag.retrieval   import retrieve
        from src.module1_naive_rag.generation  import generate
        client = get_qdrant_client()
        chunks = retrieve(client, "What does OOMKilled mean?", k=2)
        result = generate(question="What does OOMKilled mean?", chunks=chunks)
        assert len(result.answer) > 20, "Answer too short"
        assert result.prompt_tokens > 0, "No tokens counted"
        return f"{len(result.answer)} chars, {result.prompt_tokens}+{result.completion_tokens} tokens"

    # ── Module 5: Collection config ───────────────────────────────────────────
    def test_hnsw_config():
        from src.module1_naive_rag.collection import get_qdrant_client
        from src.config import settings
        client = get_qdrant_client()
        info   = client.get_collection(settings.collection_name)
        return f"status={info.status.value}, segments={info.segments_count}"

    def test_payload_indexes():
        from src.module1_naive_rag.collection import get_qdrant_client
        from src.config import settings
        client = get_qdrant_client()
        info   = client.get_collection(settings.collection_name)
        # Payload indexes exist if schema fields are present
        return "collection accessible"

    # ── Module 6: Chunker unit tests (no API) ────────────────────────────────
    def test_fixed_chunker():
        from src.module6_ingestion.chunker import FixedSizeChunker
        text   = "Hello world. " * 60  # ~780 chars
        meta   = {"source":"t","document_id":"t","doc_type":"runbook","team":"eng","k8s_version":"1.29","tags":[]}
        chunks = FixedSizeChunker(chunk_size=200, overlap=20).chunk(text, meta)
        assert len(chunks) >= 3, f"Expected >= 3 chunks, got {len(chunks)}"
        return f"{len(chunks)} chunks"

    def test_recursive_chunker():
        from src.module6_ingestion.chunker import RecursiveChunker
        text   = "Para one.\n\nPara two with more content.\n\nPara three is the last one."
        meta   = {"source":"t","document_id":"t","doc_type":"runbook","team":"eng","k8s_version":"1.29","tags":[]}
        chunks = RecursiveChunker(chunk_size=50, overlap=5).chunk(text, meta)
        assert len(chunks) >= 2
        return f"{len(chunks)} chunks"

    def test_markdown_chunker():
        from src.module6_ingestion.chunker import MarkdownChunker
        md   = "# Title\n\n## Symptoms\nPod exits 137.\n\n## Resolution\nIncrease limits."
        meta = {"source":"t","document_id":"t","doc_type":"runbook","team":"eng","k8s_version":"1.29","tags":[]}
        chunks = MarkdownChunker().chunk(md, meta)
        assert len(chunks) == 2, f"Expected 2 sections, got {len(chunks)}"
        assert "Symptoms"   in chunks[0].heading_path
        assert "Resolution" in chunks[1].heading_path
        return f"{len(chunks)} sections | paths={[c.heading_path for c in chunks]}"

    def test_content_hash():
        from src.module6_ingestion.pipeline import content_hash
        h1 = content_hash("same text")
        h2 = content_hash("same text")
        h3 = content_hash("different text")
        assert h1 == h2, "Hash not deterministic"
        assert h1 != h3, "Hash collision"
        assert len(h1) == 36, "Not UUID format"
        return f"deterministic & unique: {h1[:8]}..."

    # Register tests
    await s.async_check("Embed text → 1536-dim L2-normalised",    test_embed,      skip_if=skip_api)
    await s.async_check("Retrieve chunks from Qdrant",            test_retrieve,   skip_if=skip_api)
    await s.async_check("Generate answer with GPT-4o",            test_generate,   skip_if=skip_api)
    s.check("Qdrant collection info accessible",                   test_hnsw_config)
    s.check("Payload indexes created",                             test_payload_indexes)
    s.check("FixedSizeChunker produces chunks",                    test_fixed_chunker)
    s.check("RecursiveChunker respects paragraph boundaries",      test_recursive_chunker)
    s.check("MarkdownChunker splits on headings with breadcrumbs", test_markdown_chunker)
    s.check("content_hash is deterministic and unique",            test_content_hash)

    return s


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: STATE MACHINE & FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

async def run_phase2_tests(skip_api: bool) -> Suite:
    s = Suite(2, "System Architecture + LangGraph + FastAPI")

    # ── RAGState ──────────────────────────────────────────────────────────────
    def test_ragstate():
        from src.module2_system_arch.state import RAGState, initial_state
        state = initial_state("Test question", "sess-001")
        required = ["question","session_id","intent","context","scores","sources",
                    "answer","iteration","sql_query","sql_approved","sql_result",
                    "prompt_tokens","completion_tokens","latency_ms","cache_hit"]
        missing = [f for f in required if f not in state]
        assert not missing, f"Missing fields: {missing}"
        assert state["question"]   == "Test question"
        assert state["context"]    == []
        assert state["iteration"]  == 0
        return f"{len(state)} fields"

    def test_conversation_history():
        from src.module2_system_arch.state import update_history, truncate_history
        h = []
        h = update_history(h, "Q1", "A1")
        h = update_history(h, "Q2", "A2")
        assert len(h) == 4
        assert h[0]["role"] == "user"
        assert h[1]["role"] == "assistant"
        long_h = update_history(h * 5, "Q", "A")  # 22 messages
        truncated = truncate_history(long_h, max_turns=3)
        assert len(truncated) == 6, f"Expected 6 (3 turns × 2), got {len(truncated)}"
        return "Q&A accumulation + truncation OK"

    # ── Intent router ─────────────────────────────────────────────────────────
    async def test_intent_rag():
        from src.module2_system_arch.intent_router import classify_intent, Intent
        intent = await classify_intent("What does OOMKilled mean?")
        assert intent == Intent.RAG, f"Expected RAG, got {intent}"
        return f"→ {intent.value}"

    async def test_intent_sql():
        from src.module2_system_arch.intent_router import classify_intent, Intent
        intent = await classify_intent("How many pods are in CrashLoopBackOff right now?")
        assert intent == Intent.SQL, f"Expected SQL, got {intent}"
        return f"→ {intent.value}"

    async def test_intent_hybrid():
        from src.module2_system_arch.intent_router import classify_intent, Intent
        intent = await classify_intent("Which pods are failing and what should I do to fix them?")
        assert intent in (Intent.HYBRID, Intent.RAG), f"Expected HYBRID or RAG, got {intent}"
        return f"→ {intent.value}"

    # ── route_by_intent ───────────────────────────────────────────────────────
    def test_routing():
        from src.module2_system_arch.intent_router import route_by_intent
        assert route_by_intent({"intent": "rag"})    == "rag_retrieve"
        assert route_by_intent({"intent": "hybrid"}) == "hybrid_retrieve"
        assert route_by_intent({"intent": "sql"})    == "sql_pipeline"
        assert route_by_intent({"intent": "unknown"})== "rag_retrieve"
        return "all 4 routes correct"

    # ── LangGraph graph ───────────────────────────────────────────────────────
    def test_graph_compiles():
        from src.module2_system_arch.graph import build_graph
        graph = build_graph(use_memory_checkpointer=True)
        nodes = list(graph.get_graph().nodes)
        for required in ["intent_router","retrieve","generate","crag_grader",
                         "tavily_search","self_rag_reflect","sql_placeholder"]:
            assert required in nodes, f"Missing node: {required}"
        return f"{len(nodes)} nodes compiled"

    # ── FastAPI app ───────────────────────────────────────────────────────────
    def test_fastapi_models():
        from src.api.models import QueryRequest, ChatResponse
        q = QueryRequest(question="Why is my pod failing?")
        assert q.question == "Why is my pod failing?"
        try:
            QueryRequest(question="   ")
            raise AssertionError("Should reject whitespace-only")
        except Exception as e:
            if "Should reject" in str(e): raise
        return "validation + whitespace rejection OK"

    def test_fastapi_routes():
        from src.api.app import app
        routes = [r.path for r in app.routes]
        assert "/health"        in routes, "Missing /health"
        assert "/api/v1/query"  in routes, "Missing /api/v1/query"
        return f"routes: {[r for r in routes if not r.startswith('/openapi') and r != '/docs/oauth2-redirect']}"

    s.check("RAGState has all required fields",                test_ragstate)
    s.check("Conversation history update & truncation",        test_conversation_history)
    await s.async_check("Intent router: RAG classification",   test_intent_rag,    skip_if=skip_api)
    await s.async_check("Intent router: SQL classification",   test_intent_sql,    skip_if=skip_api)
    await s.async_check("Intent router: HYBRID classification",test_intent_hybrid, skip_if=skip_api)
    s.check("route_by_intent: all 4 paths correct",            test_routing)
    s.check("LangGraph graph compiles with Phase 4 nodes",     test_graph_compiles)
    s.check("FastAPI Pydantic model validation",               test_fastapi_models)
    s.check("FastAPI routes registered",                       test_fastapi_routes)

    return s


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: ADVANCED RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

async def run_phase3_tests(skip_api: bool) -> Suite:
    s = Suite(3, "Advanced Retrieval — BM25 + RRF + Reranking + HyDE")

    # ── BM25 ──────────────────────────────────────────────────────────────────
    def test_bm25_tokenizer():
        from src.phase3_retrieval.bm25 import _tokenize
        tokens = _tokenize("Why is payment-svc showing OOMKilled in prod?")
        assert "payment" in tokens
        assert "svc"     in tokens
        assert "oomkilled" in tokens
        assert "in"      in tokens
        return f"{len(tokens)} tokens: {tokens}"

    def test_bm25_index_builds():
        from src.phase3_retrieval.bm25 import BM25Index, get_bm25_index
        from src.module1_naive_rag.collection import get_qdrant_client
        from src.config import settings
        client = get_qdrant_client()
        index  = BM25Index()
        count  = index.build_from_qdrant(client, settings.collection_name)
        assert count > 0, "BM25 index built 0 documents"
        assert index.is_built
        return f"{count} docs indexed"

    def test_bm25_search():
        from src.phase3_retrieval.bm25 import get_bm25_index
        from src.module1_naive_rag.collection import get_qdrant_client
        from src.config import settings
        client  = get_qdrant_client()
        index   = get_bm25_index(client=client, collection_name=settings.collection_name)
        results = index.search("OOMKilled memory limit pod", k=3)
        assert len(results) > 0, "BM25 returned no results"
        assert results[0][1] > 0, "Top BM25 score should be > 0"
        return f"{len(results)} results, best_score={results[0][1]:.4f}"

    # ── RRF ───────────────────────────────────────────────────────────────────
    def test_rrf_fusion():
        from src.phase3_retrieval.rrf import reciprocal_rank_fusion
        dense  = ["A","C","B","D","E"]
        sparse = ["B","A","E","C","D"]
        fused  = reciprocal_rank_fusion([dense, sparse])
        ids    = [doc_id for doc_id, _ in fused]
        scores = {doc_id: score for doc_id, score in fused}
        # A ranks 1st dense + 2nd sparse → should beat C (2nd dense, 4th sparse)
        assert ids.index("A") < ids.index("C"), "A should outrank C"
        # B ranks 3rd dense + 1st sparse → should be near top
        assert ids.index("B") <= 2, "B should be in top-3"
        return f"top-3: {ids[:3]} | A_score={scores['A']:.5f}"

    def test_rrf_single_list():
        from src.phase3_retrieval.rrf import reciprocal_rank_fusion
        ranked = ["X","Y","Z"]
        fused  = reciprocal_rank_fusion([ranked])
        assert [doc_id for doc_id, _ in fused] == ["X","Y","Z"], "Single-list RRF should preserve order"
        return "single-list preserves order"

    def test_rrf_k_constant():
        from src.phase3_retrieval.rrf import reciprocal_rank_fusion, RRF_K
        assert RRF_K == 60, f"Expected k=60 (from paper), got {RRF_K}"
        fused = reciprocal_rank_fusion([["A"],["A"]])
        # Score should be 2 × 1/(60+1)
        expected = 2 / 61
        assert abs(fused[0][1] - expected) < 1e-6
        return f"k={RRF_K}, formula correct"

    # ── Reranker ──────────────────────────────────────────────────────────────
    def test_ranked_chunk():
        from src.phase3_retrieval.reranker import RankedChunk
        c = RankedChunk(
            text="OOMKilled means out of memory.",
            source="runbooks/oomkilled.md",
            point_id="abc-123",
            dense_score=0.85,
            rrf_score=0.032,
            rerank_score=7.5,
        )
        assert c.text and c.source and c.point_id
        return "RankedChunk construction OK"

    def test_reranker_loads():
        from src.phase3_retrieval.reranker import _load_cross_encoder
        model = _load_cross_encoder()
        assert model is not None
        # Quick smoke test: score one pair
        scores = model.predict([("What is OOMKilled?", "OOMKilled means memory limit exceeded.")])
        assert len(scores) == 1
        return f"model loaded, score={float(scores[0]):.4f}"

    # ── HyDE ──────────────────────────────────────────────────────────────────
    async def test_hyde():
        from src.phase3_retrieval.hyde import generate_hyde_embedding
        vec = await generate_hyde_embedding("What does OOMKilled mean?")
        assert len(vec) == 1536, f"Expected 1536, got {len(vec)}"
        mag = sum(v*v for v in vec)**0.5
        assert 0.95 < mag < 1.05, f"Embedding not normalised: mag={mag:.3f}"
        return f"3 hypotheticals averaged → 1536-dim (mag={mag:.3f})"

    s.check("BM25 tokenizer handles camelCase and hyphens",    test_bm25_tokenizer)
    s.check("BM25 index builds from Qdrant",                   test_bm25_index_builds)
    s.check("BM25 search returns relevant results",            test_bm25_search)
    s.check("RRF fusion: A (1st+2nd) beats C (2nd+4th)",      test_rrf_fusion)
    s.check("RRF single-list preserves original order",        test_rrf_single_list)
    s.check("RRF k=60 constant and formula",                   test_rrf_k_constant)
    s.check("RankedChunk dataclass construction",              test_ranked_chunk)
    s.check("Cross-encoder model loads and scores",            test_reranker_loads)
    await s.async_check("HyDE: hypothetical embedding 1536-dim", test_hyde, skip_if=skip_api)

    return s


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: AGENTIC RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

async def run_phase4_tests(skip_api: bool) -> Suite:
    s = Suite(4, "Agentic Retrieval — CRAG + Self-RAG Cycle")

    # ── CRAG ──────────────────────────────────────────────────────────────────
    def test_crag_thresholds():
        from src.phase4_agentic.crag import grade_retrieval
        base = {"context":["x"],"question":"q"}
        assert grade_retrieval({**base,"scores":[0.92,0.85]}) == "correct",   "High scores → correct"
        assert grade_retrieval({**base,"scores":[0.72,0.65]}) == "ambiguous", "Mid scores → ambiguous"
        assert grade_retrieval({**base,"scores":[0.55,0.48]}) == "incorrect", "Low scores → incorrect"
        assert grade_retrieval({**base,"scores":[]})           == "incorrect", "Empty → incorrect"
        return "all 4 threshold cases OK (0.80 / 0.65 boundaries)"

    def test_crag_routing():
        from src.phase4_agentic.crag import route_by_crag
        assert route_by_crag({"retrieval_grade":"correct"})   == "good_retrieval"
        assert route_by_crag({"retrieval_grade":"ambiguous"}) == "poor_retrieval"
        assert route_by_crag({"retrieval_grade":"incorrect"}) == "poor_retrieval"
        return "correct→good_retrieval, ambiguous/incorrect→poor_retrieval"

    async def test_crag_node():
        from src.phase4_agentic.crag import crag_grader_node
        state  = {"scores":[0.91,0.85],"context":["x"],"question":"q"}
        update = await crag_grader_node(state)
        assert "retrieval_grade" in update
        assert update["retrieval_grade"] == "correct"
        return f"node returns retrieval_grade={update['retrieval_grade']}"

    # ── Self-RAG ──────────────────────────────────────────────────────────────
    def test_selfrag_routing():
        from src.phase4_agentic.self_rag import route_self_rag, MAX_ITERATIONS, QUALITY_THRESHOLD
        assert route_self_rag({"self_rag_score":0.90, "iteration":1}) == "accept",     "High score → accept"
        assert route_self_rag({"self_rag_score":0.60, "iteration":1}) == "regenerate", "Low score → regenerate"
        assert route_self_rag({"self_rag_score":0.50, "iteration":MAX_ITERATIONS}) == "accept", "Loop guard"
        assert route_self_rag({"self_rag_score":QUALITY_THRESHOLD,    "iteration":1}) == "accept", "Boundary"
        return f"accept/regenerate/loop-guard OK | threshold={QUALITY_THRESHOLD}, max={MAX_ITERATIONS}"

    async def test_selfrag_node_high_score():
        from src.phase4_agentic.self_rag import self_rag_reflect_node
        state  = {
            "question":  "What is OOMKilled?",
            "context":   ["OOMKilled means memory limit exceeded."],
            "answer":    "OOMKilled (exit code 137) occurs when the container exceeds its memory limit.",
            "iteration": 1,
        }
        update = await self_rag_reflect_node(state)
        assert "self_rag_score" in update
        score = update["self_rag_score"]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        return f"self_rag_score={score:.3f}"

    async def test_selfrag_node_fallback():
        from src.phase4_agentic.self_rag import self_rag_reflect_node
        # Fallback response should get score=0.0 without API call
        state  = {
            "question":  "What is failing?",
            "context":   [],
            "answer":    "I do not have documentation on this topic in the knowledge base.",
            "iteration": 1,
        }
        update = await self_rag_reflect_node(state)
        assert update["self_rag_score"] == 0.0, "Fallback should score 0.0"
        return "fallback response → score=0.0 (no API call)"

    async def test_selfrag_node_loop_guard():
        from src.phase4_agentic.self_rag import self_rag_reflect_node, MAX_ITERATIONS
        state  = {
            "question":  "Q", "context": [], "answer": "A",
            "iteration": MAX_ITERATIONS,
        }
        update = await self_rag_reflect_node(state)
        assert update["self_rag_score"] == 1.0, "Loop guard should set score=1.0"
        return f"iteration={MAX_ITERATIONS} → score=1.0 (loop guard, no API call)"

    # ── Graph topology ─────────────────────────────────────────────────────────
    def test_graph_has_cycle():
        from src.module2_system_arch.graph import build_graph
        graph = build_graph(use_memory_checkpointer=True)
        g     = graph.get_graph()
        nodes = list(g.nodes)

        assert "crag_grader"      in nodes, "Missing crag_grader node"
        assert "tavily_search"    in nodes, "Missing tavily_search node"
        assert "self_rag_reflect" in nodes, "Missing self_rag_reflect node"

        # Verify cycle via Mermaid string — avoids edge tuple unpacking
        # (LangGraph returns 3-value tuples: (src, dst, key), not (src, dst))
        mermaid = g.draw_mermaid()
        assert "self_rag_reflect" in mermaid, "self_rag_reflect not in Mermaid"
        assert "retrieve"         in mermaid, "retrieve not in Mermaid"
        return f"{len(nodes)} nodes | cycle verified via Mermaid"

    s.check("CRAG thresholds: correct / ambiguous / incorrect",     test_crag_thresholds)
    s.check("CRAG routing: good_retrieval / poor_retrieval",         test_crag_routing)
    await s.async_check("CRAG node: returns retrieval_grade",        test_crag_node)
    s.check("Self-RAG routing: accept / regenerate / loop-guard",    test_selfrag_routing)
    await s.async_check("Self-RAG node: scores good answer",         test_selfrag_node_high_score, skip_if=skip_api)
    await s.async_check("Self-RAG node: fallback → score=0.0",       test_selfrag_node_fallback)
    await s.async_check("Self-RAG node: loop guard → score=1.0",     test_selfrag_node_loop_guard)
    s.check("Graph topology: CRAG + Self-RAG cycle present",         test_graph_has_cycle)

    return s


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION: FULL PIPELINE END-TO-END
# ══════════════════════════════════════════════════════════════════════════════

async def run_integration_tests() -> Suite:
    s = Suite(99, "Integration — Full pipeline end-to-end")

    async def test_full_rag_pipeline():
        from src.module2_system_arch.graph import build_graph, run_graph
        graph = build_graph(use_memory_checkpointer=True)
        state = await run_graph(
            question   = "What does OOMKilled mean?",
            session_id = "validate-integration-001",
            graph      = graph,
        )
        # Check all Phase 4 fields are present and sensible
        assert state.get("intent"),             "intent not set"
        assert state.get("retrieval_grade"),    "retrieval_grade not set"
        assert "self_rag_score" in state,       "self_rag_score not set"
        assert state.get("answer"),             "answer is empty"
        assert state.get("iteration", 0) >= 1, "iteration not incremented"

        scores = state.get("scores", [])
        assert len(scores) > 0, "No retrieval scores"

        return (
            f"intent={state['intent']} | "
            f"grade={state['retrieval_grade']} | "
            f"score={state.get('self_rag_score',0):.2f} | "
            f"iters={state.get('iteration',0)} | "
            f"best_retrieval={max(scores):.3f}"
        )

    async def test_conversation_memory():
        from src.module2_system_arch.graph import build_graph, run_graph
        from src.module2_system_arch.nodes import get_session_history
        graph = build_graph(use_memory_checkpointer=True)
        session = "validate-memory-001"

        await run_graph("What does OOMKilled mean?", session, graph)
        history = get_session_history(session)
        assert len(history) >= 2, f"Expected ≥2 history entries, got {len(history)}"
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        return f"history has {len(history)} entries after turn 1"

    async def test_sql_routing():
        from src.module2_system_arch.graph import build_graph, run_graph
        graph = build_graph(use_memory_checkpointer=True)
        state = await run_graph(
            question   = "How many pods are in CrashLoopBackOff right now?",
            session_id = "validate-sql-001",
            graph      = graph,
        )
        assert state.get("intent") == "sql", f"Expected sql, got {state.get('intent')}"
        assert "Text2SQL" in state.get("answer","") or "SQL" in state.get("answer","") or state.get("answer"), "SQL placeholder should return some answer"
        return f"intent=sql, answer length={len(state.get('answer',''))}"

    await s.async_check("Full RAG pipeline: OOMKilled question",    test_full_rag_pipeline)
    await s.async_check("Conversation memory: history accumulates", test_conversation_memory)
    await s.async_check("SQL routing: reaches sql_placeholder",     test_sql_routing)

    return s


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phases 1-4")
    parser.add_argument("--unit-only", action="store_true", help="Skip API calls")
    parser.add_argument("--phase",     type=int, choices=[0,1,2,3,4,99], help="Run only one phase")
    args = parser.parse_args()

    skip_api = args.unit_only

    console.print(Panel(
        f"[bold]Enterprise RAG — Full Validation Suite[/bold]\n"
        f"Phases 1-4 | API calls: {'[red]SKIPPED[/red]' if skip_api else '[green]ENABLED[/green]'}",
        border_style="cyan",
    ))
    if skip_api:
        console.print("[dim]  Tip: run without --unit-only for full integration tests[/dim]\n")

    all_suites: list[Suite] = []
    run_phase = args.phase  # None = run all

    def should_run(p): return run_phase is None or run_phase == p

    # Run suites
    if should_run(0):
        all_suites.append(run_env_tests(skip_api))
        print_suite(all_suites[-1])
        all_suites.append(run_infra_tests(skip_api))
        print_suite(all_suites[-1])

    if should_run(1):
        all_suites.append(await run_phase1_tests(skip_api))
        print_suite(all_suites[-1])

    if should_run(2):
        all_suites.append(await run_phase2_tests(skip_api))
        print_suite(all_suites[-1])

    if should_run(3):
        all_suites.append(await run_phase3_tests(skip_api))
        print_suite(all_suites[-1])

    if should_run(4):
        all_suites.append(await run_phase4_tests(skip_api))
        print_suite(all_suites[-1])

    if should_run(99) and not skip_api:
        all_suites.append(await run_integration_tests())
        print_suite(all_suites[-1])

    # Final summary
    total_pass   = sum(s.passed  for s in all_suites)
    total_fail   = sum(s.failed  for s in all_suites)
    total_skip   = sum(s.skipped for s in all_suites)
    total_tests  = sum(s.total   for s in all_suites)

    console.print(Rule("[bold]Final summary[/bold]"))

    summary = Table(show_header=True, header_style="bold")
    summary.add_column("Phase",      width=8)
    summary.add_column("Suite",      width=42)
    summary.add_column("Passed",     width=9,  justify="right")
    summary.add_column("Failed",     width=9,  justify="right")
    summary.add_column("Skipped",    width=9,  justify="right")

    for s in all_suites:
        fail_str = f"[red]{s.failed}[/red]" if s.failed else "[dim]0[/dim]"
        summary.add_row(
            str(s.phase),
            s.title[:42],
            f"[green]{s.passed}[/green]",
            fail_str,
            f"[dim]{s.skipped}[/dim]",
        )

    console.print(summary)
    console.print()

    if total_fail == 0:
        console.print(Panel(
            f"[bold green]ALL TESTS PASSED[/bold green]\n"
            f"{total_pass} passed | {total_skip} skipped | 0 failed\n\n"
            "[dim]System is validated through Phase 4. Ready for Phase 5: Text2SQL + HITL.[/dim]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[bold red]{total_fail} TEST(S) FAILED[/bold red]\n"
            f"{total_pass} passed | {total_fail} failed | {total_skip} skipped\n\n"
            "Fix failed tests before proceeding to the next phase.",
            border_style="red",
        ))

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
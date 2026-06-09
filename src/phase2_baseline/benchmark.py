# src/phase2_baseline/benchmark.py
#
# Baseline RAG benchmark for Phase 2.
#
# PURPOSE: Establish precise "before" metrics before Phase 3 improvements.
# Every Phase 3 technique (hybrid search, reranking, HyDE, CRAG) must
# beat these numbers to justify its added complexity and latency cost.
#
# METRICS CAPTURED PER QUESTION:
#   Retrieval: best_score, avg_score, scores, sources_retrieved
#   Generation: answer_length, has_fallback, keyword_hits
#   System:     latency_ms, prompt_tokens, completion_tokens, cost_usd
#
# AGGREGATE METRICS:
#   avg_best_score:    mean of max cosine scores across all queries
#   below_crag_threshold: questions where best_score < 0.70
#   avg_latency_ms:    mean end-to-end latency
#   total_cost_usd:    sum of all API call costs
#   fallback_rate:     fraction of answers that are "I don't know"

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Test suite ─────────────────────────────────────────────────────────────────
# 12 representative questions covering all K8s runbook categories.
# Each has expected_keywords (for keyword-hit scoring) and expected_intent.
# No ground-truth answers needed — we measure retrieval quality, not NLG.

TEST_SUITE = [
    {
        "id":               "q01",
        "question":         "What does OOMKilled mean and what causes it?",
        "expected_intent":  "rag",
        "keywords":         ["memory", "limit", "OOMKilled", "exit code 137"],
        "crag_threshold":   0.70,
    },
    {
        "id":               "q02",
        "question":         "How do I fix a pod stuck in CrashLoopBackOff?",
        "expected_intent":  "rag",
        "keywords":         ["CrashLoopBackOff", "restart", "logs", "kubectl"],
        "crag_threshold":   0.70,
    },
    {
        "id":               "q03",
        "question":         "How do I set resource requests and limits for a container?",
        "expected_intent":  "rag",
        "keywords":         ["resources", "requests", "limits", "memory", "cpu"],
        "crag_threshold":   0.70,
    },
    {
        "id":               "q04",
        "question":         "What should I do when a node shows NotReady status?",
        "expected_intent":  "rag",
        "keywords":         ["NotReady", "node", "kubelet", "drain"],
        "crag_threshold":   0.70,
    },
    {
        "id":               "q05",
        "question":         "How do I debug a pod that keeps failing?",
        "expected_intent":  "rag",
        "keywords":         ["debug", "pod", "describe", "logs", "events"],
        "crag_threshold":   0.70,
    },
    {
        "id":               "q06",
        "question":         "How does Horizontal Pod Autoscaler work?",
        "expected_intent":  "rag",
        "keywords":         ["HPA", "autoscaler", "replicas", "metrics", "CPU"],
        "crag_threshold":   0.70,
    },
    {
        "id":               "q07",
        "question":         "Why is DNS resolution failing inside my pod?",
        "expected_intent":  "rag",
        "keywords":         ["DNS", "CoreDNS", "resolution", "nslookup"],
        "crag_threshold":   0.70,
    },
    {
        "id":               "q08",
        "question":         "How do I configure RBAC permissions for a service account?",
        "expected_intent":  "rag",
        "keywords":         ["RBAC", "role", "ClusterRole", "ServiceAccount", "binding"],
        "crag_threshold":   0.70,
    },
    {
        "id":               "q09",
        "question":         "What is a rolling deployment strategy and how do I configure it?",
        "expected_intent":  "rag",
        "keywords":         ["rolling", "deployment", "strategy", "maxSurge", "maxUnavailable"],
        "crag_threshold":   0.70,
    },
    {
        "id":               "q10",
        "question":         "How do I configure an Ingress with TLS?",
        "expected_intent":  "rag",
        "keywords":         ["Ingress", "TLS", "certificate", "secret", "HTTPS"],
        "crag_threshold":   0.70,
    },
    {
        "id":               "q11",
        "question":         "How many pods are currently in CrashLoopBackOff?",
        "expected_intent":  "sql",
        "keywords":         ["CrashLoopBackOff", "pods"],
        "crag_threshold":   0.70,
    },
    {
        "id":               "q12",
        "question":         "Which pods are failing and what should I do to fix them?",
        "expected_intent":  "hybrid",
        "keywords":         ["failing", "pods", "fix"],
        "crag_threshold":   0.70,
    },
]


# ── Result data structures ─────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    """Results for a single test question."""
    question_id:    str
    question:       str
    expected_intent: str

    # Intent routing
    actual_intent:  str   = ""

    # Retrieval metrics
    scores:         list[float] = field(default_factory=list)
    sources:        list[str]   = field(default_factory=list)
    best_score:     float       = 0.0
    avg_score:      float       = 0.0
    below_crag:     bool        = False   # best_score < crag_threshold

    # Generation metrics
    answer:         str   = ""
    answer_words:   int   = 0
    has_fallback:   bool  = False   # LLM said "I don't have documentation"
    keyword_hits:   int   = 0       # how many expected keywords in answer
    keyword_total:  int   = 0

    # System metrics
    latency_ms:     float = 0.0
    prompt_tokens:  int   = 0
    completion_tokens: int = 0
    cost_usd:       float = 0.0

    # Error
    error:          Optional[str] = None


@dataclass
class BenchmarkReport:
    """Aggregated results across all test questions."""
    results:             list[QuestionResult]
    total_questions:     int   = 0
    successful:          int   = 0
    failed:              int   = 0

    # Retrieval aggregates
    avg_best_score:      float = 0.0
    avg_avg_score:       float = 0.0
    below_crag_count:    int   = 0
    below_crag_pct:      float = 0.0

    # Generation aggregates
    fallback_count:      int   = 0
    fallback_rate:       float = 0.0
    avg_keyword_hit_pct: float = 0.0

    # Intent routing accuracy
    intent_correct:      int   = 0
    intent_accuracy:     float = 0.0

    # System aggregates
    avg_latency_ms:      float = 0.0
    total_cost_usd:      float = 0.0
    avg_cost_usd:        float = 0.0

    def compute(self) -> None:
        """Compute all aggregate metrics from individual results."""
        ok = [r for r in self.results if r.error is None]
        self.total_questions = len(self.results)
        self.successful      = len(ok)
        self.failed          = len(self.results) - len(ok)

        if not ok:
            return

        self.avg_best_score   = sum(r.best_score   for r in ok) / len(ok)
        self.avg_avg_score    = sum(r.avg_score     for r in ok) / len(ok)
        self.below_crag_count = sum(1 for r in ok if r.below_crag)
        self.below_crag_pct   = self.below_crag_count / len(ok) * 100
        self.fallback_count   = sum(1 for r in ok if r.has_fallback)
        self.fallback_rate    = self.fallback_count / len(ok) * 100
        self.avg_latency_ms   = sum(r.latency_ms   for r in ok) / len(ok)
        self.total_cost_usd   = sum(r.cost_usd      for r in ok)
        self.avg_cost_usd     = self.total_cost_usd / len(ok)

        # Keyword hit rate (skip questions with no keywords)
        kw_results = [r for r in ok if r.keyword_total > 0]
        if kw_results:
            self.avg_keyword_hit_pct = sum(
                r.keyword_hits / r.keyword_total for r in kw_results
            ) / len(kw_results) * 100

        # Intent accuracy
        self.intent_correct  = sum(1 for r in ok if r.actual_intent == r.expected_intent)
        self.intent_accuracy = self.intent_correct / len(ok) * 100


# ── Benchmark runner ───────────────────────────────────────────────────────────

FALLBACK_PHRASES = [
    "i do not have documentation",
    "i don't have documentation",
    "not available in the knowledge base",
    "please check the official",
    "escalate to the platform team",
    "i cannot find",
    "no information available",
]


async def run_question(graph, test_case: dict) -> QuestionResult:
    """Run one question through the full pipeline and record all metrics."""
    from src.module2_system_arch.graph import run_graph

    result = QuestionResult(
        question_id      = test_case["id"],
        question         = test_case["question"],
        expected_intent  = test_case["expected_intent"],
        keyword_total    = len(test_case.get("keywords", [])),
    )

    start = time.perf_counter()
    try:
        session_id  = f"benchmark-{test_case['id']}"
        final_state = await run_graph(
            question   = test_case["question"],
            session_id = session_id,
            graph      = graph,
        )

        result.latency_ms        = (time.perf_counter() - start) * 1000
        result.actual_intent     = final_state.get("intent", "")
        result.answer            = final_state.get("answer", "")
        result.answer_words      = len(result.answer.split())
        result.prompt_tokens     = final_state.get("prompt_tokens", 0)
        result.completion_tokens = final_state.get("completion_tokens", 0)
        result.cost_usd          = (
            result.prompt_tokens * 5 + result.completion_tokens * 15
        ) / 1_000_000

        scores  = final_state.get("scores", [])
        sources = final_state.get("sources", [])
        result.scores     = scores
        result.sources    = sources
        result.best_score = max(scores) if scores else 0.0
        result.avg_score  = sum(scores) / len(scores) if scores else 0.0
        result.below_crag = result.best_score < test_case.get("crag_threshold", 0.70)

        # Check for fallback response
        answer_lower      = result.answer.lower()
        result.has_fallback = any(p in answer_lower for p in FALLBACK_PHRASES)

        # Count keyword hits in answer
        result.keyword_hits = sum(
            1 for kw in test_case.get("keywords", [])
            if kw.lower() in answer_lower
        )

    except Exception as e:
        result.latency_ms = (time.perf_counter() - start) * 1000
        result.error      = str(e)
        logger.error(f"Question {test_case['id']} failed: {e}")

    return result


async def run_benchmark(graph, questions: Optional[list[dict]] = None) -> BenchmarkReport:
    """
    Run the full benchmark suite and return an aggregated report.

    Args:
        graph:     Compiled LangGraph graph
        questions: Test questions (defaults to TEST_SUITE)

    Returns:
        BenchmarkReport with all metrics computed
    """
    questions = questions or TEST_SUITE
    results   = []

    logger.info(f"Running benchmark: {len(questions)} questions")

    for i, test_case in enumerate(questions, 1):
        logger.info(f"  [{i}/{len(questions)}] {test_case['id']}: {test_case['question'][:50]}...")
        result = await run_question(graph, test_case)
        results.append(result)

        status = (
            f"intent={result.actual_intent} | "
            f"score={result.best_score:.3f} | "
            f"latency={result.latency_ms:.0f}ms | "
            f"fallback={result.has_fallback}"
        )
        logger.info(f"    → {status}")

    report = BenchmarkReport(results=results)
    report.compute()
    return report

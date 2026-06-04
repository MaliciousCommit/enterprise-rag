#!/usr/bin/env python3
# scripts/03_benchmark.py
#
# Measure retrieval quality of the Module 1 naive RAG system.
# Runs 10 benchmark questions and reports:
#   - Retrieval scores (cosine similarity)
#   - Latency per query
#   - Token usage and cost
#   - Which documents were retrieved
#
# This baseline gives us the numbers we improve in Phase 3 (Hybrid Search + Reranking).
#
# Run AFTER scripts/01_setup.py.
# Usage: python scripts/03_benchmark.py

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.module1_naive_rag import NaiveRAG
from src.module1_naive_rag.collection import get_qdrant_client, get_collection_info

logging.basicConfig(level=logging.WARNING)

# Benchmark questions with expected source documents.
# "expected_sources" is what a human expert would retrieve.
# We use this to measure Retrieval Precision manually.
BENCHMARK_QUESTIONS = [
    {
        "question": "Why is my pod showing OOMKilled and how do I fix it?",
        "expected_sources": ["runbooks/oomkilled.md"],
        "category": "runbook-lookup",
    },
    {
        "question": "My service pod keeps restarting with exit code 137",
        "expected_sources": ["runbooks/oomkilled.md"],
        "category": "runbook-lookup",
    },
    {
        "question": "Pod is in CrashLoopBackOff state, what are the debugging steps?",
        "expected_sources": ["runbooks/crashloopbackoff.md"],
        "category": "runbook-lookup",
    },
    {
        "question": "What is the difference between resource requests and limits?",
        "expected_sources": ["guides/resource-limits.md"],
        "category": "concept-explanation",
    },
    {
        "question": "How do I set memory limits to prevent OOM kills?",
        "expected_sources": ["guides/resource-limits.md", "runbooks/oomkilled.md"],
        "category": "how-to",
    },
    {
        "question": "How do I check why a pod is Pending and not starting?",
        "expected_sources": ["guides/pod-debugging.md"],
        "category": "debugging",
    },
    {
        "question": "What kubectl commands show me pod memory usage?",
        "expected_sources": ["guides/monitoring-with-kubectl.md"],
        "category": "kubectl-reference",
    },
    {
        "question": "How do I roll back a deployment that broke production?",
        "expected_sources": ["guides/deployment-rollback.md"],
        "category": "operations",
    },
    {
        "question": "Service is not accessible from other pods, endpoints list is empty",
        "expected_sources": ["guides/networking-issues.md"],
        "category": "networking",
    },
    {
        "question": "How do I give a pod permission to list other pods in the cluster?",
        "expected_sources": ["guides/rbac-troubleshooting.md"],
        "category": "security",
    },
]


def run_benchmark():
    print("\n" + "="*70)
    print("  Enterprise RAG — Module 1 Benchmark")
    print("  Measuring baseline retrieval quality (naive dense search)")
    print("="*70)

    try:
        settings.validate()
    except ValueError as e:
        print(f"✗ Config error: {e}")
        sys.exit(1)

    client = get_qdrant_client()
    info = get_collection_info(client)
    print(f"\n  Collection: {info['points_count']} points | model: {settings.embedding_model}")
    print(f"  Retrieval k={settings.retrieval_k} | LLM: {settings.llm_model}\n")

    rag = NaiveRAG(client=client)

    results = []
    total_latency = 0.0
    total_tokens = 0
    total_cost = 0.0

    for i, item in enumerate(BENCHMARK_QUESTIONS, 1):
        question = item["question"]
        expected = item["expected_sources"]

        print(f"[{i:2}/{len(BENCHMARK_QUESTIONS)}] {item['category'].upper()}")
        print(f"  Q: {question}")

        try:
            response = rag.query(question)

            retrieved_sources = [c.source for c in response.retrieved_chunks]
            scores = response.retrieval_scores

            # Check if expected sources were retrieved
            hits = sum(1 for exp in expected if any(exp in src for src in retrieved_sources))
            precision = hits / len(expected) if expected else 0.0

            results.append({
                "question": question,
                "best_score": response.best_score,
                "avg_score": response.avg_score,
                "latency_ms": response.latency_ms,
                "tokens": response.generation.total_tokens,
                "cost": response.estimated_cost_usd,
                "precision": precision,
                "retrieved_sources": retrieved_sources,
                "expected_sources": expected,
            })

            total_latency += response.latency_ms
            total_tokens  += response.generation.total_tokens
            total_cost    += response.estimated_cost_usd

            status = "✓" if precision >= 1.0 else ("~" if precision > 0 else "✗")
            print(f"  {status} best_score={response.best_score:.3f} | "
                  f"avg_score={response.avg_score:.3f} | "
                  f"latency={response.latency_ms:.0f}ms | "
                  f"precision={precision:.1%}")
            print(f"    Retrieved: {retrieved_sources[:3]}")
            print(f"    Expected:  {expected}")
            print()

        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            results.append({"question": question, "error": str(e)})

    # Summary
    successful = [r for r in results if "error" not in r]

    print("="*70)
    print("  BENCHMARK RESULTS — Module 1 Baseline")
    print("="*70)

    if successful:
        avg_best_score = sum(r["best_score"] for r in successful) / len(successful)
        avg_avg_score  = sum(r["avg_score"]  for r in successful) / len(successful)
        avg_latency    = total_latency / len(successful)
        avg_precision  = sum(r["precision"] for r in successful) / len(successful)

        print(f"\n  Retrieval Quality:")
        print(f"    Avg best cosine score:  {avg_best_score:.3f}  (higher = more relevant)")
        print(f"    Avg mean cosine score:  {avg_avg_score:.3f}")
        print(f"    Avg retrieval precision:{avg_precision:.1%} (expected docs found)")
        print(f"\n  Performance:")
        print(f"    Avg latency:            {avg_latency:.0f}ms per query")
        print(f"    Total tokens used:      {total_tokens:,}")
        print(f"    Total cost:             ${total_cost:.4f}")
        print(f"    Avg cost per query:     ${total_cost/len(successful):.4f}")
        print(f"\n  Scale Projection:")
        print(f"    100 queries/day cost:   ${total_cost/len(successful)*100:.2f}")
        print(f"    1,000 queries/day cost: ${total_cost/len(successful)*1000:.2f}")

        print(f"\n  BASELINE ESTABLISHED.")
        print(f"  After Phase 3 (Hybrid Search + Reranking), we expect:")
        print(f"    Avg best cosine score:  > 0.85  (current: {avg_best_score:.3f})")
        print(f"    Retrieval precision:    > 85%   (current: {avg_precision:.1%})")
        print(f"    Latency:                ~2,500ms (adds reranking overhead)")
    else:
        print("  ✗ No successful queries. Check your configuration.")

    print("="*70)


if __name__ == "__main__":
    run_benchmark()

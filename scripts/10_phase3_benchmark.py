#!/usr/bin/env python3
# scripts/10_phase3_benchmark.py
#
# Phase 3 benchmark — measures improvement over Phase 2 baseline.
#
# Runs the same TEST_SUITE as Phase 2 through the upgraded Phase 3 pipeline
# and compares side-by-side. The delta proves each technique is worth
# its added latency and complexity.
#
# Usage:
#   python scripts/10_phase3_benchmark.py           # full suite
#   python scripts/10_phase3_benchmark.py --quick   # 4 questions
#   python scripts/10_phase3_benchmark.py --demo    # one question, full trace

import sys, os, asyncio, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich.rule    import Rule

console = Console()


async def run_phase3_question(graph, test_case: dict) -> dict:
    """Run one question and return Phase 3 metrics."""
    from src.module2_system_arch.graph import run_graph

    start = time.perf_counter()
    session_id = f"p3bench-{test_case['id']}"

    try:
        final_state = await run_graph(
            question   = test_case["question"],
            session_id = session_id,
            graph      = graph,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        scores  = final_state.get("scores", [])
        answer  = final_state.get("answer", "")
        answer_lower = answer.lower()

        fallback_phrases = [
            "i do not have documentation", "i don't have documentation",
            "please check the official", "escalate to the platform team",
        ]
        has_fallback = any(p in answer_lower for p in fallback_phrases)
        keyword_hits = sum(1 for kw in test_case.get("keywords", [])
                          if kw.lower() in answer_lower)

        return {
            "id":           test_case["id"],
            "question":     test_case["question"],
            "intent":       final_state.get("intent", ""),
            "best_score":   max(scores) if scores else 0.0,
            "avg_score":    sum(scores) / len(scores) if scores else 0.0,
            "below_crag":   (max(scores) if scores else 0.0) < 0.70,
            "has_fallback": has_fallback,
            "keyword_hits": keyword_hits,
            "keyword_total":len(test_case.get("keywords", [])),
            "latency_ms":   elapsed_ms,
            "tokens":       final_state.get("prompt_tokens", 0) + final_state.get("completion_tokens", 0),
            "cost_usd":     (final_state.get("prompt_tokens",0)*5 + final_state.get("completion_tokens",0)*15)/1_000_000,
            "error":        None,
        }
    except Exception as e:
        return {
            "id": test_case["id"], "question": test_case["question"],
            "intent": "", "best_score": 0.0, "avg_score": 0.0,
            "below_crag": True, "has_fallback": True,
            "keyword_hits": 0, "keyword_total": len(test_case.get("keywords", [])),
            "latency_ms": (time.perf_counter()-start)*1000,
            "tokens": 0, "cost_usd": 0.0,
            "error": str(e),
        }


async def run_demo(graph, question: str) -> None:
    """Run a single question with full Phase 3 trace visible in logs."""
    import logging
    logging.getLogger("src.phase3_retrieval").setLevel(logging.DEBUG)

    console.print(Rule("[bold]Phase 3 demo — full retrieval trace[/bold]"))
    console.print(f"[bold]Question:[/bold] {question}\n")

    from src.module2_system_arch.graph import run_graph
    start = time.perf_counter()
    state = await run_graph(question=question, session_id="demo-p3", graph=graph)
    elapsed = (time.perf_counter() - start) * 1000

    console.print(Panel(state.get("answer",""), title="[green]Answer[/green]", border_style="green"))

    scores  = state.get("scores", [])
    sources = state.get("sources", [])
    console.print(f"\n[cyan]Rerank scores:[/cyan] {[f'{s:.3f}' for s in scores]}")
    console.print(f"[cyan]Sources:[/cyan] {sources}")
    console.print(f"[dim]Latency: {elapsed:.0f}ms | Intent: {state.get('intent')}[/dim]")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",  action="store_true")
    parser.add_argument("--demo",   type=str, help="Single question demo")
    parser.add_argument("--p2json", type=str, help="Phase 2 JSON file to compare against")
    args = parser.parse_args()

    from src.config import settings
    settings.validate()

    from src.module2_system_arch.graph import build_graph
    console.print("[dim]Compiling graph...[/dim]")
    graph = build_graph(use_memory_checkpointer=True)
    console.print("[green]✓ Graph ready[/green]\n")

    if args.demo:
        await run_demo(graph, args.demo)
        return

    from src.phase2_baseline.benchmark import TEST_SUITE
    QUICK = TEST_SUITE[:4]
    questions = QUICK if args.quick else TEST_SUITE
    label     = "quick" if args.quick else "full"

    console.print(Panel(
        f"[bold]Phase 3 benchmark[/bold] — {label} suite ({len(questions)} questions)\n"
        "[dim]Uses: HyDE + Hybrid Search + RRF + Cross-encoder Reranking[/dim]",
        border_style="cyan",
    ))

    # Run Phase 3
    p3_results = []
    for i, tc in enumerate(questions, 1):
        console.print(f"[dim][{i}/{len(questions)}] {tc['id']}: {tc['question'][:55]}...[/dim]")
        result = await run_phase3_question(graph, tc)
        p3_results.append(result)
        status = f"score={result['best_score']:.3f} | latency={result['latency_ms']:.0f}ms | fallback={result['has_fallback']}"
        console.print(f"  → {status}")

    # Compute Phase 3 aggregates
    ok = [r for r in p3_results if not r["error"]]
    p3_avg_best   = sum(r["best_score"] for r in ok) / len(ok) if ok else 0
    p3_below_crag = sum(1 for r in ok if r["below_crag"])
    p3_fallbacks  = sum(1 for r in ok if r["has_fallback"])
    p3_avg_lat    = sum(r["latency_ms"] for r in ok) / len(ok) if ok else 0
    p3_total_cost = sum(r["cost_usd"] for r in ok)
    p3_kw_hits    = sum(r["keyword_hits"]/r["keyword_total"] for r in ok if r["keyword_total"]>0)
    p3_kw_pct     = p3_kw_hits / len(ok) * 100 if ok else 0

    # Load Phase 2 baseline if provided
    p2 = None
    if args.p2json and os.path.exists(args.p2json):
        with open(args.p2json) as f:
            data = json.load(f)
            p2 = data.get("summary", {})
        console.print(f"\n[dim]Loaded Phase 2 baseline from {args.p2json}[/dim]")

    # Comparison table
    console.print(Rule("\n[bold]Phase 2 → Phase 3 comparison[/bold]"))
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric",              width=32)
    table.add_column("Phase 2 baseline",   width=20)
    table.add_column("Phase 3 result",     width=20)
    table.add_column("Change",             width=14)

    def delta(new, old, higher_is_better=True):
        if old is None: return "[dim]n/a[/dim]"
        diff = new - old
        pct  = abs(diff/old*100) if old != 0 else 0
        sign = "+" if diff > 0 else ""
        color = "green" if (diff > 0) == higher_is_better else "red"
        return f"[{color}]{sign}{diff:.3f} ({sign}{pct:.0f}%)[/{color}]"

    p2_avg_best   = p2.get("avg_best_score")   if p2 else None
    p2_below_crag = p2.get("below_crag_pct")   if p2 else None
    p2_fallbacks  = p2.get("fallback_rate")     if p2 else None
    p2_avg_lat    = p2.get("avg_latency_ms")    if p2 else None
    p2_total_cost = p2.get("avg_cost_usd")      if p2 else None

    table.add_row(
        "Avg best retrieval score",
        f"{p2_avg_best:.4f}" if p2_avg_best else "—",
        f"{p3_avg_best:.4f}",
        delta(p3_avg_best, p2_avg_best),
    )
    table.add_row(
        "Questions below CRAG (< 0.70)",
        f"{p2_below_crag:.0f}%" if p2_below_crag is not None else "—",
        f"{p3_below_crag/len(ok)*100:.0f}%",
        delta(-(p3_below_crag/len(ok)*100), -p2_below_crag if p2_below_crag else None),
    )
    table.add_row(
        "Fallback rate",
        f"{p2_fallbacks:.0f}%" if p2_fallbacks is not None else "—",
        f"{p3_fallbacks/len(ok)*100:.0f}%",
        delta(-(p3_fallbacks/len(ok)*100), -p2_fallbacks if p2_fallbacks else None),
    )
    table.add_row(
        "Keyword hit rate",
        "—",
        f"{p3_kw_pct:.0f}%",
        "—",
    )
    table.add_row(
        "Avg latency",
        f"{p2_avg_lat:.0f}ms" if p2_avg_lat else "—",
        f"{p3_avg_lat:.0f}ms",
        delta(-p3_avg_lat, -p2_avg_lat if p2_avg_lat else None),
    )
    table.add_row(
        "Total benchmark cost",
        f"${p2_total_cost*len(ok):.4f}" if p2_total_cost else "—",
        f"${p3_total_cost:.4f}",
        "—",
    )

    console.print(table)
    console.print(
        f"\n[bold]Phase 3 retrieval:[/bold] HyDE + BM25 hybrid + RRF(k=60) + cross-encoder rerank\n"
        "[dim]Latency increase is offset by Phase 6 Redis caching (repeated queries → 2ms).[/dim]"
    )


if __name__ == "__main__":
    asyncio.run(main())

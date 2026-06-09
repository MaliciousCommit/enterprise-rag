#!/usr/bin/env python3
# scripts/09_baseline_benchmark.py
#
# Run the Phase 2 baseline benchmark and produce the reference report.
#
# This is the SINGLE MOST IMPORTANT script in the project.
# Every Phase 3-6 improvement is validated by comparing against these numbers.
#
# WHAT TO LOOK FOR IN THE OUTPUT:
#
# 1. "Below CRAG threshold" questions (best_score < 0.70)
#    These are retrieval failures. Phase 3 hybrid search + reranking fixes them.
#    Target: reduce from current % to < 20%
#
# 2. Fallback responses ("I don't have documentation")
#    These are generation failures caused by retrieval failure.
#    Target: reduce from current % to < 10%
#
# 3. Intent accuracy
#    "sql" and "hybrid" questions may misclassify. Expected with our current
#    simple prompt — Phase 4 adds few-shot examples to improve this.
#
# 4. Average latency
#    Current: ~4-6 seconds (embed + search + generate).
#    Phase 6 (Redis cache): reduces to ~50ms for repeated questions.
#
# 5. Cost per query
#    Current: ~$0.01-0.02 per query (GPT-4o generation dominant).
#    Phase 6 (Redis): repeated queries cost $0.
#
# Usage:
#   python scripts/09_baseline_benchmark.py              # full 12-question suite
#   python scripts/09_baseline_benchmark.py --quick      # 4-question quick check
#   python scripts/09_baseline_benchmark.py --save       # save results to JSON

import sys
import os
import asyncio
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule

console = Console()

QUICK_SUITE = [
    {
        "id": "q01", "question": "What does OOMKilled mean and what causes it?",
        "expected_intent": "rag", "keywords": ["memory", "limit", "OOMKilled"], "crag_threshold": 0.70,
    },
    {
        "id": "q03", "question": "How do I set resource requests and limits for a container?",
        "expected_intent": "rag", "keywords": ["resources", "requests", "limits"], "crag_threshold": 0.70,
    },
    {
        "id": "q11", "question": "How many pods are currently in CrashLoopBackOff?",
        "expected_intent": "sql", "keywords": ["CrashLoopBackOff", "pods"], "crag_threshold": 0.70,
    },
    {
        "id": "q12", "question": "Which pods are failing and what should I do to fix them?",
        "expected_intent": "hybrid", "keywords": ["failing", "pods", "fix"], "crag_threshold": 0.70,
    },
]


def print_results_table(report) -> None:
    """Print per-question results as a rich table."""
    table = Table(
        title="Phase 2 Baseline — Per-question results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("ID",        width=5)
    table.add_column("Question",  width=38)
    table.add_column("Intent",    width=9)
    table.add_column("Best ↑",    width=7)
    table.add_column("CRAG?",     width=7)
    table.add_column("Latency",   width=9)
    table.add_column("Kw%",       width=6)
    table.add_column("Fallback?", width=10)

    for r in report.results:
        if r.error:
            table.add_row(
                r.question_id, r.question[:37], "ERROR",
                "—", "—", f"{r.latency_ms:.0f}ms", "—",
                f"[red]ERROR[/red]"
            )
            continue

        # Intent: green if correct, red if wrong
        intent_color = "green" if r.actual_intent == r.expected_intent else "red"
        intent_str   = f"[{intent_color}]{r.actual_intent}[/{intent_color}]"

        # Score colour
        if r.best_score >= 0.80:
            score_str = f"[green]{r.best_score:.3f}[/green]"
        elif r.best_score >= 0.70:
            score_str = f"[yellow]{r.best_score:.3f}[/yellow]"
        else:
            score_str = f"[red]{r.best_score:.3f}[/red]"

        # CRAG indicator
        crag_str = "[red]YES →🌐[/red]" if r.below_crag else "[green]no[/green]"

        # Keyword hit
        kw_pct = (r.keyword_hits / r.keyword_total * 100) if r.keyword_total else 0
        kw_color = "green" if kw_pct >= 70 else "yellow" if kw_pct >= 40 else "red"
        kw_str  = f"[{kw_color}]{kw_pct:.0f}%[/{kw_color}]"

        # Fallback
        fallback_str = "[red]YES[/red]" if r.has_fallback else "[green]no[/green]"

        table.add_row(
            r.question_id,
            r.question[:37] + ("…" if len(r.question) > 37 else ""),
            intent_str,
            score_str,
            crag_str,
            f"{r.latency_ms:.0f}ms",
            kw_str,
            fallback_str,
        )

    console.print(table)


def print_summary(report) -> None:
    """Print the aggregated benchmark summary."""
    console.print(Rule("[bold]Phase 2 baseline — summary[/bold]"))

    # Retrieval quality
    retrieval = Table(title="Retrieval quality", show_header=False, box=None, padding=(0,2))
    retrieval.add_column("Metric",  style="cyan",   width=32)
    retrieval.add_column("Value",                   width=12)
    retrieval.add_column("Target (Phase 3)",        width=22)

    def score_color(v, good, ok):
        return "green" if v >= good else "yellow" if v >= ok else "red"

    avg_best = report.avg_best_score
    retrieval.add_row(
        "Avg best cosine score",
        f"[{score_color(avg_best, 0.80, 0.70)}]{avg_best:.4f}[/{score_color(avg_best, 0.80, 0.70)}]",
        "> 0.80 (Phase 3 hybrid)",
    )
    retrieval.add_row(
        "Avg avg cosine score",
        f"{report.avg_avg_score:.4f}",
        "> 0.70",
    )
    crag_pct = report.below_crag_pct
    retrieval.add_row(
        "Questions below CRAG (< 0.70)",
        f"[{'red' if crag_pct > 30 else 'yellow'}]{report.below_crag_count}/{report.successful} ({crag_pct:.0f}%)[/{'red' if crag_pct > 30 else 'yellow'}]",
        "< 20% (Phase 4 CRAG)",
    )
    console.print(Panel(retrieval, border_style="blue"))

    # Generation quality
    generation = Table(title="Generation quality", show_header=False, box=None, padding=(0,2))
    generation.add_column("Metric", style="cyan",  width=32)
    generation.add_column("Value",                 width=12)
    generation.add_column("Target",                width=22)

    fb_rate = report.fallback_rate
    generation.add_row(
        "Fallback rate (\"I don't know\")",
        f"[{'red' if fb_rate > 30 else 'yellow' if fb_rate > 15 else 'green'}]{report.fallback_count}/{report.successful} ({fb_rate:.0f}%)[/{'red' if fb_rate > 30 else 'yellow' if fb_rate > 15 else 'green'}]",
        "< 10% (Phases 3+4)",
    )
    generation.add_row(
        "Avg keyword hit rate",
        f"{report.avg_keyword_hit_pct:.0f}%",
        "> 70% (Phase 3)",
    )
    generation.add_row(
        "Intent routing accuracy",
        f"{report.intent_correct}/{report.successful} ({report.intent_accuracy:.0f}%)",
        "> 95% (Phase 4 few-shot)",
    )
    console.print(Panel(generation, border_style="green"))

    # System metrics
    system = Table(title="System metrics", show_header=False, box=None, padding=(0,2))
    system.add_column("Metric", style="cyan", width=32)
    system.add_column("Value",               width=12)
    system.add_column("Target",              width=22)

    system.add_row("Avg end-to-end latency",  f"{report.avg_latency_ms:.0f}ms",   "< 100ms (Phase 6 cache)")
    system.add_row("Avg cost per query",      f"${report.avg_cost_usd:.4f}",       "~$0 cached (Phase 6)")
    system.add_row("Total benchmark cost",    f"${report.total_cost_usd:.4f}",     "")
    console.print(Panel(system, border_style="yellow"))


def print_failure_analysis(report) -> None:
    """Identify and explain specific failure modes."""
    console.print(Rule("[bold]Failure analysis — what Phase 3 must fix[/bold]"))

    failures = [r for r in report.results if r.error is None and (r.below_crag or r.has_fallback)]

    if not failures:
        console.print("[green]No failures detected — retrieval quality is excellent.[/green]")
        return

    for r in failures:
        reasons = []
        if r.below_crag:
            reasons.append(f"Retrieval too weak (best={r.best_score:.3f} < 0.70) → Phase 3 hybrid search will fix")
        if r.has_fallback:
            reasons.append("LLM refused to answer (insufficient context) → consequence of weak retrieval")
        if r.actual_intent != r.expected_intent:
            reasons.append(f"Intent misclassified: expected={r.expected_intent} got={r.actual_intent}")

        console.print(Panel(
            f"[bold]{r.question_id}:[/bold] {r.question}\n"
            + "\n".join(f"  • {reason}" for reason in reasons),
            border_style="red",
            padding=(0, 1),
        ))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 baseline benchmark")
    parser.add_argument("--quick", action="store_true", help="Run 4-question quick suite")
    parser.add_argument("--save",  action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    from src.config import settings
    try:
        settings.validate()
    except ValueError as e:
        console.print(f"[red]Config: {e}[/red]")
        sys.exit(1)

    from src.module2_system_arch.graph import build_graph
    from src.phase2_baseline.benchmark import run_benchmark, TEST_SUITE

    console.print(Panel(
        "[bold]Phase 2 — Baseline RAG Benchmark[/bold]\n"
        "This establishes the reference metrics before Phase 3 improvements.\n"
        "[dim]Run again after Phase 3 to measure improvement.[/dim]",
        border_style="cyan",
    ))

    questions = QUICK_SUITE if args.quick else TEST_SUITE
    label     = "quick (4 questions)" if args.quick else f"full ({len(questions)} questions)"
    console.print(f"\n[dim]Suite: {label} | Building graph...[/dim]")

    graph  = build_graph(use_memory_checkpointer=True)
    console.print("[green]✓ Graph ready. Running benchmark...[/green]\n")

    report = await run_benchmark(graph, questions)

    print_results_table(report)
    console.print()
    print_summary(report)
    console.print()
    print_failure_analysis(report)

    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path      = f"phase2_baseline_{timestamp}.json"
        data      = {
            "timestamp": timestamp,
            "suite":     label,
            "summary": {
                "avg_best_score":   report.avg_best_score,
                "below_crag_pct":   report.below_crag_pct,
                "fallback_rate":    report.fallback_rate,
                "intent_accuracy":  report.intent_accuracy,
                "avg_latency_ms":   report.avg_latency_ms,
                "avg_cost_usd":     report.avg_cost_usd,
            },
            "results": [
                {
                    "id":          r.question_id,
                    "question":    r.question,
                    "intent":      r.actual_intent,
                    "best_score":  r.best_score,
                    "below_crag":  r.below_crag,
                    "has_fallback": r.has_fallback,
                    "latency_ms":  r.latency_ms,
                    "cost_usd":    r.cost_usd,
                    "keyword_hits": r.keyword_hits,
                    "keyword_total": r.keyword_total,
                }
                for r in report.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"\n[green]Results saved to {path}[/green]")

    # Final verdict
    console.print(Rule())
    console.print(
        f"\n[bold]Phase 2 baseline established.[/bold]\n"
        f"Avg retrieval score: [cyan]{report.avg_best_score:.4f}[/cyan] | "
        f"Fallback rate: [cyan]{report.fallback_rate:.0f}%[/cyan] | "
        f"CRAG fires on: [cyan]{report.below_crag_pct:.0f}%[/cyan] of queries | "
        f"Avg latency: [cyan]{report.avg_latency_ms:.0f}ms[/cyan]\n\n"
        "[dim]These numbers are your Phase 3 targets. "
        "Hybrid search + reranking should raise avg score above 0.80 "
        "and cut fallback rate below 10%.[/dim]"
    )


if __name__ == "__main__":
    asyncio.run(main())

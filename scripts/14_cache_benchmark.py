#!/usr/bin/env python3
# scripts/14_cache_benchmark.py
#
# Demonstrates the Phase 6 Redis caching latency improvement.
#
# WHAT THIS SHOWS:
# 1. Cold run: full pipeline (~5,000ms, all API calls)
# 2. Warm run: cache hits (~3ms, Redis only)
# 3. Per-tier stats: which tiers are hitting and what was saved
# 4. Cache hit rate over multiple questions
#
# Usage:
#   python scripts/14_cache_benchmark.py              # full demo
#   python scripts/14_cache_benchmark.py --flush      # flush cache, rerun
#   python scripts/14_cache_benchmark.py --stats      # show cache stats only

import sys, os, asyncio, argparse, time, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich.rule    import Rule

console = Console()

TEST_QUESTIONS = [
    "What does OOMKilled mean and how do I fix it?",
    "How do I debug a pod stuck in CrashLoopBackOff?",
    "What is the difference between resource requests and limits?",
    "How does Horizontal Pod Autoscaler work?",
    "What does OOMKilled mean and how do I fix it?",   # ← REPEAT of Q1 — should hit cache
    "How do I debug a pod stuck in CrashLoopBackOff?", # ← REPEAT of Q2 — should hit cache
]


async def run_question(graph, question: str, session_id: str) -> dict:
    from src.module2_system_arch.graph import run_graph
    start      = time.perf_counter()
    final_state = await run_graph(question, session_id, graph)
    elapsed_ms  = (time.perf_counter() - start) * 1000
    return {"state": final_state, "latency_ms": elapsed_ms}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flush", action="store_true", help="Flush cache before running")
    parser.add_argument("--stats", action="store_true", help="Show cache stats only")
    args = parser.parse_args()

    from src.config import settings
    settings.validate()

    # Check Redis
    from src.phase6_cache.client import test_redis_connection
    redis_ok = test_redis_connection()

    if not redis_ok:
        console.print(Panel(
            "[red]Redis not reachable.[/red]\n\n"
            "Start it:\n"
            "  docker compose up -d redis\n\n"
            "The benchmark still runs but caching is disabled.\n"
            "All queries will take ~5s (cold path).",
            border_style="red",
            title="Redis unavailable",
        ))
    else:
        console.print("[green]✓ Redis connected[/green]")

    from src.phase6_cache.manager import get_cache_manager
    cache = get_cache_manager()

    if args.stats:
        console.print(Panel(str(cache.stats), title="Cache stats", border_style="cyan"))
        return

    if args.flush:
        cache.flush_all()
        console.print("[yellow]Cache flushed — all queries will be cold[/yellow]\n")

    from src.module2_system_arch.graph import build_graph
    console.print("[dim]Compiling graph...[/dim]")
    graph = build_graph(use_memory_checkpointer=True)
    console.print("[green]✓ Graph ready[/green]\n")

    console.print(Panel(
        "[bold]Phase 6 — Cache Benchmark[/bold]\n"
        f"Redis: {'[green]connected[/green]' if redis_ok else '[red]unavailable[/red]'}\n"
        "Questions 5+6 repeat questions 1+2 — watch the latency drop",
        border_style="cyan",
    ))

    results = []
    table = Table(
        title="Query results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#",           width=4)
    table.add_column("Question",    width=42)
    table.add_column("Latency",     width=10)
    table.add_column("Cache",       width=8)
    table.add_column("Intent",      width=8)

    for i, question in enumerate(TEST_QUESTIONS, 1):
        console.print(f"[dim]Running Q{i}: {question[:50]}...[/dim]")

        session_id = str(uuid.uuid4())[:8]
        result = await run_question(graph, question, session_id)

        latency_ms  = result["latency_ms"]
        final_state = result["state"]
        cache_hit   = final_state.get("cache_hit", False)
        intent      = final_state.get("intent", "?")

        # Color latency: green<100ms, yellow<1000ms, red>1000ms
        if latency_ms < 100:
            lat_str = f"[green]{latency_ms:.0f}ms[/green]"
        elif latency_ms < 1000:
            lat_str = f"[yellow]{latency_ms:.0f}ms[/yellow]"
        else:
            lat_str = f"[red]{latency_ms:.0f}ms[/red]"

        cache_str = "[green]HIT[/green]" if cache_hit else "[dim]miss[/dim]"

        table.add_row(
            str(i),
            question[:41] + ("…" if len(question) > 41 else ""),
            lat_str,
            cache_str,
            intent,
        )
        results.append({
            "question":   question,
            "latency_ms": latency_ms,
            "cache_hit":  cache_hit,
        })

    console.print(table)

    # Summary
    hits   = [r for r in results if r["cache_hit"]]
    misses = [r for r in results if not r["cache_hit"]]

    if hits and misses:
        avg_cold = sum(r["latency_ms"] for r in misses) / len(misses)
        avg_warm = sum(r["latency_ms"] for r in hits)   / len(hits)
        speedup  = avg_cold / avg_warm if avg_warm > 0 else 0

        console.print(Panel(
            f"[bold]Cache performance summary[/bold]\n\n"
            f"Cold queries (miss):  {len(misses)} queries | avg [red]{avg_cold:.0f}ms[/red]\n"
            f"Warm queries (hit):   {len(hits)} queries  | avg [green]{avg_warm:.0f}ms[/green]\n"
            f"Speedup:              [bold cyan]{speedup:.0f}x faster[/bold cyan] on cache hits\n\n"
            f"[dim]Cache stats: {cache.stats}[/dim]",
            border_style="green",
        ))
    elif not redis_ok:
        cold_avg = sum(r["latency_ms"] for r in results) / len(results)
        console.print(Panel(
            f"[yellow]All {len(results)} queries ran without cache[/yellow]\n"
            f"Avg latency: {cold_avg:.0f}ms\n\n"
            "Start Redis and re-run to see the speedup:\n"
            "  docker compose up -d redis\n"
            "  python scripts/14_cache_benchmark.py",
            border_style="yellow",
        ))
    else:
        console.print("[dim]Run the benchmark twice to see cache hits on repeated questions.[/dim]")


if __name__ == "__main__":
    asyncio.run(main())

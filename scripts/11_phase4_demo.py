#!/usr/bin/env python3
# scripts/11_phase4_demo.py
#
# Phase 4 demo: CRAG routing + Self-RAG reflection cycle.
#
# WHAT TO OBSERVE:
# 1. "Specific K8s question" (e.g. OOMKilled) → CRAG grade: CORRECT
#    → Tavily NOT called → generate → Self-RAG scores high → END
#
# 2. "Vague broad question" (e.g. "which pods are failing") → CRAG grade: INCORRECT
#    → Tavily called (if API key set) → enriched context → generate → Self-RAG
#
# 3. "Self-RAG cycle": if the first answer scores below 0.80,
#    the graph loops back to retrieve_node for a second attempt
#    Look for "Self-RAG: REGENERATE" in logs
#
# 4. Graph topology: prints the Mermaid diagram showing the cycle
#
# Usage:
#   python scripts/11_phase4_demo.py                     # run demo questions
#   python scripts/11_phase4_demo.py --topology          # print graph topology
#   python scripts/11_phase4_demo.py --q "your question" # custom question

import sys, os, asyncio, argparse, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich.rule    import Rule

console = Console()

DEMO_QUESTIONS = [
    {
        "label":    "High-score RAG (CRAG: correct, no Tavily)",
        "question": "What does OOMKilled exit code 137 mean and how do I fix it?",
    },
    {
        "label":    "Vague hybrid (CRAG: poor, Tavily fallback)",
        "question": "Which pods are failing and what should I do about it?",
    },
    {
        "label":    "SQL intent (routed to placeholder)",
        "question": "How many pods are currently in CrashLoopBackOff?",
    },
]


async def run_question(graph, question: str, label: str = "") -> dict:
    """Run one question and display full Phase 4 trace."""
    from src.module2_system_arch.graph import run_graph

    if label:
        console.print(Rule(f"[bold]{label}[/bold]"))
    console.print(f"[bold]Q:[/bold] {question}\n")

    start       = time.perf_counter()
    final_state = await run_graph(
        question   = question,
        session_id = f"p4demo-{hash(question) % 10000}",
        graph      = graph,
    )
    elapsed = (time.perf_counter() - start) * 1000

    answer  = final_state.get("answer", "")
    intent  = final_state.get("intent", "?")
    grade   = final_state.get("retrieval_grade", "?")
    scores  = final_state.get("scores", [])
    sr_score = final_state.get("self_rag_score", 1.0)
    iterations = final_state.get("iteration", 1)
    tavily_used = bool(final_state.get("tavily_results"))

    # Phase 4 trace summary
    trace = Table(show_header=False, box=None, padding=(0, 2))
    trace.add_column("Stage",  style="cyan",   width=24)
    trace.add_column("Result", style="white")

    trace.add_row("Intent classified",   f"[cyan]{intent}[/cyan]")
    trace.add_row(
        "CRAG grade",
        f"[{'green' if grade=='correct' else 'red'}]{grade}[/{'green' if grade=='correct' else 'red'}] "
        f"(best score: {max(scores):.3f})" if scores else grade
    )
    trace.add_row("Tavily called",       "[yellow]YES[/yellow]" if tavily_used else "[dim]no[/dim]")
    trace.add_row("Self-RAG score",      f"{sr_score:.3f} ({'✓ accepted' if sr_score >= 0.80 else '↺ would regenerate'})")
    trace.add_row("Iterations",          str(iterations))
    trace.add_row("Total latency",       f"{elapsed:.0f}ms")

    console.print(Panel(trace, title="[blue]Phase 4 trace[/blue]", border_style="blue"))
    console.print(Panel(answer, title="[green]Answer[/green]", border_style="green"))
    console.print()

    return final_state


def show_topology(graph) -> None:
    """Print the graph topology as Mermaid diagram."""
    console.print(Rule("[bold]Phase 4 graph topology[/bold]"))
    try:
        mermaid = graph.get_graph().draw_mermaid()
        console.print(Panel(
            mermaid,
            title="[blue]Mermaid diagram (paste at mermaid.live)[/blue]",
            border_style="blue",
        ))
    except Exception as e:
        console.print(f"[yellow]Mermaid: {e}[/yellow]")

    console.print(
        "\n[bold]Key Phase 4 additions:[/bold]\n"
        "  • [cyan]crag_grader[/cyan]: grades retrieval, decides Tavily fallback\n"
        "  • [cyan]tavily_search[/cyan]: fetches web content when local retrieval fails\n"
        "  • [cyan]self_rag_reflect[/cyan]: scores answer quality\n"
        "  • [red bold]CYCLE[/red bold]: self_rag_reflect → retrieve (when score < 0.80)\n"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", action="store_true")
    parser.add_argument("--q",        type=str, help="Single question")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("src.phase4_agentic").setLevel(logging.DEBUG)
        logging.getLogger("src.phase3_retrieval").setLevel(logging.DEBUG)

    from src.config import settings
    settings.validate()

    from src.module2_system_arch.graph import build_graph
    console.print("[dim]Compiling Phase 4 graph...[/dim]")
    graph = build_graph(use_memory_checkpointer=True)
    console.print("[green]✓ Graph compiled with CRAG + Self-RAG cycle[/green]\n")

    if args.topology:
        show_topology(graph)
        return

    if args.q:
        await run_question(graph, args.q, label="Custom question")
        return

    # Run all demo questions
    console.print(Panel(
        "[bold]Phase 4 — Agentic Retrieval Demo[/bold]\n"
        "CRAG: detects poor retrieval and fetches from web\n"
        "Self-RAG: scores answer quality and can cycle back to retrieve",
        border_style="cyan",
    ))

    for demo in DEMO_QUESTIONS:
        await run_question(graph, demo["question"], demo["label"])

    show_topology(graph)

    # Check Tavily config
    if not os.getenv("TAVILY_API_KEY"):
        console.print(Panel(
            "[yellow]TAVILY_API_KEY not set.[/yellow]\n"
            "CRAG detected poor retrieval but couldn't call Tavily.\n"
            "To enable web fallback:\n"
            "  1. Get a free key at https://tavily.com\n"
            "  2. Add TAVILY_API_KEY=your-key to your .env file",
            border_style="yellow",
            title="Tavily not configured",
        ))


if __name__ == "__main__":
    asyncio.run(main())

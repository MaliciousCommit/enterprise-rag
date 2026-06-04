#!/usr/bin/env python3
# scripts/04_langgraph_query.py
#
# Interactive query interface using the Module 2 LangGraph state machine.
#
# This replaces scripts/02_query.py's flat NaiveRAG pipeline with a
# proper state machine that:
#   1. Routes questions by intent (rag/sql/hybrid)
#   2. Tracks state across nodes
#   3. Checkpoints conversation for memory
#   4. Shows the execution trace (which nodes ran, in what order)
#
# Usage:
#   python scripts/04_langgraph_query.py
#   python scripts/04_langgraph_query.py --question "Why is my pod OOMKilled?"
#
# WHAT TO OBSERVE:
#   - Intent classification: which pipeline was selected and why
#   - Node execution trace: see each node's contribution to state
#   - Latency breakdown: intent (~300ms) + retrieve (~150ms) + generate (~2s)
#   - SQL questions: see the placeholder response explaining Phase 5

import sys
import os
import asyncio
import argparse
import uuid
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()

# Build the graph ONCE at startup (not per-query)
# This is the same pattern FastAPI will use in Phase 3
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        from src.module2_system_arch.graph import build_graph
        _graph = build_graph(use_memory_checkpointer=True)
    return _graph


def print_result(final_state: dict, elapsed_ms: float) -> None:
    """Pretty-print the final RAGState after graph execution."""

    intent = final_state.get("intent", "unknown")
    answer = final_state.get("answer", "")
    context = final_state.get("context", [])
    scores  = final_state.get("scores", [])
    sources = final_state.get("sources", [])
    iteration = final_state.get("iteration", 0)
    prompt_tokens     = final_state.get("prompt_tokens", 0)
    completion_tokens = final_state.get("completion_tokens", 0)

    # ── Intent badge ──────────────────────────────────────────────────────────
    intent_color = {"rag": "blue", "sql": "green", "hybrid": "magenta"}.get(intent, "white")
    console.print(f"\n  Intent classified as: [{intent_color} bold]{intent.upper()}[/{intent_color} bold]")

    # ── Answer panel ──────────────────────────────────────────────────────────
    console.print(Panel(
        Markdown(answer),
        title=f"[bold green]Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # ── Retrieval table (only for rag/hybrid) ─────────────────────────────────
    if context and intent in ("rag", "hybrid"):
        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("Doc",    style="blue",   width=5)
        table.add_column("Score",  style="cyan",   width=8)
        table.add_column("Source", style="yellow", width=28)
        table.add_column("Preview")

        for i, (chunk, score) in enumerate(zip(context, scores or [0]*len(context)), 1):
            score_color = "green" if score > 0.80 else "yellow" if score > 0.65 else "red"
            source = sources[i-1] if i-1 < len(sources) else "unknown"
            table.add_row(
                f"[{i}]",
                f"[{score_color}]{score:.3f}[/{score_color}]",
                source.split("/")[-1][:27],
                chunk[:80].replace("\n", " ") + "...",
            )

        console.print(Panel(
            table,
            title="[bold blue]Retrieved context[/bold blue]",
            border_style="blue",
        ))

    # ── Stats footer ──────────────────────────────────────────────────────────
    console.print(
        f"  [dim]Latency: {elapsed_ms:.0f}ms | "
        f"Tokens: {prompt_tokens}↑ {completion_tokens}↓ | "
        f"Iteration: {iteration} | "
        f"Intent: {intent}[/dim]\n"
    )


async def run_query(question: str, session_id: str) -> dict:
    """Run a single query through the LangGraph pipeline."""
    from src.module2_system_arch.graph import run_graph

    graph = get_graph()
    start = time.perf_counter()
    final_state = await run_graph(question=question, session_id=session_id, graph=graph)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print_result(final_state, elapsed_ms)
    return final_state


async def run_interactive(session_id: str) -> None:
    """Interactive REPL loop."""
    console.print(Panel(
        "[bold]Enterprise RAG — Module 2 | LangGraph State Machine[/bold]\n"
        "Questions are classified by intent before retrieval.\n"
        "Try asking a SQL question to see the intent router in action.\n"
        "[dim]Commands: 'quit' to exit | 'new' for a new session[/dim]",
        border_style="cyan",
    ))

    examples = [
        ("rag",    "Why is my pod showing OOMKilled?"),
        ("rag",    "How do I roll back a bad deployment?"),
        ("sql",    "How many pods are in CrashLoopBackOff right now?"),
        ("sql",    "Which nodes have memory pressure today?"),
        ("hybrid", "Which pods are failing and what should I do to fix them?"),
    ]

    console.print("\n[bold]Example questions by intent:[/bold]")
    for intent, q in examples:
        color = {"rag": "blue", "sql": "green", "hybrid": "magenta"}[intent]
        console.print(f"  [{color}][{intent.upper()}][/{color}]  {q}")
    console.print()

    while True:
        try:
            console.print("[bold cyan]Question:[/bold cyan] ", end="")
            question = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "q", "exit"):
            console.print("[dim]Goodbye.[/dim]")
            break
        if question.lower() == "new":
            session_id = str(uuid.uuid4())[:8]
            console.print(f"[dim]New session: {session_id}[/dim]")
            continue

        try:
            await run_query(question, session_id)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Module 2 LangGraph Query Interface")
    parser.add_argument("--question", "-q", type=str, help="Single query mode")
    parser.add_argument("--session",  "-s", type=str, default=str(uuid.uuid4())[:8],
                        help="Session ID for conversation memory")
    args = parser.parse_args()

    # Validate setup
    from src.config import settings
    try:
        settings.validate()
    except ValueError as e:
        console.print(f"[red]Config error: {e}[/red]")
        sys.exit(1)

    console.print(f"\n[dim]Session ID: {args.session}[/dim]")
    console.print("[dim]Building LangGraph state machine...[/dim]")
    get_graph()  # compile once
    console.print("[green]✓ Graph compiled[/green]\n")

    if args.question:
        await run_query(args.question, args.session)
    else:
        await run_interactive(args.session)


if __name__ == "__main__":
    asyncio.run(main())

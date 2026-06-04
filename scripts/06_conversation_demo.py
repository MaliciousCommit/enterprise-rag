#!/usr/bin/env python3
# scripts/06_conversation_demo.py
#
# Demonstrates multi-turn conversation memory in LangGraph.
#
# WHAT THIS SHOWS:
# 1. Same session_id across turns → LangGraph loads previous checkpoint
# 2. Conversation history carried into the LLM → meaningful follow-ups
# 3. Different session_id → fresh conversation, no memory of previous
# 4. Graph topology inspection — see the nodes and edges
#
# WHY CONVERSATION MEMORY MATTERS:
# Without memory, every question is independent:
#   Q: "What is OOMKilled?"    → A: "OOMKilled means..."
#   Q: "How do I fix it?"      → A: "Fix what? I have no context."
#
# With memory, the LLM has the full conversation history:
#   Q: "What is OOMKilled?"    → A: "OOMKilled means..."
#   Q: "How do I fix it?"      → A: "To fix OOMKilled (which I just
#                                     described), increase the memory limit..."
#
# IMPLEMENTATION NOTE:
# LangGraph checkpoints the RAGState to MemorySaver after each node.
# On the next turn with the same session_id, LangGraph loads this state.
# The conversation history (question/answer pairs) is stored separately
# in _session_histories (see nodes.py) and injected into the LLM call.
#
# Usage:
#   python scripts/06_conversation_demo.py            # auto demo
#   python scripts/06_conversation_demo.py --inspect  # show graph topology

import sys
import os
import asyncio
import argparse
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import print as rprint

console = Console()


async def run_turn(graph, question: str, session_id: str, turn_num: int) -> dict:
    """Run one conversation turn and return the final state."""
    from src.module2_system_arch.graph import run_graph

    console.print(f"\n[bold cyan]Turn {turn_num}[/bold cyan] | session=[dim]{session_id}[/dim]")
    console.print(f"[bold]Q:[/bold] {question}")

    state = await run_graph(question=question, session_id=session_id, graph=graph)

    console.print(Panel(
        state.get("answer", ""),
        title=f"[green]A[/green] (intent=[cyan]{state.get('intent', '?')}[/cyan])",
        border_style="green",
        padding=(0, 1),
    ))

    scores = state.get("scores", [])
    if scores:
        console.print(
            f"[dim]Retrieval scores: {[f'{s:.3f}' for s in scores]} | "
            f"Tokens: {state.get('prompt_tokens',0)}↑ {state.get('completion_tokens',0)}↓[/dim]"
        )

    return state


async def demo_conversation_memory(graph) -> None:
    """
    Demo 1: Show that conversation history persists across turns.
    Same session_id → LLM remembers previous exchange.
    """
    console.print(Rule("[bold]Demo 1: Multi-turn conversation memory[/bold]"))
    console.print("[dim]Same session_id across 3 turns — the LLM remembers previous answers[/dim]\n")

    session_id = f"demo-{uuid.uuid4().hex[:8]}"

    # Turn 1: Ask a specific question
    await run_turn(graph,
        question="What does OOMKilled mean and what causes it?",
        session_id=session_id,
        turn_num=1,
    )

    # Turn 2: Follow-up referencing Turn 1
    await run_turn(graph,
        question="How do I fix the problem you just described?",
        session_id=session_id,
        turn_num=2,
    )

    # Turn 3: Deeper follow-up
    await run_turn(graph,
        question="What monitoring should I set up to prevent this from happening again?",
        session_id=session_id,
        turn_num=3,
    )


async def demo_session_isolation(graph) -> None:
    """
    Demo 2: Show that different session_ids are completely isolated.
    Session A's history doesn't leak into Session B.
    """
    console.print(Rule("[bold]Demo 2: Session isolation[/bold]"))
    console.print("[dim]Two sessions asking follow-ups — each is independent[/dim]\n")

    session_a = f"session-a-{uuid.uuid4().hex[:6]}"
    session_b = f"session-b-{uuid.uuid4().hex[:6]}"

    # Session A: about OOMKilled
    console.print(f"[yellow]Session A:[/yellow] {session_a}")
    await run_turn(graph, "What is OOMKilled?", session_a, turn_num=1)
    await run_turn(graph, "How do I fix it?",   session_a, turn_num=2)

    console.print()

    # Session B: about CrashLoopBackOff — completely separate
    console.print(f"[blue]Session B:[/blue] {session_b}")
    await run_turn(graph, "What is CrashLoopBackOff?", session_b, turn_num=1)
    await run_turn(graph, "How do I debug it?",         session_b, turn_num=2)

    console.print("\n[dim]Notice: Session B's follow-up answers about CrashLoopBackOff, "
                  "not about OOMKilled — sessions are fully isolated.[/dim]")


def inspect_graph_topology(graph) -> None:
    """
    Show the LangGraph graph topology: nodes, edges, and structure.
    LangGraph can generate Mermaid diagram syntax for visualisation.
    """
    console.print(Rule("[bold]Graph topology[/bold]"))

    try:
        # get_graph() returns the raw NetworkX graph
        # draw_mermaid() generates Mermaid diagram syntax
        mermaid = graph.get_graph().draw_mermaid()
        console.print("\n[bold]Mermaid diagram (paste at mermaid.live):[/bold]")
        console.print(Panel(mermaid, border_style="blue"))
    except Exception as e:
        console.print(f"[yellow]Mermaid generation: {e}[/yellow]")

    # Show nodes
    try:
        nodes = list(graph.get_graph().nodes)
        table = Table(title="Graph Nodes", show_header=True, header_style="bold")
        table.add_column("Node", style="cyan")
        table.add_column("Type")
        for node in nodes:
            table.add_row(str(node), "START/END" if str(node).startswith("__") else "Function node")
        console.print(table)
    except Exception as e:
        console.print(f"[yellow]Node inspection: {e}[/yellow]")

    # Show edges
    try:
        edges = list(graph.get_graph().edges)
        table = Table(title="Graph Edges", show_header=True, header_style="bold")
        table.add_column("From",       style="cyan")
        table.add_column("To",         style="green")
        for src, dst in edges:
            table.add_row(str(src), str(dst))
        console.print(table)
    except Exception as e:
        console.print(f"[yellow]Edge inspection: {e}[/yellow]")


async def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph conversation demo")
    parser.add_argument("--inspect", action="store_true", help="Show graph topology only")
    parser.add_argument("--isolation", action="store_true", help="Run session isolation demo")
    args = parser.parse_args()

    # Validate config
    from src.config import settings
    try:
        settings.validate()
    except ValueError as e:
        console.print(f"[red]Config error: {e}[/red]")
        sys.exit(1)

    # Build graph
    console.print("[dim]Compiling LangGraph state machine...[/dim]")
    from src.module2_system_arch.graph import build_graph
    graph = build_graph(use_memory_checkpointer=True)
    console.print("[green]✓ Graph ready[/green]\n")

    if args.inspect:
        inspect_graph_topology(graph)
        return

    # Run demos
    if args.isolation:
        await demo_session_isolation(graph)
    else:
        await demo_conversation_memory(graph)
        console.print()
        inspect_graph_topology(graph)

    console.print("\n[bold green]Demo complete.[/bold green]")
    console.print("[dim]Key observations:[/dim]")
    console.print("  • Same session_id → LLM receives conversation history → coherent follow-ups")
    console.print("  • Different session_id → fresh start → no cross-session leakage")
    console.print("  • MemorySaver stores checkpoints in process memory (lost on restart)")
    console.print("  • Phase 5: PostgresSaver will persist across restarts → true persistence")


if __name__ == "__main__":
    asyncio.run(main())

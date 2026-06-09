#!/usr/bin/env python3
# scripts/13_phase5_demo.py
#
# Interactive Phase 5 demo: Text2SQL with Human-in-the-Loop approval.
#
# WHAT THIS DEMONSTRATES:
# 1. SQL question → intent classified as "sql"
# 2. GPT-4o generates a SELECT query using the schema
# 3. Validator approves (SELECT-only, has LIMIT)
# 4. Graph hits interrupt() — PAUSES
# 5. YOU review the SQL in the terminal
# 6. YOU type "yes" or "no"
# 7. Graph resumes with your decision
# 8. If approved: SQL executes → results → LLM answer
# 9. If rejected: LLM explains it cannot answer without data
#
# This is the same flow as the production API:
#   First call:   POST /api/v1/query       → {"pending_approval": true, "pending_sql": "..."}
#   Second call:  POST /api/v1/sql/approve → full answer
#
# Usage:
#   python scripts/13_phase5_demo.py
#   python scripts/13_phase5_demo.py --q "How many pods are in CrashLoopBackOff?"

import sys, os, asyncio, argparse, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel   import Panel
from rich.syntax  import Syntax
from rich.rule    import Rule

console = Console()

DEMO_QUESTIONS = [
    "How many pods are in CrashLoopBackOff right now?",
    "Which pods are failing in the prod cluster?",
    "What P1 incidents are currently open?",
    "What was deployed to prod in the last 24 hours?",
    "Which nodes have memory pressure?",
    "Show me all critical alerts that are currently firing",
]


async def run_hitl_demo(graph, question: str, session_id: str) -> None:
    """Run one full HITL cycle: generate SQL → interrupt → human review → resume."""
    from src.module2_system_arch.graph import run_graph, resume_graph

    console.print(f"\n[bold]Question:[/bold] {question}")
    console.print("[dim]Running pipeline...[/dim]\n")

    # ── Step 1: First invocation — generates SQL, hits interrupt ──────────────
    state = await run_graph(question, session_id, graph)

    # ── Step 2: Handle the interrupt ──────────────────────────────────────────
    if state.get("pending_approval"):
        sql = state.get("sql_query", "")

        console.print(Panel(
            "[bold yellow]⏸ Graph paused — SQL requires your approval[/bold yellow]\n\n"
            "The system generated the following SQL query.\n"
            "Review it carefully before approving execution against the live database.",
            border_style="yellow",
        ))

        # Display the SQL with syntax highlighting
        console.print(Syntax(sql, "sql", theme="monokai", line_numbers=True))

        console.print(
            "\n[dim]This query will run against the PostgreSQL operational database.[/dim]"
        )

        # Human decision
        while True:
            console.print("\n[bold]Approve this SQL? [yellow]yes[/yellow] / [red]no[/red]:[/bold] ", end="")
            try:
                decision = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                decision = "no"
                console.print("no")

            if decision in ("yes", "y"):
                approved = True
                break
            elif decision in ("no", "n"):
                approved = False
                break
            else:
                console.print("[dim]Please type yes or no[/dim]")

        action = "[green]APPROVED[/green]" if approved else "[red]REJECTED[/red]"
        console.print(f"\n  Decision: {action}")

        # ── Step 3: Resume the graph ──────────────────────────────────────────
        console.print("[dim]Resuming graph...[/dim]")
        state = await resume_graph(session_id, approved, graph)

    # ── Step 4: Display final answer ──────────────────────────────────────────
    answer = state.get("answer", "")
    intent = state.get("intent", "?")
    sql_approved = state.get("sql_approved")
    sql_result   = state.get("sql_result", "")

    # Show SQL results if available
    if sql_result and "Error" not in sql_result:
        console.print(Panel(
            sql_result,
            title="[blue]Live cluster data (from PostgreSQL)[/blue]",
            border_style="blue",
        ))

    console.print(Panel(
        answer,
        title=f"[green]Answer[/green] (intent=[cyan]{intent}[/cyan], sql_approved={sql_approved})",
        border_style="green",
    ))

    console.print(
        f"[dim]Self-RAG score: {state.get('self_rag_score', 1.0):.3f} | "
        f"Iterations: {state.get('iteration', 1)}[/dim]\n"
    )


async def run_rag_demo(graph, question: str, session_id: str) -> None:
    """Run a regular RAG question (no SQL) for comparison."""
    from src.module2_system_arch.graph import run_graph
    state = await run_graph(question, session_id, graph)
    answer = state.get("answer", "")
    console.print(Panel(
        answer,
        title=f"[green]Answer[/green] (intent=[cyan]{state.get('intent')}[/cyan])",
        border_style="green",
    ))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q",    type=str, help="Single question to run")
    parser.add_argument("--all",  action="store_true", help="Run all demo questions")
    args = parser.parse_args()

    from src.config import settings
    settings.validate()

    # Check PostgreSQL
    from src.phase5_text2sql.database import test_connection
    if not test_connection():
        console.print(Panel(
            "[red]PostgreSQL not reachable.[/red]\n"
            "Run: docker compose up -d postgres\n"
            "Then: python scripts/12_setup_postgres.py",
            border_style="red",
        ))
        sys.exit(1)

    console.print("[green]✓ PostgreSQL connected[/green]")

    from src.module2_system_arch.graph import build_graph
    console.print("[dim]Compiling Phase 5 graph...[/dim]")
    graph = build_graph(use_memory_checkpointer=True)
    console.print("[green]✓ Graph compiled with Text2SQL + HITL interrupt[/green]")

    console.print(Panel(
        "[bold]Phase 5 — Text2SQL with Human-in-the-Loop[/bold]\n"
        "SQL questions pause for your approval before executing\n"
        "[dim]Type 'yes' to approve, 'no' to reject[/dim]",
        border_style="cyan",
    ))

    if args.q:
        session_id = str(uuid.uuid4())[:8]
        await run_hitl_demo(graph, args.q, session_id)
        return

    if args.all:
        for q in DEMO_QUESTIONS:
            session_id = str(uuid.uuid4())[:8]
            console.print(Rule())
            await run_hitl_demo(graph, q, session_id)
        return

    # Interactive loop
    console.print("\n[bold]Example SQL questions:[/bold]")
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        console.print(f"  [dim]{i}.[/dim] {q}")
    console.print("\n[dim]Commands: type a question | 'quit' to exit | 1-6 for examples[/dim]\n")

    while True:
        console.print("[bold cyan]Question:[/bold cyan] ", end="")
        try:
            question = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "q", "exit"):
            console.print("[dim]Goodbye.[/dim]")
            break
        if question.isdigit() and 1 <= int(question) <= len(DEMO_QUESTIONS):
            question = DEMO_QUESTIONS[int(question) - 1]
            console.print(f"[dim]Using: {question}[/dim]")

        session_id = str(uuid.uuid4())[:8]
        try:
            await run_hitl_demo(graph, question, session_id)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(main())

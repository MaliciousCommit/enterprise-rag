#!/usr/bin/env python3
# scripts/12_setup_postgres.py
#
# One-time setup: create the 7-table K8s operational schema and load sample data.
#
# Usage:
#   python scripts/12_setup_postgres.py
#
# Requires PostgreSQL running. Start with:
#   docker compose up -d postgres
#
# Connection configured via .env:
#   POSTGRES_HOST=localhost
#   POSTGRES_PORT=5432
#   POSTGRES_DB=enterprise_rag
#   POSTGRES_USER=raguser
#   POSTGRES_PASSWORD=ragpass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich.rule    import Rule

console = Console()


def main() -> None:
    console.print(Panel(
        "[bold]Phase 5 — PostgreSQL Setup[/bold]\n"
        "Creates 7-table K8s operational schema + sample data",
        border_style="cyan",
    ))

    from src.phase5_text2sql.database import test_connection, setup_database, get_db_connection

    # Test connection
    console.print("[dim]Testing PostgreSQL connection...[/dim]")
    if not test_connection():
        console.print(Panel(
            "[red]Cannot connect to PostgreSQL.[/red]\n\n"
            "Start it with:\n"
            "  docker compose up -d postgres\n\n"
            "Or configure .env:\n"
            "  POSTGRES_HOST=localhost\n"
            "  POSTGRES_PORT=5432\n"
            "  POSTGRES_DB=enterprise_rag\n"
            "  POSTGRES_USER=raguser\n"
            "  POSTGRES_PASSWORD=ragpass",
            border_style="red",
        ))
        sys.exit(1)

    console.print("[green]✓ PostgreSQL connected[/green]")

    # Create schema + load sample data
    console.print("\n[dim]Creating schema and loading sample data...[/dim]")
    success = setup_database()

    if not success:
        console.print("[red]✗ Setup failed — check logs above[/red]")
        sys.exit(1)

    console.print("[green]✓ Schema created (7 tables)[/green]")
    console.print("[green]✓ Sample data loaded[/green]\n")

    # Verify row counts
    console.print(Rule("[bold]Row counts[/bold]"))
    tables = ["clusters", "pods", "nodes", "incidents", "deployments", "alerts", "audit_log"]

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Table",      width=16)
    table.add_column("Row count",  width=12, justify="right")
    table.add_column("Sample",     width=40)

    from src.phase5_text2sql.database import execute_select

    for t in tables:
        count_res  = execute_select(f"SELECT COUNT(*) as n FROM {t} LIMIT 1")
        sample_res = execute_select(f"SELECT * FROM {t} LIMIT 1")
        count = count_res["rows"][0][0] if count_res["rows"] else 0
        sample = str(sample_res["rows"][0][:2]) if sample_res["rows"] else "empty"
        color = "green" if int(count) > 0 else "yellow"
        table.add_row(t, f"[{color}]{count}[/{color}]", sample[:38])

    console.print(table)

    # Run a test query
    console.print(Rule("\n[bold]Test query: failing pods[/bold]"))
    res = execute_select("""
        SELECT p.pod_name, p.namespace, p.status, p.restart_count
        FROM pods p
        JOIN clusters c ON p.cluster_id = c.id
        WHERE c.environment = 'prod'
          AND p.status IN ('CrashLoopBackOff', 'OOMKilled', 'Error', 'Failed')
        ORDER BY p.restart_count DESC
        LIMIT 10
    """)

    from src.phase5_text2sql.database import format_results_as_text
    console.print(format_results_as_text(res))

    console.print(Panel(
        "[bold green]PostgreSQL setup complete![/bold green]\n\n"
        "Next step:\n"
        "  python scripts/13_phase5_demo.py\n\n"
        "[dim]This will ask a SQL question, pause for your approval,\n"
        "then execute and return the answer.[/dim]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# scripts/07_qdrant_inspect.py
#
# Inspect the current Qdrant collection and benchmark retrieval.
#
# WHAT THIS SHOWS:
# 1. Collection configuration (HNSW params, quantization, status)
# 2. Sample point inspection (what one vector's payload looks like)
# 3. Retrieval benchmark — measures real latency across K values
# 4. Similarity score distribution across your knowledge base
#
# Usage:
#   python scripts/07_qdrant_inspect.py           # full inspection
#   python scripts/07_qdrant_inspect.py --bench   # retrieval benchmark only
#   python scripts/07_qdrant_inspect.py --upgrade # recreate with HNSW tuning

import sys
import os
import asyncio
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule

console = Console()


def inspect_collection(client) -> None:
    """Show full collection configuration and statistics."""
    from src.config import settings

    console.print(Rule("[bold]Collection inspection[/bold]"))

    # Basic info
    info = client.get_collection(settings.collection_name)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan", width=22)
    table.add_column("Value")

    table.add_row("Collection name",  settings.collection_name)
    table.add_row("Status",           str(info.status.value if info.status else "unknown"))
    table.add_row("Points (vectors)", str(info.points_count or 0))
    table.add_row("Segments",         str(info.segments_count or 0))
    table.add_row("Vector dimension", str(settings.embedding_dim))
    table.add_row("Distance metric",  "COSINE")
    table.add_row("Embedding model",  settings.embedding_model)

    console.print(Panel(table, title="[blue]Collection stats[/blue]", border_style="blue"))


def inspect_sample_points(client) -> None:
    """Show what an actual stored point looks like — the full payload."""
    from src.config import settings

    console.print(Rule("[bold]Sample point inspection[/bold]"))

    # Scroll through first 3 points
    results = client.scroll(
        collection_name=settings.collection_name,
        limit=3,
        with_payload=True,
        with_vectors=False,  # skip the 1536 floats — too verbose
    )

    points = results[0]
    if not points:
        console.print("[yellow]No points found. Run 01_setup.py first.[/yellow]")
        return

    for i, point in enumerate(points, 1):
        payload = point.payload or {}
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key",   style="cyan",  width=18)
        table.add_column("Value", style="white")

        table.add_row("point_id",    str(point.id)[:36])
        for k, v in payload.items():
            if k == "text":
                table.add_row(k, str(v)[:100] + "..." if len(str(v)) > 100 else str(v))
            else:
                table.add_row(k, str(v))

        console.print(Panel(table, title=f"[green]Point {i}[/green]", border_style="green"))


def benchmark_retrieval(client) -> None:
    """
    Measure retrieval latency across different K values.

    This separates the two components of retrieval latency:
      Embedding:   OpenAI API call (~100-150ms) — network-bound
      HNSW search: Qdrant query (~1-5ms) — CPU-bound

    The benchmark uses a CACHED embedding to isolate Qdrant's speed.
    """
    from src.config import settings
    from src.module1_naive_rag.embeddings import embed_text

    console.print(Rule("[bold]Retrieval benchmark[/bold]"))
    console.print("[dim]Embedding query (one-time cost)...[/dim]")

    # Time the embedding call
    t0 = time.perf_counter()
    query_vector = embed_text("Why is my pod showing OOMKilled?")
    embed_ms = (time.perf_counter() - t0) * 1000
    console.print(f"[dim]Embedding: {embed_ms:.0f}ms (OpenAI API, network-bound)[/dim]\n")

    # Benchmark Qdrant at different K values (using cached vector)
    k_values = [1, 3, 5, 10, 18]  # 18 = all points (brute force territory)

    table = Table(
        title="Qdrant search latency (embedding time excluded)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("K",            width=5)
    table.add_column("Latency",      width=10)
    table.add_column("Best score",   width=12)
    table.add_column("Worst score",  width=12)
    table.add_column("Top source")

    for k in k_values:
        if k > (client.get_collection(settings.collection_name).points_count or 0):
            continue

        # Run 3 times, take median (warm up cache)
        latencies = []
        last_results = None
        for _ in range(3):
            t = time.perf_counter()
            results = client.query_points(
                collection_name=settings.collection_name,
                query=query_vector,
                limit=k,
                with_payload=True,
                with_vectors=False,
            )
            latencies.append((time.perf_counter() - t) * 1000)
            last_results = results.points

        median_ms = sorted(latencies)[1]
        scores = [r.score for r in last_results]
        top_source = last_results[0].payload.get("source", "?").split("/")[-1] if last_results else "?"

        table.add_row(
            str(k),
            f"{median_ms:.2f}ms",
            f"{max(scores):.4f}" if scores else "—",
            f"{min(scores):.4f}" if scores else "—",
            top_source,
        )

    console.print(table)
    console.print(
        f"\n[dim]At our scale (18 points), HNSW uses brute force — "
        f"all sub-millisecond.\nHNSW advantage appears at >10k vectors.[/dim]"
    )


def upgrade_collection(client) -> None:
    """
    Recreate the collection with proper HNSW tuning + payload indexes.
    Re-ingests all data after recreation.
    """
    from src.module5_vector_db.collection_manager import (
        create_production_collection,
        create_payload_indexes,
        get_collection_stats,
    )
    from src.config import settings

    console.print(Rule("[bold]Upgrading collection to production config[/bold]"))
    console.print("[yellow]⚠ This will delete and recreate the collection. All data will be re-ingested.[/yellow]")

    # Recreate with tuned HNSW
    create_production_collection(client, force_recreate=True)
    console.print("[green]✓ Collection recreated with m=16, ef_construct=200[/green]")

    # Add payload indexes
    create_payload_indexes(client)
    console.print("[green]✓ Payload indexes created (doc_type, team, k8s_version)[/green]")

    # Re-ingest
    console.print("\n[dim]Re-ingesting knowledge base...[/dim]")
    from data.k8s_knowledge_base import DOCUMENTS as K8S_DOCUMENTS
    from src.module1_naive_rag.ingestion import ingest_many_documents

    stats = ingest_many_documents(client, K8S_DOCUMENTS)
    console.print(f"[green]✓ Ingested: {stats.ingested} chunks, {stats.errors} errors[/green]")

    # Final stats
    info = get_collection_stats(client)
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan", width=20)
    table.add_column("Value")
    for k, v in info.items():
        table.add_row(str(k), str(v))
    console.print(Panel(table, title="[blue]Upgraded collection[/blue]", border_style="blue"))


def show_score_distribution(client) -> None:
    """
    Show the cosine similarity score distribution for a set of test queries.
    This helps calibrate the CRAG threshold (currently 0.7).
    """
    from src.module1_naive_rag.embeddings import embed_text
    from src.config import settings

    console.print(Rule("[bold]Score distribution analysis[/bold]"))
    console.print("[dim]Running 5 sample queries to understand score distribution...[/dim]\n")

    test_queries = [
        ("High relevance",   "What does OOMKilled exit code 137 mean?"),
        ("Medium relevance", "How do I debug a failing pod?"),
        ("Low relevance",    "What is the DNS resolution flow in Kubernetes?"),
        ("Vague query",      "Which pods are failing and what should I do?"),
        ("Off-domain",       "How do I configure nginx rate limiting?"),
    ]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Query type",     width=18)
    table.add_column("Best score",     width=12)
    table.add_column("Avg score",      width=10)
    table.add_column("CRAG action",    width=18)
    table.add_column("Top source")

    for label, query in test_queries:
        vec = embed_text(query)
        results = client.query_points(
            collection_name=settings.collection_name,
            query=vec,
            limit=5,
            with_payload=True,
            with_vectors=False,
        )
        pts = results.points
        if not pts:
            continue

        scores   = [r.score for r in pts]
        best     = max(scores)
        avg      = sum(scores) / len(scores)
        source   = pts[0].payload.get("source", "?").split("/")[-1]

        # CRAG threshold logic (Phase 4)
        if best >= 0.80:
            action       = "[green]✓ High confidence[/green]"
        elif best >= 0.70:
            action       = "[yellow]~ Medium confidence[/yellow]"
        else:
            action       = "[red]→ Tavily fallback[/red]"

        table.add_row(label, f"{best:.4f}", f"{avg:.4f}", action, source)

    console.print(table)
    console.print(
        "\n[dim]CRAG threshold = 0.70 (Phase 4). "
        "Queries scoring below this trigger Tavily web search fallback.[/dim]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qdrant collection inspector")
    parser.add_argument("--bench",   action="store_true", help="Retrieval benchmark only")
    parser.add_argument("--upgrade", action="store_true", help="Recreate with HNSW tuning")
    parser.add_argument("--scores",  action="store_true", help="Score distribution analysis")
    args = parser.parse_args()

    from src.module1_naive_rag.collection import get_qdrant_client
    client = get_qdrant_client()

    if args.upgrade:
        upgrade_collection(client)
    elif args.bench:
        benchmark_retrieval(client)
    elif args.scores:
        show_score_distribution(client)
    else:
        # Full inspection
        inspect_collection(client)
        console.print()
        inspect_sample_points(client)
        console.print()
        benchmark_retrieval(client)
        console.print()
        show_score_distribution(client)

    console.print()


if __name__ == "__main__":
    main()

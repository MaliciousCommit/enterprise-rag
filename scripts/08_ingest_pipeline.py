#!/usr/bin/env python3
# scripts/08_ingest_pipeline.py
#
# Demonstrates the Module 6 ingestion pipeline.
#
# WHAT THIS DOES:
# 1. Compares all three chunking strategies on a sample K8s runbook
# 2. Re-ingests the full knowledge base using MarkdownChunker
# 3. Shows score improvements from better chunking
#
# Usage:
#   python scripts/08_ingest_pipeline.py             # compare + re-ingest
#   python scripts/08_ingest_pipeline.py --compare   # comparison only (no API calls)
#   python scripts/08_ingest_pipeline.py --reingest  # re-ingest only

import sys
import os
import asyncio
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule

console = Console()

SAMPLE_RUNBOOK = """
# OOMKilled Runbook

## Overview
OOMKilled (Out of Memory Killed) occurs when a container exceeds its memory limit.
The Linux kernel's OOM killer terminates the process with exit code 137.

## Symptoms
- Pod status shows OOMKilled
- Exit code 137 in container status
- kubectl describe pod shows Last State: OOMKilled
- Repeated restarts with increasing CrashLoopBackOff intervals

## Diagnosis

### Check Pod Status
Run kubectl get pods to identify pods in error state.
Look for restart counts greater than 3 as a signal of repeated OOM events.

### Inspect Resource Usage
Use kubectl top pod to see current memory usage.
Compare against the configured limits in the pod spec.
If usage is near 80% of limit, the pod is at risk.

### Review Limits Configuration
Check the pod's resource spec with kubectl describe pod.
Look for resources.limits.memory in the container spec.
Verify that limits are set and appropriate for the workload.

## Resolution

### Increase Memory Limit
Edit the deployment to increase resources.limits.memory.
A safe starting point is 150% of observed peak usage.
Example: kubectl set resources deployment/payment-service --limits=memory=512Mi

### Add Memory Request
If only limits are set (no requests), add a matching request.
Set resources.requests.memory to 60-70% of the limit.
This allows the scheduler to place the pod correctly.

### Implement Graceful Degradation
Configure the application to shed load when memory pressure is high.
Implement /health endpoint that returns 503 when memory > 80% of limit.
Use HPA to scale out before hitting OOM conditions.

## Prevention
Set both requests and limits for all production containers.
Configure VPA (Vertical Pod Autoscaler) for automatic limit tuning.
Set up Prometheus alerts for memory usage > 80% of limit.
Review limits quarterly as traffic patterns change.
"""


def compare_chunking_strategies() -> None:
    """Show how different strategies chunk the same document."""
    from src.module6_ingestion.chunker import compare_strategies

    console.print(Rule("[bold]Chunking strategy comparison[/bold]"))
    console.print("[dim]Same document, three strategies — notice chunk count and size differences[/dim]\n")

    results = compare_strategies(SAMPLE_RUNBOOK, source="runbooks/oomkilled.md")

    # Summary table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Strategy",   width=12)
    table.add_column("Chunks",     width=8)
    table.add_column("Min chars",  width=10)
    table.add_column("Max chars",  width=10)
    table.add_column("Avg chars",  width=10)
    table.add_column("Verdict")

    verdicts = {
        "fixed":     "[yellow]Cuts mid-sentence — avoid for structured docs[/yellow]",
        "recursive": "[blue]Clean boundaries — good for prose[/blue]",
        "markdown":  "[green]Section-aware — best for our runbooks[/green]",
    }

    for strategy, stats in results.items():
        table.add_row(
            strategy,
            str(stats["num_chunks"]),
            str(stats["min_chars"]),
            str(stats["max_chars"]),
            str(stats["avg_chars"]),
            verdicts[strategy],
        )
    console.print(table)

    # Show markdown chunks in detail (the winner)
    console.print(f"\n[bold]Markdown chunks (showing first 3 of {len(results['markdown']['chunks'])}):[/bold]")
    for chunk in results["markdown"]["chunks"][:3]:
        heading = f"[cyan]{chunk.heading_path}[/cyan]" if chunk.heading_path else "[dim]no heading[/dim]"
        console.print(Panel(
            f"[dim]path:[/dim] {heading}\n"
            f"[dim]chars:[/dim] {chunk.char_count} | [dim]tokens≈:[/dim] {chunk.token_count}\n\n"
            + chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
            border_style="green",
        ))


def reingest_with_markdown_chunker(client) -> None:
    """Re-ingest the knowledge base using the better MarkdownChunker."""
    from src.module6_ingestion.pipeline import RawDocument, ingest_documents, ChunkingStrategy
    from data.k8s_knowledge_base import DOCUMENTS

    console.print(Rule("[bold]Re-ingesting with MarkdownChunker[/bold]"))
    console.print(f"[dim]{len(DOCUMENTS)} documents → MarkdownChunker[/dim]\n")

    # Convert knowledge base documents to RawDocument format
    raw_docs = []
    for doc in DOCUMENTS:
        raw_docs.append(RawDocument(
    text        = doc["content"],
    source      = doc.get("source", doc.get("document_id", "unknown")),
    document_id = doc.get("document_id", "unknown"),
    doc_type    = doc.get("category", "runbook"),
    team        = doc.get("team", "platform"),
    k8s_version = doc.get("k8s_version", "1.29"),
    tags        = doc.get("tags", []),
    strategy    = ChunkingStrategy.MARKDOWN,
))

    stats = ingest_documents(
        client      = client,
        documents   = raw_docs,
        skip_existing = False,   # force re-ingest with new chunker
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key",   style="cyan", width=22)
    table.add_column("Value")
    table.add_row("Documents processed",  str(stats.total_documents))
    table.add_row("Chunks produced",      str(stats.total_chunks))
    table.add_row("Chunks ingested",      str(stats.ingested))
    table.add_row("Chunks skipped",       str(stats.skipped))
    table.add_row("Errors",               str(stats.errors))
    table.add_row("Embedding API calls",  str(stats.embedding_calls))
    table.add_row("Total time",           f"{stats.elapsed_seconds:.1f}s")
    table.add_row("Throughput",           f"{stats.throughput:.1f} chunks/sec")
    console.print(Panel(table, title="[green]Ingestion stats[/green]", border_style="green"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare",  action="store_true", help="Strategy comparison only")
    parser.add_argument("--reingest", action="store_true", help="Re-ingest only")
    args = parser.parse_args()

    from src.config import settings
    try:
        settings.validate()
    except ValueError as e:
        console.print(f"[red]Config: {e}[/red]")
        sys.exit(1)

    if args.compare:
        compare_chunking_strategies()
        return

    from src.module1_naive_rag.collection import get_qdrant_client
    client = get_qdrant_client()

    if args.reingest:
        reingest_with_markdown_chunker(client)
        return

    # Default: compare then re-ingest
    compare_chunking_strategies()
    console.print()
    reingest_with_markdown_chunker(client)


if __name__ == "__main__":
    main()

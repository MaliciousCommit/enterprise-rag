#!/usr/bin/env python3
# scripts/02_query.py
#
# Interactive query session against the Module 1 naive RAG system.
# Run AFTER scripts/01_setup.py has ingested the knowledge base.
#
# Usage (from project root):
#   python scripts/02_query.py
#   python scripts/02_query.py --question "Why is my pod OOMKilled?"
#   python scripts/02_query.py --k 10    # retrieve 10 chunks instead of 5

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.module1_naive_rag import NaiveRAG
from src.module1_naive_rag.collection import get_qdrant_client, get_collection_info

logging.basicConfig(
    level=logging.WARNING,   # suppress info logs for cleaner output in interactive mode
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


EXAMPLE_QUESTIONS = [
    "Why is my pod showing OOMKilled and how do I fix it?",
    "My pod is in CrashLoopBackOff, what should I do?",
    "How do I set memory and CPU limits for my pods?",
    "What is the difference between resource requests and limits?",
    "How do I debug a pod that won't start?",
    "How do I roll back a failed deployment?",
    "My service is not reachable from other pods, what do I check?",
    "How do I check which pods are using the most memory?",
    "What should I do when a node is in NotReady state?",
    "How do I grant a service account permission to list pods?",
]


def main():
    parser = argparse.ArgumentParser(description="Query the Enterprise RAG system")
    parser.add_argument("--question", "-q", type=str, help="Single question to ask")
    parser.add_argument("--k", type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument("--no-spotlighting", action="store_true",
                        help="Disable XML spotlighting (use plain separators)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Enterprise RAG — Module 1 | Naive RAG Query Interface")
    print("="*60)

    # Validate config
    try:
        settings.validate()
    except ValueError as e:
        print(f"✗ Config error: {e}")
        sys.exit(1)

    # Check collection exists and has data
    client = get_qdrant_client()
    try:
        info = get_collection_info(client)
        print(f"  Collection: {info['name']} | {info['points_count']} points | status: {info['status']}")
        if info["points_count"] == 0:
            print("  ✗ Collection is empty. Run scripts/01_setup.py first.")
            sys.exit(1)
    except Exception as e:
        print(f"  ✗ Cannot access Qdrant: {e}")
        print("  → Start Qdrant: docker compose up qdrant -d")
        sys.exit(1)

    # Create RAG instance
    rag = NaiveRAG(client=client)

    # Single question mode
    if args.question:
        rag.query_and_print(args.question, k=args.k)
        return

    # Interactive mode
    print(f"\n  Retrieval: k={args.k or settings.retrieval_k} chunks")
    print("  Type your question, 'examples' to see sample questions, or 'quit' to exit.\n")

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break
        if question.lower() == "examples":
            print("\nExample questions:")
            for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
                print(f"  {i:2}. {q}")
            print()
            continue

        try:
            rag.query_and_print(question, k=args.k)
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


if __name__ == "__main__":
    main()

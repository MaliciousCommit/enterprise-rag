#!/usr/bin/env python3
# scripts/01_setup.py
#
# Run this ONCE before querying the system.
# Creates the Qdrant collection and ingests all K8s knowledge base documents.
#
# Usage (from project root):
#   python scripts/01_setup.py
#   python scripts/01_setup.py --recreate    # wipe and re-ingest
#
# Prerequisites:
#   1. Copy .env.example to .env and set OPENAI_API_KEY
#   2. Start Qdrant: docker compose up qdrant -d
#   3. Verify Qdrant is running: curl http://localhost:6333/healthz

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path so `src` is importable regardless of CWD.
# This replaces needing to install the project with pip install -e .
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.module1_naive_rag.collection import (
    get_qdrant_client,
    create_collection,
    get_collection_info,
)
from src.module1_naive_rag.ingestion import ingest_document
from data.k8s_knowledge_base import get_all_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Set up the Enterprise RAG knowledge base")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection (WARNING: deletes all existing data)",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Enterprise RAG — Module 1 Setup")
    print("="*60)

    # Step 1: Validate config
    print("\n[1/4] Validating configuration...")
    try:
        settings.validate()
        print(f"  ✓ OPENAI_API_KEY:   set ({settings.openai_api_key[:12]}...)")
        print(f"  ✓ Qdrant host:      {settings.qdrant_host}:{settings.qdrant_port}")
        print(f"  ✓ Collection:       {settings.collection_name}")
        print(f"  ✓ Embedding model:  {settings.embedding_model}")
        print(f"  ✓ LLM model:        {settings.llm_model}")
    except ValueError as e:
        print(f"  ✗ Configuration error: {e}")
        print("\n  → Copy .env.example to .env and set your OPENAI_API_KEY")
        sys.exit(1)

    # Step 2: Connect to Qdrant
    print("\n[2/4] Connecting to Qdrant...")
    try:
        client = get_qdrant_client()
        # Test connection with a simple collections list
        collections = client.get_collections().collections
        print(f"  ✓ Connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
        print(f"  ✓ Existing collections: {[c.name for c in collections]}")
    except Exception as e:
        print(f"  ✗ Cannot connect to Qdrant: {e}")
        print("\n  → Start Qdrant with: docker compose up qdrant -d")
        print("  → Verify it's running: curl http://localhost:6333/healthz")
        sys.exit(1)

    # Step 3: Create collection
    print(f"\n[3/4] Setting up collection '{settings.collection_name}'...")
    create_collection(client, force_recreate=args.recreate)
    info = get_collection_info(client)
    print(f"  ✓ Collection status: {info['status']}")
    print(f"  ✓ Current points:    {info['points_count']}")

    if info["points_count"] > 0 and not args.recreate:
        print(f"\n  ⚠ Collection already has {info['points_count']} points.")
        print("  ⚠ Re-running ingestion will ADD duplicate points.")
        print("  ⚠ Use --recreate to wipe and re-ingest cleanly.")
        response = input("\n  Continue anyway? [y/N]: ").strip().lower()
        if response != "y":
            print("  Aborted. Use --recreate to start fresh.")
            sys.exit(0)

    # Step 4: Ingest knowledge base documents
    documents = get_all_documents()
    print(f"\n[4/4] Ingesting {len(documents)} documents...")
    print(f"  Chunk size: {settings.chunk_size} words | Overlap: {settings.chunk_overlap} words")

    total_ingested = 0
    total_errors = 0
    total_chunks = 0

    for i, doc in enumerate(documents, 1):
        print(f"\n  Document {i}/{len(documents)}: {doc['document_id']}")
        word_count = len(doc["content"].split())
        print(f"    Source:   {doc['source']}")
        print(f"    Words:    {word_count}")

        try:
            stats = ingest_document(
                client=client,
                text=doc["content"],
                source=doc["source"],
                document_id=doc["document_id"],
                metadata={
                    "category":    doc.get("category", "unknown"),
                    "k8s_version": doc.get("k8s_version", "1.29"),
                },
            )
            total_ingested += stats["ingested"]
            total_errors   += stats["errors"]
            total_chunks   += stats["chunks_created"]
            print(f"    Result:   {stats['chunks_created']} chunks → {stats['ingested']} ingested")
        except Exception as e:
            logger.error(f"Failed to ingest '{doc['document_id']}': {e}")
            total_errors += 1

    # Final summary
    print("\n" + "="*60)
    print("  INGESTION COMPLETE")
    print("="*60)
    print(f"  Documents processed: {len(documents)}")
    print(f"  Total chunks created: {total_chunks}")
    print(f"  Points ingested:     {total_ingested}")
    print(f"  Errors:              {total_errors}")

    # Verify final collection state
    final_info = get_collection_info(client)
    print(f"  Collection points:   {final_info['points_count']}")
    print(f"  Collection status:   {final_info['status']}")

    if total_errors == 0:
        print("\n  ✓ Setup complete. Run scripts/02_query.py to test the system.")
    else:
        print(f"\n  ⚠ {total_errors} errors occurred. Check logs above.")

    # Estimated embedding cost
    # Each chunk ~350 tokens avg, $0.020/M tokens
    est_tokens = total_chunks * 350
    est_cost = est_tokens * 0.020 / 1_000_000
    print(f"\n  Estimated embedding cost: ${est_cost:.4f}")
    print(f"  ({total_chunks} chunks × ~350 tokens × $0.020/M tokens)")


if __name__ == "__main__":
    main()

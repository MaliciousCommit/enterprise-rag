# Enterprise RAG — Kubernetes IT Operations

Production-grade Retrieval-Augmented Generation system built incrementally across 11 phases.

## Current Phase: Module 1 — Naive RAG Foundations

**Stack:** OpenAI (text-embedding-3-small + gpt-4o) · Qdrant · Python 3.11+

---

## Quick Start (Module 1)

### 1. Prerequisites
- Python 3.11+
- Docker + Docker Compose
- OpenAI API key

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 4. Start Qdrant
```bash
docker compose up qdrant -d

# Verify it's running:
curl http://localhost:6333/healthz
# → {"title":"qdrant - vector search engine","version":"1.9.0"}
```

### 5. Ingest the knowledge base
```bash
python scripts/01_setup.py
# Ingests 10 K8s runbooks/guides into Qdrant (~$0.0001 in embedding costs)
```

### 6. Query the system
```bash
# Interactive mode:
python scripts/02_query.py

# Single question:
python scripts/02_query.py --question "Why is my pod OOMKilled?"
```

### 7. Run benchmark
```bash
python scripts/03_benchmark.py
# Measures retrieval quality and establishes the Phase 3 improvement baseline
```

---

## Project Structure

```
enterprise_rag/
├── src/
│   ├── config.py                    # Central settings (all env vars in one place)
│   └── module1_naive_rag/
│       ├── collection.py            # Qdrant lifecycle management
│       ├── embeddings.py            # OpenAI embedding with batching + retry
│       ├── ingestion.py             # Chunk → embed → upsert pipeline
│       ├── retrieval.py             # HNSW search + RetrievedChunk model
│       ├── generation.py            # GPT-4o grounded generation
│       └── pipeline.py              # NaiveRAG class (ties everything together)
├── data/
│   └── k8s_knowledge_base.py        # 10 K8s runbooks and guides
├── scripts/
│   ├── 01_setup.py                  # Create collection + ingest docs
│   ├── 02_query.py                  # Interactive query session
│   └── 03_benchmark.py              # Measure retrieval quality
└── docker-compose.yml               # Qdrant (+ PostgreSQL, Redis in later phases)
```

---

## What This Module Teaches

- RAG fundamentals: embed → search → generate
- Qdrant HNSW vector index internals
- OpenAI embedding API with batching and retry
- Grounded prompt construction with XML spotlighting
- Chunk overlap strategy and why it matters
- Cost and latency analysis of each pipeline step

## What's Missing (Fixed in Later Phases)

| Gap | Phase |
|-----|-------|
| Hybrid search (dense + BM25) | Phase 3 |
| Cross-encoder reranking | Phase 3 |
| HyDE query expansion | Phase 3 |
| CRAG + Tavily web fallback | Phase 4 |
| Self-RAG quality reflection | Phase 4 |
| Text2SQL + HITL approval | Phase 5 |
| Redis 5-tier semantic cache | Phase 6 |
| RAGAS evaluation | Phase 7 |
| 9-layer guardrails | Phase 8 |
| LangGraph state machine | Phase 4 |
| FastAPI REST endpoint | Phase 3 |
| Kubernetes deployment | Phase 10 |

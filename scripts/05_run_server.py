#!/usr/bin/env python3
# scripts/05_run_server.py
#
# Start the Enterprise RAG FastAPI server with Uvicorn.
#
# WHAT IS UVICORN?
# Uvicorn is an ASGI (Async Server Gateway Interface) server.
# It sits between the network and FastAPI, handling:
#   - TCP connection management
#   - HTTP/1.1 and HTTP/2 parsing
#   - TLS termination (in production)
#   - Worker process management
#
# ASGI vs WSGI:
# WSGI (Flask, Django old): one thread per request, synchronous
# ASGI (FastAPI, Starlette): async event loop, many requests per thread
# Our LangGraph pipeline does async I/O (OpenAI, Qdrant) so ASGI is essential.
#
# DEVELOPMENT MODE (--reload):
# Uvicorn watches for file changes and restarts automatically.
# Great for development — no need to manually restart after code changes.
# Never use --reload in production (file watching overhead).
#
# PRODUCTION MODE (Phase 10):
# Use Gunicorn as the process manager with Uvicorn workers:
#   gunicorn src.api.app:app \
#     --workers 4 \           # 4 worker processes (one per CPU core)
#     --worker-class uvicorn.workers.UvicornWorker \
#     --bind 0.0.0.0:8000 \
#     --timeout 120           # allow 2 minutes for long LLM calls
#
# In Kubernetes (Phase 10), uvicorn runs directly (no Gunicorn needed)
# because Kubernetes handles multiple replicas:
#   CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn


def main():
    print("=" * 60)
    print("  Enterprise RAG — FastAPI Server")
    print("  http://localhost:8000")
    print("  Swagger UI: http://localhost:8000/docs")
    print("  Health:     http://localhost:8000/health")
    print("=" * 60)

    uvicorn.run(
        "src.api.app:app",      # module:attribute path to the FastAPI app
        host="0.0.0.0",         # listen on all interfaces (not just localhost)
        port=8000,
        reload=True,            # auto-restart on file changes (dev only)
        log_level="info",       # uvicorn access log level
        # In production (Phase 10 Kubernetes), remove reload=True and add:
        # workers=4,            # multiple processes
        # access_log=True,      # log every request
    )


if __name__ == "__main__":
    main()

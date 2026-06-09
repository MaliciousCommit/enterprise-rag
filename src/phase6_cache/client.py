# src/phase6_cache/client.py
#
# Redis connection management.
#
# SUPPORTS TWO BACKENDS:
#
# 1. Local Redis (docker-compose)
#    Set: REDIS_URL=redis://localhost:6379
#    Start: docker compose up -d redis
#    Best for: development, learning, this curriculum
#
# 2. Upstash Redis (serverless, HTTP)
#    Set: REDIS_URL=https://xxx.upstash.io
#         REDIS_TOKEN=your-token
#    Best for: production without managing Redis infrastructure
#    Free tier: 10,000 requests/day, 256MB storage
#    https://upstash.com
#
# GRACEFUL DEGRADATION:
# If Redis is unreachable (not running, wrong URL, network issue),
# all cache operations silently return None/False instead of raising.
# The pipeline continues without caching — just slower.
# This is the correct production behaviour: caching should never
# be a hard dependency that takes down the system.
#
# CONNECTION POOL:
# redis.Redis maintains an internal connection pool (default: 50 connections).
# The singleton pattern here means one pool shared across all requests.
# For 4 FastAPI workers: 4 pools × 50 connections = 200 max Redis connections.
# Redis default max: 10,000. Plenty of headroom.

import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_redis_client():
    """
    Return the Redis client singleton.

    Called once on first cache access. Subsequent calls return
    the same client from lru_cache.

    Returns None if Redis is unavailable — callers must handle this.
    """
    url   = os.getenv("REDIS_URL",   "redis://localhost:6379")
    token = os.getenv("REDIS_TOKEN", "")

    try:
        import redis

        # Upstash uses HTTPS + token auth
        if url.startswith("https://") and token:
            client = redis.Redis.from_url(
                url,
                password=token,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        else:
            # Local Redis (no auth)
            client = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )

        # Verify connection
        client.ping()
        logger.info(f"Redis connected: {url.split('@')[-1]}")  # hide credentials
        return client

    except Exception as e:
        logger.warning(
            f"Redis unavailable ({e}). Caching disabled. "
            "Start Redis with: docker compose up -d redis"
        )
        return None


def test_redis_connection() -> bool:
    """Test Redis connection. Returns True/False."""
    client = get_redis_client()
    if client is None:
        return False
    try:
        client.ping()
        return True
    except Exception:
        return False

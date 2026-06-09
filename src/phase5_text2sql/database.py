# src/phase5_text2sql/database.py
#
# PostgreSQL connection management and safe query execution.
#
# TWO RESPONSIBILITIES:
# 1. Connection pool management (create once, reuse across requests)
# 2. Safe query execution (timeout, row limits, error handling)
#
# WHY A ROW LIMIT?
# Without limits, a user could ask "show me all pods" and trigger:
#   SELECT * FROM pods   → potentially millions of rows if data grows
# This would: fill context window, increase token cost, slow response.
# We enforce LIMIT 100 at the execution layer (not just the SQL layer).
#
# WHY A STATEMENT TIMEOUT?
# A bad JOIN or missing index could cause a query to run for minutes.
# SET statement_timeout = '5s' in the session tells PostgreSQL to kill
# any query that runs longer than 5 seconds. This protects the database.

import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional

import psycopg2
import psycopg2.extras  # for RealDictCursor (returns rows as dicts)

from src.phase5_text2sql.schema import CREATE_TABLES_SQL, SAMPLE_DATA_SQL

logger = logging.getLogger(__name__)

# Safety limits for SQL execution
MAX_ROWS             = 100      # never return more than this many rows
STATEMENT_TIMEOUT_MS = 5_000    # kill queries that run > 5 seconds


def get_database_url() -> str:
    """
    Build the PostgreSQL connection URL from environment variables.

    Expected .env variables:
        POSTGRES_HOST     (default: localhost)
        POSTGRES_PORT     (default: 5432)
        POSTGRES_DB       (default: enterprise_rag)
        POSTGRES_USER     (default: raguser)
        POSTGRES_PASSWORD (default: ragpass)
    """
    host = os.getenv("POSTGRES_HOST",     "localhost")
    port = os.getenv("POSTGRES_PORT",     "5432")
    db   = os.getenv("POSTGRES_DB",       "enterprise_rag")
    user = os.getenv("POSTGRES_USER",     "raguser")
    pwd  = os.getenv("POSTGRES_PASSWORD", "ragpass")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


@lru_cache(maxsize=1)
def get_connection_params() -> dict:
    """Return connection kwargs for psycopg2.connect()."""
    url = get_database_url()
    # Parse the URL manually (psycopg2 doesn't accept URL format directly in all versions)
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    return {
        "host":     parsed.hostname,
        "port":     parsed.port or 5432,
        "dbname":   parsed.path.lstrip("/"),
        "user":     parsed.username,
        "password": parsed.password,
        "connect_timeout": 5,
        "application_name": "enterprise-rag",
    }


@contextmanager
def get_db_connection():
    """
    Context manager: get a PostgreSQL connection, close it on exit.

    WHY NOT A PERSISTENT CONNECTION POOL?
    For a learning curriculum with a single FastAPI process and low traffic,
    simple connection-per-request is sufficient. Each connection takes ~5ms.
    Phase 10 (production) would add connection pooling via PgBouncer or
    SQLAlchemy's QueuePool (pool_size=20, max_overflow=10).
    """
    conn = None
    try:
        conn = psycopg2.connect(**get_connection_params())
        yield conn
    finally:
        if conn and not conn.closed:
            conn.close()


def test_connection() -> bool:
    """Test that the PostgreSQL database is reachable. Returns True/False."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        logger.info("PostgreSQL: connection OK")
        return True
    except Exception as e:
        logger.error(f"PostgreSQL: connection failed — {e}")
        return False


def execute_select(sql: str) -> dict:
    """
    Execute a SELECT query safely and return results as a dict.

    SAFETY MEASURES:
    1. SET statement_timeout: kills any query exceeding 5 seconds
    2. MAX_ROWS limit: cursor.fetchmany() rather than fetchall()
    3. READ-ONLY transaction: protects against accidental writes
    4. Exception handling: database errors become clean error messages

    Returns:
        {
            "columns": ["col1", "col2", ...],
            "rows":    [["val1", "val2"], ...],
            "row_count": 5,
            "truncated": False,   # True if > MAX_ROWS results
            "error": None,        # error message string if query failed
        }

    NEVER raises — always returns a result dict.
    The caller (sql_execute_node) handles the "error" key.
    """
    try:
        with get_db_connection() as conn:
            # Open read-only transaction (prevents accidental writes)
            conn.set_session(readonly=True, autocommit=False)

            with conn.cursor() as cur:
                # Kill the query if it runs longer than the timeout
                cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")

                cur.execute(sql)

                # Get column names from cursor description
                columns = [desc[0] for desc in cur.description] if cur.description else []

                # Fetch at most MAX_ROWS + 1 (the extra row tells us if truncated)
                rows_raw = cur.fetchmany(MAX_ROWS + 1)
                truncated = len(rows_raw) > MAX_ROWS
                rows = [list(r) for r in rows_raw[:MAX_ROWS]]

                # Convert any non-serialisable types to strings
                rows = [[str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                         for v in row]
                        for row in rows]

                logger.info(
                    f"SQL executed: {len(rows)} rows returned"
                    + (" (truncated to 100)" if truncated else "")
                )

                return {
                    "columns":   columns,
                    "rows":      rows,
                    "row_count": len(rows),
                    "truncated": truncated,
                    "error":     None,
                }

    except psycopg2.errors.QueryCanceled:
        msg = f"Query cancelled: exceeded {STATEMENT_TIMEOUT_MS}ms timeout"
        logger.error(msg)
        return {"columns": [], "rows": [], "row_count": 0, "truncated": False, "error": msg}

    except psycopg2.Error as e:
        msg = f"Database error: {e.pgcode} — {e.pgerror or str(e)}"
        logger.error(msg)
        return {"columns": [], "rows": [], "row_count": 0, "truncated": False, "error": msg}

    except Exception as e:
        msg = f"Unexpected error executing SQL: {type(e).__name__}: {e}"
        logger.error(msg)
        return {"columns": [], "rows": [], "row_count": 0, "truncated": False, "error": msg}


def format_results_as_text(result: dict) -> str:
    """
    Format SQL result dict into a human-readable table string for LLM context.

    Example output:
        pod_name                     | namespace | status           | restart_count
        ─────────────────────────────┼───────────┼──────────────────┼──────────────
        payment-svc-7d9f8-xk2p      | prod      | CrashLoopBackOff | 14
        payment-svc-7d9f8-mk8l      | prod      | CrashLoopBackOff | 8

        (2 rows)
    """
    if result.get("error"):
        return f"SQL execution error: {result['error']}"

    columns  = result["columns"]
    rows     = result["rows"]
    row_count = result["row_count"]

    if not rows:
        return "Query returned 0 rows."

    # Calculate column widths
    col_widths = [len(str(c)) for c in columns]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val) if val is not None else "NULL"))

    # Header
    header    = " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(columns))
    separator = "─" * len(header)
    lines     = [header, separator]

    # Rows
    for row in rows:
        line = " | ".join(
            str(v if v is not None else "NULL").ljust(col_widths[i])
            for i, v in enumerate(row)
        )
        lines.append(line)

    suffix = f"\n({row_count} rows)"
    if result.get("truncated"):
        suffix += " [truncated to 100 rows]"

    return "\n".join(lines) + suffix


def setup_database() -> bool:
    """
    Create the schema and load sample data.
    Called by scripts/12_setup_postgres.py.
    Idempotent: safe to call multiple times (uses IF NOT EXISTS).
    """
    try:
        with get_db_connection() as conn:
            conn.set_session(autocommit=False)
            with conn.cursor() as cur:
                logger.info("Creating schema...")
                cur.execute(CREATE_TABLES_SQL)
                logger.info("Loading sample data...")
                cur.execute(SAMPLE_DATA_SQL)
            conn.commit()
        logger.info("Database setup complete")
        return True
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False

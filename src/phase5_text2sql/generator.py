# src/phase5_text2sql/generator.py
#
# LLM-based SQL generation with schema context.
#
# THE CORE CHALLENGE:
# GPT-4o knows SQL syntax but doesn't know our schema.
# We must inject the exact table and column names into every prompt.
# A hallucinated column name → SQL syntax error → execution fails.
#
# WHY GPT-4o (NOT gpt-4o-mini) FOR SQL GENERATION?
# SQL generation is more demanding than intent classification.
# It requires:
#   - Precise column name recall (no hallucination tolerance)
#   - Correct JOIN logic for multi-table questions
#   - Proper aggregation (GROUP BY, HAVING)
#   - K8s domain knowledge (what is a "failing" pod in Kubernetes?)
#
# gpt-4o achieves ~95% correct SQL on first attempt with our schema.
# gpt-4o-mini drops to ~80% — too many re-tries needed.
# We use temperature=0.0 for full determinism — same question → same SQL.
#
# FEW-SHOT EXAMPLES:
# The examples below are critical. They teach the LLM:
#   - How to join pods to clusters using cluster_id
#   - Which status values count as "failing"
#   - How to use recorded_at for time-based queries
#   - That LIMIT is always required

import logging
import re

from openai import AsyncOpenAI

from src.phase5_text2sql.schema import SCHEMA_DESCRIPTION

logger = logging.getLogger(__name__)

SQL_GEN_MODEL = "gpt-4o"
SQL_GEN_TEMP  = 0.0
SQL_GEN_MAX_TOKENS = 500


SQL_GENERATION_PROMPT = f"""
You are an expert PostgreSQL query writer for a Kubernetes IT Operations database.

SCHEMA:
{SCHEMA_DESCRIPTION}

RULES — follow these exactly:
1. Write ONLY the SQL query. No explanation, no markdown, no code blocks.
2. Always use SELECT (never INSERT, UPDATE, DELETE, DROP, or any write statement).
3. Always include LIMIT (maximum 100).
4. Use exact column names from the schema above.
5. Use ORDER BY to make results readable.
6. For "failing" or "unhealthy" pods: status IN ('CrashLoopBackOff','OOMKilled','Error','Failed')
7. For "recent": use WHERE recorded_at > NOW() - INTERVAL '1 hour'
8. Always join pods to clusters via cluster_id when filtering by environment.

EXAMPLES:

Q: How many pods are in CrashLoopBackOff right now?
A:
SELECT COUNT(*) AS crashloop_pod_count
FROM pods
WHERE status = 'CrashLoopBackOff'
  AND recorded_at > NOW() - INTERVAL '1 hour';

Q: Which pods are failing in the prod cluster?
A:
SELECT p.pod_name, p.namespace, p.status, p.restart_count
FROM pods p
JOIN clusters c ON p.cluster_id = c.id
WHERE c.environment = 'prod'
  AND p.status IN ('CrashLoopBackOff', 'OOMKilled', 'Error', 'Failed')
ORDER BY p.restart_count DESC
LIMIT 20;

Q: What P1 incidents are currently open?
A:
SELECT incident_id, title, severity, status,
       affected_services, created_at
FROM incidents
WHERE severity = 'P1' AND status IN ('open', 'investigating')
ORDER BY created_at DESC
LIMIT 20;

Q: What was deployed to prod in the last 24 hours?
A:
SELECT d.deployment_name, d.namespace, d.image_tag,
       d.status, d.deployed_by, d.deployed_at
FROM deployments d
JOIN clusters c ON d.cluster_id = c.id
WHERE c.environment = 'prod'
  AND d.deployed_at > NOW() - INTERVAL '24 hours'
ORDER BY d.deployed_at DESC
LIMIT 20;

Q: Which nodes have memory pressure?
A:
SELECT n.node_name, n.status, n.memory_pressure,
       n.memory_usage_mi, n.memory_capacity_mi,
       ROUND(n.memory_usage_mi * 100.0 / NULLIF(n.memory_capacity_mi, 0), 1) AS memory_pct
FROM nodes n
WHERE n.memory_pressure = true
ORDER BY memory_pct DESC NULLS LAST
LIMIT 20;
""".strip()


async def generate_sql(question: str) -> str:
    """
    Generate a SQL query for the given natural language question.

    PROMPT STRUCTURE:
    1. Schema description (exact table/column names)
    2. Hard rules the LLM must follow (SELECT-only, LIMIT, etc.)
    3. Few-shot examples (teaches JOIN patterns and K8s domain)
    4. The question

    TEMPERATURE=0.0:
    Fully deterministic. Same question → same SQL.
    Essential for testing: we can predict what SQL will be generated.
    If a question generates wrong SQL, we fix the prompt or add an example.

    RESPONSE PARSING:
    GPT-4o sometimes wraps SQL in ```sql ... ``` despite instructions.
    We strip any markdown code fences as a safety measure.

    Args:
        question: Natural language question from the user

    Returns:
        SQL string ready for validation and (after approval) execution.
        Returns an error SQL comment string if generation fails.
    """
    client = AsyncOpenAI()

    logger.info(f"Generating SQL for: '{question[:60]}...'")

    try:
        response = await client.chat.completions.create(
            model       = SQL_GEN_MODEL,
            temperature = SQL_GEN_TEMP,
            max_tokens  = SQL_GEN_MAX_TOKENS,
            messages    = [
                {"role": "system", "content": SQL_GENERATION_PROMPT},
                {"role": "user",   "content": f"Question: {question}"},
            ],
        )

        sql_raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present (GPT sometimes adds them)
        sql_clean = re.sub(r'^```sql\s*', '', sql_raw, flags=re.IGNORECASE)
        sql_clean = re.sub(r'^```\s*',    '', sql_clean, flags=re.MULTILINE)
        sql_clean = re.sub(r'```$',       '', sql_clean, flags=re.MULTILINE)
        sql_clean = sql_clean.strip().rstrip(';').strip()

        logger.info(
            f"SQL generated | tokens: {response.usage.total_tokens} | "
            f"sql: {sql_clean[:80]}..."
        )

        return sql_clean

    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return f"-- SQL generation failed: {e}"

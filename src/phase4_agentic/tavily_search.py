# src/phase4_agentic/tavily_search.py
#
# Tavily web search integration for CRAG fallback.
#
# WHEN THIS RUNS:
# Only when CRAG grades retrieval as "ambiguous" or "incorrect".
# NOT on every query — only when the knowledge base can't answer.
#
# WHY TAVILY OVER GOOGLE/BING API?
# Tavily is purpose-built for RAG:
#   - Returns clean, parsed text (not raw HTML to parse yourself)
#   - Provides relevance scores per result
#   - Respects robots.txt and content policies
#   - Free tier: 1000 searches/month
#   - Results are pre-filtered for factual content
#
# QUERY CONSTRUCTION FOR KUBERNETES:
# We pin queries to the cluster's k8s_version to prevent version mismatch.
# A question about "pod autoscaling" with a 1.27 cluster should not
# retrieve docs about 1.30 features that don't exist yet.
#
# MERGE STRATEGY:
# "correct"   → use local chunks only (Tavily not called)
# "ambiguous" → merge: local chunks + Tavily results (diversity)
# "incorrect" → replace: Tavily results only (local was useless)
#
# CONTEXT WINDOW MANAGEMENT:
# Tavily results are truncated to MAX_RESULT_CHARS per result.
# We take top-3 results to stay within context window budget.

import logging
import os
from typing import Optional

from src.module2_system_arch.state import RAGState

logger = logging.getLogger(__name__)

MAX_RESULTS      = 3      # Tavily results to use
MAX_RESULT_CHARS = 800    # chars per result (rough: 200 tokens)
K8S_VERSION_DEFAULT = "1.29"


async def fetch_tavily_results(
    question:    str,
    k8s_version: str = K8S_VERSION_DEFAULT,
) -> list[str]:
    """
    Search Tavily and return formatted result strings.

    QUERY CONSTRUCTION:
    We add "Kubernetes {version}" prefix to the question to:
    1. Keep results in the Kubernetes domain
    2. Pin to the cluster's version (prevents stale docs)
    3. Improve result relevance for platform engineering questions

    GRACEFUL DEGRADATION:
    If TAVILY_API_KEY is not set: logs a warning, returns empty list.
    The calling node handles empty results safely.
    If Tavily returns an error: same — empty list, warning logged.
    This means Tavily failure never crashes the pipeline.

    Returns:
        List of formatted strings, each prefixed with [WEB SOURCE: url]
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning(
            "TAVILY_API_KEY not set — skipping web search fallback. "
            "Set this in your .env file to enable CRAG web search."
        )
        return []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)

        # Pin query to Kubernetes version
        query = f"Kubernetes {k8s_version} {question}"
        logger.info(f"Tavily search: '{query[:60]}...'")

        response = client.search(
            query              = query,
            max_results        = MAX_RESULTS,
            search_depth       = "basic",  # "advanced" costs 2x credits
            include_raw_content= False,    # parsed text only (cleaner)
        )

        results = []
        for result in response.get("results", []):
            url     = result.get("url", "unknown")
            content = result.get("content", "")
            score   = result.get("score", 0.0)

            if not content:
                continue

            # Truncate to stay within context window budget
            if len(content) > MAX_RESULT_CHARS:
                content = content[:MAX_RESULT_CHARS] + "..."

            # Format with source attribution
            results.append(
                f"[WEB SOURCE: {url} | relevance={score:.3f}]\n{content}"
            )

        logger.info(f"Tavily returned {len(results)} results for: '{question[:40]}...'")
        return results

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []


async def tavily_search_node(state: RAGState) -> dict:
    """
    LangGraph node: fetch Tavily results and merge with local context.

    MERGE STRATEGY based on retrieval_grade:
    "ambiguous" → local chunks kept + Tavily appended (diversity wins)
    "incorrect" → local chunks discarded + only Tavily used (quality wins)

    WHY KEEP AMBIGUOUS LOCAL RESULTS?
    Local chunks are from our curated knowledge base (runbooks, postmortems).
    Even at 0.70 score they may contain team-specific configurations
    that no public web page would have. Tavily supplements, not replaces.

    NODE CONTRACT:
      Reads:   state["question"], state["context"], state["retrieval_grade"]
      Writes:  {"context": list[str], "sources": list[str],
                "tavily_results": list[str]}
      Returns: partial state update
    """
    question        = state["question"]
    retrieval_grade = state.get("retrieval_grade", "incorrect")
    local_context   = state.get("context", [])
    k8s_version     = K8S_VERSION_DEFAULT  # Phase 5: read from SQL cluster data

    tavily_results = await fetch_tavily_results(question, k8s_version)

    # Build merged context based on grade
    if retrieval_grade == "ambiguous":
        # Keep local chunks (may have team-specific info) + add web results
        merged_context = local_context + tavily_results
        logger.info(
            f"CRAG merge (ambiguous): {len(local_context)} local + "
            f"{len(tavily_results)} web = {len(merged_context)} total"
        )
    else:
        # "incorrect": local chunks were bad, use only web results
        # If no web results either (no API key), fall back to local anyway
        merged_context = tavily_results if tavily_results else local_context
        logger.info(
            f"CRAG replace (incorrect): {len(local_context)} local discarded, "
            f"using {len(tavily_results)} web results"
        )

    # Update sources to include web provenance
    sources = state.get("sources", [])
    if tavily_results:
        sources = list(sources) + ["tavily_web_search"]

    return {
        "context":        merged_context,
        "sources":        sources,
        "tavily_results": tavily_results,
    }

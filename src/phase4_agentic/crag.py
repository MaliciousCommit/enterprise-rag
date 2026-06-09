# src/phase4_agentic/crag.py
#
# CRAG: Corrective Retrieval-Augmented Generation
# Paper: "Corrective Retrieval Augmented Generation" (Shi et al., 2024)
# https://arxiv.org/abs/2401.15884
#
# THE CORE PROBLEM CRAG SOLVES:
# Standard RAG blindly passes retrieved chunks to the LLM regardless of quality.
# If the retrieval fails (low scores, off-topic chunks), the LLM either
# halluccinates or says "I don't know" — both are bad outcomes.
#
# CRAG ADDS A QUALITY GATE:
# After retrieval, grade the results. If quality is poor, CORRECT the
# retrieval by fetching fresh, relevant content from the web (Tavily).
# Only then generate the answer.
#
# THREE-TIER GRADING:
# ┌─────────────────────┬──────────────┬──────────────────────────┐
# │ Best cosine score   │ Grade        │ Action                   │
# ├─────────────────────┼──────────────┼──────────────────────────┤
# │ ≥ 0.80              │ CORRECT      │ Use local chunks directly │
# │ 0.65 - 0.80         │ AMBIGUOUS    │ Use local + Tavily hybrid │
# │ < 0.65              │ INCORRECT    │ Discard local, use Tavily │
# └─────────────────────┴──────────────┴──────────────────────────┘
#
# WHY USE SCORES RATHER THAN LLM GRADING?
# Option A (scores): 0ms, no API cost, uses cosine similarity threshold.
#   Weakness: cosine score is not always a perfect proxy for relevance.
# Option B (LLM grader): ~150ms, ~$0.0001, LLM reads each chunk and
#   rates: "Is this chunk relevant to the question? Yes/No/Partial"
#   More accurate but adds latency.
#
# We implement Option A (score-based) for speed, with a note that
# production systems often use Option B for better precision.
#
# KUBERNETES-SPECIFIC CAVEAT (from Module 1 Socratic Q5):
# Tavily queries are pinned to the cluster's k8s_version to prevent
# retrieving docs for the wrong Kubernetes version.
# "Kubernetes 1.29 OOMKilled" not just "OOMKilled"

import logging
from src.module2_system_arch.state import RAGState

logger = logging.getLogger(__name__)

# CRAG thresholds
CRAG_CORRECT_THRESHOLD   = 0.80   # definitely good — use local only
CRAG_AMBIGUOUS_THRESHOLD = 0.65   # maybe good — use local + web
# below 0.65: INCORRECT — discard local, use web only


def grade_retrieval(state: RAGState) -> str:
    """
    Grade the quality of retrieved chunks.

    Returns one of:
      "correct"   — local retrieval is reliable
      "ambiguous" — local retrieval may be partial
      "incorrect" — local retrieval is unreliable

    This is a synchronous, zero-cost operation (uses pre-computed scores).
    """
    scores = state.get("scores", [])

    if not scores:
        logger.warning("CRAG: no retrieval scores found → incorrect")
        return "incorrect"

    best_score = max(scores)
    avg_score  = sum(scores) / len(scores)

    if best_score >= CRAG_CORRECT_THRESHOLD:
        grade = "correct"
    elif best_score >= CRAG_AMBIGUOUS_THRESHOLD:
        grade = "ambiguous"
    else:
        grade = "incorrect"

    logger.info(
        f"CRAG grade: {grade.upper()} | "
        f"best={best_score:.4f} | avg={avg_score:.4f} | "
        f"n_chunks={len(scores)}"
    )

    return grade


def route_by_crag(state: RAGState) -> str:
    """
    LangGraph conditional edge: route based on CRAG retrieval grade.

    Returns a key that maps to a node name in the graph's path_map.

    ROUTING LOGIC:
    "correct"   → skip Tavily, go directly to generation
    "ambiguous" → fetch Tavily to supplement (handled in tavily_search_node)
    "incorrect" → fetch Tavily to replace (handled in tavily_search_node)
    """
    grade = state.get("retrieval_grade", "incorrect")

    if grade == "correct":
        return "good_retrieval"     # → generate directly
    else:
        return "poor_retrieval"     # → tavily_search → generate


async def crag_grader_node(state: RAGState) -> dict:
    """
    LangGraph node: grade retrieval quality and set routing signal.

    NODE CONTRACT:
      Reads:   state["scores"], state["context"]
      Writes:  {"retrieval_grade": str}
      Returns: partial state update

    This node executes in ~1ms (no API calls).
    It gates the expensive Tavily API call behind a quality check.
    """
    grade = grade_retrieval(state)

    return {"retrieval_grade": grade}

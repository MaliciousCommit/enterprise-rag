# src/phase4_agentic/self_rag.py
#
# Self-RAG: Self-Reflective Retrieval-Augmented Generation
# Paper: "Self-RAG: Learning to Retrieve, Generate, and Critique through
#         Self-Reflection" (Asai et al., 2023) https://arxiv.org/abs/2310.11511
#
# THE CORE INSIGHT:
# Standard RAG generates an answer and returns it — no quality check.
# The LLM might produce:
#   - A well-grounded, complete answer (great)
#   - A partially grounded answer with one hallucinated claim (dangerous)
#   - A truthful but incomplete answer that misses key steps (unhelpful)
#
# Self-RAG adds a critique step: after generating, the LLM reads its
# OWN answer alongside the question and context, then scores:
#   "Did I actually answer this from the context? Is my answer complete?"
#
# IF QUALITY IS POOR (score < threshold):
#   - The LangGraph graph CYCLES BACK to retrieve_node
#   - With a refined query (optional): make the second retrieval more targeted
#   - A loop guard (iteration < max_iterations) prevents infinite loops
#
# THE CYCLE IN LANGGRAPH:
# This is the most important architectural feature of Phase 4.
# Plain async Python cannot express "loop back to retrieve if needed"
# without complex state management. In LangGraph: one conditional edge.
#
# SCORING DIMENSIONS:
# We ask the LLM to score three things (each 0-1):
#   faithfulness: Is every claim in the answer supported by the context?
#   completeness: Does the answer address all parts of the question?
#   coherence:    Is the answer clear and well-structured?
#
# Final score = weighted average:
#   0.5 × faithfulness + 0.3 × completeness + 0.2 × coherence
# (faithfulness weighted highest — hallucination is the worst failure)
#
# REGENERATION STRATEGY:
# On second attempt, we add a hint to the retrieval that the first
# attempt was insufficient — this can shift which chunks get retrieved.
#
# LOOP GUARD:
# iteration >= MAX_ITERATIONS → always accept, even with low score.
# This prevents infinite loops if the knowledge base genuinely can't
# answer the question (we don't want to loop forever in that case).

import json
import logging
from openai import AsyncOpenAI

from src.module2_system_arch.state import RAGState

logger = logging.getLogger(__name__)

MAX_ITERATIONS     = 2       # maximum regeneration attempts
QUALITY_THRESHOLD  = 0.80    # below this: regenerate if iteration < max
SELF_RAG_MODEL     = "gpt-4o-mini"   # use mini for cost efficiency
SELF_RAG_TEMP      = 0.0     # deterministic scoring


SELF_RAG_PROMPT = """
You are evaluating the quality of a Kubernetes operations answer.
Score the answer on three dimensions (each 0.0 to 1.0):

faithfulness: Is every factual claim in the answer directly supported
              by the provided context? No hallucinated information?

completeness: Does the answer address ALL parts of the question?
              Missing steps or partial answers score lower.

coherence:    Is the answer clear, well-structured, and actionable?

Question: {question}

Context used:
{context}

Answer generated:
{answer}

Respond with ONLY valid JSON, no markdown, no explanation:
{{"faithfulness": 0.0, "completeness": 0.0, "coherence": 0.0, "reason": "brief explanation"}}
""".strip()


async def score_answer(
    question: str,
    context:  list[str],
    answer:   str,
) -> dict:
    """
    Ask the LLM to score the quality of its own answer.

    Returns a dict with:
      score:        float 0.0-1.0 (weighted composite)
      faithfulness: float 0.0-1.0
      completeness: float 0.0-1.0
      coherence:    float 0.0-1.0
      reason:       str (brief explanation)

    GRACEFUL DEGRADATION:
    If the LLM returns malformed JSON or the call fails,
    we return a score of 1.0 (accept the answer).
    Reason: a failing reflection is better than crashing the pipeline.
    """
    client = AsyncOpenAI()

    # Truncate context for the reflection call (we don't need full chunks)
    context_preview = "\n\n".join(c[:300] + "..." if len(c) > 300 else c
                                  for c in context[:3])

    prompt = SELF_RAG_PROMPT.format(
        question = question,
        context  = context_preview,
        answer   = answer,
    )

    try:
        response = await client.chat.completions.create(
            model       = SELF_RAG_MODEL,
            temperature = SELF_RAG_TEMP,
            max_tokens  = 200,
            messages    = [{"role": "user", "content": prompt}],
        )

        raw = response.choices[0].message.content.strip()

        # Strip any accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        scores = json.loads(raw)

        faithfulness = float(scores.get("faithfulness", 1.0))
        completeness = float(scores.get("completeness", 1.0))
        coherence    = float(scores.get("coherence",    1.0))
        reason       = scores.get("reason", "")

        # Weighted composite score
        composite = 0.5 * faithfulness + 0.3 * completeness + 0.2 * coherence

        logger.info(
            f"Self-RAG score: {composite:.3f} | "
            f"faithful={faithfulness:.2f} complete={completeness:.2f} coherent={coherence:.2f} | "
            f"reason: {reason[:80]}"
        )

        return {
            "score":        composite,
            "faithfulness": faithfulness,
            "completeness": completeness,
            "coherence":    coherence,
            "reason":       reason,
        }

    except Exception as e:
        logger.warning(f"Self-RAG scoring failed: {e}. Accepting answer.")
        return {
            "score":        1.0,
            "faithfulness": 1.0,
            "completeness": 1.0,
            "coherence":    1.0,
            "reason":       f"scoring_failed: {e}",
        }


async def self_rag_reflect_node(state: RAGState) -> dict:
    """
    LangGraph node: score the generated answer and set routing signal.

    NODE CONTRACT:
      Reads:   state["question"], state["context"], state["answer"],
               state["iteration"]
      Writes:  {"self_rag_score": float, "retrieval_grade": str (reset)}
      Returns: partial state update

    LOOP GUARD:
    If iteration >= MAX_ITERATIONS: skip scoring, always accept.
    This fires regardless of quality — we can't loop forever.

    ROUTING SIGNAL:
    self_rag_score is read by route_self_rag() (conditional edge function).
    Setting it here causes the next edge evaluation to route correctly.
    """
    iteration = state.get("iteration", 1)
    answer    = state.get("answer", "")
    question  = state["question"]
    context   = state.get("context", [])

    # Loop guard: don't score on last allowed iteration
    if iteration >= MAX_ITERATIONS:
        logger.info(
            f"Self-RAG: iteration={iteration} >= max={MAX_ITERATIONS}, "
            "accepting answer without scoring"
        )
        return {"self_rag_score": 1.0}

    # Don't score fallback responses (they're already known to be low quality)
    fallback_phrases = ["i do not have documentation", "escalate to the platform"]
    if any(p in answer.lower() for p in fallback_phrases):
        logger.info("Self-RAG: fallback response detected, score=0.0, will regenerate")
        return {"self_rag_score": 0.0}

    # Score the answer
    scores = await score_answer(question, context, answer)

    return {"self_rag_score": scores["score"]}


def route_self_rag(state: RAGState) -> str:
    score     = state.get("self_rag_score", 1.0)
    iteration = state.get("iteration", 1)
    intent    = state.get("intent", "rag")

    # SQL questions: never regenerate via RAG path.
    # The regenerate branch routes to retrieve_node (vector search + BM25).
    # Vector search cannot answer "which pods are failing in your cluster" —
    # that requires live PostgreSQL data, which only the SQL pipeline provides.
    # Sending a SQL question through RAG retrieval always produces a worse answer.
    if intent == "sql":
        logger.info(
            f"Self-RAG: ACCEPT (intent=sql — RAG regeneration cannot improve SQL answers)"
        )
        return "accept"

    if score >= QUALITY_THRESHOLD or iteration >= MAX_ITERATIONS:
        logger.info(f"Self-RAG: ACCEPT (score={score:.3f}, iteration={iteration})")
        return "accept"
    else:
        logger.info(
            f"Self-RAG: REGENERATE (score={score:.3f} < {QUALITY_THRESHOLD}, "
            f"iteration={iteration})"
        )
        return "regenerate"

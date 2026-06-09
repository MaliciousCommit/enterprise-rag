# src/phase3_retrieval/hyde.py
#
# HyDE: Hypothetical Document Embeddings.
# Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
# Gao et al., 2022. https://arxiv.org/abs/2212.10496
#
# THE PROBLEM HyDE SOLVES:
# Standard dense search embeds the QUESTION and finds similar ANSWERS.
# But questions and answers live in different semantic spaces:
#
#   Question: "Why does my pod keep restarting?"
#   Answer text: "CrashLoopBackOff occurs when a container repeatedly fails..."
#
# The question embedding captures "restart" semantics.
# The answer embedding captures "CrashLoopBackOff" semantics.
# These are related but NOT identical — there's a semantic gap.
#
# HyDE BRIDGES THE GAP:
# Instead of embedding the question, embed a HYPOTHETICAL ANSWER.
# A hypothetical answer lives in the same semantic space as real answers.
#
#   Question: "Why does my pod keep restarting?"
#   HyDE generates: "Pods restart repeatedly when the container process exits
#                    with a non-zero code. This is called CrashLoopBackOff.
#                    Common causes include: misconfigured command, missing env
#                    variables, or OOMKilled..."
#
# The hypothetical answer embedding is MUCH closer to the real runbook text.
# Retrieval improves significantly for vague or broad questions.
#
# WHY GENERATE MULTIPLE HYPOTHETICAL ANSWERS (n=3)?
# A single hypothetical may take one angle on the topic.
# 3 hypotheticals cover multiple angles.
# Averaging their embeddings creates a more stable, central embedding
# that represents the question's semantic neighbourhood more completely.
#
# WHEN HyDE HELPS:
#   Good for: vague questions ("what should I do about pod failures?")
#             questions using different vocabulary than the docs
#             questions about concepts (not exact names)
#
#   Less helpful for: questions with exact pod/namespace names
#                    ("show me payment-svc logs") — BM25 handles these
#
# TEMPERATURE=0.7:
# We use higher temperature than generation (0.1) intentionally.
# Diversity in hypotheticals → better coverage of the answer space.
# We don't care if a hypothetical is "correct" — we care if it's
# semantically in the right neighbourhood.

import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

HYDE_SYSTEM_PROMPT = """
You are a Kubernetes platform engineer writing a technical runbook entry.
Given a question, write a concise technical answer (3-5 sentences) as if
it were an excerpt from an official Kubernetes operations guide.
Focus on technical accuracy and use Kubernetes-specific terminology.
Do not say "I" or acknowledge the question — just write the answer directly.
""".strip()

NUM_HYPOTHETICALS = 3   # number of hypothetical answers to generate
HYDE_TEMPERATURE  = 0.7  # higher than generation — we want diversity
HYDE_MAX_TOKENS   = 200  # answers should be concise, not full runbooks


async def generate_hyde_embedding(question: str) -> list[float]:
    """
    Generate N hypothetical answers and return their averaged embedding.

    PIPELINE:
    1. Generate N hypothetical Kubernetes runbook entries for the question
    2. Embed each hypothetical answer (not the question)
    3. Average the N embeddings component-wise
    4. Return the averaged embedding for dense vector search

    The averaged embedding represents the centroid of the answer space
    for this question — closer to real answers than the question itself.

    Args:
        question: The user's original question

    Returns:
        Averaged embedding vector (1536-dim for text-embedding-3-small)

    FALLBACK:
    If HyDE generation fails for any reason (rate limit, timeout),
    we fall back to embedding the original question.
    This ensures the pipeline never fails due to HyDE.
    """
    client = AsyncOpenAI()

    # ── Step 1: Generate N hypothetical answers ─────────────────────────────
    try:
        response = await client.chat.completions.create(
            model       = "gpt-4o-mini",   # mini for speed+cost — quality sufficient for HyDE
            temperature = HYDE_TEMPERATURE,
            max_tokens  = HYDE_MAX_TOKENS,
            n           = NUM_HYPOTHETICALS,   # generate N completions in one API call
            messages    = [
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
        )

        hypotheticals = [
            choice.message.content.strip()
            for choice in response.choices
            if choice.message.content
        ]

        logger.debug(
            f"HyDE: generated {len(hypotheticals)} hypotheticals | "
            f"tokens: {response.usage.total_tokens}"
        )

    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}. Falling back to question embedding.")
        hypotheticals = []

    if not hypotheticals:
        # Fallback: embed the question directly (Phase 2 behaviour)
        return await _embed_text_async(question)

    # ── Step 2: Embed each hypothetical ────────────────────────────────────
    embeddings = []
    for hyp_text in hypotheticals:
        try:
            emb = await _embed_text_async(hyp_text)
            embeddings.append(emb)
        except Exception as e:
            logger.warning(f"HyDE embedding failed for one hypothetical: {e}")

    if not embeddings:
        logger.warning("HyDE: all embeddings failed. Falling back to question embedding.")
        return await _embed_text_async(question)

    # ── Step 3: Average the embeddings ────────────────────────────────────
    dim = len(embeddings[0])
    averaged = [
        sum(emb[i] for emb in embeddings) / len(embeddings)
        for i in range(dim)
    ]

    logger.info(
        f"HyDE: {len(embeddings)} embeddings averaged → "
        f"{dim}-dim vector for '{question[:40]}...'"
    )

    return averaged


async def _embed_text_async(text: str) -> list[float]:
    """Async wrapper for OpenAI text embedding."""
    from src.config import settings
    client = AsyncOpenAI()
    response = await client.embeddings.create(
        model = settings.embedding_model,
        input = text,
    )
    return response.data[0].embedding

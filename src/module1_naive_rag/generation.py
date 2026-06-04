# src/module1_naive_rag/generation.py
# LLM generation with grounded prompts, retry, and token tracking.
#
# PHASE EVOLUTION:
# Module 1: Single GPT-4o call with grounded prompt
# Phase 4:  Wrapped in self-RAG reflection loop (score < 0.8 -> regen, max 2x)
# Phase 8:  Output passes through PII redaction + output moderation after generation
# Phase 9:  token usage metrics sent to Prometheus counter

import logging
from dataclasses import dataclass
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from tenacity import (
    retry, retry_if_exception_type,
    stop_after_attempt, wait_exponential, before_sleep_log,
)
from src.config import settings
from src.module1_naive_rag.retrieval import RetrievedChunk, format_context

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────
#
# The system prompt is the most important safety control in Module 1.
# Every word is deliberate.
#
# "ONLY source of truth is the context" -- grounding instruction.
#   Without this, GPT-4o freely mixes parametric knowledge with retrieved context,
#   making answers sound authoritative but potentially fabricated.
#
# "If context does not contain..." -- explicit fallback behavior.
#   Without this, the LLM guesses. With this, it admits ignorance.
#
# "Cite the source document using [doc N]" -- attribution.
#   Enables the user to verify claims. Makes hallucination detectable.
#
# "If multiple documents conflict..." -- conflict resolution instruction.
#   K8s runbooks sometimes have version-specific advice that conflicts.
#   The LLM must acknowledge rather than silently pick one.

SYSTEM_PROMPT = """You are a Kubernetes operations expert assistant for our platform engineering team.

Your ONLY source of truth is the context documents provided to you inside <doc> tags.

STRICT RULES:
1. Answer ONLY from the provided context documents. Zero exceptions.
2. If the context does not contain sufficient information to answer, respond with:
   "I do not have documentation on this topic in the knowledge base. Please check the official Kubernetes documentation or escalate to the platform team at platform@company.com"
3. Always be specific and actionable. Generic advice is not acceptable in production operations.
4. Cite the source document using [doc N] notation when making specific claims.
5. If multiple documents conflict with each other, acknowledge the conflict and cite both.
6. Never use your general training knowledge -- use ONLY the content inside <doc> tags.
7. Format your answer clearly. Use numbered steps for procedures. Use bullet points for lists.""".strip()

# The user message template.
# Context goes BEFORE the question -- this follows the "Lost in the Middle"
# finding: LLMs best retain content at the start of their context window.
# The context is the most important part -- put it first.
USER_PROMPT_TEMPLATE = """Context documents:

{context}

Question: {question}

Answer (cite [doc N] for specific claims):"""


@dataclass
class GenerationResult:
    """
    Result of a single LLM generation call.
    Carries the answer text plus full metadata for observability.

    prompt_tokens + completion_tokens = total_tokens.
    In Phase 6: total_tokens is checked against the user's daily budget
    (L6 Token Budget guardrail: 100k tokens/day/user).
    In Phase 9: prompt_tokens and completion_tokens are emitted as
    Prometheus counters (llm_tokens_total{type="prompt"} etc.)

    COST REFERENCE (GPT-4o as of 2024):
    Input:  $5.00 per million tokens
    Output: $15.00 per million tokens
    """
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str       # "stop" (normal) | "length" (truncated by max_tokens)
    sources: list[str]       # unique source paths from retrieved chunks


_llm_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI()
    return _llm_client


_llm_retry = retry(
    retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@_llm_retry
def _call_llm(messages: list[dict], model: str, temperature: float) -> object:
    """
    Internal: raw OpenAI chat completion call with retry.

    Why separate from generate()?
    The retry decorator wraps ONLY the API call.
    Prompt construction, logging, and result mapping happen outside retry --
    no point retrying those, they don't fail transiently.
    """
    client = _get_client()
    return client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        # max_tokens=2048,    # optional cap on output length
        #                     # without this, GPT-4o outputs up to 4096 tokens
        # stream=False,       # Phase 10 (FastAPI) uses stream=True for SSE
    )


def generate(
    question: str,
    chunks: list[RetrievedChunk],
    model: str | None = None,
    temperature: float | None = None,
) -> GenerationResult:
    """
    Generate a grounded answer using GPT-4o with retrieved context.

    FLOW:
        chunks -> format_context() [XML spotlighting]
               -> build messages   [system + user]
               -> GPT-4o call      [~1,500ms]
               -> GenerationResult

    CONTEXT WINDOW ECONOMICS (Module 1 defaults):
        System prompt:           ~180 tokens
        5 chunks x 350 tokens:  ~1,750 tokens
        Question:                 ~20 tokens
        Safety margin:            ~50 tokens
        Total input:           ~2,000 tokens
        Expected output:         ~300 tokens
        Grand total:           ~2,300 tokens

        Cost: 2,300 x $0.000005 = $0.0115 per query
        At 1,000 queries/day: $11.50/day

        With 60% cache hit rate (Phase 6): $4.60/day

    TEMPERATURE = 0.1 RATIONALE (from Socratic Q4):
        0.0 = greedy decoding -- causes repetition loops in multi-item answers
        0.1 = near-deterministic -- prevents loops with tiny variance
        0.7 = used ONLY for HyDE hypothesis generation (Phase 3), NOT here

    Args:
        question:    User's natural language question
        chunks:      Retrieved chunks (sorted by score, highest first)
        model:       Override LLM model (default: settings.llm_model = "gpt-4o")
        temperature: Override temperature (default: settings.llm_temperature = 0.1)

    Returns:
        GenerationResult with answer text + full token usage metadata
    """
    model = model or settings.llm_model
    temperature = temperature if temperature is not None else settings.llm_temperature

    # Build the spotlighted context string
    context_str = format_context(chunks, use_spotlighting=True)

    # Build messages in OpenAI format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                context=context_str,
                question=question,
            ),
        },
    ]

    logger.info(
        f"Generating answer | model={model} | temp={temperature} | "
        f"{len(chunks)} context chunks | question='{question[:50]}...'"
    )

    # Call GPT-4o (with retry on rate limits / server errors)
    response = _call_llm(messages, model, temperature)

    answer = response.choices[0].message.content
    usage = response.usage
    finish_reason = response.choices[0].finish_reason

    if finish_reason == "length":
        logger.warning(
            "LLM response was TRUNCATED by max_tokens limit. "
            "Consider increasing max_tokens or reducing context size."
        )

    # Compute approximate input cost for logging
    input_cost = usage.prompt_tokens * 5.0 / 1_000_000
    output_cost = usage.completion_tokens * 15.0 / 1_000_000

    logger.info(
        f"Generation complete | "
        f"tokens: {usage.prompt_tokens}+{usage.completion_tokens}={usage.total_tokens} | "
        f"cost: ~${input_cost + output_cost:.4f} | "
        f"finish: {finish_reason}"
    )

    return GenerationResult(
        answer=answer,
        model=response.model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        finish_reason=finish_reason,
        sources=list({c.source for c in chunks}),
    )

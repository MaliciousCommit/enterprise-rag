# src/config.py
#
# The single source of truth for all configuration in the Enterprise RAG system.
#
# WHY A CENTRAL CONFIG MODULE?
# Every other module in this project imports Settings from here.
# This means:
#   1. Config changes require editing ONE file, not hunting through 20 modules
#   2. We can validate all config at startup (fail fast before serving traffic)
#   3. Swap config source without changing any logic (env → Vault → K8s ConfigMap)
#   4. Type annotations give us IDE autocomplete on every setting
#
# DESIGN DECISION: Pydantic BaseSettings vs plain dataclass
# We use a simple dataclass for Module 1 (fewer dependencies, easier to understand).
# From Phase 3 onwards we upgrade to pydantic.BaseSettings which adds:
#   - Automatic type coercion (str "5" → int 5)
#   - Validators (ensure OPENAI_API_KEY starts with "sk-")
#   - Nested settings models
#   - .env file parsing built in
#
# HOW VALUES ARE LOADED:
# os.getenv("KEY", "default") reads from:
#   1. System environment variables
#   2. .env file (loaded by load_dotenv() below)
#   3. Falls back to the default if neither is set
#
# KUBERNETES NOTE:
# In production (Phase 10), these values come from:
#   - Non-secret values: Kubernetes ConfigMap → mounted as env vars
#   - Secret values (API keys): Kubernetes Secret → mounted as env vars
#   - The code is identical either way — just env vars.

import os
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv

# load_dotenv() reads the .env file in the project root and injects
# all KEY=VALUE pairs as environment variables.
# Must be called BEFORE we read os.getenv() calls below.
# If .env doesn't exist, this is a no-op (safe to call always).
load_dotenv()


@dataclass
class Settings:
    """
    All configuration for the Enterprise RAG system.

    Access via the module-level `settings` singleton at the bottom of this file:
        from src.config import settings
        print(settings.qdrant_host)   # "localhost"

    NEVER instantiate Settings() yourself in other modules.
    Always use the singleton — it ensures load_dotenv() has already run.
    """

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    # The API key is read from env. Never hardcode it.
    # In Kubernetes: mounted from a Secret named "openai-credentials".
    # The openai SDK reads OPENAI_API_KEY automatically, but we store it
    # here so we can validate at startup that it's not empty.

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = field(
        default_factory=lambda: os.getenv("QDRANT_HOST", "localhost")
    )
    qdrant_port: int = field(
        default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333"))
    )
    qdrant_api_key: str | None = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY") or None
        # os.getenv returns "" if set to empty string.
        # `or None` converts "" → None, which QdrantClient expects for no auth.
    )

    # ── Qdrant Collection ─────────────────────────────────────────────────────
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "k8s_docs_m1")
    )
    # We namespace the collection name with the module that created it.
    # "k8s_docs_m1" = Kubernetes docs, built in Module 1.
    # Phase 3 adds hybrid search and re-ingests as "k8s_docs_p3".
    # This lets us A/B test retrieval quality between phases.

    # ── Embedding Model ───────────────────────────────────────────────────────
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1536"))
    )
    # CRITICAL: embedding_model and embedding_dim must stay in sync.
    # text-embedding-3-small → 1536 dims  (our choice: cheap, fast, good quality)
    # text-embedding-3-large → 3072 dims  (better quality, 5x more expensive)
    # ada-002 (legacy)       → 1536 dims  (older, slightly lower quality)
    #
    # Once you ingest documents with one model, you CANNOT change models
    # without re-ingesting all documents. The model defines the vector space.

    embed_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBED_BATCH_SIZE", "100"))
    )
    # OpenAI allows up to 2048 inputs per embedding API call.
    # We use 100 as a conservative default to avoid timeout issues.

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o")
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1"))
    )
    # Temperature rationale (from Module 1 Socratic Q4):
    # 0.0 = deterministic but causes repetition loops in multi-item answers
    # 0.1 = nearly deterministic, prevents repetition loops
    # 0.7 = used for HyDE hypothesis generation (Phase 3)
    # We use 0.1 for final answer generation throughout.

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "400"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "80"))
    )
    # chunk_size: Target word count per chunk (not tokens — we approximate).
    # 400 words ≈ 512 tokens for English technical text.
    # The overlap (80 words) ensures sentences at chunk boundaries appear
    # in at least one chunk's full context.
    #
    # Phase 6 (Document Pipeline) will compare strategies:
    # fixed-size vs semantic vs sentence-window vs hierarchical

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_K", "5"))
    )
    # k=5: number of chunks returned to the LLM.
    # Phase 3 changes this: we retrieve k=20 for reranking, then
    # the cross-encoder reranker narrows to the top 5 for the LLM.
    # The LLM always sees 5 chunks — we just get smarter about WHICH 5.

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    # ── Computed properties ───────────────────────────────────────────────────
    def validate(self) -> None:
        """
        Validate critical settings at startup.
        Raises ValueError with a clear message if anything is misconfigured.

        Called once in __post_init__ so misconfiguration fails immediately
        at import time — not 30 minutes into a production job.
        """
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        if not self.openai_api_key.startswith(("sk-", "sk-proj-")):
            raise ValueError(
                f"OPENAI_API_KEY looks invalid: '{self.openai_api_key[:10]}...'"
            )
        if self.embedding_dim not in (1536, 3072):
            raise ValueError(
                f"EMBEDDING_DIM must be 1536 or 3072, got {self.embedding_dim}"
            )
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError(
                f"LLM_TEMPERATURE must be 0.0-2.0, got {self.llm_temperature}"
            )

    def __post_init__(self) -> None:
        """Runs automatically after dataclass __init__. Configures logging."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )


# ── Module-level singleton ────────────────────────────────────────────────────
#
# All other modules import this singleton:
#   from src.config import settings
#
# We don't call settings.validate() here because it would fail for anyone
# who imports config without OPENAI_API_KEY set (e.g., CI/CD syntax checks).
# Instead, validate() is called explicitly in scripts/01_setup.py and
# scripts/02_query.py at the top of main().
settings = Settings()

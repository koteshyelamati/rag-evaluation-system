from __future__ import annotations

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central application settings loaded from environment variables / .env file.

    All variables can be overridden by setting them in the environment or in
    a .env file at the project root (see .env.example for a full reference).
    """

    # ── LLM / Embedding providers ────────────────────────────────────────────
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API key (required for RAG pipeline)")
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key (required for Ragas evaluation)")

    # ── Vector store ─────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = Field(default="./chroma_db", description="Filesystem path for ChromaDB persistence")
    COLLECTION_NAME: str = Field(default="rag_docs", description="ChromaDB collection name")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = Field(default=1000, ge=100, description="Characters per document chunk")
    CHUNK_OVERLAP: int = Field(default=200, ge=0, description="Character overlap between consecutive chunks")
    TOP_K_RESULTS: int = Field(default=5, ge=1, le=20, description="Number of chunks retrieved per query")

    # ── Query cache ───────────────────────────────────────────────────────────
    QUERY_CACHE_MAXSIZE: int = Field(default=256, ge=1, description="Maximum number of cached query results")
    QUERY_CACHE_TTL_SECONDS: float = Field(default=3600.0, ge=0, description="Cache TTL in seconds (0 = no expiry)")

    # ── Observability ─────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO", description="Logging level: DEBUG | INFO | WARNING | ERROR")
    SLOW_REQUEST_THRESHOLD_MS: float = Field(
        default=2000.0,
        ge=0,
        description="Requests slower than this (ms) are logged as warnings",
    )

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def cache_ttl(self) -> Optional[float]:
        """Return TTL as None (no expiry) when set to 0, otherwise the float value."""
        return None if self.QUERY_CACHE_TTL_SECONDS == 0 else self.QUERY_CACHE_TTL_SECONDS


settings = Settings()

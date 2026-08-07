"""Typed, environment-driven application configuration.

All runtime configuration is centralised here so the rest of the codebase never
reaches into ``os.environ`` directly. Values are read from the environment (and a
local ``.env`` file, if present) and validated at load time — a missing or malformed
setting fails fast with a clear error rather than surfacing deep inside a request.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Datastore ---------------------------------------------------------
    database_url: str = Field(
        ...,
        description="PostgreSQL connection URL, e.g. postgresql://user:pass@host:5432/db",
    )

    # --- Providers ---------------------------------------------------------
    cohere_api_key: SecretStr = Field(..., description="API key for Cohere embeddings.")
    google_api_key: SecretStr = Field(..., description="API key for Google Gemini.")

    # --- Model / retrieval tunables ---------------------------------------
    embedding_model: str = Field(
        default="embed-english-v3.0",
        description="Cohere embedding model. Must match embedding_dimension.",
    )
    embedding_dimension: int = Field(
        default=1024,
        description="Vector dimension of the embedding model (embed-english-v3.0 -> 1024).",
    )
    llm_model: str = Field(default="gemini-1.5-flash", description="Gemini chat model.")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    retriever_k: int = Field(default=3, ge=1, le=20, description="Documents to retrieve.")
    collection_name: str = Field(
        default="qa_knowledge_base",
        description="Name of the pgvector collection holding embeddings.",
    )

    # --- Observability -----------------------------------------------------
    log_level: str = Field(default="INFO")

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, value: str) -> str:
        if not value.startswith(("postgresql://", "postgresql+psycopg://", "postgres://")):
            raise ValueError(
                "database_url must be a PostgreSQL URL "
                "(postgresql://... or postgresql+psycopg://...)"
            )
        return value

    @property
    def sqlalchemy_url(self) -> str:
        """Connection URL normalised to the psycopg (v3) driver.

        ``langchain-postgres`` and our SQLAlchemy engine both require the psycopg3
        driver. Users are free to set a plain ``postgresql://`` URL in ``.env``;
        we rewrite the scheme so a driver mismatch can never happen at runtime.
        """
        url = self.database_url
        if url.startswith("postgresql+psycopg://"):
            return url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        # postgres:// (legacy alias)
        return url.replace("postgres://", "postgresql+psycopg://", 1)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Cached so configuration is parsed once per process. Tests clear the cache via
    ``get_settings.cache_clear()`` when they need to inject a different environment.
    """
    return Settings()  # values are populated from the environment / .env

"""
app/core/config.py
─────────────────
Centralised configuration using Pydantic Settings.
All values can be overridden via environment variables or a .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_provider: str = "anthropic"          # openai | anthropic | google
    llm_model: str = "claude-sonnet-4-6"

    # ── API Keys ──────────────────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_provider: str = "openai"       # openai | local
    embedding_model: str = "text-embedding-3-small"

    # ── Vector DB ─────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "trading_prompts"

    # ── Relational DB ─────────────────────────────────────────────────────
    database_url: str = "sqlite:///./trading_rag.db"

    # ── Retrieval ─────────────────────────────────────────────────────────
    default_top_k: int = 3
    min_relevance_score: float = 0.30

    # ── App ───────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()

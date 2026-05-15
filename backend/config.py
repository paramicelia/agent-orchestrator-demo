"""Settings loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq
    groq_api_key: str = ""
    groq_smart_model: str = "llama-3.3-70b-versatile"
    groq_lite_model: str = "llama-3.1-8b-instant"

    # Memory
    memory_backend: str = "chroma"  # "chroma" or "pgvector"
    memory_db_path: str = "./chroma_db"
    memory_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Required when memory_backend == "pgvector".
    # Example: postgresql://agent:agent@localhost:5432/agent_demo
    postgres_dsn: str = ""

    # FastAPI
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "info"

    # Observability (optional)
    langsmith_api_key: str = ""
    langsmith_project: str = "agent-orchestrator-demo"
    langchain_tracing_v2: bool = False


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor."""
    return Settings()

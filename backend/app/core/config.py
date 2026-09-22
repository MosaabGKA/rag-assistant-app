"""Pydantic-settings configuration for the RAG backend.

Resolved at import time from ``backend/.env`` overlaid on sensible defaults.
Persistence metadata from the notebook (``data/vector_store/config.json``) is
read lazily by the services that need it; this module does not depend on it.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
CORPUS_ROOT = PROJECT_ROOT / "corpus"
ENV_FILE = BACKEND_DIR / ".env"
VECTOR_STORE_DIR = BACKEND_DIR / "data" / "vector_store"
VECTOR_STORE_CONFIG = VECTOR_STORE_DIR / "config.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- app --------------------------------------------------------------
    app_name: str = "RAG Assistant API"
    app_version: str = "0.1.0"
    app_description: str = "ITI Final Project — RAG assistant over the e-commerce support corpus"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # -- CORS ---------------------------------------------------------------
    # JSON list of allowed origins; Streamlit runs on 8501.
    enabled_origins: str = '["http://localhost:8501","http://127.0.0.1:8501"]'

    # -- retrieval / embeddings -------------------------------------------
    vector_store_dir: Path = VECTOR_STORE_DIR
    vector_store_config: Path = VECTOR_STORE_CONFIG
    collection: str = "support_docs"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 4
    load_store_at_startup: bool = True

    # -- generation ---------------------------------------------------------
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: int = 90

    # -- corpus ------------------------------------------------------------------
    corpus_dir: Path = CORPUS_ROOT
    n_documents: int = 0
    n_chunks: int = 0

    @property
    def cors_origins(self) -> list[str]:
        import json

        try:
            return [o.strip() for o in json.loads(self.enabled_origins)]
        except Exception:
            return ["http://localhost:8501", "http://127.0.0.1:8501"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

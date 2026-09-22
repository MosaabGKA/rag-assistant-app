"""Wrapper for calling the RAG backend API.

The backend URL is read from the ``API_BASE_URL`` environment variable,
honoring ``frontend/.env`` via python-dotenv when present. A Streamlit
``.streamlit/secrets.toml`` value overrides it if set. It is never
hard-coded in the app code — url resolution is centralized here.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

try:  # python-dotenv is optional (e.g. minimal CLI installs)
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv listed in requirements
    load_dotenv = None

_FRONTEND_DIR = Path(__file__).resolve().parent
if load_dotenv:
    load_dotenv(_FRONTEND_DIR / ".env")

API_BASE_URL = os.environ.get("API_BASE_URL") or "http://127.0.0.1:8000"

try:  # Streamlit secrets take precedence when running under Streamlit.
    import streamlit as st

    API_BASE_URL = st.secrets.get("API_BASE_URL") or API_BASE_URL
except Exception:  # pragma: no cover - no streamlit runtime (CLI/tests)
    pass


class APIError(RuntimeError):
    """Raised when the backend is unreachable or returns an error status."""


def get_backend_url() -> str:
    """Return the effective backend base URL."""
    return API_BASE_URL


def health(timeout: float = 3) -> dict:
    """GET /health — ``{}`` on any failure (never raises)."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def ask(question: str, timeout: float = 300) -> dict:
    """POST /query — returns the parsed response or raises ``APIError``."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        raise APIError(f"Backend unreachable at {API_BASE_URL}: {exc}") from exc
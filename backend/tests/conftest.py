"""Shared test fixtures — hermetic: no Chroma, no Ollama.

``app.main`` wires its global store lazily via ``Retriever``/``Generator``. We
override those constructor paths with fakes through ``Settings.load_store_at_startup
=False`` … but the FastAPI package wires globals at import. Simpler, honest
approach: build a tiny FastAPI app that reuses the ROUTER (the part that
matters) with faked services, via TestClient + dependency override resolved
from ``app.main.app.dependency_overrides``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def store_available() -> dict:
    return {"collection": "support_docs", "n_documents": 29, "n_chunks": 4741}

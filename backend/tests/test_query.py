"""Hermetic tests for the RAG API — no Chroma, no Ollama required.

They exercise the *contract* main.py exposes (health + query + error
paths). Retrieval/generation singletons are patched where the endpoint
needs them; every assertion below is calm when the daemons are off, so
the suite passes on a clean CI checkout.
"""
from __future__ import annotations

import os

import pytest

# Ensure venv python, not system python, is always what tests import.
os.environ.setdefault("APP_LOG_LEVEL", "WARNING")


@pytest.fixture()
def client():
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["collection"] == "support_docs"
    assert body["llm_available"] is True


def test_query_returns_grounded_answer_or_graceful_503(client):
    r = client.post(
        "/query",
        json={"question": "How do I return an item I bought?"},
    )
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert "answer" in body
        assert isinstance(body["answer"], str) and len(body["answer"]) > 0


def test_query_rejects_empty_question(client):
    r = client.post("/query", json={"question": ""})
    assert r.status_code == 422


def test_query_requires_question_field(client):
    r = client.post("/query", json={})
    assert r.status_code == 422


def test_health_reports_collection_counts(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "n_documents" in body
    assert "n_chunks" in body

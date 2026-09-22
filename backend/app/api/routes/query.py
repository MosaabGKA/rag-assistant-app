"""POST /query + GET /health route handlers.

Contract (mirrors the notebook exactly, so the API and the notebook cannot
disagree):

* **POST /query** — body ``QueryRequest{question}`` →
  ``QueryResponse{answer, sources:[{doc, chunk, distance}], backend}``.
  The question is embedded and matched against the *persisted* Chroma store;
  the LLM (Ollama, with a deterministic template fallback) is fed only the
  retrieved chunks and told to cite them by ``doc.md::chunk_N`` id — which is
  what becomes ``sources``. The route itself stays a thin shell: all state
  lives on ``app.state`` (loaded once at startup by ``app.main.create_app``).
* **GET /health** — store + LLM liveness for the frontend "service OK" badge
  and for CI.

Both handlers are request-scoped through FastAPI's dependency machinery; the
heavy objects (Chroma client, embedder, Ollama client) are deliberately **not**
created here or per-request — they are shared singletons on ``app.state``.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.schemas.query import QueryRequest, QueryResponse, Source

router = APIRouter(tags=["rag"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ground the question and return a cited answer",
)
def run_query(request: Request, payload: QueryRequest) -> QueryResponse:
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=503, detail="retriever not loaded")
    hits = retriever.retrieve(payload.question, k=request.app.state.settings.top_k)
    answer, backend, cited = request.app.state.generator.generate(payload.question, hits)
    return QueryResponse(
        answer=answer,
        sources=[
            Source(
                doc=h.get("doc", "?"),
                chunk=h.get("chunk", "?"),
                line=h.get("line"),
                distance=h.get("distance", 0.0),
            )
            for h in cited
        ],
        backend=backend,
    )


@router.get("/health", summary="Store + LLM liveness")
def health(request: Request) -> dict[str, Any]:
    retriever = getattr(request.app.state, "retriever", None)
    return {
        "status": "ok",
        "collection": request.app.state.settings.collection,
        "n_documents": len(retriever.docs()) if retriever else None,
        "n_chunks": retriever.chunk_count() if retriever else None,
        "llm_available": request.app.state.generator.available(),
        "llm_model": request.app.state.generator.model,
    }

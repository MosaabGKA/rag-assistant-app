"""FastAPI application entrypoint.

Wires the whole RAG pipeline on top of FastAPI:

* **Startup (lifespan)** — build the retriever + generator once, cache them in
  ``app.state``, share across threads/processes; if the persisted store is
  missing we boot anyway and report it via ``/health`` instead of 500ing.
* **POST /query** — ``{question}`` -> ``{answer, sources, backend}`` using the
  shared retriever + generator.
* **GET /health** — store + LLM liveness (the frontend's "service OK" badge
  and CI).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import query as query_router
from app.core.config import get_settings
from app.schemas.query import HealthResponse, QueryRequest, QueryResponse, Source
from app.services.generation import Generator
from app.services.retrieval import Retriever
from app.utils.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)

_generator = Generator()
_retriever: Retriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store + LLM connection ONCE at startup (not per request).

    A failure is logged, not raised: if the persisted store is missing we boot
    anyway and report it via ``/health`` instead of 500ing.
    """
    global _retriever
    if settings.load_store_at_startup:
        try:
            _retriever = Retriever()
            _retriever.load()
            logger.info("retriever ready: %d chunks", _retriever.n_chunks())
        except Exception as exc:  # pragma: no cover - env-dependent
            logger.warning("store unavailable at startup (will report via /health): %s", exc)
            _retriever = None
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="RAG assistant API",
    lifespan=lifespan,
)

if settings.enabled_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.enabled_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(query_router.router, prefix="/api")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    store = _retriever.store_metrics() if _retriever else {}
    return HealthResponse(
        status="ok",
        collection=settings.collection,
        n_documents=store.get("n_documents"),
        n_chunks=store.get("n_chunks"),
        llm_available=_generator.available(),
        llm_model=_generator.model if _generator.available() else None,
    )


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(payload: QueryRequest) -> QueryResponse:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="retriever not loaded")
    hits = _retriever.retrieve(payload.question, k=settings.top_k)
    answer, backend, cited = _generator.generate(payload.question, hits)
    return QueryResponse(
        answer=answer,
        sources=[
            Source(doc=h["doc"], chunk=h["chunk"], line=h.get("line"), distance=h["distance"])
            for h in cited
        ],
        backend=backend,
    )

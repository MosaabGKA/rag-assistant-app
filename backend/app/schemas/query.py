"""API request/response models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1, max_length=1000, description="The customer's question"
    )


class Source(BaseModel):
    doc: str = Field(description="source document id, e.g. 07_check_refund_policy.md")
    chunk: str = Field(description="retrieved chunk id, e.g. chunk_47")
    line: int | None = Field(default=None, description="1-based start line in the source file")
    distance: float = Field(description="cosine distance of the retrieved chunk")


class QueryResponse(BaseModel):
    answer: str = Field(description="grounded answer from the LLM")
    sources: list[Source] = Field(default_factory=list)
    backend: str = Field(
        description="generation backend used (ollama | template-fallback)"
    )


class HealthResponse(BaseModel):
    status: str
    collection: str | None = None
    n_documents: int | None = None
    n_chunks: int | None = None
    llm_available: bool
    llm_model: str | None = None

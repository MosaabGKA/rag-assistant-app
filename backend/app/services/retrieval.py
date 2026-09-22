"""Top-k retrieval over the persisted Chroma store.

Contract consumed by app.main (zero-arg construction):

    r = Retriever()          # no settings injected
    r.load()                 # ValueError-safe; never raises
    r.loaded  -> bool
    r.n_chunks()     -> int  (METHOD)
    r.n_documents()  -> int  (METHOD)
    r.store_metrics() -> {"n_chunks","n_documents","collection"}
    r.retrieve(question, k=settings.top_k) -> [{"doc","chunk","text","distance","id"}]
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("app.retrieval")

_BACKEND = Path(__file__).resolve().parent.parent.parent
_STORE = _BACKEND / "data" / "vector_store"
_DOCS = _BACKEND.parent / "data" / "documents"
_ENC = "all-MiniLM-L6-v2"
_COL = "support_docs"


def _read_source(path: Path) -> str | None:
    """Re-extract a source document exactly like the ingestion notebook."""
    if not path.exists():
        return None
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _start_line(chunks: list[tuple[int, str]], text: str) -> dict[int, int]:
    """Map chunk ids -> 1-based start line in ``text``.

    Chunks overlap (200 chars), so each next chunk starts *inside* the
    previous one; scanning with a forwarded cursor from the previous start
    reliably finds each chunk's true beginning.
    """
    held = sorted(chunks, key=lambda c: c[0])
    out: dict[int, int] = {}
    cursor = 0
    for cid, chunk in held:
        n = text.find(chunk, cursor)
        if n < 0:
            n = text.find(chunk)
            if n < 0:
                continue
        out[cid] = text.count("\n", 0, n) + 1
        cursor = n + 1
    return out


class Retriever:
    def __init__(self, settings: Any | None = None) -> None:
        from app.core.config import get_settings

        if settings is None:
            settings = get_settings()
        self.settings = settings
        store_dir = Path(getattr(settings, "vector_store_dir", _STORE))
        if not store_dir.is_absolute():
            store_dir = _BACKEND / store_dir
        self.store_dir = store_dir
        self.collection_name = str(getattr(settings, "collection", _COL))
        self.embedding_model = str(
            getattr(settings, "embedding_model", None) or _ENC
        )
        self.top_k = max(1, int(getattr(settings, "top_k", 4)))
        self._client = None
        self._col = None
        self._emb = None
        self.loaded = False
        self._chunk_cnt = 0
        self._doc_cnt = 0
        self._lock = threading.Lock()
        self._lines: dict[str, dict[int, int]] | None = None

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> "Retriever":
        from sentence_transformers import SentenceTransformer

        with self._lock:
            if self.loaded:
                return self
            if not self.store_dir.exists():
                logger.warning("store not found: %s", self.store_dir)
                return self
            import chromadb

            try:
                client = chromadb.PersistentClient(path=str(self.store_dir))
                col = client.get_collection(self.collection_name)
            except Exception:
                logger.warning(
                    "collection %r not in %s", self.collection_name, self.store_dir
                )
                return self
            self._client = client
            self._col = col
            self._emb = SentenceTransformer(self.embedding_model, device="cpu")
            try:
                ids = col.get(include=[])["ids"]
                self._chunk_cnt = len(ids)
                ms = col.get(include=["metadatas"])["metadatas"]
                self._doc_cnt = len({m.get("doc") for m in ms if m})
            except Exception as exc:
                logger.warning("counts: %s", exc)
            self.loaded = True
        return self

    # -- metrics ------------------------------------------------------------
    def _line_map(self) -> dict[str, dict[int, int]]:
        """Map ``doc -> {chunk_id: start line}``, built once and cached.

        Only present for docs whose source file exists under
        ``<project>/data/documents``; PDFs need ``pypdf``. Falls back to {} on
        any hiccup (missing corpus, unreadable file) so retrieval never breaks.
        """
        if self._lines is not None:
            return self._lines
        lines: dict[str, dict[int, int]] = {}
        with self._lock:
            if self._lines is not None:
                return self._lines
            try:
                got = self._col.get(include=["documents", "metadatas"])
                per_doc: dict[str, list[tuple[int, str]]] = {}
                for m, txt in zip(
                    got.get("metadatas", []), got.get("documents", [])
                ):
                    if not m:
                        continue
                    doc = str(m.get("doc", "?"))
                    per_doc.setdefault(doc, []).append(
                        (int(m.get("chunk", -1)), txt or "")
                    )
                for doc, chunks in per_doc.items():
                    text = _read_source(_DOCS / doc)
                    if text is None:
                        continue
                    lines[doc] = _start_line(chunks, text)
            except Exception as exc:
                logger.warning("line map: %s", exc)
            self._lines = lines
        return self._lines

    def n_chunks(self) -> int:
        return self._chunk_cnt if self.loaded else 0

    def n_documents(self) -> int:
        return self._doc_cnt if self.loaded else 0

    def store_metrics(self) -> dict[str, Any]:
        return {
            "n_chunks": self.n_chunks(),
            "n_documents": self.n_documents(),
            "collection": self.collection_name,
        }

    # -- retrieval ----------------------------------------------------------
    def retrieve(self, question: str, k: int | None = None) -> list[dict[str, Any]]:
        if not self.loaded:
            return []
        k = k or self.top_k
        vec = self._emb.encode([question], convert_to_numpy=True).tolist()
        res = self._col.query(query_embeddings=vec, n_results=k)
        lines = self._line_map()
        hits: list[dict[str, Any]] = []
        for i, cid in enumerate(res["ids"][0]):
            m = res["metadatas"][0][i] or {}
            doc = str(m.get("doc", "?"))
            chunk = str(m.get("chunk", "?"))
            try:
                line = lines.get(doc, {}).get(int(chunk))
            except ValueError:
                line = None
            hits.append(
                {
                    "doc": doc,
                    "chunk": chunk,
                    "line": line,
                    "text": res["documents"][0][i],
                    "distance": float(res["distances"][0][i]),
                    "id": str(cid),
                }
            )
        return hits


def build_retriever(settings: Any | None = None) -> Retriever:
    r = Retriever(settings)
    r.load()
    return r

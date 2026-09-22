"""Generation service.

Grounds the answer in the retrieved chunks by sending them to a local Ollama
daemon. Robustness contract (keeps CI + the frontend green with no GPU):

* If Ollama is reachable and the model is installed -> real generation,
  ``backend == "ollama"``.
* Else -> a deterministic template answer that selects the best hits and lists
  their exact source ids, ``backend == "template"``.

The prompt is grounded (sources in, sources out): the model is told to only use
the provided chunks and to close with a single ``Sources:`` line citing the
retrieved ``doc.md::chunk_N`` ids, which the route turns into the structured
list. Everything is prompt-level — no post-hoc citation parsing, so the API
and the notebook stay interchangeable.
"""
from __future__ import annotations

import re
import threading
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.core.config import get_settings

_S = get_settings()


class CraftedSource(BaseModel):
    idx: int
    doc: str
    chunk: str
    text: str
    distance: float

    @property
    def source(self) -> str:
        return f"{self.doc}::{self.chunk}"


class GenerationResponse(BaseModel):
    answer: str = Field(description="grounded answer, Sources footer already attached")
    backend: str = Field(description="ollama | template")


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _format_sources(hits: list[dict[str, Any]]) -> str:
    items = [f"[[{i+1}]] {h['doc']}::{h['chunk']}" for i, h in enumerate(hits)]
    return "\n".join(items) if items else "(none)"


def _build_prompt(question: str, hits: list[dict[str, Any]]) -> str:
    ctx = "\n\n".join(
        f"[[{i+1}]]\n{h['text'].strip()}" for i, h in enumerate(hits)
    )
    legend = _format_sources(hits)
    return (
        "You are a support agent for an e-commerce help center. Answer the "
        "question using ONLY the chunks between <CONTEXT> and </CONTEXT>. "
        "Do not invent facts and do not use outside knowledge.\n\n"
        f"<CONTEXT>\n{ctx}\n</CONTEXT>\n\n"
        "QUESTION: {{question}}\n\n"
        "Rules:\n"
        "- Cite the exact sources you relied on, one line at the very end:\n"
        "  Sources: <doc>::<chunk>, <doc>::<chunk>  (only ids from the legend\n"
        "  below; never write bare [[n]] — always include the doc::chunk id).\n"
        f"- Legacy ids: {legend}\n"
        "- If the context does not answer the question, say so and write "
        '"Sources: (none)".\n\n'
        f"Answer:\n"
    ).replace("{{question}}", question)


def _fallback_answer(question: str, hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "I could not find that in the knowledge base.\n\nSources: (none)"
    top = hits[0]
    second = hits[1] if len(hits) > 1 else None
    body = (
        f"Based on the sources, here is the most relevant information:\n\n"
        f"{_collapse_ws(top['text'])[:600]}"
    )
    if second:
        body += f"\n\nAlso relevant:\n{_collapse_ws(second['text'])[:300]}"
    ids = ", ".join(f"{h['doc']}::{h['chunk']}" for h in hits[:2])
    body += f"\n\nSources: {ids}"
    return body


class Generator:
    """Stateless — one instance per app, created at startup. Thread-safe: the
    LLM call is guarded by a lock and availability is probed once (cached)."""

    def __init__(self) -> None:
        self.host = _S.ollama_host.rstrip("/")
        self.model = _S.ollama_model
        self.timeout = _S.ollama_timeout
        self._lock = threading.Lock()
        self._available: bool | None = None

    # -- availability ----------------------------------------------------
    def _probe(self) -> bool:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            names = {m.get("name", "") for m in r.json().get("models", [])}
            ok = any(
                n.split(":", 1)[0] == self.model.split(":", 1)[0] for n in names
            )
        except httpx.HTTPError:
            ok = False
        self._available = ok
        return ok

    def available(self) -> bool:
        if self._available is None:
            with self._lock:
                if self._available is None:
                    self._probe()
        return self._available

    def status(self) -> dict[str, str]:
        return {
            "host": self.host,
            "model": self.model,
            "available": str(self.available()),
        }

    # -- generation -------------------------------------------------------
    def generate(
        self, question: str, hits: list[dict[str, Any]]
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """Return ``(answer, backend, cited_hits)``.

        ``cited_hits`` is the subset of retrieved chunks the model actually
        cited in its trailing ``Sources:`` line (``[]`` when it wrote
        ``Sources: (none)``), so the structured ``sources`` array can never
        disagree with the answer text. Falls back to a template when Ollama
        is absent so the endpoint never 500s on a missing daemon.
        """
        if not self.available():
            answer = _fallback_answer(question, hits)
            return answer, "template-fallback", _cited_hits(answer, hits)
        answer = self._generate_ollama(question, hits)
        return answer, "ollama", _cited_hits(answer, hits)

    def _generate_ollama(self, question: str, hits: list[dict[str, Any]]) -> str:
        payload = {
            "model": self.model,
            "prompt": _build_prompt(question, hits),
            "stream": False,
            "options": {"temperature": 0, "seed": 42, "num_predict": 250},
        }
        with self._lock:
            try:
                r = httpx.post(
                    f"{self.host}/api/generate", json=payload, timeout=self.timeout
                )
                r.raise_for_status()
                text = (r.json().get("response") or "").strip()
            except httpx.HTTPError:
                return _fallback_answer(question, hits)
        if not text:
            return _fallback_answer(question, hits)
        return _canonical_footer(_ensure_sources_footer(text, hits), hits)


def _ensure_sources_footer(answer: str, hits: list[dict[str, Any]]) -> str:
    if re.search(r"(?im)^\s*sources\s*:", answer):
        return answer
    ids = ", ".join(f"{h['doc']}::{h['chunk']}" for h in hits[:2])
    return f"{answer.strip()}\n\nSources: {ids}"


def _canonical_footer(answer: str, hits: list[dict[str, Any]]) -> str:
    """Rewrite a trailing ``Sources:`` line into canonical ``doc::chunk`` ids.

    LLaMA sometimes cites with the prompt's bare legend indices (``[[1]]``)
    instead of ``20_place_order.md::66``. We map each token back to the real
    hit id it refers to (expanding ``[[n]]`` by hit order), dedupe, and rebuild
    the footer so the answer text and the structured ``sources`` array are
    always identical. ``Sources: (none)`` is left untouched.

    Only a *trailing* ``Sources:`` line is treated as the footer: LLaMA
    occasionally opens its reply with the sources list, in which case the line
    is part of the body and must not swallow the text that follows it.
    """
    text = answer or ""
    m = re.search(r"(?im)^\s*sources\s*:\s*(.+?)\s*\Z", text)
    if not m:
        return text
    body = m.group(1)
    if re.search(r"(?i)\(none\)", body):
        return text
    ids: list[str] = []
    for raw in re.split(r",", body):
        tok = raw.strip()
        if not tok:
            continue
        hit_id = re.search(r"([\w\-./]+\.\w+)::([\w.\-]+)", tok)
        if hit_id:
            ids.append(f"{hit_id.group(1)}::{hit_id.group(2)}")
            continue
        lm = re.search(r"\[\[\s*(\d+)\s*\]\]", tok)
        if lm:
            idx = int(lm.group(1)) - 1
            if 0 <= idx < len(hits):
                ids.append(f"{hits[idx]['doc']}::{hits[idx]['chunk']}")
    if not ids:
        return text
    seen: list[str] = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    footer = "Sources: " + ", ".join(seen)
    return text[: m.start()].rstrip() + "\n\n" + footer + "\n"


_REFUSAL_PATTERNS = (
    r"i could not find",
    r"i couldn'?t find",
    r"cannot answer",
    r"can'?t answer",
    r"i do not know",
    r"i don'?t know",
    r"not in the knowledge base",
    r"no information",
    r"not enough information",
    r"unable to answer",
    r"does not (answer|cover|contain)",
)


def _looks_like_refusal(answer: str) -> bool:
    """True when the answer *text* says it could not answer the question.

    The footer is excluded: LLaMA sometimes writes ``Sources: (none)`` even
    when it answered fully from context, so the body is the ground truth.
    """
    body = re.sub(r"(?im)^\s*sources\s*:.*$", "", answer or "").lower()
    return any(re.search(p, body) for p in _REFUSAL_PATTERNS)


def _cited_hits(answer: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Subset of ``hits`` behind the answer, so the structured ``sources``
    array never disagrees with the visible answer.

    Precise ``doc::chunk`` ids in the footer win. When the footer is missing
    or unparseable (LLaMA occasionally botches it) we trust the answer body:
    a genuine answer falls back to the retrieved hits — never to an empty
    list — and only a real refusal (``Sources: (none)`` *plus* refusal wording
    in the body) yields ``[]``.
    """
    text = answer or ""
    trailer = re.search(r"(?ims)^\s*sources\s*:\s*(.+?)\s*$", text)
    if trailer:
        cited = set(re.findall(r"([\w\-./]+\.\w+)::([\w.\-]+)", trailer.group(1)))
        if cited:
            return [
                h
                for h in hits
                if (str(h.get("doc")), str(h.get("chunk"))) in cited
            ]
        if re.search(r"(?i)\(none\)", trailer.group(1)) and _looks_like_refusal(text):
            return []
    if _looks_like_refusal(text):
        return []
    return hits

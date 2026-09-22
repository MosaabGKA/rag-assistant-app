"""Streamlit chat UI for the RAG assistant.

Talks only to the FastAPI backend (no direct Chroma/Ollama access).
"""
from __future__ import annotations

import re

import streamlit as st

import api_client


def _strip_footer(answer: str) -> str:
    """Drop the model's trailing ``Sources:`` line from display.

    The backend keeps the footer (it drives the structured ``sources`` array);
    here we render the cited list separately, so the raw footer would only
    duplicate it.
    """
    if not answer:
        return ""
    return re.sub(r"(?im)^\s*Sources:.*\n?", "", answer).strip()


def _format_answer(res: dict) -> str:
    msg = _strip_footer(res.get("answer", ""))
    srcs = res.get("sources", [])
    if srcs:
        lines = [
            "- [{}] `{}:{}` chunk {} (d={:.2f})".format(
                i,
                s.get("doc"),
                s.get("line") or s.get("chunk"),
                s.get("chunk"),
                s.get("distance", 0),
            )
            for i, s in enumerate(srcs, 1)
        ]
        msg += "\n\n**Sources**\n" + "\n".join(lines)
    msg += f"\n\n`backend: {res.get('backend')}`"
    return msg


st.set_page_config(page_title="Support Assistant", layout="wide")
st.title(":material/support_agent: Support Assistant")
st.caption("Grounded answers from your support documents. "
           "Built with ChromaDB + Sentence-Transformers + llama3.2 via Ollama.")

h = api_client.health()
if h.get("status") == "ok":
    llm = h.get("llm_available")
    st.sidebar.success(
        f"Store ready · {h.get('n_documents')} docs / {h.get('n_chunks')} chunks"
    )
    st.sidebar.info(
        f"LLM: {'online' if llm else 'template-fallback'} · {h.get('llm_model')}"
    )
else:
    st.sidebar.error("Backend not reachable — start the FastAPI service first.")

if "history" not in st.session_state:
    st.session_state.history = []

for role, txt in st.session_state.history:
    with st.chat_message(role):
        st.markdown(txt)

if q := st.chat_input("Ask about orders, returns, refunds, shipping…"):
    st.chat_message("user").markdown(q)
    with st.spinner("Retrieving and grounding an answer…"):
        try:
            res = api_client.ask(q)
        except api_client.APIError:
            res = None
    if res is None:
        msg = "_API error — is the backend running?_"
    else:
        msg = _format_answer(res)
    st.session_state.history.append(("user", q))
    st.session_state.history.append(("assistant", msg))
    with st.chat_message("assistant"):
        st.markdown(msg)

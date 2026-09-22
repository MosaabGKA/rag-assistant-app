# RAG Assistant

## Overview

A grounded **e-commerce support Q&A assistant**. Given a customer question
(e.g. *"How do I return an item I bought?"*) it retrieves the most relevant
chunks from a persisted FAQ corpus, grounds an answer in those chunks via a
local LLM, and always reports the exact sources it used. No hallucinated
answers: if the store can't answer, it says so and lists no fake sources.

## Architecture

```
corpus (29 FAQs from the Bitext customer-support knowledge base: orders, returns, refunds, shipping, accounts, payments)
   │  chunk (1200 chars, 200 overlap) + embed (all-MiniLM-L6-v2)
   ▼
Chroma persistent store  ── backend/data/vector_store ── 29 docs · 4741 chunks
   │  semantic top-k (default 4)
   ▼
FastAPI  /query  ──  Chroma top-k retrieval  →  Ollama (llama3.2:3b) grounded
   │                                                                 answer
   ├── /health   live store + LLM status (docs/chunks/available/model)
   └── /query    {"question": "…"} → {"answer", "sources", "backend"}
   │
   └── Streamlit UI (frontend/) + terminal CLI (frontend/cli.py)
```

## Quick start

```bash
# 1. build the vector store first (git-ignored, ~37 MB — see Corpus below)
git clone https://github.com/MosaabGKA/rag-assistant-app
cd rag-assistant-app
.venv/bin/python -m pip install -r notebooks-requirements.txt 2>/dev/null || true
#   (simplest: run notebooks/rag_pipeline.ipynb top-to-bottom to create the store)

# 2. backend
python -m venv .venv && . .venv/bin/activate
cd backend
pip install -r requirements.txt
cp .env.example .env            # defaults are fine on a stock machine
uvicorn app.main:app --reload   # http://127.0.0.1:8000

# 3. frontend (separate terminal)
cd ../frontend
pip install -r requirements.txt
streamlit run app.py            # http://localhost:8501
```

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | **Streamlit** (chat UI) + **requests**; terminal companion CLI |
| Backend | **FastAPI** + uvicorn, Pydantic-schematized `/query` & `/health` |
| Retrieval | **ChromaDB** persistent store · **sentence-transformers** `all-MiniLM-L6-v2` · 1200-char chunks (200 overlap) |
| Generation | **Ollama** · llama3.2:3b (deterministic: temperature 0, seed 42) with a template fallback |
| Data | Bitext customer-support knowledge base (26,872 pairs → 29 FAQ docs) |
| Tooling | Python 3.12, pytest, pandas, python-dotenv |

## Project structure

```
rag-assistant-app/
├── backend/                  # FastAPI service
│   ├── app/
│   │   ├── main.py           # lifespan, wiring, /health + /query routes
│   │   ├── api/routes/query.py
│   │   ├── core/config.py    # settings from .env
│   │   ├── schemas/query.py
│   │   ├── services/retrieval.py    # Chroma + line-number mapping
│   │   └── services/generation.py   # Ollama grounding + Sources footer
│   ├── data/vector_store/    # pre-built Chroma store (git-ignored)
│   ├── tests/                # hermetic pytest suite (no daemons needed)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py                # Streamlit chat UI
│   ├── api_client.py         # API_BASE_URL aware backend wrapper
│   ├── cli.py                # terminal companion
│   ├── .env                  # API_BASE_URL=http://localhost:8000
│   └── requirements.txt
├── notebooks/rag_pipeline.ipynb   # end-to-end pipeline (Phases 2.1–2.7)
├── scripts/build_corpus.py        # raw dataset → data/documents/
├── data/
│   ├── documents/            # 27 FAQ .md + 2 .pdf (committed)
│   └── raw/                  # original 27K CSV (git-ignored, ~19 MB)
├── .gitignore
└── README.md
```

## Environment variables

Read from `backend/.env` (see `backend/.env.example`); every variable has a
sane default so the app runs on a stock machine.

| Variable | Default | Purpose |
|---|---|---|
| `VECTOR_STORE_DIR` | `data/vector_store` | Chroma DB directory (relative → `backend/`) |
| `COLLECTION` | `support_docs` | Chroma collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer for query + docs |
| `TOP_K` | `4` | Retrieved chunks fed to the LLM |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server base URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Generation model |
| `OLLAMA_TIMEOUT` | `300` | Per-generation HTTP timeout (s) |
| `ALLOWED_ORIGINS` | `http://localhost:8501,...` | CORS for the Streamlit frontend |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Backend bind address |

Frontend reads `API_BASE_URL` from `frontend/.env` (falls back to
`http://127.0.0.1:8000`), **never hard-coded** in the app code.

## API

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/health` | GET  | — | `{status, collection, n_documents, n_chunks, llm_available, llm_model}` |
| `/query`  | POST | `{"question": "…"}` | `{answer, sources:[{doc, chunk, distance}], backend}` |

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I return an item I bought?"}'
```

Ollama is optional at runtime: when it is offline the API falls back to a
template answer and reports `backend: "template-fallback"` so the endpoint
never 500s on a missing daemon.

## Evaluation results (Phase 2.6)

12 questions spanning distinct support topics, assessed on three criteria:

| Metric | Result |
|---|---|
| Retrieval top-1 hit rate — accepted doc is the #1 chunk | **92% (11/12)** |
| Retrieval top-k hit rate — accepted doc in top-4 | **92% (11/12)** |
| Answers with citations (every answer ends in `Sources:`) | **92% (11/12)** |

The single miss is *"I cannot log in to my new account"*:
`21_recover_password.md` is retrieved instead of the equally-valid
`22_registration_problems.md` / `11_create_account.md`, because the neighbour
doc legitimately answers it. Cases like this are scored against the *set* of
accepted documents. Repeated runs are deterministic (temperature 0, seed 42),
so the table in `notebooks/rag_pipeline.ipynb` reproduces exactly.

## Screenshots

> (Add a couple of screenshots of the running app — e.g. the Streamlit chat
> with a grounded, cited answer and the `/health` status sidebar. Place PNGs in
> `docs/screenshots/` and embed them here.)

## Quality gates

```bash
cd backend
pytest            # hermetic — no Chroma/Ollama daemon needed, fully offline
python -m pytest tests  # same suite, explicit path
```

The corpus is built by `scripts/build_corpus.py` (see *Corpus & offline
storage*). Notebook `notebooks/rag_pipeline.ipynb` consumes `data/documents/`,
builds the vector store and the RAG baseline end-to-end;
`backend/data/vector_store/config.json` records the persisted build (collection
`support_docs`, 29 docs, 4741 chunks, top-k 4, `all-MiniLM-L6-v2`,
`llama3.2:3b`).

## Corpus & offline storage

Everything needed to run the app is stored in this repository, so no download
is required at runtime:

* `data/documents/` — the retrieval corpus, **27 per-topic FAQ markdown files
  + 2 PDFs** (29 docs). This is a *condensed* local copy of the Bitext
  knowledge base, not the full 26,872-row dataset.
* `backend/data/vector_store/` — **not committed** (git-ignored, ~37 MB).
  Regenerate it by running `notebooks/rag_pipeline.ipynb` top-to-bottom; section
  2.5 persists Chroma under `backend/data/vector_store/` and writes
  `config.json` recording the build (collection `support_docs`, 29 docs,
  4741 chunks, top-k 4, `all-MiniLM-L6-v2`, `llama3.2:3b`).
* `data/raw/` — the **original downloaded dataset** (Bitext 27K-instruct CSV,
  ~19 MB). Git-ignored (re-downloadable from Hugging Face). The `data/documents/`
  corpus is generated from it by `scripts/build_corpus.py`: text is cleaned
  (`{{X}}` → `(X)`, whitespace collapsed), pairs are grouped by `intent`, and one
  FAQ markdown per topic is written to `data/documents/`, with two intents
  (`check_refund_policy`, `cancel_order`) also exported as PDFs.

## License

Coursework — ITI (Information Technology Institute) Advanced AI final
project. 

Retrieval corpus: the public **Bitext customer-support knowledge
base** (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`,
Hugging Face, 26,872 instruction/response pairs, license CDLA-Sharing-1.0),
condensed into 29 per-topic FAQ documents. 

All *code* is original to this
repository.

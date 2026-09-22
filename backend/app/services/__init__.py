"""Service layer — retrieval, generation and the orchestration pipeline.

Kept import- and startup- side-effect free on purpose: importing any module
here never touches the vector store, the embedder or Ollama, so tests and
``uvicorn`` can load the package without a GPU / daemon. Real work only starts
when ``Retriever.load()`` is called (once, from the app lifespan).
"""

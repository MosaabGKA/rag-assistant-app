"""Application package root.

Importing ``app`` must stay completely side-effect free (no vector store, no
model, no Ollama, no logging bootstrap) so pytest collection and
``fastapi --help`` behave identically whether or not the index exists.
"""

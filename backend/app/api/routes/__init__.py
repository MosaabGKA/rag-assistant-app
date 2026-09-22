"""API route modules — intentionally empty.

Routes are discovered through their own module objects (``main.py`` imports
``from app.api.routes.query import router``), never through package-level
attribute lookups, so this ``__init__`` must not import anything — an eager
``from . import query`` here triggers the classic partially-initialised circular
import during startup.
"""

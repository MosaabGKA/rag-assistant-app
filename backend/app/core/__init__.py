"""Application core — settings + (later) module-level singletons.

Exposed so notebook cells, tests and the API read the *same* Settings
instance (``get_settings`` is memoized) instead of each building their own.
"""
from app.core.config import get_settings

__all__ = ["get_settings"]

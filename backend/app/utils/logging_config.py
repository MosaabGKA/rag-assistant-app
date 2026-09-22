"""Minimal logging bootstrap.

One module-level function::

    get_logger(name) -> logging.Logger

Calling it the *first time* installs a single stderr handler on the root
logger (so line noise is constant across the notebook / uvicorn / pytest)
and sets ``app`` to WARNING. Calling it again is a no-op. Idempotent and
side-effect safe — safe to call at import time in every module.
"""
from __future__ import annotations

import logging
import sys
import threading

_lock = threading.Lock()
_installed = False


def get_logger(name: str) -> logging.Logger:
    global _installed
    if not _installed:
        with _lock:
            if not _installed:
                _setup()
                _installed = True
    return logging.getLogger(name)


def _setup() -> None:
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s | %(message)s"))
    root.addHandler(handler)

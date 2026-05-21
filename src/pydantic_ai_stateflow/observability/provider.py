"""Back-compat shim.

The legacy ``ObservabilityProvider`` class was deleted in SP1 T11.
Apps now configure observability via ``ObservabilityConfig`` passed
to ``sf.create_app(observability=...)``.

This module re-exports ``has_logfire`` for any downstream code that
imports it from here.
"""

from __future__ import annotations

import importlib

__all__ = ["has_logfire"]


def has_logfire() -> bool:
    """Soft import — True iff `logfire` is importable in this process."""
    try:
        mod = importlib.import_module("logfire")
        return mod is not None
    except Exception:
        return False

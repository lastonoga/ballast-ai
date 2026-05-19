"""Central logging entry point for ``pydantic_ai_stateflow``.

## Library defaults

At framework import time we attach a ``NullHandler`` to the
``pydantic_ai_stateflow`` root logger, so the library NEVER spams the
host application's stderr by default — same pattern stdlib recommends
for libraries. Hosts that want to see framework logs do one of:

1. **Set the env var** ``STATEFLOW_LOG_LEVEL`` (e.g. ``INFO`` /
   ``DEBUG``). The framework auto-configures a ``StreamHandler`` on
   ``sys.stderr`` at that level on first ``configure()`` call (or on
   first ``get_logger()`` after the env var is read at import time).
2. **Call ``configure(level=...)``** explicitly from app startup.
3. **Wire their own logging.dictConfig** that targets the
   ``pydantic_ai_stateflow`` namespace — the library does nothing
   special, regular Python logging hierarchy rules apply.

## Submodule loggers

Every module that wants to emit a log gets its own logger via
``get_logger(__name__)``. Because every submodule is rooted under
``pydantic_ai_stateflow.*``, all framework logs flow through the
single root logger, and apps can selectively enable DEBUG for one
subsystem (e.g.
``logging.getLogger('pydantic_ai_stateflow.api.streaming').setLevel(
'DEBUG')``).

## Why not the existing ``observability`` package

``pydantic_ai_stateflow.observability`` covers **OTel/Logfire span
emission** — a structured tracing concern. Python ``logging`` is a
different concern (free-form diagnostic text). Both live side-by-side;
a single request can produce both spans and log lines.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TextIO


ROOT_NAMESPACE = "pydantic_ai_stateflow"
"""Namespace under which all framework loggers are rooted."""

_DEFAULT_FORMAT = "%(asctime)s %(name)-50s %(levelname)-7s %(message)s"

_configured = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger rooted under ``pydantic_ai_stateflow``.

    Pass ``__name__`` from a framework module — since every framework
    module's ``__name__`` starts with ``pydantic_ai_stateflow.``, the
    returned logger inherits levels and handlers from the framework
    root. Calling with no argument returns the root logger itself.

    Apps that want to instrument their own code with the same logger
    hierarchy can pass a string starting with ``pydantic_ai_stateflow.``
    too — it's just a regular ``logging.getLogger`` lookup.
    """
    return logging.getLogger(name or ROOT_NAMESPACE)


def _attach_null_handler() -> None:
    """Library default: NullHandler so we don't print to stderr until
    the host opts in. Idempotent."""
    root = logging.getLogger(ROOT_NAMESPACE)
    if not any(isinstance(h, logging.NullHandler) for h in root.handlers):
        root.addHandler(logging.NullHandler())


def configure(
    *,
    level: int | str | None = None,
    stream: TextIO | None = None,
    fmt: str | None = None,
) -> None:
    """Attach a ``StreamHandler`` to the framework root logger.

    Idempotent: only the first call wires up a handler. Subsequent
    calls update the level on the root logger (so hosts can dial
    DEBUG up/down without re-adding handlers).

    Args:
      level: explicit level (``logging.DEBUG``, ``'INFO'``, …). When
        ``None``, reads ``STATEFLOW_LOG_LEVEL`` from env; if that's
        also unset, defaults to ``INFO``.
      stream: file-like to write to. Defaults to ``sys.stderr`` so
        log lines don't pollute stdout (which servers often pipe to
        access-log channels).
      fmt: custom format string. Defaults to a compact
        timestamp/name/level/message format.
    """
    global _configured  # noqa: PLW0603

    root = logging.getLogger(ROOT_NAMESPACE)
    resolved_level = level if level is not None else os.environ.get(
        "STATEFLOW_LOG_LEVEL", "INFO",
    )
    if isinstance(resolved_level, str):
        resolved_level = resolved_level.upper()
    root.setLevel(resolved_level)

    if _configured:
        return

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(fmt or _DEFAULT_FORMAT))
    root.addHandler(handler)
    _configured = True


_attach_null_handler()

# Eager configure when the host has set STATEFLOW_LOG_LEVEL. Lets users
# get framework logs without modifying app code — just set an env var
# before running. App-driven ``configure(level=...)`` overrides this.
if os.environ.get("STATEFLOW_LOG_LEVEL"):
    configure()


__all__ = ["ROOT_NAMESPACE", "configure", "get_logger"]

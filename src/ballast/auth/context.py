"""Ambient ``current_user_id`` ContextVar."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_current_user_id: ContextVar[str | None] = ContextVar(
    "current_user_id", default=None,
)


def current_user_id() -> str | None:
    """Return the user id bound to the current context, if any."""
    return _current_user_id.get()


@contextmanager
def acting_as(user_id: str) -> Iterator[None]:
    """Bind ``user_id`` to the current context for the duration of the
    block. API middleware wraps each request handler with this so
    downstream code (repos, channels) reads the caller's identity from
    the ambient context instead of plumbing it through every signature.
    """
    tok = _current_user_id.set(user_id)
    try:
        yield
    finally:
        _current_user_id.reset(tok)


__all__ = ["acting_as", "current_user_id"]

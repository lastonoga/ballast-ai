"""Shared session/transaction plumbing for ``Sql*Repository`` classes.

Per-method ``async with self._sessionmaker() as session, session.begin():``
is dialect-correct but visually noisy and forces every cross-cutting
concern (observability spans, retry-on-serialization-failure, …) to
be patched into N methods across multiple files. Two named context
managers express intent and give cross-cutting concerns a single
hook:

  - ``_session()`` opens a session for read-only work; no transaction
    block is opened (SQLAlchemy autobegin still applies on a write,
    but nothing here writes).
  - ``_tx()`` opens a session AND ``session.begin()`` so the body
    commits on clean exit / rolls back on exception.

The mixin owns the ``async_sessionmaker``; subclasses should NOT
define ``__init__`` unless they need additional state.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class SqlSessionMixin:
    """Inject + share the session-lifecycle plumbing."""

    def __init__(
        self, session_factory: "async_sessionmaker[AsyncSession]",
    ) -> None:
        self._sessionmaker = session_factory

    @asynccontextmanager
    async def _session(self) -> "AsyncIterator[AsyncSession]":
        """Read-only session — no transaction block."""
        async with self._sessionmaker() as session:
            yield session

    @asynccontextmanager
    async def _tx(self) -> "AsyncIterator[AsyncSession]":
        """Read/write session inside an explicit transaction.

        Commits on clean exit, rolls back on exception.
        """
        async with self._sessionmaker() as session, session.begin():
            yield session


__all__ = ["SqlSessionMixin"]

"""Event log repository — append + read-by-seq for SSE resume."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from sqlalchemy import asc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import col

from ballast.persistence.events.domain import ThreadEvent


@runtime_checkable
class EventLogRepository(Protocol):
    """Durable per-thread append-only event log.

    Writers (the ``DurableAgent`` workflow) call ``append`` for every
    event emitted by ``agent.run_stream``. Readers (the SSE endpoint)
    call ``read_since`` on reconnect to catch up on events the client
    missed while disconnected.

    Both operations are scoped to a single ``thread_id``. ``seq`` is
    monotonic PER THREAD (not global) — implementations are free to
    use ``MAX(seq)+1`` semantics or a per-thread sequence generator.
    """

    async def append(
        self,
        *,
        thread_id: UUID,
        kind: str,
        payload: dict[str, Any],
    ) -> ThreadEvent:
        """Append one event; returns the persisted row (with assigned ``seq``)."""
        ...

    async def read_since(
        self,
        thread_id: UUID,
        *,
        after_seq: int = 0,
        limit: int = 1000,
    ) -> list[ThreadEvent]:
        """Return events with ``seq > after_seq`` in ascending order."""
        ...

    async def latest_seq(self, thread_id: UUID) -> int:
        """Largest ``seq`` for ``thread_id``, or ``0`` if none."""
        ...


class InMemoryEventLogRepository:
    """In-memory implementation for dev / tests / single-process apps.

    Thread-safe enough for asyncio — all mutations happen on the event
    loop. For multi-process / distributed deployments, swap for a
    persistent implementation (Postgres / Redis / etc).
    """

    def __init__(self) -> None:
        self._events: dict[UUID, list[ThreadEvent]] = {}

    async def append(
        self,
        *,
        thread_id: UUID,
        kind: str,
        payload: dict[str, Any],
    ) -> ThreadEvent:
        bucket = self._events.setdefault(thread_id, [])
        seq = (bucket[-1].seq + 1) if bucket else 1
        event = ThreadEvent(
            thread_id=thread_id,
            seq=seq,
            kind=kind,
            payload=dict(payload),
        )
        bucket.append(event)
        return event

    async def read_since(
        self,
        thread_id: UUID,
        *,
        after_seq: int = 0,
        limit: int = 1000,
    ) -> list[ThreadEvent]:
        bucket = self._events.get(thread_id, [])
        out = [e for e in bucket if e.seq > after_seq]
        return out[:limit]

    async def latest_seq(self, thread_id: UUID) -> int:
        bucket = self._events.get(thread_id)
        return bucket[-1].seq if bucket else 0


class SqlEventLogRepository:
    """SQLAlchemy-backed event log. Works on Postgres AND SQLite.

    Uses only dialect-portable types (``JSON`` variant on sqlite,
    ``JSONB`` on postgres — see ``domain.py``).

    Owns its session lifecycle: each method opens a session via the
    injected ``async_sessionmaker``. ``append`` derives the next ``seq``
    via ``MAX(seq)+1`` inside a single ``session.begin()`` block. For
    high-contention multi-writer workloads on the same thread, prefer a
    per-thread sequence generator — for the framework's typical
    one-workflow-per-thread shape, ``MAX(seq)+1`` is sufficient.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = session_factory

    async def append(
        self,
        *,
        thread_id: UUID,
        kind: str,
        payload: dict[str, Any],
    ) -> ThreadEvent:
        async with self._sessionmaker() as session, session.begin():
            max_seq_stmt = select(func.max(col(ThreadEvent.seq))).where(
                col(ThreadEvent.thread_id) == thread_id,
            )
            current_max = (await session.execute(max_seq_stmt)).scalar()
            seq = (current_max or 0) + 1
            event = ThreadEvent(
                thread_id=thread_id,
                seq=seq,
                kind=kind,
                payload=dict(payload),
            )
            session.add(event)
            await session.flush()
            await session.refresh(event)
        return event

    async def read_since(
        self,
        thread_id: UUID,
        *,
        after_seq: int = 0,
        limit: int = 1000,
    ) -> list[ThreadEvent]:
        async with self._sessionmaker() as session:
            stmt = (
                select(ThreadEvent)
                .where(
                    col(ThreadEvent.thread_id) == thread_id,
                    col(ThreadEvent.seq) > after_seq,
                )
                .order_by(asc(col(ThreadEvent.seq)))
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def latest_seq(self, thread_id: UUID) -> int:
        async with self._sessionmaker() as session:
            stmt = select(func.max(col(ThreadEvent.seq))).where(
                col(ThreadEvent.thread_id) == thread_id,
            )
            current_max = (await session.execute(stmt)).scalar()
            return int(current_max or 0)

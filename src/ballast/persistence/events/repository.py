"""Event log repository — append + read-by-seq for SSE resume."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID

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

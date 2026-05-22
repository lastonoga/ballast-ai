from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from ballast.persistence.outbox.domain import OutboxEvent


@runtime_checkable
class OutboxRepository(Protocol):
    async def enqueue(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        workflow_id: UUID | None = None,
    ) -> OutboxEvent: ...

    async def list_undelivered(
        self, *, limit: int = 100,
    ) -> list[OutboxEvent]: ...

    async def mark_delivered(self, id: UUID) -> None: ...


class InMemoryOutboxRepository:
    def __init__(self) -> None:
        self._rows: list[OutboxEvent] = []

    async def enqueue(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        workflow_id: UUID | None = None,
    ) -> OutboxEvent:
        event = OutboxEvent(
            id=uuid4(),
            event_type=event_type,
            payload=dict(payload),
            workflow_id=workflow_id,
            delivered_at=None,
            created_at=datetime.now(tz=UTC),
        )
        self._rows.append(event)
        return event

    async def list_undelivered(
        self, *, limit: int = 100,
    ) -> list[OutboxEvent]:
        out = [r for r in self._rows if r.delivered_at is None]
        return out[:limit]

    async def mark_delivered(self, id: UUID) -> None:
        for r in self._rows:
            if r.id == id:
                r.delivered_at = datetime.now(tz=UTC)
                return

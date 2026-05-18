from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic_ai_stateflow.persistence.outbox.domain import OutboxEvent


@runtime_checkable
class OutboxRepository(Protocol):
    async def enqueue(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        tenant_id: UUID,
        workflow_id: UUID | None = None,
    ) -> OutboxEvent: ...

    async def list_undelivered(
        self, *, tenant_id: UUID, limit: int = 100
    ) -> list[OutboxEvent]: ...

    async def mark_delivered(self, id: UUID, *, tenant_id: UUID) -> None: ...


class InMemoryOutboxRepository:
    def __init__(self) -> None:
        self._rows: list[OutboxEvent] = []

    async def enqueue(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        tenant_id: UUID,
        workflow_id: UUID | None = None,
    ) -> OutboxEvent:
        event = OutboxEvent(
            id=uuid4(),
            tenant_id=tenant_id,
            event_type=event_type,
            payload=dict(payload),
            workflow_id=workflow_id,
            delivered_at=None,
            created_at=datetime.now(tz=UTC),
        )
        self._rows.append(event)
        return event

    async def list_undelivered(
        self, *, tenant_id: UUID, limit: int = 100
    ) -> list[OutboxEvent]:
        out = [
            r
            for r in self._rows
            if r.tenant_id == tenant_id and r.delivered_at is None
        ]
        return out[:limit]

    async def mark_delivered(self, id: UUID, *, tenant_id: UUID) -> None:
        for i, r in enumerate(self._rows):
            if r.id == id and r.tenant_id == tenant_id:
                self._rows[i] = r.model_copy(
                    update={"delivered_at": datetime.now(tz=UTC)}
                )
                return

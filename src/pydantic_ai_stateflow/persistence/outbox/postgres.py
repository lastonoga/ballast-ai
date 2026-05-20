from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import asc, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from pydantic_ai_stateflow.persistence.outbox.domain import OutboxEvent


class PostgresOutboxRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def enqueue(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        workflow_id: UUID | None = None,
    ) -> OutboxEvent:
        event = OutboxEvent(
            event_type=event_type,
            payload=dict(payload),
            workflow_id=workflow_id,
        )
        self._s.add(event)
        await self._s.flush()
        await self._s.refresh(event)
        return event

    async def list_undelivered(
        self, *, limit: int = 100,
    ) -> list[OutboxEvent]:
        stmt = (
            select(OutboxEvent)
            .where(col(OutboxEvent.delivered_at).is_(None))
            .order_by(asc(col(OutboxEvent.created_at)))
            .limit(limit)
        )
        return list((await self._s.execute(stmt)).scalars().all())

    async def mark_delivered(self, id: UUID) -> None:
        stmt = (
            update(OutboxEvent)
            .where(col(OutboxEvent.id) == id)
            .values(delivered_at=datetime.now(tz=UTC))
        )
        await self._s.execute(stmt)

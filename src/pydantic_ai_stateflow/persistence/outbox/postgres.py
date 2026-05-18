from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import asc, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from pydantic_ai_stateflow.persistence.outbox.domain import OutboxEvent
from pydantic_ai_stateflow.persistence.outbox.persistence import OutboxRow


class PostgresOutboxRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def enqueue(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        tenant_id: UUID,
        workflow_id: UUID | None = None,
    ) -> OutboxEvent:
        row = OutboxRow(
            tenant_id=tenant_id,
            event_type=event_type,
            payload=dict(payload),
            workflow_id=workflow_id,
        )
        self._s.add(row)
        await self._s.flush()
        await self._s.refresh(row)
        return OutboxEvent.from_row(row)

    async def list_undelivered(
        self, *, tenant_id: UUID, limit: int = 100
    ) -> list[OutboxEvent]:
        stmt = (
            select(OutboxRow)
            .where(
                col(OutboxRow.tenant_id) == tenant_id,
                col(OutboxRow.delivered_at).is_(None),
            )
            .order_by(asc(col(OutboxRow.created_at)))
            .limit(limit)
        )
        rows = (await self._s.execute(stmt)).scalars().all()
        return [OutboxEvent.from_row(r) for r in rows]

    async def mark_delivered(self, id: UUID, *, tenant_id: UUID) -> None:
        stmt = (
            update(OutboxRow)
            .where(
                col(OutboxRow.id) == id,
                col(OutboxRow.tenant_id) == tenant_id,
            )
            .values(delivered_at=datetime.now(tz=UTC))
        )
        await self._s.execute(stmt)

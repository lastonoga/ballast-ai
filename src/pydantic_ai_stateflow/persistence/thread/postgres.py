"""PostgreSQL-backed ThreadRepository using SQLAlchemy AsyncSession."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import asc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from pydantic_ai_stateflow.persistence.thread.domain import Message, Thread
from pydantic_ai_stateflow.persistence.thread.persistence import MessageRow, ThreadRow


class PostgresThreadRepository:
    """SQLAlchemy/PostgreSQL implementation of the ThreadRepository protocol.

    Accepts an ``AsyncSession`` from the caller; flush()-not-commit() is used
    so the caller controls the transaction boundary (Unit-of-Work pattern).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        purpose: str,
        purpose_metadata: dict[str, Any],
        actor_id: str,
        tenant_id: UUID,
    ) -> Thread:
        row = ThreadRow(
            tenant_id=tenant_id,
            purpose=purpose,
            purpose_metadata=dict(purpose_metadata),
            actor_id=actor_id,
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)
        return Thread.from_row(row)

    async def load(self, id: UUID, *, tenant_id: UUID) -> Thread | None:
        stmt = select(ThreadRow).where(
            col(ThreadRow.id) == id,
            col(ThreadRow.tenant_id) == tenant_id,
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return Thread.from_row(row)

    async def add_message(
        self,
        thread_id: UUID,
        *,
        role: str,
        parts: list[dict[str, Any]],
        tenant_id: UUID,
    ) -> Message:
        # Verify thread exists for this tenant (FK + tenant isolation).
        stmt = select(ThreadRow).where(
            col(ThreadRow.id) == thread_id,
            col(ThreadRow.tenant_id) == tenant_id,
        )
        result = await self._session.execute(stmt)
        if result.scalar_one_or_none() is None:
            raise KeyError(f"Thread {thread_id} not found for tenant {tenant_id}")

        msg_row = MessageRow(
            tenant_id=tenant_id,
            thread_id=thread_id,
            role=role,
            parts=list(parts),
        )
        self._session.add(msg_row)
        await self._session.flush()
        await self._session.refresh(msg_row)
        return Message.from_row(msg_row)

    async def history(
        self,
        thread_id: UUID,
        *,
        tenant_id: UUID,
        limit: int = 100,
    ) -> list[Message]:
        stmt = (
            select(MessageRow)
            .where(
                col(MessageRow.thread_id) == thread_id,
                col(MessageRow.tenant_id) == tenant_id,
            )
            .order_by(asc(col(MessageRow.created_at)))
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        return [Message.from_row(r) for r in rows]

"""PostgreSQL-backed ThreadRepository using SQLAlchemy AsyncSession."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import asc, desc, select, update
from sqlalchemy import delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from pydantic_ai_stateflow.persistence.thread.domain import Message, Thread, ThreadStatus
from pydantic_ai_stateflow.persistence.thread.persistence import MessageRow, ThreadRow
from pydantic_ai_stateflow.persistence.thread.repository import (
    ThreadClosedError,
    _walk_active_branch,
)


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
        parent_id: UUID | None = None,
    ) -> Message:
        # Verify thread exists for this tenant AND is open.
        stmt = select(ThreadRow).where(
            col(ThreadRow.id) == thread_id,
            col(ThreadRow.tenant_id) == tenant_id,
        )
        result = await self._session.execute(stmt)
        thread_row = result.scalar_one_or_none()
        if thread_row is None:
            raise KeyError(f"Thread {thread_id} not found for tenant {tenant_id}")
        if thread_row.status == ThreadStatus.CLOSED.value:
            raise ThreadClosedError(
                f"Thread {thread_id} is closed; cannot add message"
            )
        # ARCHIVED threads remain appendable; only CLOSED is terminal.

        msg_row = MessageRow(
            tenant_id=tenant_id,
            thread_id=thread_id,
            role=role,
            parts=list(parts),
            parent_id=parent_id,
        )
        self._session.add(msg_row)
        await self._session.flush()
        await self._session.refresh(msg_row)
        return Message.from_row(msg_row)

    async def close(self, thread_id: UUID, *, tenant_id: UUID) -> Thread:
        now = datetime.now(tz=UTC)
        stmt = (
            update(ThreadRow)
            .where(col(ThreadRow.id) == thread_id, col(ThreadRow.tenant_id) == tenant_id)
            .values(status=ThreadStatus.CLOSED.value, closed_at=now)
        )
        result = await self._session.execute(stmt)
        if result.rowcount == 0:  # type: ignore[attr-defined]
            raise KeyError(f"Thread {thread_id} not found for tenant {tenant_id}")
        # Reload to return canonical view
        loaded = await self.load(thread_id, tenant_id=tenant_id)
        assert loaded is not None
        return loaded

    async def history(
        self,
        thread_id: UUID,
        *,
        tenant_id: UUID,
        limit: int = 100,
    ) -> list[Message]:
        # Fetch ALL messages for the thread (no SQL-level limit) so the
        # tree walker can pick the active branch correctly. ``limit`` is
        # applied after walking. For threads with thousands of branches
        # this becomes wasteful; an iter-5 optimization is to push the
        # walk into a recursive CTE or to keep a denormalized
        # ``thread.active_path`` column.
        stmt = (
            select(MessageRow)
            .where(
                col(MessageRow.thread_id) == thread_id,
                col(MessageRow.tenant_id) == tenant_id,
            )
            .order_by(asc(col(MessageRow.created_at)))
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        msgs = [Message.from_row(r) for r in rows]
        return _walk_active_branch(msgs, limit=limit)

    async def siblings(
        self, message_id: UUID, *, tenant_id: UUID,
    ) -> list[Message]:
        target_stmt = select(MessageRow).where(
            col(MessageRow.id) == message_id,
            col(MessageRow.tenant_id) == tenant_id,
        )
        target_result = await self._session.execute(target_stmt)
        target = target_result.scalar_one_or_none()
        if target is None:
            return []

        # Find rows that share parent_id (including the target itself).
        # NULL == NULL semantics: SQL treats NULL as not-equal even to
        # itself, so split the query on whether parent_id is NULL.
        if target.parent_id is None:
            stmt = (
                select(MessageRow)
                .where(
                    col(MessageRow.thread_id) == target.thread_id,
                    col(MessageRow.tenant_id) == tenant_id,
                    col(MessageRow.parent_id).is_(None),
                )
                .order_by(asc(col(MessageRow.created_at)))
            )
        else:
            stmt = (
                select(MessageRow)
                .where(
                    col(MessageRow.thread_id) == target.thread_id,
                    col(MessageRow.tenant_id) == tenant_id,
                    col(MessageRow.parent_id) == target.parent_id,
                )
                .order_by(asc(col(MessageRow.created_at)))
            )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        return [Message.from_row(r) for r in rows]

    async def list_(
        self,
        *,
        tenant_id: UUID,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Thread]:
        stmt = select(ThreadRow).where(col(ThreadRow.tenant_id) == tenant_id)
        if not include_archived:
            stmt = stmt.where(col(ThreadRow.status) != ThreadStatus.ARCHIVED.value)
        stmt = stmt.order_by(desc(col(ThreadRow.created_at))).limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        return [Thread.from_row(r) for r in rows]

    async def rename(
        self, thread_id: UUID, *, title: str | None, tenant_id: UUID
    ) -> Thread:
        stmt = (
            update(ThreadRow)
            .where(col(ThreadRow.id) == thread_id, col(ThreadRow.tenant_id) == tenant_id)
            .values(title=title)
        )
        result = await self._session.execute(stmt)
        if result.rowcount == 0:  # type: ignore[attr-defined]
            raise KeyError(f"Thread {thread_id} not found for tenant {tenant_id}")
        loaded = await self.load(thread_id, tenant_id=tenant_id)
        assert loaded is not None
        return loaded

    async def archive(self, thread_id: UUID, *, tenant_id: UUID) -> Thread:
        stmt = (
            update(ThreadRow)
            .where(col(ThreadRow.id) == thread_id, col(ThreadRow.tenant_id) == tenant_id)
            .values(status=ThreadStatus.ARCHIVED.value)
        )
        result = await self._session.execute(stmt)
        if result.rowcount == 0:  # type: ignore[attr-defined]
            raise KeyError(f"Thread {thread_id} not found for tenant {tenant_id}")
        loaded = await self.load(thread_id, tenant_id=tenant_id)
        assert loaded is not None
        return loaded

    async def unarchive(self, thread_id: UUID, *, tenant_id: UUID) -> Thread:
        stmt = (
            update(ThreadRow)
            .where(col(ThreadRow.id) == thread_id, col(ThreadRow.tenant_id) == tenant_id)
            .values(status=ThreadStatus.OPEN.value)
        )
        result = await self._session.execute(stmt)
        if result.rowcount == 0:  # type: ignore[attr-defined]
            raise KeyError(f"Thread {thread_id} not found for tenant {tenant_id}")
        loaded = await self.load(thread_id, tenant_id=tenant_id)
        assert loaded is not None
        return loaded

    async def delete(self, thread_id: UUID, *, tenant_id: UUID) -> None:
        # Manually delete child messages first (no DB-level CASCADE configured).
        # Idempotent: unknown thread / wrong tenant → no-op.
        msg_stmt = sa_delete(MessageRow).where(
            col(MessageRow.thread_id) == thread_id,
            col(MessageRow.tenant_id) == tenant_id,
        )
        await self._session.execute(msg_stmt)
        thread_stmt = sa_delete(ThreadRow).where(
            col(ThreadRow.id) == thread_id,
            col(ThreadRow.tenant_id) == tenant_id,
        )
        await self._session.execute(thread_stmt)
        await self._session.flush()

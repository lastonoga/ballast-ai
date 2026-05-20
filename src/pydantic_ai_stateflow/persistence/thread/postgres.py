"""PostgreSQL-backed ThreadRepository using SQLAlchemy AsyncSession.

Operates directly on the ``Thread`` / ``Message`` SQLModel classes (no
separate Row models — ``table=True`` SQLModels ARE the persistence row
AND the API/domain payload). The adapter does session.add + flush +
refresh; the caller controls the transaction boundary via
``SqlAlchemyUnitOfWork``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import asc, desc, select
from sqlalchemy import delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from pydantic_ai_stateflow.logging import get_logger
from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.observability.trace_names import TraceName
from pydantic_ai_stateflow.persistence.thread.domain import Message, Thread, ThreadStatus
from pydantic_ai_stateflow.persistence.thread.repository import (
    ThreadClosedError,
    _walk_active_branch,
)

_log = get_logger(__name__)


class PostgresThreadRepository:
    """SQLAlchemy/PostgreSQL implementation of ``ThreadRepository``.

    Accepts an ``AsyncSession`` from the caller; uses ``flush`` (not
    ``commit``) so the caller owns transaction lifetimes via
    ``SqlAlchemyUnitOfWork``.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @traced(
        TraceName.THREAD_CREATE,
        attrs=lambda _self, *, agent, **__: {
            "agent": agent, "backend": "postgres",
        },
    )
    async def create(
        self,
        *,
        agent: str,
        metadata: dict[str, Any] | None = None,
    ) -> Thread:
        thread = Thread(
            agent=agent,
            metadata_=dict(metadata or {}),
        )
        self._session.add(thread)
        await self._session.flush()
        await self._session.refresh(thread)
        return thread

    async def load(self, id: UUID) -> Thread | None:
        stmt = select(Thread).where(col(Thread.id) == id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    @traced(
        TraceName.THREAD_ADD_MESSAGE,
        attrs=lambda _self, thread_id, *, role, **__: {
            "thread_id": str(thread_id),
            "role": role,
            "backend": "postgres",
        },
    )
    async def add_message(
        self,
        thread_id: UUID,
        *,
        role: str,
        parts: list[dict[str, Any]],
        parent_id: UUID | None = None,
    ) -> Message:
        thread = await self.load(thread_id)
        if thread is None:
            raise KeyError(f"Thread {thread_id} not found")
        if thread.status == ThreadStatus.CLOSED:
            raise ThreadClosedError(
                f"Thread {thread_id} is closed; cannot add message",
            )

        msg = Message(
            thread_id=thread_id,
            role=role,
            parts=list(parts),
            parent_id=parent_id,
        )
        self._session.add(msg)
        await self._session.flush()
        await self._session.refresh(msg)
        _log.debug(
            "PostgresThreadRepository.add_message: thread=%s id=%s role=%s "
            "parent=%s parts=%d",
            thread_id, msg.id, role, parent_id, len(msg.parts),
        )
        return msg

    async def add_message_with_id(
        self,
        thread_id: UUID,
        *,
        id: UUID,
        role: str,
        parts: list[dict[str, Any]],
        parent_id: UUID | None = None,
    ) -> Message:
        thread = await self.load(thread_id)
        if thread is None:
            raise KeyError(f"Thread {thread_id} not found")
        if thread.status == ThreadStatus.CLOSED:
            raise ThreadClosedError(
                f"Thread {thread_id} is closed; cannot add message",
            )
        existing_stmt = select(Message).where(col(Message.id) == id)
        existing = (await self._session.execute(existing_stmt)).scalar_one_or_none()
        if existing is not None:
            return existing

        msg = Message(
            id=id,
            thread_id=thread_id,
            role=role,
            parts=list(parts),
            parent_id=parent_id,
        )
        self._session.add(msg)
        await self._session.flush()
        await self._session.refresh(msg)
        _log.debug(
            "PostgresThreadRepository.add_message_with_id: thread=%s "
            "id=%s role=%s parent=%s parts=%d",
            thread_id, msg.id, role, parent_id, len(msg.parts),
        )
        return msg

    async def all_messages(
        self,
        thread_id: UUID,
        *,
        limit: int = 1000,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .where(col(Message.thread_id) == thread_id)
            .order_by(asc(col(Message.created_at)))
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def close(self, thread_id: UUID) -> Thread:
        thread = await self.load(thread_id)
        if thread is None:
            raise KeyError(f"Thread {thread_id} not found")
        thread.status = ThreadStatus.CLOSED
        thread.closed_at = datetime.now(tz=UTC)
        await self._session.flush()
        return thread

    @traced(
        TraceName.THREAD_HISTORY,
        attrs=lambda _self, thread_id, *, limit=100, **__: {
            "thread_id": str(thread_id),
            "limit": limit,
            "backend": "postgres",
        },
    )
    async def history(
        self,
        thread_id: UUID,
        *,
        limit: int = 100,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .where(col(Message.thread_id) == thread_id)
            .order_by(asc(col(Message.created_at)))
        )
        result = await self._session.execute(stmt)
        msgs = list(result.scalars().all())
        return _walk_active_branch(msgs, limit=limit)

    async def siblings(self, message_id: UUID) -> list[Message]:
        target_stmt = select(Message).where(col(Message.id) == message_id)
        target_result = await self._session.execute(target_stmt)
        target = target_result.scalar_one_or_none()
        if target is None:
            return []

        if target.parent_id is None:
            stmt = (
                select(Message)
                .where(
                    col(Message.thread_id) == target.thread_id,
                    col(Message.parent_id).is_(None),
                )
                .order_by(asc(col(Message.created_at)))
            )
        else:
            stmt = (
                select(Message)
                .where(
                    col(Message.thread_id) == target.thread_id,
                    col(Message.parent_id) == target.parent_id,
                )
                .order_by(asc(col(Message.created_at)))
            )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Thread]:
        stmt = select(Thread)
        if not include_archived:
            stmt = stmt.where(col(Thread.status) != ThreadStatus.ARCHIVED)
        stmt = stmt.order_by(desc(col(Thread.created_at))).limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_metadata(
        self, thread_id: UUID, *, metadata: dict[str, Any],
    ) -> Thread:
        thread = await self.load(thread_id)
        if thread is None:
            raise KeyError(f"Thread {thread_id} not found")
        thread.metadata_ = dict(metadata)
        await self._session.flush()
        return thread

    async def archive(self, thread_id: UUID) -> Thread:
        thread = await self.load(thread_id)
        if thread is None:
            raise KeyError(f"Thread {thread_id} not found")
        thread.status = ThreadStatus.ARCHIVED
        await self._session.flush()
        return thread

    async def unarchive(self, thread_id: UUID) -> Thread:
        thread = await self.load(thread_id)
        if thread is None:
            raise KeyError(f"Thread {thread_id} not found")
        thread.status = ThreadStatus.OPEN
        await self._session.flush()
        return thread

    async def delete(self, thread_id: UUID) -> None:
        # Manually delete child messages first (no DB-level CASCADE).
        # Idempotent: unknown thread → no-op.
        await self._session.execute(
            sa_delete(Message).where(col(Message.thread_id) == thread_id),
        )
        await self._session.execute(
            sa_delete(Thread).where(col(Thread.id) == thread_id),
        )
        await self._session.flush()

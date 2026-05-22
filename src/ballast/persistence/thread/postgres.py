"""PostgreSQL-backed ThreadRepository using SQLAlchemy AsyncSession.

Operates directly on the ``Thread`` / ``Message`` SQLModel classes (no
separate Row models — ``table=True`` SQLModels ARE the persistence row
AND the API/domain payload). The repo owns its session lifecycle: each
mutating method opens a session, runs inside ``session.begin()``, and
commits on clean exit. After-commit it self-emits the corresponding
signal (currently ``message_added`` for add/upsert) so the signal only
fires for state that actually landed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import asc, desc, select
from sqlalchemy import delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import col

from ballast.logging import get_logger
from ballast.observability.spans import traced
from ballast.observability.trace_names import TraceName
from ballast.persistence.thread.domain import Message, Thread, ThreadStatus
from ballast.persistence.thread.repository import ThreadClosedError

_log = get_logger(__name__)


class PostgresThreadRepository:
    """SQLAlchemy/PostgreSQL implementation of ``ThreadRepository``.

    Owns its session lifecycle: each method opens a fresh session via the
    injected ``async_sessionmaker`` and commits per-call. Signal emission
    (``message_added``) happens after commit so subscribers never observe
    state that was rolled back.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = session_factory

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
        async with self._sessionmaker() as session, session.begin():
            thread = Thread(
                agent=agent,
                metadata_=dict(metadata or {}),
            )
            session.add(thread)
            await session.flush()
            await session.refresh(thread)
        return thread

    async def load(self, id: UUID) -> Thread | None:
        async with self._sessionmaker() as session:
            stmt = select(Thread).where(col(Thread.id) == id)
            result = await session.execute(stmt)
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
        id: str | None = None,
        silent: bool = False,
    ) -> Message:
        async with self._sessionmaker() as session, session.begin():
            thread = await self._load_in_session(session, thread_id)
            if thread is None:
                raise KeyError(f"Thread {thread_id} not found")
            if thread.status == ThreadStatus.CLOSED:
                raise ThreadClosedError(
                    f"Thread {thread_id} is closed; cannot add message",
                )
            if id is not None:
                existing_stmt = select(Message).where(col(Message.id) == id)
                existing = (
                    await session.execute(existing_stmt)
                ).scalar_one_or_none()
                if existing is not None:
                    return existing

            msg = Message(
                id=id or str(uuid4()),
                thread_id=thread_id,
                role=role,
                parts=list(parts),
            )
            session.add(msg)
            await session.flush()
            await session.refresh(msg)
            _log.debug(
                "PostgresThreadRepository.add_message: thread=%s id=%s role=%s "
                "parts=%d",
                thread_id, msg.id, role, len(msg.parts),
            )
        # session committed at this point.
        if not silent:
            # Lazy import — keep persistence loadable in contexts where the
            # signal infrastructure isn't wired (eval scripts, migrations).
            from ballast.events import message_added  # noqa: PLC0415

            await message_added.send(
                sender=self, thread_id=thread_id, message=msg,
            )
        return msg

    async def upsert_message(
        self,
        thread_id: UUID,
        *,
        id: str,
        role: str,
        parts: list[dict[str, Any]],
        silent: bool = False,
    ) -> Message:
        async with self._sessionmaker() as session, session.begin():
            thread = await self._load_in_session(session, thread_id)
            if thread is None:
                raise KeyError(f"Thread {thread_id} not found")
            if thread.status == ThreadStatus.CLOSED:
                raise ThreadClosedError(
                    f"Thread {thread_id} is closed; cannot upsert message",
                )
            existing_stmt = select(Message).where(col(Message.id) == id)
            existing = (
                await session.execute(existing_stmt)
            ).scalar_one_or_none()
            if existing is not None:
                existing.role = role
                existing.parts = list(parts)
                await session.flush()
                msg = existing
            else:
                msg = Message(
                    id=id,
                    thread_id=thread_id,
                    role=role,
                    parts=list(parts),
                )
                session.add(msg)
                await session.flush()
                await session.refresh(msg)
        # session committed at this point.
        if not silent:
            from ballast.events import message_added  # noqa: PLC0415

            await message_added.send(
                sender=self, thread_id=thread_id, message=msg,
            )
        return msg

    @traced(
        TraceName.THREAD_HISTORY,
        attrs=lambda _self, thread_id, *, limit=1000, **__: {
            "thread_id": str(thread_id),
            "limit": limit,
            "backend": "postgres",
        },
    )
    async def history(
        self,
        thread_id: UUID,
        *,
        limit: int = 1000,
    ) -> list[Message]:
        async with self._sessionmaker() as session:
            stmt = (
                select(Message)
                .where(col(Message.thread_id) == thread_id)
                .order_by(asc(col(Message.created_at)))
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def delete_messages(
        self, thread_id: UUID, *, ids: list[str],
    ) -> None:
        if not ids:
            return
        async with self._sessionmaker() as session, session.begin():
            await session.execute(
                sa_delete(Message).where(
                    col(Message.thread_id) == thread_id,
                    col(Message.id).in_(ids),
                ),
            )

    async def close(self, thread_id: UUID) -> Thread:
        async with self._sessionmaker() as session, session.begin():
            thread = await self._load_in_session(session, thread_id)
            if thread is None:
                raise KeyError(f"Thread {thread_id} not found")
            thread.status = ThreadStatus.CLOSED
            thread.closed_at = datetime.now(tz=UTC)
            await session.flush()
        return thread

    async def list_(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Thread]:
        async with self._sessionmaker() as session:
            stmt = select(Thread)
            if not include_archived:
                stmt = stmt.where(col(Thread.status) != ThreadStatus.ARCHIVED)
            stmt = (
                stmt.order_by(desc(col(Thread.created_at)))
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def update_metadata(
        self, thread_id: UUID, *, metadata: dict[str, Any],
    ) -> Thread:
        async with self._sessionmaker() as session, session.begin():
            thread = await self._load_in_session(session, thread_id)
            if thread is None:
                raise KeyError(f"Thread {thread_id} not found")
            thread.metadata_ = dict(metadata)
            await session.flush()
        return thread

    async def archive(self, thread_id: UUID) -> Thread:
        async with self._sessionmaker() as session, session.begin():
            thread = await self._load_in_session(session, thread_id)
            if thread is None:
                raise KeyError(f"Thread {thread_id} not found")
            thread.status = ThreadStatus.ARCHIVED
            await session.flush()
        return thread

    async def unarchive(self, thread_id: UUID) -> Thread:
        async with self._sessionmaker() as session, session.begin():
            thread = await self._load_in_session(session, thread_id)
            if thread is None:
                raise KeyError(f"Thread {thread_id} not found")
            thread.status = ThreadStatus.OPEN
            await session.flush()
        return thread

    async def delete(self, thread_id: UUID) -> None:
        # Manually delete child messages first (no DB-level CASCADE).
        # Idempotent: unknown thread → no-op.
        async with self._sessionmaker() as session, session.begin():
            await session.execute(
                sa_delete(Message).where(col(Message.thread_id) == thread_id),
            )
            await session.execute(
                sa_delete(Thread).where(col(Thread.id) == thread_id),
            )

    @staticmethod
    async def _load_in_session(
        session: AsyncSession, thread_id: UUID,
    ) -> Thread | None:
        stmt = select(Thread).where(col(Thread.id) == thread_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

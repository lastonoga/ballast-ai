from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic_ai_stateflow.persistence.thread.domain import Message, Thread, ThreadStatus


class ThreadClosedError(Exception):
    """Raised when trying to add a message to a closed thread."""


@runtime_checkable
class ThreadRepository(Protocol):
    """Port for thread + message persistence.

    All methods require `tenant_id` — multi-tenant first-class per spec 1.12.

    Threads have a lifecycle: created OPEN, may be ARCHIVED (still readable +
    appendable), may be CLOSED (terminal — no further messages). Adding
    messages to a CLOSED thread raises `ThreadClosedError`; ARCHIVED threads
    accept messages. ``delete`` is a hard delete (idempotent, cascades to
    messages).
    """

    async def create(
        self,
        *,
        purpose: str,
        purpose_metadata: dict[str, Any],
        actor_id: str,
        tenant_id: UUID,
    ) -> Thread: ...
    async def load(self, id: UUID, *, tenant_id: UUID) -> Thread | None: ...
    async def add_message(
        self, thread_id: UUID, *, role: str, parts: list[dict[str, Any]], tenant_id: UUID
    ) -> Message: ...
    async def history(
        self, thread_id: UUID, *, tenant_id: UUID, limit: int = 100
    ) -> list[Message]: ...
    async def close(self, thread_id: UUID, *, tenant_id: UUID) -> Thread: ...
    async def list_(
        self,
        *,
        tenant_id: UUID,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Thread]: ...
    async def rename(
        self, thread_id: UUID, *, title: str | None, tenant_id: UUID
    ) -> Thread: ...
    async def archive(self, thread_id: UUID, *, tenant_id: UUID) -> Thread: ...
    async def unarchive(self, thread_id: UUID, *, tenant_id: UUID) -> Thread: ...
    async def delete(self, thread_id: UUID, *, tenant_id: UUID) -> None: ...


class InMemoryThreadRepository:
    """In-memory implementation for unit tests."""

    def __init__(self) -> None:
        self._threads: dict[UUID, Thread] = {}
        self._messages: dict[UUID, list[Message]] = {}

    async def create(
        self,
        *,
        purpose: str,
        purpose_metadata: dict[str, Any],
        actor_id: str,
        tenant_id: UUID,
    ) -> Thread:
        thread = Thread(
            id=uuid4(),
            tenant_id=tenant_id,
            purpose=purpose,
            purpose_metadata=dict(purpose_metadata),
            workflow_id=None,
            actor_id=actor_id,
            status=ThreadStatus.OPEN,
            title=None,
            created_at=datetime.now(tz=UTC),
            closed_at=None,
        )
        self._threads[thread.id] = thread
        self._messages[thread.id] = []
        return thread

    async def load(self, id: UUID, *, tenant_id: UUID) -> Thread | None:
        thread = self._threads.get(id)
        if thread is None or thread.tenant_id != tenant_id:
            return None
        return thread

    async def add_message(
        self, thread_id: UUID, *, role: str, parts: list[dict[str, Any]], tenant_id: UUID
    ) -> Message:
        thread = self._threads.get(thread_id)
        if thread is None or thread.tenant_id != tenant_id:
            raise KeyError(f"Thread {thread_id} not found for tenant {tenant_id}")
        if thread.status == ThreadStatus.CLOSED:
            raise ThreadClosedError(f"Thread {thread_id} is closed; cannot add message")
        msg = Message(
            id=uuid4(),
            tenant_id=tenant_id,
            thread_id=thread_id,
            role=role,
            parts=list(parts),
            created_at=datetime.now(tz=UTC),
        )
        self._messages[thread_id].append(msg)
        return msg

    async def history(
        self, thread_id: UUID, *, tenant_id: UUID, limit: int = 100
    ) -> list[Message]:
        thread = self._threads.get(thread_id)
        if thread is None or thread.tenant_id != tenant_id:
            return []
        return self._messages[thread_id][:limit]

    async def close(self, thread_id: UUID, *, tenant_id: UUID) -> Thread:
        thread = self._threads.get(thread_id)
        if thread is None or thread.tenant_id != tenant_id:
            raise KeyError(f"Thread {thread_id} not found for tenant {tenant_id}")
        closed = thread.model_copy(update={
            "status": ThreadStatus.CLOSED,
            "closed_at": datetime.now(tz=UTC),
        })
        self._threads[thread_id] = closed
        return closed

    async def list_(
        self,
        *,
        tenant_id: UUID,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Thread]:
        threads = [t for t in self._threads.values() if t.tenant_id == tenant_id]
        if not include_archived:
            threads = [t for t in threads if t.status != ThreadStatus.ARCHIVED]
        threads.sort(key=lambda t: t.created_at, reverse=True)
        return threads[offset : offset + limit]

    async def rename(
        self, thread_id: UUID, *, title: str | None, tenant_id: UUID
    ) -> Thread:
        thread = self._threads.get(thread_id)
        if thread is None or thread.tenant_id != tenant_id:
            raise KeyError(f"Thread {thread_id} not found for tenant {tenant_id}")
        updated = thread.model_copy(update={"title": title})
        self._threads[thread_id] = updated
        return updated

    async def archive(self, thread_id: UUID, *, tenant_id: UUID) -> Thread:
        thread = self._threads.get(thread_id)
        if thread is None or thread.tenant_id != tenant_id:
            raise KeyError(f"Thread {thread_id} not found for tenant {tenant_id}")
        updated = thread.model_copy(update={"status": ThreadStatus.ARCHIVED})
        self._threads[thread_id] = updated
        return updated

    async def unarchive(self, thread_id: UUID, *, tenant_id: UUID) -> Thread:
        thread = self._threads.get(thread_id)
        if thread is None or thread.tenant_id != tenant_id:
            raise KeyError(f"Thread {thread_id} not found for tenant {tenant_id}")
        updated = thread.model_copy(update={"status": ThreadStatus.OPEN})
        self._threads[thread_id] = updated
        return updated

    async def delete(self, thread_id: UUID, *, tenant_id: UUID) -> None:
        thread = self._threads.get(thread_id)
        if thread is None or thread.tenant_id != tenant_id:
            return  # idempotent: deleting an unknown thread is a no-op
        self._threads.pop(thread_id, None)
        self._messages.pop(thread_id, None)

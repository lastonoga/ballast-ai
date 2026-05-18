from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic_ai_stateflow.persistence.thread.domain import Message, Thread


@runtime_checkable
class ThreadRepository(Protocol):
    """Port for thread + message persistence.

    All methods require `tenant_id` — multi-tenant first-class per spec 1.12.
    """

    async def create(
        self, *, purpose: str, purpose_metadata: dict[str, Any], actor_id: str, tenant_id: UUID
    ) -> Thread: ...
    async def load(self, id: UUID, *, tenant_id: UUID) -> Thread | None: ...
    async def add_message(
        self, thread_id: UUID, *, role: str, parts: list[dict[str, Any]], tenant_id: UUID
    ) -> Message: ...
    async def history(
        self, thread_id: UUID, *, tenant_id: UUID, limit: int = 100
    ) -> list[Message]: ...


class InMemoryThreadRepository:
    """In-memory implementation for unit tests."""

    def __init__(self) -> None:
        self._threads: dict[UUID, Thread] = {}
        self._messages: dict[UUID, list[Message]] = {}

    async def create(
        self, *, purpose: str, purpose_metadata: dict[str, Any], actor_id: str, tenant_id: UUID
    ) -> Thread:
        thread = Thread(
            id=uuid4(),
            tenant_id=tenant_id,
            purpose=purpose,
            purpose_metadata=dict(purpose_metadata),
            workflow_id=None,
            actor_id=actor_id,
            created_at=datetime.now(tz=UTC),
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

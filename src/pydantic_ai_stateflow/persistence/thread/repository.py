from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic_ai_stateflow.persistence.thread.domain import Message, Thread, ThreadStatus


class ThreadClosedError(Exception):
    """Raised when trying to add a message to a closed thread."""


def _walk_active_branch(
    msgs: list[Message], *, limit: int = 100,
) -> list[Message]:
    """Walk the message tree picking ``max(created_at)`` at every fork.

    Used by both ``InMemoryThreadRepository.history`` and
    ``PostgresThreadRepository.history`` after their respective fetch:
    same tree-walking semantics regardless of storage. Pure function over
    a list of ``Message`` rows; doesn't care which thread they belong to
    (callers pre-filter to one thread).

    **Mixed legacy/branched data handling.** Pre-branching data has
    every row with ``parent_id IS NULL``; new data has explicit
    ``parent_id`` links. To make both shapes (and any combination) walk
    cleanly we **virtually link** every NULL-parent row to the
    previous-by-``created_at`` row before walking. So:

    - all-NULL legacy thread → behaves like a linear list.
    - all-linked new thread → strict tree walk.
    - mixed (legacy prefix + new branches) → legacy chain is implicit,
      explicit ``parent_id``s on newer rows override.

    Returns at most ``limit`` messages along the active path.
    """
    if not msgs:
        return []

    linear = sorted(msgs, key=lambda m: m.created_at)
    virt_parent: dict[UUID, UUID | None] = {}
    prev: UUID | None = None
    for m in linear:
        virt_parent[m.id] = m.parent_id if m.parent_id is not None else prev
        prev = m.id

    children_by_parent: dict[UUID | None, list[Message]] = {}
    for m in msgs:
        children_by_parent.setdefault(virt_parent[m.id], []).append(m)

    path: list[Message] = []
    current_parent: UUID | None = None
    while True:
        candidates = children_by_parent.get(current_parent, [])
        if not candidates:
            break
        chosen = max(candidates, key=lambda m: m.created_at)
        path.append(chosen)
        if len(path) >= limit:
            break
        current_parent = chosen.id
    return path


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
        self,
        thread_id: UUID,
        *,
        role: str,
        parts: list[dict[str, Any]],
        tenant_id: UUID,
        parent_id: UUID | None = None,
    ) -> Message:
        """Append a message to the thread tree.

        ``parent_id`` is the id of the message this one replies to.
        Pass ``None`` only for a thread's very first user turn.
        Implementations don't validate that ``parent_id`` lives in the
        same thread/tenant — callers are responsible. (Callers usually
        derive it from a prior ``history()`` call, so the invariant holds
        by construction.)
        """
        ...
    async def history(
        self, thread_id: UUID, *, tenant_id: UUID, limit: int = 100
    ) -> list[Message]:
        """Return the **active branch** of the thread.

        Walks the message tree from the root (``parent_id IS NULL``)
        forward, picking ``max(created_at)`` at every fork. The returned
        list is the linear conversation slice the agent should see —
        previously regenerated branches are silently skipped (still
        present in storage, available via ``siblings``).

        Returns ``[]`` for unknown threads, foreign tenants, or empty
        threads.
        """
        ...
    async def siblings(
        self, message_id: UUID, *, tenant_id: UUID,
    ) -> list[Message]:
        """Return all messages that share ``parent_id`` with ``message_id``.

        Includes the queried message itself. Sorted by ``created_at`` asc
        (oldest first). Used by the threads router to enrich each
        message's response with branch-picker metadata.

        Returns ``[]`` for unknown ids or foreign tenants.
        """
        ...
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
        self,
        thread_id: UUID,
        *,
        role: str,
        parts: list[dict[str, Any]],
        tenant_id: UUID,
        parent_id: UUID | None = None,
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
            parent_id=parent_id,
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
        all_msgs = self._messages[thread_id]
        return _walk_active_branch(all_msgs, limit=limit)

    async def siblings(
        self, message_id: UUID, *, tenant_id: UUID,
    ) -> list[Message]:
        # Search across all tenant-owned threads; the in-memory store
        # doesn't index messages by id so we scan. Fine for tests.
        for tid, msgs in self._messages.items():
            thread = self._threads.get(tid)
            if thread is None or thread.tenant_id != tenant_id:
                continue
            target = next((m for m in msgs if m.id == message_id), None)
            if target is None:
                continue
            sibs = [m for m in msgs if m.parent_id == target.parent_id]
            sibs.sort(key=lambda m: m.created_at)
            return sibs
        return []

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

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic_ai_stateflow.logging import get_logger
from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.observability.trace_names import TraceName
from pydantic_ai_stateflow.persistence.thread.domain import Message, Thread, ThreadStatus

_log = get_logger(__name__)


class ThreadClosedError(Exception):
    """Raised when trying to add a message to a closed thread."""


def _walk_active_branch(
    msgs: list[Message], *, limit: int = 100,
) -> list[Message]:
    """Walk the message tree picking ``max(created_at)`` at every fork.

    **Mixed legacy/branched data handling.** Pre-branching data has
    every row with ``parent_id IS NULL``; new data has explicit
    ``parent_id`` links. We virtually link every NULL-parent row to
    the previous-by-``created_at`` row before walking, so legacy
    threads behave like a linear list and mixed threads keep both
    semantics.
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

    The framework does NOT presume tenancy or actor identity — those
    are app-side concerns. Apps that need scoping put it in
    ``Thread.metadata_`` (free-form dict, optionally validated by a
    ``StateflowAgent.metadata_model``) and filter / authorize at their
    own layer (custom router, repo wrapper, RLS policy, …).

    Threads have a lifecycle: created OPEN, may be ARCHIVED (still
    readable + appendable), may be CLOSED (terminal — no further
    messages). Adding messages to a CLOSED thread raises
    ``ThreadClosedError``; ARCHIVED threads accept messages. ``delete``
    is a hard delete (idempotent, cascades to messages).
    """

    async def create(
        self,
        *,
        agent: str,
        metadata: dict[str, Any] | None = None,
    ) -> Thread:
        """Create a new thread bound to ``agent`` with optional ``metadata``."""
        ...
    async def load(self, id: UUID) -> Thread | None: ...
    async def add_message(
        self,
        thread_id: UUID,
        *,
        role: str,
        parts: list[dict[str, Any]],
        parent_id: UUID | None = None,
    ) -> Message: ...
    async def add_message_with_id(
        self,
        thread_id: UUID,
        *,
        id: UUID,
        role: str,
        parts: list[dict[str, Any]],
        parent_id: UUID | None = None,
    ) -> Message:
        """Persist a message with a caller-supplied id (idempotent).

        Used by the canonical ``ThreadHistoryAdapter.append`` flow:
        assistant-ui generates the message id client-side and calls
        ``POST /threads/{id}/messages``, which forwards to this method.
        If the id already exists in this thread, returns the existing
        row unchanged (no-op) — so a retry doesn't duplicate.
        """
        ...
    async def history(
        self, thread_id: UUID, *, limit: int = 100,
    ) -> list[Message]: ...
    async def all_messages(
        self, thread_id: UUID, *, limit: int = 1000,
    ) -> list[Message]:
        """All persisted messages for ``thread_id`` (no active-branch walk).

        Returns the full tree as a flat list — siblings included.
        Callers that need branch navigation (e.g. the HTTP messages
        endpoint feeding assistant-ui's ``ThreadHistoryAdapter.load``)
        get every message with its ``parent_id`` and rebuild the tree
        client-side.

        ``history`` is still the right call when you need just the
        active branch (e.g. building model-prompt context).
        """
        ...
    async def siblings(self, message_id: UUID) -> list[Message]: ...
    async def close(self, thread_id: UUID) -> Thread: ...
    async def list_(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Thread]: ...
    async def update_metadata(
        self, thread_id: UUID, *, metadata: dict[str, Any],
    ) -> Thread:
        """Replace ``Thread.metadata_`` wholesale. Generic mutation hook
        for apps that store presentation/scope data (title, tenant_id,
        user_id, …) inside metadata."""
        ...
    async def archive(self, thread_id: UUID) -> Thread: ...
    async def unarchive(self, thread_id: UUID) -> Thread: ...
    async def delete(self, thread_id: UUID) -> None: ...


class InMemoryThreadRepository:
    """In-memory implementation for unit tests."""

    def __init__(self) -> None:
        self._threads: dict[UUID, Thread] = {}
        self._messages: dict[UUID, list[Message]] = {}

    @traced(
        TraceName.THREAD_CREATE,
        attrs=lambda _self, *, agent, **__: {"agent": agent},
    )
    async def create(
        self,
        *,
        agent: str,
        metadata: dict[str, Any] | None = None,
    ) -> Thread:
        thread = Thread(
            id=uuid4(),
            agent=agent,
            metadata_=dict(metadata or {}),
            workflow_id=None,
            status=ThreadStatus.OPEN,
            created_at=datetime.now(tz=UTC),
            closed_at=None,
        )
        self._threads[thread.id] = thread
        self._messages[thread.id] = []
        _log.info(
            "InMemoryThreadRepository.create: id=%s agent=%s",
            thread.id, agent,
        )
        return thread

    async def load(self, id: UUID) -> Thread | None:
        return self._threads.get(id)

    @traced(
        TraceName.THREAD_ADD_MESSAGE,
        attrs=lambda _self, thread_id, *, role, **__: {
            "thread_id": str(thread_id),
            "role": role,
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
        thread = self._threads.get(thread_id)
        if thread is None:
            raise KeyError(f"Thread {thread_id} not found")
        if thread.status == ThreadStatus.CLOSED:
            raise ThreadClosedError(
                f"Thread {thread_id} is closed; cannot add message",
            )
        msg = Message(
            id=uuid4(),
            thread_id=thread_id,
            role=role,
            parts=list(parts),
            parent_id=parent_id,
            created_at=datetime.now(tz=UTC),
        )
        self._messages[thread_id].append(msg)
        _log.debug(
            "InMemoryThreadRepository.add_message: thread=%s id=%s role=%s "
            "parent=%s parts=%d",
            thread_id, msg.id, role, parent_id, len(msg.parts),
        )
        return msg

    @traced(
        TraceName.THREAD_HISTORY,
        attrs=lambda _self, thread_id, *, limit=100, **__: {
            "thread_id": str(thread_id),
            "limit": limit,
        },
    )
    async def add_message_with_id(
        self,
        thread_id: UUID,
        *,
        id: UUID,
        role: str,
        parts: list[dict[str, Any]],
        parent_id: UUID | None = None,
    ) -> Message:
        thread = self._threads.get(thread_id)
        if thread is None:
            raise KeyError(f"Thread {thread_id} not found")
        if thread.status == ThreadStatus.CLOSED:
            raise ThreadClosedError(
                f"Thread {thread_id} is closed; cannot add message",
            )
        existing = next(
            (m for m in self._messages[thread_id] if m.id == id),
            None,
        )
        if existing is not None:
            # Idempotent — same id already persisted (retry or duplicate
            # append after a network blip). Return as-is.
            return existing
        msg = Message(
            id=id,
            thread_id=thread_id,
            role=role,
            parts=list(parts),
            parent_id=parent_id,
            created_at=datetime.now(tz=UTC),
        )
        self._messages[thread_id].append(msg)
        _log.debug(
            "InMemoryThreadRepository.add_message_with_id: thread=%s "
            "id=%s role=%s parent=%s parts=%d",
            thread_id, msg.id, role, parent_id, len(msg.parts),
        )
        return msg

    async def history(
        self, thread_id: UUID, *, limit: int = 100,
    ) -> list[Message]:
        thread = self._threads.get(thread_id)
        if thread is None:
            return []
        all_msgs = self._messages[thread_id]
        return _walk_active_branch(all_msgs, limit=limit)

    async def all_messages(
        self, thread_id: UUID, *, limit: int = 1000,
    ) -> list[Message]:
        thread = self._threads.get(thread_id)
        if thread is None:
            return []
        msgs = sorted(self._messages[thread_id], key=lambda m: m.created_at)
        return msgs[:limit]

    async def siblings(self, message_id: UUID) -> list[Message]:
        for tid, msgs in self._messages.items():
            if tid not in self._threads:
                continue
            target = next((m for m in msgs if m.id == message_id), None)
            if target is None:
                continue
            sibs = [m for m in msgs if m.parent_id == target.parent_id]
            sibs.sort(key=lambda m: m.created_at)
            return sibs
        return []

    async def close(self, thread_id: UUID) -> Thread:
        thread = self._threads.get(thread_id)
        if thread is None:
            raise KeyError(f"Thread {thread_id} not found")
        thread.status = ThreadStatus.CLOSED
        thread.closed_at = datetime.now(tz=UTC)
        return thread

    async def list_(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Thread]:
        threads = list(self._threads.values())
        if not include_archived:
            threads = [t for t in threads if t.status != ThreadStatus.ARCHIVED]
        threads.sort(key=lambda t: t.created_at, reverse=True)
        return threads[offset : offset + limit]

    async def update_metadata(
        self, thread_id: UUID, *, metadata: dict[str, Any],
    ) -> Thread:
        thread = self._threads.get(thread_id)
        if thread is None:
            raise KeyError(f"Thread {thread_id} not found")
        thread.metadata_ = dict(metadata)
        return thread

    async def archive(self, thread_id: UUID) -> Thread:
        thread = self._threads.get(thread_id)
        if thread is None:
            raise KeyError(f"Thread {thread_id} not found")
        thread.status = ThreadStatus.ARCHIVED
        return thread

    async def unarchive(self, thread_id: UUID) -> Thread:
        thread = self._threads.get(thread_id)
        if thread is None:
            raise KeyError(f"Thread {thread_id} not found")
        thread.status = ThreadStatus.OPEN
        return thread

    async def delete(self, thread_id: UUID) -> None:
        self._threads.pop(thread_id, None)
        self._messages.pop(thread_id, None)

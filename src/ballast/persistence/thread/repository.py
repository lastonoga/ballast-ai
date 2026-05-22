from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from ballast.logging import get_logger
from ballast.observability.spans import traced
from ballast.observability.trace_names import TraceName
from ballast.persistence.thread.domain import Message, Thread, ThreadStatus

_log = get_logger(__name__)


class ThreadClosedError(Exception):
    """Raised when trying to add a message to a closed thread."""


@runtime_checkable
class ThreadRepository(Protocol):
    """Port for thread + message persistence.

    The framework does NOT presume tenancy or actor identity — those
    are app-side concerns. Apps that need scoping put it in
    ``Thread.metadata_`` (free-form dict, optionally validated by a
    ``BallastAgent.metadata_model``) and filter / authorize at their
    own layer (custom router, repo wrapper, RLS policy, …).

    Threads have a lifecycle: created OPEN, may be ARCHIVED (still
    readable + appendable), may be CLOSED (terminal — no further
    messages). Adding messages to a CLOSED thread raises
    ``ThreadClosedError``; ARCHIVED threads accept messages. ``delete``
    is a hard delete (idempotent, cascades to messages).

    Messages are stored as a **flat linear list** (no parent_id, no
    tree). Edit / regenerate flows collapse to truncate-then-append
    via ``sync_messages_from_body`` at the streaming endpoint.
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
        id: str | None = None,
        silent: bool = False,
    ) -> Message:
        """Append a message to ``thread_id``.

        ``id`` is optional: omit to let the repo mint a fresh UUID4;
        supply to honor a caller-chosen id (e.g. assistant-ui's short
        client id round-tripping via the body sync). If the supplied
        id already exists in this thread, the existing row is returned
        unchanged — gives free idempotency on retries.

        ``silent=True`` skips firing the :data:`ballast.events.message_added`
        signal — use for migrations / seeds / tests that want to write a
        row without the default log + publish handler running.
        """
        ...
    async def history(
        self, thread_id: UUID, *, limit: int = 1000,
    ) -> list[Message]:
        """Linear message list for ``thread_id`` ordered by ``created_at``."""
        ...
    async def upsert_message(
        self,
        thread_id: UUID,
        *,
        id: str,
        role: str,
        parts: list[dict[str, Any]],
        silent: bool = False,
    ) -> Message:
        """Insert or replace a message by id.

        Same semantics as ``add_message`` for first-time inserts, but
        an existing row with the same id has its ``role`` and ``parts``
        REPLACED in place — ``created_at`` is preserved so the message
        keeps its position in the linear history.

        Used by streaming-event APIs (see
        ``runtime.thread_events.ThreadEventStream``) to mutate a single
        UI-visible message across multiple snapshots without appending
        N rows.

        ``silent=True`` skips firing the
        :data:`ballast.events.message_added` signal."""
        ...
    async def delete_messages(
        self, thread_id: UUID, *, ids: list[str],
    ) -> None:
        """Delete messages by id. Unknown ids are ignored (idempotent)."""
        ...
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
        id: str | None = None,
        silent: bool = False,
    ) -> Message:
        thread = self._threads.get(thread_id)
        if thread is None:
            raise KeyError(f"Thread {thread_id} not found")
        if thread.status == ThreadStatus.CLOSED:
            raise ThreadClosedError(
                f"Thread {thread_id} is closed; cannot add message",
            )
        if id is not None:
            existing = next(
                (m for m in self._messages[thread_id] if m.id == id), None,
            )
            if existing is not None:
                return existing
        msg = Message(
            id=id or str(uuid4()),
            thread_id=thread_id,
            role=role,
            parts=list(parts),
            created_at=datetime.now(tz=UTC),
        )
        self._messages[thread_id].append(msg)
        _log.debug(
            "InMemoryThreadRepository.add_message: thread=%s id=%s role=%s "
            "parts=%d",
            thread_id, msg.id, role, len(msg.parts),
        )
        if not silent:
            # Lazy import — ``ballast.events`` is independent of the
            # persistence layer at import time; importing inline keeps
            # the persistence module loadable in contexts where the
            # signal infrastructure isn't wired (eval scripts, raw
            # migrations).
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
        thread = self._threads.get(thread_id)
        if thread is None:
            raise KeyError(f"Thread {thread_id} not found")
        if thread.status == ThreadStatus.CLOSED:
            raise ThreadClosedError(
                f"Thread {thread_id} is closed; cannot upsert message",
            )
        existing = next(
            (m for m in self._messages[thread_id] if m.id == id), None,
        )
        if existing is not None:
            # In-place replace; preserve created_at so linear position
            # in history doesn't shift.
            existing.role = role
            existing.parts = list(parts)
            if not silent:
                from ballast.events import message_added  # noqa: PLC0415

                await message_added.send(
                    sender=self, thread_id=thread_id, message=existing,
                )
            return existing
        msg = Message(
            id=id,
            thread_id=thread_id,
            role=role,
            parts=list(parts),
            created_at=datetime.now(tz=UTC),
        )
        self._messages[thread_id].append(msg)
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
        },
    )
    async def history(
        self, thread_id: UUID, *, limit: int = 1000,
    ) -> list[Message]:
        thread = self._threads.get(thread_id)
        if thread is None:
            return []
        msgs = sorted(self._messages[thread_id], key=lambda m: m.created_at)
        return msgs[:limit]

    async def delete_messages(
        self, thread_id: UUID, *, ids: list[str],
    ) -> None:
        if not ids or thread_id not in self._messages:
            return
        drop = set(ids)
        self._messages[thread_id] = [
            m for m in self._messages[thread_id] if m.id not in drop
        ]

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

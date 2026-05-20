from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from pydantic_ai_stateflow.persistence.thread.persistence import MessageRow, ThreadRow


class ThreadStatus(StrEnum):
    """Thread lifecycle states.

    - OPEN: default. Messages may be appended.
    - ARCHIVED: hidden from the default list view but still readable and
      appendable (assistant-ui's "archived" pane). Apps may unarchive.
    - CLOSED: terminal. No further messages can be appended
      (``add_message`` raises ``ThreadClosedError``). Closing reason is
      application-specific (HITL resolved, onboarding done, etc.) — the
      framework provides the primitive, callers decide when to call
      ``repo.close(...)``.
    """

    OPEN = "open"
    ARCHIVED = "archived"
    CLOSED = "closed"


class Thread(BaseModel):
    """A conversation thread bound to one ``StateflowAgent``.

    ``agent`` is the registry key (== ``StateflowAgent.name``) that
    decides which agent runs against this thread when messages arrive.
    Set once at create-time; the per-thread agent never changes
    afterwards — HITL/tool-call flows rely on the tool registry being
    stable for the same thread.

    ``metadata`` is a free-form dict validated against the registered
    agent's ``metadata_model`` (when present) at create-time. Typical
    keys: ``"relations"`` (FKs to app-side entities) and ``"context"``
    (per-thread settings the agent reads from ``ctx.deps``). The
    metadata model is owned by the agent class — see
    ``pydantic_ai_stateflow.runtime.agents``.
    """

    model_config = ConfigDict(frozen=True)
    id: UUID
    tenant_id: UUID
    agent: str
    """Registry key — matches the ``name`` ClassVar of the
    ``StateflowAgent`` subclass that handles this thread."""
    metadata: dict[str, Any]
    """Validated against ``agent``'s ``metadata_model``; free-form when
    the agent declares no model."""
    workflow_id: UUID | None
    actor_id: str
    status: ThreadStatus
    title: str | None = None
    created_at: datetime
    closed_at: datetime | None

    @classmethod
    def from_row(cls, row: ThreadRow) -> Thread:
        return cls(
            id=row.id,
            tenant_id=row.tenant_id,
            agent=row.agent,
            # ThreadRow uses ``metadata_`` (Python trailing-underscore)
            # to dodge the SQLAlchemy ``metadata`` reserved-attr clash;
            # domain Thread is Pydantic and has no such clash.
            metadata=row.metadata_,
            workflow_id=row.workflow_id,
            actor_id=row.actor_id,
            status=ThreadStatus(row.status),
            title=row.title,
            created_at=row.created_at,
            closed_at=row.closed_at,
        )


class Message(BaseModel):
    """One message in a thread. Threads are conversation TREES, not lists.

    ``parent_id`` is the id of the message this one replies to:

    - first user turn of a thread:           ``parent_id = None``
    - assistant reply to a user turn:        ``parent_id = <that user's id>``
    - follow-up user turn after assistant:   ``parent_id = <that assistant's id>``

    Multiple children of the same parent are *branches* — created by
    regenerating an assistant reply (``trigger='regenerate-message'``) or
    by editing a user turn. The "active" branch surfaced in
    ``ThreadRepository.history(...)`` is the one whose path picks
    ``max(created_at)`` at every fork (i.e. the most recently created
    sibling wins). UI branch-pickers can show all siblings, but cross-
    reload state of which sibling the user clicked is not persisted —
    that's an explicit MVP scope decision (see iter 4 round 2 plan).
    """

    model_config = ConfigDict(frozen=True)
    id: UUID
    tenant_id: UUID
    thread_id: UUID
    role: str
    parts: list[dict[str, Any]]
    parent_id: UUID | None = None
    created_at: datetime

    @classmethod
    def from_row(cls, row: MessageRow) -> Message:
        return cls(
            id=row.id,
            tenant_id=row.tenant_id,
            thread_id=row.thread_id,
            role=row.role,
            parts=row.parts,
            parent_id=row.parent_id,
            created_at=row.created_at,
        )

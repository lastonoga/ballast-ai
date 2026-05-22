"""Thread + Message — SQLModel domain types (single-class per entity).

These classes ARE the persistence rows AND the API/domain models, per
SQLModel's design intent. No ``ThreadRow`` / ``Thread`` dual hierarchy:
one class with ``table=True``, used everywhere.

**Framework presumption is intentionally minimal.** No ``tenant_id``,
no ``actor_id``, no ``title`` — identity / ownership / display are
app-side concerns. Apps that need them put structured values into
``Thread.metadata_`` (JSON-aliased as ``"metadata"``) and (optionally)
validate via ``BallastAgent.metadata_model``.

.. note::
   SQLAlchemy Declarative reserves the attribute name ``metadata`` for
   its ``MetaData`` class-attr, so the Python attribute is
   ``metadata_`` (trailing underscore). The SQL column AND the JSON
   field name are both ``"metadata"``, via ``sa_column=Column("metadata", ...)``
   and Pydantic's ``alias="metadata"``. With
   ``populate_by_name=True`` callers can construct either way; API
   layers should ``model_dump(by_alias=True)`` to keep the wire shape
   stable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


class ThreadStatus(StrEnum):
    """Thread lifecycle states.

    - OPEN: default. Messages may be appended.
    - ARCHIVED: hidden from the default list view but still readable and
      appendable. Apps may unarchive.
    - CLOSED: terminal. No further messages can be appended
      (``add_message`` raises ``ThreadClosedError``).
    """

    OPEN = "open"
    ARCHIVED = "archived"
    CLOSED = "closed"


class Thread(SQLModel, table=True):
    """A conversation thread bound to one ``BallastAgent``.

    ``agent`` is the registry key (== ``BallastAgent.name``).
    ``metadata_`` is free-form, validated by the agent's
    ``metadata_model`` at create-time. Apps put any per-thread scope
    (user_id, tenant_id, title, …) here.
    """

    __tablename__ = "threads"
    model_config = {"populate_by_name": True}  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    agent: str
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        alias="metadata",
        sa_column=Column(
            "metadata", JSONB, nullable=False, server_default="{}",
        ),
    )
    workflow_id: UUID | None = Field(default=None, index=True)
    status: ThreadStatus = Field(default=ThreadStatus.OPEN, index=True)
    created_at: datetime = Field(
        default_factory=_now_utc,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    closed_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )


class Message(SQLModel, table=True):
    """One message in a thread — flat linear list, no tree structure.

    ``id`` is a free-form string (NOT a UUID) so client-supplied ids —
    e.g. assistant-ui's short random strings — round-trip 1:1 without
    coercion. Backend-issued messages default to ``str(uuid4())`` for
    uniqueness.

    No ``parent_id``: the UI runtime (``@assistant-ui/react-ai-sdk``
    over Vercel ``useChat``) renders a flat message array and has no
    branch switcher. Edit / regenerate collapse to "truncate then
    append" via the body-vs-DB diff in ``POST /threads/{id}/messages``.
    """

    __tablename__ = "messages"

    id: str = Field(
        default_factory=lambda: str(uuid4()), primary_key=True,
    )
    thread_id: UUID = Field(foreign_key="threads.id", index=True)
    role: str  # "system" / "user" / "assistant" / "tool"
    parts: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, server_default="[]"),
    )
    created_at: datetime = Field(
        default_factory=_now_utc,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

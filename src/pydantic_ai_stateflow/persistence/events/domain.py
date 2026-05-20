"""``ThreadEvent`` — single durable row per emitted agent-stream event.

Per-thread monotonic ``seq`` lets SSE consumers cheaply ask "give me
everything after N" on reconnect (Last-Event-ID resume). The row also
carries the raw event payload so the consumer doesn't need to re-run
the workflow to recover what was emitted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


class ThreadEvent(SQLModel, table=True):
    """One event from a ``StateflowDurableAgent`` run, persisted for replay.

    ``seq`` is monotonic PER THREAD (not global). Apps that need a
    global ordering can sort on ``(created_at, id)``.
    """

    __tablename__ = "thread_events"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    thread_id: UUID = Field(index=True, nullable=False)
    seq: int = Field(nullable=False)
    """Per-thread monotonic sequence — Last-Event-ID for SSE resume."""

    kind: str = Field(nullable=False)
    """Event kind hint (``message-delta``, ``tool-call``, ``done``, …).

    Lets consumers cheaply filter / route without parsing the payload.
    The exact set of kinds is up to the agent / encoder.
    """

    payload: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    """The full event body — what gets serialized into SSE ``data:``."""

    created_at: datetime = Field(
        default_factory=_now_utc,
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )

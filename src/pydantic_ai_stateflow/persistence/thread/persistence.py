from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


class ThreadRow(SQLModel, table=True):
    __tablename__ = "threads"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = Field(foreign_key="tenants.id", index=True)
    purpose: str  # ThreadPurpose enum value or domain-specific str
    purpose_metadata: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    workflow_id: UUID | None = Field(default=None, index=True)
    actor_id: str
    status: str = Field(default="open", index=True)  # ThreadStatus enum value
    title: str | None = Field(default=None, nullable=True)
    created_at: datetime = Field(
        default_factory=_now_utc,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    closed_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )


class MessageRow(SQLModel, table=True):
    __tablename__ = "messages"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = Field(foreign_key="tenants.id", index=True)
    thread_id: UUID = Field(foreign_key="threads.id", index=True)
    role: str  # "system" / "user" / "assistant" / "tool"
    # Self-FK to the message this one replies to. NULL only for the very
    # first user turn of a thread. Siblings (same parent_id) are branches —
    # produced by ``trigger='regenerate-message'`` or user-message edits.
    # See ``Message`` domain class for the tree-walking contract.
    parent_id: UUID | None = Field(
        default=None, foreign_key="messages.id", index=True, nullable=True,
    )
    parts: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, server_default="[]"),
    )
    created_at: datetime = Field(
        default_factory=_now_utc,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

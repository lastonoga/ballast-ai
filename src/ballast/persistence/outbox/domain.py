"""Outbox event — single SQLModel (no separate Row).

Transactional-outbox pattern: writes go in the same DB transaction as
business mutations; a dispatcher polls undelivered rows and publishes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

# JSONB on Postgres, JSON on every other dialect (sqlite, mysql, …).
_JSON_PORTABLE = JSONB().with_variant(JSON(), "sqlite")


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


class OutboxEvent(SQLModel, table=True):
    """A pending event awaiting external delivery."""

    __tablename__ = "outbox"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    event_type: str
    payload: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(_JSON_PORTABLE, nullable=False, server_default="{}"),
    )
    workflow_id: UUID | None = Field(default=None)
    delivered_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True, index=True),
    )
    created_at: datetime = Field(
        default_factory=_now_utc,
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )

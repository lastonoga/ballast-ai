"""Notes domain types."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


class Note(SQLModel, table=True):
    """Persisted note (also the agent/UI projection).

    Single SQLModel-with-``table=True`` per the framework's convention:
    the row IS the API/domain payload. Tools and UI receive instances
    directly; the SQL repo persists the same class.
    """

    __tablename__ = "notes"

    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    title: str
    body: str
    created_at: datetime = SQLField(
        default_factory=_now_utc,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = SQLField(
        default_factory=_now_utc,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

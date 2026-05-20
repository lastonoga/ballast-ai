"""Notes domain types."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import ConfigDict
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel


class Note(SQLModel, table=True):
    """Persisted note (also the agent/UI projection).

    Frozen via the SQLModel side is impractical; this single class is
    handed back to tools, serialized to the UI, and can be inserted via
    a future SQLAlchemy session unchanged.
    """

    __tablename__ = "notes"

    model_config = ConfigDict(frozen=False)

    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    title: str
    body: str
    created_at: datetime
    updated_at: datetime

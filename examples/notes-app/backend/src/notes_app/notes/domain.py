"""Notes domain types.

`NoteRow` is the persistence row (SQLModel table — iteration 4+ will swap
the in-memory repo for a Postgres-backed impl with this row shape).

`Note` is the immutable domain projection — what tools return to the model
and what the API hands to the UI. Keeping the two split lets the
persistence shape evolve (indexes, audit columns) without leaking into the
agent-visible payload.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel


class NoteRow(SQLModel, table=True):
    """Persistence row for a note (one row per note, scoped by tenant)."""

    __tablename__ = "notes"

    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = SQLField(index=True)
    title: str
    body: str
    created_at: datetime
    updated_at: datetime


class Note(BaseModel):
    """Domain projection of a note.

    Immutable; safe to hand back to tools and serialize directly to the UI.
    Tools return this type so pydantic-ai includes the saved `id` in the
    tool-result the model sees (so it can chain follow-up actions).
    """

    model_config = ConfigDict(frozen=True)

    id: UUID
    tenant_id: UUID
    title: str
    body: str
    created_at: datetime
    updated_at: datetime

"""Note repository — Protocol + in-memory impl.

The Protocol is what the agent tools depend on; the in-memory impl is what
iteration 3 actually wires up. A Postgres/DBOS-backed impl is intentionally
deferred to iteration 4+ (see TODO below) so the example stays self-contained.

All methods are tenant-scoped. `update` raises `KeyError` if the note does
not exist OR belongs to a different tenant; `delete` is idempotent (silent
no-op for unknown / wrong-tenant note ids — the model shouldn't be told
about other tenants' notes either way).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from notes_app.notes.domain import Note


@runtime_checkable
class NoteRepository(Protocol):
    """Tenant-scoped note storage.

    Iteration 3 ships only `InMemoryNoteRepository`; iteration 4+ will add
    a Postgres/SQLModel-backed impl (see TODO at bottom of this file) once
    the engine grows a DBOS-backed unit-of-work boundary suitable for the
    notes-app.
    """

    async def create(
        self, *, title: str, body: str, tenant_id: UUID,
    ) -> Note: ...

    async def list_(
        self, *, tenant_id: UUID, limit: int = 100,
    ) -> list[Note]: ...

    async def get(
        self, note_id: UUID, *, tenant_id: UUID,
    ) -> Note | None: ...

    async def search(
        self, query: str, *, tenant_id: UUID, limit: int = 20,
    ) -> list[Note]: ...

    async def update(
        self,
        note_id: UUID,
        *,
        title: str | None,
        body: str | None,
        tenant_id: UUID,
    ) -> Note: ...

    async def delete(self, note_id: UUID, *, tenant_id: UUID) -> None: ...


class InMemoryNoteRepository:
    """Process-local note store, fresh per `__init__`.

    Suitable for dev / tests / the iteration-3 dogfood smoke. Not threadsafe
    across multiple worker processes — that's fine for a single-uvicorn dev
    server.
    """

    def __init__(self) -> None:
        self._notes: dict[UUID, Note] = {}

    @staticmethod
    def _now() -> datetime:
        return datetime.now(tz=UTC)

    async def create(
        self, *, title: str, body: str, tenant_id: UUID,
    ) -> Note:
        now = self._now()
        note = Note(
            id=uuid4(),
            tenant_id=tenant_id,
            title=title,
            body=body,
            created_at=now,
            updated_at=now,
        )
        self._notes[note.id] = note
        return note

    async def list_(
        self, *, tenant_id: UUID, limit: int = 100,
    ) -> list[Note]:
        rows = [n for n in self._notes.values() if n.tenant_id == tenant_id]
        rows.sort(key=lambda n: n.created_at, reverse=True)
        return rows[:limit]

    async def get(
        self, note_id: UUID, *, tenant_id: UUID,
    ) -> Note | None:
        note = self._notes.get(note_id)
        if note is None or note.tenant_id != tenant_id:
            return None
        return note

    async def search(
        self, query: str, *, tenant_id: UUID, limit: int = 20,
    ) -> list[Note]:
        needle = query.casefold().strip()
        if not needle:
            return []
        hits = [
            n
            for n in self._notes.values()
            if n.tenant_id == tenant_id
            and needle in (n.title + " " + n.body).casefold()
        ]
        hits.sort(key=lambda n: n.created_at, reverse=True)
        return hits[:limit]

    async def update(
        self,
        note_id: UUID,
        *,
        title: str | None,
        body: str | None,
        tenant_id: UUID,
    ) -> Note:
        existing = self._notes.get(note_id)
        if existing is None or existing.tenant_id != tenant_id:
            raise KeyError(f"note {note_id} not found for tenant {tenant_id}")
        updated = existing.model_copy(
            update={
                "title": title if title is not None else existing.title,
                "body": body if body is not None else existing.body,
                "updated_at": self._now(),
            },
        )
        self._notes[note_id] = updated
        return updated

    async def delete(self, note_id: UUID, *, tenant_id: UUID) -> None:
        existing = self._notes.get(note_id)
        if existing is not None and existing.tenant_id == tenant_id:
            del self._notes[note_id]
        # Idempotent: silent no-op on unknown / wrong-tenant ids.


# TODO(iteration 4+): add `PostgresNoteRepository` using `NoteRow` +
# `pydantic_ai_stateflow.persistence.uow.UnitOfWork`. The Protocol above is
# the boundary; the agent tools should not need to change.

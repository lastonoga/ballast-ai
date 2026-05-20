"""Note repository — Protocol + in-memory impl.

The Protocol is what the agent tools depend on; the in-memory impl is what
iteration 3 actually wires up. A Postgres/DBOS-backed impl is intentionally
deferred to iteration 4+ (see TODO below) so the example stays self-contained.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from notes_app.notes.domain import Note


@runtime_checkable
class NoteRepository(Protocol):
    """Note storage."""

    async def create(self, *, title: str, body: str) -> Note: ...

    async def list_(self, *, limit: int = 100) -> list[Note]: ...

    async def get(self, note_id: UUID) -> Note | None: ...

    async def search(self, query: str, *, limit: int = 20) -> list[Note]: ...

    async def update(
        self,
        note_id: UUID,
        *,
        title: str | None,
        body: str | None,
    ) -> Note: ...

    async def delete(self, note_id: UUID) -> None: ...


class InMemoryNoteRepository:
    """Process-local note store, fresh per `__init__`."""

    def __init__(self) -> None:
        self._notes: dict[UUID, Note] = {}

    @staticmethod
    def _now() -> datetime:
        return datetime.now(tz=UTC)

    async def create(self, *, title: str, body: str) -> Note:
        now = self._now()
        note = Note(
            id=uuid4(),
            title=title,
            body=body,
            created_at=now,
            updated_at=now,
        )
        self._notes[note.id] = note
        return note

    async def list_(self, *, limit: int = 100) -> list[Note]:
        rows = list(self._notes.values())
        rows.sort(key=lambda n: n.created_at, reverse=True)
        return rows[:limit]

    async def get(self, note_id: UUID) -> Note | None:
        return self._notes.get(note_id)

    async def search(self, query: str, *, limit: int = 20) -> list[Note]:
        needle = query.casefold().strip()
        if not needle:
            return []
        hits = [
            n
            for n in self._notes.values()
            if needle in (n.title + " " + n.body).casefold()
        ]
        hits.sort(key=lambda n: n.created_at, reverse=True)
        return hits[:limit]

    async def update(
        self,
        note_id: UUID,
        *,
        title: str | None,
        body: str | None,
    ) -> Note:
        existing = self._notes.get(note_id)
        if existing is None:
            raise KeyError(f"note {note_id} not found")
        updated = existing.model_copy(
            update={
                "title": title if title is not None else existing.title,
                "body": body if body is not None else existing.body,
                "updated_at": self._now(),
            },
        )
        self._notes[note_id] = updated
        return updated

    async def delete(self, note_id: UUID) -> None:
        # Idempotent: silent no-op on unknown ids.
        self._notes.pop(note_id, None)

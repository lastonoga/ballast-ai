"""Note repository — Protocol + InMemory + SQL impls.

The Protocol is what the agent tools depend on; ``InMemoryNoteRepository``
is what tests use, and ``SqlNoteRepository`` is what production wires
when ``NOTES_APP_DATABASE_URL`` points at a real database (sqlite by
default, postgres in prod). The class name intentionally isn't
``Postgres*``: the impl uses only dialect-portable SQLAlchemy types
and runs on either backend.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from sqlalchemy import or_, select
from sqlalchemy import delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import col

from notes_app.models.note import Note


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


class SqlNoteRepository:
    """SQLAlchemy-backed note store — sqlite + postgres compatible.

    Commit-per-method, mirroring ``SqlThreadRepository``: each
    mutating call opens a fresh session via the injected
    ``async_sessionmaker`` and commits on clean exit. Reads use a plain
    session (no transaction needed).

    No signal emissions — notes aren't part of the framework's
    chat-message-signal flow.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = session_factory

    @staticmethod
    def _now() -> datetime:
        return datetime.now(tz=UTC)

    async def create(self, *, title: str, body: str) -> Note:
        now = self._now()
        async with self._sessionmaker() as session, session.begin():
            note = Note(
                id=uuid4(),
                title=title,
                body=body,
                created_at=now,
                updated_at=now,
            )
            session.add(note)
            await session.flush()
            await session.refresh(note)
        return note

    async def list_(self, *, limit: int = 100) -> list[Note]:
        async with self._sessionmaker() as session:
            stmt = (
                select(Note)
                .order_by(col(Note.created_at).desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get(self, note_id: UUID) -> Note | None:
        async with self._sessionmaker() as session:
            stmt = select(Note).where(col(Note.id) == note_id)
            return (await session.execute(stmt)).scalar_one_or_none()

    async def search(self, query: str, *, limit: int = 20) -> list[Note]:
        needle = query.strip()
        if not needle:
            return []
        pattern = f"%{needle}%"
        async with self._sessionmaker() as session:
            stmt = (
                select(Note)
                .where(
                    or_(
                        col(Note.title).ilike(pattern),
                        col(Note.body).ilike(pattern),
                    ),
                )
                .order_by(col(Note.created_at).desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def update(
        self,
        note_id: UUID,
        *,
        title: str | None,
        body: str | None,
    ) -> Note:
        async with self._sessionmaker() as session, session.begin():
            stmt = select(Note).where(col(Note.id) == note_id)
            existing = (await session.execute(stmt)).scalar_one_or_none()
            if existing is None:
                raise KeyError(f"note {note_id} not found")
            if title is not None:
                existing.title = title
            if body is not None:
                existing.body = body
            existing.updated_at = self._now()
            await session.flush()
            await session.refresh(existing)
        return existing

    async def delete(self, note_id: UUID) -> None:
        # Idempotent: silent no-op on unknown ids.
        async with self._sessionmaker() as session, session.begin():
            await session.execute(
                sa_delete(Note).where(col(Note.id) == note_id),
            )


# ── Module-level singleton ──────────────────────────────────────────────
# App-specific repository. Imported directly by callers that need it
# (avoids passing through constructor DI). Tests swap via
# ``monkeypatch.setattr("notes_app.repositories.note.notes_repo", mock)``.

notes_repo: NoteRepository = InMemoryNoteRepository()

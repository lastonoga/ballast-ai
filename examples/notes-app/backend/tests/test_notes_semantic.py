"""``NotesSemantic`` — notes-app semantic source over notes_repo."""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from ballast.memory.semantic import SemanticSource
from notes_app.memory.semantic_sources import NotesSemantic, notes_semantic
from notes_app.repositories.note import InMemoryNoteRepository


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemoryNoteRepository]:
    fresh = InMemoryNoteRepository()
    monkeypatch.setattr("notes_app.repositories.note.notes_repo", fresh)
    yield fresh


def test_module_singleton_exists_and_named() -> None:
    assert isinstance(notes_semantic, NotesSemantic)
    assert notes_semantic.name == "notes"


def test_satisfies_semantic_source_protocol() -> None:
    assert isinstance(notes_semantic, SemanticSource)


@pytest.mark.asyncio
async def test_recent_returns_recent_notes(repo: InMemoryNoteRepository) -> None:
    n1 = await repo.create(title="t1", body="b1")
    n2 = await repo.create(title="t2", body="b2")
    results = await notes_semantic.recent(days=30)
    ids = {n.id for n in results}
    assert {n1.id, n2.id} <= ids


@pytest.mark.asyncio
async def test_search_substring(repo: InMemoryNoteRepository) -> None:
    await repo.create(title="ml notes", body="machine learning content")
    await repo.create(title="fashion", body="trends")
    results = await notes_semantic.search(query="machine")
    titles = {n.title for n in results}
    assert "ml notes" in titles
    assert "fashion" not in titles

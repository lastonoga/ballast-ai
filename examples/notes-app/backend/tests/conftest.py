"""Shared pytest fixtures for the notes-app backend tests."""

from __future__ import annotations

import pytest

from notes_app.notes.repository import InMemoryNoteRepository


@pytest.fixture
def repo() -> InMemoryNoteRepository:
    """Fresh in-memory note repo per test (no cross-test leakage)."""
    return InMemoryNoteRepository()

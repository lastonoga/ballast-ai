"""Shared pytest fixtures for the notes-app backend tests."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from notes_app.notes.repository import InMemoryNoteRepository


@pytest.fixture
def repo() -> InMemoryNoteRepository:
    """Fresh in-memory note repo per test (no cross-test leakage)."""
    return InMemoryNoteRepository()


@pytest.fixture
def tenant_id() -> UUID:
    """Random tenant id per test."""
    return uuid4()

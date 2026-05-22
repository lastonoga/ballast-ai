"""Notes-app thread repository — extends framework's InMemoryThreadRepository.

The app owns its repository so it can layer on app-specific behaviour
(e.g. custom indexes, soft-delete semantics) without touching the
framework. Subclasses the reference InMemory impl since the notes-app
demo doesn't need Postgres yet.
"""
from __future__ import annotations

from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
)


class NotesThreadRepository(InMemoryThreadRepository):
    """Notes-app's ThreadRepository. Inherits InMemory impl as-is."""


# Module-level singleton. ``main.py`` imports this and passes to
# ``sf.create_app`` so the framework's Engine resolves the same instance
# everywhere.
thread_repo: NotesThreadRepository = NotesThreadRepository()


__all__ = ["NotesThreadRepository", "thread_repo"]

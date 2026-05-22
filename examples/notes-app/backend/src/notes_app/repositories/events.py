"""Notes-app event log repository — extends framework's InMemory impl.

Mirrors ``notes_app.repositories.thread``: the app owns the repo
instance even when the implementation is the framework's reference
in-memory one, so it can later swap in a Postgres-backed subclass
without touching call sites.
"""
from __future__ import annotations

from ballast.persistence import InMemoryEventLogRepository


class NotesEventLogRepository(InMemoryEventLogRepository):
    """Notes-app's event log. Inherits InMemory impl as-is."""


event_log: NotesEventLogRepository = NotesEventLogRepository()


__all__ = ["NotesEventLogRepository", "event_log"]

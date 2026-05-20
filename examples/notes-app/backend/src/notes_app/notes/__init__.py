"""Notes domain — domain types, repository protocol, in-memory impl.

``build_notes_router`` lives in ``notes_app.notes.routes`` but is NOT
re-exported here because routes import ``NotesAgent``, which itself
pulls in ``notes_app.notes.domain.Note`` — re-exporting would create a
circular import. ``main.py`` imports it directly from the submodule.
"""

from notes_app.notes.domain import Note
from notes_app.notes.repository import InMemoryNoteRepository, NoteRepository

__all__ = [
    "InMemoryNoteRepository",
    "Note",
    "NoteRepository",
]

"""Notes domain — domain types, repository protocol, in-memory impl.

Tools and ``NoteToolDeps`` live next to ``NotesAgent`` in ``notes_app.agent``
(one file per agent — they're the same unit of code).
"""

from notes_app.notes.domain import Note, NoteRow
from notes_app.notes.repository import InMemoryNoteRepository, NoteRepository

__all__ = [
    "InMemoryNoteRepository",
    "Note",
    "NoteRepository",
    "NoteRow",
]

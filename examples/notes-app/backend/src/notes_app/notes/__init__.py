"""Notes domain — domain types, repository protocol, in-memory impl."""

from notes_app.notes.domain import Note
from notes_app.notes.repository import InMemoryNoteRepository, NoteRepository

__all__ = [
    "InMemoryNoteRepository",
    "Note",
    "NoteRepository",
]

"""Notes-app semantic memory sources — wrap ``notes_repo`` for agent exposure.

This module declares the typed accessors the LLM agent sees as tools
when ``SemanticMemory(sources=[notes_semantic])`` is wired into the
Ballast builder.

Each method is marked with ``@memory_tool``. Its docstring becomes the
tool description the LLM reads when deciding which tool to call. Keep
docstrings concrete + decision-oriented.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ballast.memory.semantic import DomainSemanticSource, memory_tool

from notes_app.models.note import Note


class NotesSemantic(DomainSemanticSource):
    """Read-only semantic view over the user's notes."""

    name = "notes"

    @memory_tool
    async def recent(self, days: int = 7) -> list[Note]:
        """Return notes the user has created or edited in the last
        `days` days. Most recent first. Use when the user references
        recent work without a specific identifier."""
        from notes_app.repositories.note import notes_repo  # noqa: PLC0415

        all_notes = await notes_repo.list_()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        filtered = [
            n for n in all_notes
            if getattr(n, "created_at", cutoff) >= cutoff
        ]
        # Sort newest first if created_at exists.
        filtered.sort(
            key=lambda n: getattr(n, "created_at", cutoff),
            reverse=True,
        )
        return filtered

    @memory_tool
    async def search(self, query: str, limit: int = 10) -> list[Note]:
        """Find notes whose title or body matches `query` (substring,
        case-insensitive). Use when the user references a note by topic
        or keyword rather than by id."""
        from notes_app.repositories.note import notes_repo  # noqa: PLC0415

        hits = await notes_repo.search(query)
        return hits[:limit]


notes_semantic: NotesSemantic = NotesSemantic()

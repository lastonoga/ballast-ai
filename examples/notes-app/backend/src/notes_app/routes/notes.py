"""HTTP routes owned by the notes-app (not the framework).

The framework no longer ships ``POST /threads`` because every app has
its own opinion about how new threads come into being (which agent,
what metadata, what side-effects). Notes-app's create endpoint binds
every thread to ``NotesAgent`` and runs ``body.metadata`` through
``validate_thread_metadata`` so that, if/when ``NotesAgent.metadata_model``
gets a schema, the call site is already wired.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from ballast.persistence.thread.repository import ThreadRepository
from ballast.runtime import validate_thread_metadata

from notes_app.agents.notes import NotesAgent


class _CreateThreadBody(BaseModel):
    metadata: dict[str, Any] | None = None


def build_notes_router(repo: ThreadRepository) -> APIRouter:
    """App-owned routes for the notes domain — currently just ``POST /threads``."""
    router = APIRouter()

    @router.post("/threads", status_code=201)
    async def create_thread(body: _CreateThreadBody) -> dict[str, Any]:
        metadata = validate_thread_metadata(NotesAgent, body.metadata)
        thread = await repo.create(
            agent=NotesAgent.name,
            metadata=metadata,
        )
        return thread.model_dump(mode="json", by_alias=True)

    return router

"""FastAPI entry point for the notes-app backend.

Wires:
  - in-memory thread repository (no Postgres yet)
  - in-memory note repository bound through ``app.state.container``
    via an ``on_startup`` hook (per spec 4A.0.7).
  - notes-app routes (``POST /threads``) — see
    ``notes_app.notes.routes``
  - framework threads router (read/lifecycle/delete) + framework
    streaming router (resolves ``NotesAgent`` via registry per request).
  - ``StateflowAgent`` registry: a ``NotesAgent`` instance is registered
    under ``name="notes"`` at boot.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic_ai_stateflow.api import CORSConfig
from pydantic_ai_stateflow.api.streaming import build_streaming_router
from pydantic_ai_stateflow.api.threads import build_threads_router
from pydantic_ai_stateflow.observability import ObservabilityProvider
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
    ThreadRepository,
)
from pydantic_ai_stateflow.runtime import Engine, register_agent

from notes_app.agent import NotesAgent
from notes_app.notes import InMemoryNoteRepository, NoteRepository
from notes_app.notes.routes import build_notes_router

load_dotenv()


def build_app(
    *,
    thread_repo: ThreadRepository | None = None,
    notes_agent: NotesAgent | None = None,
    notes_repo: NoteRepository | None = None,
) -> FastAPI:
    """Construct the FastAPI app."""
    repo = thread_repo or InMemoryThreadRepository()
    notes = notes_repo or InMemoryNoteRepository()
    agent = notes_agent or NotesAgent(notes_repo=notes)

    register_agent(agent)

    engine = Engine(
        providers=[
            ObservabilityProvider(
                service_name="app",
                environment="dev",
                instrument_pydantic_ai=True,
                instrument_httpx=True,
            ),
        ],
    )

    notes_router = build_notes_router(repo)
    threads_router = build_threads_router(thread_repo=repo)
    streaming_router = build_streaming_router(thread_repo=repo)

    async def _bind_domain_repos(app: FastAPI) -> None:
        """Bind app-level repos onto the framework Container (spec 4A.0.7)."""
        app.state.container.bind(NoteRepository, notes)

    app: FastAPI = engine.fastapi_app(
        extra_routers=[notes_router, threads_router, streaming_router],
        cors=CORSConfig.permissive_dev(),
        on_startup=[_bind_domain_repos],
    )

    app.state.notes_repo = notes
    app.state.thread_repo = repo
    app.state.notes_agent = agent
    return app


app: FastAPI = build_app()


def main() -> None:  # pragma: no cover
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("notes_app.main:app", host=host, port=port, reload=True)

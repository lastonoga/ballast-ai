"""FastAPI entry point for the notes-app backend.

Wires:
  - in-memory thread repository (no Postgres yet)
  - in-memory note repository bound through ``app.state.container``
    via an ``on_startup`` hook (per spec 4A.0.7).
  - threads router (read/update/delete) + an app-owned
    ``POST /threads`` create endpoint that calls
    ``validate_thread_metadata(NotesAgent, body.metadata)`` before
    persisting.
  - ``StateflowAgent`` registry: a ``NotesAgent`` instance is registered
    under ``name="notes"`` at boot; ``Thread.agent == "notes"`` makes the
    framework's streaming router resolve and drive it per request.
  - streaming router takes only ``thread_repo`` — the agent, deps, and
    model settings are resolved from the registry via ``thread.agent``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from uuid import UUID

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI
from pydantic import BaseModel
from pydantic_ai_stateflow.api import CORSConfig
from pydantic_ai_stateflow.api.deps import get_tenant_id
from pydantic_ai_stateflow.api.streaming import build_streaming_router
from pydantic_ai_stateflow.api.threads import build_threads_router
from pydantic_ai_stateflow.observability import ObservabilityProvider, has_logfire
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
    ThreadRepository,
)
from pydantic_ai_stateflow.runtime import (
    Engine,
    register_agent,
    validate_thread_metadata,
)

from notes_app.agent import NotesAgent
from notes_app.notes.repository import InMemoryNoteRepository, NoteRepository

if TYPE_CHECKING:
    pass

load_dotenv()


class _CreateThreadBody(BaseModel):
    actor_id: str = "user"
    metadata: dict[str, Any] | None = None


def _build_create_thread_router(repo: ThreadRepository) -> APIRouter:
    """App-owned ``POST /threads`` — the framework no longer ships one.

    Notes-app threads are always bound to the ``"notes"`` agent. Metadata
    (if any) is validated via the framework's registry-backed
    ``validate_thread_metadata`` — currently ``NotesAgent.metadata_model``
    is ``None`` so anything passes through, but the call site is ready
    for a future schema without further wiring.
    """
    router = APIRouter()

    @router.post("/threads", status_code=201)
    async def create_thread(
        body: _CreateThreadBody,
        tenant_id: UUID = Depends(get_tenant_id),
    ) -> dict[str, Any]:
        metadata = validate_thread_metadata(NotesAgent, body.metadata)
        thread = await repo.create(
            agent=NotesAgent.name,
            metadata=metadata,
            actor_id=body.actor_id,
            tenant_id=tenant_id,
        )
        return thread.model_dump(mode="json")

    return router


def build_app(
    *,
    thread_repo: ThreadRepository | None = None,
    notes_agent: NotesAgent | None = None,
    notes_repo: NoteRepository | None = None,
) -> FastAPI:
    """Construct the FastAPI app.

    Args:
      thread_repo: defaults to ``InMemoryThreadRepository``.
      notes_agent: a pre-built ``NotesAgent``. Defaults to one bound to
        the ``notes_repo`` arg; tests inject a variant with a
        ``TestModel``-backed pydantic-ai ``Agent`` (see
        ``tests/test_smoke.py::_fake_notes_agent``).
      notes_repo: ``NoteRepository`` bound into the container. Defaults
        to a fresh ``InMemoryNoteRepository``.
    """
    repo = thread_repo or InMemoryThreadRepository()
    notes = notes_repo or InMemoryNoteRepository()
    agent = notes_agent or NotesAgent(notes_repo=notes)

    # Register the StateflowAgent so build_streaming_router can resolve it
    # by ``Thread.agent`` per request. Registration is idempotent —
    # re-creating the app in tests just overwrites the same name.
    register_agent(agent)

    engine = Engine(
        providers=[
            ObservabilityProvider(
                service_name="app",
                environment="dev",
                instrument_pydantic_ai=True,
                instrument_httpx=True,
            )
        ]
    )
    
    threads_router = build_threads_router(thread_repo=repo)
    create_router = _build_create_thread_router(repo)
    streaming_router = build_streaming_router(thread_repo=repo)

    async def _bind_domain_repos(app: FastAPI) -> None:
        """Bind app-level repos onto the framework Container (spec 4A.0.7)."""
        app.state.container.bind(NoteRepository, notes)

    app: FastAPI = engine.fastapi_app(
        extra_routers=[create_router, threads_router, streaming_router],
        # permissive_dev() covers assistant-ui Next.js (localhost:3000)
        # and Vite (localhost:3003).
        cors=CORSConfig.permissive_dev(),
        on_startup=[_bind_domain_repos],
    )

    # Expose for integration tests + future admin endpoints.
    app.state.notes_repo = notes
    app.state.thread_repo = repo
    app.state.notes_agent = agent
    return app


# Module-level ASGI handle for ``uvicorn notes_app.main:app``.
app: FastAPI = build_app()


def main() -> None:  # pragma: no cover — convenience entry-point
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("notes_app.main:app", host=host, port=port, reload=True)

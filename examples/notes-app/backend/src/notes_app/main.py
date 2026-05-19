"""FastAPI entry point for the notes-app backend.

Wires:
  - in-memory thread repository (no Postgres yet)
  - in-memory note repository bound through ``app.state.container``
    via an ``on_startup`` hook (per spec 4A.0.7 — no module-level
    singletons; see ``docs/superpowers/guides/domain-repo-scaffold.md``).
  - threads router (CRUD)
  - streaming router backed by ``build_streaming_router(thread_repo=,
    agent=, deps_factory=, model_settings=)``. The agent is built lazily
    on first request to keep the import side-effect-free; the deps
    factory injects per-request ``NoteToolDeps(repo=container.get(
    NoteRepository), tenant_id=...)`` into the pydantic-ai agent so its
    CRUD tools act on the right tenant.
  - empty provider list — still doesn't need DBOS / Postgres.

The actual Vercel AI SDK wire encoding, body parsing, event taxonomy,
and tool-approval round-trip are delegated to
``pydantic_ai.ui.vercel_ai.VercelAIAdapter`` (see
``src/pydantic_ai_stateflow/api/streaming/router.py``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic_ai_stateflow.api import CORSConfig
from pydantic_ai_stateflow.api.streaming import build_streaming_router
from pydantic_ai_stateflow.api.threads import build_threads_router
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
    ThreadRepository,
)
from pydantic_ai_stateflow.runtime import Engine

from notes_app.agent import (
    build_agent,
    build_notes_deps_factory,
)
from notes_app.notes.repository import InMemoryNoteRepository, NoteRepository

if TYPE_CHECKING:
    from pydantic_ai import Agent

load_dotenv()


class _LazyAgent:
    """Defer ``build_agent()`` until the first streaming request.

    Constructed at app-boot time, it forwards every ``Agent`` attribute
    access (``run_stream_events``, ``output_type``, etc.) to a real
    ``Agent`` instance built on first touch. Lets the app boot in
    environments without ``OPENROUTER_API_KEY`` (e.g. the threads-only
    smoke tests) and only requires the env var on the first real
    streaming request.
    """

    def __init__(self) -> None:
        self._agent: Agent[Any, Any] | None = None

    def _ensure(self) -> Agent[Any, Any]:
        if self._agent is None:
            self._agent = build_agent()
        return self._agent

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ensure(), name)


def build_app(
    *,
    thread_repo: ThreadRepository | None = None,
    agent: Agent[Any, Any] | None = None,
    notes_repo: NoteRepository | None = None,
) -> FastAPI:
    """Construct the FastAPI app.

    Args:
      thread_repo: defaults to ``InMemoryThreadRepository``.
      agent: a pre-built pydantic-ai ``Agent``. Defaults to a lazy
        OpenRouter-backed agent that hits OpenRouter only on first use.
        Tests pass an in-memory ``TestModel``-backed agent to avoid the
        network.
      notes_repo: ``NoteRepository`` bound into the container. Defaults
        to a fresh ``InMemoryNoteRepository``.
    """
    repo = thread_repo or InMemoryThreadRepository()
    notes = notes_repo or InMemoryNoteRepository()
    resolved_agent: Agent[Any, Any] = agent if agent is not None else _LazyAgent()  # type: ignore[assignment]

    engine = Engine(providers=[])
    threads_router = build_threads_router(thread_repo=repo)

    async def _bind_domain_repos(app: FastAPI) -> None:
        """Bind app-level repos onto the framework Container (spec 4A.0.7)."""
        app.state.container.bind(NoteRepository, notes)

    app: FastAPI = engine.fastapi_app(
        extra_routers=[threads_router],
        # F8: permissive_dev() covers the assistant-ui Next.js dev shell
        # (localhost:3000) and the Vite dev port (localhost:3003).
        cors=CORSConfig.permissive_dev(),
        on_startup=[_bind_domain_repos],
    )

    streaming_router = build_streaming_router(
        thread_repo=repo,
        agent=resolved_agent,
        deps_factory=build_notes_deps_factory(notes),
    )
    app.include_router(streaming_router)

    # Expose for integration tests + future admin endpoints.
    app.state.notes_repo = notes
    app.state.thread_repo = repo
    return app


# Module-level ASGI handle for ``uvicorn notes_app.main:app``.
# Engine.boot() runs in lifespan; the agent stays lazy (constructed on
# the first POST to /threads/{id}/messages).
app: FastAPI = build_app()


def main() -> None:  # pragma: no cover — convenience entry-point
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("notes_app.main:app", host=host, port=port, reload=True)

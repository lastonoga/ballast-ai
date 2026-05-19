"""FastAPI entry point for the notes-app backend (iteration 3).

Wires:
  - in-memory thread repository (no Postgres yet)
  - in-memory note repository bound through ``app.state.container``
    via an ``on_startup`` hook (per spec 4A.0.7 — no module-level
    singletons; see ``docs/superpowers/guides/domain-repo-scaffold.md``).
  - threads router (CRUD)
  - streaming router backed by ``build_notes_runner(...)`` which injects
    per-request ``NoteToolDeps(repo=container.get(NoteRepository),
    tenant_id=...)`` into the pydantic-ai agent so its CRUD tools can act
    on the right tenant.
  - empty provider list — iteration 3 still doesn't need DBOS / Postgres

The app object is lazily built so importing this module never hits
OpenRouter; the agent is only constructed at first request (or eagerly
via ``build_app``).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import UUID

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic_ai_stateflow.api import CORSConfig
from pydantic_ai_stateflow.api.streaming import (
    AgentRunner,
    StreamEvent,
    build_streaming_router,
)
from pydantic_ai_stateflow.api.streaming.router import _PostMessageBody
from pydantic_ai_stateflow.api.threads import build_threads_router
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
    ThreadRepository,
)
from pydantic_ai_stateflow.runtime import Engine

from notes_app.agent import build_agent, build_notes_runner
from notes_app.notes.repository import InMemoryNoteRepository, NoteRepository

load_dotenv()


def _lazy_runner(app: FastAPI) -> AgentRunner:
    """Defer agent construction until the first streamed request.

    Resolves the ``NoteRepository`` from ``app.state.container`` at
    construction time (which is the first request, *after* the
    ``on_startup`` hook has run and bound the repo).

    Lets the app boot in environments without ``OPENROUTER_API_KEY``
    (e.g. the threads-only smoke test path).
    """
    _cached: dict[str, AgentRunner] = {}

    async def _runner(
        *,
        thread_id: UUID,
        run_id: UUID,
        message: _PostMessageBody,
        tenant_id: UUID,
    ) -> AsyncIterator[StreamEvent]:
        if "runner" not in _cached:
            repo = app.state.container.get(NoteRepository)
            _cached["runner"] = build_notes_runner(build_agent(), repo)
        async for event in _cached["runner"](
            thread_id=thread_id,
            run_id=run_id,
            message=message,
            tenant_id=tenant_id,
        ):
            yield event

    return _runner


def build_app(
    *,
    thread_repo: ThreadRepository | None = None,
    agent_runner: AgentRunner | None = None,
    notes_repo: NoteRepository | None = None,
) -> FastAPI:
    """Construct the FastAPI app.

    Args:
      thread_repo: defaults to ``InMemoryThreadRepository``.
      agent_runner: defaults to a lazy OpenRouter-backed runner that
        resolves its ``NoteRepository`` from ``app.state.container``.
        Tests pass a fake here to avoid hitting the network.
      notes_repo: the ``NoteRepository`` to bind into the container.
        Defaults to a fresh ``InMemoryNoteRepository``. Tests may pass
        their own instance to inspect what the agent wrote.
    """
    repo = thread_repo or InMemoryThreadRepository()
    notes = notes_repo or InMemoryNoteRepository()

    engine = Engine(providers=[])
    threads_router = build_threads_router(thread_repo=repo)

    async def _bind_domain_repos(app: FastAPI) -> None:
        """Bind app-level repos onto the framework Container (spec 4A.0.7)."""
        app.state.container.bind(NoteRepository, notes)

    app: FastAPI = engine.fastapi_app(
        extra_routers=[threads_router],
        # F8: permissive_dev() covers the assistant-ui Next.js dev shell
        # (localhost:3000) and the Vite dev port (localhost:3003) so a
        # real browser frontend can hit this backend without a proxy.
        # Replace with an explicit ``CORSConfig(allow_origins=[...])`` in
        # production.
        cors=CORSConfig.permissive_dev(),
        on_startup=[_bind_domain_repos],
    )

    # Streaming router needs a closure over `app` so its runner can
    # resolve the NoteRepository from the container at first request.
    runner = agent_runner or _lazy_runner(app)
    streaming_router = build_streaming_router(
        thread_repo=repo,
        agent_runner=runner,
    )
    app.include_router(streaming_router)

    # Expose the notes repo on app.state so integration tests (and future
    # admin/debug endpoints) can inspect what the agent wrote without
    # round-tripping through the container.
    app.state.notes_repo = notes
    return app


# Module-level ASGI handle for ``uvicorn notes_app.main:app``.
# Built eagerly so uvicorn's lifespan kicks ``Engine.boot()``; the agent
# itself is still lazy (constructed on first /messages request).
app: FastAPI = build_app()


def main() -> None:  # pragma: no cover — convenience entry-point
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("notes_app.main:app", host=host, port=port, reload=True)

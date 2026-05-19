"""FastAPI entry point for the notes-app backend (iteration 2 / 2.1).

Wires:
  - in-memory thread repository (no Postgres yet)
  - threads router (CRUD)
  - streaming router (AG-UI SSE) backed by the framework's `make_runner`
    adapter on top of an OpenRouter-backed pydantic-ai Agent
  - empty provider list — iteration 2 doesn't need DBOS / persistence
    providers

The app object is lazily built so importing this module never hits
OpenRouter; the agent is only constructed at first request (or eagerly
via `build_app`).
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
    make_runner,
)
from pydantic_ai_stateflow.api.streaming.router import _PostMessageBody
from pydantic_ai_stateflow.api.threads import build_threads_router
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
    ThreadRepository,
)
from pydantic_ai_stateflow.runtime import Engine

from notes_app.agent import build_agent

load_dotenv()


def _lazy_runner() -> AgentRunner:
    """Defer agent construction until the first streamed request.

    Lets the app boot in environments without `OPENROUTER_API_KEY`
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
            _cached["runner"] = make_runner(build_agent(), text_field="reply")
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
) -> FastAPI:
    """Construct the FastAPI app.

    Args:
      thread_repo: defaults to `InMemoryThreadRepository`.
      agent_runner: defaults to a lazy OpenRouter-backed `make_runner`.
        Tests pass a fake here to avoid hitting the network.
    """
    repo = thread_repo or InMemoryThreadRepository()
    runner = agent_runner or _lazy_runner()

    engine = Engine(providers=[])
    threads_router = build_threads_router(thread_repo=repo)
    streaming_router = build_streaming_router(
        thread_repo=repo,
        agent_runner=runner,
    )
    app: FastAPI = engine.fastapi_app(
        extra_routers=[threads_router, streaming_router],
        # F8: permissive_dev() covers the assistant-ui Next.js dev shell
        # (localhost:3000) and the Vite dev port (localhost:3003) so a
        # real browser frontend can hit this backend without a proxy.
        # Replace with an explicit `CORSConfig(allow_origins=[...])` in
        # production.
        cors=CORSConfig.permissive_dev(),
    )
    return app


# Module-level ASGI handle for `uvicorn notes_app.main:app`.
# Built eagerly so uvicorn's lifespan kicks Engine.boot(); the agent itself
# is still lazy (constructed on first /messages request).
app: FastAPI = build_app()


def main() -> None:  # pragma: no cover — convenience entry-point
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("notes_app.main:app", host=host, port=port, reload=True)

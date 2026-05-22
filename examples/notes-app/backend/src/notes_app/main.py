"""FastAPI entry point — wires Infra + mounts routes.

App-specific singletons (repos, flows, agents) live in their own
modules and are imported here directly — no constructor DI for app
state. The framework (``sf.create_app``) still owns its own infra
bundle (thread repo, event log, event stream).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pydantic_ai_stateflow as sf
from dbos import DBOSConfig
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic_ai_stateflow.observability.config import ObservabilityConfig
from pydantic_ai_stateflow.persistence import InMemoryEventLogRepository
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
)
from pydantic_ai_stateflow.runtime.event_stream import InProcessEventStream
from pydantic_ai_stateflow.settings import get_settings

from notes_app.agents.notes import notes_agent
from notes_app.agents.todo_approval import approval_agent
from notes_app.repositories.note import notes_repo
from notes_app.routes.notes import build_notes_router
from notes_app.routes.streaming import _AGENT_BY_NAME, router as streaming_router
from notes_app.routes.workflows import router as workflows_router
from notes_app.workflows.brainstorm import brainstorm
from notes_app.workflows.todo_approval import todo_flow


load_dotenv()


def _dbos_db_url() -> str:
    url = get_settings().dbos.database_url
    if url:
        return url
    return f"sqlite:///{Path(tempfile.gettempdir()) / 'notes-app.dbos.sqlite'}"


infra = sf.Infra(
    thread_repo=InMemoryThreadRepository(),
    event_log=InMemoryEventLogRepository(),
    event_stream=InProcessEventStream(),
)

app: FastAPI = sf.create_app(
    infra=infra,
    cors=sf.CORSConfig.permissive_dev(),
    dbos=DBOSConfig(name="notes-app", system_database_url=_dbos_db_url()),
    observability=ObservabilityConfig(
        service_name="app",
        environment="dev",
        instrument_pydantic_ai=True,
        instrument_httpx=True,
        instrument_fastapi=False,
    ),
    extra_routers=[
        build_notes_router(infra.thread_repo),
        workflows_router,
        streaming_router,
    ],
)

# Tests + frontend introspection.
app.state.notes_repo = notes_repo
app.state.notes_agent = notes_agent
app.state.todo_approval_agent = approval_agent
app.state.brainstorm_flow = brainstorm
app.state.todo_flow = todo_flow


def main() -> None:  # pragma: no cover
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("notes_app.main:app", host=host, port=port, reload=True)


__all__ = ["_AGENT_BY_NAME", "app", "main", "notes_repo"]

"""FastAPI entry point — wires app-owned repos into ``sf.create_app``.

App-specific singletons (repos, flows, agents) live in their own
modules and are imported here directly — no constructor DI for app
state. The framework's ``sf.create_app`` takes the thread repo +
event log + event stream as explicit kwargs and stashes them on the
process-wide ``Engine`` (read via ``sf.get_engine()``).
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
from pydantic_ai_stateflow.settings import get_settings

from notes_app.agents.notes import notes_agent
from notes_app.agents.todo_approval import approval_agent
from notes_app.repositories.events import event_log
from notes_app.repositories.note import notes_repo
from notes_app.repositories.thread import thread_repo
from notes_app.routes.notes import build_notes_router
from notes_app.agents import agents
from notes_app.routes.streaming import router as streaming_router
from notes_app.routes.workflows import router as workflows_router
from notes_app.streams import event_stream
from notes_app.workflows.brainstorm import brainstorm
from notes_app.workflows.todo_approval import todo_flow


load_dotenv()


def _dbos_db_url() -> str:
    url = get_settings().dbos.database_url
    if url:
        return url
    return f"sqlite:///{Path(tempfile.gettempdir()) / 'notes-app.dbos.sqlite'}"


app: FastAPI = sf.create_app(
    thread_repo=thread_repo,
    event_log=event_log,
    event_stream=event_stream,
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
        build_notes_router(thread_repo),
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


__all__ = ["agents", "app", "main", "notes_repo"]

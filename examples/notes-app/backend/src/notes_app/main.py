"""FastAPI entry point for the notes-app backend.

Wires the demo via ``sf.create_app(...)`` — no factory pattern, no
``or InMemoryX()`` fallbacks. Tests use ``sf.testing.TestEngine``.

App-specific settings live in ``notes_app.settings``; framework
settings in ``pydantic_ai_stateflow.settings``.
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
from pydantic_ai_stateflow.runtime import (
    InProcessEventStream,
    ThreadEventBroadcaster,
)
from pydantic_ai_stateflow.settings import get_settings

from notes_app.agent import NotesAgent
from notes_app.brainstorm_flow import build_brainstorm_flow
from notes_app.notes import InMemoryNoteRepository
from notes_app.notes.routes import build_notes_router
from notes_app.todo_approval_agent import NotesTodoApprovalAgent
from notes_app.todo_flow import TodoApprovalFlow


load_dotenv()


def _dbos_db_url() -> str:
    url = get_settings().dbos.database_url
    if url:
        return url
    return f"sqlite:///{Path(tempfile.gettempdir()) / 'notes-app.dbos.sqlite'}"


# ── Singletons (explicit construction at module top — no DI magic) ──────

notes_repo = InMemoryNoteRepository()
thread_repo = InMemoryThreadRepository()
event_log = InMemoryEventLogRepository()
event_stream = InProcessEventStream()

todo_flow = TodoApprovalFlow(
    notes_repo=notes_repo,
    thread_repo=thread_repo,
    event_log=event_log,
    event_stream=event_stream,
)

broadcaster = ThreadEventBroadcaster(
    thread_repo=thread_repo, event_log=event_log, event_stream=event_stream,
)

brainstorm = build_brainstorm_flow(todo_flow=todo_flow, broadcaster=broadcaster)

notes_agent = NotesAgent(
    notes_repo=notes_repo,
    thread_repo=thread_repo,
    event_log=event_log,
    event_stream=event_stream,
    todo_flow=todo_flow,
    config_name="notes-app-notes-agent",
)
approval_agent = NotesTodoApprovalAgent(notes_repo=notes_repo)

notes_router = build_notes_router(thread_repo)

app: FastAPI = sf.create_app(
    workflows=[brainstorm],
    agents=[notes_agent, approval_agent],
    thread_repo=thread_repo,
    event_log=event_log,
    event_stream=event_stream,
    dbos=DBOSConfig(name="notes-app", system_database_url=_dbos_db_url()),
    cors=sf.CORSConfig.permissive_dev(),
    observability=ObservabilityConfig(
        service_name="app",
        environment="dev",
        instrument_pydantic_ai=True,
        instrument_httpx=True,
        instrument_fastapi=False,
    ),
    extra_routers=[notes_router],
)

# Tests + frontend introspection.
app.state.notes_repo = notes_repo
app.state.notes_agent = notes_agent
app.state.todo_approval_agent = approval_agent
app.state.todo_flow = todo_flow
app.state.brainstorm_flow = brainstorm


def main() -> None:  # pragma: no cover
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("notes_app.main:app", host=host, port=port, reload=True)

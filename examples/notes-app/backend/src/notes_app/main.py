"""FastAPI entry point for the notes-app backend.

Wires the demo via ``sf.create_app(...)`` — no factory pattern, no
``or InMemoryX()`` fallbacks. Tests use ``sf.testing.TestEngine``
with ``dependency_overrides`` for swapping deps; they do NOT call
``build_app(...)`` (which is kept only as a back-compat shim during
the migration window).
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
from pydantic_ai_stateflow.persistence import (
    EventLogRepository,
    InMemoryEventLogRepository,
)
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
    ThreadRepository,
)
from pydantic_ai_stateflow.runtime import (
    EventStream,
    InProcessEventStream,
    ThreadEventBroadcaster,
)

from notes_app.agent import NotesAgent
from notes_app.brainstorm_flow import BrainstormFlow, build_brainstorm_flow
from notes_app.notes import InMemoryNoteRepository, NoteRepository
from notes_app.notes.routes import build_notes_router
from notes_app.todo_approval_agent import NotesTodoApprovalAgent
from notes_app.todo_flow import TodoApprovalFlow

load_dotenv()


def _dbos_db_url() -> str:
    """Default DBOS system DB path for the notes-app demo (SQLite).

    Honors ``DBOS_DATABASE_URL`` env var when set (e.g. for Postgres).
    Otherwise uses a per-process SQLite file under the system tempdir so
    repeated dev restarts don't accumulate state in the project root.
    """
    override = os.environ.get("DBOS_DATABASE_URL")
    if override:
        return override
    return f"sqlite:///{Path(tempfile.gettempdir()) / 'notes-app.dbos.sqlite'}"


def build_app(
    *,
    thread_repo: ThreadRepository | None = None,
    notes_agent: NotesAgent | None = None,
    notes_repo: NoteRepository | None = None,
    todo_approval_agent: NotesTodoApprovalAgent | None = None,
    todo_flow: TodoApprovalFlow | None = None,
    brainstorm_flow: BrainstormFlow | None = None,
    event_log: EventLogRepository | None = None,
    event_stream: EventStream | None = None,
    manage_dbos_lifecycle: bool = True,
) -> FastAPI:
    """Legacy entry point — wraps ``sf.create_app(...)`` for tests
    that haven't migrated to ``sf.testing.TestEngine`` yet.

    SP1 T10 migrates the test suite; this shim is deleted in SP1 T11.
    """
    repo = thread_repo or InMemoryThreadRepository()
    notes = notes_repo or InMemoryNoteRepository()
    log = event_log or InMemoryEventLogRepository()
    stream = event_stream or InProcessEventStream()
    flow = todo_flow or TodoApprovalFlow(
        notes_repo=notes,
        thread_repo=repo,
        event_log=log,
        event_stream=stream,
    )
    agent = notes_agent or NotesAgent(
        notes_repo=notes,
        thread_repo=repo,
        event_log=log,
        event_stream=stream,
        todo_flow=flow,
        # Stable name so DBOS workflow recovery rebinds the instance
        # to in-flight runs after a process restart.
        config_name="notes-app-notes-agent",
    )
    approval_agent = todo_approval_agent or NotesTodoApprovalAgent(
        notes_repo=notes,
    )
    # Broadcaster shared across long-running workflows that want to
    # push live progress events into the parent thread without the
    # user having an open ``useChat`` stream there. Backed by the
    # same event_log + event_stream as the framework's other SSE-
    # delivered events (``message-added``, ``thread-created``).
    broadcaster = ThreadEventBroadcaster(
        thread_repo=repo, event_log=log, event_stream=stream,
    )
    bstorm = brainstorm_flow or build_brainstorm_flow(
        todo_flow=flow, broadcaster=broadcaster,
    )

    dbos_config = (
        DBOSConfig(name="notes-app", system_database_url=_dbos_db_url())
        if manage_dbos_lifecycle
        else None
    )

    notes_router = build_notes_router(repo)

    app = sf.create_app(
        workflows=[bstorm],
        agents=[agent, approval_agent],
        thread_repo=repo,
        event_log=log,
        event_stream=stream,
        dbos=dbos_config,
        manage_dbos_lifecycle=manage_dbos_lifecycle,
        cors=sf.CORSConfig.permissive_dev(),
        observability=ObservabilityConfig(
            service_name="app",
            environment="dev",
            instrument_pydantic_ai=True,
            instrument_httpx=True,
            # FastAPI route spans pollute traces with one root span
            # per HTTP request — irrelevant for agent observability.
            # Agent runs already have their own spans via
            # ``instrument_pydantic_ai``.
            instrument_fastapi=False,
        ),
        extra_routers=[notes_router],
    )

    # Tests introspect these via ``app.state``.
    app.state.notes_repo = notes
    app.state.thread_repo = repo
    app.state.notes_agent = agent
    app.state.todo_approval_agent = approval_agent
    app.state.todo_flow = flow
    app.state.brainstorm_flow = bstorm
    return app


app: FastAPI = build_app()


def main() -> None:  # pragma: no cover
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("notes_app.main:app", host=host, port=port, reload=True)

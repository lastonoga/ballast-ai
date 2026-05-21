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

To see logfire traces, set ``LOGFIRE_TOKEN`` env var before starting the
server. Without it, telemetry is a no-op.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dbos import DBOSConfig

from pydantic_ai_stateflow.durable import Durable
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic_ai_stateflow.api import CORSConfig
from pydantic_ai_stateflow.api.dbos_router import build_dbos_router
from pydantic_ai_stateflow.api.streaming import build_streaming_router
from pydantic_ai_stateflow.api.threads import build_threads_router
from pydantic_ai_stateflow.observability import ObservabilityProvider
from pydantic_ai_stateflow.persistence import (
    EventLogRepository,
    InMemoryEventLogRepository,
)
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
    ThreadRepository,
)
from pydantic_ai_stateflow.runtime import (
    Engine,
    EventStream,
    EventStreamProvider,
    InProcessEventStream,
    ThreadEventBroadcaster,
    register_agent,
)

from notes_app.agent import NotesAgent
from pydantic_ai_stateflow.api.workflow_router import build_workflow_router

from notes_app.brainstorm_flow import BrainstormFlow, build_brainstorm_flow
from notes_app.notes import InMemoryNoteRepository, NoteRepository
from notes_app.notes.routes import build_notes_router
from notes_app.todo_approval_agent import NotesTodoApprovalAgent
from notes_app.todo_flow import TodoApprovalFlow

load_dotenv()


def _default_dbos_database_url() -> str:
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
    """Construct the FastAPI app.

    DBOS is launched (and destroyed) inside the FastAPI lifespan when
    ``manage_dbos_lifecycle=True`` (the default). Tests that supply
    their own DBOS runtime (see ``tests/conftest.py``) pass
    ``manage_dbos_lifecycle=False`` so the test fixture stays in charge.
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

    register_agent(agent)
    register_agent(approval_agent)

    engine = Engine(
        providers=[
            ObservabilityProvider(
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
            EventStreamProvider(stream=stream, log=log),
        ],
    )

    notes_router = build_notes_router(repo)
    threads_router = build_threads_router(thread_repo=repo)
    streaming_router = build_streaming_router(
        thread_repo=repo,
        event_log=log,
        event_stream=stream,
    )
    # DBOS introspection + control (workflow tree, cancel/resume/fork).
    # Thread-scoped via ``/dbos/threads/{id}/workflows`` (filters by the
    # ``agent-run:{thread_id}:`` prefix that StateflowDurableAgent mints).
    dbos_router = build_dbos_router()
    # Transitional: mount the auto-generated @sf.workflow route directly
    # until T9 rewrites main.py to use sf.create_app(workflows=[bstorm]).
    brainstorm_workflow_router = build_workflow_router(bstorm)

    async def _bind_domain_repos(app: FastAPI) -> None:
        """Bind app-level repos onto the framework Container (spec 4A.0.7)."""
        app.state.container.bind(NoteRepository, notes)

    startup_hooks = [_bind_domain_repos]
    shutdown_hooks: list = []

    if manage_dbos_lifecycle:
        async def _launch_dbos(_app: FastAPI) -> None:
            # ``Durable.init(...)`` registers the singleton;
            # ``Durable.launch()`` starts the workflow runtime. Both must
            # happen before any ``@Durable.workflow`` runs — including
            # the durable ``TodoApprovalFlow.run`` that ``propose_todo``
            # kicks off via ``Durable.start_workflow``.
            Durable.init(
                DBOSConfig(
                    name="notes-app",
                    system_database_url=_default_dbos_database_url(),
                ),
            )
            Durable.launch()

        async def _destroy_dbos(_app: FastAPI) -> None:
            # ``destroy_registry=False`` — leave @DBOS.workflow
            # registrations intact for tests / subsequent boots in the
            # same process.
            Durable.destroy(destroy_registry=False)

        startup_hooks.append(_launch_dbos)
        shutdown_hooks.append(_destroy_dbos)

    app: FastAPI = engine.fastapi_app(
        extra_routers=[
            notes_router, threads_router, streaming_router,
            dbos_router, brainstorm_workflow_router,
        ],
        cors=CORSConfig.permissive_dev(),
        on_startup=startup_hooks,
        on_shutdown=shutdown_hooks,
    )

    app.state.notes_repo = notes
    app.state.thread_repo = repo
    app.state.notes_agent = agent
    app.state.todo_approval_agent = approval_agent
    app.state.todo_flow = flow
    app.state.brainstorm_flow = bstorm
    # ``get_workflow_instance("brainstorm-flow")`` resolves from this dict;
    # T9 will replace with sf.create_app(workflows=[bstorm]) which populates
    # app.state.workflows automatically.
    app.state.workflows = {"brainstorm-flow": bstorm}
    return app


app: FastAPI = build_app()


def main() -> None:  # pragma: no cover
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("notes_app.main:app", host=host, port=port, reload=True)

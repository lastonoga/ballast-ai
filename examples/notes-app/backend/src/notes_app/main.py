"""FastAPI entry point for the notes-app backend.

Wires the demo via ``sf.create_app(infra=...)`` and explicitly mounts
the app-owned routes (``/workflows/brainstorm-flow``,
``/threads/{id}/messages``, ``/threads/{id}/cancel``). The framework
no longer auto-generates streaming / cancel / workflow routes —
apps own their URL shape and dispatch logic.

App-specific settings live in ``notes_app.settings``; framework
settings in ``pydantic_ai_stateflow.settings``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import UUID

import pydantic_ai_stateflow as sf
from dbos import DBOSConfig, SetWorkflowID
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from pydantic_ai_stateflow.durable import Durable
from pydantic_ai_stateflow.errors import ThreadNotFound
from pydantic_ai_stateflow.observability.config import ObservabilityConfig
from pydantic_ai_stateflow.persistence import InMemoryEventLogRepository
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
)
from pydantic_ai_stateflow.runtime.event_stream import InProcessEventStream
from pydantic_ai_stateflow.settings import get_settings

from notes_app.agent import NotesAgent
from notes_app.brainstorm_flow import (
    BrainstormFlow,
    BrainstormTask,
    build_brainstorm_flow,
)
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


# ── Infra bundle (singleton repos + streams) ────────────────────────────

notes_repo = InMemoryNoteRepository()
infra = sf.Infra(
    thread_repo=InMemoryThreadRepository(),
    event_log=InMemoryEventLogRepository(),
    event_stream=InProcessEventStream(),
)

# ── Flows + agents (only app-specific deps in constructors) ─────────────

todo_flow = TodoApprovalFlow(notes_repo=notes_repo)
brainstorm = build_brainstorm_flow(todo_flow=todo_flow)
notes_agent = NotesAgent(
    notes_repo=notes_repo,
    todo_flow=todo_flow,
    config_name="notes-app-notes-agent",
)
approval_agent = NotesTodoApprovalAgent(notes_repo=notes_repo)

# Agent dispatch table — app owns this lookup. The framework no
# longer maintains a registry; ``Thread.agent`` is an opaque string
# that the app resolves to the right ``StateflowAgent`` instance.
_AGENT_BY_NAME = {
    NotesAgent.name: notes_agent,
    NotesTodoApprovalAgent.name: approval_agent,
}

notes_router = build_notes_router(infra.thread_repo)

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
    extra_routers=[notes_router],
)

# Tests + frontend introspection — expose singletons via ``app.state``.
app.state.notes_repo = notes_repo
app.state.notes_agent = notes_agent
app.state.todo_approval_agent = approval_agent
app.state.todo_flow = todo_flow
app.state.brainstorm_flow = brainstorm


# ── App-owned routes ────────────────────────────────────────────────


@app.post("/workflows/brainstorm-flow", response_model=dict)
async def start_brainstorm(
    task: BrainstormTask,
    ctx: sf.RunContext = Depends(sf.get_run_context),
) -> dict:
    """Kick off a brainstorm workflow with a deterministic id.

    Same ``(parent_thread, topic)`` collapses to one in-flight workflow
    (matches the historical ``brainstorm_router.py`` behaviour)."""
    workflow_id = BrainstormFlow.workflow_id(task)
    with SetWorkflowID(workflow_id):
        handle = await Durable.start_workflow(brainstorm.run, ctx, task)
    return {"workflow_id": handle.workflow_id}


@app.post("/threads/{thread_id}/messages")
async def stream_messages(
    request: Request,
    thread_id: UUID,
    ctx: sf.RunContext = Depends(sf.get_run_context),
) -> object:
    """Stream a fresh assistant turn for ``thread_id``.

    Resolves the per-thread agent from the app's dispatch table and
    delegates to the framework's ``stream_response`` primitive (which
    handles body-vs-DB sync, durable / inline dispatch, Vercel-AI
    streaming, approval-resume detection)."""
    thread = await ctx.thread_repo.load(thread_id)
    if thread is None:
        raise ThreadNotFound(
            f"thread {thread_id} not found",
            context={"thread_id": str(thread_id)},
        )
    agent = _AGENT_BY_NAME[thread.agent]
    return await sf.stream_response(
        request=request,
        thread_id=thread_id,
        agent=agent,
        ctx=ctx,
    )


@app.post("/threads/{thread_id}/cancel")
async def cancel_thread(
    thread_id: UUID,
    ctx: sf.RunContext = Depends(sf.get_run_context),
) -> dict:
    """Cancel every active workflow for ``thread_id`` (durable agents only)."""
    thread = await ctx.thread_repo.load(thread_id)
    if thread is None:
        raise ThreadNotFound(
            f"thread {thread_id} not found",
            context={"thread_id": str(thread_id)},
        )
    agent = _AGENT_BY_NAME[thread.agent]
    await sf.cancel_thread_workflows(
        thread_id=thread_id, agent=agent, ctx=ctx,
    )
    return {"cancelled": True}


def main() -> None:  # pragma: no cover
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("notes_app.main:app", host=host, port=port, reload=True)

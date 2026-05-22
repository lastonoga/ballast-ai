"""Streaming + cancellation endpoints for per-thread agent runs.

Resolves the per-thread agent from the app's dispatch table (``_AGENT_BY_NAME``)
and delegates to the framework's ``stream_response`` / ``cancel_thread_workflows``
primitives. Framework infra is reached via ``sf.get_engine()`` — the
process-wide singleton wired by ``sf.create_app`` at startup.
"""

from __future__ import annotations

from uuid import UUID

import pydantic_ai_stateflow as sf
from fastapi import APIRouter, Request
from pydantic_ai_stateflow.errors import ThreadNotFound

from notes_app.agents.notes import NotesAgent, notes_agent
from notes_app.agents.todo_approval import NotesTodoApprovalAgent, approval_agent

router = APIRouter()


# App-owned agent dispatch — framework doesn't have a registry.
# Exported so tests can swap an agent for the duration of a test
# (see ``tests/test_smoke.py::test_threads_crud_and_streaming_fake``).
_AGENT_BY_NAME = {
    NotesAgent.name: notes_agent,
    NotesTodoApprovalAgent.name: approval_agent,
}


@router.post("/threads/{thread_id}/messages")
async def stream_messages(
    request: Request,
    thread_id: UUID,
) -> object:
    """Stream a fresh assistant turn for ``thread_id``.

    Resolves the per-thread agent from the app's dispatch table and
    delegates to the framework's ``stream_response`` primitive."""
    engine = sf.get_engine()
    thread = await engine.thread_repo.load(thread_id)
    if thread is None:
        raise ThreadNotFound(
            f"thread {thread_id} not found",
            context={"thread_id": str(thread_id)},
        )
    agent = _AGENT_BY_NAME[thread.agent]
    return await sf.stream_response(
        request=request, thread_id=thread_id, agent=agent,
    )


@router.post("/threads/{thread_id}/cancel")
async def cancel_thread(
    thread_id: UUID,
) -> dict:
    """Cancel every active workflow for ``thread_id`` (durable agents only)."""
    engine = sf.get_engine()
    thread = await engine.thread_repo.load(thread_id)
    if thread is None:
        raise ThreadNotFound(
            f"thread {thread_id} not found",
            context={"thread_id": str(thread_id)},
        )
    agent = _AGENT_BY_NAME[thread.agent]
    await sf.cancel_thread_workflows(thread_id=thread_id, agent=agent)
    return {"cancelled": True}

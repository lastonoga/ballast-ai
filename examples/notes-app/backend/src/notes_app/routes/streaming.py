"""Streaming + cancellation endpoints for per-thread agent runs.

Resolves the per-thread agent from the app's ``agents`` ``Registry``
(``notes_app.agents.agents``) and delegates to the framework's
``stream_response`` / ``cancel_thread_workflows`` primitives. Framework
infra is reached via ``sf.get_engine()`` — the process-wide singleton
wired by ``sf.create_app`` at startup.
"""

from __future__ import annotations

from uuid import UUID

import pydantic_ai_stateflow as sf
from fastapi import APIRouter, Request
from pydantic_ai_stateflow.errors import ThreadNotFound

from notes_app.agents import agents

router = APIRouter()


@router.post("/threads/{thread_id}/messages")
async def stream_messages(
    request: Request,
    thread_id: UUID,
) -> object:
    """Stream a fresh assistant turn for ``thread_id``.

    Resolves the per-thread agent via the app's ``agents`` registry
    and delegates to the framework's ``stream_response`` primitive.
    """
    engine = sf.get_engine()
    thread = await engine.thread_repo.load(thread_id)
    if thread is None:
        raise ThreadNotFound(
            f"thread {thread_id} not found",
            context={"thread_id": str(thread_id)},
        )
    return await sf.stream_response(
        request=request, thread_id=thread_id, agent=agents.get(thread.agent),
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
    await sf.cancel_thread_workflows(
        thread_id=thread_id, agent=agents.get(thread.agent),
    )
    return {"cancelled": True}

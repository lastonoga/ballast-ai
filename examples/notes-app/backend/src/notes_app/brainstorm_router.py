"""``POST /workflows/brainstorm-todo`` — kicks off a BrainstormFlow run.

Fire-and-forget: starts the DBOS workflow asynchronously and returns
its id immediately. The user-visible result lands via the existing
SSE mechanism — TodoApprovalFlow emits ``thread-created`` into the
parent thread's event log when it spawns the helper, and on user
approve, ``message-added`` lands in the parent.
"""

from __future__ import annotations

from uuid import UUID

from dbos import SetWorkflowID
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pydantic_ai_stateflow import Durable
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository

from notes_app.brainstorm_flow import BrainstormFlow


DEFAULT_TOPIC = "Идеи для todo на эту неделю"


class BrainstormRequest(BaseModel):
    thread_id: UUID = Field(description="Parent thread that will receive the helper")
    topic: str | None = None


class BrainstormResponse(BaseModel):
    workflow_id: str
    parent_thread_id: UUID
    topic: str


def build_brainstorm_router(
    *,
    flow: BrainstormFlow,
    thread_repo: ThreadRepository,
) -> APIRouter:
    """``POST /workflows/brainstorm-todo``.

    Body: ``{thread_id: UUID, topic?: str}``. ``thread_id`` is the
    parent (where the user clicked the button); the workflow spawns a
    helper thread for HITL approval and posts the result back there.
    """
    router = APIRouter()

    @router.post("/workflows/brainstorm-todo", response_model=BrainstormResponse)
    async def start_brainstorm(req: BrainstormRequest) -> BrainstormResponse:
        thread = await thread_repo.load(req.thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="parent thread not found")

        topic = (req.topic or "").strip() or DEFAULT_TOPIC

        # Deterministic-ish workflow id so the same (thread, topic) doesn't
        # spawn parallel duplicates if the user double-clicks. ``parent +
        # topic`` is enough — different topics on the same thread can run
        # concurrently, identical retries collapse to the same workflow.
        workflow_id = f"brainstorm:{req.thread_id}:{abs(hash(topic))}"
        with SetWorkflowID(workflow_id):
            handle = await Durable.start_workflow(
                flow.run, topic=topic, parent_thread_id=req.thread_id,
            )

        return BrainstormResponse(
            workflow_id=handle.workflow_id,
            parent_thread_id=req.thread_id,
            topic=topic,
        )

    return router


__all__ = ["BrainstormRequest", "BrainstormResponse", "build_brainstorm_router"]

"""HTTP entry points for app-owned durable workflows."""

from __future__ import annotations

import pydantic_ai_stateflow as sf
from dbos import SetWorkflowID
from fastapi import APIRouter, Depends
from pydantic_ai_stateflow.durable import Durable

from notes_app.models.brainstorm import BrainstormTask
from notes_app.workflows.brainstorm import BrainstormFlow, brainstorm

router = APIRouter()


@router.post("/workflows/brainstorm-flow", response_model=dict)
async def start_brainstorm(
    task: BrainstormTask,
    ctx: sf.RunContext = Depends(sf.get_run_context),
) -> dict:
    """Kick off a brainstorm workflow with a deterministic id.

    Same ``(parent_thread, topic)`` collapses to one in-flight workflow
    (matches the historical ``brainstorm_router.py`` behaviour)."""
    with SetWorkflowID(BrainstormFlow.workflow_id(task)):
        handle = await Durable.start_workflow(brainstorm.run, ctx, task)
    return {"workflow_id": handle.workflow_id}

"""DBOS introspection + control router.

Exposes DBOS workflow inspection (list / detail / steps) and
re-execution primitives (cancel / resume / fork) as HTTP endpoints so
an operator UI can render the durable execution tree and surgically
re-run failed steps.

Endpoints (all under ``{prefix}``):

  - ``GET    /dbos/threads/{thread_id}/workflows``      → list workflows for a thread
        query: ``?prefix=`` repeatable; defaults to ``[agent-run:{tid}:]``
  - ``GET    /dbos/workflows/{workflow_id}``            → workflow status + IO
  - ``GET    /dbos/workflows/{workflow_id}/steps``      → step list (one level)
  - ``GET    /dbos/workflows/{workflow_id}/children``   → child workflows
        (workflows with parent_workflow_id == this id)
  - ``POST   /dbos/workflows/{workflow_id}/cancel``     → mark cancelled
  - ``POST   /dbos/workflows/{workflow_id}/resume``     → re-execute from where it stopped
  - ``POST   /dbos/workflows/{workflow_id}/fork``       → fork from a specific step
    body: ``{"start_step": int, "queue_name": str?, "queue_partition_key": str?}``

The per-thread listing defaults to the prefix
``agent-run:{thread_id}:`` (set by ``DurableAgent.enqueue_run``)
to scope workflows to a thread. Apps with their own naming scheme
(e.g. ``brainstorm:{tid}:…``) pass extra ``?prefix=`` values to
surface those workflows in the same view. The ``/children`` endpoint
plus ``StepInfo.child_workflow_id`` together let a UI render the
full nested execution tree of patterns like ``DivergentConvergent``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from dbos import DBOS

from ballast.durable import Durable
from fastapi import APIRouter, Query

from ballast.errors import WorkflowNotFound
from pydantic import BaseModel

from ballast.logging import get_logger

if TYPE_CHECKING:
    pass

_log = get_logger(__name__)


def _thread_workflow_prefix(thread_id: UUID) -> str:
    """Same prefix DurableAgent.enqueue_run uses for workflow ids."""
    return f"agent-run:{thread_id}:"


def _wf_to_dict(wf: Any) -> dict[str, Any]:
    """Serialize a DBOS ``WorkflowStatus`` to a JSON-safe dict.

    DBOS's ``WorkflowStatus`` is a plain class (not a pydantic model)
    with attribute access; pickled outputs / exceptions can't go on
    the wire as-is, so we stringify them defensively.
    """
    out: dict[str, Any] = {
        "workflow_id": getattr(wf, "workflow_id", None),
        "status": getattr(wf, "status", None),
        "name": getattr(wf, "name", None),
        "class_name": getattr(wf, "class_name", None),
        "config_name": getattr(wf, "config_name", None),
        "queue_name": getattr(wf, "queue_name", None),
        "queue_partition_key": getattr(wf, "queue_partition_key", None),
        "created_at": getattr(wf, "created_at", None),
        "updated_at": getattr(wf, "updated_at", None),
        "executor_id": getattr(wf, "executor_id", None),
        "app_version": getattr(wf, "app_version", None),
        "parent_workflow_id": getattr(wf, "parent_workflow_id", None),
        "forked_from": getattr(wf, "forked_from", None),
        "was_forked_from": getattr(wf, "was_forked_from", False),
        "recovery_attempts": getattr(wf, "recovery_attempts", None),
    }
    # Output / error: best-effort string repr (real values may be
    # arbitrary Python objects from pickle).
    output = getattr(wf, "output", None)
    if output is not None:
        try:
            out["output"] = repr(output)
        except Exception:
            out["output"] = "<unrepresentable output>"
    error = getattr(wf, "error", None)
    if error is not None:
        out["error"] = f"{type(error).__name__}: {error}"
    return out


def _step_to_dict(step: Any) -> dict[str, Any]:
    """Serialize a DBOS ``StepInfo`` TypedDict to a JSON-safe dict.

    Step output is arbitrary Python; coerce via ``repr`` to keep the
    endpoint's content-type clean.
    """
    raw_output = step.get("output")
    raw_error = step.get("error")
    return {
        "function_id": step.get("function_id"),
        "function_name": step.get("function_name"),
        "child_workflow_id": step.get("child_workflow_id"),
        "started_at_epoch_ms": step.get("started_at_epoch_ms"),
        "completed_at_epoch_ms": step.get("completed_at_epoch_ms"),
        "output": (
            None if raw_output is None
            else (
                raw_output if isinstance(
                    raw_output, (str, int, float, bool, dict, list),
                )
                else repr(raw_output)
            )
        ),
        "error": (
            None if raw_error is None
            else f"{type(raw_error).__name__}: {raw_error}"
        ),
    }


class _ForkBody(BaseModel):
    """Body for ``POST /dbos/workflows/{id}/fork``."""

    start_step: int
    queue_name: str | None = None
    queue_partition_key: str | None = None


def _build_dbos_router() -> APIRouter:
    """Build the DBOS introspection + control router."""
    router = APIRouter()

    @router.get("/dbos/threads/{thread_id}/workflows")
    async def list_thread_workflows(
        thread_id: UUID,
        limit: int = 100,
        offset: int = 0,
        prefix: list[str] | None = Query(
            default=None,
            description=(
                "Repeatable workflow_id prefix filter. Defaults to "
                "['agent-run:{thread_id}:'] — the prefix DurableAgent "
                "mints. Apps with their own naming scheme (e.g. brainstorm "
                "workflows mint 'brainstorm:{thread_id}:…') pass additional "
                "prefixes here to surface those in the same listing."
            ),
        ),
    ) -> list[dict[str, Any]]:
        """List workflows for ``thread_id`` (parent + forked + resumed).

        Defaults to filtering by the ``agent-run:{thread_id}:`` prefix
        that ``DurableAgent`` mints. Pass ``?prefix=`` (one or
        more) to widen the filter — for example
        ``?prefix=agent-run:UUID:&prefix=brainstorm:UUID:`` surfaces
        the agent runs AND the app's brainstorm workflows.
        Newest-first.
        """
        prefixes = prefix if prefix else [_thread_workflow_prefix(thread_id)]
        wfs = await Durable.list_workflows(
            workflow_id_prefix=prefixes,
            sort_desc=True,
            limit=limit,
            offset=offset,
            load_input=False,
            load_output=False,
        )
        return [_wf_to_dict(w) for w in wfs]

    @router.get("/dbos/workflows/{workflow_id}/children")
    async def list_workflow_children(
        workflow_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List workflows whose ``parent_workflow_id == workflow_id``.

        Lets the UI drill from any workflow into its direct children
        without having to know the children's id naming scheme. Use
        in combination with the steps endpoint to render the full
        execution tree (steps + nested workflows) of a pattern like
        ``DivergentConvergent``."""
        wfs = await Durable.list_workflows(
            parent_workflow_id=workflow_id,
            sort_desc=False,  # preserve enqueue order so fan-out reads left-to-right
            limit=limit,
            load_input=False,
            load_output=False,
        )
        return [_wf_to_dict(w) for w in wfs]

    @router.get("/dbos/workflows/{workflow_id}")
    async def get_workflow(workflow_id: str) -> dict[str, Any]:
        """Status + IO for one workflow."""
        wfs = await Durable.list_workflows(
            workflow_ids=[workflow_id],
            limit=1,
            load_input=True,
            load_output=True,
        )
        if not wfs:
            raise WorkflowNotFound(
                f"workflow {workflow_id} not found",
                context={"workflow_id": workflow_id},
            )
        return _wf_to_dict(wfs[0])

    @router.get("/dbos/workflows/{workflow_id}/steps")
    async def list_workflow_steps(
        workflow_id: str,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List recorded steps for one workflow (durable execution log)."""
        steps = await Durable.list_workflow_steps(
            workflow_id, limit=limit, offset=offset,
        )
        return [_step_to_dict(s) for s in steps]

    @router.post("/dbos/workflows/{workflow_id}/cancel")
    async def cancel_workflow(workflow_id: str) -> dict[str, Any]:
        """Mark a workflow as cancelled. Idempotent on terminal workflows."""
        await Durable.cancel_workflow(workflow_id)
        return {"workflow_id": workflow_id, "cancelled": True}

    @router.post("/dbos/workflows/{workflow_id}/resume")
    async def resume_workflow(workflow_id: str) -> dict[str, Any]:
        """Resume execution of a cancelled / interrupted workflow."""
        handle = await Durable.resume_workflow(workflow_id)
        return {
            "workflow_id": workflow_id,
            "resumed": True,
            "handle_workflow_id": getattr(handle, "workflow_id", None),
        }

    @router.post("/dbos/workflows/{workflow_id}/fork")
    async def fork_workflow(
        workflow_id: str, body: _ForkBody,
    ) -> dict[str, Any]:
        """Fork a workflow from a specific step into a new workflow id.

        Useful for "re-run from this step" semantics in the UI:
        completed steps before ``start_step`` are reused; everything
        from ``start_step`` onward re-executes fresh under the new id.
        """
        handle = await Durable.fork_workflow(
            workflow_id,
            body.start_step,
            queue_name=body.queue_name,
            queue_partition_key=body.queue_partition_key,
        )
        return {
            "source_workflow_id": workflow_id,
            "start_step": body.start_step,
            "forked_workflow_id": getattr(handle, "workflow_id", None),
        }

    return router


# ── Module-level router ──────────────────────────────────────────────
# The router captures no per-app deps (Durable is a static facade), so
# it's built once at import.

dbos_router = _build_dbos_router()


__all__ = ["dbos_router"]

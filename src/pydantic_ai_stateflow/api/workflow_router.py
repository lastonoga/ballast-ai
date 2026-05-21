"""Generate ``POST /workflows/{name}`` endpoints from @sf.workflow instances.

For each workflow instance registered in ``app.state.workflows``,
``sf.create_app`` mounts a router built by ``build_workflow_router``.
The handler validates the request body against the workflow's
``input_type``, computes a deterministic workflow id (from
``cls.workflow_id(input)`` if defined, otherwise from a content hash),
starts the durable workflow via ``Durable.start_workflow``, and either:

- returns ``{workflow_id, started_at}`` immediately (default,
  fire-and-forget), OR
- awaits the workflow and returns the output model (``blocking=True``).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from dbos import SetWorkflowID
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from pydantic_ai_stateflow.api.deps import get_workflow_instance
from pydantic_ai_stateflow.durable import Durable
from pydantic_ai_stateflow.runtime.workflows import workflow_metadata


class WorkflowStartResponse(BaseModel):
    """Body returned by the fire-and-forget workflow endpoint."""

    workflow_id: str
    started_at: datetime


def _default_workflow_id(name: str, input_obj: BaseModel) -> str:
    """Default deterministic id: ``<name>:<sha256(json)[:16]>``."""
    json_bytes = input_obj.model_dump_json(exclude_none=False).encode("utf-8")
    digest = hashlib.sha256(json_bytes).hexdigest()[:16]
    return f"{name}:{digest}"


def _resolve_workflow_id(cls: type, name: str, input_obj: BaseModel) -> str:
    """Use ``cls.workflow_id(input)`` if defined, else default hash."""
    custom = getattr(cls, "workflow_id", None)
    if callable(custom):
        return str(custom(input_obj))
    return _default_workflow_id(name, input_obj)


def build_workflow_router(
    instance: object,
    *,
    prefix: str = "/workflows",
) -> APIRouter:
    """Build an APIRouter exposing ``POST {prefix}/{name}`` for one workflow.

    The instance's class must be ``@sf.workflow``-decorated.
    """
    name, input_type, output_type, blocking = workflow_metadata(instance)
    cls = type(instance)

    router = APIRouter()
    resolver = get_workflow_instance(name)

    if blocking:
        # Blocking mode: await the workflow, return the output model.
        #
        # NOTE: ``from __future__ import annotations`` turns all annotations
        # into strings (PEP 563). FastAPI resolves them from the function's
        # ``__annotations__`` dict at route registration time — if it sees
        # the string ``"input_type"`` instead of the actual class it has no
        # schema to build and falls back to treating the parameter as a query
        # param. The fix is to patch ``__annotations__`` after definition so
        # FastAPI sees the live types.
        async def _start_blocking(
            body: Any,
            request: Request,
            wf: Any = Depends(resolver),
        ) -> Any:
            del request  # unused; FastAPI sees the resolver via Depends
            workflow_id = _resolve_workflow_id(cls, name, body)
            with SetWorkflowID(workflow_id):
                handle = await Durable.start_workflow(wf.run, body)
            result = await handle.get_result()
            return result

        _start_blocking.__annotations__ = {
            "body": input_type,
            "request": Request,
            "wf": Any,
            "return": output_type,
        }

        router.add_api_route(
            f"{prefix}/{name}",
            _start_blocking,
            methods=["POST"],
            response_model=output_type,
            name=f"workflow__{name.replace('-', '_')}",
        )
    else:
        # Fire-and-forget: return handle id + start time.
        async def _start(
            body: Any,
            request: Request,
            wf: Any = Depends(resolver),
        ) -> Any:
            del request
            workflow_id = _resolve_workflow_id(cls, name, body)
            started_at = datetime.now(tz=timezone.utc)
            with SetWorkflowID(workflow_id):
                handle = await Durable.start_workflow(wf.run, body)
            return WorkflowStartResponse(
                workflow_id=handle.workflow_id,
                started_at=started_at,
            )

        _start.__annotations__ = {
            "body": input_type,
            "request": Request,
            "wf": Any,
            "return": WorkflowStartResponse,
        }

        router.add_api_route(
            f"{prefix}/{name}",
            _start,
            methods=["POST"],
            response_model=WorkflowStartResponse,
            name=f"workflow__{name.replace('-', '_')}",
        )

    return router


__all__ = ["WorkflowStartResponse", "build_workflow_router"]

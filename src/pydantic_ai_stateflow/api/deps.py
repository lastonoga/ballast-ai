"""FastAPI dependency providers for stateflow apps.

Resolves framework infrastructure from ``request.app.state`` —
populated by ``sf.create_app()``. Routes import these and use
``Depends(get_X)`` in their handler signatures.

Test-time override: standard FastAPI pattern,
``app.dependency_overrides[get_thread_repo] = lambda: my_test_repo``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import HTTPException, Request

from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository

if TYPE_CHECKING:
    from pydantic_ai_stateflow.persistence.events.repository import (
        EventLogRepository,
    )
    from pydantic_ai_stateflow.runtime.event_stream import EventStream


def _get_state(request: Request, attr: str, friendly: str) -> Any:
    val = getattr(request.app.state, attr, None)
    if val is None:
        raise HTTPException(
            status_code=500,
            detail=f"{friendly} not attached to app.state — "
                   "call sf.create_app() or set it explicitly",
        )
    return val


def get_thread_repo(request: Request) -> ThreadRepository:
    """Resolve the ``ThreadRepository`` from ``app.state.thread_repo``."""
    return cast(ThreadRepository, _get_state(request, "thread_repo", "ThreadRepository"))


def get_event_log(request: Request) -> "EventLogRepository":
    """Resolve the ``EventLogRepository`` from ``app.state.event_log``."""
    return cast(
        "EventLogRepository",
        _get_state(request, "event_log", "EventLogRepository"),
    )


def get_event_stream(request: Request) -> "EventStream":
    """Resolve the ``EventStream`` from ``app.state.event_stream``."""
    return cast("EventStream", _get_state(request, "event_stream", "EventStream"))


def get_workflow_instance(name: str):
    """Return a Depends-compatible factory that resolves a workflow
    instance by its kebab-name from ``app.state.workflows``."""

    def _resolver(request: Request) -> Any:
        workflows = getattr(request.app.state, "workflows", None)
        if workflows is None:
            raise HTTPException(
                status_code=500,
                detail="app.state.workflows missing — sf.create_app() "
                       "should have populated it",
            )
        try:
            return workflows[name]
        except KeyError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"No workflow instance registered under {name!r}",
            ) from exc

    _resolver.__name__ = f"get_workflow_instance__{name.replace('-', '_')}"
    return _resolver


def get_agent_instance(name: str):
    """Return a Depends-compatible factory that resolves an agent
    instance by its kebab-name from ``app.state.agents``."""

    def _resolver(request: Request) -> Any:
        agents = getattr(request.app.state, "agents", None)
        if agents is None:
            raise HTTPException(
                status_code=500,
                detail="app.state.agents missing — sf.create_app() "
                       "should have populated it",
            )
        try:
            return agents[name]
        except KeyError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"No agent instance registered under {name!r}",
            ) from exc

    _resolver.__name__ = f"get_agent_instance__{name.replace('-', '_')}"
    return _resolver


__all__ = [
    "get_agent_instance",
    "get_event_log",
    "get_event_stream",
    "get_thread_repo",
    "get_workflow_instance",
]

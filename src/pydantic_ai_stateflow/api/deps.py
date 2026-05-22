"""FastAPI dependency providers for stateflow apps.

Resolves the framework ``Engine`` from ``request.app.state.engine`` —
populated by ``sf.create_app(thread_repo=..., event_log=..., event_stream=...)``.
Routes import these and use ``Depends(get_X)`` in their handler signatures.

Test-time override: standard FastAPI pattern,
``app.dependency_overrides[get_thread_repo] = lambda: my_test_repo``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi import Request

from pydantic_ai_stateflow.errors import ConfigurationInvariantViolation
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository

if TYPE_CHECKING:
    from pydantic_ai_stateflow.persistence.events.repository import (
        EventLogRepository,
    )
    from pydantic_ai_stateflow.runtime.engine import Engine
    from pydantic_ai_stateflow.runtime.event_stream import EventStream


def get_engine_dep(request: Request) -> "Engine":
    """Resolve the ``Engine`` from ``app.state.engine``."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise ConfigurationInvariantViolation(
            "Engine not attached to app.state",
            hint="call sf.create_app(thread_repo=..., event_log=..., event_stream=...) at app construction",
            context={"attr": "engine"},
        )
    return engine


def get_thread_repo(request: Request) -> ThreadRepository:
    """Resolve the ``ThreadRepository`` from the Engine."""
    return cast(ThreadRepository, get_engine_dep(request).thread_repo)


def get_event_log(request: Request) -> "EventLogRepository":
    """Resolve the ``EventLogRepository`` from the Engine."""
    return cast("EventLogRepository", get_engine_dep(request).event_log)


def get_event_stream(request: Request) -> "EventStream":
    """Resolve the ``EventStream`` from the Engine."""
    return cast("EventStream", get_engine_dep(request).event_stream)


__all__ = [
    "get_engine_dep",
    "get_event_log",
    "get_event_stream",
    "get_thread_repo",
]

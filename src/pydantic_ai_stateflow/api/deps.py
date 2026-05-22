"""FastAPI dependency providers for stateflow apps.

Resolves framework infrastructure from ``request.app.state.infra`` —
populated by ``sf.create_app(infra=...)``. Routes import these and use
``Depends(get_X)`` in their handler signatures.

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
    from pydantic_ai_stateflow.runtime.event_stream import EventStream
    from pydantic_ai_stateflow.runtime.infra import Infra, RunContext


def get_infra(request: Request) -> "Infra":
    """Resolve the ``Infra`` bundle from ``app.state.infra``."""
    infra = getattr(request.app.state, "infra", None)
    if infra is None:
        raise ConfigurationInvariantViolation(
            "Infra not attached to app.state",
            hint="call sf.create_app(infra=...) at app construction",
            context={"attr": "infra"},
        )
    return infra


def get_run_context(request: Request) -> "RunContext":
    """Mint a per-request ``RunContext`` from ``app.state.infra``.

    No per-call fields are set (``parent_thread_id`` / ``workflow_id``
    are ``None``). Apps that need them can build their own context via
    ``ctx.with_(...)`` or ``infra.context(...)`` inside the handler.
    """
    return get_infra(request).context()


def get_thread_repo(request: Request) -> ThreadRepository:
    """Resolve the ``ThreadRepository`` from the Infra bundle."""
    return cast(ThreadRepository, get_infra(request).thread_repo)


def get_event_log(request: Request) -> "EventLogRepository":
    """Resolve the ``EventLogRepository`` from the Infra bundle."""
    return cast("EventLogRepository", get_infra(request).event_log)


def get_event_stream(request: Request) -> "EventStream":
    """Resolve the ``EventStream`` from the Infra bundle."""
    return cast("EventStream", get_infra(request).event_stream)


__all__ = [
    "get_event_log",
    "get_event_stream",
    "get_infra",
    "get_run_context",
    "get_thread_repo",
]

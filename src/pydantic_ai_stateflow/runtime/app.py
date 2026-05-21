"""``sf.create_app()`` — the canonical entry point for stateflow apps.

Replaces the old ``Engine`` + ``build_app`` machinery. Builds a FastAPI
app with:

- ``app.state.{thread_repo, event_log, event_stream}`` populated
- ``app.state.workflows[name] = instance`` for each @sf.workflow
- ``app.state.agents[name] = instance`` for each @sf.stateflow_agent
- Built-in routers mounted (health, threads, streaming, dbos)
- One auto-generated workflow router per workflow instance
  (``POST /workflows/<name>``)
- DBOS launched + destroyed via FastAPI lifespan
- ObservabilityConfig.install() called before any of the above

No DI container. No Provider list. Apps pass instances directly.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI

from pydantic_ai_stateflow.api.dbos_router import dbos_router
from pydantic_ai_stateflow.api.health import build_health_router, health_router
from pydantic_ai_stateflow.api.streaming.router import streaming_router
from pydantic_ai_stateflow.api.threads import threads_router
from pydantic_ai_stateflow.api.workflow_router import build_workflow_router
from pydantic_ai_stateflow.durable import Durable
from pydantic_ai_stateflow.observability.config import ObservabilityConfig
from pydantic_ai_stateflow.persistence import (
    InMemoryEventLogRepository,
    InMemoryThreadRepository,
)
from pydantic_ai_stateflow.runtime.agents import StateflowAgent
from pydantic_ai_stateflow.runtime.event_stream import InProcessEventStream
from pydantic_ai_stateflow.runtime.workflows import workflow_metadata

if TYPE_CHECKING:
    from dbos import DBOSConfig

    from pydantic_ai_stateflow.api.cors import CORSConfig
    from pydantic_ai_stateflow.persistence.events.repository import (
        EventLogRepository,
    )
    from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository
    from pydantic_ai_stateflow.runtime.event_stream import EventStream

LifespanHook = Callable[[FastAPI], Awaitable[None]]

_logger = logging.getLogger("pydantic_ai_stateflow.app")


def create_app(
    *,
    # Construction targets — INSTANCES ONLY (no classes, no factories).
    workflows: Sequence[object] = (),
    agents: Sequence[StateflowAgent] = (),
    # Cross-cutting infra (sane defaults; apps override)
    thread_repo: "ThreadRepository | None" = None,
    event_log: "EventLogRepository | None" = None,
    event_stream: "EventStream | None" = None,
    # DBOS
    dbos: "DBOSConfig | None" = None,
    manage_dbos_lifecycle: bool = True,
    # Observability
    observability: ObservabilityConfig | None = None,
    # HTTP
    cors: "CORSConfig | None" = None,
    extra_routers: Sequence[APIRouter] = (),
    health_checks: dict[str, Callable[[], Awaitable[bool]]] | None = None,
    on_startup: Sequence[LifespanHook] = (),
    on_shutdown: Sequence[LifespanHook] = (),
) -> FastAPI:
    """Construct the FastAPI app for a Stateflow service.

    Order of operations:
    1. ``ObservabilityConfig.install()`` configures Logfire (no-op if
       already installed with same config).
    2. Repos default to InMemory variants when not supplied.
    3. ``app.state`` populated with repos for ``Depends(get_*)`` resolution.
    4. For each instance in ``workflows=``: read decorator metadata,
       store in ``app.state.workflows[name]``, mount auto-generated router.
    5. For each instance in ``agents=``: read decorator metadata, store
       in ``app.state.agents[name]``.
    6. Built-in routers mounted: health, threads, streaming, dbos.
       Then per-workflow routers. Then ``extra_routers``.
    7. Lifespan registered: launches DBOS on startup, destroys on shutdown,
       runs caller's ``on_startup`` / ``on_shutdown`` hooks.
    8. ``observability.instrument_app(app)`` attaches FastAPI integration.
    """
    # 1. Observability first.
    if observability is not None:
        observability.install()

    # 2. Default repos.
    resolved_thread_repo = thread_repo or InMemoryThreadRepository()
    resolved_event_log = event_log or InMemoryEventLogRepository()
    resolved_event_stream = event_stream or InProcessEventStream()

    # 7. Lifespan.
    startup_hooks: list[LifespanHook] = list(on_startup)
    shutdown_hooks: list[LifespanHook] = list(on_shutdown)

    if manage_dbos_lifecycle and dbos is not None:
        async def _launch_dbos(_app: FastAPI) -> None:
            Durable.init(dbos)
            Durable.launch()

        async def _destroy_dbos(_app: FastAPI) -> None:
            Durable.destroy(destroy_registry=False)

        startup_hooks.insert(0, _launch_dbos)
        shutdown_hooks.append(_destroy_dbos)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        for hook in startup_hooks:
            try:
                await hook(_app)
            except Exception:
                _logger.exception(
                    "startup hook %r raised; aborting boot",
                    getattr(hook, "__qualname__", repr(hook)),
                )
                raise
        try:
            yield
        finally:
            for hook in reversed(shutdown_hooks):
                try:
                    await hook(_app)
                except Exception:
                    _logger.exception(
                        "shutdown hook %r raised; continuing",
                        getattr(hook, "__qualname__", repr(hook)),
                    )

    app = FastAPI(lifespan=_lifespan)

    # 3. app.state population.
    app.state.thread_repo = resolved_thread_repo
    app.state.event_log = resolved_event_log
    app.state.event_stream = resolved_event_stream
    app.state.workflows = {}
    app.state.agents = {}

    # 4. Workflow instances.
    for instance in workflows:
        name, _input_t, _output_t, _blocking = workflow_metadata(instance)
        if name in app.state.workflows:
            raise ValueError(
                f"Duplicate workflow instance for {name!r}: "
                f"{app.state.workflows[name]!r} and {instance!r}",
            )
        app.state.workflows[name] = instance

    # 5. Agent instances.
    for agent_instance in agents:
        agent_cls = type(agent_instance)
        agent_name = getattr(agent_cls, "name", None)
        if agent_name is None:
            raise TypeError(
                f"Agent instance {agent_instance!r} (class {agent_cls.__name__}) "
                f"has no ``name`` ClassVar — was @sf.stateflow_agent applied?",
            )
        if agent_name in app.state.agents:
            raise ValueError(
                f"Duplicate agent instance for {agent_name!r}",
            )
        app.state.agents[agent_name] = agent_instance

    # 6. Mount built-in routers.
    if health_checks is not None:
        app.include_router(build_health_router(checks=health_checks))
    else:
        app.include_router(health_router)
    app.include_router(threads_router)
    app.include_router(streaming_router)
    app.include_router(dbos_router)

    # 6b. Auto-generated workflow routers.
    for instance in workflows:
        app.include_router(build_workflow_router(instance))

    # 6c. Extra routers from the app.
    for r in extra_routers:
        app.include_router(r)

    # CORS.
    if cors is not None:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors.allow_origins),
            allow_methods=list(cors.allow_methods),
            allow_headers=list(cors.allow_headers),
            allow_credentials=cors.allow_credentials,
            expose_headers=list(cors.expose_headers),
            max_age=cors.max_age,
        )

    # 8. FastAPI observability instrumentation.
    if observability is not None:
        observability.instrument_app(app)

    # SP2 — install structured error handlers.
    from pydantic_ai_stateflow.api.error_middleware import install_error_handlers
    install_error_handlers(app)

    return app


__all__ = ["create_app"]

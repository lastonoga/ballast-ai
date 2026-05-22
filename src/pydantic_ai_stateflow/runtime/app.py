"""``sf.create_app()`` — the canonical entry point for stateflow apps.

Builds a FastAPI app with:

- ``app.state.infra = infra`` (apps read ``request.app.state.infra``)
- Built-in routers mounted (health, threads, dbos)
- DBOS launched + destroyed via FastAPI lifespan
- ``ObservabilityConfig.install()`` called before any of the above

No DI container, no auto-generated workflow routes, no agent registry.
Apps own their own routes (``stream_response`` primitive available for
streaming endpoints) and pass the ``Infra`` bundle in once at startup.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI

from pydantic_ai_stateflow.api.dbos_router import dbos_router
from pydantic_ai_stateflow.api.health import build_health_router, health_router
from pydantic_ai_stateflow.api.threads import threads_router
from pydantic_ai_stateflow.durable import Durable
from pydantic_ai_stateflow.observability.config import ObservabilityConfig

if TYPE_CHECKING:
    from dbos import DBOSConfig

    from pydantic_ai_stateflow.api.cors import CORSConfig
    from pydantic_ai_stateflow.runtime.infra import Infra

LifespanHook = Callable[[FastAPI], Awaitable[None]]

_logger = logging.getLogger("pydantic_ai_stateflow.app")


def create_app(
    *,
    # Cross-cutting infrastructure bundle (repos + event log + stream).
    infra: "Infra",
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
    2. ``app.state.infra = infra`` so dependency providers + apps can
       reach the cross-cutting singletons.
    3. Built-in routers mounted: health, threads, dbos. Then
       ``extra_routers`` (apps mount their own streaming/cancel/etc).
    4. Lifespan registered: launches DBOS on startup, destroys on shutdown,
       runs caller's ``on_startup`` / ``on_shutdown`` hooks.
    5. ``observability.instrument_app(app)`` attaches FastAPI integration.
    """
    # 1. Observability first.
    if observability is not None:
        observability.install()

    # 2. Lifespan.
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
    app.state.infra = infra

    # 4. Mount built-in routers.
    if health_checks is not None:
        app.include_router(build_health_router(checks=health_checks))
    else:
        app.include_router(health_router)
    app.include_router(threads_router)
    app.include_router(dbos_router)

    # 4b. Extra routers from the app.
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

    # 5. FastAPI observability instrumentation.
    if observability is not None:
        observability.instrument_app(app)

    # SP2 — install structured error handlers.
    from pydantic_ai_stateflow.api.error_middleware import install_error_handlers
    install_error_handlers(app)

    return app


__all__ = ["create_app"]

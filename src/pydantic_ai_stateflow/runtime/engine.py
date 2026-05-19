from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, TypeVar

from fastapi import APIRouter, FastAPI

from pydantic_ai_stateflow.runtime.container import Container, DefaultContainer
from pydantic_ai_stateflow.runtime.provider import ServiceProvider

if TYPE_CHECKING:
    from pydantic_ai_stateflow.api.cors import CORSConfig

T = TypeVar("T")

Invariant = Callable[[Container], Awaitable[None]]
LifespanHook = Callable[[FastAPI], Awaitable[None]]

_logger = logging.getLogger("pydantic_ai_stateflow.engine")


class EngineInvariantViolation(Exception):  # noqa: N818
    """Raised when a bootstrap-time invariant check fails.

    Engine.boot propagates this so the application start fails fast
    instead of running with a broken configuration (per spec 4H).
    """


class Engine:
    """Orchestrator: registers providers + runs bootstrap invariants.

    Per spec 4H:
    - Providers register in user-declared order (no auto-DAG)
    - All providers register before any invariants run
    - Invariants raise EngineInvariantViolation to abort startup
    - Container is exposed publicly so FastAPI / CLI callers can
      `Depends(get_container)` rather than reaching for a global.
    """

    def __init__(
        self,
        *,
        providers: list[ServiceProvider],
        invariants: list[Invariant] | None = None,
        container: Container | None = None,
    ) -> None:
        self.container: Container = container if container is not None else DefaultContainer()
        self._providers = list(providers)
        self._invariants = list(invariants or [])
        self._booted = False

    async def boot(self) -> None:
        if self._booted:
            raise RuntimeError("Engine already booted")
        # Lazy import to avoid a cycle: observability.provider imports
        # EngineInvariantViolation from this module.
        from pydantic_ai_stateflow.observability.provider import ObservabilityProvider

        for provider in self._providers:
            # Spec 4H: ObservabilityProvider must be first. Before invoking
            # any non-Observability provider, set a tripwire on the container
            # — if ObservabilityProvider runs later it will detect this and
            # raise EngineInvariantViolation.
            if not isinstance(provider, ObservabilityProvider) and not getattr(
                self.container,
                "_observability_registered",
                False,
            ):
                self.container._observability_first_violated = True  # type: ignore[attr-defined]
            await provider.register(self.container)
        for invariant in self._invariants:
            await invariant(self.container)
        self._booted = True

    def fastapi_app(
        self,
        *,
        extra_routers: list[APIRouter] | None = None,
        health_checks: dict[str, Callable[[], Awaitable[bool]]] | None = None,
        cors: CORSConfig | None = None,
        on_startup: list[LifespanHook] | None = None,
        on_shutdown: list[LifespanHook] | None = None,
    ) -> FastAPI:
        """Build a FastAPI app with the Container/Engine wired in.

        - Attaches `app.state.container` and `app.state.engine` (spec 4A.0.7).
        - Registers a lifespan that calls `engine.boot()` once (idempotent
          guard against double-boot on re-entry).
        - Mounts `/healthz` by default.
        - Mounts any `extra_routers` provided.

        Optional knobs (additive — defaults preserve prior behaviour):

        - ``cors``: install Starlette's ``CORSMiddleware`` from a
          :class:`pydantic_ai_stateflow.api.CORSConfig`. When ``None``
          (the default) no CORS middleware is installed and cross-origin
          browser requests will be blocked by the browser.
        - ``on_startup``: list of ``async def hook(app) -> None``
          callables awaited (in declared order) *after* ``engine.boot()``.
          A hook exception is logged and re-raised so startup fails fast.
        - ``on_shutdown``: list of ``async def hook(app) -> None``
          callables awaited in REVERSE order during shutdown. Exceptions
          are logged but swallowed so later hooks still run.

        Observability is NOT auto-attached here — install via
        `ObservabilityProvider` in the provider list (spec 4H: provider order).
        """
        from pydantic_ai_stateflow.api.health import build_health_router

        startup_hooks: list[LifespanHook] = list(on_startup or [])
        shutdown_hooks: list[LifespanHook] = list(on_shutdown or [])

        @asynccontextmanager
        async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
            if not self._booted:
                await self.boot()
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
                            "shutdown hook %r raised; continuing with "
                            "remaining hooks",
                            getattr(hook, "__qualname__", repr(hook)),
                        )

        app = FastAPI(lifespan=_lifespan)
        app.state.container = self.container
        app.state.engine = self
        app.include_router(build_health_router(checks=health_checks))
        for r in extra_routers or []:
            app.include_router(r)
        if cors is not None:
            # Import lazily so the CORS surface stays optional at
            # import-time and matches the rest of the runtime layer.
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
        return app

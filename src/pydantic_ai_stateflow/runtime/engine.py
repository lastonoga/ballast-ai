from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypeVar

from fastapi import APIRouter, FastAPI

from pydantic_ai_stateflow.runtime.container import Container, DefaultContainer
from pydantic_ai_stateflow.runtime.provider import ServiceProvider

T = TypeVar("T")

Invariant = Callable[[Container], Awaitable[None]]


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
        for provider in self._providers:
            await provider.register(self.container)
        for invariant in self._invariants:
            await invariant(self.container)
        self._booted = True

    def fastapi_app(
        self,
        *,
        extra_routers: list[APIRouter] | None = None,
        health_checks: dict[str, Callable[[], Awaitable[bool]]] | None = None,
    ) -> FastAPI:
        """Build a FastAPI app with the Container/Engine wired in.

        - Attaches `app.state.container` and `app.state.engine` (spec 4A.0.7).
        - Registers a lifespan that calls `engine.boot()` once (idempotent
          guard against double-boot on re-entry).
        - Mounts `/healthz` by default.
        - Mounts any `extra_routers` provided.

        Observability is NOT auto-attached here — install via
        `ObservabilityProvider` in the provider list (spec 4H: provider order).
        """
        from pydantic_ai_stateflow.api.health import build_health_router

        @asynccontextmanager
        async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
            if not self._booted:
                await self.boot()
            yield

        app = FastAPI(lifespan=_lifespan)
        app.state.container = self.container
        app.state.engine = self
        app.include_router(build_health_router(checks=health_checks))
        for r in extra_routers or []:
            app.include_router(r)
        return app

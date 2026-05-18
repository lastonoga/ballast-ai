from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

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

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from pydantic_ai_stateflow.runtime.engine import EngineInvariantViolation

if TYPE_CHECKING:
    from fastapi import FastAPI
    from sqlalchemy.ext.asyncio import AsyncEngine

    from pydantic_ai_stateflow.runtime.container import Container


def has_logfire() -> bool:
    """Soft import — True iff `logfire` is importable in this process."""
    try:
        mod = importlib.import_module("logfire")
        return mod is not None
    except Exception:
        return False


class ObservabilityProvider:
    """Configures logfire (when present) and registers the `must-be-first`
    bootstrap invariant.

    Soft dependency: if `logfire` is not installed, every method is a no-op
    so the test suite (and applications that don't want telemetry) keep
    working. Spec 4D, 4H.
    """

    def __init__(
        self,
        *,
        service_name: str = "pydantic-ai-stateflow",
        environment: str = "dev",
        instrument_pydantic_ai: bool = True,
        instrument_httpx: bool = True,
        instrument_fastapi_app: FastAPI | None = None,
        instrument_sqlalchemy_engine: AsyncEngine | None = None,
        configure_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.service_name = service_name
        self.environment = environment
        self._instr_pai = instrument_pydantic_ai
        self._instr_httpx = instrument_httpx
        self._fastapi_app = instrument_fastapi_app
        self._sa_engine = instrument_sqlalchemy_engine
        self._configure_kwargs = dict(configure_kwargs or {})

    async def register(self, container: Container) -> None:
        # Spec 4H invariant — observability registers FIRST. Engine.boot()
        # sets `_observability_first_violated = True` on the container before
        # invoking any non-ObservabilityProvider; if we see that flag, we
        # ran after at least one other provider and must abort.
        if getattr(container, "_observability_first_violated", False):
            raise EngineInvariantViolation(
                "ObservabilityProvider must register first (spec 4H).",
            )
        if not has_logfire():
            # mark so subsequent providers don't trip the invariant
            container._observability_registered = True  # type: ignore[attr-defined]
            return
        import logfire  # noqa: WPS433  (soft import)

        logfire.configure(
            service_name=self.service_name,
            environment=self.environment,
            **self._configure_kwargs,
        )
        if self._instr_pai and hasattr(logfire, "instrument_pydantic_ai"):
            logfire.instrument_pydantic_ai()
        if self._instr_httpx and hasattr(logfire, "instrument_httpx"):
            logfire.instrument_httpx()
        if self._fastapi_app is not None and hasattr(logfire, "instrument_fastapi"):
            logfire.instrument_fastapi(self._fastapi_app)
        if self._sa_engine is not None and hasattr(logfire, "instrument_sqlalchemy"):
            logfire.instrument_sqlalchemy(engine=self._sa_engine)
        container._observability_registered = True  # type: ignore[attr-defined]

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any

from pydantic_ai_stateflow.logging import get_logger
from pydantic_ai_stateflow.observability.cost import (
    CostExtractor,
    configure_cost_extractors,
)
from pydantic_ai_stateflow.runtime.engine import EngineInvariantViolation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastapi import FastAPI
    from sqlalchemy.ext.asyncio import AsyncEngine

    from pydantic_ai_stateflow.runtime.container import Container

_logger = get_logger(__name__)


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
        instrument_fastapi: bool = True,
        instrument_fastapi_app: FastAPI | None = None,
        instrument_sqlalchemy_engine: AsyncEngine | None = None,
        configure_kwargs: dict[str, Any] | None = None,
        cost_extractors: Sequence[CostExtractor] | None = None,
    ) -> None:
        self.service_name = service_name
        self.environment = environment
        self._instr_pai = instrument_pydantic_ai
        self._instr_httpx = instrument_httpx
        self._instr_fastapi = instrument_fastapi
        # ``instrument_fastapi_app`` is back-compat only — callers that
        # constructed the FastAPI app before the provider can still pass
        # it explicitly. The preferred path is ``Engine.fastapi_app`` →
        # ``ObservabilityProvider.instrument_app`` (called automatically
        # post-construction).
        self._fastapi_app = instrument_fastapi_app
        self._sa_engine = instrument_sqlalchemy_engine
        self._configure_kwargs = dict(configure_kwargs or {})
        # ``None`` → framework default (OpenRouterCostExtractor). Pass an
        # explicit sequence (including ``[]``) to override. Extractors are
        # used by the ``ModelResponse.cost`` fallback patch — they
        # supply a real billed cost from upstream when genai-prices'
        # static catalogue doesn't know the model.
        self._cost_extractors = cost_extractors
        # Idempotency tracking.
        self._logfire_configured = False
        self._fastapi_instrumented = False

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
            _logger.info(
                "ObservabilityProvider: logfire not installed, telemetry is a no-op",
            )
            # mark so subsequent providers don't trip the invariant
            container._observability_registered = True  # type: ignore[attr-defined]
            return
        import logfire  # noqa: WPS433  (soft import)

        logfire.configure(
            service_name=self.service_name,
            environment=self.environment,
            **self._configure_kwargs,
        )
        self._logfire_configured = True
        token_present = bool(os.environ.get("LOGFIRE_TOKEN"))
        if token_present:
            _logger.info(
                "logfire configured (service_name=%s, environment=%s, token=present)",
                self.service_name,
                self.environment,
            )
        else:
            _logger.info(
                "logfire configured (service_name=%s, environment=%s, token=MISSING) "
                "— set LOGFIRE_TOKEN to ship telemetry; otherwise spans are dropped",
                self.service_name,
                self.environment,
            )
        if self._instr_pai and hasattr(logfire, "instrument_pydantic_ai"):
            try:
                logfire.instrument_pydantic_ai()
            except Exception:
                _logger.warning(
                    "logfire.instrument_pydantic_ai failed",
                    exc_info=True,
                )
        if self._instr_httpx and hasattr(logfire, "instrument_httpx"):
            try:
                logfire.instrument_httpx()
            except Exception:
                _logger.warning(
                    "logfire.instrument_httpx failed — install "
                    "'logfire[httpx]' (or opentelemetry-instrumentation-httpx) "
                    "to enable HTTP-client spans",
                    exc_info=True,
                )
        if self._fastapi_app is not None and hasattr(logfire, "instrument_fastapi"):
            try:
                logfire.instrument_fastapi(self._fastapi_app)
                self._fastapi_instrumented = True
            except Exception:
                _logger.warning(
                    "logfire.instrument_fastapi failed — install "
                    "'logfire[fastapi]' to enable HTTP-server spans",
                    exc_info=True,
                )
        if self._sa_engine is not None and hasattr(logfire, "instrument_sqlalchemy"):
            try:
                logfire.instrument_sqlalchemy(engine=self._sa_engine)
            except Exception:
                _logger.warning(
                    "logfire.instrument_sqlalchemy failed",
                    exc_info=True,
                )

        # Cost-fallback patch on ``ModelResponse.cost`` — makes the
        # ``operation.cost`` span attribute populate with the upstream-
        # reported billed cost when genai-prices doesn't know the model.
        # See ``pydantic_ai_stateflow.observability.cost`` for the
        # extractor strategy contract.
        configure_cost_extractors(self._cost_extractors)

        container._observability_registered = True  # type: ignore[attr-defined]

    def instrument_app(self, app: FastAPI) -> None:
        """Instrument a FastAPI app with logfire after construction.

        Called by ``Engine.fastapi_app`` once the app instance exists,
        so callers don't have to plumb the app back into the provider's
        constructor. Idempotent — calling twice (or after a prior
        ``register`` that already passed ``instrument_fastapi_app``)
        is a no-op.

        Soft no-op when ``logfire`` isn't installed or
        ``instrument_fastapi=False`` was set on the provider.
        """
        if not self._instr_fastapi:
            return
        if self._fastapi_instrumented:
            return
        if not has_logfire():
            return
        import logfire  # noqa: WPS433  (soft import)

        if not hasattr(logfire, "instrument_fastapi"):
            return
        try:
            logfire.instrument_fastapi(app)
        except Exception:
            # Logfire raises if the FastAPI integration extra isn't
            # installed (``logfire[fastapi]``). Don't break startup —
            # log loudly and continue without HTTP-level spans.
            _logger.warning(
                "logfire.instrument_fastapi failed — HTTP spans disabled. "
                "Install 'logfire[fastapi]' (or opentelemetry-instrumentation-fastapi) "
                "to enable.",
                exc_info=True,
            )
            return
        self._fastapi_instrumented = True
        _logger.info("logfire.instrument_fastapi attached to FastAPI app")

"""``GoalDriftDetector`` — agent surface for Goal Drift Detection."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext

from ballast.capabilities.base import BallastCapability
from ballast.drift._core import DriftEngine
from ballast.drift._handlers import GoalDriftError
from ballast.drift._protocols import DriftCheckSignal, DriftContext

_log = logging.getLogger("ballast.drift.capability")

MetadataProvider = Callable[[RunContext[Any] | None, ModelRequestContext], dict[str, Any]]


def _empty_metadata(
    _ctx: RunContext[Any] | None,
    _request_context: ModelRequestContext,
) -> dict[str, Any]:
    return {}


class GoalDriftDetector(BallastCapability):
    """Per-step drift monitor wrapping a ``DriftEngine``.

    Per-run isolation via ``for_run`` (counters live on the clone, not the
    base instance). The base instance carries the (immutable) ``engine``
    + ``metadata_provider``.

    Hook strategy:
      - ``before_model_request`` — starts the monotonic clock (idempotent).
      - ``after_model_request`` — increments counters from the response
        (tool call count, tokens), constructs ``DriftCheckSignal`` +
        ``DriftContext``, calls ``engine.maybe_check``. Engine exceptions
        are swallowed (drift detection is a sidecar — must not crash agent).
        ``GoalDriftError`` is the ONE exception that propagates, by design.
    """

    name = "goal_drift_detector"

    def __init__(
        self, *,
        engine: DriftEngine,
        metadata_provider: MetadataProvider = _empty_metadata,
    ) -> None:
        self._engine = engine
        self._metadata_provider = metadata_provider
        # Per-run state (only populated on the clone returned by for_run)
        self._step_index = 0
        self._tool_calls = 0
        self._tokens_used = 0
        self._started_at: float | None = None

    async def for_run(self, ctx: RunContext[Any]) -> GoalDriftDetector:
        return GoalDriftDetector(
            engine=self._engine,
            metadata_provider=self._metadata_provider,
        )

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        # NOTE: positional (not keyword-only) per pydantic-ai AbstractCapability signature
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        if self._started_at is None:
            self._started_at = time.monotonic()
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        self._step_index += 1
        for part in response.parts:
            if isinstance(part, ToolCallPart):
                self._tool_calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._tokens_used += (
                getattr(usage, "input_tokens", 0)
                + getattr(usage, "output_tokens", 0)
            )

        signal = DriftCheckSignal(
            step_index=self._step_index,
            tool_calls=self._tool_calls,
            tokens_used=self._tokens_used,
            seconds_elapsed=(
                time.monotonic() - self._started_at if self._started_at else 0.0
            ),
        )
        drift_ctx = DriftContext(
            messages=list(request_context.messages),
            run_ctx=ctx,
            workflow_input=None,
            metadata=self._metadata_provider(ctx, request_context),
        )

        try:
            await self._engine.maybe_check(signal, drift_ctx)
        except GoalDriftError:
            raise  # intentional hard stop
        except Exception:
            _log.exception("drift engine failed in after_model_request (swallowed)")

        return response


__all__ = ["GoalDriftDetector", "MetadataProvider"]

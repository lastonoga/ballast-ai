"""Built-in ``DriftHandler`` implementations + ``GoalDriftError``.

Apps choose what happens on drift via one or more handlers:

- ``LogOnly`` — write WARNING; never blocks.
- ``EmitDriftEvent(sink)`` — push a structured event to a caller-supplied
  async sink (e.g., a thread-event publisher).
- ``RaiseDriftError`` — raise ``GoalDriftError(verdict)`` to abort the run.
- ``EscalateToHITL(channel, card_factory)`` — open a HITL request and
  BLOCK until the human responds (sequential handler-chain semantics).
- ``Compose(*handlers)`` — run handlers in declared order; non-Raise
  exceptions are swallowed individually so a flaky handler never blocks
  the rest of the chain.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from ballast.drift._protocols import DriftContext, DriftHandler
from ballast.drift._verdict import DriftVerdictBase
from ballast.errors import BallastError

_log = logging.getLogger("ballast.drift")


class GoalDriftError(BallastError):  # noqa: N818
    """Raised by ``RaiseDriftError`` handler to abort the workflow/run.

    Propagates from ``DriftEngine.maybe_check`` and ``Compose.handle``.
    Caller's exception handler (DBOS workflow runtime / FastAPI / etc.)
    is responsible for whatever cleanup / retry / escalation applies.
    """

    code = "BALLAST_GOAL_DRIFT"
    status_code = 409

    def __init__(self, verdict: DriftVerdictBase) -> None:
        self.verdict = verdict
        super().__init__(
            f"GoalDriftError: {verdict.reason}",
            hint=(
                "The agent's goal-drift judge requested an interrupt. "
                "Adjust the drift strategy, expand the goal context, or "
                "remove ``RaiseDriftError`` from handlers if a hard stop "
                "isn't desired."
            ),
            context={"verdict": verdict.model_dump()},
        )


class LogOnly:
    """Write a WARNING log entry. Never blocks."""

    async def handle(self, verdict: DriftVerdictBase, ctx: DriftContext) -> None:
        _log.warning("goal drift detected: %s", verdict.reason)


class EmitDriftEvent:
    """Push a structured event to a caller-supplied async sink.

    Apps wire ``sink`` to whatever they want (thread-event publisher,
    OTel attribute, metrics counter). Verdict is ``model_dump()``-ed
    into the payload.
    """

    def __init__(
        self, *,
        sink: Callable[[str, dict[str, Any]], Awaitable[None]],
        event_name: str = "goal_drift",
    ) -> None:
        self._sink = sink
        self._event_name = event_name

    async def handle(self, verdict: DriftVerdictBase, ctx: DriftContext) -> None:
        await self._sink(self._event_name, verdict.model_dump())


class RaiseDriftError:
    """Raise ``GoalDriftError(verdict)`` — aborts the calling flow."""

    async def handle(self, verdict: DriftVerdictBase, ctx: DriftContext) -> None:
        raise GoalDriftError(verdict)


class EscalateToHITL:
    """Open a HITL request via a ``HITLChannel`` and BLOCK until verdict.

    The caller supplies:
      - ``channel``: any ``HITLChannel``-compatible object with
        ``async def request(payload, *, timeout) -> verdict``.
      - ``card_factory``: ``Callable[[DriftVerdictBase], BaseModel]``
        — builds the payload (apps may use a domain-specific
        ``ApprovalCard`` subclass).
      - ``timeout``: optional duration before the channel returns / raises.

    Blocking semantics: handler does not return until human responds (or
    timeout fires). Other handlers later in the chain run AFTER this.
    """

    def __init__(
        self, *,
        channel: Any,
        card_factory: Callable[[DriftVerdictBase], Any],
        timeout: timedelta | None = None,
    ) -> None:
        self._channel = channel
        self._card_factory = card_factory
        self._timeout = timeout

    async def handle(self, verdict: DriftVerdictBase, ctx: DriftContext) -> None:
        payload = self._card_factory(verdict)
        await self._channel.request(payload, timeout=self._timeout)


class Compose:
    """Run handlers in declared order, isolating non-Raise exceptions."""

    def __init__(self, *handlers: DriftHandler) -> None:
        if not handlers:
            raise ValueError("Compose requires at least one handler")
        self._handlers = handlers

    async def handle(self, verdict: DriftVerdictBase, ctx: DriftContext) -> None:
        for h in self._handlers:
            try:
                await h.handle(verdict, ctx)
            except GoalDriftError:
                raise  # Intentional hard-stop — propagate.
            except Exception:
                _log.exception(
                    "drift handler %r failed (swallowed)",
                    type(h).__name__,
                )


__all__ = [
    "Compose",
    "EmitDriftEvent",
    "EscalateToHITL",
    "GoalDriftError",
    "LogOnly",
    "RaiseDriftError",
]

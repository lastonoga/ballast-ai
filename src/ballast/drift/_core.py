"""``DriftEngine`` — pure-function pipeline for drift detection.

Single entry point ``maybe_check(signal, ctx)`` orchestrates:
  1. strategy.should_check(signal) → maybe abort (cheap path)
  2. goal_source.goal(ctx) + window.slice(ctx)
  3. prompt.build(...) → judge.run(...) (typed verdict)
  4. for each handler: handle(verdict, ctx) (failure-isolated)

Failure modes:
  - judge exception → swallowed + logged → return None
  - non-Raise handler exception → swallowed per-handler + logged → chain continues
  - GoalDriftError from any handler → propagates (intentional hard-stop)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ballast.drift._handlers import GoalDriftError
from ballast.drift._protocols import (
    DriftCheckSignal,
    DriftCheckStrategy,
    DriftContext,
    DriftHandler,
    GoalSource,
    PromptBuilder,
    TraceWindow,
)
from ballast.drift._verdict import DefaultDriftVerdict, DriftVerdictBase

_log = logging.getLogger("ballast.drift")


@dataclass
class DriftEngine:
    """Compose strategy + window + goal + prompt + judge + handlers.

    Capability and workflow surfaces both call ``maybe_check``; they
    differ only in how they assemble ``DriftCheckSignal`` + ``DriftContext``.
    """

    strategy:      DriftCheckStrategy
    window:        TraceWindow
    goal_source:   GoalSource
    prompt:        PromptBuilder
    judge:         Any  # pydantic-ai Agent[None, DriftVerdictBase-subclass]
    handlers:      list[DriftHandler] = field(default_factory=list)
    verdict_model: type[DriftVerdictBase] = DefaultDriftVerdict

    async def maybe_check(
        self, signal: DriftCheckSignal, ctx: DriftContext,
    ) -> DriftVerdictBase | None:
        """Run one drift check. Returns verdict if check fired, else None."""
        if not self.strategy.should_check(signal):
            return None

        trace = await self.window.slice(ctx)
        if not trace:
            return None

        goal = await self.goal_source.goal(ctx)
        prompt = self.prompt.build(goal, trace)

        try:
            judge_result = await self.judge.run(
                prompt, output_type=self.verdict_model,
            )
            verdict: DriftVerdictBase = judge_result.output
        except Exception:
            _log.exception("drift judge failed (swallowed)")
            return None

        if verdict.should_interrupt:
            for handler in self.handlers:
                try:
                    await handler.handle(verdict, ctx)
                except GoalDriftError:
                    raise
                except Exception:
                    _log.exception(
                        "drift handler %r failed (swallowed)",
                        type(handler).__name__,
                    )
        return verdict


__all__ = ["DriftEngine"]

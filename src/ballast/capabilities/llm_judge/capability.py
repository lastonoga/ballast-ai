"""``JudgeAfterRun`` — plug :class:`LLMJudge` into pydantic-ai's
``after_run`` lifecycle hook.

:class:`LLMJudge` itself is just a grading primitive — it doesn't know
about agents, threads, or capabilities. ``JudgeAfterRun`` is the
adapter that:

  1. fires the judge after every agent run,
  2. optionally persists the verdict as a UI card,
  3. optionally invokes a custom callback (HITL, metrics, alerts).

Three behaviours combine: observe / persist / dispatch. Each one is
opt-in by passing the corresponding constructor arg.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic_ai import RunContext

from ballast.capabilities.base import BallastCapability
from ballast.capabilities.llm_judge._errors import JudgeUnavailable
from ballast.capabilities.llm_judge.persistence import (
    persist_verdict_as_thread_event,
)

if TYPE_CHECKING:
    from pydantic_ai.run import AgentRunResult

    from ballast.capabilities.llm_judge._models import JudgeVerdict
    from ballast.capabilities.llm_judge.judge import LLMJudge


_log = logging.getLogger("ballast.judge_after_run")


class JudgeAfterRun(BallastCapability):
    """Auto-grade the final agent output via :class:`LLMJudge` after
    each run.

    Wire it on the agent like any other capability::

        class NotesAgent(DurableAgent):
            def build_agent(self):
                return Agent(
                    model=...,
                    capabilities=[
                        BudgetGuard(max_iterations=20),
                        JudgeAfterRun(
                            LLMJudge(
                                "Answer is grounded in retrieved notes",
                                threshold=0.7,
                            ),
                            thread_id_from=lambda ctx: ctx.deps.thread_id,
                        ),
                    ],
                    ...
                )

    Per-run isolation: stateless across runs (the judge itself is
    stateless too); no ``for_run`` override needed.
    """

    name = "judge_after_run"

    def __init__(
        self,
        judge: "LLMJudge",
        *,
        subject: str = "assistant-turn",
        thread_id_from: Callable[[RunContext[Any]], UUID | None] | None = None,
        on_verdict: Callable[
            ["JudgeVerdict", RunContext[Any]], Awaitable[None],
        ] | None = None,
        fail_open: bool = True,
    ) -> None:
        self.judge = judge
        self.subject = subject
        self.thread_id_from = thread_id_from
        self.on_verdict = on_verdict
        self.fail_open = fail_open

    async def after_run(
        self,
        ctx: RunContext[Any],
        *,
        result: "AgentRunResult[Any]",
    ) -> "AgentRunResult[Any]":
        """Grade ``result.output``, persist + dispatch as configured.

        Returns the result unchanged — judge observes, doesn't mutate.

        Failure modes:
          - ``JudgeFailed`` (from a ``sync=True`` judge crossing
            threshold) ALWAYS propagates — the app explicitly asked
            for a hard gate.
          - ``JudgeUnavailable`` (infra: model 5xx, timeout, network)
            is swallowed when ``fail_open=True`` (default). Log and
            continue, so a flaky judge model never holds up the
            user-facing reply. Set ``fail_open=False`` to propagate.
        """
        try:
            verdict = await self.judge.grade(result.output)
        except JudgeUnavailable as exc:
            if not self.fail_open:
                raise
            _log.warning(
                "judge unavailable; skipping verdict for "
                "subject=%r: %s", self.subject, exc,
            )
            return result

        if self.thread_id_from is not None:
            thread_id = self.thread_id_from(ctx)
            if thread_id is not None:
                await persist_verdict_as_thread_event(
                    thread_id, verdict, subject=self.subject,
                )

        if self.on_verdict is not None:
            await self.on_verdict(verdict, ctx)

        return result


__all__ = ["JudgeAfterRun"]

from __future__ import annotations

import inspect
import itertools
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Generic, TypeVar
from uuid import UUID

from dbos import DBOS, DBOSConfiguredInstance
from pydantic import BaseModel

from pydantic_ai_stateflow.capabilities.helpers import (
    Critique,
    SemanticLoopDetected,
    TypedLoopGuard,
)
from pydantic_ai_stateflow.patterns.errors import ReflectionExhausted
from pydantic_ai_stateflow.patterns.loop_recovery import AbortOnLoop, LoopRecoveryPolicy

InT = TypeVar("InT")
OutT = TypeVar("OutT")

Writer = Callable[[InT], Awaitable[OutT]] | Callable[[InT], OutT]
Critic = Callable[[Any], Awaitable[Critique]] | Callable[[Any], Critique]
FeedbackRenderer = Callable[[InT, list[Critique]], InT]


def _default_feedback_renderer(task: Any, feedback: list[Critique]) -> Any:
    return task


async def _ensure_async(fn: Callable[..., Any], *args: Any) -> Any:
    result = fn(*args)
    if inspect.isawaitable(result):
        return await result
    return result


_instance_counter = itertools.count()


@DBOS.dbos_class()
class Reflection(DBOSConfiguredInstance, Generic[InT, OutT]):
    """Writer -> Critic -> optional loop-guard -> repeat (bounded).

    Loop-guard ordering: AFTER critique. If draft fails critique AND
    guard detects repetition, ``loop_recovery.handle(...)`` decides:
    abort (raise), accept-last (warn + return), or escalate (HITL).

    Satisfies ``Pattern[InT, OutT]`` structurally (not via inheritance
    from ``Pattern``; ``DBOSConfiguredInstance`` is a DBOS-runtime base
    needed so member workflows / steps can be addressed by instance name
    during recovery).

    ``.run`` is a @DBOS.workflow; ``_write`` / ``_critique`` are @DBOS.step
    so retries / replays are deterministic across crashes.

    The ``for`` loop is BOUNDED by max_iterations (STATEFLOW013 compliant).
    """

    name: ClassVar[str] = "reflection"

    def __init__(
        self,
        writer: Writer[InT, OutT],
        critic: Critic,
        *,
        max_iterations: int = 5,
        loop_guard: TypedLoopGuard[OutT] | None = None,
        loop_recovery: LoopRecoveryPolicy[OutT] | None = None,
        feedback_renderer: FeedbackRenderer[InT] = _default_feedback_renderer,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        super().__init__(config_name=f"reflection-{next(_instance_counter)}")
        self.writer = writer
        self.critic = critic
        self.max_iterations = max_iterations
        self.loop_guard = loop_guard
        self.loop_recovery: LoopRecoveryPolicy[OutT] = loop_recovery or AbortOnLoop()
        self.feedback_renderer = feedback_renderer

    @DBOS.workflow()
    async def run(self, task: InT, *, tenant_id: UUID) -> OutT:
        feedback: list[Critique] = []
        for _ in range(self.max_iterations):
            rendered = self.feedback_renderer(task, feedback)
            draft = await self._write(rendered)
            critique = await self._critique(draft)
            if critique.passed:
                return draft
            feedback.append(critique)
            if self.loop_guard is not None:
                try:
                    await self.loop_guard.check(draft)
                except SemanticLoopDetected:
                    return await self.loop_recovery.handle(
                        ctx=None, draft=draft, feedback=feedback
                    )
        raise ReflectionExhausted(
            iterations=self.max_iterations, last_feedback=feedback
        )

    @DBOS.step()
    async def _write(self, rendered: InT) -> OutT:
        result: OutT = await _ensure_async(self.writer, rendered)
        return result

    @DBOS.step()
    async def _critique(self, draft: OutT) -> Critique:
        payload = draft.model_dump() if isinstance(draft, BaseModel) else draft
        verdict = await _ensure_async(self.critic, payload)
        if not isinstance(verdict, Critique):
            raise TypeError(
                f"critic must return Critique, got {type(verdict).__name__}"
            )
        return verdict

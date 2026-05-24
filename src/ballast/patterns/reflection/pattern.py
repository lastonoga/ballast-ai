"""``Reflection`` — writer → critic → refiner durable loop.

The simplest "self-correcting agent" shape: a writer produces a draft,
a critic assesses it, and if the critique fails the writer gets the
feedback and tries again — up to ``max_iter`` times.

Wrapped as a ``@Durable.workflow`` so every iteration is recoverable
across crashes: the writer call + critic call inside one step boundary
are memoised by DBOS, so on replay we skip already-completed work
and only re-run the unfinished tail.

Progress narration: each iteration boundary emits a
:class:`ReflectionEvent` on :data:`reflection_progress`. Apps open
``progress_to_thread(thread_id)`` around ``Reflection.run(task)`` to
get live "writer → critic → refine" UI cards out of the box; no
narration code in the pattern itself.
"""
from __future__ import annotations

import itertools
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from dbos import DBOSConfiguredInstance

from ballast.capabilities.helpers import Critique
from ballast.durable import Durable
from ballast.patterns.reflection._critic import (
    CriticCallable,
    to_critic_callable,
)
from ballast.patterns.reflection._errors import ReflectionExhausted
from ballast.patterns.reflection._events import (
    ReflectionEvent,
    reflection_progress,
)

# Forward-ref type only used for the constructor signature.
_LLMJudgeT = Any

InT = TypeVar("InT")
OutT = TypeVar("OutT")

Writer = Callable[[InT, list[Critique]], Awaitable[OutT]]
"""Writer signature: takes the task + accumulated critique history,
returns a draft. Apps wrap a ``BallastAgent`` themselves — keeping
Writer a plain callable means the pattern stays agnostic to whether
the writer is an LLM, a deterministic function, a mock, or anything
else with the right shape."""


_instance_counter = itertools.count()


@Durable.dbos_class()
class Reflection(DBOSConfiguredInstance, Generic[InT, OutT]):
    """Self-correcting writer/critic loop.

    Args:
        writer: ``async (task, critiques) -> draft``. Receives the
            accumulated critique history each iteration so the writer
            can incorporate feedback.
        critic: Either an :class:`LLMJudge` (auto-adapted to a
            ``Critique``-returning callable) or a plain
            ``async (draft) -> Critique`` callable.
        max_iter: Hard cap on attempts. The loop terminates with
            :class:`ReflectionExhausted` when reached without a
            passing critique.
        config_name: DBOS configured-instance name. Auto-generated if
            omitted; override for stable replay across deploys.
    """

    def __init__(
        self,
        *,
        writer: Writer[InT, OutT],
        critic: "_LLMJudgeT | CriticCallable",
        max_iter: int = 3,
        config_name: str | None = None,
    ) -> None:
        if max_iter < 1:
            raise ValueError(
                f"Reflection: ``max_iter`` must be >= 1, got {max_iter!r}",
            )
        resolved_name = config_name or (
            f"reflection-{next(_instance_counter)}"
        )
        super().__init__(config_name=resolved_name)
        self._writer = writer
        self._critic = to_critic_callable(critic)
        self.max_iter = max_iter

    @Durable.workflow()
    async def run(self, task: InT) -> OutT:
        """Run the writer/critic loop until the critic passes or
        ``max_iter`` is hit.

        Returns the draft that passed the critic. Raises
        :class:`ReflectionExhausted` (carrying the last critique) on
        no convergence.
        """
        critiques: list[Critique] = []
        last_critique: Critique | None = None
        last_draft: OutT | None = None

        for iter_num in range(1, self.max_iter + 1):
            draft = await self._write_step(task, critiques)
            last_draft = draft
            await self._publish(ReflectionEvent(
                type="draft",
                iter=iter_num,
                payload={"draft": _stringify(draft)},
            ))

            critique = await self._critique_step(draft)
            last_critique = critique
            critiques.append(critique)
            await self._publish(ReflectionEvent(
                type="critique",
                iter=iter_num,
                payload=critique.model_dump(mode="json"),
            ))

            if critique.passed:
                await self._publish(ReflectionEvent(
                    type="passed", iter=iter_num, payload={},
                ))
                return draft

            if iter_num < self.max_iter:
                await self._publish(ReflectionEvent(
                    type="refine", iter=iter_num, payload={},
                ))

        # Loop fell through: ``max_iter`` reached without passing.
        assert last_critique is not None
        assert last_draft is not None
        await self._publish(ReflectionEvent(
            type="exhausted",
            iter=self.max_iter,
            payload={"last_critique": last_critique.model_dump(mode="json")},
        ))
        raise ReflectionExhausted(
            iterations=self.max_iter,
            last_critique=last_critique,
        )

    # ── per-iteration steps (memoised by DBOS) ──────────────────────

    @Durable.step()
    async def _write_step(
        self, task: InT, critiques: list[Critique],
    ) -> OutT:
        """One writer call, isolated as a DBOS step so a crash before
        the critic returns doesn't re-fire the writer on replay."""
        return await self._writer(task, critiques)

    @Durable.step()
    async def _critique_step(self, draft: OutT) -> Critique:
        """One critic call, isolated as a DBOS step for the same
        crash-recovery reason."""
        return await self._critic(draft)

    # ── progress emission ───────────────────────────────────────────

    async def _publish(self, event: ReflectionEvent) -> None:
        """Fire the typed signal so observers see the event and the
        default chat router (if connected) emits a UI card."""
        await reflection_progress.send(sender=self, event=event)


def _stringify(value: Any) -> str:
    """Best-effort string representation of a draft for the UI card."""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump_json"):
        try:
            return value.model_dump_json()
        except Exception:
            pass
    return repr(value)


__all__ = ["Reflection", "Writer"]

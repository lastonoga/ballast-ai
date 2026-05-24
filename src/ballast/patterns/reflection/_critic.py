"""Adapter that lets :class:`LLMJudge` plug in as a Reflection critic.

Reflection's loop talks in :class:`Critique` (passed / issues /
suggestions / confidence). LLMJudge talks in :class:`JudgeVerdict`
(reason / pass_ / score / model_used / latency_ms). Translation is
straightforward but lives in its own file so neither side has to
know about the other:

  - the judge stays a general-purpose grader;
  - the reflection pattern stays critic-shape-agnostic;
  - apps can supply a plain ``callable(output) -> Critique`` without
    pulling LLMJudge / pydantic-evals into their dep graph.

The adapter is the only seam between the two abstractions.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from ballast.capabilities.helpers import Critique

if TYPE_CHECKING:
    from ballast.capabilities.llm_judge import LLMJudge


CriticCallable = Callable[[Any], Awaitable[Critique]]


def to_critic_callable(
    critic: "LLMJudge | CriticCallable",
) -> CriticCallable:
    """Normalise the supported critic shapes into one callable
    ``async (output) -> Critique``.

    - ``LLMJudge`` → grade in ``sync=False`` mode (we want the verdict
      regardless), translate ``JudgeVerdict`` → ``Critique`` (passed
      from ``pass_``, issues from ``reason``, confidence from ``score``).
    - already-callable → returned as-is.
    """
    from ballast.capabilities.llm_judge import LLMJudge  # noqa: PLC0415

    if isinstance(critic, LLMJudge):
        judge = critic

        async def _judge_critic(output: Any) -> Critique:
            verdict = await judge.grade(output, sync=False)
            return Critique(
                passed=verdict.pass_,
                issues=[verdict.reason] if not verdict.pass_ else [],
                suggestions=[],
                confidence=verdict.score,
            )

        return _judge_critic

    return critic


__all__ = ["CriticCallable", "to_critic_callable"]

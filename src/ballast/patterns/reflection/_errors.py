"""Reflection-specific exceptions."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ballast.patterns.errors import PatternError

if TYPE_CHECKING:
    from ballast.capabilities.helpers import Critique


class ReflectionExhausted(PatternError):  # noqa: N818
    """``Reflection.run`` hit ``max_iter`` without the critic passing.

    Attaches the last critique so handlers can surface the unresolved
    issues to the caller (HITL escalation, error UI, retry-with-
    different-rubric, …) instead of just "exhausted".
    """

    code = "BALLAST_PATTERN_REFLECTION_EXHAUSTED"
    status_code = 500

    def __init__(self, *, iterations: int, last_critique: "Critique") -> None:
        self.iterations = iterations
        self.last_critique = last_critique
        super().__init__(
            f"Reflection exhausted after {iterations} iteration(s); "
            f"last critique: passed={last_critique.passed}, "
            f"issues={last_critique.issues!r}",
            hint=(
                "Raise ``max_iter``, soften the critic's passing "
                "criteria, improve the writer's prompt, or wrap "
                "in a try/except and route to HITL with "
                "``e.last_critique`` for context."
            ),
            context={
                "iterations": iterations,
                "last_critique": last_critique.model_dump(mode="json"),
            },
        )


__all__ = ["ReflectionExhausted"]

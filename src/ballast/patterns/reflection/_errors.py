"""Reflection-specific exceptions."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ballast.patterns.errors import PatternError

if TYPE_CHECKING:
    from ballast.capabilities.helpers import Critique


_OutT = TypeVar("_OutT")


class ReflectionExhausted(PatternError, Generic[_OutT]):  # noqa: N818
    """``Reflection.run`` hit ``max_iter`` without the critic passing.

    Carries:

      - ``iterations`` — how many writer/critic rounds ran
      - ``last_critique`` — what the critic said last (issues +
        suggestions for the next-best step)
      - ``last_draft`` — the most recent draft the writer produced.
        Useful when the caller wants to "save anyway" / "show the
        best-effort attempt" instead of dropping work on exhaustion.

    Handlers route to HITL with the critique for context OR persist
    ``last_draft`` as a best-effort fallback.
    """

    code = "BALLAST_PATTERN_REFLECTION_EXHAUSTED"
    status_code = 500

    def __init__(
        self,
        *,
        iterations: int,
        last_critique: "Critique",
        last_draft: _OutT,
    ) -> None:
        self.iterations = iterations
        self.last_critique = last_critique
        self.last_draft = last_draft
        super().__init__(
            f"Reflection exhausted after {iterations} iteration(s); "
            f"last critique: passed={last_critique.passed}, "
            f"issues={last_critique.issues!r}",
            hint=(
                "Raise ``max_iter``, soften the critic's passing "
                "criteria, improve the writer's prompt, or wrap "
                "in a try/except and route to HITL with "
                "``e.last_critique`` for context (or persist "
                "``e.last_draft`` as a best-effort save)."
            ),
            context={
                "iterations": iterations,
                "last_critique": last_critique.model_dump(mode="json"),
                "last_draft": _safe_dump(last_draft),
            },
        )


def _safe_dump(value: Any) -> Any:
    """JSON-safe representation of a draft for the error context."""
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


__all__ = ["ReflectionExhausted"]

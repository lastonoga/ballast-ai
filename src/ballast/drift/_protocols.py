"""``DriftCheckSignal`` + ``DriftContext`` + 5 Protocols for drift detection.

Vehicles (this module's first half):
  ``DriftCheckSignal`` — cheap ping passed to ``DriftCheckStrategy.should_check``
  on every agent step. No I/O; constructed even when judge does not run.

  ``DriftContext`` — full state passed to window / goal source / handlers
  only when ``should_check`` returns True. May carry references to a
  ``RunContext`` (agent surface) or workflow input (workflow surface).

Protocols (added in Task 3): ``DriftCheckStrategy``, ``TraceWindow``,
``GoalSource``, ``PromptBuilder``, ``DriftHandler``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ModelMessage


@dataclass
class DriftCheckSignal:
    """Lightweight ping for ``DriftCheckStrategy.should_check``.

    Passed on every step (cheap to construct, no I/O).
    """

    step_index: int
    """Number of ``after_model_request`` invocations seen so far (1-based)."""

    tool_calls: int
    """Cumulative tool-call count across all model responses in this run."""

    tokens_used: int
    """Cumulative input+output tokens across all model responses."""

    seconds_elapsed: float
    """Monotonic time since the first hook fire."""


@dataclass
class DriftContext:
    """Full context for window / goal / handler.

    Built ONLY when ``DriftCheckStrategy.should_check`` returns True.
    Read-only by convention; framework does not mutate after construction.
    """

    messages: list["ModelMessage"]
    """Message history at the moment of the check (may be empty in workflow surface)."""

    run_ctx: "RunContext[Any] | None"
    """Available only in agent surface. ``None`` in workflow surface."""

    workflow_input: Any = None
    """Available only in workflow surface. ``None`` in agent surface."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Application-populated scratch (e.g. ``{"budget": {...}}`` for OnBudgetThreshold)."""


__all__ = ["DriftCheckSignal", "DriftContext"]

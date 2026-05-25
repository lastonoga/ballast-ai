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
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ModelMessage

from ballast.drift._verdict import DriftVerdictBase


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


@runtime_checkable
class DriftCheckStrategy(Protocol):
    """When to fire the judge.

    Implementations may be stateful (e.g., ``EveryNToolCalls`` tracks the
    last fire count). ``should_check`` is called on every agent step.
    """

    def should_check(self, signal: DriftCheckSignal) -> bool: ...


@runtime_checkable
class TraceWindow(Protocol):
    """What slice of message history to show the judge."""

    async def slice(self, ctx: DriftContext) -> list["ModelMessage"]: ...


@runtime_checkable
class GoalSource(Protocol):
    """Where the original objective comes from."""

    async def goal(self, ctx: DriftContext) -> str: ...


@runtime_checkable
class PromptBuilder(Protocol):
    """How to ask the judge.

    Returns the user prompt for the judge agent. The judge's system prompt
    is owned by the judge agent itself (see ``make_default_judge``).
    """

    def build(self, goal: str, trace: list["ModelMessage"]) -> str: ...


@runtime_checkable
class DriftHandler(Protocol):
    """What to do on drift.

    Multiple handlers run in declared order. Exceptions from non-Raise
    handlers are swallowed individually (see ``DriftEngine.maybe_check``).
    """

    async def handle(self, verdict: DriftVerdictBase, ctx: DriftContext) -> None: ...


__all__ = [
    "DriftCheckSignal", "DriftContext",
    "DriftCheckStrategy", "TraceWindow", "GoalSource",
    "PromptBuilder", "DriftHandler",
]

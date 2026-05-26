"""``PlanExecutionError`` — raised when a step fails and ``RePlanPolicy``
returns ``None`` (fail-loud)."""
from __future__ import annotations

from typing import Any

from ballast.errors import BallastError
from ballast.patterns.plan_execute._plan import PlannedStep


class PlanExecutionError(BallastError):  # noqa: N818
    """A step's execution failed and the configured ``RePlanPolicy``
    declined to provide a new plan.

    Carries ``failed_step`` and ``partial_outputs`` for debugging /
    higher-level recovery in calling workflows.
    """

    code = "BALLAST_PLAN_EXECUTION"
    status_code = 422

    def __init__(
        self,
        message: str,
        *,
        failed_step: PlannedStep,
        partial_outputs: dict[str, Any],
    ) -> None:
        self.failed_step = failed_step
        self.partial_outputs = partial_outputs
        super().__init__(
            message,
            hint=(
                "A planned step failed and no replan policy supplied a recovery plan. "
                "Inspect failed_step + partial_outputs to decide whether to retry, "
                "expand the planner's instructions, or wire a custom RePlanPolicy."
            ),
            context={
                "failed_step_id": failed_step.id,
                "failed_step_kind": failed_step.kind,
                "completed_step_ids": sorted(partial_outputs),
            },
        )


__all__ = ["PlanExecutionError"]

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic_ai_stateflow.errors import PatternError as _BasePatternError


class PatternError(_BasePatternError):
    """Root for all Pattern-raised errors. Catch this to handle any pattern failure.

    Re-exported as ``pydantic_ai_stateflow.errors.PatternError``; this
    module keeps the historical import path stable.
    """


class ReflectionExhausted(PatternError):  # noqa: N818
    """Reflection.run exhausted max_iterations without the critic returning passed=True."""

    code = "STATEFLOW_PATTERN_REFLECTION_EXHAUSTED"
    status_code = 500

    def __init__(self, *, iterations: int, last_feedback: list[Any]) -> None:
        self.iterations = iterations
        self.last_feedback = last_feedback
        super().__init__(
            f"ReflectionExhausted: {iterations} iterations without convergence; "
            f"last_feedback={last_feedback!r}",
            hint=(
                "Raise ``max_iterations``, soften the critic's passing "
                "criteria, or improve the worker prompt."
            ),
            context={"iterations": iterations, "last_feedback": list(last_feedback)},
        )


class MutationRejected(PatternError):  # noqa: N818
    """A MutationPipeline stage returned RejectedAt and RaiseOnReject was in effect."""

    code = "STATEFLOW_PATTERN_MUTATION_REJECTED"
    status_code = 500

    def __init__(self, *, stage: str, reason: str, actor_id: str | None = None) -> None:
        self.stage = stage
        self.reason = reason
        self.actor_id = actor_id
        super().__init__(
            f"MutationRejected at stage={stage!r}: {reason}",
            hint=(
                "Use ``DropOnReject`` to skip rejected proposals instead "
                "of raising, or adjust the stage's reject criteria."
            ),
            context={
                "stage": stage,
                "reason": reason,
                "actor_id": actor_id,
            },
        )


class HITLTimedOut(PatternError):  # noqa: N818
    """The HITL gate timed out before any authorized actor responded."""

    code = "STATEFLOW_PATTERN_HITL_TIMED_OUT"
    status_code = 504

    def __init__(self, *, request_id: UUID) -> None:
        self.request_id = request_id
        super().__init__(
            f"HITLTimedOut: request_id={request_id}",
            hint=(
                "Raise the HITL gate ``timeout`` or ensure an actor is "
                "subscribed to the channel."
            ),
            context={"request_id": str(request_id)},
        )


class HITLDenied(PatternError):  # noqa: N818
    """Defense-in-depth authz failure inside HITLGate (responder lacks permission)."""

    code = "STATEFLOW_PATTERN_HITL_DENIED"
    status_code = 403

    def __init__(self, *, actor_id: str, votes: dict[str, Any]) -> None:
        self.actor_id = actor_id
        self.votes = votes
        super().__init__(
            f"HITLDenied: actor_id={actor_id!r} votes={votes!r}",
            hint=(
                "Grant the actor the required role, or change the gate's "
                "``Policy`` to admit this actor."
            ),
            context={"actor_id": actor_id, "votes": dict(votes)},
        )


class InsufficientDivergence(PatternError):  # noqa: N818
    """``DivergentConvergent`` finished the divergent phase with fewer
    distinct hypotheses than ``min_hypotheses`` requires.

    Holds enough context for the caller to decide between retry with a
    relaxed config, dropping the run, or escalating to a human."""

    code = "STATEFLOW_PATTERN_INSUFFICIENT_DIVERGENCE"
    status_code = 500

    def __init__(
        self, *,
        produced: int,
        required: int,
        branch_outcomes: dict[str, str] | None = None,
    ) -> None:
        self.produced = produced
        self.required = required
        self.branch_outcomes = dict(branch_outcomes or {})
        super().__init__(
            f"InsufficientDivergence: produced={produced} required={required} "
            f"branch_outcomes={self.branch_outcomes!r}",
            hint=(
                "Lower ``min_hypotheses``, raise ``best_of_n``, or add "
                "branches with higher temperature."
            ),
            context={
                "produced": produced,
                "required": required,
                "branch_outcomes": dict(self.branch_outcomes),
            },
        )

from __future__ import annotations

from typing import Any
from uuid import UUID


class PatternError(Exception):
    """Root for all Pattern-raised errors. Catch this to handle any pattern failure."""


class ReflectionExhausted(PatternError):  # noqa: N818
    """Reflection.run exhausted max_iterations without the critic returning passed=True."""

    def __init__(self, *, iterations: int, last_feedback: list[Any]) -> None:
        self.iterations = iterations
        self.last_feedback = last_feedback
        super().__init__(
            f"ReflectionExhausted: {iterations} iterations without convergence; "
            f"last_feedback={last_feedback!r}"
        )


class MutationRejected(PatternError):  # noqa: N818
    """A MutationPipeline stage returned RejectedAt and RaiseOnReject was in effect."""

    def __init__(self, *, stage: str, reason: str, actor_id: str | None = None) -> None:
        self.stage = stage
        self.reason = reason
        self.actor_id = actor_id
        super().__init__(f"MutationRejected at stage={stage!r}: {reason}")


class HITLTimedOut(PatternError):  # noqa: N818
    """The HITL gate timed out before any authorized actor responded."""

    def __init__(self, *, request_id: UUID) -> None:
        self.request_id = request_id
        super().__init__(f"HITLTimedOut: request_id={request_id}")


class HITLDenied(PatternError):  # noqa: N818
    """Defense-in-depth authz failure inside HITLGate (responder lacks permission)."""

    def __init__(self, *, actor_id: str, votes: dict[str, Any]) -> None:
        self.actor_id = actor_id
        self.votes = votes
        super().__init__(f"HITLDenied: actor_id={actor_id!r} votes={votes!r}")

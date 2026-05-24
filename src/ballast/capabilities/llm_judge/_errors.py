"""Judge-specific exceptions.

Two distinct failure modes — they warrant different responses:

  - :class:`JudgeFailed` — the judge DID grade the output and ruled
    against it. Verdict is real; route to HITL or retry the producer.
  - :class:`JudgeUnavailable` — the judge MODEL itself never reached
    a verdict (timeout / rate-limit / 5xx). Usually fail-open (skip,
    log, continue) rather than block the user-facing turn.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ballast.errors import BallastError

if TYPE_CHECKING:
    from ballast.capabilities.llm_judge._models import JudgeVerdict


class JudgeFailed(BallastError):
    """Raised when ``LLMJudge.grade(..., sync=True)`` returns a verdict
    below the configured threshold. The verdict is attached as
    ``context['verdict']`` so handlers can route into HITL with the
    rationale already in hand.
    """

    code = "BALLAST_JUDGE_FAILED"
    status_code = 500

    def __init__(self, *, verdict: "JudgeVerdict") -> None:
        self.verdict = verdict
        super().__init__(
            f"LLMJudge verdict failed: score={verdict.score:.2f} "
            f"(threshold not met). Reason: {verdict.reason}",
            hint=(
                "Either raise the rubric's threshold, soften the "
                "rubric, or wire a HITL escalation that consumes the "
                "verdict via ``context['verdict']``."
            ),
            context={"verdict": verdict.model_dump(mode="json")},
        )


class JudgeUnavailable(BallastError):
    """Raised when the judge model itself failed (network, rate-limit,
    5xx, timeout) and the ``max_retries`` budget was exhausted.

    Callers usually catch this separately from :class:`JudgeFailed`
    and decide to fail-open: log the underlying error, skip the
    gate, continue the turn.
    """

    code = "BALLAST_JUDGE_UNAVAILABLE"
    status_code = 503

    def __init__(
        self,
        *,
        attempts: int,
        last_error: BaseException,
        model_used: str,
    ) -> None:
        self.attempts = attempts
        self.last_error = last_error
        self.model_used = model_used
        super().__init__(
            f"LLMJudge model {model_used!r} unavailable after "
            f"{attempts} attempt(s): {type(last_error).__name__}: "
            f"{last_error}",
            hint=(
                "Transient model error. Raise ``max_retries`` for "
                "more resilience, switch ``model=`` to a faster / "
                "more reliable judge, or wrap the grade call in a "
                "try/except ``JudgeUnavailable`` to fail-open."
            ),
            context={
                "attempts": attempts,
                "model": model_used,
                "last_error_type": type(last_error).__name__,
            },
        )


__all__ = ["JudgeFailed", "JudgeUnavailable"]

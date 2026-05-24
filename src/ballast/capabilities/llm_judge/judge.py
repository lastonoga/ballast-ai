"""``LLMJudge`` — runtime quality gate.

Thin wrapper around ``pydantic_evals.evaluators.llm_as_a_judge`` so the
same primitive that batches CI evaluations is also usable in the hot
path (after an agent turn, after a retrieval, after a tool-call
result). Production callers typically:

  - fire-and-forget (``sync=False``, default) — judge runs in the
    background, verdict is persisted/logged, the user-facing response
    is not blocked;
  - hard-gate (``sync=True``) — raise :class:`JudgeFailed` on verdicts
    below ``threshold`` so the caller can route into HITL or retry.

:class:`LLMJudge` is intentionally a plain helper class, NOT a
``BallastCapability`` subclass. The judge grades arbitrary objects
(assistant turn, tool argument, retrieved chunk, intermediate plan) —
forcing it through pydantic-ai's capability lifecycle hooks would
conflate "the judging primitive" with "auto-grade on every turn".
Apps that want the latter wrap an :class:`LLMJudge` inside
:class:`JudgeAfterRun` (or their own capability).

Two grading modes:

  - ``direct``   — single output graded against a rubric. Forwards to
                   ``pydantic_evals.judge_*``.
  - ``pairwise`` — compare two outputs against the rubric, return the
                   winner. Pydantic-evals does not ship pairwise OOB;
                   the agent + prompt live in :mod:`._pairwise`.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal

from ballast.capabilities.llm_judge._errors import JudgeFailed
from ballast.capabilities.llm_judge._models import (
    JudgeVerdict,
    PairwiseVerdict,
)
from ballast.capabilities.llm_judge._pairwise import (
    build_pairwise_prompt,
    make_pairwise_agent,
)
from ballast.capabilities.llm_judge._retry import retry_with_backoff

if TYPE_CHECKING:
    from pydantic_ai.settings import ModelSettings


_default_model_settings: "ModelSettings | None" = None


def set_default_judge_model(
    model: "str | object",
    *,
    model_settings: "ModelSettings | None" = None,
) -> None:
    """Process-wide default for every ``LLMJudge(model=None, ...)``.

    Two settings combined behind one call:

    - ``model`` — forwarded to pydantic-evals so batch CI evaluators
      use the same judge as runtime grading.
    - ``model_settings`` — framework-side default for any
      ``LLMJudge(model_settings=None)``. Pydantic-evals doesn't have
      a global slot for this, so we track it ourselves and inject at
      grade-time.

    Use it once at app boot::

        from ballast import set_default_judge_model
        from pydantic_ai.models.openrouter import OpenRouterModelSettings

        set_default_judge_model(
            "openrouter:qwen/qwen3.6-plus",
            model_settings=OpenRouterModelSettings(
                temperature=0.0,                       # deterministic
                openrouter_reasoning={"effort": "none"},
            ),
        )
    """
    from pydantic_evals.evaluators.llm_as_a_judge import (  # noqa: PLC0415
        set_default_judge_model as _upstream,
    )

    _upstream(model)
    global _default_model_settings
    _default_model_settings = model_settings


def get_default_judge_model_settings() -> "ModelSettings | None":
    """Read the global default ``ModelSettings``; ``None`` if unset."""
    return _default_model_settings


def _resolve_default_model() -> str:
    """Read pydantic-evals' configured default model id as a string."""
    from pydantic_evals.evaluators.llm_as_a_judge import (  # noqa: PLC0415
        _default_model,
    )

    if isinstance(_default_model, str):
        return _default_model
    return getattr(_default_model, "name", repr(_default_model))


class LLMJudge:
    """LLM-as-a-judge quality gate.

    Args:
        rubric: Natural-language criterion the judge applies. Prefer
            binary / categorical phrasings ("does X reference at least
            one source") over abstract 1-10 scales — the latter
            regresses to the mean per published evaluator-research.
        model: pydantic-ai model id (``"openai:gpt-5.2"``,
            ``"anthropic:claude-3-7-sonnet"``, …). ``None`` uses the
            pydantic-evals default — change globally with
            :func:`set_default_judge_model`.
        mode: ``"direct"`` (default) grades one output;
            ``"pairwise"`` compares two via :meth:`grade_pairwise`.
        threshold: ``score < threshold`` → ``pass_`` ignored, raises
            :class:`JudgeFailed` when ``sync=True``.
        sync: Default for :meth:`grade`. ``True`` raises on threshold
            miss; ``False`` returns the verdict regardless and the
            caller decides what to do.
        model_settings: Forwarded to pydantic-ai's agent. Use this to
            cap ``temperature`` (a judge benefits from ~0.0 — you want
            deterministic verdicts, not "creative" ones) or set
            ``max_tokens``.
        max_retries: Transient-error budget for the judge model call.
            ``0`` (default) = no retry; the original error propagates
            as :class:`JudgeUnavailable`. Use ``2``–``3`` if your
            stack tolerates a small extra latency budget on transient
            blips.
        retry_backoff_base_s: First-attempt backoff (seconds). Doubles
            on each subsequent retry.
    """

    def __init__(
        self,
        rubric: str,
        *,
        model: str | None = None,
        mode: Literal["direct", "pairwise"] = "direct",
        threshold: float = 0.5,
        sync: bool = False,
        model_settings: "ModelSettings | None" = None,
        max_retries: int = 0,
        retry_backoff_base_s: float = 0.5,
    ) -> None:
        if not rubric or not rubric.strip():
            raise ValueError("LLMJudge: ``rubric`` must be non-empty")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"LLMJudge: ``threshold`` must be in [0, 1], "
                f"got {threshold!r}",
            )
        if max_retries < 0:
            raise ValueError(
                f"LLMJudge: ``max_retries`` must be >= 0, "
                f"got {max_retries!r}",
            )
        self.rubric = rubric
        self.model = model
        self.mode = mode
        self.threshold = threshold
        self.sync = sync
        self._explicit_model_settings = model_settings
        self.max_retries = max_retries
        self.retry_backoff_base_s = retry_backoff_base_s

    @property
    def model_settings(self) -> "ModelSettings | None":
        """Per-instance ``model_settings`` if set, else the global
        default (:func:`set_default_judge_model` ``model_settings=``).

        Property (not stored attribute) so global-default changes after
        ``LLMJudge(...)`` construction still affect subsequent grades —
        matches how the global model id propagates.
        """
        if self._explicit_model_settings is not None:
            return self._explicit_model_settings
        return _default_model_settings

    async def grade(
        self,
        output: Any,
        *,
        input_: Any | None = None,
        expected: Any | None = None,
        sync: bool | None = None,
    ) -> JudgeVerdict:
        """Grade ``output`` against the rubric.

        Routes to the most-specific pydantic-evals helper based on
        which optional args are supplied:

          - ``output``                                 → ``judge_output``
          - ``output`` + ``input_``                    → ``judge_input_output``
          - ``output`` + ``expected``                  → ``judge_output_expected``
          - ``output`` + ``input_`` + ``expected``     → ``judge_input_output_expected``

        ``sync`` overrides the instance default — pass ``True`` to
        raise :class:`JudgeFailed` on threshold-miss, ``False`` to
        always return the verdict.

        Raises :class:`JudgeUnavailable` if the underlying model call
        fails ``max_retries + 1`` times in a row.
        """
        from pydantic_evals.evaluators import llm_as_a_judge  # noqa: PLC0415

        sync_resolved = self.sync if sync is None else sync
        model_id = self.model or _resolve_default_model()

        async def _call_judge() -> Any:
            if input_ is not None and expected is not None:
                return await llm_as_a_judge.judge_input_output_expected(
                    input_, output, expected, self.rubric,
                    model=self.model, model_settings=self.model_settings,
                )
            if input_ is not None:
                return await llm_as_a_judge.judge_input_output(
                    input_, output, self.rubric,
                    model=self.model, model_settings=self.model_settings,
                )
            if expected is not None:
                return await llm_as_a_judge.judge_output_expected(
                    output, expected, self.rubric,
                    model=self.model, model_settings=self.model_settings,
                )
            return await llm_as_a_judge.judge_output(
                output, self.rubric,
                model=self.model, model_settings=self.model_settings,
            )

        started = time.perf_counter()
        grading = await retry_with_backoff(
            _call_judge,
            max_retries=self.max_retries,
            backoff_base_s=self.retry_backoff_base_s,
            model_id=model_id,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        verdict = JudgeVerdict(
            reason=grading.reason,
            pass_=grading.pass_,
            score=grading.score,
            model_used=model_id,
            latency_ms=latency_ms,
        )
        if sync_resolved and verdict.score < self.threshold:
            raise JudgeFailed(verdict=verdict)
        return verdict

    async def grade_pairwise(
        self,
        a: Any,
        b: Any,
        *,
        rubric_override: str | None = None,
    ) -> PairwiseVerdict:
        """Compare ``a`` vs ``b`` against the rubric (or override).

        Returns the winner + reason; ``threshold`` / ``sync`` do not
        apply here (there is no scalar score to threshold against).

        Same ``max_retries`` policy as :meth:`grade` — exhausted →
        :class:`JudgeUnavailable`.
        """
        rubric = rubric_override or self.rubric
        model_id = self.model or _resolve_default_model()
        prompt = build_pairwise_prompt(rubric, a, b)
        agent = make_pairwise_agent(
            model_id, model_settings=self.model_settings,
        )

        started = time.perf_counter()
        result = await retry_with_backoff(
            lambda: agent.run(prompt),
            max_retries=self.max_retries,
            backoff_base_s=self.retry_backoff_base_s,
            model_id=model_id,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        return PairwiseVerdict(
            winner=result.output.winner,
            reason=result.output.reason,
            model_used=model_id,
            latency_ms=latency_ms,
        )


__all__ = [
    "LLMJudge",
    "get_default_judge_model_settings",
    "set_default_judge_model",
]

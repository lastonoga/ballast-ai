"""LLM-as-a-Judge — runtime quality gate.

Thin wrapper around ``pydantic_evals.evaluators.llm_as_a_judge`` so the
same judge primitive that batches CI evaluations is also usable in the
hot path (after an agent turn, after a retrieval, after a tool-call
result). Production callers typically:

  - fire-and-forget (``sync=False``, default) — judge runs in the
    background, verdict is persisted/logged, the user-facing response
    is not blocked;
  - hard-gate (``sync=True``) — raise ``JudgeFailed`` on verdicts below
    ``threshold`` so the caller can route into HITL or retry.

``LLMJudge`` is intentionally a plain helper class, NOT a
``BallastCapability`` subclass. The judge grades arbitrary objects
(assistant turn, tool argument, retrieved chunk, intermediate plan) —
forcing it through pydantic-ai's capability lifecycle hooks would
conflate "the judging primitive" with "auto-grade on every turn".
Apps that want the latter wrap an ``LLMJudge`` instance inside a
capability of their own.

Two grading modes:

  - ``direct``   — single output graded against a rubric. Forwards to
                   ``pydantic_evals.judge_output`` /
                   ``judge_input_output_expected``.
  - ``pairwise`` — compare two outputs against the rubric, return the
                   winner. Pydantic-evals does not ship pairwise out
                   of the box; we own a tiny pydantic-ai Agent for it.

Verdict persistence is a separate opt-in helper
(``persist_verdict_as_thread_event``) — judges grade everything, but
only some callers want the verdict on the thread's event log.
"""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from ballast.capabilities.base import BallastCapability
from ballast.errors import BallastError

if TYPE_CHECKING:
    from pydantic_ai.run import AgentRunResult
    from pydantic_ai.settings import ModelSettings


_DEFAULT_PAIRWISE_MODEL = "openai:gpt-5.2"


class JudgeFailed(BallastError):
    """Raised when ``LLMJudge.grade(..., sync=True)`` returns a verdict
    below the configured threshold. The verdict is attached as
    ``context['verdict']`` so handlers can route into HITL with the
    rationale already in hand.

    Distinct from :class:`JudgeUnavailable` — this means the judge
    DID grade the output and ruled against it. Infrastructure-level
    failures (timeout, rate limit, model 5xx) raise ``JudgeUnavailable``
    instead.
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

    Separate from :class:`JudgeFailed` because the policy is different:
    failed-verdict is the judge doing its job and saying "this output
    is bad"; unavailable is the judge infrastructure not reaching the
    model at all. Callers usually want to log + skip (fail-open) on
    unavailability rather than blocking the user-facing turn.
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


class JudgeVerdict(BaseModel, populate_by_name=True):
    """Result of one grading call.

    ``pass_`` / ``score`` mirror pydantic-evals' ``GradingOutput``;
    ``model_used`` + ``latency_ms`` are framework additions so
    production monitoring can attribute cost / latency to the judge.

    The wire-serialised JSON uses ``pass`` (alias) so it round-trips
    cleanly with ``GradingOutput``.
    """

    reason: str
    pass_: bool = Field(validation_alias="pass", serialization_alias="pass")
    score: float
    model_used: str
    latency_ms: int


class PairwiseVerdict(BaseModel):
    """Result of one pairwise comparison.

    ``winner`` ∈ {``"a"``, ``"b"``, ``"tie"``}. ``reason`` is the
    judge's CoT-style justification — same role as ``JudgeVerdict.reason``.
    """

    winner: Literal["a", "b", "tie"]
    reason: str
    model_used: str
    latency_ms: int


def set_default_judge_model(model: str) -> None:
    """Process-wide default model for every ``LLMJudge(model=None, ...)``.

    Thin pass-through to
    ``pydantic_evals.evaluators.llm_as_a_judge.set_default_judge_model``
    — same setting that pydantic-evals' batch CI evaluators consult,
    so direct-mode runtime grading and CI use the same model unless
    overridden per-instance.

    Call once at app boot before any judge instance grades anything::

        from ballast import set_default_judge_model

        set_default_judge_model("openrouter:qwen/qwen-3.6-72b-instruct")
    """
    from pydantic_evals.evaluators.llm_as_a_judge import (  # noqa: PLC0415
        set_default_judge_model as _upstream,
    )

    _upstream(model)


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
            ``JudgeFailed`` when ``sync=True``.
        sync: Default for :meth:`grade`. ``True`` raises on threshold
            miss; ``False`` returns the verdict regardless and the
            caller decides what to do.
        model_settings: Forwarded to pydantic-ai's agent. Use this to
            cap ``temperature`` (a judge benefits from ~0.0 — you want
            deterministic verdicts, not "creative" ones) or set
            ``max_tokens``.
        max_retries: Transient-error budget for the judge model call.
            ``0`` (default) = no retry; the original error propagates
            as :class:`JudgeUnavailable`. ``n`` = up to ``n`` retries
            with exponential backoff (``0.5s``, ``1s``, ``2s``, …)
            before giving up. Use ``2``–``3`` if the rest of your stack
            tolerates a small extra latency budget on transient blips.
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
        self.model_settings = model_settings
        self.max_retries = max_retries
        self.retry_backoff_base_s = retry_backoff_base_s

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
        fails ``max_retries + 1`` times in a row (transient errors are
        retried with exponential backoff).
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
        grading = await self._retry(_call_judge, model_id=model_id)
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

    async def _retry(
        self,
        call: Callable[[], Awaitable[Any]],
        *,
        model_id: str,
    ) -> Any:
        """Execute ``call`` with up to ``self.max_retries`` retries on
        any exception. Backoff doubles each attempt starting from
        ``self.retry_backoff_base_s``.

        Translates the final exception into :class:`JudgeUnavailable`
        so callers can distinguish infrastructure failure from
        :class:`JudgeFailed` (real low-score verdict).
        """
        import asyncio  # noqa: PLC0415

        attempts = self.max_retries + 1
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                return await call()
            except Exception as exc:  # noqa: BLE001 — wrap all transients
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                backoff = self.retry_backoff_base_s * (2 ** attempt)
                await asyncio.sleep(backoff)
        assert last_error is not None  # for type-checker — loop ran at least once
        raise JudgeUnavailable(
            attempts=attempts, last_error=last_error, model_used=model_id,
        )

    async def grade_pairwise(
        self,
        a: Any,
        b: Any,
        *,
        rubric_override: str | None = None,
    ) -> PairwiseVerdict:
        """Compare ``a`` vs ``b`` against the rubric (or override).

        Pairwise comparison is empirically the most robust judge mode
        for subjective metrics (tone, helpfulness, style) — single-shot
        scoring is fragile because the model has no anchor for what
        "good" looks like.

        Returns the winner + reason; ``threshold`` / ``sync`` do not
        apply here (there is no scalar score to threshold against).
        """
        from pydantic_ai import Agent  # noqa: PLC0415

        rubric = rubric_override or self.rubric
        model_id = self.model or _resolve_default_model()

        prompt = (
            "Compare two outputs against the rubric and choose the "
            "stronger one. If they are equally good or equally bad, "
            "return ``tie``. Always explain your reasoning before the "
            "final verdict.\n\n"
            f"<Rubric>{rubric}</Rubric>\n"
            f"<OutputA>{_stringify(a)}</OutputA>\n"
            f"<OutputB>{_stringify(b)}</OutputB>"
        )
        agent: Agent[None, _PairwiseGrading] = Agent(
            model=model_id,
            output_type=_PairwiseGrading,
            system_prompt=(
                "You are a strict but fair judge comparing two outputs "
                "against a user-supplied rubric. Always reason step by "
                "step before returning a verdict."
            ),
            model_settings=self.model_settings,
        )

        started = time.perf_counter()
        result = await self._retry(lambda: agent.run(prompt), model_id=model_id)
        latency_ms = int((time.perf_counter() - started) * 1000)

        return PairwiseVerdict(
            winner=result.output.winner,
            reason=result.output.reason,
            model_used=model_id,
            latency_ms=latency_ms,
        )


class _PairwiseGrading(BaseModel):
    """Internal output schema for the pairwise judge agent."""

    reason: str
    winner: Literal["a", "b", "tie"]


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    try:
        import json  # noqa: PLC0415

        return json.dumps(value, default=str)
    except Exception:
        return repr(value)


def _resolve_default_model() -> str:
    """Read pydantic-evals' configured default model id as a string."""
    from pydantic_evals.evaluators.llm_as_a_judge import (  # noqa: PLC0415
        _default_model,
    )

    if isinstance(_default_model, str):
        return _default_model
    return getattr(_default_model, "name", repr(_default_model))


async def persist_verdict_as_thread_event(
    thread_id: UUID,
    verdict: "JudgeVerdict | PairwiseVerdict",
    *,
    subject: str,
) -> None:
    """Optionally write a judge verdict into the thread's event log so
    the SSE consumer can render it as a card and the audit log keeps
    a durable record.

    Callers decide WHEN to persist — every site that wants the
    verdict surfaced calls this explicitly. Not auto-wired so judges
    used for fire-and-forget telemetry stay free of I/O.

    ``subject`` is a free-form label (``"assistant-turn"``,
    ``"tool-call:create_note"``, ``"retrieved-chunk:42"``) that
    explains WHAT was graded — the verdict on its own is meaningless
    without the subject.
    """
    from ballast.runtime.engine import get_ballast  # noqa: PLC0415

    engine = get_ballast()
    # Reuse the broadcaster — verdict is logically the same shape as
    # any other ``data-*`` UI card the framework emits (renders via
    # ``makeAssistantDataUI({name: "judge-verdict"})`` on the FE).
    # The broadcaster already routes through ``MessageAddedPayload`` so
    # the payload shape stays leak-proof.
    await engine.broadcaster.emit_raw(
        thread_id,
        part={
            "type": "data-judge-verdict",
            "data": {
                "subject": subject,
                **verdict.model_dump(mode="json", by_alias=True),
            },
        },
        persistent=True,
    )


class JudgeAfterRun(BallastCapability):
    """Auto-grade the final agent output via ``LLMJudge`` after each run.

    Plugs an :class:`LLMJudge` into pydantic-ai's ``after_run`` hook so
    every agent turn produces a verdict without app code having to call
    the judge explicitly. Three behaviours combined as you opt in:

      1. **Just observe** — pass ``judge`` only. Verdict is computed
         per-run; if ``judge.sync=True`` and threshold is missed,
         ``JudgeFailed`` propagates out of the agent's ``run`` call.

      2. **Persist to thread** — pass ``thread_id_from(ctx)`` returning
         the destination thread. Each verdict lands as a
         ``data-judge-verdict`` card in the SSE feed via
         :func:`persist_verdict_as_thread_event`.

      3. **Custom side-effect** — pass ``on_verdict(verdict, ctx)`` for
         arbitrary work (Slack alert, OTel attribute, metric counter,
         HITL escalation, …). Runs AFTER persistence (if enabled).

    Wire it on the agent like any other capability::

        class NotesAgent(DurableAgent):
            def build_agent(self):
                return Agent(
                    model=...,
                    capabilities=[
                        BudgetGuard(max_iterations=20),
                        JudgeAfterRun(
                            LLMJudge(
                                "Answer is grounded in retrieved notes",
                                threshold=0.7,
                            ),
                            thread_id_from=lambda ctx: ctx.deps.thread_id,
                        ),
                    ],
                    ...
                )

    Per-run isolation: the capability is stateless across runs (the
    judge itself is stateless too); no ``for_run`` override needed.
    """

    name = "judge_after_run"

    def __init__(
        self,
        judge: "LLMJudge",
        *,
        subject: str = "assistant-turn",
        thread_id_from: Callable[[RunContext[Any]], UUID | None] | None = None,
        on_verdict: Callable[
            ["JudgeVerdict", RunContext[Any]], Awaitable[None],
        ] | None = None,
    ) -> None:
        self.judge = judge
        self.subject = subject
        self.thread_id_from = thread_id_from
        self.on_verdict = on_verdict

    async def after_run(
        self,
        ctx: RunContext[Any],
        *,
        result: "AgentRunResult[Any]",
    ) -> "AgentRunResult[Any]":
        """Grade ``result.output``, persist + dispatch as configured.

        Returns the result unchanged — judge does not mutate the
        agent's output, just observes / escalates. If the underlying
        ``LLMJudge`` is configured ``sync=True`` and the verdict misses
        threshold, ``JudgeFailed`` propagates here.
        """
        verdict = await self.judge.grade(result.output)

        if self.thread_id_from is not None:
            thread_id = self.thread_id_from(ctx)
            if thread_id is not None:
                await persist_verdict_as_thread_event(
                    thread_id, verdict, subject=self.subject,
                )

        if self.on_verdict is not None:
            await self.on_verdict(verdict, ctx)

        return result


__all__ = [
    "JudgeAfterRun",
    "JudgeFailed",
    "JudgeUnavailable",
    "JudgeVerdict",
    "LLMJudge",
    "PairwiseVerdict",
    "persist_verdict_as_thread_event",
    "set_default_judge_model",
]

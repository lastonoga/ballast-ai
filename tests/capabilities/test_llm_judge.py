"""``LLMJudge`` runtime-gate tests.

We stub the underlying pydantic-evals ``judge_*`` functions and the
pairwise pydantic-ai agent so the suite doesn't hit a real model.
What we're actually asserting is the framework's contract: routing
between the four pydantic-evals signatures, threshold behaviour,
verdict shape, and the JudgeFailed escalation path.
"""
from __future__ import annotations

from typing import Any

import pytest
from pydantic_evals.evaluators.llm_as_a_judge import GradingOutput

from ballast.capabilities.llm_judge import (
    JudgeAfterRun,
    JudgeFailed,
    JudgeUnavailable,
    JudgeVerdict,
    LLMJudge,
    PairwiseVerdict,
    set_default_judge_model,
)


@pytest.fixture
def stub_judge_funcs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace pydantic-evals ``judge_*`` with deterministic stubs and
    record which one was invoked with what args."""
    calls: dict[str, Any] = {}

    def _make_stub(name: str, *, grading: GradingOutput):
        async def _stub(*args: Any, **kwargs: Any) -> GradingOutput:
            calls.setdefault("name", name)
            calls.setdefault("args", args)
            calls.setdefault("kwargs", kwargs)
            return grading
        return _stub

    grading = GradingOutput(reason="looks ok", pass_=True, score=0.8)
    from pydantic_evals.evaluators import llm_as_a_judge

    for fn_name in (
        "judge_output",
        "judge_input_output",
        "judge_output_expected",
        "judge_input_output_expected",
    ):
        monkeypatch.setattr(
            llm_as_a_judge, fn_name, _make_stub(fn_name, grading=grading),
        )
    return calls


@pytest.mark.asyncio
async def test_grade_with_output_only_routes_to_judge_output(
    stub_judge_funcs: dict[str, Any],
) -> None:
    judge = LLMJudge("Output is non-empty", threshold=0.5)
    verdict = await judge.grade("hello")
    assert stub_judge_funcs["name"] == "judge_output"
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.pass_ is True
    assert verdict.score == 0.8
    assert verdict.reason == "looks ok"
    assert verdict.model_used  # populated, content not asserted


@pytest.mark.asyncio
async def test_grade_with_input_routes_to_judge_input_output(
    stub_judge_funcs: dict[str, Any],
) -> None:
    judge = LLMJudge("Output responds to input")
    await judge.grade("hi back", input_="hi")
    assert stub_judge_funcs["name"] == "judge_input_output"


@pytest.mark.asyncio
async def test_grade_with_expected_routes_to_judge_output_expected(
    stub_judge_funcs: dict[str, Any],
) -> None:
    judge = LLMJudge("Output matches expected")
    await judge.grade("hello", expected="hello")
    assert stub_judge_funcs["name"] == "judge_output_expected"


@pytest.mark.asyncio
async def test_grade_with_input_and_expected_routes_to_full_judge(
    stub_judge_funcs: dict[str, Any],
) -> None:
    judge = LLMJudge("Output matches expected given input")
    await judge.grade("hello", input_="hi", expected="hello")
    assert stub_judge_funcs["name"] == "judge_input_output_expected"


@pytest.mark.asyncio
async def test_sync_mode_raises_judgefailed_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sync=True`` + score < threshold → JudgeFailed with verdict
    attached (so HITL handlers can read the rationale)."""
    failing = GradingOutput(reason="off topic", pass_=False, score=0.2)

    async def _stub(*_args: Any, **_kwargs: Any) -> GradingOutput:
        return failing

    from pydantic_evals.evaluators import llm_as_a_judge
    monkeypatch.setattr(llm_as_a_judge, "judge_output", _stub)

    judge = LLMJudge("Strict rubric", threshold=0.5, sync=True)
    with pytest.raises(JudgeFailed) as exc_info:
        await judge.grade("garbage")
    assert exc_info.value.verdict.score == 0.2
    assert exc_info.value.verdict.reason == "off topic"
    # Context exposes the full verdict so escalation handlers can
    # surface it to the user without re-running the judge.
    assert exc_info.value.context["verdict"]["score"] == 0.2


@pytest.mark.asyncio
async def test_async_default_returns_failing_verdict_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sync=False`` (the default) returns the verdict regardless of
    threshold — the caller decides what to do."""
    failing = GradingOutput(reason="off topic", pass_=False, score=0.2)

    async def _stub(*_args: Any, **_kwargs: Any) -> GradingOutput:
        return failing

    from pydantic_evals.evaluators import llm_as_a_judge
    monkeypatch.setattr(llm_as_a_judge, "judge_output", _stub)

    judge = LLMJudge("Strict rubric", threshold=0.5)
    verdict = await judge.grade("garbage")
    assert verdict.pass_ is False
    assert verdict.score == 0.2


@pytest.mark.asyncio
async def test_per_call_sync_overrides_instance_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing = GradingOutput(reason="bad", pass_=False, score=0.1)

    async def _stub(*_args: Any, **_kwargs: Any) -> GradingOutput:
        return failing

    from pydantic_evals.evaluators import llm_as_a_judge
    monkeypatch.setattr(llm_as_a_judge, "judge_output", _stub)

    judge = LLMJudge("R", threshold=0.5, sync=False)
    with pytest.raises(JudgeFailed):
        await judge.grade("x", sync=True)


def test_constructor_rejects_empty_rubric() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        LLMJudge("")


def test_constructor_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        LLMJudge("R", threshold=1.5)


@pytest.mark.asyncio
async def test_pairwise_returns_winner_from_stub_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``grade_pairwise`` builds an internal pydantic-ai Agent via the
    ``make_pairwise_agent`` factory. Patch the factory so we return a
    fake agent and skip the real OpenAI provider init."""
    from ballast.capabilities.llm_judge import _pairwise as pairwise_mod

    class _FakeRunResult:
        def __init__(
            self, output: pairwise_mod._PairwiseGrading,
        ) -> None:
            self.output = output

    class _FakeAgent:
        async def run(self, *args: Any, **kwargs: Any) -> _FakeRunResult:
            del args, kwargs
            return _FakeRunResult(pairwise_mod._PairwiseGrading(
                reason="B is more detailed", winner="b",
            ))

    monkeypatch.setattr(
        "ballast.capabilities.llm_judge.judge.make_pairwise_agent",
        lambda *_a, **_kw: _FakeAgent(),
    )

    judge = LLMJudge("Detailed answer", mode="pairwise")
    verdict = await judge.grade_pairwise("short", "looong and detailed")
    assert isinstance(verdict, PairwiseVerdict)
    assert verdict.winner == "b"
    assert "detailed" in verdict.reason.lower()
    assert verdict.model_used


# ── JudgeAfterRun (capability wrapper) ────────────────────────────────


class _FakeAgentRunResult:
    def __init__(self, output: Any) -> None:
        self.output = output


class _FakeRunContext:
    def __init__(self, deps: Any = None) -> None:
        self.deps = deps


@pytest.mark.asyncio
async def test_judge_after_run_invokes_judge_and_returns_result_unchanged(
    stub_judge_funcs: dict[str, Any],
) -> None:
    """``after_run`` hook calls ``judge.grade(result.output)`` and
    returns the result unchanged — judge observes, doesn't mutate."""
    cap = JudgeAfterRun(LLMJudge("Output is non-empty"))
    result = _FakeAgentRunResult(output="hello")
    returned = await cap.after_run(_FakeRunContext(), result=result)
    assert returned is result
    # The judge stub returns score=0.8 → no exception, just observed.
    assert stub_judge_funcs["name"] == "judge_output"


@pytest.mark.asyncio
async def test_judge_after_run_persists_when_thread_id_supplied(
    stub_judge_funcs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``thread_id_from(ctx)`` → routes verdict through
    ``persist_verdict_as_thread_event``."""
    del stub_judge_funcs
    from uuid import uuid4

    persisted: list[tuple[Any, JudgeVerdict, str]] = []

    async def _fake_persist(thread_id, verdict, *, subject):
        persisted.append((thread_id, verdict, subject))

    # Patch where ``capability.py`` binds the symbol — the canonical
    # location ``persistence`` module isn't enough because
    # ``capability.py`` did ``from .persistence import …`` at import.
    monkeypatch.setattr(
        "ballast.capabilities.llm_judge.capability."
        "persist_verdict_as_thread_event",
        _fake_persist,
    )

    tid = uuid4()
    cap = JudgeAfterRun(
        LLMJudge("R"),
        subject="assistant-turn",
        thread_id_from=lambda _ctx: tid,
    )
    await cap.after_run(
        _FakeRunContext(),
        result=_FakeAgentRunResult(output="hi"),
    )
    assert len(persisted) == 1
    assert persisted[0][0] == tid
    assert persisted[0][2] == "assistant-turn"


@pytest.mark.asyncio
async def test_judge_after_run_invokes_on_verdict_callback(
    stub_judge_funcs: dict[str, Any],
) -> None:
    del stub_judge_funcs
    seen: list[JudgeVerdict] = []

    async def _on_verdict(v: JudgeVerdict, _ctx: Any) -> None:
        seen.append(v)

    cap = JudgeAfterRun(LLMJudge("R"), on_verdict=_on_verdict)
    await cap.after_run(
        _FakeRunContext(),
        result=_FakeAgentRunResult(output="hi"),
    )
    assert len(seen) == 1
    assert seen[0].pass_ is True


@pytest.mark.asyncio
async def test_judge_after_run_propagates_judgefailed_on_sync_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrapping a ``sync=True`` judge means threshold failures abort
    the agent run — the capability does not swallow ``JudgeFailed``."""
    failing = GradingOutput(reason="bad", pass_=False, score=0.1)

    async def _stub(*_args: Any, **_kwargs: Any) -> GradingOutput:
        return failing

    from pydantic_evals.evaluators import llm_as_a_judge
    monkeypatch.setattr(llm_as_a_judge, "judge_output", _stub)

    cap = JudgeAfterRun(LLMJudge("R", threshold=0.5, sync=True))
    with pytest.raises(JudgeFailed):
        await cap.after_run(
            _FakeRunContext(),
            result=_FakeAgentRunResult(output="x"),
        )


# ── retry / JudgeUnavailable ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_grade_raises_judge_unavailable_with_zero_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_retries=0`` (default): one attempt, exception wrapped as
    ``JudgeUnavailable`` with ``attempts=1``."""
    calls = {"n": 0}

    async def _flaky(*_args: Any, **_kwargs: Any) -> GradingOutput:
        calls["n"] += 1
        raise RuntimeError("simulated 429")

    from pydantic_evals.evaluators import llm_as_a_judge
    monkeypatch.setattr(llm_as_a_judge, "judge_output", _flaky)

    judge = LLMJudge("R", max_retries=0)
    with pytest.raises(JudgeUnavailable) as exc_info:
        await judge.grade("x")
    assert calls["n"] == 1
    assert exc_info.value.attempts == 1
    assert isinstance(exc_info.value.last_error, RuntimeError)


@pytest.mark.asyncio
async def test_grade_retries_until_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First two calls raise, third succeeds → verdict returned, no
    JudgeUnavailable."""
    calls = {"n": 0}

    async def _flaky_then_ok(*_args: Any, **_kwargs: Any) -> GradingOutput:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("model timed out")
        return GradingOutput(reason="ok", pass_=True, score=0.9)

    # Skip the asyncio.sleep so the test stays fast.
    async def _no_sleep(_seconds: float) -> None:
        return None

    from pydantic_evals.evaluators import llm_as_a_judge
    monkeypatch.setattr(llm_as_a_judge, "judge_output", _flaky_then_ok)
    monkeypatch.setattr("asyncio.sleep", _no_sleep)

    judge = LLMJudge("R", max_retries=2, retry_backoff_base_s=0.01)
    verdict = await judge.grade("x")
    assert calls["n"] == 3
    assert verdict.score == 0.9


@pytest.mark.asyncio
async def test_grade_exhausts_retries_and_raises_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All ``max_retries + 1`` attempts fail → JudgeUnavailable with
    ``attempts=`` count."""
    calls = {"n": 0}

    async def _always_failing(*_args: Any, **_kwargs: Any) -> GradingOutput:
        calls["n"] += 1
        raise ConnectionError("no route to host")

    async def _no_sleep(_seconds: float) -> None:
        return None

    from pydantic_evals.evaluators import llm_as_a_judge
    monkeypatch.setattr(llm_as_a_judge, "judge_output", _always_failing)
    monkeypatch.setattr("asyncio.sleep", _no_sleep)

    judge = LLMJudge("R", max_retries=2, retry_backoff_base_s=0.01)
    with pytest.raises(JudgeUnavailable) as exc_info:
        await judge.grade("x")
    assert calls["n"] == 3  # initial + 2 retries
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_error, ConnectionError)


def test_constructor_rejects_negative_max_retries() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        LLMJudge("R", max_retries=-1)


def test_set_default_judge_model_forwards_to_pydantic_evals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``set_default_judge_model`` is a thin pass-through to the
    pydantic-evals function; it should be called with the same
    argument."""
    seen: list[Any] = []

    def _capture(model: Any) -> None:
        seen.append(model)

    from pydantic_evals.evaluators import llm_as_a_judge
    monkeypatch.setattr(llm_as_a_judge, "set_default_judge_model", _capture)

    set_default_judge_model("openrouter:qwen/qwen-3.6-72b-instruct")
    assert seen == ["openrouter:qwen/qwen-3.6-72b-instruct"]

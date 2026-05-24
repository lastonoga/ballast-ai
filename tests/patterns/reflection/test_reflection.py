"""``Reflection`` durable loop tests.

Use plain callable writer/critic to keep tests deterministic and
out of the LLM-call path. DBOS is launched once per module via the
``fresh_dbos_executor`` fixture so the ``@Durable.workflow`` / step
decorators have a runtime to register against.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from ballast import Critique
from ballast.events.context import progress_to_thread
from ballast.patterns.reflection import (
    Reflection,
    ReflectionEvent,
    ReflectionExhausted,
    reflection_progress,
)


@pytest.mark.asyncio
async def test_reflection_passes_first_iteration(
    fresh_dbos_executor: None,
) -> None:
    """Critic passes on first draft → run returns immediately, no
    refine event."""
    calls = {"writer": 0, "critic": 0}

    async def writer(task: str, history: list[Critique]) -> str:
        calls["writer"] += 1
        return f"draft({task})"

    async def critic(draft: str) -> Critique:
        calls["critic"] += 1
        return Critique(passed=True, confidence=1.0)

    refl = Reflection[str, str](
        writer=writer, critic=critic,
        config_name=f"test-refl-{uuid4()}",
    )
    out = await refl.run("topic")

    assert out == "draft(topic)"
    assert calls == {"writer": 1, "critic": 1}


@pytest.mark.asyncio
async def test_reflection_loops_then_passes(
    fresh_dbos_executor: None,
) -> None:
    """First critic fails with issues; second draft passes. The
    writer must receive the accumulated critique history."""
    seen_history: list[list[Critique]] = []

    async def writer(task: str, history: list[Critique]) -> str:
        seen_history.append(list(history))
        return f"draft_v{len(history) + 1}"

    async def critic(draft: str) -> Critique:
        if draft.endswith("v1"):
            return Critique(
                passed=False, issues=["too short"], confidence=0.3,
            )
        return Critique(passed=True, confidence=0.9)

    refl = Reflection[str, str](
        writer=writer, critic=critic, max_iter=5,
        config_name=f"test-refl-{uuid4()}",
    )
    out = await refl.run("x")

    assert out == "draft_v2"
    assert len(seen_history) == 2
    assert seen_history[0] == []                     # first writer call
    assert len(seen_history[1]) == 1                  # second call sees one critique
    assert seen_history[1][0].issues == ["too short"]


@pytest.mark.asyncio
async def test_reflection_exhausted_raises_with_last_critique(
    fresh_dbos_executor: None,
) -> None:
    """All critiques fail → ReflectionExhausted carries iterations +
    last_critique so handlers can route to HITL with context."""

    async def writer(task: str, history: list[Critique]) -> str:
        return f"v{len(history) + 1}"

    async def critic(draft: str) -> Critique:
        return Critique(
            passed=False,
            issues=[f"still wrong at {draft}"],
            suggestions=["try again"],
            confidence=0.1,
        )

    refl = Reflection[str, str](
        writer=writer, critic=critic, max_iter=3,
        config_name=f"test-refl-{uuid4()}",
    )
    with pytest.raises(ReflectionExhausted) as exc_info:
        await refl.run("topic")
    assert exc_info.value.iterations == 3
    assert exc_info.value.last_critique.issues == ["still wrong at v3"]
    # ``last_draft`` is the most recent writer output — callers can
    # persist it as a best-effort fallback instead of dropping work.
    assert exc_info.value.last_draft == "v3"


@pytest.mark.asyncio
async def test_reflection_emits_typed_progress_events(
    fresh_dbos_executor: None,
) -> None:
    """Every iteration boundary should fire on reflection_progress.

    Subscribe a side-channel receiver and assert the exact sequence:
    draft → critique → refine → draft → critique → passed.
    """
    seen: list[ReflectionEvent] = []

    async def _collect(sender: Any, *, event: ReflectionEvent, **_: Any) -> None:
        seen.append(event)

    reflection_progress.connect(_collect)

    async def writer(task: str, history: list[Critique]) -> str:
        return f"v{len(history) + 1}"

    async def critic(draft: str) -> Critique:
        return Critique(passed=draft.endswith("v2"))

    refl = Reflection[str, str](
        writer=writer, critic=critic, max_iter=3,
        config_name=f"test-refl-{uuid4()}",
    )
    await refl.run("t")

    types = [e.type for e in seen]
    assert types == ["draft", "critique", "refine", "draft", "critique", "passed"]
    # The "critique" event payload is the Critique itself.
    critique_event = next(e for e in seen if e.type == "critique" and e.iter == 1)
    assert critique_event.payload["passed"] is False


@pytest.mark.asyncio
async def test_reflection_default_chat_router_no_op_without_progress_scope(
    fresh_dbos_executor: None,
) -> None:
    """Without ``progress_to_thread(...)`` scope, the bundled chat
    router is a no-op — no broadcaster lookup, no exception."""

    async def writer(task: str, history: list[Critique]) -> str:
        return "ok"

    async def critic(draft: str) -> Critique:
        return Critique(passed=True)

    refl = Reflection[str, str](
        writer=writer, critic=critic,
        config_name=f"test-refl-{uuid4()}",
    )
    out = await refl.run("t")
    assert out == "ok"


def test_constructor_rejects_zero_max_iter() -> None:
    async def w(t: Any, h: list[Critique]) -> Any: return t
    async def c(d: Any) -> Critique: return Critique(passed=True)

    with pytest.raises(ValueError, match=">= 1"):
        Reflection[Any, Any](writer=w, critic=c, max_iter=0)


# ── critic adapter ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_judge_critic_adapts_to_critique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``LLMJudge`` plugs in directly as a critic — the adapter
    translates ``JudgeVerdict`` → ``Critique`` (passed ← pass_,
    issues ← [reason], confidence ← score)."""
    from pydantic_evals.evaluators.llm_as_a_judge import GradingOutput

    from ballast.capabilities.llm_judge import LLMJudge
    from ballast.patterns.reflection import to_critic_callable

    async def _stub(*_args: Any, **_kwargs: Any) -> GradingOutput:
        return GradingOutput(reason="too vague", pass_=False, score=0.3)

    from pydantic_evals.evaluators import llm_as_a_judge
    monkeypatch.setattr(llm_as_a_judge, "judge_output", _stub)

    judge = LLMJudge("Be specific", threshold=0.7)
    critic = to_critic_callable(judge)
    critique = await critic("vague answer")

    assert critique.passed is False
    assert critique.issues == ["too vague"]
    assert critique.confidence == 0.3


@pytest.mark.asyncio
async def test_to_critic_callable_passes_through_plain_callable() -> None:
    """A regular async callable is returned unchanged."""
    from ballast.patterns.reflection import to_critic_callable

    async def my_critic(out: Any) -> Critique:
        return Critique(passed=True)

    assert to_critic_callable(my_critic) is my_critic

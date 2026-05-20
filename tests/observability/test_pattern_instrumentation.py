"""Smoke: the wrapped patterns still satisfy their old contract.

We don't assert on logfire output (covered in test_traced); we assert that
existing pattern behaviour is unchanged after wrapping `.run` with @traced.
"""
from __future__ import annotations

import pytest

from pydantic_ai_stateflow.capabilities.helpers import Critique
from pydantic_ai_stateflow.patterns import Reflection


@pytest.mark.asyncio
async def test_reflection_run_still_works_after_instrumentation(
    fresh_dbos_executor: None,
):
    calls = {"writer": 0, "critic": 0}

    async def writer(task: str) -> str:
        calls["writer"] += 1
        return f"draft:{task}"

    async def critic(draft: str) -> Critique:
        calls["critic"] += 1
        return Critique(passed=True, feedback="ok")

    ref: Reflection[str, str] = Reflection(writer, critic, max_iterations=1)
    out = await ref.run("input")
    assert out == "draft:input"
    assert calls == {"writer": 1, "critic": 1}

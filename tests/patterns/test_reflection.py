from __future__ import annotations

import hashlib
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel

from ballast.capabilities.helpers import Critique, TypedLoopGuard
from ballast.patterns import AbortOnLoop, Reflection, ReflectionExhausted


class Draft(BaseModel):
    body: str
    revision: int = 0


@pytest.mark.asyncio
async def test_reflection_returns_first_passing_draft(
    fresh_dbos_executor: None,
) -> None:
    write_calls = {"n": 0}

    async def writer_fn(task: str) -> Draft:
        write_calls["n"] += 1
        return Draft(body=f"draft_{write_calls['n']}", revision=write_calls["n"])

    async def critic_fn(payload: Any) -> Critique:
        return Critique(passed=True)

    pattern = Reflection[str, Draft](
        writer_fn,
        critic_fn,
        max_iterations=3,
    )
    result = await pattern.run("seed")
    assert result.body == "draft_1"
    assert write_calls["n"] == 1


@pytest.mark.asyncio
async def test_reflection_loops_until_passed(
    fresh_dbos_executor: None,
) -> None:
    write_calls = {"n": 0}

    async def writer_fn(task: str) -> Draft:
        write_calls["n"] += 1
        return Draft(body=f"v{write_calls['n']}", revision=write_calls["n"])

    async def critic_fn(payload: Any) -> Critique:
        rev = payload.get("revision", 0) if isinstance(payload, dict) else 0
        return Critique(
            passed=rev >= 3,
            issues=[] if rev >= 3 else [f"need more (rev={rev})"],
        )

    pattern = Reflection[str, Draft](
        writer_fn,
        critic_fn,
        max_iterations=5,
    )
    result = await pattern.run("seed")
    assert result.revision == 3
    assert write_calls["n"] == 3


@pytest.mark.asyncio
async def test_reflection_raises_exhausted_when_never_passes(
    fresh_dbos_executor: None,
) -> None:
    async def writer_fn(task: str) -> Draft:
        return Draft(body="never-good")

    async def critic_fn(payload: Any) -> Critique:
        return Critique(passed=False, issues=["still bad"])

    pattern = Reflection[str, Draft](
        writer_fn,
        critic_fn,
        max_iterations=2,
    )
    with pytest.raises(ReflectionExhausted) as exc:
        await pattern.run("seed")
    assert exc.value.iterations == 2
    assert len(exc.value.last_feedback) == 2


@pytest.mark.asyncio
async def test_reflection_invokes_loop_recovery_on_repeated_drafts(
    fresh_dbos_executor: None,
) -> None:
    class _IdentityEmbedder:
        async def embed(self, text: str) -> list[float]:
            h = hashlib.md5(text.encode()).digest()
            return [float(b) for b in h[:6]]

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [await self.embed(t) for t in texts]

    async def writer_fn(task: str) -> Draft:
        return Draft(body="identical")

    async def critic_fn(payload: Any) -> Critique:
        return Critique(passed=False, issues=["bad"])

    class AcceptLast:
        async def handle(
            self, ctx: Any, draft: Draft, feedback: list[Critique]
        ) -> Draft:
            return draft

    guard: TypedLoopGuard[Draft] = TypedLoopGuard(
        embedder=_IdentityEmbedder(),
        selector=lambda d: d.body,
        threshold=0.99,
        window=2,
    )
    pattern = Reflection[str, Draft](
        writer_fn,
        critic_fn,
        max_iterations=10,
        loop_guard=guard,
        loop_recovery=AcceptLast(),
    )
    result = await pattern.run("seed")
    assert result.body == "identical"


@pytest.mark.asyncio
async def test_reflection_default_loop_recovery_is_abort_on_loop() -> None:
    pattern = Reflection[str, Draft](
        lambda t: Draft(body="x"),
        lambda p: Critique(passed=True),
    )
    assert isinstance(pattern.loop_recovery, AbortOnLoop)

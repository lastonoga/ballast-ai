"""``RememberTurn`` — capability that writes episodes after successful turns."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import (
    Episode, EpisodicMemory, RememberTurn,
)


class _FakeSource:
    name = "fake"
    def __init__(self): self.remembered: list[Episode] = []
    async def recall(self, **_): return []
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode): self.remembered.append(episode)


class _FakeResult:
    """Minimal stand-in for AgentRunResult."""
    def __init__(self, output: str = "hello"): self.output = output


class _FakeDeps:
    parent_thread_id = None
    user_id = "u-1"


class _FakeCtx:
    """Minimal stand-in for RunContext."""
    def __init__(self): self.deps = _FakeDeps()


@pytest.mark.asyncio
async def test_remember_turn_writes_on_pass() -> None:
    src = _FakeSource()
    mem = EpisodicMemory(sources=[src])
    cap = RememberTurn(memory=mem, gate=lambda *_: True)
    result = _FakeResult()
    out = await cap.after_run(_FakeCtx(), result=result)
    assert out is result    # MUST return the result unchanged
    assert len(src.remembered) == 1
    assert src.remembered[0].preview != ""


@pytest.mark.asyncio
async def test_remember_turn_skips_when_gate_fails() -> None:
    src = _FakeSource()
    mem = EpisodicMemory(sources=[src])
    cap = RememberTurn(memory=mem, gate=lambda *_: False)
    await cap.after_run(_FakeCtx(), result=_FakeResult())
    assert src.remembered == []


@pytest.mark.asyncio
async def test_remember_turn_swallows_exceptions() -> None:
    """Failures inside RememberTurn must NOT block the agent turn."""
    class _BrokenSource:
        name = "broken"
        async def recall(self, **_): return []
        async def hydrate(self, episode, *, detail): return episode
        async def remember(self, episode): raise RuntimeError("oops")
    mem = EpisodicMemory(sources=[_BrokenSource()])
    cap = RememberTurn(memory=mem, gate=lambda *_: True)
    result = _FakeResult()
    out = await cap.after_run(_FakeCtx(), result=result)
    assert out is result   # still returned, exception swallowed

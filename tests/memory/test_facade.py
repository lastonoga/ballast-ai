"""``EpisodicMemory`` facade — dispatches recall via strategy."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import (
    DetailLevel, Episode, EpisodicMemory, RecallResult, ScoredEpisode,
)
from ballast.memory.episodic.strategies import TopK


class _FakeSource:
    def __init__(self, returns): self.name = "fake"; self._r = returns
    async def recall(self, **_): return self._r
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: self.last_remembered = episode


def _se(id_: str) -> ScoredEpisode:
    return ScoredEpisode(
        episode=Episode(id=id_, source="fake",
                        occurred_at=datetime.now(UTC),
                        scope=Scope(), preview="p"),
        score=0.9,
    )


@pytest.mark.asyncio
async def test_episodic_for_runs_strategy() -> None:
    src = _FakeSource([_se("a"), _se("b")])
    mem = EpisodicMemory(sources=[src], default_strategy=TopK(k=1))
    out = await mem.episodic_for(intent="x")
    assert isinstance(out, RecallResult)
    assert len(out.episodes) == 1


@pytest.mark.asyncio
async def test_default_scope_builder_called_if_no_scope_passed() -> None:
    captured: list[Scope] = []
    class _SpySrc:
        name = "spy"
        async def recall(self, *, intent, scope, k, detail):
            captured.append(scope); return []
        async def hydrate(self, episode, *, detail): return episode
        async def remember(self, episode) -> None: return None
    spy = _SpySrc()
    mem = EpisodicMemory(
        sources=[spy],
        default_scope_builder=lambda: Scope(user_id="from-builder"),
    )
    await mem.episodic_for(intent="x")
    assert captured[0].user_id == "from-builder"


@pytest.mark.asyncio
async def test_remember_fans_out_to_writable_sources() -> None:
    writable = _FakeSource([])
    class _ReadOnly:
        name = "ro"
        async def recall(self, **_): return []
        async def hydrate(self, episode, *, detail): return episode
        async def remember(self, episode) -> None:
            raise NotImplementedError()
    mem = EpisodicMemory(sources=[writable, _ReadOnly()])
    ep = Episode(id="x", source="fake",
                 occurred_at=datetime.now(UTC), scope=Scope(), preview="p")
    await mem.remember(ep)
    assert writable.last_remembered.id == "x"

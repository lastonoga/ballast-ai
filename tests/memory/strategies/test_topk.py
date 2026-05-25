"""``TopK`` strategy — parallel query → merge → first K."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import DetailLevel, Episode, ScoredEpisode
from ballast.memory.episodic.strategies import TopK


class _FakeSource:
    def __init__(self, name: str, returns: list[ScoredEpisode]) -> None:
        self.name = name
        self._returns = returns
    async def recall(self, *, intent, scope, k, detail):
        return self._returns
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


def _ep(i: str) -> Episode:
    return Episode(
        id=i, source="x", occurred_at=datetime.now(UTC),
        scope=Scope(), preview="p",
    )


@pytest.mark.asyncio
async def test_topk_returns_top_k_by_merged_score() -> None:
    src1 = _FakeSource("s1", [
        ScoredEpisode(episode=_ep("a"), score=0.9),
        ScoredEpisode(episode=_ep("b"), score=0.4),
    ])
    src2 = _FakeSource("s2", [
        ScoredEpisode(episode=_ep("c"), score=0.8),
    ])
    out = await TopK(k=2).execute(
        intent="x", sources=[src1, src2], scope=Scope(),
    )
    assert len(out.episodes) == 2


@pytest.mark.asyncio
async def test_topk_resilient_to_source_failure() -> None:
    class _Broken:
        name = "broken"
        async def recall(self, **_): raise RuntimeError("down")
        async def hydrate(self, episode, *, detail): return episode
        async def remember(self, episode) -> None: return None

    ok = _FakeSource("ok", [ScoredEpisode(episode=_ep("a"), score=0.9)])
    out = await TopK(k=5).execute(
        intent="x", sources=[_Broken(), ok], scope=Scope(),
    )
    assert len(out.episodes) == 1
    assert out.episodes[0].episode.id == "a"

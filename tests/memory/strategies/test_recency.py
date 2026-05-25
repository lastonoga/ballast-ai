"""``Recency`` — sort by occurred_at desc; scores ignored."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import Episode, ScoredEpisode
from ballast.memory.episodic.strategies import Recency


class _FakeSource:
    def __init__(self, returns): self.name = "x"; self._r = returns
    async def recall(self, **_): return self._r
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


def _se(id_: str, dt: datetime, score: float) -> ScoredEpisode:
    return ScoredEpisode(
        episode=Episode(id=id_, source="x", occurred_at=dt,
                        scope=Scope(), preview="p"),
        score=score,
    )


@pytest.mark.asyncio
async def test_recency_orders_by_occurred_at_desc() -> None:
    now = datetime.now(UTC)
    src = _FakeSource([
        _se("old", now - timedelta(days=7), 0.9),
        _se("new", now,                     0.1),
        _se("mid", now - timedelta(days=2), 0.5),
    ])
    out = await Recency(n=3).execute(
        intent="x", sources=[src], scope=Scope(),
    )
    assert [se.episode.id for se in out.episodes] == ["new", "mid", "old"]


@pytest.mark.asyncio
async def test_recency_n_caps_results() -> None:
    now = datetime.now(UTC)
    src = _FakeSource([
        _se(f"e-{i}", now - timedelta(days=i), 0.0) for i in range(10)
    ])
    out = await Recency(n=3).execute(intent="x", sources=[src], scope=Scope())
    assert len(out.episodes) == 3

"""``MapReduce`` strategy — LLM-driven digest for large result sets."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import Episode, ScoredEpisode
from ballast.memory.episodic.strategies import MapReduce


class _FakeSource:
    def __init__(self, returns): self.name = "x"; self._r = returns
    async def recall(self, **_): return self._r
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


def _se(id_: str) -> ScoredEpisode:
    return ScoredEpisode(
        episode=Episode(id=id_, source="x", occurred_at=datetime.now(UTC),
                        scope=Scope(), preview=f"preview {id_}"),
        score=0.5,
    )


@pytest.mark.asyncio
async def test_map_reduce_strategy_calls_map_and_reduce() -> None:
    map_calls: list[str] = []
    reduce_calls: list[int] = []

    async def map_fn(ep: ScoredEpisode) -> str:
        map_calls.append(ep.episode.id)
        return f"M({ep.episode.id})"

    async def reduce_fn(items: list[str]) -> str:
        reduce_calls.append(len(items))
        return ", ".join(items)

    src = _FakeSource([_se(str(i)) for i in range(5)])
    out = await MapReduce(
        max_items=10,
        map_fn=map_fn,
        reduce_fn=reduce_fn,
    ).execute(intent="x", sources=[src], scope=Scope())

    assert sorted(map_calls) == ["0", "1", "2", "3", "4"]
    assert reduce_calls == [5]
    assert len(out.episodes) == 1                              # single synthesized episode
    assert out.episodes[0].episode.preview == "M(0), M(1), M(2), M(3), M(4)"

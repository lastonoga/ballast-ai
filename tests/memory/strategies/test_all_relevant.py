"""``AllRelevant`` — return everything above a threshold."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import Episode, ScoredEpisode
from ballast.memory.episodic.strategies import AllRelevant


class _FakeSource:
    def __init__(self, returns): self.name = "x"; self._r = returns
    async def recall(self, **_): return self._r
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


def _se(id_: str, score: float) -> ScoredEpisode:
    return ScoredEpisode(
        episode=Episode(id=id_, source="x", occurred_at=datetime.now(UTC),
                        scope=Scope(), preview="p"),
        score=score,
    )


@pytest.mark.asyncio
async def test_all_relevant_filters_by_threshold() -> None:
    src = _FakeSource([_se("a", 0.9), _se("b", 0.4), _se("c", 0.7)])
    out = await AllRelevant(threshold=0.5).execute(
        intent="x", sources=[src], scope=Scope(),
    )
    ids = {se.episode.id for se in out.episodes}
    assert ids == {"a", "c"}

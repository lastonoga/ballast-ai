"""``Cluster`` — semantic dedup: one medoid per cluster."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import Episode, ScoredEpisode
from ballast.memory.episodic.strategies import Cluster


class _FakeEmbedder:
    """Returns fixed embeddings keyed by episode preview content."""
    _table = {
        "alpha":  [1.0, 0.0],
        "alpha2": [0.95, 0.05],   # very close to alpha
        "beta":   [0.0, 1.0],
        "beta2":  [0.05, 0.95],   # very close to beta
    }
    async def embed(self, text): return self._table[text]
    async def embed_batch(self, texts): return [self._table[t] for t in texts]


class _FakeSource:
    def __init__(self, returns): self.name = "x"; self._r = returns
    async def recall(self, **_): return self._r
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


def _se(preview: str) -> ScoredEpisode:
    return ScoredEpisode(
        episode=Episode(id=preview, source="x", occurred_at=datetime.now(UTC),
                        scope=Scope(), preview=preview),
        score=0.5,
    )


@pytest.mark.asyncio
async def test_cluster_returns_one_per_cluster() -> None:
    src = _FakeSource([_se("alpha"), _se("alpha2"), _se("beta"), _se("beta2")])
    out = await Cluster(n_clusters=2, embedder=_FakeEmbedder()).execute(
        intent="x", sources=[src], scope=Scope(),
    )
    assert len(out.episodes) == 2
    ids = {se.episode.id for se in out.episodes}
    # One representative from each cluster
    assert (("alpha" in ids or "alpha2" in ids)
            and ("beta" in ids or "beta2" in ids))

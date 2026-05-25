"""``VectorEpisodicSource`` — pgvector-backed semantic recall."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import DetailLevel, Episode
from ballast.memory.episodic.sources import VectorEpisodicSource


def _pad(vec: list[float], dim: int = 1536) -> list[float]:
    return vec + [0.0] * (dim - len(vec))


class _FakeEmbedder:
    """Returns 1536-dim embeddings keyed by text."""

    _table = {
        "ML in production": _pad([1.0, 0.0, 0.0, 0.0]),
        "fashion trends": _pad([0.0, 1.0, 0.0, 0.0]),
        "ml deployment": _pad([0.99, 0.01, 0.0, 0.0]),
    }

    async def embed(self, text: str) -> list[float]:
        return self._table[text]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._table[t] for t in texts]


@pytest.mark.asyncio
async def test_vector_source_remember_then_recall(
    session_factory,
) -> None:
    src = VectorEpisodicSource(
        sessionmaker=session_factory,
        embedder=_FakeEmbedder(),
    )
    ep_ml = Episode(
        id="ep-1",
        source="vector",
        occurred_at=datetime(2026, 5, 25, tzinfo=UTC),
        scope=Scope(user_id="u-1"),
        preview="ML in production",
        summary="ML in production",
    )
    ep_fashion = Episode(
        id="ep-2",
        source="vector",
        occurred_at=datetime(2026, 5, 24, tzinfo=UTC),
        scope=Scope(user_id="u-1"),
        preview="fashion trends",
        summary="fashion trends",
    )
    await src.remember(ep_ml)
    await src.remember(ep_fashion)

    out = await src.recall(
        intent="ml deployment",
        scope=Scope(user_id="u-1"),
        k=2,
        detail=DetailLevel.SUMMARY,
    )
    # ml deployment ≈ ML in production (cosine ~1.0), fashion ~0
    assert out[0].episode.id == "ep-1"

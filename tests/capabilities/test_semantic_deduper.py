import pytest

from pydantic_ai_stateflow.capabilities.helpers import (
    SemanticDeduper,
    SemanticLoopDetected,
)


class _IdentityEmbedder:
    """Returns a deterministic 'embedding' so we can craft exact-match scenarios."""
    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = mapping

    async def embed(self, text: str) -> list[float]:
        return self._mapping[text]

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


@pytest.mark.asyncio
async def test_deduper_does_not_fire_below_window():
    """First few snapshots fill the window; no detection yet."""
    e = _IdentityEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    d = SemanticDeduper(e)
    await d.add_and_check("a", threshold=0.95, window=3)
    await d.add_and_check("b", threshold=0.95, window=3)
    # No exception — fewer than `window` snapshots seen


@pytest.mark.asyncio
async def test_deduper_fires_when_window_filled_with_similar():
    """Three near-identical embeddings exceed cosine threshold."""
    e = _IdentityEmbedder({
        "x1": [1.0, 0.0],
        "x2": [1.0, 0.0],
        "x3": [1.0, 0.0],
    })
    d = SemanticDeduper(e)
    await d.add_and_check("x1", threshold=0.95, window=3)
    await d.add_and_check("x2", threshold=0.95, window=3)
    with pytest.raises(SemanticLoopDetected):
        await d.add_and_check("x3", threshold=0.95, window=3)


@pytest.mark.asyncio
async def test_deduper_does_not_fire_when_diverse():
    """Window of dissimilar embeddings is allowed."""
    e = _IdentityEmbedder({
        "a": [1.0, 0.0, 0.0],
        "b": [0.0, 1.0, 0.0],
        "c": [0.0, 0.0, 1.0],
        "d": [-1.0, 0.0, 0.0],
    })
    d = SemanticDeduper(e)
    for s in ["a", "b", "c", "d"]:
        await d.add_and_check(s, threshold=0.95, window=3)
    # Never raises


@pytest.mark.asyncio
async def test_deduper_sliding_window_drops_old_entries():
    """After window is full, oldest entry is dropped on insert."""
    e = _IdentityEmbedder({
        "old": [1.0, 0.0],
        "n1": [0.0, 1.0],
        "n2": [0.0, 1.0],
        "n3": [0.0, 1.0],
    })
    d = SemanticDeduper(e)
    await d.add_and_check("old", threshold=0.95, window=2)
    await d.add_and_check("n1", threshold=0.95, window=2)
    # n2 + n1 == similar; old should have been dropped
    with pytest.raises(SemanticLoopDetected):
        await d.add_and_check("n2", threshold=0.95, window=2)

from __future__ import annotations

import hashlib

import pytest
from pydantic import BaseModel

from ballast.patterns import (
    SemanticDedup,
    SemanticDedupConfig,
)


class _StableEmbedder:
    """Deterministic md5-derived vector with a soft override.

    The override lets a test pin two strings to identical vectors (i.e.
    cosine = 1.0) so we can assert dedup catches them, without faking
    the entire embedder contract."""

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self._aliases = aliases or {}

    def _vec(self, text: str) -> list[float]:
        key = self._aliases.get(text, text)
        h = hashlib.md5(key.encode()).digest()
        return [float(b) for b in h[:8]]

    async def embed(self, text: str) -> list[float]:
        return self._vec(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class _Item(BaseModel):
    title: str
    body: str


@pytest.mark.asyncio
async def test_returns_empty_for_empty_input() -> None:
    dedup = SemanticDedup[_Item](
        embedder=_StableEmbedder(),
        projector=lambda i: i.title,
    )
    assert await dedup.run([]) == []


@pytest.mark.asyncio
async def test_preserves_distinct_items_in_order() -> None:
    items = [_Item(title=t, body="b") for t in ("a", "b", "c")]
    dedup = SemanticDedup[_Item](
        embedder=_StableEmbedder(),
        projector=lambda i: i.title,
    )
    result = await dedup.run(items)
    assert [i.title for i in result] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_collapses_aliased_duplicate_keeping_first() -> None:
    # Pin "a-second" to embed identically to "a-first" by aliasing them
    # to the same hash input. Threshold 0.99 → only exact-cosine match
    # collapses.
    aliases = {"a-first": "same", "a-second": "same"}
    items = [
        _Item(title="a-first", body="short"),
        _Item(title="a-second", body="much longer body"),
        _Item(title="b", body="b"),
    ]
    dedup = SemanticDedup[_Item](
        embedder=_StableEmbedder(aliases=aliases),
        projector=lambda i: i.title,
        config=SemanticDedupConfig(threshold=0.99, keep="first"),
    )
    result = await dedup.run(items)
    assert [i.title for i in result] == ["a-first", "b"]


@pytest.mark.asyncio
async def test_keep_longest_swaps_to_longer_projection() -> None:
    aliases = {"x-short": "same", "x-long-and-detailed": "same"}
    items = [
        _Item(title="x-short", body=""),
        _Item(title="x-long-and-detailed", body=""),
    ]
    dedup = SemanticDedup[_Item](
        embedder=_StableEmbedder(aliases=aliases),
        projector=lambda i: i.title,
        config=SemanticDedupConfig(threshold=0.99, keep="longest"),
    )
    result = await dedup.run(items)
    assert [i.title for i in result] == ["x-long-and-detailed"]


@pytest.mark.asyncio
async def test_threshold_validation_rejects_out_of_range() -> None:
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        SemanticDedupConfig(threshold=1.5)

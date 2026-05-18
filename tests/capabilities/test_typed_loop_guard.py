import hashlib

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.capabilities.helpers import (
    SemanticLoopDetected,
    TypedLoopGuard,
)


class _IdentityEmbedder:
    async def embed(self, text: str) -> list[float]:
        h = hashlib.md5(text.encode()).digest()
        return [float(b) for b in h[:6]]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class Draft(BaseModel):
    rationale: str
    score: int


@pytest.mark.asyncio
async def test_guard_fires_when_same_field_repeats() -> None:
    guard = TypedLoopGuard[Draft](
        embedder=_IdentityEmbedder(),
        selector=lambda d: d.rationale,
        threshold=0.99,
        window=2,
    )
    await guard.check(Draft(rationale="same reason", score=1))
    with pytest.raises(SemanticLoopDetected):
        await guard.check(Draft(rationale="same reason", score=2))


@pytest.mark.asyncio
async def test_guard_allows_diverse_field_values() -> None:
    guard = TypedLoopGuard[Draft](
        embedder=_IdentityEmbedder(),
        selector=lambda d: d.rationale,
        threshold=0.99,
        window=2,
    )
    await guard.check(Draft(rationale="first", score=1))
    await guard.check(Draft(rationale="completely different", score=2))


@pytest.mark.asyncio
async def test_guard_supports_list_selector() -> None:
    """Selector may return a list of strings (multiple fields to check)."""
    guard = TypedLoopGuard[Draft](
        embedder=_IdentityEmbedder(),
        selector=lambda d: [d.rationale, str(d.score)],
        threshold=0.99,
        window=3,
    )
    await guard.check(Draft(rationale="A", score=1))
    await guard.check(Draft(rationale="B", score=1))
    with pytest.raises(SemanticLoopDetected):
        await guard.check(Draft(rationale="C", score=1))

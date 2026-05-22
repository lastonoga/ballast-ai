import pytest

from ballast.capabilities.helpers import Embedder


class _FakeEmbedder:
    """Structural impl for testing Protocol satisfaction."""
    async def embed(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def test_fake_embedder_satisfies_protocol():
    assert isinstance(_FakeEmbedder(), Embedder)


@pytest.mark.asyncio
async def test_fake_embedder_embed_returns_vector():
    e = _FakeEmbedder()
    v = await e.embed("hello")
    assert v == [5.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_fake_embedder_embed_batch():
    e = _FakeEmbedder()
    vs = await e.embed_batch(["a", "bb"])
    assert vs == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]

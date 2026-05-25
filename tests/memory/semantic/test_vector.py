"""``VectorSemanticSource`` ABC — instantiation + protocol conformance."""
from __future__ import annotations

from typing import Any

import pytest

from ballast.memory.semantic import (
    SemanticSource,
    VectorSemanticSource,
    memory_tool,
)


class _DummyEmbedder:
    async def embed(self, text: str) -> list[float]: return [0.0]
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: return [[0.0] for _ in texts]


class _DummyMaker:
    """Just a sessionmaker placeholder for ABC-shape testing."""
    def __call__(self): raise NotImplementedError


class _SearchSource(VectorSemanticSource):
    name = "search"

    @memory_tool
    async def search(self, query: str) -> list[Any]:
        """Search."""
        return []


def test_subclass_satisfies_semantic_source_protocol() -> None:
    src = _SearchSource(embedder=_DummyEmbedder(), sessionmaker=_DummyMaker())
    assert isinstance(src, SemanticSource)
    assert src.name == "search"


def test_subclass_must_set_name() -> None:
    """Subclassing without overriding ``name`` should fail at instantiation
    or at attribute access — the ABC declares ``name: ClassVar[str]`` as required."""
    class _NoName(VectorSemanticSource):
        pass
    # Without name, the class still instantiates (no metaclass check), but
    # `name` attribute access raises AttributeError if not set.
    with pytest.raises(AttributeError):
        _NoName(embedder=_DummyEmbedder(), sessionmaker=_DummyMaker()).name  # type: ignore[misc]

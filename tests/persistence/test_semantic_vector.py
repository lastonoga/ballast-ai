"""``VectorSemanticSource._vector_search`` — generic cosine-search helper."""
from __future__ import annotations

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import Field, SQLModel

from ballast.memory.semantic import VectorSemanticSource, memory_tool


class _DocRow(SQLModel, table=True):
    """Throwaway test table — pads vectors to 1536 (the standard dim)."""

    __tablename__ = "_test_semantic_doc_row"

    id:        str = Field(primary_key=True)
    text:      str
    embedding: list[float] = Field(sa_column=Column(Vector(1536), nullable=False))


def _pad(vec: list[float]) -> list[float]:
    return vec + [0.0] * (1536 - len(vec))


class _FakeEmbedder:
    _table = {
        "machine learning":   _pad([1.0, 0.0, 0.0]),
        "ml model":           _pad([0.95, 0.05, 0.0]),
        "fashion trends":     _pad([0.0, 1.0, 0.0]),
    }
    async def embed(self, text: str) -> list[float]:
        return self._table[text]
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._table[t] for t in texts]


class _DocsSemantic(VectorSemanticSource):
    name = "docs"

    @memory_tool
    async def search(self, query: str, k: int = 3) -> list[_DocRow]:
        """Find docs whose text is semantically similar to `query`."""
        return await self._vector_search(
            query=query,
            table=_DocRow,
            embedding_column=_DocRow.embedding,
            k=k,
        )


@pytest.mark.asyncio
async def test_vector_search_returns_cosine_ordered(
    session_factory,
) -> None:
    # Seed two docs
    async with session_factory() as session:
        async with session.begin():
            session.add_all([
                _DocRow(id="ml", text="machine learning",
                        embedding=_pad([1.0, 0.0, 0.0])),
                _DocRow(id="fashion", text="fashion trends",
                        embedding=_pad([0.0, 1.0, 0.0])),
            ])

    src = _DocsSemantic(embedder=_FakeEmbedder(), sessionmaker=session_factory)
    results = await src.search(query="ml model", k=2)
    assert results[0].id == "ml"


@pytest.mark.asyncio
async def test_vector_search_respects_k(
    session_factory,
) -> None:
    async with session_factory() as session:
        async with session.begin():
            session.add_all([
                _DocRow(id=f"row-{i}", text="machine learning",
                        embedding=_pad([1.0 - i * 0.1, 0.0, 0.0]))
                for i in range(5)
            ])

    src = _DocsSemantic(embedder=_FakeEmbedder(), sessionmaker=session_factory)
    results = await src.search(query="ml model", k=3)
    assert len(results) == 3

import pytest

from ballast.patterns.mapreduce import Chunker, Reducer


class _StringChunker:
    def chunk(self, doc: str) -> list[str]:
        return doc.split()


class _SumReducer:
    async def reduce(self, items: list[int]) -> list[int]:
        return [sum(items)]


def test_chunker_protocol_satisfied_by_structural_class():
    assert isinstance(_StringChunker(), Chunker)


def test_reducer_protocol_satisfied_by_structural_class():
    assert isinstance(_SumReducer(), Reducer)


@pytest.mark.asyncio
async def test_chunker_returns_chunks():
    c = _StringChunker()
    assert c.chunk("hello world foo") == ["hello", "world", "foo"]


@pytest.mark.asyncio
async def test_reducer_returns_aggregate():
    r = _SumReducer()
    assert await r.reduce([1, 2, 3, 4]) == [10]

from __future__ import annotations

from uuid import uuid4

import pytest

from pydantic_ai_stateflow.patterns import MapReduce


class WordChunker:
    def chunk(self, doc: str) -> list[str]:
        return doc.split()


class UniqueLowercaseReducer:
    async def reduce(self, items: list[str]) -> list[str]:
        seen: list[str] = []
        for it in items:
            if it.lower() not in seen:
                seen.append(it.lower())
        return seen


@pytest.mark.asyncio
async def test_mapreduce_processes_chunks_and_reduces(
    fresh_dbos_executor: None,
) -> None:
    async def extractor(chunk: str) -> str | None:
        return chunk.upper() if chunk else None

    pattern = MapReduce[str, str, str](
        chunker=WordChunker(),
        extractor=extractor,
        reducer=UniqueLowercaseReducer(),
        concurrency=2,
    )
    result = await pattern.run("foo bar foo baz", tenant_id=uuid4())
    assert sorted(result) == ["bar", "baz", "foo"]


@pytest.mark.asyncio
async def test_mapreduce_filters_none_extractor_outputs(
    fresh_dbos_executor: None,
) -> None:
    async def extractor(chunk: str) -> str | None:
        return chunk if chunk.startswith("a") else None

    pattern = MapReduce[str, str, str](
        chunker=WordChunker(),
        extractor=extractor,
        reducer=UniqueLowercaseReducer(),
    )
    result = await pattern.run("apple banana avocado cherry", tenant_id=uuid4())
    assert sorted(result) == ["apple", "avocado"]


@pytest.mark.asyncio
async def test_mapreduce_empty_doc_returns_empty(
    fresh_dbos_executor: None,
) -> None:
    async def extractor(chunk: str) -> str | None:
        return chunk

    pattern = MapReduce[str, str, str](
        chunker=WordChunker(),
        extractor=extractor,
        reducer=UniqueLowercaseReducer(),
    )
    result = await pattern.run("", tenant_id=uuid4())
    assert result == []


def test_mapreduce_default_concurrency_is_8() -> None:
    pattern = MapReduce[str, str, str](
        chunker=WordChunker(),
        extractor=lambda c: c,
        reducer=UniqueLowercaseReducer(),
    )
    assert pattern.concurrency == 8

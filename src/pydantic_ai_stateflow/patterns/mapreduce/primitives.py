from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

Doc = TypeVar("Doc", contravariant=True)
Chunk = TypeVar("Chunk")
Item = TypeVar("Item")


@runtime_checkable
class Chunker(Protocol[Doc, Chunk]):
    """Split a `Doc` into a list of `Chunk`s for parallel processing.

    `chunk` is sync because chunking is typically pure (string split,
    tokenizer slice). If your chunker needs IO, wrap it in a thin async
    facade — the framework wraps the call in @DBOS.step regardless.
    """

    def chunk(self, doc: Doc) -> list[Chunk]: ...


@runtime_checkable
class Reducer(Protocol[Item]):
    """Reduce a list of extractor outputs into a final list.

    Typical implementations: dedupe, rank, score-threshold-filter,
    merge-equivalent. Async because reducers may call agents or DBs.
    """

    async def reduce(self, items: list[Item]) -> list[Item]: ...

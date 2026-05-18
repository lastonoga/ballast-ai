from __future__ import annotations

import asyncio
import inspect
import itertools
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Generic, TypeVar
from uuid import UUID

from dbos import DBOS, DBOSConfiguredInstance

from pydantic_ai_stateflow.patterns.mapreduce.primitives import Chunker, Reducer

Doc = TypeVar("Doc")
Chunk = TypeVar("Chunk")
Item = TypeVar("Item")

Extractor = Callable[[Chunk], Awaitable[Item | None]] | Callable[[Chunk], Item | None]


async def _ensure_async(fn: Callable[..., Any], *args: Any) -> Any:
    result = fn(*args)
    if inspect.isawaitable(result):
        return await result
    return result


_instance_counter = itertools.count()


@DBOS.dbos_class()
class MapReduce(DBOSConfiguredInstance, Generic[Doc, Chunk, Item]):
    """Chunk -> parallel-extract -> reduce, bounded by ``concurrency``.

    Each ``_extract_one`` runs as its own @DBOS.step so failures retry in
    isolation. ``asyncio.Semaphore`` bounds in-flight extractors so a
    10K-chunk document does not thunder-herd downstream services.

    Satisfies ``Pattern[Doc, list[Item]]`` structurally (not via
    inheritance). ``DBOSConfiguredInstance`` is required so per-instance
    workflows / steps can be addressed by config_name during recovery.
    """

    name: ClassVar[str] = "map_reduce"

    def __init__(
        self,
        chunker: Chunker[Doc, Chunk],
        extractor: Extractor[Chunk, Item],
        reducer: Reducer[Item],
        *,
        concurrency: int = 8,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        super().__init__(config_name=f"map_reduce-{next(_instance_counter)}")
        self.chunker = chunker
        self.extractor = extractor
        self.reducer = reducer
        self.concurrency = concurrency

    @DBOS.workflow()
    async def run(self, doc: Doc, *, tenant_id: UUID) -> list[Item]:
        chunks = await self._chunk(doc)
        if not chunks:
            return []
        sem = asyncio.Semaphore(self.concurrency)

        async def _bounded(chunk: Chunk) -> Item | None:
            async with sem:
                return await self._extract_one(chunk)

        results = await asyncio.gather(*[_bounded(c) for c in chunks])
        non_null = [r for r in results if r is not None]
        return await self._reduce(non_null)

    @DBOS.step()
    async def _chunk(self, doc: Doc) -> list[Chunk]:
        return self.chunker.chunk(doc)

    @DBOS.step()
    async def _extract_one(self, chunk: Chunk) -> Item | None:
        return await _ensure_async(self.extractor, chunk)  # type: ignore[no-any-return]

    @DBOS.step()
    async def _reduce(self, items: list[Item]) -> list[Item]:
        return await self.reducer.reduce(items)

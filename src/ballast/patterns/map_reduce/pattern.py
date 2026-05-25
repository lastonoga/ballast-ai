"""``map_reduce_llm`` — generic parallel map + reduce primitive.

Reused by:
  - ``MapReduceStrategy`` in memory recall (large result sets)
  - Future long-document RAG (per-chunk extract + reduce)
  - Custom apps with embarrassingly-parallel LLM steps
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

InT     = TypeVar("InT")
MapT    = TypeVar("MapT")
ReduceT = TypeVar("ReduceT")


async def map_reduce_llm(
    items: list[InT],
    *,
    map_step:    Callable[[InT], Awaitable[MapT]],
    reduce_step: Callable[[list[MapT]], Awaitable[ReduceT]],
    map_concurrency:    int = 8,
    collapse_threshold: int | None = None,
) -> ReduceT:
    """Map each item in parallel (bounded by ``map_concurrency``),
    then reduce. If ``collapse_threshold`` is set and mapped output
    exceeds it, perform recursive batch-reduce before the final reduce.

    Empty ``items`` short-circuits to ``reduce_step([])`` (single call).
    """
    if not items:
        return await reduce_step([])

    sem = asyncio.Semaphore(map_concurrency)

    async def _bounded(x: InT) -> MapT:
        async with sem:
            return await map_step(x)

    mapped: list[MapT] = await asyncio.gather(*(_bounded(x) for x in items))

    if collapse_threshold is not None and len(mapped) > collapse_threshold:
        batches: list[list[MapT]] = [
            mapped[i:i + collapse_threshold]
            for i in range(0, len(mapped), collapse_threshold)
        ]
        partial: list[MapT] = []
        for batch in batches:
            partial.append(await reduce_step(batch))   # type: ignore[arg-type]
        mapped = partial

    return await reduce_step(mapped)


__all__ = ["map_reduce_llm"]

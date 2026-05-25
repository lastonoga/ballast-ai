"""``map_reduce_llm`` — parallel per-item map + reduce. Generic primitive."""
from __future__ import annotations

import pytest

from ballast.patterns.map_reduce import map_reduce_llm


@pytest.mark.asyncio
async def test_simple_map_reduce(fresh_dbos_executor: None) -> None:
    """Map doubles each int; reduce sums."""
    async def double(x: int) -> int: return x * 2

    async def sum_all(xs: list[int]) -> int: return sum(xs)

    out = await map_reduce_llm(
        items=[1, 2, 3, 4, 5],
        map_step=double,
        reduce_step=sum_all,
    )
    assert out == 30   # (1+2+3+4+5)*2


@pytest.mark.asyncio
async def test_empty_items_short_circuits(
    fresh_dbos_executor: None,
) -> None:
    """Zero items → reduce called once with []."""
    async def map_fn(_: int) -> int: raise AssertionError("should not run")

    async def reduce_fn(xs: list[int]) -> str: return f"got {len(xs)} items"

    out = await map_reduce_llm(
        items=[], map_step=map_fn, reduce_step=reduce_fn,
    )
    assert out == "got 0 items"


@pytest.mark.asyncio
async def test_collapse_threshold_triggers_recursive_reduce(
    fresh_dbos_executor: None,
) -> None:
    """When mapped output exceeds collapse_threshold, batches are
    reduced before the final reduce."""
    reduce_calls: list[int] = []

    async def passthrough(x: int) -> int: return x

    async def sum_reduce(xs: list[int]) -> int:
        reduce_calls.append(len(xs))
        return sum(xs)

    out = await map_reduce_llm(
        items=list(range(10)),
        map_step=passthrough,
        reduce_step=sum_reduce,
        collapse_threshold=3,
    )
    assert out == sum(range(10))   # final value
    # First batches of 3 reduced, then their results re-reduced.
    assert len(reduce_calls) > 1

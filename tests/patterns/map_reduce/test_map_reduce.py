"""``MapReduce`` — durable class-based parallel map+reduce."""
from __future__ import annotations

import pytest

from ballast.patterns.map_reduce import MapReduce


@pytest.mark.asyncio
async def test_callable_api_basic(fresh_dbos_executor: None) -> None:
    """Low-level: pass async callables for map+reduce."""
    async def double(x: int) -> int: return x * 2
    async def sum_all(xs: list[int]) -> int: return sum(xs)

    mr = MapReduce[int, int, int](map_step=double, reduce_step=sum_all)
    out = await mr.run([1, 2, 3, 4, 5])
    assert out == 30


@pytest.mark.asyncio
async def test_empty_items_short_circuits(fresh_dbos_executor: None) -> None:
    async def map_fn(_: int) -> int: raise AssertionError("should not run")
    async def reduce_fn(xs: list[int]) -> str: return f"got {len(xs)}"

    mr = MapReduce[int, int, str](map_step=map_fn, reduce_step=reduce_fn)
    out = await mr.run([])
    assert out == "got 0"


@pytest.mark.asyncio
async def test_collapse_threshold_triggers_recursive_reduce(
    fresh_dbos_executor: None,
) -> None:
    reduce_calls: list[int] = []

    async def passthrough(x: int) -> int: return x
    async def sum_reduce(xs: list[int]) -> int:
        reduce_calls.append(len(xs))
        return sum(xs)

    mr = MapReduce[int, int, int](
        map_step=passthrough, reduce_step=sum_reduce, collapse_threshold=3,
    )
    out = await mr.run(list(range(10)))
    assert out == sum(range(10))
    assert len(reduce_calls) > 1


def test_xor_validation_map_phase() -> None:
    async def step(x): return x
    with pytest.raises(ValueError, match="map_step"):
        MapReduce(map_step=step, map_agent=object(),  # type: ignore[arg-type]
                  reduce_step=step)


def test_xor_validation_reduce_phase() -> None:
    async def step(x): return x
    with pytest.raises(ValueError, match="reduce_step"):
        MapReduce(map_step=step,
                  reduce_step=step, reduce_agent=object())  # type: ignore[arg-type]


def test_xor_validation_neither_provided() -> None:
    with pytest.raises(ValueError, match="map_step"):
        MapReduce()  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_retries_on_transient_failure(fresh_dbos_executor: None) -> None:
    """retries=2 → up to 3 total attempts; succeed on 3rd."""
    attempts = {"count": 0}

    async def flaky(x: int) -> int:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")
        return x * 10

    async def sum_all(xs: list[int]) -> int: return sum(xs)

    mr = MapReduce[int, int, int](
        map_step=flaky, reduce_step=sum_all,
        retries=2, retry_backoff_seconds=0.0,
    )
    out = await mr.run([5])
    assert out == 50
    assert attempts["count"] == 3

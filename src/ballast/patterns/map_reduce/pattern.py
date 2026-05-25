"""``MapReduce`` — durable parallel map + reduce pattern.

Two API levels:

**Low-level** (full control):
    mr = MapReduce(map_step=async_fn, reduce_step=async_fn)
    out = await mr.run(items)

**High-level** (use agents — framework wires them):
    mr = MapReduce(map_agent=SummarizerAgent(), reduce_agent=DigestAgent())
    out = await mr.run(items)

For each phase: either ``_step`` (raw callable) OR ``_agent`` is set,
not both. Validated at construction.

Used internally by ``MapReduceStrategy`` for episodic-memory recall
digesting; stands alone for long-document RAG, parallel scraping
pipelines, etc.

DBOS-backed: ``run`` is a workflow; per-item map calls + the final
reduce are steps with configurable retries. Crashes mid-flow recover:
already-completed map results are replayed from DBOS state, only the
unfinished tail re-runs.
"""
from __future__ import annotations

import asyncio
import itertools
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from dbos import DBOSConfiguredInstance

from ballast.durable import Durable

if TYPE_CHECKING:
    from ballast.runtime.agents import BallastAgent


InT     = TypeVar("InT")
MapT    = TypeVar("MapT")
ReduceT = TypeVar("ReduceT")

MapStep    = Callable[[Any], Awaitable[Any]]
ReduceStep = Callable[[list[Any]], Awaitable[Any]]

_instance_counter = itertools.count()


@Durable.dbos_class()
class MapReduce(DBOSConfiguredInstance, Generic[InT, MapT, ReduceT]):
    """Durable parallel map+reduce. Stateless across runs; one instance
    is reusable for many ``run(items)`` calls."""

    def __init__(
        self,
        *,
        map_step:    MapStep | None = None,
        map_agent:   "BallastAgent | None" = None,
        reduce_step: ReduceStep | None = None,
        reduce_agent: "BallastAgent | None" = None,
        map_concurrency:    int = 8,
        collapse_threshold: int | None = None,
        retries:            int = 0,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        super().__init__(
            config_name=f"{type(self).__qualname__}-{next(_instance_counter)}",
        )

        # XOR validation: exactly one of step/agent per phase.
        if (map_step is None) == (map_agent is None):
            raise ValueError(
                "MapReduce: provide exactly one of `map_step` or `map_agent`",
            )
        if (reduce_step is None) == (reduce_agent is None):
            raise ValueError(
                "MapReduce: provide exactly one of `reduce_step` or `reduce_agent`",
            )

        self._map_step = map_step
        self._map_agent = map_agent
        self._reduce_step = reduce_step
        self._reduce_agent = reduce_agent
        self._map_concurrency = map_concurrency
        self._collapse_threshold = collapse_threshold
        self._retries = retries
        self._retry_backoff = retry_backoff_seconds

    @Durable.workflow()
    async def run(self, items: list[Any]) -> Any:
        """Map each item in parallel (bounded by ``map_concurrency``),
        then reduce. If ``collapse_threshold`` is set and mapped output
        exceeds it, perform recursive batch-reduce before the final reduce.

        Empty ``items`` short-circuits to a single ``_reduce([])`` call.
        """
        if not items:
            return await self._reduce([])

        sem = asyncio.Semaphore(self._map_concurrency)

        async def _bounded(x: Any) -> Any:
            async with sem:
                return await self._map_one(x)

        mapped: list[Any] = await asyncio.gather(*(_bounded(x) for x in items))

        if self._collapse_threshold is not None and len(mapped) > self._collapse_threshold:
            batches = [
                mapped[i : i + self._collapse_threshold]
                for i in range(0, len(mapped), self._collapse_threshold)
            ]
            partial: list[Any] = []
            for batch in batches:
                partial.append(await self._reduce(batch))
            mapped = partial

        return await self._reduce(mapped)

    @Durable.step()
    async def _map_one(self, item: Any) -> Any:
        """One map call. Memoised by DBOS. Retried up to ``retries`` times
        on transient failure with exponential backoff."""
        attempt = 0
        while True:
            try:
                if self._map_step is not None:
                    return await self._map_step(item)
                # Agent path: run with `item` as the user prompt; return result.output.
                result = await self._map_agent.run(item)  # type: ignore[union-attr]
                return result.output
            except Exception:
                attempt += 1
                if attempt > self._retries:
                    raise
                await asyncio.sleep(self._retry_backoff * (2 ** (attempt - 1)))

    @Durable.step()
    async def _reduce(self, items: list[Any]) -> Any:
        """One reduce call. Same memoisation/retry semantics."""
        attempt = 0
        while True:
            try:
                if self._reduce_step is not None:
                    return await self._reduce_step(items)
                prompt = "\n".join(str(x) for x in items)
                result = await self._reduce_agent.run(prompt)  # type: ignore[union-attr]
                return result.output
            except Exception:
                attempt += 1
                if attempt > self._retries:
                    raise
                await asyncio.sleep(self._retry_backoff * (2 ** (attempt - 1)))


__all__ = ["MapReduce"]

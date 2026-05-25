"""``as_workflow`` — adapt CoALAUnit to @Durable.workflow with per-phase steps."""
from __future__ import annotations

import itertools
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from dbos import DBOSConfiguredInstance

from ballast.coala._protocol import CoALAUnit
from ballast.durable import Durable

InT      = TypeVar("InT")
ObsT     = TypeVar("ObsT")
ContextT = TypeVar("ContextT")
OutT     = TypeVar("OutT")

_instance_counter = itertools.count()


@Durable.dbos_class()
class _CoALAWorkflow(
    DBOSConfiguredInstance, Generic[InT, ObsT, ContextT, OutT],
):
    """DBOS-configured wrapper turning a CoALAUnit into a durable
    workflow with per-phase ``@Durable.step`` memoisation.

    The unit lives on ``self._unit`` (instance state on a
    ``DBOSConfiguredInstance``) — never pickled per step call. Each
    phase becomes a real step: on workflow replay, already-completed
    phases skip; only the unfinished tail re-runs.
    """

    def __init__(self, unit: CoALAUnit[InT, ObsT, ContextT, OutT]) -> None:
        super().__init__(
            config_name=(
                f"{type(self).__qualname__}-{type(unit).__name__}-"
                f"{next(_instance_counter)}"
            ),
        )
        self._unit = unit

    @Durable.workflow()
    async def run(self, input: InT) -> OutT:
        observation = await self._observe(input)
        context     = await self._retrieve(observation)
        output      = await self._act(observation, context)
        await self._learn(observation, context, output)
        return output

    @Durable.step()
    async def _observe(self, input: InT) -> ObsT:
        return await self._unit.observe(input)

    @Durable.step()
    async def _retrieve(self, observation: ObsT) -> ContextT:
        return await self._unit.retrieve(observation)

    @Durable.step()
    async def _act(self, observation: ObsT, context: ContextT) -> OutT:
        return await self._unit.act(observation, context)

    @Durable.step()
    async def _learn(
        self, observation: ObsT, context: ContextT, output: OutT,
    ) -> None:
        return await self._unit.learn(observation, context, output)


def as_workflow(
    unit: CoALAUnit[InT, Any, Any, OutT],
) -> Callable[[InT], Awaitable[OutT]]:
    """Wrap a CoALAUnit as a @Durable.workflow runner with per-phase steps.

    Each phase becomes a ``@Durable.step`` — memoised on replay,
    retryable. Crash mid-lifecycle: already-completed phases skip;
    only the unfinished tail re-runs.

    Returns a plain async callable. The unit is stored on a
    ``DBOSConfiguredInstance`` subclass instance, so it is never pickled
    as a step or workflow argument — units may freely close over agents,
    HTTP clients, repositories, etc.
    """
    wrapper = _CoALAWorkflow(unit)
    return wrapper.run


__all__ = ["as_workflow"]

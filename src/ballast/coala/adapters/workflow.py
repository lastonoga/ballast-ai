"""``as_workflow`` — adapt CoALAUnit to @Durable.workflow."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from ballast.coala._protocol import CoALAUnit
from ballast.durable import Durable

InT  = TypeVar("InT")
OutT = TypeVar("OutT")


def as_workflow(
    unit: CoALAUnit[InT, Any, Any, OutT],
) -> Callable[[InT], Awaitable[OutT]]:
    """Wrap a CoALAUnit as a @Durable.workflow runner.

    The unit instance is captured via closure and is NOT passed as a
    workflow or step argument. This avoids DBOS's pickle-serialization
    of step args for the unit itself, which would fail for units that
    close over agents, HTTP clients, asyncio locks, or other
    non-picklable objects.

    Design note (deviation from original plan):
      The plan called for four module-level ``@Durable.step``-decorated
      helpers (``_observe_step``, ``_retrieve_step``, etc.) each
      receiving ``unit`` as a positional arg. DBOS serialises step args
      via ``pickle``, so passing a ``CoALAUnit`` instance as a step arg
      would fail for any unit that closes over an unpicklable object
      (e.g., a ``pydantic_ai.Agent``, an ``httpx.AsyncClient``, or an
      asyncio ``Lock``). Per the task's fallback guidance, the four
      phases are instead called directly inside the single
      ``@Durable.workflow`` body. The workflow itself only receives the
      plain ``input`` arg (pickle-safe primitives / Pydantic models as
      the caller provides). Per-phase memoisation is not available in
      this variant; the entire lifecycle re-runs on replay. If
      per-phase step memoisation becomes a requirement, the unit must
      expose a ``@Durable.step``-decorated class method or be wrapped
      in a ``DBOSConfiguredInstance`` subclass.

    Returns a plain async callable. The returned function is registered
    with DBOS at call time — create one runner per unit instance.
    """
    unit_type_name = type(unit).__name__

    @Durable.workflow()
    async def runner(input: InT) -> OutT:
        observation = await unit.observe(input)
        context     = await unit.retrieve(observation)
        output      = await unit.act(observation, context)
        await unit.learn(observation, context, output)
        return output

    runner.__name__ = f"coala_workflow_{unit_type_name}"
    runner.__doc__  = (type(unit).__doc__ or "").strip() or None
    return runner


__all__ = ["as_workflow"]

"""``as_capability`` — adapt CoALAUnit to BallastCapability."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ballast.capabilities.base import BallastCapability
from ballast.coala._protocol import CoALAUnit

_log = logging.getLogger("ballast.coala.capability")

_OBSERVATION_KEY = "_coala_observation"
_CONTEXT_KEY = "_coala_context"

GateFn = Callable[[Any], "bool | Awaitable[bool]"]


def as_capability(
    unit: CoALAUnit,
    *,
    gate: GateFn | None = None,
) -> BallastCapability:
    """Wrap a CoALAUnit as a pydantic-ai capability for an agent.

    Phase → hook mapping:

      observe + retrieve → ``before_model_request``. The full
        ``ModelRequestContext`` is passed to ``observe`` so the unit can
        inspect messages, settings, or parameters as needed. Observation +
        retrieved context are stashed on ``ctx.deps`` for later ``learn``
        access.

      act → the agent's own ``.iter()`` loop. NOT framework-mediated.

      learn → ``after_run``, gated by optional ``gate`` callback.
        Failures inside ``learn`` are swallowed + logged so memory-write
        bugs never block user-facing replies.
    """

    class _CoALACapability(BallastCapability):
        name = f"coala_{type(unit).__name__}"

        async def before_model_request(
            self, ctx: Any, request_context: Any
        ) -> Any:
            observation = await unit.observe(request_context)
            context = await unit.retrieve(observation)
            _stash(ctx, _OBSERVATION_KEY, observation)
            _stash(ctx, _CONTEXT_KEY, context)
            return request_context

        async def after_run(self, ctx: Any, *, result: Any) -> Any:
            try:
                if gate is not None:
                    g = gate(result)
                    passed = await g if asyncio.iscoroutine(g) else g
                    if not passed:
                        return result
                observation = _unstash(ctx, _OBSERVATION_KEY)
                context = _unstash(ctx, _CONTEXT_KEY)
                output = getattr(result, "output", result)
                await unit.learn(observation, context, output)
            except Exception:
                _log.exception("CoALA learn() failed (swallowed)")
            return result

    return _CoALACapability()


def _stash(ctx: Any, key: str, value: Any) -> None:
    deps = getattr(ctx, "deps", None)
    if isinstance(deps, dict):
        deps[key] = value
    else:
        setattr(ctx, key, value)


def _unstash(ctx: Any, key: str) -> Any:
    deps = getattr(ctx, "deps", None)
    if isinstance(deps, dict):
        return deps.get(key)
    return getattr(ctx, key, None)


__all__ = ["as_capability"]

"""``with_drift_monitor`` — workflow surface for Goal Drift Detection.

Decorator: wraps an async function (typically a ``@Durable.workflow``
body) and runs a background tick loop that polls the drift engine's
strategy on a configurable interval.

Known limitation: in messageless contexts (workflows without an agent
loop), ``DriftContext.messages == []`` and built-in ``TraceWindow`` impls
return ``[]`` → ``DriftEngine.maybe_check`` short-circuits to ``None``.
Apps that want workflow drift detection must supply a custom ``TraceWindow``
(e.g., one that reads state from a database via ``ctx.workflow_input``).
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, TypeVar

from ballast.drift._core import DriftEngine
from ballast.drift._handlers import GoalDriftError
from ballast.drift._protocols import DriftCheckSignal, DriftContext

_log = logging.getLogger("ballast.drift.workflow")

T = TypeVar("T")


def with_drift_monitor(
    *,
    engine: DriftEngine,
    tick_seconds: float = 1.0,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Wrap an async function so a drift-monitor task runs alongside it.

    The monitor task is cancelled in ``finally`` regardless of how the
    body returns (success, exception, cancellation).
    """
    if tick_seconds <= 0:
        raise ValueError("tick_seconds must be > 0")

    def deco(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            fn_input = args[0] if args else next(iter(kwargs.values()), None)
            monitor = asyncio.create_task(
                _monitor_loop(engine, fn_input, tick_seconds),
                name=f"drift-monitor:{fn.__name__}",
            )
            try:
                return await fn(*args, **kwargs)
            finally:
                monitor.cancel()
                with suppress(asyncio.CancelledError):
                    await monitor
        return wrapper
    return deco


async def _monitor_loop(
    engine: DriftEngine, fn_input: Any, tick_seconds: float,
) -> None:
    """Periodic polling of the drift engine."""
    start = time.monotonic()
    tick = 0
    while True:
        try:
            await asyncio.sleep(tick_seconds)
        except asyncio.CancelledError:
            raise
        tick += 1
        signal = DriftCheckSignal(
            step_index=tick,
            tool_calls=0,
            tokens_used=0,
            seconds_elapsed=time.monotonic() - start,
        )
        ctx = DriftContext(
            messages=[],
            run_ctx=None,
            workflow_input=fn_input,
            metadata={},
        )
        try:
            await engine.maybe_check(signal, ctx)  # type: ignore[arg-type]
        except GoalDriftError:
            # Workflow-side hard-stop policy: log and continue ticking;
            # the wrapper's body owns the workflow lifecycle, the monitor
            # cannot itself abort it. Raising from the background task
            # would only crash the monitor coroutine, which is useless.
            _log.warning("GoalDriftError fired from workflow monitor; body unaffected")
        except Exception:
            _log.exception("drift monitor tick failed (swallowed)")

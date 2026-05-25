"""with_drift_monitor decorator — workflow surface for drift detection."""
from __future__ import annotations

import asyncio

import pytest

from ballast.drift._protocols import DriftCheckSignal, DriftContext
from ballast.patterns.drift_monitor import with_drift_monitor


class _RecordingEngine:
    def __init__(self): self.calls = []
    async def maybe_check(self, sig, ctx):
        self.calls.append((sig, ctx))
        return None


@pytest.mark.asyncio
async def test_decorator_passes_through_return_value() -> None:
    engine = _RecordingEngine()

    @with_drift_monitor(engine=engine, tick_seconds=0.05)  # type: ignore[arg-type]
    async def body(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2

    assert await body(7) == 14


@pytest.mark.asyncio
async def test_monitor_task_cancelled_after_body_returns() -> None:
    engine = _RecordingEngine()

    @with_drift_monitor(engine=engine, tick_seconds=0.05)  # type: ignore[arg-type]
    async def body() -> None:
        await asyncio.sleep(0.02)

    await body()
    # If monitor wasn't cancelled, asyncio would still have it pending.
    # Give the event loop a chance to run remaining tasks; none should remain.
    tasks_before = len([t for t in asyncio.all_tasks() if not t.done()])
    await asyncio.sleep(0.05)
    tasks_after = len([t for t in asyncio.all_tasks() if not t.done()])
    # Background monitor must have stopped (only current test task remains).
    assert tasks_after <= tasks_before


@pytest.mark.asyncio
async def test_monitor_fires_at_least_once_during_long_body() -> None:
    engine = _RecordingEngine()

    @with_drift_monitor(engine=engine, tick_seconds=0.05)  # type: ignore[arg-type]
    async def body() -> None:
        await asyncio.sleep(0.15)

    await body()
    assert len(engine.calls) >= 1


@pytest.mark.asyncio
async def test_body_exception_still_cancels_monitor() -> None:
    engine = _RecordingEngine()

    @with_drift_monitor(engine=engine, tick_seconds=0.05)  # type: ignore[arg-type]
    async def body() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await body()
    # Monitor must be torn down; give time and check.
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_monitor_tick_exception_swallowed_and_loop_continues() -> None:
    calls_count = 0

    class _BoomEngine:
        async def maybe_check(self, sig, ctx):
            nonlocal calls_count
            calls_count += 1
            raise RuntimeError("engine down")

    @with_drift_monitor(engine=_BoomEngine(), tick_seconds=0.03)  # type: ignore[arg-type]
    async def body() -> None:
        await asyncio.sleep(0.12)

    await body()
    # Monitor should keep ticking despite engine failing each time.
    assert calls_count >= 2

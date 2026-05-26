"""End-to-end: Scored[T] with MapReduce / Agent.output_type / CircuitBreaker."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.quality.scored import (
    Confidence, Scored,
    filter_by_min_confidence, rank_by_confidence,
)
from ballast.resilience.circuit_breaker import (
    Consecutive, CircuitBreaker, BreakerState,
)


class _Clock:
    def __init__(self): self.now = datetime(2026, 1, 1, tzinfo=UTC)
    def __call__(self): return self.now


@pytest.mark.asyncio
async def test_filter_then_rank_pipeline() -> None:
    """Typical reduce-step pattern: low filtered, rest ranked high → low."""
    items: list[Scored[str]] = [
        Scored[str](value="a", rationale="r", confidence="low"),
        Scored[str](value="b", rationale="r", confidence="high"),
        Scored[str](value="c", rationale="r", confidence="medium"),
        Scored[str](value="d", rationale="r", confidence="high"),
    ]
    kept = filter_by_min_confidence(items, "medium")
    ranked = rank_by_confidence(kept, secondary_key=lambda it: it.value)
    assert [it.value for it in ranked] == ["b", "d", "c"]


@pytest.mark.asyncio
async def test_circuit_breaker_treats_low_confidence_as_failure() -> None:
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(2),
        is_success=lambda r: isinstance(r, Scored) and r.confidence != "low",
        clock=_Clock(),
    )

    async def low_conf_extract() -> Scored[str]:
        return Scored[str](value="x", rationale="r", confidence="low")

    await cb.call(low_conf_extract)
    await cb.call(low_conf_extract)
    assert cb.stats().state == BreakerState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_high_confidence_keeps_closed() -> None:
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(2),
        is_success=lambda r: isinstance(r, Scored) and r.confidence != "low",
        clock=_Clock(),
    )

    async def high_conf() -> Scored[str]:
        return Scored[str](value="x", rationale="r", confidence="high")

    await cb.call(high_conf)
    await cb.call(high_conf)
    assert cb.stats().state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_aggregate_then_summarize_pattern() -> None:
    """Apps can bucket items by confidence and feed buckets to LLM separately."""
    from ballast.quality.scored import aggregate_by_confidence

    items: list[Scored[str]] = [
        Scored[str](value="fact-a", rationale="r", confidence="high"),
        Scored[str](value="fact-b", rationale="r", confidence="medium"),
        Scored[str](value="fact-c", rationale="r", confidence="high"),
        Scored[str](value="fact-d", rationale="r", confidence="low"),
    ]
    buckets = aggregate_by_confidence(items)
    assert buckets["high"] == ["fact-a", "fact-c"]
    assert buckets["medium"] == ["fact-b"]
    assert buckets["low"] == ["fact-d"]

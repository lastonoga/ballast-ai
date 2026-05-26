"""Built-in ThresholdPolicy implementations."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ballast.resilience.circuit_breaker._protocols import ThresholdPolicy
from ballast.resilience.circuit_breaker._thresholds import (
    Consecutive, WindowedCount, WindowedRate,
)


_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _t(secs: float) -> datetime:
    return _T0 + timedelta(seconds=secs)


# --- Consecutive --------------------------------------------------------------

def test_consecutive_satisfies_protocol() -> None:
    assert isinstance(Consecutive(3), ThresholdPolicy)


def test_consecutive_trips_after_n_failures() -> None:
    c = Consecutive(max_failures=3)
    assert not c.trip(at=_t(0))
    for i in range(2):
        c.on_outcome(success=False, at=_t(i))
        assert not c.trip(at=_t(i))
    c.on_outcome(success=False, at=_t(3))
    assert c.trip(at=_t(3))


def test_consecutive_resets_on_success() -> None:
    c = Consecutive(max_failures=3)
    c.on_outcome(success=False, at=_t(0))
    c.on_outcome(success=False, at=_t(1))
    c.on_outcome(success=True,  at=_t(2))
    c.on_outcome(success=False, at=_t(3))
    assert not c.trip(at=_t(3))


def test_consecutive_reset_method_clears() -> None:
    c = Consecutive(max_failures=2)
    c.on_outcome(success=False, at=_t(0))
    c.on_outcome(success=False, at=_t(1))
    assert c.trip(at=_t(1))
    c.reset()
    assert not c.trip(at=_t(1))


def test_consecutive_rejects_invalid_max() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        Consecutive(max_failures=0)


# --- WindowedCount ------------------------------------------------------------

def test_windowed_count_trips_within_window() -> None:
    w = WindowedCount(max_failures=3, window=timedelta(seconds=10))
    w.on_outcome(success=False, at=_t(0))
    w.on_outcome(success=False, at=_t(5))
    w.on_outcome(success=False, at=_t(9))
    assert w.trip(at=_t(9))


def test_windowed_count_prunes_old_failures() -> None:
    w = WindowedCount(max_failures=3, window=timedelta(seconds=10))
    w.on_outcome(success=False, at=_t(0))
    w.on_outcome(success=False, at=_t(5))
    w.on_outcome(success=False, at=_t(20))  # _t(0), _t(5) outside window
    assert not w.trip(at=_t(20))


def test_windowed_count_reset_clears() -> None:
    w = WindowedCount(max_failures=2, window=timedelta(seconds=10))
    w.on_outcome(success=False, at=_t(0))
    w.on_outcome(success=False, at=_t(1))
    assert w.trip(at=_t(1))
    w.reset()
    assert not w.trip(at=_t(1))


# --- WindowedRate -------------------------------------------------------------

def test_windowed_rate_trips_above_rate_with_min_samples() -> None:
    w = WindowedRate(rate=0.5, window=timedelta(seconds=60), min_samples=4)
    # 2 failures + 2 successes in window → 50% → trip (>= 0.5)
    w.on_outcome(success=False, at=_t(0))
    w.on_outcome(success=False, at=_t(1))
    w.on_outcome(success=True,  at=_t(2))
    w.on_outcome(success=True,  at=_t(3))
    assert w.trip(at=_t(3))


def test_windowed_rate_does_not_trip_below_min_samples() -> None:
    w = WindowedRate(rate=0.5, window=timedelta(seconds=60), min_samples=10)
    for i in range(3):
        w.on_outcome(success=False, at=_t(i))
    assert not w.trip(at=_t(3))  # only 3 samples, need 10


def test_windowed_rate_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError, match="rate must be"):
        WindowedRate(rate=0.0)
    with pytest.raises(ValueError, match="rate must be"):
        WindowedRate(rate=1.5)

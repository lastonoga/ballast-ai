"""BreakerState + BreakerStats + CircuitOpenError."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ballast.errors import BallastError
from ballast.resilience.circuit_breaker._state import (
    BreakerState, BreakerStats, CircuitOpenError,
)


def test_breaker_state_string_enum() -> None:
    assert BreakerState.CLOSED == "closed"
    assert BreakerState.OPEN == "open"
    assert BreakerState.HALF_OPEN == "half_open"


def test_breaker_stats_required_fields() -> None:
    stats = BreakerStats(
        scope="tool:search", state=BreakerState.OPEN,
        consecutive_failures=5, total_failures=10, total_successes=2,
        opened_at=datetime(2026, 5, 26, tzinfo=UTC),
        will_attempt_recovery_at=datetime(2026, 5, 26, 0, 0, 30, tzinfo=UTC),
        probe_attempts=0, probe_max=1,
    )
    assert stats.scope == "tool:search"
    assert stats.state == BreakerState.OPEN
    assert stats.consecutive_failures == 5
    assert stats.probe_max == 1


def test_breaker_stats_model_dump_serializable() -> None:
    stats = BreakerStats(
        scope="x", state=BreakerState.CLOSED,
        consecutive_failures=0, total_failures=0, total_successes=0,
        opened_at=None, will_attempt_recovery_at=None,
        probe_attempts=0, probe_max=1,
    )
    dumped = stats.model_dump(mode="json")
    assert dumped["scope"] == "x"
    assert dumped["state"] == "closed"


def test_circuit_open_error_subclass_of_ballast_error() -> None:
    assert issubclass(CircuitOpenError, BallastError)
    assert CircuitOpenError.code == "BALLAST_CIRCUIT_OPEN"
    assert CircuitOpenError.status_code == 503


def test_circuit_open_error_carries_stats() -> None:
    stats = BreakerStats(
        scope="api", state=BreakerState.OPEN,
        consecutive_failures=5, total_failures=5, total_successes=0,
        opened_at=datetime(2026, 5, 26, tzinfo=UTC),
        will_attempt_recovery_at=datetime(2026, 5, 26, 0, 0, 30, tzinfo=UTC),
        probe_attempts=0, probe_max=1,
    )
    exc = CircuitOpenError(stats)
    assert exc.stats is stats
    assert "api" in str(exc)

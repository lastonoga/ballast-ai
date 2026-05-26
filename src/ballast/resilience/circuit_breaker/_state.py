"""``BreakerState`` enum + ``BreakerStats`` pydantic snapshot + ``CircuitOpenError``.

State enum is a string-valued ``Enum`` for clean JSON serialization.
Stats is a pydantic BaseModel for OTel attribute fitness + dashboard
fitness. Error carries the snapshot for downstream observability.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from ballast.errors import BallastError


class BreakerState(str, Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


class BreakerStats(BaseModel):
    """Snapshot for observability / logfire / dashboards."""

    scope:                    str
    state:                    BreakerState
    consecutive_failures:     int
    total_failures:           int
    total_successes:          int
    opened_at:                datetime | None
    will_attempt_recovery_at: datetime | None
    probe_attempts:           int
    probe_max:                int


class CircuitOpenError(BallastError):  # noqa: N818
    """Raised by ``RaiseError`` fallback when breaker rejects an invocation."""

    code = "BALLAST_CIRCUIT_OPEN"
    status_code = 503

    def __init__(self, stats: BreakerStats) -> None:
        self.stats = stats
        super().__init__(
            f"Circuit breaker open for scope {stats.scope!r}",
            hint=(
                "Wait for the recovery_after window, or supply a non-Raise "
                "fallback policy when constructing the CircuitBreaker."
            ),
            context={"breaker_stats": stats.model_dump(mode="json")},
        )


__all__ = ["BreakerState", "BreakerStats", "CircuitOpenError"]

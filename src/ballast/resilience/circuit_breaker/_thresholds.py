"""Built-in ``ThresholdPolicy`` implementations.

Apps choose when the breaker opens:

- ``Consecutive(N)`` — trip after N consecutive failures (any success resets).
- ``WindowedCount(N, window)`` — trip if >= N failures in the trailing window.
- ``WindowedRate(rate, window, min_samples)`` — trip if failure rate is
  high in the window, gated by a minimum sample count.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta


class Consecutive:
    """Trip after N consecutive failures. Any success resets."""

    def __init__(self, max_failures: int = 5) -> None:
        if max_failures < 1:
            raise ValueError("max_failures must be >= 1")
        self._max = max_failures
        self._count = 0

    def on_outcome(self, *, success: bool, at: datetime) -> None:
        self._count = 0 if success else self._count + 1

    def trip(self, *, at: datetime) -> bool:
        return self._count >= self._max

    def reset(self) -> None:
        self._count = 0


class WindowedCount:
    """Trip if >= ``max_failures`` failures in the trailing ``window``."""

    def __init__(
        self, max_failures: int = 5,
        window: timedelta = timedelta(seconds=60),
    ) -> None:
        if max_failures < 1:
            raise ValueError("max_failures must be >= 1")
        if window.total_seconds() <= 0:
            raise ValueError("window must be > 0")
        self._max = max_failures
        self._window = window
        self._failures: deque[datetime] = deque()

    def on_outcome(self, *, success: bool, at: datetime) -> None:
        if not success:
            self._failures.append(at)
        self._prune(at)

    def trip(self, *, at: datetime) -> bool:
        self._prune(at)
        return len(self._failures) >= self._max

    def reset(self) -> None:
        self._failures.clear()

    def _prune(self, at: datetime) -> None:
        cutoff = at - self._window
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()


class WindowedRate:
    """Trip if failure_count / total_count >= rate over ``window``,
    provided total_count >= ``min_samples``."""

    def __init__(
        self, rate: float = 0.5,
        window: timedelta = timedelta(seconds=60),
        min_samples: int = 10,
    ) -> None:
        if not 0.0 < rate <= 1.0:
            raise ValueError("rate must be in (0, 1]")
        if window.total_seconds() <= 0:
            raise ValueError("window must be > 0")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        self._rate = rate
        self._window = window
        self._min = min_samples
        self._outcomes: deque[tuple[datetime, bool]] = deque()

    def on_outcome(self, *, success: bool, at: datetime) -> None:
        self._outcomes.append((at, success))
        self._prune(at)

    def trip(self, *, at: datetime) -> bool:
        self._prune(at)
        if len(self._outcomes) < self._min:
            return False
        failures = sum(1 for _, ok in self._outcomes if not ok)
        return (failures / len(self._outcomes)) >= self._rate

    def reset(self) -> None:
        self._outcomes.clear()

    def _prune(self, at: datetime) -> None:
        cutoff = at - self._window
        while self._outcomes and self._outcomes[0][0] < cutoff:
            self._outcomes.popleft()


__all__ = ["Consecutive", "WindowedCount", "WindowedRate"]

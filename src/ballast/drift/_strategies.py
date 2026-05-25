"""Built-in ``DriftCheckStrategy`` implementations.

Apps choose when the LLM judge fires:

- ``AfterEveryStep`` — every agent step (precise, expensive).
- ``EveryNToolCalls(n)`` — every N tool calls.
- ``EveryNSteps(n)`` — every N model responses.
- ``Periodic(seconds)`` — every N seconds of wall time.
- ``OnBudgetThreshold(fraction, budget_fn)`` — once when consumed / max
  crosses the fraction (e.g., 50% of token budget burnt).
- ``Compose(*strategies)`` — OR-combination; fires if any component fires.
"""
from __future__ import annotations

from collections.abc import Callable

from ballast.drift._protocols import DriftCheckSignal, DriftCheckStrategy


class AfterEveryStep:
    """Fire on every agent step."""

    def should_check(self, signal: DriftCheckSignal) -> bool:
        return True


class EveryNToolCalls:
    """Fire when tool-call count has advanced by N since last fire."""

    def __init__(self, n: int = 5) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self._n = n
        self._last = 0

    def should_check(self, signal: DriftCheckSignal) -> bool:
        if signal.tool_calls >= self._last + self._n:
            self._last = signal.tool_calls
            return True
        return False


class EveryNSteps:
    """Fire when step_index has advanced by N since last fire."""

    def __init__(self, n: int = 3) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self._n = n
        self._last = 0

    def should_check(self, signal: DriftCheckSignal) -> bool:
        if signal.step_index >= self._last + self._n:
            self._last = signal.step_index
            return True
        return False


class Periodic:
    """Fire once each elapsed window of ``seconds``."""

    def __init__(self, seconds: float = 30.0) -> None:
        if seconds <= 0:
            raise ValueError("seconds must be > 0")
        self._seconds = seconds
        self._last = 0.0

    def should_check(self, signal: DriftCheckSignal) -> bool:
        if signal.seconds_elapsed >= self._last + self._seconds:
            self._last = signal.seconds_elapsed
            return True
        return False


class OnBudgetThreshold:
    """Fire ONCE when ``consumed / max`` crosses ``fraction``.

    Reads budget state via a caller-supplied ``budget_fn`` returning
    ``(consumed, max_total)``. Stays quiet once it has fired until the
    consumed value drops back below the threshold (which never happens
    in practice — the fire is effectively one-shot).
    """

    def __init__(
        self, *,
        fraction: float = 0.5,
        budget_fn: Callable[[], tuple[int, int]],
    ) -> None:
        if not 0.0 < fraction < 1.0:
            raise ValueError("fraction must be in (0, 1)")
        self._fraction = fraction
        self._budget_fn = budget_fn
        self._fired = False

    def should_check(self, signal: DriftCheckSignal) -> bool:
        consumed, total = self._budget_fn()
        if total <= 0:
            return False
        crossed = (consumed / total) >= self._fraction
        if crossed and not self._fired:
            self._fired = True
            return True
        if not crossed:
            # Allow re-fire if consumed somehow drops back (defensive).
            self._fired = False
        return False


class Compose:
    """OR-combination — fires if any wrapped strategy fires this tick."""

    def __init__(self, *strategies: DriftCheckStrategy) -> None:
        if not strategies:
            raise ValueError("Compose requires at least one strategy")
        self._strategies = strategies

    def should_check(self, signal: DriftCheckSignal) -> bool:
        # Short-circuit on first True so all subsequent strategies still
        # see this signal on the NEXT call (no skipped ticks).
        return any(s.should_check(signal) for s in self._strategies)


__all__ = [
    "AfterEveryStep",
    "Compose",
    "EveryNSteps",
    "EveryNToolCalls",
    "OnBudgetThreshold",
    "Periodic",
]

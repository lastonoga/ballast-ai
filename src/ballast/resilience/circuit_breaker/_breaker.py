"""``CircuitBreaker`` core + ``_ScopeBucket`` per-scope state machine."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from ballast.resilience.circuit_breaker._fallbacks import RaiseError
from ballast.resilience.circuit_breaker._protocols import (
    FallbackPolicy, ScopeKey, ThresholdFactory, ThresholdPolicy,
)
from ballast.resilience.circuit_breaker._scope import global_scope
from ballast.resilience.circuit_breaker._state import (
    BreakerState, BreakerStats,
)
from ballast.resilience.circuit_breaker._thresholds import Consecutive

T = TypeVar("T")


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _default_threshold_factory() -> ThresholdPolicy:
    return Consecutive(5)


def _default_is_success(_result: Any) -> bool:
    return True


class CircuitBreaker:
    """Protects async function invocations.

    Apps call ``.call(fn, *args, ctx=..., **kwargs)``. ``ctx`` (mapping)
    is fed to ``scope_key`` to determine which sub-bucket tracks this
    invocation. One CircuitBreaker can multiplex many scopes (e.g., one
    breaker, many tools).
    """

    def __init__(
        self, *,
        threshold_factory: ThresholdFactory               = _default_threshold_factory,
        fallback:          FallbackPolicy | None          = None,
        scope_key:         ScopeKey                        = global_scope,
        recovery_after:    timedelta                       = timedelta(seconds=30),
        probe_max:         int                             = 1,
        is_failure_exc:    tuple[type[BaseException], ...] = (Exception,),
        ignored_exc:       tuple[type[BaseException], ...] = (asyncio.CancelledError,),
        is_success:        Callable[[Any], bool]           = _default_is_success,
        name:              str                             = "circuit_breaker",
        clock:             Callable[[], datetime]          = _default_clock,
    ) -> None:
        if probe_max < 1:
            raise ValueError("probe_max must be >= 1")
        if recovery_after.total_seconds() <= 0:
            raise ValueError("recovery_after must be > 0")
        self._threshold_factory = threshold_factory
        self._fallback: FallbackPolicy = fallback if fallback is not None else RaiseError()
        self._scope_key = scope_key
        self._recovery_after = recovery_after
        self._probe_max = probe_max
        self._is_failure_exc = is_failure_exc
        self._ignored_exc = ignored_exc
        self._is_success = is_success
        self._name = name
        self._clock = clock
        self._scopes: dict[str, _ScopeBucket] = {}

    async def call(
        self, fn: Callable[..., Awaitable[T]],
        *args: Any, ctx: Mapping[str, Any] | None = None, **kwargs: Any,
    ) -> T:
        scope = self._scope_key(ctx or {})
        bucket = self._scopes.setdefault(scope, _ScopeBucket(self, scope))
        return await bucket.call(fn, args, kwargs)

    def stats(self, scope: str = "global") -> BreakerStats:
        """Snapshot for observability."""
        bucket = self._scopes.get(scope)
        if bucket is None:
            return BreakerStats(
                scope=scope, state=BreakerState.CLOSED,
                consecutive_failures=0, total_failures=0, total_successes=0,
                opened_at=None, will_attempt_recovery_at=None,
                probe_attempts=0, probe_max=self._probe_max,
            )
        return bucket.snapshot()

    def reset(self, scope: str | None = None) -> None:
        """Force CLOSED. ``None`` → all scopes."""
        targets = (
            list(self._scopes.values()) if scope is None
            else ([self._scopes[scope]] if scope in self._scopes else [])
        )
        for bucket in targets:
            bucket.force_closed()


class _ScopeBucket:
    """Per-scope state machine + counters + asyncio lock."""

    def __init__(self, owner: CircuitBreaker, scope: str) -> None:
        self._owner = owner
        self._scope = scope
        self._state = BreakerState.CLOSED
        self._threshold: ThresholdPolicy = owner._threshold_factory()
        self._opened_at: datetime | None = None
        self._probe_attempts = 0
        self._consecutive_failures = 0
        self._total_failures = 0
        self._total_successes = 0
        self._lock = asyncio.Lock()

    async def call(self, fn: Callable[..., Awaitable[Any]], args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        async with self._lock:
            now = self._owner._clock()
            self._maybe_transition_to_half_open(now)

            if self._state == BreakerState.OPEN:
                return await self._owner._fallback.on_rejected(
                    self.snapshot(), fn, args, kwargs,
                )

            if (self._state == BreakerState.HALF_OPEN
                    and self._probe_attempts >= self._owner._probe_max):
                return await self._owner._fallback.on_rejected(
                    self.snapshot(), fn, args, kwargs,
                )

            if self._state == BreakerState.HALF_OPEN:
                self._probe_attempts += 1

        # Execute outside the lock — fn may be long-running.
        try:
            result = await fn(*args, **kwargs)
        except BaseException as exc:
            # ignored_exc: re-raise without recording — treated as transparent.
            if isinstance(exc, self._owner._ignored_exc):
                raise
            # is_failure_exc: record as failure + re-raise.
            if isinstance(exc, self._owner._is_failure_exc):
                async with self._lock:
                    self._record(success=False, at=self._owner._clock())
                raise
            # Any other BaseException (not in is_failure_exc): just propagate,
            # not recorded as a circuit-breaker failure.
            raise

        success = self._owner._is_success(result)
        async with self._lock:
            self._record(success=success, at=self._owner._clock())
        return result

    def _record(self, *, success: bool, at: datetime) -> None:
        if success:
            self._consecutive_failures = 0
            self._total_successes += 1
        else:
            self._consecutive_failures += 1
            self._total_failures += 1
        self._threshold.on_outcome(success=success, at=at)

        if self._state == BreakerState.HALF_OPEN:
            if success:
                self._transition_to_closed()
            else:
                self._transition_to_open(at)
            return

        if self._state == BreakerState.CLOSED and self._threshold.trip(at=at):
            self._transition_to_open(at)

    def _maybe_transition_to_half_open(self, now: datetime) -> None:
        if (self._state == BreakerState.OPEN
                and self._opened_at is not None
                and now >= self._opened_at + self._owner._recovery_after):
            self._state = BreakerState.HALF_OPEN
            self._probe_attempts = 0

    def _transition_to_open(self, at: datetime) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = at

    def _transition_to_closed(self) -> None:
        self._state = BreakerState.CLOSED
        self._opened_at = None
        self._probe_attempts = 0
        self._threshold.reset()

    def force_closed(self) -> None:
        self._transition_to_closed()
        self._consecutive_failures = 0

    def snapshot(self) -> BreakerStats:
        will_recover = (
            self._opened_at + self._owner._recovery_after
            if self._opened_at is not None else None
        )
        return BreakerStats(
            scope=self._scope, state=self._state,
            consecutive_failures=self._consecutive_failures,
            total_failures=self._total_failures,
            total_successes=self._total_successes,
            opened_at=self._opened_at,
            will_attempt_recovery_at=will_recover,
            probe_attempts=self._probe_attempts,
            probe_max=self._owner._probe_max,
        )


__all__ = ["CircuitBreaker"]

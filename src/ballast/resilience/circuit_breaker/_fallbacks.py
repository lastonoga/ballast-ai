"""Built-in ``FallbackPolicy`` implementations."""
from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import timedelta
from typing import Any, NoReturn

from ballast.resilience.circuit_breaker._protocols import FallbackPolicy
from ballast.resilience.circuit_breaker._state import BreakerStats, CircuitOpenError

_log = logging.getLogger("ballast.resilience.circuit_breaker")


class RaiseError:
    """Default: raise ``CircuitOpenError`` carrying the stats snapshot."""

    async def on_rejected(
        self, stats: BreakerStats,
        fn: Callable[..., Awaitable[Any]],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> NoReturn:
        raise CircuitOpenError(stats)


class ReturnValue:
    """Return a stored sentinel value when rejected."""

    def __init__(self, value: Any) -> None:
        self._value = value

    async def on_rejected(self, stats, fn, args, kwargs) -> Any:
        return self._value


class CallFallback:
    """Dispatch to an alternative async callable.

    If the fallback's signature accepts a ``stats`` keyword parameter,
    the breaker stats are passed through; otherwise the call is made with
    only the original args + kwargs.
    """

    def __init__(self, fallback_fn: Callable[..., Awaitable[Any]]) -> None:
        self._fn = fallback_fn
        sig = inspect.signature(fallback_fn)
        self._wants_stats = "stats" in sig.parameters

    async def on_rejected(self, stats, fn, args, kwargs) -> Any:
        if self._wants_stats:
            return await self._fn(*args, stats=stats, **kwargs)
        return await self._fn(*args, **kwargs)


class EscalateToHITL:
    """Open HITL request via a ``HITLChannel`` and BLOCK until human verdict."""

    def __init__(
        self, *,
        channel: Any,
        card_factory: Callable[[BreakerStats], Any],
        timeout: timedelta | None = None,
    ) -> None:
        self._channel = channel
        self._card_factory = card_factory
        self._timeout = timeout

    async def on_rejected(self, stats, fn, args, kwargs) -> Any:
        payload = self._card_factory(stats)
        return await self._channel.request(payload, timeout=self._timeout)


class Chain:
    """Try each policy in order; return first non-raising result.

    Logs swallowed exceptions between attempts; raises the LAST exception
    if every policy fails.
    """

    def __init__(self, *policies: FallbackPolicy) -> None:
        if not policies:
            raise ValueError("Chain requires at least one policy")
        self._policies = policies

    async def on_rejected(self, stats, fn, args, kwargs) -> Any:
        last_exc: BaseException | None = None
        for p in self._policies:
            try:
                return await p.on_rejected(stats, fn, args, kwargs)
            except Exception as exc:
                _log.exception(
                    "fallback policy %r failed (trying next)",
                    type(p).__name__,
                )
                last_exc = exc
        assert last_exc is not None
        raise last_exc


__all__ = [
    "CallFallback", "Chain", "EscalateToHITL",
    "RaiseError", "ReturnValue",
]

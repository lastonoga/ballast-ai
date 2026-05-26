"""``ThresholdPolicy`` + ``FallbackPolicy`` Protocols + typing aliases.

Apps wire pluggable policies into ``CircuitBreaker``:

  * ``ThresholdPolicy`` answers "should the breaker open now?" given
    outcome history. Stateful per scope; each scope gets its own
    instance via ``ThresholdFactory``.

  * ``FallbackPolicy`` answers "what to return / raise when an invocation
    is rejected (Open state or denied probe)?". Shared across scopes.

  * ``ScopeKey`` maps an invocation's context dict to a scope string.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from ballast.resilience.circuit_breaker._state import BreakerStats


ScopeKey = Callable[[Mapping[str, Any]], str]
"""Maps invocation context dict → scope key string."""

ThresholdFactory = Callable[[], "ThresholdPolicy"]
"""Per-scope ThresholdPolicy is constructed via factory so each scope
has isolated state."""


@runtime_checkable
class ThresholdPolicy(Protocol):
    """When does the breaker open?"""

    def on_outcome(self, *, success: bool, at: datetime) -> None: ...
    def trip(self, *, at: datetime) -> bool: ...
    def reset(self) -> None: ...


@runtime_checkable
class FallbackPolicy(Protocol):
    """What to do when invocation is rejected (Open state or denied probe)."""

    async def on_rejected(
        self,
        stats: BreakerStats,
        fn:    Callable[..., Awaitable[Any]],
        args:  tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any: ...


__all__ = [
    "FallbackPolicy", "ScopeKey", "ThresholdFactory", "ThresholdPolicy",
]

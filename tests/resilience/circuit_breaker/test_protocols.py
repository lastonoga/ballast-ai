"""ThresholdPolicy + FallbackPolicy Protocols + ScopeKey alias."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.resilience.circuit_breaker._protocols import (
    FallbackPolicy, ScopeKey, ThresholdFactory, ThresholdPolicy,
)
from ballast.resilience.circuit_breaker._state import BreakerStats


def test_threshold_policy_runtime_checkable() -> None:
    class _Stub:
        def on_outcome(self, *, success, at): pass
        def trip(self, *, at): return False
        def reset(self): pass

    assert isinstance(_Stub(), ThresholdPolicy)

    class _Missing:
        def trip(self, *, at): return False

    assert not isinstance(_Missing(), ThresholdPolicy)


def test_fallback_policy_runtime_checkable() -> None:
    class _Stub:
        async def on_rejected(self, stats, fn, args, kwargs):
            return None

    assert isinstance(_Stub(), FallbackPolicy)


def test_scope_key_typing_alias_is_callable() -> None:
    sk: ScopeKey = lambda ctx: "x"
    assert sk({}) == "x"


def test_threshold_factory_typing_alias_is_callable() -> None:
    class _ThrStub:
        def on_outcome(self, *, success, at): pass
        def trip(self, *, at): return False
        def reset(self): pass

    tf: ThresholdFactory = lambda: _ThrStub()
    assert isinstance(tf(), ThresholdPolicy)

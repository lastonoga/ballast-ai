# Circuit Breaker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `CircuitBreaker` resilience primitive in `src/ballast/resilience/circuit_breaker/` with three runtime adapters (agent capability / workflow decorator / PlanAndExecute Step wrapper). Per-scope state buckets, configurable threshold + fallback policies, classic Open/Half-Open/Closed state machine, mockable clock for tests.

**Architecture:** Core class `CircuitBreaker` with `.call(fn, *args, ctx=..., **kwargs)` method. Per-scope `_ScopeBucket` holds its own `ThresholdPolicy` instance (via factory), counters, state, asyncio.Lock. Plug-in Protocols for threshold + fallback + scope-key. Three thin adapters layered on top.

**Tech Stack:** Python 3.11+, pydantic v2 (BreakerStats model), `asyncio.Lock` for per-bucket safety, `datetime` + injectable `clock` callable, existing `BallastError` / `BallastCapability` / pydantic-ai conventions.

**Spec:** `docs/superpowers/specs/2026-05-26-circuit-breaker-design.md`

---

## File Structure (reference for all tasks)

```
src/ballast/resilience/                            # NEW top-level subpackage
  __init__.py                                       # cross-subpackage re-exports (Task 11)
  circuit_breaker/
    __init__.py                                     # public exports (Task 11)
    _state.py                                       # BreakerState + BreakerStats + CircuitOpenError (Task 1)
    _protocols.py                                   # ThresholdPolicy + FallbackPolicy + ScopeKey (Task 2)
    _scope.py                                       # global_scope / per_tool_scope / per_step_scope (Task 3)
    _thresholds.py                                  # Consecutive / WindowedCount / WindowedRate (Task 4)
    _fallbacks.py                                   # RaiseError / ReturnValue / CallFallback / EscalateToHITL / Chain (Task 5)
    _breaker.py                                     # CircuitBreaker + _ScopeBucket (Task 6)
    _adapters/
      __init__.py
      workflow.py                                   # as_workflow_decorator (Task 7)
      step.py                                       # BreakerStep + as_step (Task 8)
      capability.py                                 # as_capability (Task 9)

tests/resilience/
  __init__.py
  circuit_breaker/
    __init__.py
    test_state.py                                   # Task 1
    test_protocols.py                               # Task 2
    test_scope.py                                   # Task 3
    test_thresholds.py                              # Task 4
    test_fallbacks.py                               # Task 5
    test_breaker.py                                 # Task 6
    test_adapters_workflow.py                       # Task 7
    test_adapters_step.py                           # Task 8
    test_adapters_capability.py                     # Task 9
    test_integration.py                             # Task 10
```

Top-level `ballast.__init__.py` gets `CircuitBreaker` re-export in Task 11.

---

## Task 1: `BreakerState` + `BreakerStats` + `CircuitOpenError`

**Files:**
- Create: `src/ballast/resilience/__init__.py` (empty package marker)
- Create: `src/ballast/resilience/circuit_breaker/__init__.py` (empty for now)
- Create: `src/ballast/resilience/circuit_breaker/_state.py`
- Create: `tests/resilience/__init__.py` (empty)
- Create: `tests/resilience/circuit_breaker/__init__.py` (empty)
- Create: `tests/resilience/circuit_breaker/test_state.py`

- [ ] **Step 1: Write the failing test (`tests/resilience/circuit_breaker/test_state.py`)**

```python
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
    # str enum can be serialized cleanly
    assert str(BreakerState.OPEN) == "BreakerState.OPEN" or BreakerState.OPEN.value == "open"


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
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/resilience/circuit_breaker/test_state.py -v`
Expected: `ModuleNotFoundError: No module named 'ballast.resilience'`.

- [ ] **Step 3: Create empty package markers**

- `src/ballast/resilience/__init__.py` — empty
- `src/ballast/resilience/circuit_breaker/__init__.py` — empty (populated in Task 11)
- `tests/resilience/__init__.py` — empty
- `tests/resilience/circuit_breaker/__init__.py` — empty

- [ ] **Step 4: Implement `src/ballast/resilience/circuit_breaker/_state.py`**

```python
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
```

- [ ] **Step 5: Run — confirm pass**

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/resilience src/ballast/resilience/circuit_breaker tests/resilience
git commit -m "feat(circuit-breaker): BreakerState + BreakerStats + CircuitOpenError"
```

---

## Task 2: `ThresholdPolicy` + `FallbackPolicy` Protocols + typing aliases

**Files:**
- Create: `src/ballast/resilience/circuit_breaker/_protocols.py`
- Create: `tests/resilience/circuit_breaker/test_protocols.py`

- [ ] **Step 1: Failing test (`tests/resilience/circuit_breaker/test_protocols.py`)**

```python
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
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement `src/ballast/resilience/circuit_breaker/_protocols.py`**

```python
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
    """When does the breaker open?

    Stateful — implementations track samples per scope. The framework
    calls ``on_outcome`` after every fn invocation; ``trip`` answers
    "open now?". ``reset`` is called when the state transitions back to
    CLOSED (Half-Open success).
    """

    def on_outcome(self, *, success: bool, at: datetime) -> None: ...
    def trip(self, *, at: datetime) -> bool: ...
    def reset(self) -> None: ...


@runtime_checkable
class FallbackPolicy(Protocol):
    """What to do when invocation is rejected (Open state or denied probe).

    Receives a snapshot of breaker stats + the original args. Returns
    whatever the caller should see, or raises.
    """

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
```

- [ ] **Step 4: Run — confirm pass**

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/resilience/circuit_breaker/_protocols.py tests/resilience/circuit_breaker/test_protocols.py
git commit -m "feat(circuit-breaker): ThresholdPolicy + FallbackPolicy Protocols + typing aliases"
```

---

## Task 3: Built-in `ScopeKey` helpers

**Files:**
- Create: `src/ballast/resilience/circuit_breaker/_scope.py`
- Create: `tests/resilience/circuit_breaker/test_scope.py`

- [ ] **Step 1: Failing test**

```python
"""Built-in ScopeKey helpers."""
from __future__ import annotations

from ballast.resilience.circuit_breaker._scope import (
    global_scope, per_step_scope, per_tool_scope,
)


def test_global_scope_constant() -> None:
    assert global_scope({}) == "global"
    assert global_scope({"tool_name": "x"}) == "global"


def test_per_tool_scope_uses_tool_name() -> None:
    assert per_tool_scope({"tool_name": "search"}) == "tool:search"
    assert per_tool_scope({}) == "tool:unknown"


def test_per_step_scope_uses_step_id() -> None:
    assert per_step_scope({"step_id": "s1"}) == "step:s1"
    assert per_step_scope({}) == "step:unknown"
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement `src/ballast/resilience/circuit_breaker/_scope.py`**

```python
"""Built-in ``ScopeKey`` helpers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def global_scope(_ctx: Mapping[str, Any]) -> str:
    """All invocations share one scope. The simplest case."""
    return "global"


def per_tool_scope(ctx: Mapping[str, Any]) -> str:
    """One scope per tool name. Wire when CB protects tool calls."""
    return f"tool:{ctx.get('tool_name', 'unknown')}"


def per_step_scope(ctx: Mapping[str, Any]) -> str:
    """One scope per PlanAndExecute step id. Wire when CB protects DAG nodes."""
    return f"step:{ctx.get('step_id', 'unknown')}"


__all__ = ["global_scope", "per_step_scope", "per_tool_scope"]
```

- [ ] **Step 4: Run — confirm pass**

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/resilience/circuit_breaker/_scope.py tests/resilience/circuit_breaker/test_scope.py
git commit -m "feat(circuit-breaker): built-in ScopeKey helpers (global / per_tool / per_step)"
```

---

## Task 4: Built-in `ThresholdPolicy` impls (`Consecutive`, `WindowedCount`, `WindowedRate`)

**Files:**
- Create: `src/ballast/resilience/circuit_breaker/_thresholds.py`
- Create: `tests/resilience/circuit_breaker/test_thresholds.py`

- [ ] **Step 1: Failing tests**

```python
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
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement `src/ballast/resilience/circuit_breaker/_thresholds.py`**

```python
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
```

- [ ] **Step 4: Run — confirm pass**

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/resilience/circuit_breaker/_thresholds.py tests/resilience/circuit_breaker/test_thresholds.py
git commit -m "feat(circuit-breaker): ThresholdPolicy impls (Consecutive / WindowedCount / WindowedRate)"
```

---

## Task 5: Built-in `FallbackPolicy` impls

**Files:**
- Create: `src/ballast/resilience/circuit_breaker/_fallbacks.py`
- Create: `tests/resilience/circuit_breaker/test_fallbacks.py`

- [ ] **Step 1: Failing tests**

```python
"""Built-in FallbackPolicy implementations."""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from ballast.resilience.circuit_breaker._fallbacks import (
    CallFallback, Chain, EscalateToHITL, RaiseError, ReturnValue,
)
from ballast.resilience.circuit_breaker._protocols import FallbackPolicy
from ballast.resilience.circuit_breaker._state import BreakerState, BreakerStats, CircuitOpenError


def _stats() -> BreakerStats:
    return BreakerStats(
        scope="x", state=BreakerState.OPEN,
        consecutive_failures=5, total_failures=5, total_successes=0,
        opened_at=None, will_attempt_recovery_at=None,
        probe_attempts=0, probe_max=1,
    )


async def _noop_fn(*args, **kwargs):
    return "real"


@pytest.mark.asyncio
async def test_raise_error_raises_circuit_open_error() -> None:
    with pytest.raises(CircuitOpenError) as exc:
        await RaiseError().on_rejected(_stats(), _noop_fn, (), {})
    assert exc.value.stats.scope == "x"


@pytest.mark.asyncio
async def test_return_value_returns_stored() -> None:
    fb = ReturnValue("cached")
    out = await fb.on_rejected(_stats(), _noop_fn, (), {})
    assert out == "cached"


@pytest.mark.asyncio
async def test_call_fallback_invokes_without_stats_param() -> None:
    seen: list[tuple] = []

    async def my_fb(a, b, *, c=None):
        seen.append((a, b, c))
        return "fb"

    out = await CallFallback(my_fb).on_rejected(_stats(), _noop_fn, (1, 2), {"c": "x"})
    assert out == "fb"
    assert seen == [(1, 2, "x")]


@pytest.mark.asyncio
async def test_call_fallback_invokes_with_stats_param() -> None:
    captured: dict = {}

    async def my_fb(a, *, stats=None):
        captured["a"] = a
        captured["stats"] = stats
        return "ok"

    out = await CallFallback(my_fb).on_rejected(_stats(), _noop_fn, (42,), {})
    assert out == "ok"
    assert captured["a"] == 42
    assert isinstance(captured["stats"], BreakerStats)


@pytest.mark.asyncio
async def test_escalate_to_hitl_calls_channel_request_blocking() -> None:
    requested = []

    class _Card:
        def __init__(self, stats): self.stats = stats

    class _FakeChannel:
        async def request(self, payload, *, timeout=None):
            requested.append({"payload": payload, "timeout": timeout})
            return "human_verdict"

    out = await EscalateToHITL(
        channel=_FakeChannel(),
        card_factory=_Card,
        timeout=timedelta(minutes=5),
    ).on_rejected(_stats(), _noop_fn, (), {})

    assert out == "human_verdict"
    assert len(requested) == 1
    assert isinstance(requested[0]["payload"], _Card)
    assert requested[0]["timeout"] == timedelta(minutes=5)


@pytest.mark.asyncio
async def test_chain_returns_first_success() -> None:
    class _Bad:
        async def on_rejected(self, *args): raise RuntimeError("nope")

    class _Ok:
        async def on_rejected(self, *args): return "from_ok"

    out = await Chain(_Bad(), _Ok()).on_rejected(_stats(), _noop_fn, (), {})
    assert out == "from_ok"


@pytest.mark.asyncio
async def test_chain_raises_last_when_all_fail() -> None:
    class _Bad1:
        async def on_rejected(self, *args): raise RuntimeError("first")

    class _Bad2:
        async def on_rejected(self, *args): raise ValueError("second")

    with pytest.raises(ValueError, match="second"):
        await Chain(_Bad1(), _Bad2()).on_rejected(_stats(), _noop_fn, (), {})


def test_chain_requires_at_least_one_policy() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Chain()
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement `src/ballast/resilience/circuit_breaker/_fallbacks.py`**

```python
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
    """Open HITL request via a ``HITLChannel`` and BLOCK until human verdict.

    The card factory builds the channel payload from the breaker stats.
    Returns whatever the channel returns (typed verdict).
    """

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
```

- [ ] **Step 4: Run — confirm pass**

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/resilience/circuit_breaker/_fallbacks.py tests/resilience/circuit_breaker/test_fallbacks.py
git commit -m "feat(circuit-breaker): FallbackPolicy impls (RaiseError / ReturnValue / CallFallback / EscalateToHITL / Chain)"
```

---

## Task 6: `CircuitBreaker` core class + `_ScopeBucket` state machine

**Files:**
- Create: `src/ballast/resilience/circuit_breaker/_breaker.py`
- Create: `tests/resilience/circuit_breaker/test_breaker.py`

- [ ] **Step 1: Failing tests**

```python
"""CircuitBreaker core — .call(), per-scope state, transitions."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ballast.resilience.circuit_breaker._breaker import CircuitBreaker
from ballast.resilience.circuit_breaker._fallbacks import RaiseError, ReturnValue
from ballast.resilience.circuit_breaker._scope import global_scope, per_tool_scope
from ballast.resilience.circuit_breaker._state import (
    BreakerState, CircuitOpenError,
)
from ballast.resilience.circuit_breaker._thresholds import Consecutive


# ---- Mockable clock --------------------------------------------------------

class _Clock:
    def __init__(self, start: datetime): self.now = start
    def advance(self, td: timedelta) -> None: self.now += td
    def __call__(self) -> datetime: return self.now


def _mk_clock() -> _Clock:
    return _Clock(datetime(2026, 1, 1, tzinfo=UTC))


# ---- Tests -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_closed_passes_through() -> None:
    cb = CircuitBreaker(threshold_factory=lambda: Consecutive(3), clock=_mk_clock())

    async def ok(): return "out"

    assert await cb.call(ok) == "out"
    assert cb.stats().state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_closed_to_open_after_threshold() -> None:
    cb = CircuitBreaker(threshold_factory=lambda: Consecutive(2), clock=_mk_clock())

    async def boom(): raise RuntimeError("nope")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(boom)
    assert cb.stats().state == BreakerState.OPEN


@pytest.mark.asyncio
async def test_open_invokes_fallback_with_raise_error_default() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(threshold_factory=lambda: Consecutive(1), clock=clock)

    async def boom(): raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await cb.call(boom)

    async def ok(): return "fresh"

    with pytest.raises(CircuitOpenError):
        await cb.call(ok)


@pytest.mark.asyncio
async def test_open_returns_via_return_value_fallback() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        fallback=ReturnValue("cached"),
        clock=clock,
    )

    async def boom(): raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await cb.call(boom)

    async def ok(): return "fresh"

    assert await cb.call(ok) == "cached"


@pytest.mark.asyncio
async def test_open_to_half_open_after_recovery() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        recovery_after=timedelta(seconds=10),
        clock=clock,
    )

    async def boom(): raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await cb.call(boom)
    assert cb.stats().state == BreakerState.OPEN

    clock.advance(timedelta(seconds=11))
    # First call after recovery — probe (HALF_OPEN); succeeds → CLOSED
    async def ok(): return "out"
    assert await cb.call(ok) == "out"
    assert cb.stats().state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_half_open_probe_failure_reopens() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        recovery_after=timedelta(seconds=10),
        clock=clock,
    )

    async def boom(): raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await cb.call(boom)
    clock.advance(timedelta(seconds=11))

    with pytest.raises(RuntimeError):
        await cb.call(boom)
    assert cb.stats().state == BreakerState.OPEN


@pytest.mark.asyncio
async def test_half_open_extra_probes_rejected() -> None:
    clock = _mk_clock()
    rejected = []

    class _CapturingFallback:
        async def on_rejected(self, stats, fn, args, kwargs):
            rejected.append(stats.state)
            return "rejected"

    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        recovery_after=timedelta(seconds=10),
        probe_max=1,
        fallback=_CapturingFallback(),
        clock=clock,
    )

    async def boom(): raise RuntimeError("nope")
    async def slow(): await asyncio.sleep(0.05); return "ok"

    # OPEN it
    with pytest.raises(RuntimeError):
        await cb.call(boom)

    # Advance past recovery
    clock.advance(timedelta(seconds=11))

    # Concurrent probes — one allowed, rest rejected
    results = await asyncio.gather(
        cb.call(slow), cb.call(slow), cb.call(slow),
        return_exceptions=True,
    )
    # At least one rejected via fallback
    assert "rejected" in results


@pytest.mark.asyncio
async def test_ignored_exception_does_not_count_as_failure() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        ignored_exc=(KeyError,),
        clock=clock,
    )

    async def boom(): raise KeyError("ignored")

    with pytest.raises(KeyError):
        await cb.call(boom)
    assert cb.stats().state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_is_failure_exc_filters_other_exceptions() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        is_failure_exc=(RuntimeError,),
        clock=clock,
    )

    async def boom(): raise ValueError("other type")

    with pytest.raises(ValueError):
        await cb.call(boom)
    # ValueError NOT in is_failure_exc → not counted
    assert cb.stats().state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_is_success_predicate_treats_returned_value_as_failure() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(2),
        is_success=lambda r: r != "bad",
        clock=clock,
    )

    async def maybe_bad(): return "bad"

    await cb.call(maybe_bad)
    await cb.call(maybe_bad)
    assert cb.stats().state == BreakerState.OPEN


@pytest.mark.asyncio
async def test_per_scope_isolation() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        scope_key=per_tool_scope,
        clock=clock,
    )

    async def boom(): raise RuntimeError("nope")
    async def ok(): return "out"

    # Trip tool:search
    with pytest.raises(RuntimeError):
        await cb.call(boom, ctx={"tool_name": "search"})
    assert cb.stats("tool:search").state == BreakerState.OPEN

    # tool:other still CLOSED
    assert await cb.call(ok, ctx={"tool_name": "other"}) == "out"
    assert cb.stats("tool:other").state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_reset_forces_closed() -> None:
    clock = _mk_clock()
    cb = CircuitBreaker(threshold_factory=lambda: Consecutive(1), clock=clock)

    async def boom(): raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await cb.call(boom)
    assert cb.stats().state == BreakerState.OPEN

    cb.reset()
    assert cb.stats().state == BreakerState.CLOSED


def test_constructor_validates_probe_max() -> None:
    with pytest.raises(ValueError, match="probe_max"):
        CircuitBreaker(probe_max=0)
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement `src/ballast/resilience/circuit_breaker/_breaker.py`**

```python
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
        fallback:          FallbackPolicy                  = None,        # default: RaiseError()
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

    async def call(self, fn, args, kwargs):
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

        # Execute outside the lock — fn may be long.
        try:
            result = await fn(*args, **kwargs)
        except self._owner._ignored_exc:
            raise
        except self._owner._is_failure_exc:
            async with self._lock:
                self._record(success=False, at=self._owner._clock())
            raise
        except BaseException:
            # Not in is_failure_exc → don't record, just propagate.
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
```

- [ ] **Step 4: Run — confirm pass**

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/resilience/circuit_breaker/_breaker.py tests/resilience/circuit_breaker/test_breaker.py
git commit -m "feat(circuit-breaker): CircuitBreaker core + _ScopeBucket state machine"
```

---

## Task 7: `as_workflow_decorator` adapter

**Files:**
- Create: `src/ballast/resilience/circuit_breaker/_adapters/__init__.py` (empty for now)
- Create: `src/ballast/resilience/circuit_breaker/_adapters/workflow.py`
- Create: `tests/resilience/circuit_breaker/test_adapters_workflow.py`

- [ ] **Step 1: Failing test**

```python
"""as_workflow_decorator — decorates an async fn, runs it through CircuitBreaker."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ballast.resilience.circuit_breaker._adapters.workflow import as_workflow_decorator
from ballast.resilience.circuit_breaker._breaker import CircuitBreaker
from ballast.resilience.circuit_breaker._fallbacks import ReturnValue
from ballast.resilience.circuit_breaker._scope import per_tool_scope
from ballast.resilience.circuit_breaker._state import BreakerState
from ballast.resilience.circuit_breaker._thresholds import Consecutive


class _Clock:
    def __init__(self): self.now = datetime(2026, 1, 1, tzinfo=UTC)
    def __call__(self): return self.now


@pytest.mark.asyncio
async def test_decorator_passes_through_when_closed() -> None:
    cb = CircuitBreaker(clock=_Clock())

    @as_workflow_decorator(cb)
    async def body(x: int) -> int:
        return x * 2

    assert await body(5) == 10


@pytest.mark.asyncio
async def test_decorator_opens_after_failures_then_uses_fallback() -> None:
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(2),
        fallback=ReturnValue("fallback"),
        clock=_Clock(),
    )

    @as_workflow_decorator(cb)
    async def body() -> str:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await body()
    with pytest.raises(RuntimeError):
        await body()
    # Now CB is OPEN — third call returns fallback
    assert await body() == "fallback"


@pytest.mark.asyncio
async def test_decorator_propagates_scope_ctx() -> None:
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        scope_key=per_tool_scope,
        clock=_Clock(),
    )

    @as_workflow_decorator(cb, scope_ctx={"tool_name": "publish_wf"})
    async def body() -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await body()
    assert cb.stats("tool:publish_wf").state == BreakerState.OPEN
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement `src/ballast/resilience/circuit_breaker/_adapters/workflow.py`**

```python
"""``as_workflow_decorator`` — workflow surface for CircuitBreaker."""
from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

from ballast.resilience.circuit_breaker._breaker import CircuitBreaker

T = TypeVar("T")


def as_workflow_decorator(
    breaker: CircuitBreaker, *,
    scope_ctx: Mapping[str, Any] | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorate an async function so its invocations flow through the breaker.

    ``scope_ctx`` is forwarded to ``breaker.scope_key`` to determine the
    breaker's per-scope bucket. None → ``scope_key`` receives ``{}``.
    """
    ctx = scope_ctx or {}

    def deco(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await breaker.call(fn, *args, ctx=ctx, **kwargs)

        return wrapper

    return deco


__all__ = ["as_workflow_decorator"]
```

- [ ] **Step 4: Run — confirm pass**

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/resilience/circuit_breaker/_adapters/__init__.py src/ballast/resilience/circuit_breaker/_adapters/workflow.py tests/resilience/circuit_breaker/test_adapters_workflow.py
git commit -m "feat(circuit-breaker): as_workflow_decorator adapter"
```

---

## Task 8: `BreakerStep` + `as_step` adapter

**Files:**
- Create: `src/ballast/resilience/circuit_breaker/_adapters/step.py`
- Create: `tests/resilience/circuit_breaker/test_adapters_step.py`

- [ ] **Step 1: Failing test**

```python
"""BreakerStep — wraps a Step (PlanAndExecute) through the breaker."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._protocols import StepContext
from ballast.resilience.circuit_breaker._adapters.step import BreakerStep, as_step
from ballast.resilience.circuit_breaker._breaker import CircuitBreaker
from ballast.resilience.circuit_breaker._fallbacks import ReturnValue
from ballast.resilience.circuit_breaker._scope import per_step_scope
from ballast.resilience.circuit_breaker._state import BreakerState
from ballast.resilience.circuit_breaker._thresholds import Consecutive


class _Clock:
    def __init__(self): self.now = datetime(2026, 1, 1, tzinfo=UTC)
    def __call__(self): return self.now


class _OkStep:
    async def execute(self, plan_input, dep_outputs, ctx):
        return "ok"


class _BoomStep:
    async def execute(self, plan_input, dep_outputs, ctx):
        raise RuntimeError("step failed")


def _ctx(step_id: str = "s1", kind: str = "callable") -> StepContext:
    return StepContext(
        plan=Plan(steps=[PlannedStep(id=step_id, kind=kind)]),
        step=PlannedStep(id=step_id, kind=kind),
        step_registry=None,
    )


@pytest.mark.asyncio
async def test_breaker_step_passes_through_when_closed() -> None:
    cb = CircuitBreaker(clock=_Clock())
    wrapped = as_step(cb, _OkStep())
    out = await wrapped.execute(plan_input=None, dep_outputs={}, ctx=_ctx())
    assert out == "ok"


@pytest.mark.asyncio
async def test_breaker_step_uses_per_step_scope() -> None:
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        scope_key=per_step_scope,
        clock=_Clock(),
    )
    wrapped = as_step(cb, _BoomStep())

    with pytest.raises(RuntimeError):
        await wrapped.execute(plan_input=None, dep_outputs={}, ctx=_ctx("s1"))
    assert cb.stats("step:s1").state == BreakerState.OPEN
    # Other step id still CLOSED
    assert cb.stats("step:s2").state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_breaker_step_routes_to_fallback_when_open() -> None:
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        fallback=ReturnValue("fallback"),
        clock=_Clock(),
    )
    wrapped = as_step(cb, _BoomStep())

    with pytest.raises(RuntimeError):
        await wrapped.execute(plan_input=None, dep_outputs={}, ctx=_ctx())
    out = await wrapped.execute(plan_input=None, dep_outputs={}, ctx=_ctx())
    assert out == "fallback"


def test_breaker_step_constructor_via_as_step() -> None:
    cb = CircuitBreaker(clock=_Clock())
    bs = as_step(cb, _OkStep())
    assert isinstance(bs, BreakerStep)
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement `src/ballast/resilience/circuit_breaker/_adapters/step.py`**

```python
"""``BreakerStep`` + ``as_step`` — wraps a PlanAndExecute Step through the breaker."""
from __future__ import annotations

from typing import Any

from ballast.resilience.circuit_breaker._breaker import CircuitBreaker


class BreakerStep:
    """Wraps any ``Step`` (PlanAndExecute) — invocations flow through the breaker.

    The breaker's ``scope_key`` receives ``{"step_id": ctx.step.id, "step_kind": ctx.step.kind}``,
    so ``per_step_scope`` gives per-DAG-node isolation out of the box.
    """

    def __init__(self, *, breaker: CircuitBreaker, wrapped: Any) -> None:
        self._breaker = breaker
        self._wrapped = wrapped

    async def execute(self, plan_input, dep_outputs, ctx):
        scope_ctx = {"step_id": ctx.step.id, "step_kind": ctx.step.kind}
        return await self._breaker.call(
            self._wrapped.execute, plan_input, dep_outputs, ctx,
            ctx=scope_ctx,
        )


def as_step(breaker: CircuitBreaker, wrapped: Any) -> BreakerStep:
    """Wrap ``wrapped`` (any object implementing ``Step.execute``) through ``breaker``."""
    return BreakerStep(breaker=breaker, wrapped=wrapped)


__all__ = ["BreakerStep", "as_step"]
```

- [ ] **Step 4: Run — confirm pass**

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/resilience/circuit_breaker/_adapters/step.py tests/resilience/circuit_breaker/test_adapters_step.py
git commit -m "feat(circuit-breaker): BreakerStep + as_step adapter (PlanAndExecute integration)"
```

---

## Task 9: `as_capability` adapter

**Files:**
- Create: `src/ballast/resilience/circuit_breaker/_adapters/capability.py`
- Create: `tests/resilience/circuit_breaker/test_adapters_capability.py`

- [ ] **Step 1: Failing test**

```python
"""as_capability — wraps agent.run() through the breaker via after_run hook."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ballast.resilience.circuit_breaker._adapters.capability import as_capability
from ballast.resilience.circuit_breaker._breaker import CircuitBreaker
from ballast.resilience.circuit_breaker._fallbacks import RaiseError, ReturnValue
from ballast.resilience.circuit_breaker._state import BreakerState
from ballast.resilience.circuit_breaker._thresholds import Consecutive


class _Clock:
    def __init__(self): self.now = datetime(2026, 1, 1, tzinfo=UTC)
    def __call__(self): return self.now


@pytest.mark.asyncio
async def test_as_capability_returns_ballast_capability_instance() -> None:
    from ballast.capabilities.base import BallastCapability
    cap = as_capability(CircuitBreaker(clock=_Clock()))
    assert isinstance(cap, BallastCapability)


@pytest.mark.asyncio
async def test_capability_records_failure_when_result_carries_exception() -> None:
    cb = CircuitBreaker(threshold_factory=lambda: Consecutive(2), clock=_Clock())
    cap = as_capability(cb)
    per_run = await cap.for_run(ctx=None)

    # Simulate two runs failing (recorded via after_run callback)
    class _FakeResult:
        output = "anything"

    # Use cap.after_run with synthetic ctx + result; cap counts via breaker.
    # We mimic an exception by raising in after_run path — but after_run is
    # called AFTER agent.run succeeded. So failures must come from the agent
    # raising during run, which doesn't trigger after_run at all. Therefore
    # capability tracks agent OUTPUT via is_success predicate.
    cb_with_pred = CircuitBreaker(
        threshold_factory=lambda: Consecutive(2),
        is_success=lambda res: getattr(res, "output", None) != "bad",
        clock=_Clock(),
    )
    cap2 = as_capability(cb_with_pred)
    per_run2 = await cap2.for_run(ctx=None)

    class _BadResult:
        output = "bad"

    await per_run2.after_run(ctx=None, result=_BadResult())
    await per_run2.after_run(ctx=None, result=_BadResult())
    # After two "bad" results, breaker should be OPEN
    assert cb_with_pred.stats().state == BreakerState.OPEN
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement `src/ballast/resilience/circuit_breaker/_adapters/capability.py`**

```python
"""``as_capability`` — agent surface for CircuitBreaker.

Wraps full ``agent.run()`` invocations via the ``after_run`` hook.
Each agent run's result is fed through the breaker's ``is_success``
predicate; failures advance the counter.

Per-tool wrapping at the pydantic-ai tool level is OUT OF SCOPE for the
first cut — apps that need per-tool CB use the workflow decorator + a
manual ``breaker.call(...)`` wrapper around their tool function, OR
``BreakerStep`` for PlanAndExecute DAG nodes.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import RunContext

from ballast.capabilities.base import BallastCapability
from ballast.resilience.circuit_breaker._breaker import CircuitBreaker

_log = logging.getLogger("ballast.resilience.circuit_breaker.capability")


class _CBCapability(BallastCapability):
    """Tracks agent.run() outcomes through the configured CircuitBreaker."""

    name = "circuit_breaker"

    def __init__(self, breaker: CircuitBreaker) -> None:
        self._breaker = breaker

    async def for_run(self, ctx: RunContext[Any]) -> "_CBCapability":
        # Stateless wrapper — same breaker shared across runs (it has its
        # own per-scope state already).
        return self

    async def after_run(self, ctx: RunContext[Any], *, result: Any) -> Any:
        # Feed the run outcome through the breaker as an "invocation". We
        # use a synthetic no-op function so the breaker's bookkeeping
        # (success vs failure via is_success) updates correctly.
        async def _noop() -> Any:
            return result

        try:
            await self._breaker.call(_noop, ctx={"agent_run": True})
        except Exception:
            # Breaker rejected (OPEN with RaiseError) or fn unexpected error.
            # Swallow — we don't want to crash the agent run because the CB
            # complains; let downstream observers see breaker.stats() instead.
            _log.exception("circuit-breaker after_run propagation swallowed")
        return result


def as_capability(breaker: CircuitBreaker) -> BallastCapability:
    """Wrap ``breaker`` as a ``BallastCapability`` for agent runs."""
    return _CBCapability(breaker)


__all__ = ["as_capability"]
```

- [ ] **Step 4: Run — confirm pass**

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/resilience/circuit_breaker/_adapters/capability.py tests/resilience/circuit_breaker/test_adapters_capability.py
git commit -m "feat(circuit-breaker): as_capability adapter (agent run outcome tracking)"
```

---

## Task 10: Integration smoke test

**Files:**
- Create: `tests/resilience/circuit_breaker/test_integration.py`

- [ ] **Step 1: Write integration tests**

```python
"""End-to-end integration: CircuitBreaker with all primitives composed."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ballast.resilience.circuit_breaker._breaker import CircuitBreaker
from ballast.resilience.circuit_breaker._fallbacks import (
    CallFallback, Chain, RaiseError, ReturnValue,
)
from ballast.resilience.circuit_breaker._scope import per_tool_scope
from ballast.resilience.circuit_breaker._state import (
    BreakerState, CircuitOpenError,
)
from ballast.resilience.circuit_breaker._thresholds import (
    Consecutive, WindowedCount, WindowedRate,
)


class _Clock:
    def __init__(self, start: datetime): self.now = start
    def advance(self, td: timedelta): self.now += td
    def __call__(self): return self.now


def _clock() -> _Clock:
    return _Clock(datetime(2026, 1, 1, tzinfo=UTC))


@pytest.mark.asyncio
async def test_full_lifecycle_with_recovery() -> None:
    """CLOSED → OPEN → HALF_OPEN → CLOSED through a real recovery cycle."""
    clock = _clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(2),
        recovery_after=timedelta(seconds=5),
        clock=clock,
    )

    async def flaky(succeed: bool):
        if not succeed:
            raise RuntimeError("boom")
        return "ok"

    # 2 failures → OPEN
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(flaky, False)
    assert cb.stats().state == BreakerState.OPEN

    # Still OPEN before recovery — raises CircuitOpenError
    with pytest.raises(CircuitOpenError):
        await cb.call(flaky, True)

    # Advance past recovery; first probe succeeds → CLOSED
    clock.advance(timedelta(seconds=6))
    assert await cb.call(flaky, True) == "ok"
    assert cb.stats().state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_call_fallback_chain_with_hitl_simulated() -> None:
    """Chain: CallFallback → ReturnValue. First-success-wins."""
    clock = _clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(1),
        fallback=Chain(
            CallFallback(lambda *a, **kw: _failing_fallback()),
            ReturnValue("ultimate_fallback"),
        ),
        clock=clock,
    )

    async def boom(): raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await cb.call(boom)

    # Now OPEN; Chain tries CallFallback (fails), then ReturnValue (succeeds)
    out = await cb.call(boom)
    assert out == "ultimate_fallback"


async def _failing_fallback():
    raise RuntimeError("fallback also broken")


@pytest.mark.asyncio
async def test_per_tool_scope_isolation_under_load() -> None:
    """Multiple scopes don't share state, even under concurrent load."""
    clock = _clock()
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(2),
        scope_key=per_tool_scope,
        clock=clock,
    )

    async def call_for_tool(name: str, succeed: bool) -> Any:
        async def fn(): 
            if not succeed:
                raise RuntimeError("nope")
            return f"ok:{name}"
        return await cb.call(fn, ctx={"tool_name": name})

    # Trip tool A
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await call_for_tool("a", False)
    assert cb.stats("tool:a").state == BreakerState.OPEN

    # Tool B still passes through
    assert await call_for_tool("b", True) == "ok:b"
    assert cb.stats("tool:b").state == BreakerState.CLOSED
```

- [ ] **Step 2: Run — confirm pass**

Run: `uv run pytest tests/resilience/circuit_breaker/test_integration.py -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/resilience/circuit_breaker/test_integration.py
git commit -m "test(circuit-breaker): end-to-end integration scenarios"
```

---

## Task 11: Public API re-exports

**Files:**
- Modify: `src/ballast/resilience/circuit_breaker/__init__.py`
- Modify: `src/ballast/resilience/circuit_breaker/_adapters/__init__.py`
- Modify: `src/ballast/resilience/__init__.py`
- Modify: `src/ballast/__init__.py`

- [ ] **Step 1: Subpackage `__init__.py`**

`src/ballast/resilience/circuit_breaker/__init__.py`:
```python
"""Circuit Breaker — resilience primitive for protecting async function invocations."""
from ballast.resilience.circuit_breaker._adapters.capability import as_capability
from ballast.resilience.circuit_breaker._adapters.step import BreakerStep, as_step
from ballast.resilience.circuit_breaker._adapters.workflow import as_workflow_decorator
from ballast.resilience.circuit_breaker._breaker import CircuitBreaker
from ballast.resilience.circuit_breaker._fallbacks import (
    CallFallback, Chain, EscalateToHITL, RaiseError, ReturnValue,
)
from ballast.resilience.circuit_breaker._protocols import (
    FallbackPolicy, ScopeKey, ThresholdFactory, ThresholdPolicy,
)
from ballast.resilience.circuit_breaker._scope import (
    global_scope, per_step_scope, per_tool_scope,
)
from ballast.resilience.circuit_breaker._state import (
    BreakerState, BreakerStats, CircuitOpenError,
)
from ballast.resilience.circuit_breaker._thresholds import (
    Consecutive, WindowedCount, WindowedRate,
)

__all__ = [
    "BreakerState", "BreakerStats", "BreakerStep",
    "CallFallback", "Chain", "CircuitBreaker", "CircuitOpenError",
    "Consecutive", "EscalateToHITL", "FallbackPolicy",
    "RaiseError", "ReturnValue", "ScopeKey",
    "ThresholdFactory", "ThresholdPolicy",
    "WindowedCount", "WindowedRate",
    "as_capability", "as_step", "as_workflow_decorator",
    "global_scope", "per_step_scope", "per_tool_scope",
]
```

`src/ballast/resilience/circuit_breaker/_adapters/__init__.py`:
```python
"""Runtime adapters for CircuitBreaker — capability / workflow / step."""
from ballast.resilience.circuit_breaker._adapters.capability import as_capability
from ballast.resilience.circuit_breaker._adapters.step import BreakerStep, as_step
from ballast.resilience.circuit_breaker._adapters.workflow import as_workflow_decorator

__all__ = ["BreakerStep", "as_capability", "as_step", "as_workflow_decorator"]
```

- [ ] **Step 2: Top-level `ballast.resilience.__init__.py`**

```python
"""Resilience primitives — Circuit Breaker, future Retry/Bulkhead/RateLimiter."""
from ballast.resilience.circuit_breaker import (
    CircuitBreaker, CircuitOpenError,
)

__all__ = ["CircuitBreaker", "CircuitOpenError"]
```

- [ ] **Step 3: Edit `src/ballast/__init__.py`**

Add to imports:
```python
from ballast.resilience.circuit_breaker import CircuitBreaker
```

Add `"CircuitBreaker"` to `__all__` (alphabetical).

- [ ] **Step 4: Smoke import**

```
uv run python -c "from ballast import CircuitBreaker; print('ok')"
```
Expected: `ok`.

```
uv run python -c "
from ballast.resilience.circuit_breaker import (
    CircuitBreaker, BreakerState, BreakerStats, CircuitOpenError,
    ThresholdPolicy, FallbackPolicy, ScopeKey, ThresholdFactory,
    Consecutive, WindowedCount, WindowedRate,
    RaiseError, ReturnValue, CallFallback, EscalateToHITL, Chain,
    global_scope, per_tool_scope, per_step_scope,
    as_capability, as_workflow_decorator, as_step, BreakerStep,
)
print('circuit_breaker subpackage ok')
"
```
Expected: `circuit_breaker subpackage ok`.

- [ ] **Step 5: Run full framework suite**

Run: `uv run pytest tests/ --tb=short -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/__init__.py src/ballast/resilience/__init__.py src/ballast/resilience/circuit_breaker/__init__.py src/ballast/resilience/circuit_breaker/_adapters/__init__.py
git commit -m "feat(ballast): re-export CircuitBreaker at top level + subpackage public API"
```

---

## Task 12: Final smoke

- [ ] **Step 1: Run framework suite**

Run: `uv run pytest tests/ --tb=short -q`
Expected: green. All ~50 new tests plus existing 582+ still pass.

- [ ] **Step 2: Run circuit-breaker suite specifically**

Run: `uv run pytest tests/resilience/circuit_breaker/ -v`
Expected: all green.

- [ ] **Step 3: Smoke import the whole framework**

```
uv run python -c "
from ballast import (
    Ballast, BallastSettings,
    CircuitBreaker,
    PlanAndExecute,
    CoALABase, CoALAUnit, as_workflow, as_tool, as_capability,
    GoalDriftDetector, with_drift_monitor,
)
from ballast.resilience.circuit_breaker import (
    BreakerState, Consecutive, RaiseError, per_tool_scope,
    as_workflow_decorator, as_step, BreakerStep,
)
print('all imports ok')
"
```
Expected: `all imports ok`.

- [ ] **Step 4: Commit any cleanup**

```bash
git status
git add -u && git commit -m "chore(circuit-breaker): final smoke cleanup" || echo "nothing to commit"
```

---

## Self-Review (against the spec)

**Spec coverage:**

| Spec section | Tasks |
|---|---|
| File structure + public API | Tasks 1-11 |
| `BreakerState` + `BreakerStats` + `CircuitOpenError` | Task 1 |
| Protocols + typing aliases | Task 2 |
| `ScopeKey` helpers | Task 3 |
| `ThresholdPolicy` impls (Consecutive / WindowedCount / WindowedRate) | Task 4 |
| `FallbackPolicy` impls (Raise / Return / Call / Escalate / Chain) | Task 5 |
| `CircuitBreaker` + `_ScopeBucket` core | Task 6 |
| `as_workflow_decorator` | Task 7 |
| `BreakerStep` + `as_step` | Task 8 |
| `as_capability` | Task 9 |
| Integration smoke | Task 10 |
| Public re-exports | Task 11 |
| Final smoke | Task 12 |

**Placeholder scan:** No TBDs/TODOs/vague-step-without-code. Every step has complete code or exact command + expected output.

**Type consistency:**
- `BreakerStats` fields (`scope`, `state`, `consecutive_failures`, `total_failures`, `total_successes`, `opened_at`, `will_attempt_recovery_at`, `probe_attempts`, `probe_max`) used consistently in Tasks 1, 6, 7, 8, 9, 10.
- `CircuitBreaker.__init__` kwargs consistent in Task 6 (definition) and Tasks 7-10 (callers).
- `ThresholdPolicy.on_outcome(*, success, at) / trip(*, at) / reset()` signature consistent in Task 2 (Protocol), Task 4 (impls), Task 6 (caller).
- `FallbackPolicy.on_rejected(stats, fn, args, kwargs)` consistent in Tasks 2, 5, 6, 7, 9, 10.
- `Step.execute(plan_input, dep_outputs, ctx)` from PlanAndExecute consistent in Task 8 (BreakerStep wraps it).
- `Clock` callable shape consistent across `CircuitBreaker.__init__` and all test fixtures.

**Known plan-vs-spec gap:** Per-tool wrapping at the pydantic-ai *tool* level (vs whole-agent-run) is documented in Task 9 as OUT OF SCOPE for first cut. Apps that need per-tool CB use `breaker.call(...)` manually around their tool function. This is consistent with spec's "first cut" focus.

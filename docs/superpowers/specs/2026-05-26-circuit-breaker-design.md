# Circuit Breaker — Design Spec

**Date:** 2026-05-26
**Status:** Approved (proceeding to plan)
**Source motivation:** "Архитектура и надёжность агентных LLM-систем в Production" — section on Circuit Breakers / mandatory final states (degraded fallback / context refresh / HITL escalation).

## Problem

Agentic LLM systems retrying a failing API in a tight loop burn budget without progress (classic "loop-happy" failure mode). The framework already has retry semantics in DBOS, but no circuit-breaker primitive: a stateful guard that — after N failures — refuses further attempts until a recovery window has passed, routing rejected calls to a deterministic fallback (cached value / alternative method / human escalation).

This spec defines `CircuitBreaker` as a core resilience primitive with three runtime adapters (capability / workflow decorator / PlanAndExecute Step wrapper).

## Goals

- One core `CircuitBreaker` primitive — `.call(fn, *args, ctx=...)` protects any async invocation.
- Per-scope isolation via configurable `scope_key: Callable[[ctx], str]`. One breaker can multiplex many independent scopes (e.g., per-tool).
- Configurable failure recognition: exception filter + success predicate.
- Pluggable `ThresholdPolicy` (when to open) with built-ins: `Consecutive`, `WindowedCount`, `WindowedRate`.
- Pluggable `FallbackPolicy` (what to do when rejected) with built-ins: `RaiseError`, `ReturnValue`, `CallFallback`, `EscalateToHITL`, `Chain`.
- Classic Closed / Open / Half-Open state machine; configurable `recovery_after` + `probe_max`.
- Three runtime adapters: `as_capability(breaker)` (agent surface), `as_workflow_decorator(breaker)` (workflow body), `as_step(breaker, wrapped)` (PlanAndExecute Step wrapping).

## Non-goals

- Persistent CB state (Redis / DB) — first cut is in-memory per process. Apps that need cluster-aware CB write a custom subclass.
- Bulkhead pattern (concurrent invocation limit) — separate resilience primitive.
- Rate limiter — separate primitive.
- Hedging (parallel duplicate calls, first-wins) — separate primitive.
- Per-scope distinct fallback / threshold within one breaker — first cut shares config across scopes. Apps wire multiple breakers when rules differ.

## Architecture

### File structure

```
src/ballast/resilience/                       # NEW top-level subpackage
  __init__.py                                 # cross-subpackage re-exports
  circuit_breaker/
    __init__.py                               # subpackage public exports
    _protocols.py                             # ThresholdPolicy + FallbackPolicy + ScopeKey
    _state.py                                 # BreakerState enum, BreakerStats, CircuitOpenError
    _thresholds.py                            # Consecutive / WindowedCount / WindowedRate
    _fallbacks.py                             # RaiseError / ReturnValue / CallFallback / EscalateToHITL / Chain
    _scope.py                                 # global_scope / per_tool_scope / per_step_scope helpers
    _breaker.py                               # CircuitBreaker + _ScopeBucket
    _adapters/
      __init__.py
      capability.py                           # as_capability(breaker)
      workflow.py                             # as_workflow_decorator(breaker)
      step.py                                 # as_step(breaker, wrapped) for PlanAndExecute

tests/resilience/
  __init__.py
  circuit_breaker/
    __init__.py
    test_state.py
    test_thresholds.py
    test_fallbacks.py
    test_breaker.py
    test_adapters_capability.py
    test_adapters_workflow.py
    test_adapters_step.py
    test_integration.py
```

**Placement:** `ballast.resilience/` is a new top-level subpackage for cross-cutting reliability concerns. CB lives here; future fellows (`Retry` policies, `Bulkhead`, `RateLimiter`, `Hedging`) join the same cluster.

### Public API

`from ballast.resilience.circuit_breaker import ...`:
- `CircuitBreaker` (core class)
- `BreakerState` (enum: `CLOSED` / `OPEN` / `HALF_OPEN`)
- `BreakerStats` (pydantic BaseModel snapshot for observability)
- `CircuitOpenError` (raised under `RaiseError` fallback)
- Protocols: `ThresholdPolicy`, `FallbackPolicy`
- Typing alias: `ScopeKey`, `ThresholdFactory`
- Built-in thresholds: `Consecutive`, `WindowedCount`, `WindowedRate`
- Built-in fallbacks: `RaiseError`, `ReturnValue`, `CallFallback`, `EscalateToHITL`, `Chain`
- Built-in scopes: `global_scope`, `per_tool_scope`, `per_step_scope`
- Adapters: `as_capability`, `as_workflow_decorator`, `as_step`, `BreakerStep`

Top-level `from ballast import CircuitBreaker` — yes, for consistency with `PlanAndExecute`, `GoalDriftDetector`.

## Components

### State + error types

```python
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


class CircuitOpenError(BallastError):
    """Raised by RaiseError fallback when breaker rejects an invocation."""
    code = "BALLAST_CIRCUIT_OPEN"
    status_code = 503

    def __init__(self, stats: BreakerStats) -> None:
        self.stats = stats
        super().__init__(
            f"Circuit breaker open for scope {stats.scope!r}",
            hint="Wait for recovery_after window, or supply a non-Raise fallback policy.",
            context={"breaker_stats": stats.model_dump(mode="json")},
        )
```

### Protocols + typing

```python
ScopeKey = Callable[[Mapping[str, Any]], str]
"""Maps invocation context dict → scope key string."""

ThresholdFactory = Callable[[], "ThresholdPolicy"]
"""Per-scope ThresholdPolicy is constructed via factory so each scope
has isolated state."""

@runtime_checkable
class ThresholdPolicy(Protocol):
    """When does the breaker open?

    Stateful — implementations track samples per scope. Framework calls
    on_outcome() after every fn invocation; trip() answers "open now?".
    Each scope gets its own ThresholdPolicy instance (via ``ThresholdFactory``).
    """
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
```

### Built-in scopes

```python
def global_scope(_ctx: Mapping[str, Any]) -> str:
    return "global"

def per_tool_scope(ctx: Mapping[str, Any]) -> str:
    return f"tool:{ctx.get('tool_name', 'unknown')}"

def per_step_scope(ctx: Mapping[str, Any]) -> str:
    return f"step:{ctx.get('step_id', 'unknown')}"
```

### Built-in thresholds

#### `Consecutive`

```python
class Consecutive:
    """Trip after N consecutive failures. Any success resets the counter."""
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
```

#### `WindowedCount`

```python
class WindowedCount:
    """Trip if >= max_failures occurred in the trailing `window`."""
    def __init__(self, max_failures: int = 5,
                 window: timedelta = timedelta(seconds=60)) -> None:
        if max_failures < 1:
            raise ValueError("max_failures must be >= 1")
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
```

#### `WindowedRate`

```python
class WindowedRate:
    """Trip if failure_count / total_count >= rate over `window`,
    provided total_count >= min_samples."""
    def __init__(self, rate: float = 0.5,
                 window: timedelta = timedelta(seconds=60),
                 min_samples: int = 10) -> None:
        if not 0.0 < rate <= 1.0:
            raise ValueError("rate must be in (0, 1]")
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
```

### Built-in fallbacks

```python
class RaiseError:
    async def on_rejected(self, stats, fn, args, kwargs):
        raise CircuitOpenError(stats)


class ReturnValue:
    def __init__(self, value: Any) -> None:
        self._value = value
    async def on_rejected(self, stats, fn, args, kwargs):
        return self._value


class CallFallback:
    """Dispatch to alternative async callable.
    If ``fallback_fn``'s signature accepts ``stats`` kwarg, it is passed."""
    def __init__(self, fallback_fn: Callable[..., Awaitable[Any]]) -> None:
        self._fn = fallback_fn
        sig = inspect.signature(fallback_fn)
        self._wants_stats = "stats" in sig.parameters

    async def on_rejected(self, stats, fn, args, kwargs):
        if self._wants_stats:
            return await self._fn(*args, stats=stats, **kwargs)
        return await self._fn(*args, **kwargs)


class EscalateToHITL:
    """Open ApprovalCard via channel; BLOCKS until human verdict.
    Returns whatever the channel returns."""
    def __init__(self, *,
                 channel: Any,                                    # HITLChannel duck-typed
                 card_factory: Callable[[BreakerStats], Any],
                 timeout: timedelta | None = None) -> None:
        self._channel = channel
        self._card_factory = card_factory
        self._timeout = timeout

    async def on_rejected(self, stats, fn, args, kwargs):
        payload = self._card_factory(stats)
        return await self._channel.request(payload, timeout=self._timeout)


class Chain:
    """Try each policy in order; return first non-raising result.
    Exceptions between attempts are logged + swallowed; final raises."""
    def __init__(self, *policies: FallbackPolicy) -> None:
        if not policies:
            raise ValueError("Chain requires at least one policy")
        self._policies = policies

    async def on_rejected(self, stats, fn, args, kwargs):
        last_exc: Exception | None = None
        for p in self._policies:
            try:
                return await p.on_rejected(stats, fn, args, kwargs)
            except Exception as exc:
                _log.exception("fallback policy %r failed (trying next)", type(p).__name__)
                last_exc = exc
        assert last_exc is not None
        raise last_exc
```

### Core `CircuitBreaker`

```python
class CircuitBreaker:
    """Protects async function invocations.

    Apps call .call(fn, *args, ctx=..., **kwargs). ctx (mapping) feeds
    scope_key to determine which sub-bucket tracks this invocation.
    One CircuitBreaker can multiplex many scopes (e.g., one breaker, many tools).
    """

    def __init__(
        self, *,
        threshold_factory: ThresholdFactory                = lambda: Consecutive(5),
        fallback:          FallbackPolicy                  = RaiseError(),
        scope_key:         ScopeKey                        = global_scope,
        recovery_after:    timedelta                       = timedelta(seconds=30),
        probe_max:         int                             = 1,
        is_failure_exc:    tuple[type[Exception], ...]     = (Exception,),
        ignored_exc:       tuple[type[Exception], ...]     = (asyncio.CancelledError,),
        is_success:        Callable[[Any], bool]           = lambda _r: True,
        name:              str                             = "circuit_breaker",
        clock:             Callable[[], datetime]          = lambda: datetime.now(UTC),
    ) -> None:
        if probe_max < 1:
            raise ValueError("probe_max must be >= 1")
        self._threshold_factory = threshold_factory
        self._fallback = fallback
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
        """Force CLOSED. None → reset all scopes."""
        targets = list(self._scopes.values()) if scope is None else [self._scopes[scope]]
        for bucket in targets:
            bucket.force_closed()
```

### `_ScopeBucket` — per-scope state machine

```python
class _ScopeBucket:
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

        # Execute outside lock — fn may be long.
        try:
            result = await fn(*args, **kwargs)
        except self._owner._ignored_exc:  # type: ignore[misc]
            raise
        except self._owner._is_failure_exc as exc:  # type: ignore[misc]
            async with self._lock:
                self._record(success=False, at=self._owner._clock())
            raise
        except Exception:
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
```

### Adapters

#### `as_capability(breaker)`

Wraps tool calls via pydantic-ai capability hook. Apps register the capability on their agent; the breaker tracks tool invocations under `per_tool_scope` (or any custom `scope_key` from the breaker).

```python
class _CBCapability(BallastCapability):
    name = "circuit_breaker"
    def __init__(self, breaker: CircuitBreaker) -> None:
        self._breaker = breaker
    # Tool-call wrapping is pydantic-ai version-specific; first cut wraps
    # via after_model_request inspection of tool calls if a true wrap_tool_call
    # hook is unavailable. Implementation chooses the path.
    ...

def as_capability(breaker: CircuitBreaker) -> BallastCapability:
    return _CBCapability(breaker)
```

#### `as_workflow_decorator(breaker, scope_ctx=None)`

```python
def as_workflow_decorator(
    breaker: CircuitBreaker, *,
    scope_ctx: Mapping[str, Any] | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            return await breaker.call(fn, *args, ctx=scope_ctx, **kwargs)
        return wrapper
    return deco
```

#### `as_step(breaker, wrapped) → BreakerStep`

```python
class BreakerStep:
    """Wraps any Step (PlanAndExecute) — failures trip per-step or shared scope."""
    def __init__(self, *, breaker: CircuitBreaker, wrapped: Step) -> None:
        self._breaker = breaker
        self._wrapped = wrapped

    async def execute(self, plan_input, dep_outputs, ctx):
        scope_ctx = {"step_id": ctx.step.id, "step_kind": ctx.step.kind}
        return await self._breaker.call(
            self._wrapped.execute, plan_input, dep_outputs, ctx,
            ctx=scope_ctx,
        )


def as_step(breaker: CircuitBreaker, wrapped: Step) -> BreakerStep:
    return BreakerStep(breaker=breaker, wrapped=wrapped)
```

## Data flow

```
[caller] await breaker.call(fn, *args, ctx={"tool_name": "search"})
    │
    ▼
scope = scope_key(ctx) → "tool:search"
bucket = self._scopes.setdefault(scope, _ScopeBucket(self, scope))
    │
    ▼ async with bucket._lock
state check:
  - if OPEN and now >= opened_at + recovery_after → state := HALF_OPEN, probe_attempts := 0
  - if OPEN (still in cooldown) → fallback.on_rejected(...)
  - if HALF_OPEN and probe_attempts >= probe_max → fallback.on_rejected(...)
  - if HALF_OPEN (probe allowed) → probe_attempts++
    │
    ▼ release lock, execute fn
result, success = await fn(*args, **kwargs), is_success(result)
exception:
  - in ignored_exc → re-raise unchanged
  - in is_failure_exc → record failure, re-raise
  - otherwise → re-raise unchanged
    │
    ▼ async with bucket._lock
record(success, at=clock())
  - threshold.on_outcome(success, at)
  - if HALF_OPEN: success → CLOSED + threshold.reset(); failure → OPEN
  - if CLOSED + threshold.trip(at): OPEN
return result
```

## Error handling

| Layer | Behaviour |
|---|---|
| `fn` raises in `ignored_exc` | Re-raised; no state change (CancelledError never trips CB) |
| `fn` raises in `is_failure_exc` | Failure recorded; trip check; original exc re-raised |
| `fn` raises other exception | Re-raised unchanged; NOT recorded (programmer error, not CB concern) |
| Breaker OPEN / probe denied | `fallback.on_rejected(stats, fn, args, kwargs)` |
| `RaiseError` fallback | `CircuitOpenError(stats)` raised |
| `CallFallback.fn` raises | Propagates (caller's contract) |
| `EscalateToHITL` timeout | `TimeoutError` from `Durable.recv_async` propagates |
| `Chain` fallback all fail | Last exception re-raised |

## Testing strategy

```
tests/resilience/circuit_breaker/
  test_state.py             # BreakerState enum, BreakerStats serialization
  test_thresholds.py        # Consecutive: trip after N, reset on success
                            # WindowedCount: deque eviction, window correctness
                            # WindowedRate: min_samples gate, rate calc
                            # Each policy: trip/reset/on_outcome semantics
                            # Factory isolation: two factories → two instances
  test_fallbacks.py         # RaiseError raises CircuitOpenError
                            # ReturnValue returns stored value
                            # CallFallback: with/without stats kwarg
                            # EscalateToHITL: blocking on channel.request
                            # Chain: ordering, exception isolation, final propagation
  test_breaker.py           # Core flows with mocked clock:
                            # - CLOSED → OPEN at threshold trip
                            # - OPEN call → fallback invoked
                            # - OPEN → HALF_OPEN after recovery_after (advance clock)
                            # - HALF_OPEN probe success → CLOSED, threshold reset
                            # - HALF_OPEN probe failure → OPEN, timer re-armed
                            # - HALF_OPEN probe_attempts > probe_max → reject
                            # - ignored_exc bypasses state changes
                            # - is_success(result)=False counts as failure
                            # - per-scope isolation (different scope_key results)
                            # - reset(scope) forces CLOSED for that scope only
                            # - reset() with no arg resets all scopes
                            # - stats() returns valid snapshot
  test_adapters_capability.py  # CB capability tracks tool calls per-name
  test_adapters_workflow.py    # Decorator: scope_ctx propagated; CB integrated
  test_adapters_step.py        # BreakerStep wraps Step, PlanAndExecute interop
  test_integration.py          # End-to-end:
                               # - CB + RaiseError → workflow fails
                               # - CB + CallFallback → workflow recovers
                               # - CB + Chain([CallFallback, EscalateToHITL]) → cascade
```

Time-mocking pattern: pass `clock: Callable[[], datetime] = lambda: datetime.now(UTC)` to `CircuitBreaker.__init__`. Tests pass a controllable clock object.

## Integration с существующими primitives

- **`BudgetGuard`** — ортогональны: budget = tokens, CB = failure rate. Может вместе сидеть в `capabilities=[]`. `BudgetExceeded` exception app may add to `is_failure_exc` so CB tripping triggers fallback (e.g., switch to cheaper model via `CallFallback`).
- **`GoalDriftDetector`** — drift verdict не считается failure (CB реагирует на exceptions/rejected outputs). Если апп хочет — `EmitDriftEvent` handler пишет в metric, отдельный CB по этому metric'у — follow-up.
- **`PlanAndExecute`** — нативно через `as_step(breaker, wrapped_step)`. Per-step или shared scope зависит от `scope_key` breaker'а.
- **`HITLChannel`** — нативно через `EscalateToHITL` fallback.
- **`@Durable.workflow`** — нативно через `as_workflow_decorator`. CB state живёт в process-памяти; на DBOS workflow replay CB перезапустится "холодным". Это известное ограничение first cut; persistent CB state — отдельная задача.

## Out of scope

- Persistent CB state (Redis / Postgres) — first cut in-memory.
- Bulkhead pattern.
- Rate limiter.
- Hedging.
- Per-scope distinct fallback / threshold within one breaker.
- Cluster-wide CB state synchronization.
- Frontend visualisation of CB state.
- Auto-tuning thresholds via historical data.

## Open questions for review

None — all design decisions resolved during brainstorm.

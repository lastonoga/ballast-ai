# 13. Resilience — circuit breakers

**Prerequisites:** [11-budget-and-loops.md](11-budget-and-loops.md).

## Introduction

`BudgetGuard` protects a single run from running too long. But what happens when *every* run starts hitting budget? Or when an upstream provider goes flaky and 30% of model calls time out? Or when one tool starts raising on every invocation? Per-run guards keep each individual run safe; they have no idea anything is systemically wrong.

That's the gap circuit breakers fill. A circuit breaker is *cross-run* — it observes failures across many calls and, once a threshold is crossed, "opens" and rejects subsequent calls without trying them. This protects the upstream system from being hammered while it recovers, and protects your users from waiting on calls that are almost certain to fail.

The framework's `CircuitBreaker` is a from-scratch implementation (no third-party library) tuned for agentic workloads: scope-aware (per-tool / per-step / per-anything buckets), composable with capabilities and workflows and PlanAndExecute steps via three adapters, and bridge-friendly with `BudgetGuard` and `GoalDriftDetector` so a per-run failure can count toward the cross-run breaker.

## The mental model

The classic state machine, three states:

```
CLOSED  ──────[threshold hit]─────► OPEN
   ▲                                  │
   │                                  ▼
   └──[probes succeed]──── HALF_OPEN ◄┘   (after recovery delay)
                              │
                              └──[probe fails]──► OPEN
```

- **CLOSED** — normal operation. Calls go through. Failures accumulate in a `ThresholdPolicy`.
- **OPEN** — threshold crossed. Calls are rejected by the `FallbackPolicy` (raise, return sentinel, escalate, etc.) without ever reaching the wrapped function.
- **HALF_OPEN** — recovery window elapsed. The breaker lets a few "probe" calls through. If probes succeed, transition back to CLOSED. If a probe fails, back to OPEN.

The breaker is *scope-aware*. You don't have one breaker per process — you have one breaker per *scope*, where scope is derived from call context. A common scope is "per tool," meaning each tool has its own bucket of failures; if `search_notes` goes down, the breaker for `search_notes` opens without affecting `create_note`.

## The simplest case

```python
from ballast.resilience.circuit_breaker import (
    CircuitBreaker,
    Consecutive,
    RaiseError,
)

breaker = CircuitBreaker(
    threshold_factory=lambda: Consecutive(max_failures=5),
    fallback=RaiseError(),
    recovery_after=timedelta(seconds=30),
)

# Direct use:
result = await breaker.call(lambda: my_flaky_async_fn(arg))
```

After 5 consecutive failures, the breaker opens. Subsequent `.call()` invocations raise `CircuitOpenError(stats)` without invoking `my_flaky_async_fn`. 30 seconds later, the breaker transitions to HALF_OPEN and lets one probe through.

Three things to notice:

- **`threshold_factory`, not `threshold`.** Each scope bucket gets its own threshold instance; the factory is called per-scope. This matters when the threshold has state (windowed counters).
- **`fallback` is a policy, not a value.** It's an object with a `.handle(stats) -> Any` method. `RaiseError()` raises; `ReturnValue(...)` returns; `EscalateToHITL(...)` opens a card.
- **`recovery_after`** controls when HALF_OPEN starts. Too short → you'll thrash. Too long → users wait longer than necessary. 30s-2min is typical.

## Threshold policies — when to open

Three built-in options:

### `Consecutive(max_failures=5)`

Trip after N failures in a row. Any success resets the counter. Right for systems where intermittent failures are normal — only sustained runs of failure should trip.

### `WindowedCount(max_failures=5, window=timedelta(seconds=60))`

Trip when ≥ N failures occur within the trailing window. Successes don't reset; old failures fall out as the window slides. Right for "more than 5 errors per minute is bad regardless of the in-between pattern."

### `WindowedRate(rate=0.5, window=timedelta(seconds=60), min_samples=10)`

Trip when failure rate ≥ `rate` over the window, gated by `min_samples` (don't trip on small samples). Right for high-throughput systems where you care about the *percentage* not the absolute count.

Pick based on your traffic shape: low-throughput (under 1 req/sec) → `Consecutive`. Medium-throughput → `WindowedCount`. High-throughput → `WindowedRate`.

## Fallback policies — what to do when open

Five built-in options:

### `RaiseError()`

Raises `CircuitOpenError(stats)`. The caller catches and decides. This is the most flexible because the caller has full context.

### `ReturnValue(value=...)`

Returns a sentinel. Right for read-paths where a cached / default response is better than an error.

### `CallFallback(fallback_fn=...)`

Calls an alternative async function. Right when you have a degraded mode — e.g., the breaker around your `enhanced_search` opens, fall back to `basic_search`. The fallback can take an optional `stats` kwarg if it wants context.

### `EscalateToHITL(channel=..., card_factory=..., timeout=...)`

Opens an approval card asking a human to make the call. The breaker waits for the verdict. Right for low-throughput high-stakes paths (the breaker for "send-money tool" opens → don't silently fail, ask a human).

### `Chain(*policies)`

Try each policy in order, swallowing exceptions until one succeeds. If all fail, raise the last exception. Right for `Chain(CallFallback(degraded_mode), ReturnValue(cached))` — try the fallback, fall back to a cached value.

## Scope keys — buckets

The `scope_key` function maps a call context to a string. Each unique string gets its own breaker bucket.

```python
def global_scope(ctx) -> str:
    return "global"

def per_tool_scope(ctx) -> str:
    return f"tool:{ctx.get('tool_name', 'unknown')}"

def per_step_scope(ctx) -> str:
    return f"step:{ctx.get('step_id', 'unknown')}"
```

The framework ships these three helpers. Custom scope keys are one-liners — return whatever string identifies the bucket you want.

A typical pattern: per-provider scoping for model calls (one breaker per OpenAI vs Anthropic vs local model), so an Anthropic outage doesn't take down OpenAI calls.

## The three adapters

`CircuitBreaker` is the core. Three adapters expose it on the framework's main deployment surfaces:

### `as_capability(breaker)`

Wraps an entire agent run. The breaker counts the run as a failure if the run raises (or if `is_success(result)` returns False — useful for treating low-confidence outputs as failures, see chapter 6):

```python
breaker = CircuitBreaker(
    threshold_factory=lambda: Consecutive(3),
    is_success=lambda r: isinstance(r, Scored) and r.confidence != "low",
)

agent = Agent(
    model=...,
    output_type=Scored[Fact],
    capabilities=[as_capability(breaker)],
)
```

### `as_workflow_decorator(breaker, *, scope_ctx=None)`

Wraps a `@Durable.workflow` function. Each invocation flows through the breaker.

```python
@Durable.workflow
@as_workflow_decorator(breaker)
async def my_workflow(input: dict) -> dict:
    ...
```

### `as_step(breaker, wrapped)`

Wraps a `PlanAndExecute` Step. The scope automatically includes `step_id` and `step_kind` so per-step breakers work without extra wiring:

```python
my_step = LLMStep(...)
guarded_step = as_step(breaker, my_step)

# Register and use in a PlanAndExecute DAG.
```

## Bridging to `BudgetGuard`

By default, the breaker treats *all* exceptions as failures. Sometimes you want it narrower — e.g., only count `BudgetExhausted` as a breaker failure, ignore everything else (those are application errors, not transport degradation):

```python
breaker = CircuitBreaker(
    threshold_factory=lambda: Consecutive(3),
    is_failure_exc=(BudgetExhausted,),   # only count budget exhaustion
    fallback=ReturnValue(degraded_answer),
)
```

Now: every run with a budget exhaustion increments the breaker. After 3 consecutive, the breaker opens. New runs short-circuit to the degraded answer without even trying. After `recovery_after`, the breaker probes again.

This is the cross-run signal you want when the model provider is degrading: rather than every user waiting through the full budget before giving up, the breaker fails them fast.

## `ignored_exc` — never count these

```python
breaker = CircuitBreaker(
    ...,
    ignored_exc=(asyncio.CancelledError, KeyboardInterrupt),
)
```

These exceptions pass through without affecting breaker state. The default already ignores `CancelledError` (a cancellation is not a system failure).

## `is_success` — outcome predicates

By default, "success" means "didn't raise." Override to treat specific *outcomes* as failures:

```python
breaker = CircuitBreaker(
    threshold_factory=lambda: Consecutive(5),
    is_success=lambda result: result.status == "ok",
)
```

This is what makes `Scored` integration clean — a low-confidence result is a "failure" without the run raising.

## Manual reset

When you've fixed the underlying problem and want the breaker to close immediately:

```python
breaker.reset()                       # all scopes
breaker.reset(scope="tool:search")    # one scope
```

Useful in incident response (closed an outage, want to bring traffic back without waiting for `recovery_after`).

## In-memory state limitation

The breaker's state lives in process memory. Across multiple replicas, each one has its own breaker state. Consequences:

- **3-replica deployment with `Consecutive(5)`** effectively means "5 failures per replica" = 15 failures total before all three are open. Be aware.
- **A replica restart resets that replica's breakers.** Hot restarts after a deploy mean fresh breakers; ongoing issues might "open again" minutes later.

For a shared breaker across replicas, you'd need a Redis or DB-backed state store — not currently shipped. The in-memory limitation is acceptable for most cases because: (a) replicas usually fail in correlated ways (one provider outage affects all of them), so per-replica state still trips reasonably together; (b) the breaker is meant to be a coarse safety mechanism, not a precise traffic shaper.

If you need shared state, the right move is to put the breaker in a *single dedicated* process (e.g., a sidecar) and route calls through it. Or wait until the framework ships a shared backend.

## Observability

Every state transition emits an event you can hook into for metrics:

```python
breaker = CircuitBreaker(
    ...,
    on_state_change=lambda scope, old, new, stats: metrics.event(
        "circuit_breaker", scope=scope, from_=old.value, to=new.value
    ),
)
```

Even without explicit instrumentation, `CircuitOpenError.stats` carries everything you need for postmortem analysis: scope, consecutive_failures, total_failures, opened_at, will_attempt_recovery_at, probe_attempts.

## Common mistakes

- **One global breaker for everything.** A single tool failure shouldn't take down unrelated tools. Use `per_tool_scope` (or similar) so failures stay localized.
- **`recovery_after` too short.** If you set it to 5 seconds and the upstream takes a minute to recover, you'll thrash between OPEN and HALF_OPEN. Start at 30s, lengthen if you see thrashing.
- **`Consecutive` for high-throughput systems.** With 1000 req/sec, 5 consecutive failures is a tiny fraction of a second — you'll trip on every transient blip. Use `WindowedRate` instead.
- **Catching `CircuitOpenError` and retrying immediately.** That's exactly what the breaker is there to prevent. If you must retry, do so with backoff much larger than `recovery_after`.
- **Forgetting `is_failure_exc` filtering.** Without it, every exception (including user-input validation errors) counts toward the breaker. Filter to transport/system errors only.

## What this chapter did NOT cover

- The `EscalateToHITL` fallback policy details — chapter 21.
- Composing with `BudgetGuard.snapshot()` for drift detection — chapter 12.
- `PlanAndExecute` step wrapping — chapter 18.
- Per-replica vs shared state design trade-offs — out of scope for now; covered in `docs/explanation/architecture-overview.md`.

## Where to go next

→ [14-patterns-intro.md](14-patterns-intro.md) — composable workflow shapes built on top of agents.

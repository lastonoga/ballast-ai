# How to add a Circuit Breaker to a tool

**Problem:** A flaky external API (search service, scraper, third-party LLM) is failing intermittently. Your agent retries blindly, burning budget and never giving up. You want repeated failures to open the breaker → block further calls for a cooldown → optionally fall back to a deterministic path.

**Solution:** Wrap the tool body with `CircuitBreaker.call(...)` or use `as_workflow_decorator`. Choose a `ThresholdPolicy` (when to open) and a `FallbackPolicy` (what to do when open).

## Minimum: wrap the tool body manually

```python
from datetime import timedelta
from ballast.resilience.circuit_breaker import (
    CircuitBreaker, Consecutive, RaiseError, per_tool_scope,
)

cb = CircuitBreaker(
    threshold_factory=lambda: Consecutive(3),    # 3 fails in a row → open
    recovery_after=timedelta(seconds=30),        # auto-attempt recovery after 30s
)

@notes_agent.tool
async def search_web(query: str) -> str:
    return await cb.call(_external_search, query, ctx={"tool_name": "search_web"})


async def _external_search(query: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"https://flaky.api/search?q={query}", timeout=5.0)
        r.raise_for_status()
        return r.text
```

When `_external_search` raises 3 times in a row, the breaker opens. The next call raises `CircuitOpenError` (HTTP 503). After 30 seconds, the breaker enters Half-Open: one probe is allowed; if it succeeds, the breaker closes; if it fails, the cooldown re-arms.

## Pick a better threshold for flaky APIs

`Consecutive` is brittle — one successful call resets it. For flaky APIs, prefer `WindowedRate`:

```python
from ballast.resilience.circuit_breaker import WindowedRate

cb = CircuitBreaker(
    threshold_factory=lambda: WindowedRate(
        rate=0.5,                              # open if 50%+ failures
        window=timedelta(seconds=60),          # in trailing 60s
        min_samples=10,                        # require at least 10 calls
    ),
)
```

Now the breaker opens when failure rate is genuinely high — not every time you hit 3 transient errors in a calm period.

## Fall back instead of raising

`RaiseError` (default) bubbles `CircuitOpenError` to the caller. For graceful degradation, use a fallback chain:

```python
from ballast.resilience.circuit_breaker import (
    CircuitBreaker, CallFallback, EscalateToHITL, Chain,
)

async def cached_search_fallback(query: str, *, stats):
    # Reuse last successful result from your cache
    return await my_cache.get_search(query) or "(no recent cache; try again later)"

cb = CircuitBreaker(
    fallback=Chain(
        CallFallback(cached_search_fallback),  # try cache first
        EscalateToHITL(                         # if that fails too: ask human
            channel=ui_card_channel,
            card_factory=lambda stats: ServiceDownCard(stats=stats),
            timeout=timedelta(minutes=15),
        ),
    ),
)
```

`Chain` tries each fallback in order; first non-raising result wins.

## Per-tool isolation (one breaker, many tools)

A single `CircuitBreaker` can multiplex many tools — use `scope_key`:

```python
cb = CircuitBreaker(
    threshold_factory=lambda: WindowedRate(0.5, timedelta(seconds=60), min_samples=10),
    scope_key=per_tool_scope,        # one bucket per tool name
)

@notes_agent.tool
async def search_web(query: str):
    return await cb.call(_external_search, query, ctx={"tool_name": "search_web"})

@notes_agent.tool
async def fetch_url(url: str):
    return await cb.call(_external_fetch, url, ctx={"tool_name": "fetch_url"})
```

If `search_web` flaps, only its bucket opens; `fetch_url` keeps working. `cb.stats("tool:search_web").state` shows the live state for dashboards.

## Decorator style for whole workflows

```python
from ballast.resilience.circuit_breaker import as_workflow_decorator
from ballast import Durable

@as_workflow_decorator(cb, scope_ctx={"workflow_name": "publish"})
@Durable.workflow()
async def publish_post(draft: PostDraft) -> str:
    ...
```

Same `CircuitBreaker` instance; the decorator wraps the workflow body.

## Inside a Plan-and-Execute DAG step

```python
from ballast.resilience.circuit_breaker import as_step
from ballast.patterns.plan_execute import LLMStep

protected_llm = as_step(cb, LLMStep(registry))
registry.register_step("llm", protected_llm)
```

Each step that the planner emits gets per-step CB scope (via `per_step_scope`).

## What counts as "failure"

By default any `Exception` (except `asyncio.CancelledError`) counts. Customize:

```python
cb = CircuitBreaker(
    is_failure_exc=(httpx.HTTPStatusError, asyncio.TimeoutError),   # only these
    ignored_exc=(asyncio.CancelledError, KeyboardInterrupt),         # never fail
    is_success=lambda r: r and r.get("status") == "ok",              # negate by result
)
```

`is_success` lets you treat a returned object as a failure (e.g. response with `status: "error"` field).

## Inspect state from a dashboard endpoint

```python
@app.get("/metrics/circuit-breakers")
async def cb_metrics():
    return {
        "search_web": cb.stats("tool:search_web").model_dump(mode="json"),
        "fetch_url":  cb.stats("tool:fetch_url").model_dump(mode="json"),
    }
```

`BreakerStats` includes state, counters, `opened_at`, `will_attempt_recovery_at`, probe attempts. Perfect for logfire or a custom dashboard.

## Manual reset

If you fixed the underlying API and want to skip the cooldown:

```python
cb.reset("tool:search_web")   # one scope
cb.reset()                     # all scopes
```

## Caveats

- **In-memory state.** CB state lives in process memory. On DBOS workflow replay, the breaker resets. For cluster-aware breaker state, you'd need a custom subclass backed by Redis — not currently in the framework.
- **`_ScopeBucket` uses an asyncio.Lock.** Calls within a scope are serialized; calls across scopes are independent. If you need maximum throughput for hundreds of scopes simultaneously, this is fine. If you need lock-free, use `global_scope`.
- **Don't combine CB with infinite retries inside the protected function** — the retries will burn through the threshold before the breaker can react.

## Related

- [cap-tokens-and-iterations.md](../cost-control/cap-tokens-and-iterations.md) — token + iteration caps
- [require-approval-for-dangerous-tools.md](../trust-and-safety/require-approval-for-dangerous-tools.md) — HITL channel for `EscalateToHITL` fallback
- Reference: `reference/resilience/circuit-breaker.md`
- Explanation: [article-pain-points.md](../../explanation/article-pain-points.md) #9

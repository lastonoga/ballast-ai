# 22. Observability

**Prerequisites:** [07-capabilities.md](07-capabilities.md), [08-running-an-app.md](08-running-an-app.md).

## Introduction

Classical observability — uptime, latency, error rate — answers "is the service healthy?" For an agentic app, that's not enough. The service can be perfectly healthy while answering every question wrong. An agent run with valid HTTP 200 responses, normal latency, and no exceptions can still be producing hallucinated facts and burning thousands of tokens per request. The service-level metrics don't see any of it.

What you need: per-run visibility into the LLM calls (prompts in, completions out, token counts), the tool dispatch, the capability state at each turn, and the cross-call patterns (which workflows, which steps, which retries). The framework's observability is built on top of [logfire](https://logfire.pydantic.dev/), the pydantic team's OpenTelemetry-backed observability product. It instruments pydantic-ai, FastAPI, DBOS, and asyncpg automatically; your job is to enable it and add app-specific spans where useful.

This chapter covers the configuration surface, the `@traced` decorator for adding your own spans, how OTel context flows across DBOS workflow boundaries, what to put in span attributes (and what to keep out), and how to use logfire's UI to actually answer the questions you'll have in production.

## The mental model

Logfire is OpenTelemetry under the hood. Every meaningful operation is a *span*: a named unit of work with start time, end time, attributes (key-value metadata), and parent-child relationships. Spans nest: a FastAPI request is a span; the agent call inside it is a child; the LLM call inside that is a grandchild; the tool calls each agent makes are siblings.

What you get end-to-end when wired correctly:

- HTTP request span → workflow span → agent.run spans → model call spans (prompt, completion, tokens, model name) → tool call spans → DB query spans.
- One trace per user request. Click any span; see its full subtree. See which agent took longest, which tool was called how many times, what the prompts looked like.

The framework's contribution: it pre-instruments pydantic-ai (model calls and tool dispatch), FastAPI (request handling), DBOS (workflow + step boundaries), and asyncpg (database queries). Your contribution: enable logfire, add your own spans for app-specific operations (`@traced`), and put useful attributes on them.

## Enabling logfire

The simplest path:

```python
from ballast import Ballast, BallastSettings

settings = BallastSettings()    # reads LOGFIRE_TOKEN from env

app = (
    Ballast(settings)
    .with_dbos()
    .with_observability()   # initializes logfire + auto-instrumentation
    .fastapi(cors="dev")
)
```

`with_observability()` reads `settings.observability`:

```
BALLAST_OBSERVABILITY__LOGFIRE_TOKEN=...
BALLAST_OBSERVABILITY__SERVICE_NAME=my-app
BALLAST_OBSERVABILITY__ENVIRONMENT=production
BALLAST_OBSERVABILITY__INSTRUMENT_FASTAPI=true
BALLAST_OBSERVABILITY__INSTRUMENT_ASYNCPG=true
BALLAST_OBSERVABILITY__INSTRUMENT_PYDANTIC_AI=true
```

Without `LOGFIRE_TOKEN`, observability gracefully degrades to no-op. You can develop without a logfire account; you just don't get the SaaS UI.

## What you get automatically

After enabling:

- **Every pydantic-ai model call** becomes a span with attributes: `model.name`, `model.provider`, `usage.input_tokens`, `usage.output_tokens`, the prompt, the completion (configurable).
- **Every tool call** becomes a child span: `tool.name`, the arguments (as JSON), the result.
- **Every FastAPI request** becomes the root span: `http.method`, `http.target`, `http.status_code`, latency.
- **Every DBOS workflow / step** becomes a span: `dbos.workflow_id`, `dbos.step_name`, the inputs/outputs (if `INSTRUMENT_DBOS=true`).
- **Every asyncpg query** becomes a span: SQL text (parameterized), latency, row count.

These nest correctly: the HTTP request span is the parent of the workflow, which is the parent of the agent run, which is the parent of the model calls and tool calls. One coherent trace per request.

## The `@traced` decorator for app-side spans

When you have an app-side function that's meaningful to see in traces, wrap with `@traced`:

```python
from ballast.observability import traced

@traced("my_app.compute_recommendations")
async def compute_recommendations(user_id: str, filters: dict) -> list[Recommendation]:
    ...

# Or with attrs from arguments:
@traced(
    "my_app.compute_recommendations",
    attrs=lambda user_id, filters: {"user_id": user_id, "filter_count": len(filters)},
)
async def compute_recommendations(user_id, filters):
    ...
```

If logfire isn't installed, `@traced` is a no-op. So you can sprinkle it wherever and it doesn't add overhead in dev/test without logfire.

The `attrs` callable receives the wrapped function's arguments and returns a dict merged into the span's attributes. Keep it cheap — it runs on every call.

## What to put in span attributes

Good attributes (high-signal, low-cardinality):

- IDs (user, tenant, thread, workflow)
- Operation names, model names, tool names
- Counts (item count, retry count, branch count)
- Outcome flags (success/failure, decision, status)

Bad attributes (high-cardinality, low-signal, or PII):

- Full message bodies (use logfire's prompt logging if you need them, with sampling)
- Raw user input (potential PII)
- Random UUIDs that don't help filtering
- Large nested data structures

Logfire's UI lets you filter by attributes; cardinality matters for query performance. `user_id` is fine if you have thousands of users; `prompt_text` as an attribute is bad even if you have only one prompt.

## OTel context across DBOS boundaries

DBOS workflows run in isolated execution contexts — by default, OTel's active span context doesn't propagate across the DBOS boundary, which would break the trace tree (the workflow span would be an orphan instead of a child of the request span).

The framework's `Durable.workflow` / `Durable.step` decorators *automatically* inject and re-attach the OTel carrier:

```python
@Durable.workflow
async def my_workflow(input):
    # When this is called from inside an HTTP request,
    # the workflow span is a child of the request span — automatically.
    ...
```

`Durable.enqueue` and `Durable.start_workflow` also inject the carrier so async-launched workflows still nest correctly under the caller's trace.

You don't have to think about this for the shipped patterns. If you write your own pattern or use `DBOS.workflow` directly (bypassing the `Durable` facade), use `ballast.observability.otel_carrier.inject_otel_carrier()` / `otel_context_from(carrier)` to manage it.

## Cost extraction

LLM provider responses include token counts and (sometimes) cost. The framework extracts these into span attributes so you can sum spend across traces:

- `usage.input_tokens`, `usage.output_tokens` — standard for all providers.
- `usage.cost_usd` — when provider response includes it (OpenRouter does; OpenAI requires post-hoc calculation).

In the logfire UI, you can write a query like "sum(usage.cost_usd) where user_id = X and timestamp > yesterday" to see per-user spend.

## Cross-service trace correlation

If your app makes HTTP calls to other services, propagating the OTel context lets you see one trace span both services. The framework's `Durable.enqueue` / `start_workflow` inject the W3C `traceparent` header for outbound HTTP calls when you use the standard `httpx` / `aiohttp` clients (which logfire instruments).

For raw socket connections or non-instrumented clients, use:

```python
from ballast.observability import otel_carrier

carrier = otel_carrier.inject_otel_carrier()
# Pass `carrier` as headers / metadata to the downstream call
```

The downstream service extracts via `otel_carrier.otel_context_from(carrier)`.

## Sampling for high-volume production

Logfire's default is "record everything," which works in dev and low-volume prod. At scale (thousands of requests / sec), you sample:

```python
import logfire

logfire.configure(
    ...,
    sampling=logfire.SamplingOptions(
        head_sample_rate=0.1,   # 10% of all traces
    ),
)
```

The framework respects whatever sampling you configure. Common patterns:

- **Sample by user.** 100% of internal team's traces; 1% of public traffic.
- **Sample by outcome.** 100% of errors; 1% of successes.
- **Sample by route.** 100% of `/healthz`; 10% of `/chat`.

The logfire docs cover the configuration in depth.

## Logs vs spans

Two different things:

- **Spans** are units of work with structured attributes. Best for "what happened over time" — every operation is a span.
- **Logs** (lines) are point-in-time messages. Best for "this notable thing happened" — warnings, errors, audit events.

Don't try to use logs for everything; use spans for operations and logs for events within them. Logfire shows logs inline with the span tree, so a warning logged inside `compute_recommendations` shows up under that span.

## Local development without a logfire account

Two options:

1. **Skip observability entirely.** `.with_observability()` without `LOGFIRE_TOKEN` is a no-op; nothing is sent.
2. **Use the local logfire CLI.** `logfire dev` (from the logfire SDK) starts a local UI for inspecting traces during dev without sending to the SaaS.

Either is fine. The framework doesn't require a logfire account; the integration is opt-in.

## Bridging to an OTel collector instead of logfire SaaS

Logfire writes OTLP; you can configure it to send to your own OTel collector instead of (or in addition to) the SaaS:

```python
logfire.configure(
    send_to_logfire="if-token-present",   # SaaS if token set; else local
    additional_span_processors=[
        BatchSpanProcessor(OTLPSpanExporter(endpoint="https://my-otel-collector/v1/traces")),
    ],
)
```

The framework's spans then show up in your own Honeycomb / Datadog / Tempo / Jaeger setup with the same structure. You give up the logfire UI but gain self-hosted control.

## Common mistakes

- **No observability at all.** First production incident, you'll wish you had it. Wire it on day one with a free logfire tier.
- **High-cardinality attributes.** UUIDs, timestamps, free-text — they break logfire's query performance. Use IDs that filter to small groups.
- **PII in spans.** Once it's in the trace, it's there. Sanitize at the boundary; never put raw user content into attributes without redaction.
- **`@traced` on every function.** Span explosion. Add traces at meaningful boundaries — request handlers, pattern entry points, expensive operations — not on every helper.
- **Trace correlation broken because of an un-instrumented client.** If your HTTP client isn't on logfire's auto-instrumentation list, propagate the carrier manually.

## What this chapter did NOT cover

- The exact logfire UI features — see logfire's docs.
- Per-pattern span attributes — covered in each pattern's chapter.
- Eval-driven quality dashboards — chapter 23.
- DBOS workflow inspector tree view — chapter 24 + the `/dbos` route in your app.

## Where to go next

→ [23-evals.md](23-evals.md) — programmatic quality evaluation.

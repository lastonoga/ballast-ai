# How to add logfire tracing

**Problem:** Your agent runs in production. When something goes wrong (drift, infinite loop, tool failure, slow request), you can't tell which step caused it. You need end-to-end traces: workflow → agent.run → each tool call → each LLM call → token usage + latency.

**Solution:** `logfire` integration is baked in. One `logfire.configure()` call at startup; framework primitives auto-emit spans + structured attributes. Add `@traced(...)` to your own functions for custom spans.

## Minimum

### 1. Install + configure

```bash
uv add logfire
```

`main.py`:
```python
import logfire

logfire.configure(
    token=settings.logfire_token,        # from logfire.dev
    service_name="my-agent-app",
    service_version="1.0.0",
)
# Optional: capture pydantic-ai spans, FastAPI requests, asyncpg queries
logfire.instrument_pydantic_ai()
logfire.instrument_fastapi(app)
logfire.instrument_asyncpg()
```

That's the whole setup. Framework primitives that already use logfire spans:
- `Ballast.fastapi_app()` instruments routes
- `@Durable.workflow` / `@Durable.step` produce DBOS spans
- pydantic-ai's agent loop produces model-call spans
- `@traced` (Ballast's wrapper) for custom workflow / pattern phases
- All `BallastCapability` hooks → spans

## Add custom spans to your code

```python
from ballast import traced


@traced(name="my_custom_step", attributes={"step.type": "data-prep"})
async def prepare_input(data: dict) -> str:
    # ... heavy work
    return cleaned
```

Spans appear in logfire with the name + attributes + duration. Nested within whatever parent span is active.

## Span attributes

`@traced(...)` accepts:
- `name: str` — span name
- `attributes: dict[str, Any]` — static attributes (set at decoration time)
- Inside the function: `logfire.span(...).set_attribute(key, value)` for dynamic

```python
from ballast import traced
import logfire


@traced(name="fetch_user")
async def fetch_user(user_id: str) -> User:
    with logfire.span("db.query") as span:
        span.set_attribute("user_id", user_id)
        user = await db.fetch(user_id)
        span.set_attribute("user.tier", user.tier)
        return user
```

## Trace patterns + capabilities

Already done — you don't need to add anything:

```python
mr = MapReduce(map_agent=extractor, reduce_agent=synthesizer)
# Auto-spans: mapreduce.run, mapreduce.map_one (per chunk), mapreduce.reduce
result = await mr.run(chunks)
```

```python
agent = Agent(model=..., capabilities=[BudgetGuard(...), GoalDriftDetector(...)])
# Auto-spans: agent.run, model.request, capability.budget_guard.after_model_request, ...
```

logfire shows the full tree in its UI.

## Cost extraction

The framework extracts token costs from LLM responses for spans:

```python
from ballast import OpenRouterCostExtractor, configure_cost_extractors


configure_cost_extractors(
    OpenRouterCostExtractor(),
    # ... add OpenAIDetailsCostExtractor, AnthropicCostExtractor, etc.
)
```

Now every model-call span carries `cost.input_usd`, `cost.output_usd`, `cost.total_usd`. Build dashboards / budget alerts on top.

## Cross-service trace correlation (OTel context)

If your agent calls downstream services that ALSO emit OTel spans, propagate context:

```python
from ballast.observability import inject_otel_headers


async def call_downstream(payload):
    headers = inject_otel_headers({})
    async with httpx.AsyncClient() as client:
        await client.post("https://other-service/api", json=payload, headers=headers)
```

On the downstream side, `extract_otel_headers(request.headers)` re-attaches the context. Spans link across services in logfire.

## Span on pattern progress

`DivergentConvergent`'s `on_progress` events automatically become attributes on the pattern's span:

```python
pattern = DivergentConvergent(
    divergent_agent=...,
    convergent_agent=...,
    branch_count=8,
    on_progress=my_callback,            # also auto-mirrored to span events
)
```

`logfire` shows branch lifecycle events inline.

## DBOS span naming

DBOS uses `workflow.{function_name}` and `step.{function_name}` by default. To customize:

```python
@Durable.workflow(name="research_publish_pipeline")
async def my_workflow(input): ...
```

Span name in logfire becomes `workflow.research_publish_pipeline`.

## Sampling

For high-volume production, sample to control cost:

```python
import logfire

logfire.configure(
    token=settings.logfire_token,
    trace_sample_rate=0.1,        # 10% of root traces
)
```

`@traced` spans within a sampled trace are always captured. Sampling is at the trace level, not the span level — keeps span trees coherent.

## Log lines vs spans

Use `logfire.info`/`warning`/`error` for log lines that DON'T need timing:

```python
import logfire


@agent.tool
async def search_web(ctx, query: str) -> str:
    logfire.info("searching", query=query, user_id=ctx.deps.user_id)
    return await _search(query)
```

These appear in the span timeline as events; no span overhead.

## Local dev without logfire account

```python
import logfire

logfire.configure(
    send_to_logfire=False,         # don't ship to logfire.dev
    console=logfire.ConsoleOptions(colors="always"),
)
```

Now spans print to your terminal — useful for local development.

## Caveats

- **Don't put PII in attributes.** Attributes are visible in logfire UI to anyone with project access. Use `logfire.hash_value(...)` or redact before attaching.
- **`@traced` adds a function-call boundary.** Don't decorate trivially-cheap functions (called 10k+ times in a tight loop) — span overhead becomes meaningful.
- **`logfire.instrument_*` calls must be after `logfire.configure()`.** Order matters at startup.
- **Don't double-instrument.** Calling `instrument_pydantic_ai()` twice creates duplicate spans.

## Bridge to OTel

If your org uses OTel collector instead of logfire SaaS:

```python
import logfire
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


tracer_provider = TracerProvider()
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4318"))
)
trace.set_tracer_provider(tracer_provider)

logfire.configure(
    send_to_logfire=False,
    additional_span_processors=[],
)
```

Logfire's `@instrument_*` decorators work the same; spans now flow to your OTel pipeline.

## Related

- [run-llm-judge-evaluation.md](run-llm-judge-evaluation.md) — `JudgeAfterRun` writes verdicts as span attributes
- [build-eval-dataset-from-traces.md](build-eval-dataset-from-traces.md) — replay traces as eval cases
- Reference: `reference/observability/traced.md`
- Reference: `reference/observability/cost-extractors.md`

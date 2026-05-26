# Customization Everywhere

> Every framework opinion is a Protocol or a callable hook. The framework ships sensible defaults; apps swap any of them. No metaclass magic, no plugin systems — just typed contracts.

## The rule

For every decision point inside Ballast, there are two things:
1. A `Protocol` (or callable type alias) defining the contract.
2. One or more built-in implementations registered as defaults.

Apps that need different behavior write their own Protocol implementation and pass it through the constructor / fluent setter. Built-ins remain available for everyone else.

## Composition surfaces

### Patterns: callable + agent slots

`MapReduce`, `Reflection`, `PlanAndExecute` etc. accept **agents** OR **plain callables** in their step slots. Apps choose granularity:

```python
# Plain callable map_step — pure Python logic
mr = MapReduce(map_step=my_extract_fn, reduce_step=my_synthesize_fn)

# Agent map_step — LLM dispatch per item
mr = MapReduce(map_agent=extractor_agent, reduce_agent=synthesizer_agent)

# Mix
mr = MapReduce(map_agent=extractor_agent, reduce_step=deterministic_reducer)
```

### Capabilities: stackable, ordered, per-run isolated

`BallastCapability` subclasses stack in order on `Agent(capabilities=[...])`. Each gets its own per-run clone via `for_run(ctx)`. No singleton state across runs.

Apps write their own:

```python
class MyCustomCapability(BallastCapability):
    name = "my_custom"

    async def for_run(self, ctx): return self.__class__()    # isolated per run

    async def after_model_request(self, ctx, *, request_context, response):
        # Inspect / mutate / log
        return response
```

### Resilience: Protocol per dimension

`CircuitBreaker` has FOUR pluggable dimensions:

| Dimension | Protocol | Built-ins |
|---|---|---|
| When to trip | `ThresholdPolicy` | `Consecutive`, `WindowedCount`, `WindowedRate` |
| What to do on rejection | `FallbackPolicy` | `RaiseError`, `ReturnValue`, `CallFallback`, `EscalateToHITL`, `Chain` |
| What scope counts as "one breaker" | `ScopeKey` (callable) | `global_scope`, `per_tool_scope`, `per_step_scope` |
| What counts as a "failure" | `is_failure_exc` + `is_success` predicates | configurable kwargs |

Apps mix and match:

```python
cb = CircuitBreaker(
    threshold_factory=lambda: WindowedRate(rate=0.4, window=timedelta(seconds=60), min_samples=20),
    fallback=Chain(
        CallFallback(use_cached_response),
        EscalateToHITL(channel=my_channel, card_factory=ServiceDownCard),
    ),
    scope_key=per_tool_scope,
    is_success=lambda r: getattr(r, "confidence", "high") != "low",
)
```

### CoALA: 4 phases, override what you need

`CoALABase` provides:
- `observe` — defaults to identity
- `retrieve` — abstract (must override)
- `act` — abstract (must override)
- `learn` — defaults to no-op

```python
class MyUnit(CoALABase[Input, Observation, Context, Output]):
    async def retrieve(self, obs):       # only override what you need
        return await my_kb.search(obs.intent)
    async def act(self, obs, ctx):
        return await my_agent.run(prompt_with(ctx))
    # observe + learn use defaults
```

Then choose deployment surface:
```python
agent.tools = [as_tool(MyUnit())]                  # one tool
flow = as_workflow(MyUnit())                       # durable workflow
agent.capabilities = [as_capability(MyUnit())]     # agent capability
```

### Goal Drift: 5 Protocols × built-ins

| What | Protocol | Built-ins |
|---|---|---|
| When to judge | `DriftCheckStrategy` | `AfterEveryStep`, `EveryNToolCalls(n)`, `EveryNSteps(n)`, `Periodic(seconds)`, `OnBudgetThreshold`, `Compose(*)` |
| What slice to show judge | `TraceWindow` | `FullTrace`, `LastNMessages(n)`, `SinceLastUserMessage`, `TokenBudgetWindow(max)` |
| Where original goal comes from | `GoalSource` | `FirstUserMessage`, `LastUserMessage`, `WorkflowInput`, `ExplicitGoal(text)` |
| How to ask judge | `PromptBuilder` | `DefaultPromptBuilder` |
| What to do on drift | `DriftHandler` | `LogOnly`, `EmitDriftEvent`, `RaiseDriftError`, `EscalateToHITL`, `Compose(*)` |

```python
engine = DriftEngine(
    strategy=Compose(EveryNToolCalls(3), OnBudgetThreshold(fraction=0.7, budget_fn=...)),
    window=TokenBudgetWindow(max_tokens=2000),
    goal_source=Summarized(my_summarizer_agent, every_n=20),    # custom summary-based
    prompt=MyCustomPromptBuilder(),
    judge=make_default_judge(model="anthropic:claude-3-5-haiku"),
    handlers=[
        LogOnly(),
        EmitDriftEvent(sink=my_metrics_sink),
        EscalateToHITL(channel=my_channel, card_factory=DriftCard),
    ],
)
```

### Plan-and-Execute: Step Protocol + StepRegistry

`Step` Protocol with one method `execute(plan_input, dep_outputs, ctx)`. Built-in implementations:
- `LLMStep` — dispatches to a registered Agent with prompt template
- `CallableStep` — dispatches to a registered async function
- `UnitStep` — dispatches to a registered CoALAUnit
- `WorkflowStep` — dispatches to a registered Durable.workflow

Apps register their own Step kinds:

```python
class MyCustomStep:
    """Apps can implement Step Protocol for any custom dispatch logic."""
    async def execute(self, plan_input, dep_outputs, ctx):
        # Whatever
        return my_result

registry = StepRegistry.with_defaults()
registry.register_step("my_kind", MyCustomStep())
# Now planner can emit PlannedStep(kind="my_kind", ...)
```

### Quality: typed wrappers with generic params

`Scored[T, ConfidenceT]` — generic in BOTH the wrapped type AND the confidence type. Default `ConfidenceT = Literal["low", "medium", "high"]`; apps override:

```python
Scored[Fact, int]                              # 1-5 numeric scale
Scored[Fact, Literal["safe", "uncertain"]]     # binary
```

Helpers (`filter_by_min_confidence`, `rank_by_confidence`, `aggregate_by_confidence`) work with the default Literal; custom `ConfidenceT` apps write their own helpers.

### HITL: channel Protocol with multiple impls

`HITLChannel` Protocol with one method:
```python
async def request(self, payload: InT, *, timeout: timedelta | None = None) -> VerdictT: ...
```

Built-ins:
- `UICardChannel` — REST `/approvals` panel via SSE
- `ThreadChannel` — in-chat marker (helper-message style)

Apps write their own (e.g. Slack channel, email channel, Telegram bot channel):

```python
class SlackApprovalChannel(DBOSHITLChannel[MyPayload, MyVerdict]):
    async def deliver(self, *, request_id, workflow_id, respond_topic, payload):
        await slack_client.post(payload.to_slack_block(), respond_to=respond_topic)

    async def decode_verdict(self, raw):
        return MyVerdict.model_validate(raw)
```

### Grounded: pluggable repositories + Selector

`Ref[T].hydrate(repo)` accepts any repo Protocol with `async def get(id) -> T | None`. `Selector` (planned) lets apps wire query DSLs.

### Persistence: swap any repository

`ThreadRepository`, `ApprovalCardRepository`, etc. are Protocols. Built-ins: `InMemoryThreadRepository`, `SqlThreadRepository`. Apps wire whatever:

```python
ballast.with_thread_repo(MyMongoThreadRepo())
ballast.with_approval_repo(MyRedisApprovalRepo())
```

## What ISN'T pluggable (and why)

A few things are intentionally hardcoded:
- **DBOS workflow primitives** — we use DBOS; swapping the durable backend is a fork-level change, not a Protocol.
- **pydantic v2 for data contracts** — used pervasively; replacing it is also fork-level.
- **pydantic-ai Agent / Tool shape** — we wrap the AbstractCapability surface; swapping would require rewriting most of the framework.
- **The HTTP layer (FastAPI)** — `Ballast.fastapi_app()` returns a FastAPI instance. Apps that want a different framework can build routes manually using `build_*_router` functions; just not via the high-level builder.

These are the framework's "non-negotiable" choices. Everything else is a Protocol.

## Pattern: "framework provides default, app overrides via constructor kwarg"

Almost every Ballast component follows this shape:

```python
def __init__(self, *, my_thing: MyProtocol = _default_my_thing_factory(), ...):
    self._my_thing = my_thing
```

So apps that want defaults write `MyComponent()`; apps that want custom write `MyComponent(my_thing=MyImpl())`. No global registries, no metaclass tricks.

## See also

- [why-ballast.md](why-ballast.md) — mission + design philosophy
- [architecture-overview.md](architecture-overview.md) — layer responsibilities
- [article-pain-points.md](article-pain-points.md) — every solution → what's pluggable in it

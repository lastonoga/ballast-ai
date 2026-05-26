# How to add Goal Drift Detection

**Problem:** Your agent works on a long-running task. Over many tool calls / refinements, it gradually loses sight of the original user goal — reasoning later in the trace stops referencing the initial intent. The agent "succeeds" (responds 200 OK) but solves the wrong problem.

**Solution:** `GoalDriftDetector` capability — periodically fires an LLM judge against the original goal + recent trace; if drift is detected, emits a thread event / escalates to HITL / raises an error.

## Minimum

```python
from ballast import GoalDriftDetector
from ballast.drift import (
    DriftEngine,
    EveryNToolCalls, LastNMessages, FirstUserMessage,
    DefaultPromptBuilder, make_default_judge,
    EmitDriftEvent,
)


async def thread_event_sink(event_name: str, payload: dict) -> None:
    # Wire to your thread-event publisher
    await broadcaster.send(event_name, payload)


detector = GoalDriftDetector(
    engine=DriftEngine(
        strategy=EveryNToolCalls(5),                  # check every 5 tool calls
        window=LastNMessages(10),                     # show judge last 10 msgs
        goal_source=FirstUserMessage(),               # original = first user prompt
        prompt=DefaultPromptBuilder(),
        judge=make_default_judge(),                   # cheap judge model
        handlers=[EmitDriftEvent(sink=thread_event_sink)],
    ),
)

agent = Agent(model=..., capabilities=[detector])
```

That's it. The judge fires every 5 tool calls. When drift is detected, a `goal_drift` thread event is emitted to your UI/log/metrics sink.

## What the judge sees

The judge agent receives:
- **Goal** — the first user message (per `FirstUserMessage` source)
- **Recent trace** — last 10 messages (per `LastNMessages(10)` window)
- **Prompt** — "Has the agent drifted from the goal? Reply with structured verdict."

The judge returns `DefaultDriftVerdict(should_interrupt: bool, reason: str, score: float, category: Literal["on_track", "loose", "drifted"], suggested_action: str | None)`.

If `should_interrupt=True`, configured handlers fire (in your case: emit a thread event).

## Pick when to check

Five built-in strategies, all pluggable via `DriftCheckStrategy` Protocol:

```python
from ballast.drift import (
    AfterEveryStep,                                       # every LLM step
    EveryNToolCalls, EveryNSteps,
    Periodic,                                             # by wall time
    OnBudgetThreshold,                                    # at 50% of budget
    ComposeStrategy,                                      # OR-combine
)

detector = GoalDriftDetector(engine=DriftEngine(
    strategy=ComposeStrategy(
        EveryNToolCalls(5),                               # baseline cadence
        OnBudgetThreshold(fraction=0.7, budget_fn=lambda: budget_guard.snapshot()["input_tokens"] / 20_000),
    ),
    ...,
))
```

## Pick where the original goal comes from

```python
from ballast.drift import (
    FirstUserMessage,            # long-running session — anchor on first prompt
    LastUserMessage,             # per-turn — anchor on current ask
    WorkflowInput,                # for @with_drift_monitor — anchor on workflow input
    ExplicitGoal,                 # statically pinned at construction
)
```

For multi-turn chat where the user might change topic, use `LastUserMessage`. For research-style tasks, use `FirstUserMessage`. For a workflow where the input encodes the goal, use `WorkflowInput`.

## Pick how to react

Five built-in handlers, all pluggable via `DriftHandler` Protocol:

```python
from ballast.drift import (
    LogOnly,                       # silent warn-log
    EmitDriftEvent,                # thread event → UI / metrics
    RaiseDriftError,               # hard fail — DBOS catches
    EscalateToHITL,                # open ApprovalCard for human decision
    ComposeHandler,                # chain multiple
)

handlers=[
    LogOnly(),                                       # for observability always
    EmitDriftEvent(sink=publish_to_thread),          # for UI awareness
    EscalateToHITL(                                  # for high-stakes scenarios
        channel=ui_card_channel,
        card_factory=lambda v: DriftCard(reason=v.reason),
    ),
]
```

For low-risk agents, `EmitDriftEvent` alone is enough (the user sees a banner; can interrupt manually). For high-risk (financial / publishing), add `EscalateToHITL` so the human MUST decide before the agent continues. For pure CI/CD/batch flows, `RaiseDriftError` lets DBOS handle it with retries / dead-letter queue.

## Bridge from BudgetGuard

`OnBudgetThreshold` strategy reads from `DriftContext.metadata["budget"]`. Wire `BudgetGuard.snapshot()` into the detector's `metadata_provider`:

```python
from ballast import BudgetGuard, GoalDriftDetector
from ballast.drift import OnBudgetThreshold

budget = BudgetGuard(max_iterations=20, max_input_tokens=50_000)

def attach_budget(ctx, request_context):
    # Read the per-run BudgetGuard clone via your own mechanism
    # (apps wire this — there's no automatic discovery yet)
    for cap in agent.capabilities:
        if isinstance(cap, BudgetGuard):
            return {"budget": cap.snapshot()}
    return {}

detector = GoalDriftDetector(
    engine=DriftEngine(
        strategy=OnBudgetThreshold(
            fraction=0.5,
            budget_fn=lambda: budget.snapshot()["input_tokens"] / 50_000,
        ),
        ...
    ),
    metadata_provider=attach_budget,
)
```

The detector fires when budget consumption crosses 50% — earlier than a pure step-count check.

## On a workflow body (no agent)

For long-running `@Durable.workflow` flows that don't have an agent loop, use the decorator:

```python
from ballast import with_drift_monitor

@with_drift_monitor(engine=DriftEngine(
    strategy=Periodic(seconds=60),
    goal_source=WorkflowInput(),
    window=...,                              # MUST be custom — default windows return [] when no agent loop
    ...,
))
@Durable.workflow()
async def long_research_flow(input: ResearchInput): ...
```

**Known limitation:** built-in `TraceWindow` impls return `[]` outside an agent loop. Apps that want workflow drift detection must supply a custom `TraceWindow` that reads state from a database / shared scratchpad / etc.

## Custom verdict shape

If you want richer verdicts (e.g. tagging the drift with categories your dashboards understand):

```python
from ballast.drift import DriftVerdictBase

class MyDriftVerdict(DriftVerdictBase):
    drift_tags: list[Literal["scope-creep", "off-topic", "goal-change"]]
    user_intent_changed: bool

detector = GoalDriftDetector(
    engine=DriftEngine(
        verdict_model=MyDriftVerdict,
        judge=Agent(model=..., output_type=MyDriftVerdict, system_prompt="..."),
        ...,
    ),
)
```

Framework reads only `should_interrupt + reason`; everything else is for your handlers + dashboards.

## Caveats

- **Adds LLM cost.** Every `should_check=True` triggers an extra LLM call. Use cheap judge models (`gpt-4o-mini`, `claude-haiku-4-5`) and tune your strategy interval.
- **Fail-safe by design.** Judge exceptions are swallowed and logged. Drift detection itself never breaks the user-facing reply.
- **In-memory state.** Counters live on the per-run clone (via `for_run`) — DBOS workflow replay starts the counter fresh, but the agent's full trace is already in `request_context.messages` so the judge still has correct input.

## Related

- [handle-flaky-external-api.md](../reliability/handle-flaky-external-api.md) — resilience for individual tools
- [cap-tokens-and-iterations.md](cap-tokens-and-iterations.md) — bridge via `OnBudgetThreshold`
- Reference: `reference/capabilities/goal-drift-detector.md`
- Explanation: [article-pain-points.md](../../explanation/article-pain-points.md) #19

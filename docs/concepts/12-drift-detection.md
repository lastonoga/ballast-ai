# 12. Drift detection

**Prerequisites:** [07-capabilities.md](07-capabilities.md), [11-budget-and-loops.md](11-budget-and-loops.md).

## Introduction

`BudgetGuard` catches an agent that runs forever. `SemanticLoopDetector` catches one that keeps saying the same thing. Neither catches the worst failure mode of all: an agent that stays busy, doesn't loop, doesn't exhaust its budget, and produces a confident final answer — to the wrong question.

Goal drift is the most expensive class of agent failure because it's invisible from the inside. The model is "working." Logs look healthy. Tools return data. The output validates. Only when a human reads the answer do they notice: "this isn't what I asked for at all." By then you've spent the tokens, executed any side-effecting tools, and shipped a wrong answer to a user.

The framework's answer is `GoalDriftDetector`: a capability that periodically asks an LLM judge "does the recent trace still address the original goal?" and takes action when the judge says no. The whole thing is built on five replaceable Protocols so you can wire it however your app needs — when to fire, what slice of trace to show, what counts as the goal, how to ask, what to do on a positive verdict.

## The mental model

Drift detection is *out-of-band evaluation*. You don't ask the agent itself "are you still on task?" — agents are bad at this self-assessment. You spin up a separate judge model whose only job is to look at the trace and answer one question: does this still address the original goal?

The judge runs asynchronously. The user-facing agent never blocks on it (a slow judge would defeat the purpose). When the judge produces a verdict, a handler decides what to do: log it, emit an event, escalate to a human, or hard-stop the run.

Five plug-in points let you tune all of this:

```
DriftEngine:
  strategy:    when to fire the judge
  window:      what slice of trace to show
  goal_source: what's the "original goal"
  prompt:      how to ask
  handlers:    what to do on positive verdict
```

Defaults work; customize anything you need.

## The simplest case

```python
from ballast.drift import (
    GoalDriftDetector,
    DriftEngine,
    EveryNSteps,
    LastNMessages,
    FirstUserMessage,
    DefaultPromptBuilder,
    LogOnly,
    make_default_judge,
)

drift_engine = DriftEngine(
    strategy=EveryNSteps(n=3),
    window=LastNMessages(n=10),
    goal_source=FirstUserMessage(),
    prompt=DefaultPromptBuilder(),
    judge=make_default_judge(),     # uses .with_judge_defaults() model
    handlers=[LogOnly()],
)

detector = GoalDriftDetector(engine=drift_engine)

agent = Agent(model="openai:gpt-4o", capabilities=[detector])
```

What this does: every 3 agent steps, take the last 10 messages, treat the first user message as the goal, ask the default judge "is this trace still pursuing this goal?", log a warning if it says no.

Drop-in for "I want some drift visibility but don't want to make decisions yet" — the LogOnly handler turns drift into telemetry. You watch the logs, see what trips it, then tune.

## The five protocols, one by one

### `DriftCheckStrategy` — when to fire

The judge isn't free (it's another LLM call) so you don't run it every step. The strategy decides when:

- **`AfterEveryStep()`** — fire on every step. Expensive; use only for short, high-stakes agents.
- **`EveryNToolCalls(n=5)`** — fire when the cumulative tool-call count advances by n. Use when drift typically shows up as "the agent keeps calling tools."
- **`EveryNSteps(n=3)`** — fire every n steps. Simple time-based check.
- **`Periodic(seconds=30.0)`** — fire at most once per elapsed wall-clock window. Use for long-running agents where steps vary in duration.
- **`OnBudgetThreshold(fraction=0.5, budget_fn=...)`** — fire once when budget consumption crosses a threshold. Bridges to `BudgetGuard.snapshot()`.
- **`Compose(s1, s2, ...)`** — OR-combine: fires if any strategy says yes.

For most agents, `EveryNSteps(n=3)` or `OnBudgetThreshold(fraction=0.5, budget_fn=guard.snapshot)` is the right starting point.

### `TraceWindow` — what to show the judge

The judge sees a slice of the message history. Showing all of it is expensive; showing too little misses context.

- **`FullTrace()`** — everything. Expensive for long runs.
- **`LastNMessages(n=10)`** — tail of the trace. Cheapest; works when drift shows up recently.
- **`SinceLastUserMessage()`** — from the last user prompt onward. Useful in conversational agents where each user turn is a new mini-goal.
- **`TokenBudgetWindow(max_tokens=4000)`** — trim from the head until under a token cap. Predictable cost.

`LastNMessages(n=10)` is the default and works for most cases. Switch to `TokenBudgetWindow` if message sizes vary wildly.

### `GoalSource` — what's the goal

The judge needs to know what the goal *is* to compare the trace against. Where does that come from?

- **`FirstUserMessage()`** — first user message in the trace. Right for single-turn or task-driven agents.
- **`LastUserMessage()`** — most recent user message. Right for conversational agents where each turn redefines the goal.
- **`WorkflowInput()`** — the workflow's input payload, stringified. Right for durable workflows where the "real" goal is the workflow argument, not whatever the user typed.
- **`ExplicitGoal(text=...)`** — static goal text. Right for agents wrapping a fixed task (e.g., "summarize the document below" — that's the goal forever).

### `PromptBuilder` — how to ask

The judge gets a prompt: here's the goal, here's the trace, is it still on-task? `DefaultPromptBuilder()` produces a reasonable default. For domain-specific judges (where the question should be "is this still pursuing the legal compliance review?"), implement your own:

```python
class LegalReviewPromptBuilder:
    def build(self, goal: str, trace: list[ModelMessage]) -> str:
        return f"""You are auditing whether this agent's trace is still
        focused on the legal compliance review described below.

        GOAL: {goal}

        TRACE: {format_trace(trace)}

        Has the agent drifted to discussing unrelated topics?
        """
```

### `DriftHandler` — what to do on positive verdict

When the judge says "yes, this drifted," what should happen?

- **`LogOnly()`** — write a WARNING. Doesn't block, doesn't interrupt. Use during initial rollout.
- **`EmitDriftEvent(sink=..., event_name="goal_drift")`** — push a structured event to your sink (metrics, audit log). Doesn't block.
- **`RaiseDriftError()`** — raise `GoalDriftError(verdict)`. Hard-stops the run; user sees a failed `agent.run`.
- **`EscalateToHITL(channel=..., card_factory=..., timeout=...)`** — open a HITL card and wait for human response before continuing.
- **`Compose(h1, h2, ...)`** — run handlers in order; one raising doesn't stop the others (except `RaiseDriftError`, which intentionally stops everything).

Common production combination:

```python
handlers=[
    LogOnly(),
    EmitDriftEvent(sink=metrics_sink, event_name="agent.drift_detected"),
    EscalateToHITL(channel=ui_channel, card_factory=drift_card, timeout=timedelta(minutes=5)),
]
```

Log it, emit a metric, ask a human. If the human approves continuing, the run proceeds; if not, the workflow aborts.

## The `DriftEngine`

```python
@dataclass
class DriftEngine:
    strategy: DriftCheckStrategy
    window: TraceWindow
    goal_source: GoalSource
    prompt: PromptBuilder
    judge: Any                                       # pydantic-ai Agent
    handlers: list[DriftHandler] = field(default_factory=list)
    verdict_model: type[DriftVerdictBase] = DefaultDriftVerdict
```

Method: `async maybe_check(signal, ctx) -> DriftVerdictBase | None` — returns a verdict if the check fired this step, `None` otherwise.

You normally don't call `maybe_check` yourself. `GoalDriftDetector` (the capability wrapper) calls it from `after_model_request`.

The `verdict_model` defaults to `DefaultDriftVerdict` (drifted: bool, rationale: str, confidence: Confidence). Override it if your judge should produce a richer verdict shape — e.g., `SeverityVerdict(drifted: bool, severity: Literal["minor", "moderate", "severe"], rationale: str)`.

## Failsafe semantics

The judge is another LLM call. It can fail (provider outage, timeout, malformed response). The framework's contract: **a failed judge never breaks the user-facing run**. Specifically:

- Judge timeout → log, treat as "no verdict this round," move on.
- Judge raises → log, swallow, move on.
- Judge returns malformed verdict → log, swallow, move on.

The only way drift detection blocks the run is via `RaiseDriftError()` (intentional stop) or `EscalateToHITL()` (waiting for human). Everything else is best-effort.

This is the right default because the alternative — drift detection itself being a single point of failure — is much worse than missing the occasional drift signal.

## Composing with `BudgetGuard`

The most common bridge: fire the drift judge halfway through the budget.

```python
guard = BudgetGuard(max_iterations=20)

drift_engine = DriftEngine(
    strategy=OnBudgetThreshold(fraction=0.5, budget_fn=guard.snapshot),
    window=LastNMessages(n=10),
    goal_source=FirstUserMessage(),
    prompt=DefaultPromptBuilder(),
    judge=make_default_judge(),
    handlers=[EmitDriftEvent(sink=metrics_sink), EscalateToHITL(...)],
)

agent = Agent(model=..., capabilities=[guard, GoalDriftDetector(engine=drift_engine)])
```

What this gives you: the drift judge fires exactly once per run, at the moment the agent has consumed half its budget. By then there's enough trace to evaluate; before half-budget, the agent often hasn't done enough to drift yet.

This pattern is the single best argument for `BudgetGuard.snapshot()` existing.

## Composition with `CoALAUnit`

The framework ships `goal_drift_as_unit(...)` (chapter 19), a thin adapter that wraps a `DriftEngine` as a CoALA unit. Useful when you want the drift judge to run as a step in a PlanAndExecute DAG rather than as a hook in an agent's loop. Same engine, different deployment surface.

## Tuning what counts as drift

The judge gets a prompt that says, roughly, "Is the trace still pursuing the goal?" with sensible defaults. In practice, you'll want to tune two things:

- **What "drift" means in your domain.** A coding agent that switches from "fix the bug" to "explain the architecture" probably drifted. A research agent that follows a citation trail might *look* like drift but is actually doing its job. Customize via `PromptBuilder`.
- **How confident the judge has to be.** `DefaultDriftVerdict` has a `confidence` field (low/medium/high). Apps often want to gate handlers: only escalate to HITL if `confidence == "high"`. Implement a custom `DriftHandler.handle` that inspects the verdict before acting.

## Common mistakes

- **Firing the judge every step.** Don't. Even a fast judge is dozens of tokens; running on a 50-step agent costs more than the agent itself.
- **Using `FirstUserMessage()` for conversational agents.** In a chat, the "goal" changes every turn. Use `LastUserMessage()` or `SinceLastUserMessage()` window.
- **`RaiseDriftError()` as the default handler.** Drift judges aren't perfect; they have false positives. Start with `LogOnly` + `EmitDriftEvent`, watch the false-positive rate, *then* graduate to harder actions.
- **Hoping the judge will catch *every* drift.** It won't. Drift detection is one signal in a stack; combine with output-level grading (chapter 23) for end-to-end coverage.

## What this chapter did NOT cover

- The `CircuitBreaker` for cross-run resilience — chapter 13.
- Output-level quality grading (after-the-fact, not in-flight) — chapter 23.
- The `goal_drift_as_unit` adapter and CoALA composition — chapter 19.
- Writing a custom `DriftVerdictBase` subclass with richer fields — covered in the reference for `ballast.drift`.

## Where to go next

→ [13-resilience.md](13-resilience.md) — circuit breakers for cross-run resilience.

# 7. Capabilities

**Prerequisites:** [01-agents.md](01-agents.md), [04-dependencies-and-state.md](04-dependencies-and-state.md).

## Introduction

So far you've defined agents with tools and typed outputs. You've passed per-run state through `deps`. What's missing is everything *cross-cutting* — budget caps, loop detection, drift checks, output grading, HITL approval, observability — the stuff that should apply uniformly across many agents and many tools, without being copy-pasted into every tool body.

The framework's answer is **capabilities**. A capability is an object you attach to an `Agent` via `capabilities=[...]`. It hooks into the agent's lifecycle at well-defined points (before the model request, after the model request, after the run, around the run) and gets to observe or modify what the agent does. Capabilities compose — you stack any combination, and the framework orchestrates them.

This chapter covers the `BallastCapability` protocol, what each hook does and when it fires, the `for_run` pattern that makes stateful capabilities safe across concurrent runs, what's in the box (budget guards, loop detectors, drift detectors, approval bridges), and how to write your own.

## The mental model

Capabilities are *aspect-oriented* additions to the agent's run loop. They don't change what the agent *does* — they observe, count, gate, and (rarely) modify.

The right mental model is middleware. Just as HTTP middleware wraps a request-response cycle with cross-cutting logic (auth, logging, compression), capabilities wrap an agent's run with cross-cutting logic (budget enforcement, drift detection, output grading). The agent doesn't know capabilities exist; the framework orchestrates them invisibly.

The reason this layer exists separately from tools: tools are *what the agent does*, capabilities are *how the agent is governed*. Mixing the two would force every tool to remember to call your budget check, your drift logger, your HITL bridge. The capability layer keeps governance logic out of business logic.

## The hooks

`BallastCapability` is an abstract base. Five hooks; you override the ones you care about and leave the rest to no-op defaults.

### `for_run(ctx) -> BallastCapability`

Called once at the start of each `agent.run(...)`. The default returns `self` (stateless capabilities). Stateful capabilities return a *fresh clone* so per-run counters don't leak across runs:

```python
class MyGuard(BallastCapability):
    def __init__(self, threshold: int):
        self.threshold = threshold
        self._counter = 0

    async def for_run(self, ctx):
        return MyGuard(threshold=self.threshold)  # fresh counter per run
```

**The single most important rule** in capability design: per-run state goes on the clone returned by `for_run`. Don't put counters / accumulators / per-request state on the shared instance. Chapter 4 covers this in the deps context; here the same rule applies to capabilities.

### `before_model_request(ctx, request_context)`

Called before each LLM call within the agent's loop. You see the prompt that's about to go to the model:

```python
async def before_model_request(self, ctx, request_context):
    # request_context.messages is the message history
    # request_context.model_settings is provider-specific config
    # Return the request_context (possibly modified); raising aborts the run.
    return request_context
```

This is where token-counting, drift checks (deciding whether to fire a judge), and similar pre-call concerns live. You can return a modified `request_context` to alter what the model receives — typically you don't, and returning the input unchanged is the right move.

### `after_model_request(ctx, request_context, response)`

Called after each LLM call. You see the response (a `ModelResponse` with one or more parts — text or tool calls):

```python
async def after_model_request(self, ctx, *, request_context, response):
    # response.parts is a list of TextPart / ToolCallPart
    # response.usage carries input/output token counts
    self._iterations += 1
    self._tokens += response.usage.input_tokens + response.usage.output_tokens
    if self._iterations > self.max_iterations:
        raise BudgetExhausted(...)
    return response
```

This is where the bulk of capability logic happens. `BudgetGuard`, `SemanticLoopDetector`, and `GoalDriftDetector` all hook here. The response can be modified — but in practice you only ever inspect, not mutate.

### `after_run(ctx, *, result)`

Called once after the agent's loop exits — `result` is the final `AgentRunResult`:

```python
async def after_run(self, ctx, *, result):
    verdict = await self.judge.grade(result.output)
    await self.persistence.save(verdict)
    return result   # return the result (possibly with attached metadata)
```

This is where output-level grading (`JudgeAfterRun`), final persistence, and end-of-run cleanup live. The `result.output` is the validated typed output (or `DeferredToolRequests` if the agent ended waiting for approval).

### `wrap_run(ctx, *, handler)`

Called *around* the entire agent run. `handler` is a callable that runs the actual agent loop. You can call it once, multiple times, or not at all:

```python
async def wrap_run(self, ctx, *, handler):
    result = await handler()
    while isinstance(result.output, DeferredToolRequests):
        # Open HITL cards, collect verdicts, re-run with deferred_tool_results
        approvals = await self.collect_approvals(result.output.approvals)
        result = await handler.__self__.run(
            None,
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals=approvals, calls={}),
            deps=ctx.deps,
        )
    return result
```

`wrap_run` is the most powerful hook — and the least used. The framework's `ApprovalCapability` uses it (chapter 21) because it needs to potentially re-run the agent multiple times to handle cascading approval rounds. Most capabilities don't need this much control.

## The lifecycle in one diagram

```
agent.run(input):
  ├─ capability.for_run(ctx)  →  cloned capability (per-run state isolated)
  │
  ├─ wrap_run(ctx, handler):
  │     [handler() begins the actual agent loop]
  │     │
  │     ├─ before_model_request(ctx, request_context)
  │     ├─ [MODEL CALL]
  │     ├─ after_model_request(ctx, request_context, response)
  │     │
  │     ├─ [TOOL CALLS — if model called any tools]
  │     │
  │     ├─ ... loop iterations ...
  │     │
  │     └─ [LOOP EXITS — final answer or DeferredToolRequests]
  │
  └─ after_run(ctx, result=final_result)
     return final_result
```

Each capability gets its own `for_run` clone; each clone receives its own sequence of `before_model_request` → `after_model_request` → `after_run` calls. Multiple capabilities run in declared order (the order you pass them in `capabilities=[...]`).

## Stacking is the standard pattern

The article's "compounding error problem" — agents that succeed 85% per step compound to 20% over 10 steps — has a flip side. If multiple independent guards each catch 80% of their failure mode, your composite reliability multiplies *up*: 1 - (0.2 × 0.2 × 0.2) ≈ 99%. That's the math behind capability stacking.

The pattern:

```python
from ballast import (
    BudgetGuard,
    SemanticLoopDetector,
    TypedLoopGuard,
    GoalDriftDetector,
    ApprovalCapability,
    JudgeAfterRun,
)

agent = Agent(
    model="openai:gpt-4o",
    capabilities=[
        BudgetGuard(max_iterations=15, max_input_tokens=30_000),
        SemanticLoopDetector(embedder=my_embedder, threshold=0.95),
        TypedLoopGuard(output_type=MyOutput),
        GoalDriftDetector(engine=my_drift_engine),
        ApprovalCapability(tool_card_map=my_card_map),
        JudgeAfterRun(judge=my_quality_judge),
    ],
)
```

Six guards, six failure modes:

- **BudgetGuard** — resource cap (tokens, iterations)
- **SemanticLoopDetector** — repeated near-identical responses (loop-happiness)
- **TypedLoopGuard** — output convergence between Pattern iterations
- **GoalDriftDetector** — semantic drift from the original goal
- **ApprovalCapability** — HITL bridge for `requires_approval=True` tools
- **JudgeAfterRun** — output-level quality grading

Each one is cheap (in-process counters, embeddings, occasional async LLM call for the judges). Stacking is the production-default pattern.

## A tour of what's in the box

The next several chapters cover each in depth. Here's the lineup so you know what's coming:

- **`BudgetGuard`** (chapter 11) — iteration + token caps. The most fundamental.
- **`SemanticLoopDetector`** (chapter 11) — cosine-similarity over recent responses to catch loop-happiness.
- **`TypedLoopGuard`** (chapter 11) — convergence detection between Pattern iterations.
- **`PIIGuard`** (chapter 11) — redacts PII from prompts before they hit the model.
- **`GroundedRetry`** (chapter 11) — targeted retry when the LLM fails to ground a `Ref[T]`.
- **`GoalDriftDetector`** (chapter 12) — async LLM judge against the original goal.
- **`CircuitBreaker.as_capability()`** (chapter 13) — wraps the agent run in a circuit breaker for cross-run resilience.
- **`ApprovalCapability`** (chapter 21) — bridges pydantic-ai's `requires_approval=True` to HITL cards.
- **`JudgeAfterRun`** (chapter 23) — output grading via `LLMJudge`.

Each is a `BallastCapability` subclass; each follows the per-run isolation rule; each composes with the others.

## Writing your own

The smallest viable custom capability:

```python
from ballast import BallastCapability

class _LogEveryRequest(BallastCapability):
    name = "log_every_request"

    async def after_model_request(self, ctx, *, request_context, response):
        logger.info(
            "model_request",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response

agent = Agent(model=..., capabilities=[_LogEveryRequest()])
```

That's the whole pattern. Stateless capability, one hook, ~10 lines. Drop it in.

For a stateful capability, override `for_run` to return a clone:

```python
class _MaxToolCallsPerRun(BallastCapability):
    name = "max_tool_calls"

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self._count = 0

    async def for_run(self, ctx):
        return _MaxToolCallsPerRun(max_calls=self.max_calls)  # fresh counter

    async def after_model_request(self, ctx, *, request_context, response):
        from pydantic_ai.messages import ToolCallPart
        for part in response.parts:
            if isinstance(part, ToolCallPart):
                self._count += 1
        if self._count > self.max_calls:
            raise RuntimeError(f"too many tool calls ({self._count}); limit {self.max_calls}")
        return response
```

You can also use `wrap_run` for capabilities that need to re-run the agent (like `ApprovalCapability`) or that need to control the whole loop from outside. That's a smaller use case; reach for hooks first.

## Hard rules

Pull these out so they don't get lost:

1. **Per-run state goes on the `for_run` clone, not the shared instance.** Otherwise concurrent runs corrupt each other's state.
2. **Don't put long-lived state on capability instance attrs.** Same rule; restating because it's the most common mistake.
3. **Hooks should be fast.** Each one runs synchronously in the agent's loop. Slow hook = slow agent. Reach for `asyncio.create_task(...)` if you need to log/persist without blocking.
4. **Don't raise from hooks unless you intend to abort the run.** The agent loop catches the exception and propagates it. If you raise from `after_model_request`, the user sees a failed `agent.run`.
5. **Don't modify the response without good reason.** `before/after_model_request` *can* return a modified value; almost no built-in capability does this. Inspection is the dominant pattern.
6. **Order matters.** Capabilities run in declared order. If `BudgetGuard` should fire before `JudgeAfterRun`, list `BudgetGuard` first.

## What this chapter did NOT cover

- How to actually deploy this agent into a running app — chapter 8.
- The specific behavior of each shipped capability — chapters 11, 12, 13, 21, 23.
- How capabilities compose with Patterns — chapters 14-17.
- The CoALA unit pattern (which has its own deployment as a capability) — chapter 19.

## Where to go next

→ [08-running-an-app.md](08-running-an-app.md) — assembling agents into a runnable FastAPI service.

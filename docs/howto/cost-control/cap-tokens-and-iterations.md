# Capping tokens and iterations

## Introduction

Every production deployment of an LLM-powered agent eventually meets the same problem: the agent loops. It calls a tool, gets an unexpected error, tries the same tool again with slightly different arguments, gets another error, tries again, and again, and again — until your token budget is exhausted, your CI job is killed, or your invoice arrives.

The math behind this is brutal and unforgiving. Suppose every individual step the agent takes succeeds 85% of the time — a respectable number for any non-trivial tool call or reasoning step. Run a ten-step workflow and the probability of clean success drops to **roughly twenty percent**. Run a twenty-step workflow and you are below five percent. This is the **compounding error problem**, and it explains why agentic systems that demo beautifully in a Jupyter notebook turn into runaway disasters in production.

The conclusion is not that LLMs need to be smarter. The conclusion is that *every step has to be observably bounded*. Not bounded by hope, by good prompt engineering, or by the model's good judgment — bounded by **explicit, capability-level limits enforced before the request leaves your process**.

Ballast ships `BudgetGuard` exactly for this purpose. It sits at the framework's capability layer and intercepts model requests, counting iterations and tokens. When any configured limit is reached, the agent's run is cut short with a typed exception. No silent failures, no surprise invoices, no late-night incidents.

This guide walks through how to set up `BudgetGuard`, how to combine it with other guards to address different classes of runaway behavior, how to catch and recover from budget exhaustion, and where the boundaries of budget enforcement actually lie.

## The mental model

Think of `BudgetGuard` as a credit card spending limit on your agent. The limit isn't there to second-guess every purchase — it's there so that when something goes wrong, the damage is bounded.

There are three independent budgets you can set:

- **Iterations** — how many model requests the agent is allowed to make in a single `run()`. This is the most fundamental cap because every LLM call is also (a) potential tool invocations, (b) potential cost, and (c) latency.
- **Input tokens** — cumulative tokens sent to the model across all iterations within one `run()`. This is what prevents context-window thrash from blowing up your spend.
- **Output tokens** — cumulative tokens generated. Mostly relevant for verbose models or chain-of-thought modes.

Each is enforced independently. The first one to trip wins, and the entire run is aborted with `BudgetExhausted`.

The key word is *cumulative*. `BudgetGuard` does not measure per-call usage — it measures total usage across the agent's full loop, since that's where runaway behaviour accumulates. A single tool call producing 50,000 tokens is fine; a tool call repeated thirty times producing 50,000 tokens *each time* is not.

## Quickstart

Attach `BudgetGuard` to any pydantic-ai `Agent` via the `capabilities` argument:

```python
from ballast import BudgetGuard
from pydantic_ai import Agent

agent = Agent(
    model="openai:gpt-4o-mini",
    system_prompt="Answer the user's question concisely.",
    capabilities=[
        BudgetGuard(
            max_iterations=10,
            max_input_tokens=20_000,
            max_output_tokens=4_000,
        ),
    ],
)
```

That's it. From now on, every `await agent.run(...)` call enforces those limits. If the agent enters a loop and hits any of them, the next request raises `BudgetExhausted` instead of going out to the model.

The exception is typed and inherits from `BallastError`, so it integrates cleanly with your existing exception handling:

```python
from ballast import BudgetExhausted

try:
    result = await agent.run("Summarize last week's PRs")
except BudgetExhausted as exc:
    logger.warning("agent ran out of budget: %s", exc)
    # Fall back to a deterministic path, or surface a friendly error to the user.
    return "I'm having trouble with that — please try again with a narrower question."
```

The numbers you choose depend on your model and use case. For a chat-style agent answering one question, ten iterations and twenty thousand input tokens is usually generous — the agent should answer in two or three model calls. For a research workflow that may legitimately call ten tools, you'd set higher limits. For batch processing where you absolutely must not exceed a per-job spend, you might pick numbers as low as five iterations and ten thousand tokens.

## Why one cap isn't enough

`BudgetGuard` is necessary but not sufficient. It's a *resource* limit, not a *behavior* limit. The agent can stay under iteration count and still produce useless output by quietly repeating itself, drifting from the original goal, or chasing a tangent.

This is where the framework's other guards come in. They're designed to compose — each addresses a different failure mode, and stacking them is the standard production pattern.

```python
from ballast import (
    BudgetGuard,
    SemanticLoopDetector,
    TypedLoopGuard,
    GoalDriftDetector,
)
from ballast.drift import (
    DriftEngine, EveryNToolCalls, LastNMessages, FirstUserMessage,
    DefaultPromptBuilder, make_default_judge, EmitDriftEvent,
)

agent = Agent(
    model="openai:gpt-4o",
    capabilities=[
        BudgetGuard(max_iterations=15, max_input_tokens=30_000),
        SemanticLoopDetector(embedder=my_embedder, threshold=0.95, window=3),
        TypedLoopGuard(output_type=MyOutput),
        GoalDriftDetector(DriftEngine(
            strategy=EveryNToolCalls(5),
            window=LastNMessages(10),
            goal_source=FirstUserMessage(),
            prompt=DefaultPromptBuilder(),
            judge=make_default_judge(),
            handlers=[EmitDriftEvent(sink=publish_to_thread)],
        )),
    ],
)
```

What each one does:

- **`BudgetGuard`** stops the agent from spending more resources than you've authorized. The blunt instrument.
- **`SemanticLoopDetector`** stops the agent from emitting near-identical model responses in succession. Catches the classic "I'll just call this tool one more time with the same arguments" pattern that wastes budget without making progress.
- **`TypedLoopGuard`** stops the agent when typed outputs converge between iterations of a Pattern (like `Reflection`'s writer-critic-refiner cycle). Catches the case where the critic and refiner agree on nothing and ping-pong forever.
- **`GoalDriftDetector`** runs an asynchronous LLM judge against the original goal and a recent slice of the trace. Catches the case where the agent stays "busy" — passing all the above checks — but is solving a completely different problem than the one the user asked for.

Each guard catches a different *class* of runaway behavior. The math from before applies in reverse: if each guard catches eighty percent of its respective failure mode, your composite reliability on any given step climbs from around 85% to well above 99%. That's the practical answer to the compounding error problem.

The cost of stacking is low. `BudgetGuard`, `SemanticLoopDetector`, and `TypedLoopGuard` are pure in-process counters with no LLM calls. Only `GoalDriftDetector` makes additional model calls, and its strategy parameter lets you control how often (every five tool calls is a reasonable default).

## Catching and recovering

The `BudgetExhausted` exception is your hook for graceful degradation. The framework deliberately does not try to "recover" automatically — silent retry would be exactly the loop behavior you're trying to prevent. Instead, the framework hands control back to your code, where you can apply whatever recovery policy fits your application.

A simple recovery is to log the incident and return a static fallback:

```python
try:
    result = await agent.run(query)
    return result.output
except BudgetExhausted as exc:
    logger.warning("agent budget exhausted on query=%r: %s", query, exc)
    return "I couldn't process that fully. Please rephrase or try again later."
```

A more sophisticated approach is to switch to a cheaper or more deterministic backend on exhaustion:

```python
try:
    return (await premium_agent.run(query)).output
except BudgetExhausted:
    logger.info("premium agent exhausted, falling back to deterministic path")
    return await deterministic_handler.handle(query)
```

If your application is built on `Durable` workflows, budget exhaustion can be handled at the workflow level — DBOS catches the exception and applies your retry policy, or you wrap the `agent.run` call in a `try`/`except` and emit a thread event so the UI can surface the issue to the user.

The point is that `BudgetExhausted` is a *signal*, not a failure mode you have to fight. Treat it as you would any business exception: catch it where you have enough context to decide what to do.

## Bridging to Circuit Breakers

If budget exhaustion happens *repeatedly* across runs, that's signal of a deeper problem: a model regression, a prompt that's no longer working, an upstream API that's gone bad. Wrapping the agent in a `CircuitBreaker` lets you detect this pattern and trip a higher-level cutoff:

```python
from datetime import timedelta
from ballast import BudgetExhausted
from ballast.resilience.circuit_breaker import (
    CircuitBreaker, Consecutive, Chain, ReturnValue, EscalateToHITL,
)

cb = CircuitBreaker(
    threshold_factory=lambda: Consecutive(3),
    recovery_after=timedelta(minutes=2),
    is_failure_exc=(BudgetExhausted, RuntimeError),
    fallback=Chain(
        ReturnValue("Service is temporarily reduced; please try again shortly."),
        EscalateToHITL(channel=ops_channel, card_factory=ServiceDegradedCard),
    ),
)

async def run_with_protection(query: str) -> str:
    result = await cb.call(lambda: agent.run(query))
    return result.output
```

After three consecutive `BudgetExhausted` exceptions, the breaker opens. Subsequent calls go directly to the fallback — first returning the polite degradation message to users, and (in parallel via `Chain`) opening a HITL ticket so an operator can investigate. After two minutes of cooldown, the breaker enters Half-Open and probes a single request; if it succeeds, normal service resumes.

This is the "mandatory final state" pattern that production-grade agentic systems converge on: don't let a single broken component cascade, and don't pretend silence is success.

## Persistence-of-state notes

`BudgetGuard` state lives in process memory, isolated per agent run. The `BudgetGuard` instance you pass into `Agent(capabilities=[...])` is a *config holder*; the actual counters live on the per-run clone returned by the framework's `for_run` hook. Two concurrent runs of the same agent do not share counters, and you don't have to wrap or scope anything yourself.

This per-run isolation is also why the same `BudgetGuard` instance can be safely shared across an agent's many invocations — there's no global state to leak. You construct it once when building the agent and forget about it.

You can read the live counters via `BudgetGuard.snapshot()` from within the agent run (typically from a custom capability or a metadata-provider callback) — this is how `GoalDriftDetector`'s `OnBudgetThreshold` strategy decides when to fire. The snapshot is a flat dict suitable for OpenTelemetry attributes or a dashboard:

```python
snap = budget.snapshot()
# {
#   "iterations": 7, "max_iterations": 15,
#   "input_tokens": 18_240, "max_input_tokens": 30_000,
#   "output_tokens": 3_120, "max_output_tokens": None,
# }
```

## Tuning the numbers

Picking the right limits is a matter of profiling, not theory. The recommended approach:

1. Start with generous limits (twenty iterations, fifty thousand input tokens). Run your application in development for a week.
2. Collect `BudgetGuard.snapshot()` after each successful agent run via your tracing layer. Plot the distribution of iterations and tokens used.
3. Set production limits at the 99th percentile of the development distribution, plus 50% headroom.

This procedure keeps real users from hitting limits while still bounding worst-case spend. If you set limits without measurement, you'll either over-spend (limits too generous) or generate spurious `BudgetExhausted` errors for legitimate use cases (limits too tight).

For latency-sensitive applications (chat UIs), bias toward tighter iteration limits — every iteration is round-trip latency. For background-processing applications (batch jobs, scheduled workflows), bias toward higher iteration limits but tighter token limits — the worst-case spend is what matters, not the worst-case latency.

## What `BudgetGuard` does *not* protect

Honesty here is important. `BudgetGuard` measures *what the model reports* in `response.usage` — input + output tokens at the model API boundary. It does *not* count:

- **Tool-execution time or external API tokens.** If your agent calls a tool that calls another LLM through a different code path (e.g., a sub-agent inside a tool body), those tokens are not counted by the parent's `BudgetGuard`. Each agent has its own guard; account for nested costs separately.
- **Tokens used by capabilities themselves.** `GoalDriftDetector` runs a judge model, which has its own cost. That cost is not deducted from the parent agent's budget.
- **Embedding API calls** done by `SemanticLoopDetector` or other capabilities relying on `Embedder`. These are typically negligible (embeddings are cheap), but if your `Embedder` implementation hits a paid API, that cost is separate.
- **Streaming responses** — token counts are still reported by the model at the end of each call, so streaming behaves the same as non-streaming from the guard's perspective. There is no per-token accounting *during* a stream.

For comprehensive cost tracking across all these surfaces, layer in `configure_cost_extractors(OpenRouterCostExtractor(), …)` and use logfire to view aggregated cost spans. `BudgetGuard` is your safety net at the agent level; cost extractors are your *accounting* across the whole system.

## When to use `BudgetGuard` and when to use `CircuitBreaker`

These two primitives are often confused because both can "stop" an agent. The distinction is clean:

- **`BudgetGuard` enforces a budget *within* a single agent run.** It says "this one invocation cannot exceed N iterations / tokens." Per-run scope, automatically reset on each new `agent.run()`.
- **`CircuitBreaker` enforces *system-level* health across many runs.** It says "if calls to this thing keep failing, stop trying for a while." Cross-run scope, manually reset or auto-reset after a cooldown.

You typically want both. `BudgetGuard` protects you from a single runaway agent run. `CircuitBreaker` protects you from a systematic failure that would otherwise produce many runaway runs in a row, each individually budget-bounded but collectively catastrophic.

Wire `CircuitBreaker.is_failure_exc=(BudgetExhausted, …)` to bridge them — if you're seeing budget exhaustions piling up, the breaker is the right place to take wider action (alert ops, fall back to a simpler implementation, page on-call).

## Testing budget enforcement

Budget exhaustion is a path your code needs to handle, which means it's a path you should test. The pydantic-ai `TestModel` makes this trivial — combine it with a tool that always wants to be called again to deterministically exhaust iterations:

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from ballast import BudgetGuard, BudgetExhausted


@pytest.mark.asyncio
async def test_budget_exhausts_after_iterations() -> None:
    agent = Agent(
        model=TestModel(call_tools=["loop_tool"]),
        capabilities=[BudgetGuard(max_iterations=3)],
    )

    @agent.tool
    async def loop_tool(ctx) -> str:
        return "keep going"

    with pytest.raises(BudgetExhausted):
        await agent.run("loop forever")
```

This runs in milliseconds, makes no real model calls, and verifies that your downstream `except BudgetExhausted` handlers will actually fire when production conditions create the same pattern.

For more nuanced tests — verifying that a custom recovery path engages, or that a `CircuitBreaker` wrapped around the agent opens after N exhaustions — the same `TestModel` pattern works; you just add more orchestration around the `try`/`except`.

## Common mistakes

A few patterns that look reasonable but lead to trouble:

- **Setting limits high "just to be safe."** This defeats the purpose. If your limit is ten thousand iterations, the agent will reach a hundred iterations of repeated nonsense before exhausting it. Choose limits at the 99th percentile of expected use, not at the worst case the LLM can dream up.
- **Catching `BudgetExhausted` and immediately calling `agent.run` again.** This is the polling-loop anti-pattern. The exception was raised because the agent's reasoning failed; retrying without changing the input will probably fail the same way, just doubling your cost. If you want retries, wrap the agent in `CircuitBreaker` with a custom fallback policy that actually changes something (different prompt, different model, escalation to HITL).
- **Sharing a `BudgetGuard` instance across unrelated agents.** Don't. Each agent should have its own — the per-run isolation is automatic, but mixing agents conflates their counters in your logs and complicates debugging.
- **Treating budget exhaustion as a bug to silence.** If you find yourself frequently catching `BudgetExhausted` and swallowing it, your limits are wrong (too tight) or your agent is broken (too loop-prone). Either way, the right answer is investigation, not suppression.

## Related

- [detect-goal-drift.md](detect-goal-drift.md) — catches the failure mode where the agent stays within budget but solves the wrong problem
- [../reliability/handle-flaky-external-api.md](../reliability/handle-flaky-external-api.md) — `CircuitBreaker` for cross-run protection
- [../observability-and-evals/add-tracing.md](../observability-and-evals/add-tracing.md) — surface budget counters in logfire dashboards
- Reference: `reference/capabilities/budget-guard.md`

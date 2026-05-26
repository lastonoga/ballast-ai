# 11. Budget and loops

**Prerequisites:** [07-capabilities.md](07-capabilities.md).

## Introduction

An agent that's left alone can do four destructive things: run forever (no termination), spend more tokens than you authorized (no cost cap), loop on near-identical responses (loop-happiness), or thrash between equivalent outputs across pattern iterations (convergence failure). Each one is its own failure mode with its own signature, and each one is invisible from outside — the agent looks busy, the logs look fine, and the bill arrives at the end of the month.

The framework's answer is multiple independent guards that each catch one failure mode cheaply. The math is the inverse of the compounding-error problem: if each guard catches 80% of its target mode, stacking three of them yields ~99% composite reliability. That's why the production-default app stacks all of them.

This chapter walks through the five built-in guards in `ballast.capabilities`: `BudgetGuard` (resource caps), `SemanticLoopDetector` (response repetition), `TypedLoopGuard` (pattern convergence), `PIIGuard` (data egress), and `GroundedRetry` (validation feedback). Each is one constructor, one or two hooks, ten lines of effort.

## The mental model

Five different "the agent is doing something bad" signals, five different guards. They don't overlap meaningfully — turning one off doesn't get you the others' coverage.

- **`BudgetGuard`** — *resource* exhaustion. You set a ceiling; it raises when crossed.
- **`SemanticLoopDetector`** — *semantic* repetition. The model is saying the same thing turn after turn.
- **`TypedLoopGuard`** — *output* convergence. Pattern iterations stop improving the answer.
- **`PIIGuard`** — *data egress*. Sensitive content reaches the model when it shouldn't.
- **`GroundedRetry`** — *validation feedback*. Targeted retries when output validation fails, instead of generic "try again."

Stack them in `capabilities=[...]`. Order matters (chapter 7): the first guard to raise wins, so put cheap checks before expensive ones.

## `BudgetGuard` — the floor

The single most important capability you'll ever attach. Without a budget cap, a malformed prompt can run an agent for thousands of turns; a tool that returns large output can blow your token quota in one request.

```python
from ballast import BudgetGuard, BudgetExhausted

guard = BudgetGuard(
    max_iterations=20,           # default
    max_input_tokens=30_000,     # optional; None = unlimited
    max_output_tokens=10_000,    # optional; None = unlimited
)

agent = Agent(model=..., capabilities=[guard])

try:
    result = await agent.run("...")
except BudgetExhausted as exc:
    logger.warning("budget exhausted: %s details=%s", exc.reason, exc.details)
    return fallback_answer
```

`BudgetExhausted.reason` is one of `"max_iterations"`, `"max_input_tokens"`, `"max_output_tokens"`. `details` carries the actual counts. Both are useful for metrics and dashboards.

Two implementation details worth knowing:

- **Per-run isolation.** `for_run` returns a fresh clone with `iterations=0` and `tokens=0`. Concurrent runs of the same agent don't interfere.
- **`snapshot()` exposes live state.** `guard.snapshot()` returns `{"iterations": ..., "max_iterations": ..., "input_tokens": ..., ...}`. Useful for cross-capability composition — `GoalDriftDetector`'s `OnBudgetThreshold` strategy reads this to fire its judge when half the budget is spent.

### Tuning, not theory

Don't pick `max_iterations` by deciding "20 sounds reasonable." Pick it by running your agent on representative tasks and measuring p99 iteration count, then setting the cap at 1.5-2× that. Same for tokens.

If your p99 is 8 iterations and you set `max_iterations=20`, you have 2.5× headroom — comfortable. If you set `max_iterations=8` and a real user trips it, you'll get false-positive alerts. If you set `max_iterations=100` to "be safe," you don't have a budget cap, you have a budget rumor.

## `SemanticLoopDetector` — catching loop-happiness

Sometimes the model gets stuck repeating itself: same answer, slightly rephrased, turn after turn. `BudgetGuard` will eventually catch it (after exhausting iterations) but you'd rather fail fast.

```python
from ballast import SemanticLoopDetector

detector = SemanticLoopDetector(
    embedder=my_openai_embedder,   # any Embedder Protocol
    threshold=0.95,                # cosine similarity above which → loop
    window=3,                      # compare against last 3 responses
)

agent = Agent(model=..., capabilities=[detector])
```

How it works: every `after_model_request`, the detector embeds the response text and compares cosine similarity against the embeddings of the last `window` responses. If similarity > `threshold`, raise.

The `embedder` is a Protocol — you implement whichever embedding API you have (OpenAI, Voyage, Cohere, local model). The framework doesn't ship a default because embedder choice is app-specific.

Tuning: `threshold=0.95` is conservative (most loops cross 0.95 by turn 3). For chattier agents where rephrasing is natural, drop to 0.90. For very strict agents, 0.97.

## `TypedLoopGuard` — convergence detection

`SemanticLoopDetector` works within an agent's loop. `TypedLoopGuard` works *between* pattern iterations — typically inside `Reflection` (chapter 15) where a critic-refiner pair shouldn't oscillate.

```python
from ballast.capabilities.helpers import TypedLoopGuard

guard = TypedLoopGuard(
    embedder=my_embedder,
    selector=lambda output: output.summary,   # what to embed from the output
    threshold=0.95,
    window=3,
)

# Used inside a pattern:
for i in range(max_iterations):
    output = await draft_or_refine()
    await guard.check(output)   # raises if convergence detected
```

The `selector` extracts the string (or list of strings) to embed from a typed output. Different from `SemanticLoopDetector` in that you control *what* gets compared — useful when your output has a noisy field (timestamp, ID) that would otherwise hide the convergence.

## `PIIGuard` — preventing data egress

If your prompts include user-uploaded text, that text might contain PII (names, SSNs, emails, etc.) that you don't want sent to the LLM provider — for compliance, privacy, or contractual reasons.

```python
from ballast.capabilities import PIIGuard

guard = PIIGuard(
    detector=my_pii_detector,    # PIIDetector Protocol
    redactor=lambda text, spans: redact_with_brackets(text, spans),
)

agent = Agent(model=..., capabilities=[guard])
```

The detector identifies PII spans; the redactor replaces them with placeholders. Both run on every model request, mutating prompts before they leave your process.

Built-in: `RegexDetector(patterns=[...])` for simple patterns. For production, plug in a real PII detection library (presidio, scrubadub) that satisfies the `PIIDetector` Protocol.

`PIIGuard` also has `wrap_run_event_stream`, which redacts PII from *streaming* responses on the way back. Useful when the model might echo PII it saw (or invented) into its reply.

## `GroundedRetry` — better retry on validation failure

When pydantic-ai's output validation fails, the default behavior is to send a generic "your output didn't match the schema; try again" to the model and retry. `GroundedRetry` makes that feedback more specific:

```python
from ballast.capabilities import GroundedRetry

retry = GroundedRetry(max_retries=3)

agent = Agent(
    model=...,
    output_type=ResearchSummary,    # contains Ref[Project] fields (chapter 5)
    capabilities=[retry],
)
```

When validation fails on a `Ref[T]` field (LLM picked an ID not in the candidate set), `GroundedRetry` formats a precise message: "The id 'xyz' is not in the candidate set. Valid ids: [...]". The model has a much better chance of self-correcting on retry.

For non-`Ref` schemas, the default pydantic-ai retry behavior is already fine; `GroundedRetry`'s value is specifically in the grounded-output case.

## Stacking: the production default

```python
from ballast import (
    BudgetGuard,
    SemanticLoopDetector,
    PIIGuard,
    GroundedRetry,
)

agent = Agent(
    model="openai:gpt-4o",
    output_type=Scored[ResearchSummary],
    capabilities=[
        PIIGuard(detector=presidio_detector, redactor=presidio_redactor),
        BudgetGuard(max_iterations=15, max_input_tokens=30_000),
        SemanticLoopDetector(embedder=my_embedder, threshold=0.95),
        GroundedRetry(max_retries=3),
    ],
)
```

Order rationale:

1. **`PIIGuard` first.** It modifies requests before they go out. Other guards work on the modified version.
2. **`BudgetGuard` second.** Cheap counter check; should fire before anything expensive runs.
3. **`SemanticLoopDetector` third.** Embedding call — modestly expensive.
4. **`GroundedRetry` last.** Only kicks in on validation failure; passive otherwise.

You don't need all five for every agent. The minimum for production:

- `BudgetGuard` — always.
- `PIIGuard` — if your agent sees user-uploaded text.
- `GroundedRetry` — if your output uses `Ref[T]`.

The semantic loop guards are useful enough that they're the next things to add, but they're not life-or-death.

## What recovery looks like when a guard trips

A guard raising is a signal, not a crash. Catch the exception at the agent boundary and decide what to do:

```python
try:
    result = await agent.run(query)
    return result.output
except BudgetExhausted as exc:
    # Log, alert, return degraded response
    logger.warning("budget exhausted: %s", exc.reason)
    metrics.incr("agent.budget_exhausted", tags={"reason": exc.reason})
    return DegradedResponse(reason="took too long; please rephrase")
except LoopDetected as exc:
    logger.info("loop detected after %d turns", exc.turn_count)
    return DegradedResponse(reason="model got stuck; retrying with different prompt")
```

In production:

- **`BudgetExhausted` should alert.** If 1% of runs hit budget, fine — tail of the distribution. If 10%, your cap is too tight or your prompts are degrading.
- **`LoopDetected` should log without alerting.** It's a *successful* catch, not a failure. The signal in aggregate (rate of loop-detection) is what matters.
- **`GroundedHydrationError` (chapter 5) should alert per-occurrence in the first weeks.** Once you've seen the failure modes, you can tune down to aggregate alerts.

## Why stacking isn't just defense-in-depth

Three guards with 80% catch rates don't give you 80% reliability — they give you ~99%. The math:

```
P(miss) = P(miss_1) × P(miss_2) × P(miss_3) = 0.2 × 0.2 × 0.2 = 0.008
```

That's the inverse of the compounding-error problem (which says agents that succeed 85% per step compound to 20% over 10 steps). It works in the reverse direction too: independent guards multiply success probability up.

The "independent" qualifier matters. Two guards that catch the same failure mode (two budget caps) don't compound. The framework's lineup is deliberately diverse — they catch different things — so stacking really does multiply.

## Common mistakes

- **Setting `max_iterations` too high "to be safe."** A cap you never hit isn't a cap. If p99 of normal runs is below 5 iterations, set the cap at 10, not 100. The cap should *catch* runaway behavior, not allow it.
- **Reusing one `BudgetGuard` instance across multiple agents.** Don't share. Each agent should have its own (Capability instances are cheap; the `for_run` clone handles per-run state but the *config* is per-agent).
- **Catching `BudgetExhausted` too broadly.** Don't `except Exception:` — you'll swallow real bugs. Catch the specific exception.
- **Not embedding the right thing in `SemanticLoopDetector`.** Default selector reads response text. For tool-calling agents that mostly produce tool-call parts, text might be empty. Provide a custom selector.

## What this chapter did NOT cover

- `GoalDriftDetector` — chapter 12; semantic drift is a different failure mode from semantic looping.
- `CircuitBreaker` — chapter 13; cross-run resilience, not within-run.
- Writing custom guards — chapter 7 covers the `BallastCapability` protocol; just subclass it.
- Metrics dashboards for budget / loop signals — chapter 22.

## Where to go next

→ [12-drift-detection.md](12-drift-detection.md) — when the agent stays under budget but answers the wrong question.

# How to add a budget guard

**Problem:** Your agent might loop indefinitely, exhausting your token budget without producing a useful answer.

**Solution:** Attach a `BudgetGuard` capability with iteration + token caps. When either limit is reached, the agent stops with `BudgetExhausted`.

## Minimum code

```python
from ballast import BudgetGuard
from pydantic_ai import Agent

agent = Agent(
    model="openai:gpt-4o-mini",
    system_prompt="Answer the user's question.",
    capabilities=[
        BudgetGuard(
            max_iterations=10,
            max_input_tokens=20_000,
            max_output_tokens=4_000,
        ),
    ],
)

result = await agent.run("How does diffusion training work?")
```

If the agent exceeds 10 iterations OR 20K input tokens OR 4K output tokens, the next request raises `BudgetExhausted`. Your caller should `try`/`except` it.

## Catch the exception

```python
from ballast import BudgetExhausted

try:
    result = await agent.run("Hard question...")
except BudgetExhausted as exc:
    # Log, fall back to a cheaper model, emit a thread event, etc.
    logger.warning("agent budget exhausted: %s", exc)
    return "Sorry, that took too many iterations — try rephrasing."
```

## Combine with other guards

`BudgetGuard` is just one capability. Stack it with `SemanticLoopDetector` (catches repeated outputs) and `TypedLoopGuard` (catches convergence between Reflection iterations):

```python
agent = Agent(
    model=...,
    capabilities=[
        BudgetGuard(max_iterations=15, max_input_tokens=30_000),
        SemanticLoopDetector(embedder=my_embedder, threshold=0.95, window=3),
        TypedLoopGuard(output_type=MyOutput),
    ],
)
```

Each guards a different failure mode. Stacking them is the standard production pattern.

## Bridge to Circuit Breaker

If `BudgetExhausted` recurring across runs means the agent is broken — wire it into a `CircuitBreaker`'s `is_failure_exc`:

```python
from ballast.resilience.circuit_breaker import CircuitBreaker, Consecutive
from ballast import BudgetExhausted

cb = CircuitBreaker(
    threshold_factory=lambda: Consecutive(3),
    is_failure_exc=(BudgetExhausted, RuntimeError),
)

result = await cb.call(lambda: agent.run("..."))
# After 3 consecutive budget exhaustions, CB opens → fallback fires.
```

## Caveats

- `max_input_tokens` / `max_output_tokens` are CUMULATIVE across all LLM calls in one `agent.run()`. Not per-call.
- BudgetGuard does NOT count tool execution time or external API tokens — only LLM-side input/output reported by `response.usage`.
- For per-tool budgeting, wrap individual tools in `CircuitBreaker` instead.

## Related

- [add-circuit-breaker-to-tool.md](add-circuit-breaker-to-tool.md) — protect external APIs from cascading failures
- [add-goal-drift-detector.md](add-goal-drift-detector.md) — async judge that catches semantic divergence
- Reference: `reference/capabilities/budget-guard.md`

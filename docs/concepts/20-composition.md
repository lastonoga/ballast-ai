# 20. Composition — combining everything

**Prerequisites:** chapters 14–19.

**What you'll learn:** how to combine patterns + cognitive units + capabilities + resilience primitives into a single coherent pipeline; how nested durable workflows behave on replay; the trade-offs between deep nesting and pattern composition.

## Sections

1. The composability story so far
2. Nesting patterns: `Reflection` inside `MapReduce` reduce; `DivergentConvergent` inside `MapReduce` map
3. CoALAUnit-as-step inside `PlanAndExecute`
4. CoALAUnit-as-capability auto-firing on agent runs that are themselves steps
5. Resilience wrapping at every layer: `CircuitBreaker` on tools, on steps, on workflows
6. The outer `@Durable.workflow` as orchestrator
7. Replay semantics for nested DBOS workflows
8. When to flatten vs when to nest
9. End-to-end example: research → critique → publish-with-approval pipeline
10. Where to go next

## Next

[21-human-in-the-loop.md](21-human-in-the-loop.md) — the human surfaces.

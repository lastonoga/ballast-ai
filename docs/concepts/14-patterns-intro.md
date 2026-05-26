# 14. Patterns — when one Agent isn't enough

**Prerequisites:** [01-agents.md](01-agents.md), [07-capabilities.md](07-capabilities.md).

**What you'll learn:** what a "pattern" is in Ballast (a `@Durable.workflow`-wrapped composition of agents/callables); the `Pattern` protocol; how the framework's patterns achieve replay-safety; the three families (refinement, fan-out, planning) and when to pick which.

## Sections

1. When you outgrow a single Agent run
2. The `Pattern` protocol and what it requires
3. Durability via `@Durable.workflow` and `DBOSConfiguredInstance`
4. The three families:
   - Refinement (Reflection)
   - Fan-out (MapReduce, DivergentConvergent)
   - Planning (PlanAndExecute)
5. Apps usually compose, not subclass, patterns
6. Patterns vs Capabilities — different layers
7. Patterns inside a higher-level workflow
8. Where to go next

## Next

[15-reflection.md](15-reflection.md) — the refinement family, starting with Writer-Critic-Refiner.

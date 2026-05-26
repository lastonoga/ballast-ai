# 12. Drift detection

**Prerequisites:** [07-capabilities.md](07-capabilities.md), [11-budget-and-loops.md](11-budget-and-loops.md).

**What you'll learn:** the "agent stays busy but solves the wrong problem" failure mode; how `GoalDriftDetector` runs an asynchronous LLM judge against the original goal; the five pluggable protocols that let you customize when, what, and how the judge fires.

## Sections

1. Goal drift: the silent failure mode
2. The judge-based approach: asynchronous LLM verdict on a slice of the trace
3. The `DriftEngine` and its five plug-in points
4. `DriftCheckStrategy` — when to fire (`AfterEveryStep`, `EveryNToolCalls`, `Periodic`, `OnBudgetThreshold`, `Compose`)
5. `TraceWindow` — what slice of history to show the judge
6. `GoalSource` — where the "original goal" comes from
7. `PromptBuilder` — how to ask the judge
8. `DriftHandler` — what to do on positive drift verdict (`LogOnly`, `EmitDriftEvent`, `EscalateToHITL`, `RaiseDriftError`)
9. Failsafe semantics: judge errors never break the user-facing reply
10. The workflow surface: `with_drift_monitor` decorator
11. Composition with `BudgetGuard` via `OnBudgetThreshold`
12. Where to go next

## Next

[13-resilience.md](13-resilience.md) — circuit breakers for cross-run resilience.

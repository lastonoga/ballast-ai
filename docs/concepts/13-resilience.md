# 13. Resilience — circuit breakers

**Prerequisites:** [11-budget-and-loops.md](11-budget-and-loops.md).

**What you'll learn:** the difference between per-run guards (BudgetGuard) and cross-run resilience (CircuitBreaker); the classic Closed/Open/Half-Open state machine; how to wire CB into agents, workflows, and PlanAndExecute steps; how to compose threshold + fallback policies.

## Sections

1. The cross-run failure mode: many runs each hit their limit
2. `CircuitBreaker` core: `.call(fn, ctx=...)` with scope-aware bucket
3. State machine: Closed → Open → Half-Open → Closed/Open
4. `ThresholdPolicy` — when to open (`Consecutive`, `WindowedCount`, `WindowedRate`)
5. `FallbackPolicy` — what to do when rejected (`RaiseError`, `ReturnValue`, `CallFallback`, `EscalateToHITL`, `Chain`)
6. `ScopeKey` — `global_scope` / `per_tool_scope` / `per_step_scope` / custom
7. The three adapters: `as_capability`, `as_workflow_decorator`, `as_step` (for PlanAndExecute)
8. Bridging to `BudgetGuard`: `is_failure_exc=(BudgetExhausted, ...)`
9. Manual reset for incident recovery
10. In-memory state limitation and what it means in practice
11. Where to go next

## Next

[14-patterns-intro.md](14-patterns-intro.md) — composable workflow shapes built on top of agents.

# 11. Budget and loops

**Prerequisites:** [07-capabilities.md](07-capabilities.md).

**What you'll learn:** the compounding-error problem and why agentic systems need multiple independent guards; how `BudgetGuard`, `SemanticLoopDetector`, and `TypedLoopGuard` each protect against a different failure mode; how to stack them.

## Sections

1. The compounding error problem in one paragraph
2. `BudgetGuard`: iteration + token caps; `BudgetExhausted` exception
3. `SemanticLoopDetector`: cosine-similarity over recent responses
4. `TypedLoopGuard`: catches convergence between Pattern iterations
5. `PIIGuard`: redacts sensitive content before LLM sees it
6. `GroundedRetry`: targeted retry when the LLM fails to ground a Ref
7. Why stacking matters and what each catches
8. Tuning limits via measurement, not theory
9. Recovery patterns when a guard trips
10. Where to go next

## Next

[12-drift-detection.md](12-drift-detection.md) — semantic drift, not just resource limits.

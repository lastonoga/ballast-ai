# 25. Custom extensions

**Prerequisites:** chapters 7, 11, 13, 14, 19, 21, 23 (the Protocol-defining ones).

**What you'll learn:** how to extend every Ballast surface via its Protocol; concrete templates for writing your own Capability, Pattern, Step, HITL Channel, Threshold Policy, Fallback Policy, Scorer, GoalSource/TraceWindow/DriftHandler.

## Sections

1. The Protocol-first design rationale (recap)
2. Writing a custom `BallastCapability` — per-run state, hooks, isolation rules
3. Writing a custom `Pattern` — `@Durable.workflow` + idempotency
4. Writing a custom `Step` for PlanAndExecute — registration + signature
5. Writing a custom `HITLChannel` — extending `DBOSHITLChannel` for the suspend+resume
6. Writing a custom `ThresholdPolicy` for CircuitBreaker
7. Writing a custom `FallbackPolicy` for CircuitBreaker
8. Writing a custom `Scorer` for evals
9. Writing a custom `GoalSource` / `TraceWindow` / `DriftHandler` for GoalDriftDetector
10. Writing a custom `Embedder` for SemanticLoopDetector / DivergentConvergent
11. Writing a custom `ThreadRepository` / `ApprovalCardRepository`
12. Publishing as a separate package
13. End

## Reading suggestion

If you got this far in one sitting, you've covered the framework end to end. Time to actually build something — return to [tutorial/](../tutorial/) for the end-to-end project, or pick a recipe from [howto/](../howto/) when you hit a specific problem.

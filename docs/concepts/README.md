# Concepts

This is the **learning path** for Ballast. Read the chapters in order; each one introduces ONE concept and builds on the previous ones.

> **Diátaxis fit:** this is the "tutorial-adjacent" learning track — between the [tutorial/](../tutorial/) (single end-to-end project) and the [how-to/](../howto/) (recipe for one specific pain). Read these to understand the framework. Read tutorials to build something. Read how-tos when you're stuck.

## When to use this

- You're new to Ballast and want to understand the whole thing before building anything.
- You've already used parts of it and want to fill in gaps.
- You're evaluating the framework and need to know what's in the box.

If you have a *specific* problem ("my agent loops forever"), skip to [howto/](../howto/). If you want to *build something concrete*, go to [tutorial/](../tutorial/).

## The path

Each chapter has explicit **Prerequisites** at the top — what you need to have read first. Reading out of order is fine if you already know the prerequisites; otherwise the new material won't make sense.

### Foundations — what an Agent is

1. [01-agents.md](01-agents.md) — Agent is the unit of work. What Ballast adds vs vanilla pydantic-ai.
2. [02-tools.md](02-tools.md) — Tools let the agent take action. Typed args, requires_approval.
3. [03-structured-output.md](03-structured-output.md) — Forcing the agent's reply into a typed shape.
4. [04-dependencies-and-state.md](04-dependencies-and-state.md) — RunContext, deps_type, per-run isolation.

### Building blocks — typed contracts

5. [05-grounded-references.md](05-grounded-references.md) — Ref[T] anti-hallucination for entity IDs.
6. [06-confidence-and-quality.md](06-confidence-and-quality.md) — Scored[T] for rationale + confidence signals.

### Cross-cutting concerns

7. [07-capabilities.md](07-capabilities.md) — BallastCapability protocol; stacking hooks per agent run.

### Running a real app

8. [08-running-an-app.md](08-running-an-app.md) — The Ballast() builder; Engine; FastAPI app factory.
9. [09-persistence.md](09-persistence.md) — Thread / message / approval state in Postgres.
10. [10-testing.md](10-testing.md) — TestModel for agents; DBOS SQLite fixture for workflows.

### Production hardening

11. [11-budget-and-loops.md](11-budget-and-loops.md) — Multi-guard composition: BudgetGuard, SemanticLoopDetector, TypedLoopGuard.
12. [12-drift-detection.md](12-drift-detection.md) — GoalDriftDetector; the 5 plug-in protocols.
13. [13-resilience.md](13-resilience.md) — CircuitBreaker; threshold + fallback policies; per-tool isolation.

### Patterns — composable workflows

14. [14-patterns-intro.md](14-patterns-intro.md) — Pattern protocol; when one Agent isn't enough.
15. [15-reflection.md](15-reflection.md) — Writer-Critic-Refiner with iteration cap.
16. [16-mapreduce.md](16-mapreduce.md) — Sharded extraction for documents bigger than context.
17. [17-divergent-convergent.md](17-divergent-convergent.md) — Variety via parallel exploration + convergent synthesis.

### Multi-step orchestration

18. [18-plan-and-execute.md](18-plan-and-execute.md) — Planner emits typed DAG; framework dispatches steps.
19. [19-cognitive-units.md](19-cognitive-units.md) — CoALAUnit: observe/retrieve/act/learn; three deployment adapters.
20. [20-composition.md](20-composition.md) — Combining patterns + units + capabilities in one pipeline.

### Humans, observability, evals

21. [21-human-in-the-loop.md](21-human-in-the-loop.md) — HITLChannel, UICardChannel, ApprovalCapability, HelperAgent.
22. [22-observability.md](22-observability.md) — Logfire tracing; cost extractors; @traced.
23. [23-evals.md](23-evals.md) — LLMJudge; Dataset from traces; Scorer protocol.

### Going deeper

24. [24-durability.md](24-durability.md) — DBOS facade; @Durable.workflow / step; replay semantics.
25. [25-custom-extensions.md](25-custom-extensions.md) — Writing your own capability / pattern / step / channel / scorer.

## Reading time

End-to-end ≈ 4-6 hours of focused reading. Each chapter is 10-20 minutes plus the time to try the examples.

If you only have an hour, read chapters 1, 2, 3, 7, 8. That covers the core mental model and how to build a working app.

If you only have ten minutes, read [why-ballast.md](../explanation/why-ballast.md) and skim [README.md](../README.md).

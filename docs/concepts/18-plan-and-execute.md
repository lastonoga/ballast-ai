# 18. Plan-and-Execute

**Prerequisites:** [14-patterns-intro.md](14-patterns-intro.md), [13-resilience.md](13-resilience.md).

**What you'll learn:** why ReAct doesn't scale to multi-step workflows; how `PlanAndExecute` separates planning (one expensive call) from execution (framework dispatch); the `Plan` data model with DAG validation; the `Step` protocol and `StepRegistry`; how to recover from step failures.

## Sections

1. ReAct's scaling problem: one LLM call per micro-step + one-step-ahead lookahead
2. The Plan-then-Execute alternative: typed DAG → framework dispatcher
3. `Plan` and `PlannedStep` models; cycle/dangling/duplicate validation
4. The four built-in step kinds: `llm`, `callable`, `unit`, `workflow`
5. `StepRegistry`: app-side registration of agents / callables / units / workflows
6. The `Step` protocol: writing your own step kind
7. Wave-by-wave DAG execution with `asyncio.gather` + semaphore
8. `RePlanPolicy` for failure recovery; `FailLoud` default
9. Wrapping steps with `CircuitBreaker` via `as_step`
10. Composition: PlanAndExecute inside an outer Durable workflow
11. Where to go next

## Next

[19-cognitive-units.md](19-cognitive-units.md) — the cognitive primitive.

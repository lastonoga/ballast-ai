# How to plan-then-execute multi-step tasks

**Pain:** Your task has 5+ steps with dependencies. ReAct calls a dollar-tier LLM on every micro-decision and only plans one step ahead, producing suboptimal trajectories + drift. You want a *planner-first* approach: one expensive LLM call to emit a typed DAG; then the framework dispatches each step (LLM / callable / sub-workflow / CoALA unit) — independent branches run in parallel.

**Solution:** `PlanAndExecute` pattern. Planner agent returns a typed `Plan` (validated DAG); `StepRegistry` maps step kinds to dispatchers; `@Durable.step`-memoised step execution is replay-safe.

## Minimum

```python
from pydantic_ai import Agent
from ballast import PlanAndExecute
from ballast.patterns.plan_execute import Plan, StepRegistry, FailLoud


# 1. Register what the planner can reference (agents / callables / units / workflows)
registry = StepRegistry.with_defaults()
registry.register_agent("researcher", researcher_agent)
registry.register_agent("summarizer", summarizer_agent)
registry.register_callable("scrape_url", scrape_url_async)


# 2. The planner — typed output_type=Plan
planner = Agent(
    model="openai:gpt-4o",
    system_prompt=(
        "You are a planner. Output a DAG of steps. Each step has:\n"
        "  - id: unique name\n"
        "  - kind: 'llm' | 'callable' | 'unit' | 'workflow'\n"
        "  - params: kind-specific (e.g. {'agent_name': 'researcher', 'prompt_template': ...})\n"
        "  - depends_on: list of step ids this depends on (optional)\n"
        "\n"
        "Available agents: researcher, summarizer.\n"
        "Available callables: scrape_url.\n"
    ),
    output_type=Plan,
)


# 3. Wire + run
pattern = PlanAndExecute(planner=planner, registry=registry)
outputs = await pattern.run({"topic": "ML safety 2026"})

# outputs is dict[step_id, step_output]:
print(outputs["final_summary"])
```

The planner emits something like:
```python
Plan(steps=[
    PlannedStep(id="research", kind="llm", params={
        "agent_name": "researcher",
        "prompt_template": "Research: {plan_input.topic}",
    }),
    PlannedStep(id="scrape", kind="callable", params={"fn_name": "scrape_url", "args": {"url": "..."}}),
    PlannedStep(id="final_summary", kind="llm", params={
        "agent_name": "summarizer",
        "prompt_template": "Summarize based on:\n{research}\n{scrape}",
    }, depends_on=["research", "scrape"]),
])
```

Framework executes: `research` + `scrape` in parallel (no deps); `final_summary` after both complete.

## Built-in step kinds

| Kind | What it dispatches to | Params |
|---|---|---|
| `llm` | Registered Agent via `registry.register_agent(name, agent)` | `agent_name`, `prompt_template` (supports `{plan_input.x}` / `{dep_id.field}` substitution), optional `output_field` |
| `callable` | Registered async fn via `registry.register_callable(name, fn)` | `fn_name`, optional `args` dict |
| `unit` | Registered CoALAUnit via `registry.register_unit(name, unit)` | `unit_name`, optional `input_from` (dep id) |
| `workflow` | Registered async fn via `registry.register_workflow(name, fn)` | `workflow_name`, optional `input_from` (dep id) |

## Custom step kinds

`Step` Protocol — apps implement:

```python
from ballast.patterns.plan_execute import Step


class MyCustomStep:
    async def execute(self, plan_input, dep_outputs, ctx) -> dict:
        # custom logic
        return {"custom_result": ...}


registry.register_step("my_kind", MyCustomStep())
# Planner can now emit PlannedStep(kind="my_kind", ...)
```

## Recovery from step failure

By default, `PlanAndExecute` uses `FailLoud` — first step failure raises `PlanExecutionError(failed_step, partial_outputs)`. For adaptive recovery, write a custom `RePlanPolicy`:

```python
from ballast.patterns.plan_execute import RePlanPolicy, Plan


class _SwapToFallbackPlan:
    async def on_step_failure(self, plan, failed_step, error, partial_outputs):
        if failed_step.id == "scrape":
            # Use cached version instead
            return Plan(steps=[
                *[s for s in plan.steps if s.id != "scrape"],
                PlannedStep(id="scrape", kind="callable",
                            params={"fn_name": "cached_scrape"}),
            ])
        return None    # FailLoud for any other step


pattern = PlanAndExecute(
    planner=planner, registry=registry,
    replan_policy=_SwapToFallbackPlan(),
)
```

The policy can return a new `Plan` (executor continues with preserved `partial_outputs`) or `None` (raise).

## Wrap steps with CircuitBreaker

```python
from ballast.resilience.circuit_breaker import CircuitBreaker, as_step, per_step_scope


cb = CircuitBreaker(scope_key=per_step_scope)

protected_llm = as_step(cb, LLMStep(registry))
registry.register_step("llm", protected_llm)   # override default

protected_callable = as_step(cb, CallableStep(registry))
registry.register_step("callable", protected_callable)
```

Now any step (regardless of kind) gets per-step CB scope. Cascading failures don't bring down the whole DAG.

## Inside an outer durable workflow

```python
from ballast import Durable


@Durable.workflow()
async def research_and_publish(topic: str) -> str:
    outputs = await pattern.run({"topic": topic})
    summary = outputs["final_summary"]
    
    if needs_approval(summary):
        verdict = await approval_channel.request(summary)
        if verdict.decision == "reject":
            return "Cancelled by user"
    
    return await publisher.publish(summary)
```

`PlanAndExecute.run` is `@Durable.workflow` itself — DBOS replays correctly. Wrapping in an outer workflow gives you a stable parent for HITL gates + post-processing.

## Caveats

- **Planner output is structurally validated** (cycles / dangling deps / dup ids → `ValidationError` before execution). Bad plans fail fast.
- **Wave-by-wave execution**, not continuous. All ready steps in one wave run via `asyncio.gather`; next wave waits for full completion. Simpler; can upgrade to continuous later if needed.
- **Step args must be picklable** for DBOS step memoisation. Don't pass unpicklable closures (e.g. lambdas with bound DB clients) as args — register them in `StepRegistry` once instead.
- **`max_parallel` (default 8)** caps concurrent step execution. Tune for your model rate limits.
- **Re-planning is unbounded in first cut.** `FailLoud` is the only built-in policy. Custom `OnFailure(max_replans=N)` policies are app-owned to avoid infinite-replan risk.

## Related

- [build-cognitive-units.md](build-cognitive-units.md) — CoALAUnit deployed as a step via `UnitStep`
- [../reliability/handle-flaky-external-api.md](../reliability/handle-flaky-external-api.md) — wrap steps with CircuitBreaker
- [../scaling-context/process-large-documents.md](../scaling-context/process-large-documents.md) — MapReduce when you want fan-out, not a DAG
- Reference: `reference/patterns/plan-and-execute.md`
- Explanation: [../../explanation/article-pain-points.md](../../explanation/article-pain-points.md) #14

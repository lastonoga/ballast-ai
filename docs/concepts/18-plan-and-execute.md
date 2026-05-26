# 18. Plan-and-Execute

**Prerequisites:** [14-patterns-intro.md](14-patterns-intro.md), [13-resilience.md](13-resilience.md).

## Introduction

The dominant agentic shape since 2023 has been ReAct: the model decides one tool call, observes the result, decides the next, observes, and so on. For two- or three-step tasks it works. For genuine multi-step workflows — "research the topic, draft a summary, fact-check it, then publish if it passes" — ReAct accumulates problems: an LLM call per micro-step, one-step-ahead lookahead, no notion of parallelism, and no separation between planning and execution.

`PlanAndExecute` flips the architecture: one expensive planning call produces a typed DAG of steps with declared dependencies; the framework dispatches each step (LLM call, plain async function, CoALA unit, sub-workflow) when its dependencies are satisfied. Independent steps run in parallel. Failures are handled by a policy you control. Replay-safety is automatic.

This chapter covers the `Plan` data model, the four built-in step kinds, the `StepRegistry` that maps step names to implementations, the `RePlanPolicy` for failure recovery, and how to wrap any step in a `CircuitBreaker` for cross-run resilience.

## The mental model

```
1. planner_agent.run(task)
        │
        ▼
   Plan { steps: [PlannedStep, ...] }   # typed DAG, validated for cycles
        │
        ▼
2. Framework dispatcher:
        │
        ├── wave 1: steps with no deps → asyncio.gather (bounded)
        ├── wave 2: steps whose deps satisfied → asyncio.gather
        ├── ...
        └── final: produce per-step outputs dict
```

The planner pays one expensive LLM call. The executor pays the per-step cost — but only the per-step cost, not the planning overhead at each step. For a 6-step plan with 3 of them running in parallel, you've roughly halved latency vs. sequential ReAct.

## The `Plan` model

```python
from ballast.patterns.plan_execute import Plan, PlannedStep

class PlannedStep(BaseModel):
    id: str                              # unique within plan
    kind: str                            # registry key: "llm" / "callable" / "unit" / "workflow" / custom
    params: dict[str, Any] = {}          # kind-specific config
    depends_on: list[str] = []           # IDs of other steps this depends on
    description: str = ""                # human-readable rationale

class Plan(BaseModel):
    steps: list[PlannedStep] = []
    rationale: str = ""
```

The model validates: step IDs are unique, no dangling dependencies, no cycles (DFS check). Invalid plans are rejected before execution starts.

The planner produces `Plan` as its `output_type`:

```python
planner = Agent(
    model="openai:gpt-4o",
    output_type=Plan,
    system_prompt=PLANNER_SYSTEM_PROMPT,   # explains available step kinds + registry contents
)
```

The system prompt is where you teach the planner what steps are available and how to use them.

## The four built-in step kinds

### `llm` — call a registered agent with a templated prompt

```python
PlannedStep(
    id="extract_facts",
    kind="llm",
    params={
        "agent_name": "extractor",
        "prompt_template": "Extract facts from: {plan_input.document}",
    },
    depends_on=[],
)
```

The `prompt_template` is rendered with `{plan_input.field}` and `{dep_id.field}` substitutions. The agent is looked up in the registry by `agent_name`.

### `callable` — call a registered async function

```python
PlannedStep(
    id="parse_csv",
    kind="callable",
    params={"fn_name": "parse_csv", "args": {"delimiter": ","}},
    depends_on=[],
)
```

The function is invoked as `fn(plan_input=..., dep_outputs=..., **args)`. Used for deterministic data manipulation that doesn't need an LLM.

### `unit` — run a CoALA unit through its 4-phase lifecycle

```python
PlannedStep(
    id="research",
    kind="unit",
    params={"unit_name": "research_summarize", "input_from": "extract_facts"},
    depends_on=["extract_facts"],
)
```

The framework dispatches `observe` → `retrieve` → `act` → `learn` on the registered unit. Chapter 19 covers CoALA units in depth.

### `workflow` — call a registered async workflow

```python
PlannedStep(
    id="publish",
    kind="workflow",
    params={"workflow_name": "publish_with_approval", "input_from": "research"},
    depends_on=["research"],
)
```

Used when a step is itself a non-trivial durable workflow (e.g., wraps HITL approval).

## The `StepRegistry`

The registry is the bridge between step names in the plan and actual implementations in your app:

```python
from ballast.patterns.plan_execute import StepRegistry

registry = StepRegistry.with_defaults()   # pre-registers "llm" / "callable" / "unit" / "workflow"

registry.register_agent("extractor", extractor_agent)
registry.register_agent("synthesizer", synthesizer_agent)
registry.register_callable("parse_csv", parse_csv_fn)
registry.register_unit("research_summarize", research_unit)
registry.register_workflow("publish_with_approval", publish_workflow)
```

The planner's system prompt lists what's in the registry, so the model knows what step kinds and names are available. Anything not registered won't dispatch.

For custom step kinds:

```python
class MyApiStep:
    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        # ... call my external API
        ...

registry.register_step("api_call", MyApiStep())
```

## The `Step` Protocol

```python
class Step(Protocol):
    async def execute(
        self,
        plan_input: Any,
        dep_outputs: dict[str, Any],
        ctx: StepContext,
    ) -> Any: ...
```

`plan_input` is the original input to the pattern. `dep_outputs` is a dict mapping dependency IDs to their outputs. `ctx` carries metadata (step ID, step kind, current wave, etc.) for observability and breaker scoping.

Any object satisfying this protocol can be registered as a step kind.

## The `PlanAndExecute` pattern

```python
from ballast.patterns.plan_execute import PlanAndExecute, FailLoud

pe = PlanAndExecute(
    planner=planner_agent,
    registry=registry,
    replan_policy=FailLoud(),     # default
    max_parallel=8,
)

result = await pe.run({"document": ...})
```

Returns a dict mapping step IDs to their outputs. Your downstream code picks whichever step result you actually care about.

## Wave-by-wave dispatch

The executor topologically sorts steps into waves: wave 1 has all steps with no unmet dependencies; wave 2 has steps whose dependencies all completed in wave 1; and so on. Each wave runs via `asyncio.gather` bounded by `max_parallel`.

```
Plan steps:
  A: depends_on=[]
  B: depends_on=[]
  C: depends_on=[A]
  D: depends_on=[A, B]
  E: depends_on=[D]

Wave 1: A, B    (run in parallel)
Wave 2: C, D    (run in parallel after wave 1)
Wave 3: E       (after wave 2)
```

Independent steps in the same wave parallelize for free.

## `RePlanPolicy` — recovering from failures

```python
class RePlanPolicy(Protocol):
    async def on_step_failure(
        self,
        plan: Plan,
        failed_step: PlannedStep,
        error: Exception,
        partial_outputs: dict[str, Any],
    ) -> Plan | None: ...
```

Return `None` to fail loud (raise `PlanExecutionError`). Return a new `Plan` to continue execution with a revised DAG.

The default `FailLoud` always returns `None`. A real re-planning policy might:

```python
class LLMReplan:
    async def on_step_failure(self, plan, failed_step, error, partial_outputs):
        # Ask the planner to revise based on what worked and what didn't.
        prompt = f"Previous plan failed at {failed_step.id} with {error}. Revise."
        result = await replanner.run(prompt)
        return result.output
```

Re-planning is powerful but adds latency and complexity. Start with `FailLoud`; add re-planning when you can measure that failure recovery is worth the spend.

## Wrapping steps with `CircuitBreaker`

```python
from ballast.resilience.circuit_breaker import CircuitBreaker, as_step

breaker = CircuitBreaker(
    threshold_factory=lambda: Consecutive(3),
    fallback=ReturnValue({"status": "degraded"}),
)

original_step = LLMStep(...)
guarded_step = as_step(breaker, original_step)

registry.register_step("llm_guarded", guarded_step)
```

The breaker's scope_key receives `{"step_id": ..., "step_kind": ...}`, so per-step buckets work out of the box. Three consecutive failures of the same step kind → breaker opens → subsequent invocations short-circuit.

## Composition: PlanAndExecute inside a workflow

```python
@Durable.workflow
async def my_top_level_flow(input: dict) -> dict:
    pe_result = await pe.run(input)
    final = await post_process(pe_result["synthesize"])
    return {"output": final, "intermediate": pe_result}
```

`PlanAndExecute.run()` is itself a `@Durable.workflow`. Calling it from another workflow nests cleanly — DBOS handles the parent/child relationship and the inner workflow gets its own ID for the inspector tree.

## Common mistakes

- **Planner sees registry items but doesn't know how to use them.** Write the planner's system prompt with explicit examples: "Use kind='llm' with agent_name='extractor' when you need to extract facts."
- **Steps with implicit dependencies.** If step B *uses* output of step A but doesn't declare `depends_on=["A"]`, the executor might run them in parallel and B gets no input. Always declare dependencies.
- **`max_parallel` too high for the model provider.** Same problem as MapReduce — rate limits. 8 is usually safe; tune from there.
- **Re-planning on every failure.** Each re-plan is an LLM call. If every transient blip triggers re-planning, you've doubled cost. Combine re-plan with a circuit breaker: re-plan when the breaker opens, not on every error.
- **Letting the planner emit too many steps.** A 30-step plan is a planner that didn't decompose well. Cap the plan size in the system prompt: "Output at most 8 steps."

## What this chapter did NOT cover

- The CoALA unit pattern — chapter 19.
- Composing PlanAndExecute with capability stacks — chapter 20.
- Writing a custom step kind — chapter 25.
- Re-planning policies in depth — `docs/explanation/architecture-overview.md`.

## Where to go next

→ [19-cognitive-units.md](19-cognitive-units.md) — the CoALA cognitive primitive.

# Plan-and-Execute Pattern — Design Spec

**Date:** 2026-05-26
**Status:** Approved (proceeding to plan)
**Source motivation:** "Архитектура и надёжность агентных LLM-систем в Production" — section on Plan-and-Execute as the production-friendly alternative to ReAct (separates strategic planning from tactical execution; planner does NOT participate in execution; lightweight executors close each node).

## Problem

ReAct loop calls a dollar-tier LLM on every mini-step and plans only one move ahead. For multi-stage durable pipelines this is expensive, slow, and prone to drift. Plan-and-Execute decouples:

1. **Planning** — one expensive call to a planner-agent that emits the full DAG of steps with dependencies.
2. **Execution** — framework dispatches each step to a registered handler (LLM call, callable, CoALA unit, sub-workflow). Independent steps run in parallel.

The framework currently has `MapReduce` / `Reflection` / `DivergentConvergent` patterns but no general-purpose planner-driven pipeline runner. This spec defines `PlanAndExecute`.

## Goals

- Single pattern entry point — `PlanAndExecute(planner=, registry=).run(input)`.
- Plan shape = DAG with explicit `depends_on`. Linear sequences expressible as degenerate DAGs.
- Pluggable step kinds via `Step` Protocol + `StepRegistry`. Built-ins: `llm` / `callable` / `unit` / `workflow`.
- Framework dispatcher executes steps; no executor-agent middleman.
- Re-plan extension via `RePlanPolicy` Protocol (first cut: `FailLoud` only).
- Durable through DBOS — `@Durable.dbos_class()` + `@Durable.step` per phase. Crash recovery preserves completed steps' outputs.
- Composable with existing primitives — `MapReduce` / `CoALAUnit` / sub-workflow as a step.

## Non-goals

- Continuous (per-step) scheduling — first cut is wave-by-wave.
- Conditional steps as first-class concept; apps express via replan policy or custom step impl.
- Per-step timeouts as plan properties (apps wrap in `asyncio.wait_for` inside their Step).
- Streaming intermediate step outputs out of pattern — follow-up if needed.
- Multi-planner consensus / adaptive replan with bounded loop — first cut is `FailLoud`-only.
- Plan visualisation tooling.

## Architecture

### File structure

```
src/ballast/patterns/plan_execute/
  __init__.py             # public re-exports
  _protocols.py           # Step + RePlanPolicy Protocols + StepContext vehicle
  _plan.py                # Plan + PlannedStep BaseModels + Plan.validator
  _steps.py               # built-in Step impls: LLMStep / CallableStep / UnitStep / WorkflowStep
  _registry.py            # StepRegistry — apps register agents/callables/units/workflows
  _policies.py            # RePlanPolicy built-ins: FailLoud
  _errors.py              # PlanExecutionError
  _executor.py            # DAG traversal helper (orchestrates _execute_step calls)
  _pattern.py             # PlanAndExecute(DBOSConfiguredInstance) — entry point

tests/patterns/plan_execute/
  __init__.py
  test_plan.py            # Plan validator
  test_steps.py           # each built-in Step impl
  test_registry.py        # StepRegistry get/register + error messages
  test_executor.py        # DAG traversal: linear / diamond / parallel root, deadlock
  test_policies.py        # FailLoud + custom replan
  test_pattern.py         # PlanAndExecute.run end-to-end (DBOS fixture)
```

**Размещение:** под `src/ballast/patterns/` рядом с `map_reduce/`, `mutation/`, `hitl/`. Consistent с MapReduce'ом — собственный subpackage т.к. много файлов.

### Public API

`from ballast.patterns.plan_execute import ...`:
- `PlanAndExecute` (entry point)
- `Plan`, `PlannedStep` (data models)
- `Step`, `RePlanPolicy` (Protocols)
- `StepContext` (vehicle)
- `StepRegistry`
- `LLMStep`, `CallableStep`, `UnitStep`, `WorkflowStep` (built-in impls)
- `FailLoud` (policy)
- `PlanExecutionError`

Top-level `from ballast import PlanAndExecute` — yes, для согласованности с `MapReduce` / `Reflection`.

## Components

### Vehicle types

```python
@dataclass
class StepContext:
    """Read-only context passed to Step.execute."""
    plan: Plan                           # full DAG
    step: PlannedStep                    # which step is executing
    step_registry: "StepRegistry"        # for cross-step lookups
    workflow_id: str | None              # DBOS workflow id (None outside workflow)
```

### Data models

```python
class PlannedStep(BaseModel):
    id: str
    kind: str                            # registry key (e.g. "llm", "callable", "unit", "workflow", or app-defined)
    params: dict[str, Any]               # kind-specific config
    depends_on: list[str] = []
    description: str = ""                # planner's human-readable rationale


class Plan(BaseModel):
    steps: list[PlannedStep]
    rationale: str = ""

    @model_validator(mode="after")
    def _validate_dag(self) -> "Plan":
        # 1. Unique step ids
        # 2. No dangling deps (every id in depends_on exists)
        # 3. No cycles (topological sort succeeds)
        # 4. Steps list may be empty (no-op plan is valid)
        ...
```

### Protocols

```python
@runtime_checkable
class Step(Protocol):
    """How to execute one planned step. Stateless; framework invokes with resolved inputs."""
    async def execute(
        self,
        plan_input: Any,
        dep_outputs: dict[str, Any],
        ctx: StepContext,
    ) -> Any: ...


@runtime_checkable
class RePlanPolicy(Protocol):
    """When/whether to invoke planner again after step failure."""
    async def on_step_failure(
        self,
        plan: Plan,
        failed_step: PlannedStep,
        error: Exception,
        partial_outputs: dict[str, Any],
    ) -> Plan | None:
        """Return new Plan to continue with (preserving partial_outputs), or None to fail loud."""
```

### Built-in Step impls

#### `LLMStep`

```python
class LLMStep:
    """Run a registered Agent with a templated prompt.

    Planner emits: PlannedStep(kind="llm", params={
        "agent_name": "researcher",
        "prompt_template": "Summarize {dep_a} given {plan_input.topic}",
        "output_field": "summary",   # optional — extract sub-field of agent output
    })

    Apps register: registry.register_agent("researcher", researcher_agent).
    """
    def __init__(self, registry: StepRegistry): self._registry = registry

    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        params = ctx.step.params
        agent = self._registry.get_agent(params["agent_name"])
        prompt = self._render(params["prompt_template"], plan_input, dep_outputs)
        result = await agent.run(prompt)
        output = result.output
        if "output_field" in params:
            output = getattr(output, params["output_field"], output)
        return output

    def _render(self, template: str, plan_input, dep_outputs) -> str:
        """Simple substitution: {plan_input}, {plan_input.x}, {dep_id}, {dep_id.field}.
        f-string-like; no full Jinja."""
        ...
```

#### `CallableStep`

```python
class CallableStep:
    """Run a registered async function.

    Planner emits: PlannedStep(kind="callable", params={"fn_name": "scrape", "args": {...}})
    Apps register: registry.register_callable("scrape", scrape_url_async)
    """
    def __init__(self, registry: StepRegistry): self._registry = registry

    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        params = ctx.step.params
        fn = self._registry.get_callable(params["fn_name"])
        return await fn(
            plan_input=plan_input,
            dep_outputs=dep_outputs,
            **params.get("args", {}),
        )
```

#### `UnitStep`

```python
class UnitStep:
    """Run a registered CoALAUnit through its 4-phase lifecycle.

    Planner emits: PlannedStep(kind="unit", params={
        "unit_name": "research_summarize",
        "input_from": "research",       # optional — use dep output as unit input
    })
    Apps register: registry.register_unit("research_summarize", ResearchSummarize())
    """
    def __init__(self, registry: StepRegistry): self._registry = registry

    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        params = ctx.step.params
        unit = self._registry.get_unit(params["unit_name"])
        unit_input = dep_outputs[params["input_from"]] if "input_from" in params else plan_input
        obs = await unit.observe(unit_input)
        retrieved = await unit.retrieve(obs)
        out = await unit.act(obs, retrieved)
        await unit.learn(obs, retrieved, out)
        return out
```

#### `WorkflowStep`

```python
class WorkflowStep:
    """Run a registered Durable workflow as a sub-step.

    Planner emits: PlannedStep(kind="workflow", params={
        "workflow_name": "publish_note",
        "input_from": "draft",     # optional
    })
    Apps register: registry.register_workflow("publish_note", publish_note_flow)
    """
    def __init__(self, registry: StepRegistry): self._registry = registry

    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        params = ctx.step.params
        workflow = self._registry.get_workflow(params["workflow_name"])
        wf_input = dep_outputs[params["input_from"]] if "input_from" in params else plan_input
        return await workflow(wf_input)
```

### `StepRegistry`

```python
class StepRegistry:
    """Apps populate before PlanAndExecute.run(). Planner emits step.kind + step.params;
    framework dispatches via this registry."""

    def __init__(self) -> None:
        self._steps:     dict[str, Step]      = {}
        self._agents:    dict[str, Agent]     = {}
        self._callables: dict[str, Callable]  = {}
        self._units:     dict[str, CoALAUnit] = {}
        self._workflows: dict[str, Callable]  = {}

    def register_step(self, kind: str, impl: Step) -> None: ...
    def register_agent(self, name: str, agent: Agent) -> None: ...
    def register_callable(self, name: str, fn: Callable) -> None: ...
    def register_unit(self, name: str, unit: CoALAUnit) -> None: ...
    def register_workflow(self, name: str, wf: Callable) -> None: ...

    def get_step(self, kind: str) -> Step:
        if kind not in self._steps:
            raise KeyError(f"step kind {kind!r} not registered; "
                           f"available: {sorted(self._steps)}")
        return self._steps[kind]
    # matching getters for agent/callable/unit/workflow, with helpful KeyError messages

    @classmethod
    def with_defaults(cls) -> "StepRegistry":
        """Pre-register LLMStep/CallableStep/UnitStep/WorkflowStep with their kinds."""
        r = cls()
        r.register_step("llm",      LLMStep(r))
        r.register_step("callable", CallableStep(r))
        r.register_step("unit",     UnitStep(r))
        r.register_step("workflow", WorkflowStep(r))
        return r
```

### Built-in `RePlanPolicy`

```python
class FailLoud:
    """No re-planning. Step failure raises PlanExecutionError."""
    async def on_step_failure(self, plan, failed_step, error, partial_outputs) -> None:
        return None
```

### `PlanAndExecute` pattern

```python
@Durable.dbos_class()
class PlanAndExecute(DBOSConfiguredInstance, Generic[InT, OutT]):
    """Plan-and-Execute pattern — planner emits DAG, framework executes.

    Two-phase durable workflow:
      1. _plan_step — call planner.run(input) → Plan (memoised)
      2. _execute_dag — traverse + run steps with asyncio.gather on independent branches
         Each _execute_step call is @Durable.step (memoised; replay-safe).
    """

    def __init__(
        self, *,
        planner: Agent[None, Plan],
        registry: StepRegistry,
        replan_policy: RePlanPolicy | None = None,
        max_parallel: int = 8,
    ) -> None:
        super().__init__(config_name=f"{type(self).__qualname__}-{next(_instance_counter)}")
        self._planner = planner
        self._registry = registry
        self._replan_policy = replan_policy or FailLoud()
        self._max_parallel = max_parallel

    @Durable.workflow()
    async def run(self, input: InT) -> dict[str, Any]:
        """Returns dict of {step_id: output} for ALL completed steps.
        Apps extract the final result by terminal step_id lookup."""
        plan = await self._plan_step(input)
        outputs = await self._execute_dag(input, plan)
        return outputs

    @Durable.step()
    async def _plan_step(self, input: InT) -> Plan:
        result = await self._planner.run(_serialize_input(input))
        return result.output

    async def _execute_dag(self, plan_input: InT, plan: Plan) -> dict[str, Any]:
        """Wave-by-wave DAG traversal with asyncio.gather. Not a step itself —
        dispatches per-step calls (which ARE steps)."""
        outputs: dict[str, Any] = {}
        pending = {s.id: s for s in plan.steps}
        sem = asyncio.Semaphore(self._max_parallel)

        while pending:
            ready = [s for s in pending.values()
                     if all(dep in outputs for dep in s.depends_on)]
            if not ready:
                raise RuntimeError(
                    f"plan deadlock: {len(pending)} steps remain, none ready. "
                    f"Pending: {list(pending)}"
                )

            async def _run_one(step: PlannedStep) -> tuple[str, Any]:
                async with sem:
                    return step.id, await self._execute_step(plan, step, plan_input, outputs)

            batch_results = await asyncio.gather(
                *(_run_one(s) for s in ready),
                return_exceptions=True,
            )
            failure_handled = False
            for result in batch_results:
                if isinstance(result, Exception):
                    # Find which ready step failed (matching by exception identity is fragile;
                    # use parallel index since gather preserves order).
                    failed_step = ready[batch_results.index(result)]
                    new_plan = await self._replan_policy.on_step_failure(
                        plan=plan, failed_step=failed_step,
                        error=result, partial_outputs=outputs,
                    )
                    if new_plan is None:
                        raise PlanExecutionError(
                            f"step {failed_step.id!r} failed: {result}",
                            failed_step=failed_step,
                            partial_outputs=outputs,
                        ) from result
                    plan = new_plan
                    pending = {s.id: s for s in plan.steps if s.id not in outputs}
                    failure_handled = True
                    break  # restart loop with new plan
                else:
                    step_id, output = result
                    outputs[step_id] = output
                    del pending[step_id]
            if failure_handled:
                continue
        return outputs

    @Durable.step()
    async def _execute_step(
        self,
        plan: Plan,
        planned: PlannedStep,
        plan_input: InT,
        outputs: dict[str, Any],
    ) -> Any:
        """One step execution — memoised per (config_name, step args)."""
        step_impl = self._registry.get_step(planned.kind)
        dep_outputs = {dep_id: outputs[dep_id] for dep_id in planned.depends_on}
        ctx = StepContext(
            plan=plan, step=planned,
            step_registry=self._registry,
            workflow_id=Durable.current_workflow_id(),
        )
        return await step_impl.execute(plan_input, dep_outputs, ctx)
```

## Data flow

```
[caller] PlanAndExecute(planner, registry).run(input)
    │
    ▼
[@Durable.workflow] run(input)
    │
    ├── [@Durable.step] _plan_step(input) → Plan        (memoised — planner LLM call)
    │
    ▼
_execute_dag(input, plan):
  loop:
    ready = [s for s in pending if all deps satisfied]
    if no ready and pending nonempty → RuntimeError(deadlock)
    if no pending → return outputs

    batch = await asyncio.gather(
      *[_execute_step(plan, s, input, outputs) for s in ready],
      return_exceptions=True
    )

    for result in batch:
      if Exception:
        new_plan = await replan_policy.on_step_failure(plan, failed_step, error, outputs)
        if new_plan is None: raise PlanExecutionError
        else: plan = new_plan; rebuild pending; continue
      else:
        outputs[step.id] = result; remove from pending

[@Durable.step] _execute_step(plan, planned, plan_input, outputs):
  step_impl = registry.get_step(planned.kind)
  dep_outputs = {dep_id: outputs[dep_id] for dep_id in planned.depends_on}
  ctx = StepContext(...)
  return await step_impl.execute(plan_input, dep_outputs, ctx)
```

## Error handling

| Layer | Behavior |
|---|---|
| `Plan.__init__` validator (cycle/dangling/dup) | `pydantic.ValidationError` immediately |
| `_plan_step` planner exception | Bubbles up (no fail-safe); DBOS retry per workflow config |
| `_execute_step` exception | Caught by `gather(return_exceptions=True)`; routed to `replan_policy.on_step_failure` |
| `FailLoud` returns `None` | `PlanExecutionError(failed_step, partial_outputs)` raised, chained `from error` |
| Custom replan returns `Plan` | Executor restarts loop preserving completed `outputs` |
| `_execute_dag` deadlock | `RuntimeError("plan deadlock: ...")` |
| `StepRegistry.get_*` missing | `KeyError` with helpful "available: [...]" message — bubbles via gather to replan_policy |

`PlanExecutionError` subclass of `BallastError`; `code="BALLAST_PLAN_EXECUTION"`; carries `failed_step: PlannedStep` and `partial_outputs: dict[str, Any]` attributes.

## DAG execution semantics

**Wave-by-wave:**
- Each iteration finds all `ready` steps (all `depends_on` satisfied), runs them via `asyncio.gather` with semaphore (`max_parallel`).
- Next wave starts after current wave fully completes (including failure handling).
- Alternative (continuous scheduling — start step as soon as its deps complete, regardless of other parallel steps) is **out of scope** for first cut. Wave model is simpler; can upgrade later without breaking API.

**Empty plan:** `_execute_dag` returns `{}` immediately. Valid.

**Single-step plan:** Degenerate DAG; one wave with one step. Works.

**Failed step in parallel batch:**
- `gather(return_exceptions=True)` ensures sibling steps in the wave complete; their outputs are saved to `partial_outputs` before invoking `replan_policy`.
- If multiple steps fail simultaneously — take the first (by `ready` order). Apps observe full `partial_outputs` and can decide via custom `replan_policy`.

**Replan semantics:**
- Completed steps' outputs preserved across replans. Planner can reference them in `new_plan` by step.id — those values are available as deps.
- Replan is unbounded in first cut — `FailLoud` is the only built-in policy. Apps writing custom `OnFailure(max_replans=N)` are responsible for termination.

**Idempotency:**
- `_execute_step` is `@Durable.step` on `DBOSConfiguredInstance`. DBOS memoises by `(config_name, function name, args)`.
- All `_execute_step` args (`plan`, `planned`, `plan_input`, `outputs`) must be picklable. `Plan`/`PlannedStep` are pydantic BaseModels (picklable); `outputs` is a dict of picklable values (apps' step outputs must be picklable too). `plan_input` is caller-supplied — apps responsible.

## Testing strategy

```
tests/patterns/plan_execute/
  test_plan.py            # Plan validator: cycles, dangling deps, dup ids, empty plan, terminal-only
  test_steps.py           # LLMStep:
                          #   - prompt template rendering: {plan_input}, {plan_input.x}, {dep_id}, {dep_id.field}
                          #   - output_field extraction
                          #   - missing agent raises KeyError with available list
                          # CallableStep:
                          #   - fn invocation with plan_input + dep_outputs + extra args
                          #   - missing callable raises KeyError
                          # UnitStep:
                          #   - 4-phase invocation order (observe → retrieve → act → learn)
                          #   - input_from dep redirection
                          # WorkflowStep:
                          #   - workflow invocation with input
                          #   - input_from dep redirection
  test_registry.py        # register/get parity for each kind
                          # with_defaults pre-registration
                          # KeyError messages list available names
  test_executor.py        # DAG traversal:
                          #   - linear (a → b → c)
                          #   - diamond (a → b/c → d)
                          #   - all-parallel root (3 independent steps)
                          #   - max_parallel semaphore caps concurrency (sleep-based test)
                          #   - empty plan returns {}
                          #   - deadlock detection (manually constructed bypass-validator plan)
  test_policies.py        # FailLoud returns None → PlanExecutionError raised with failed_step + partial_outputs
                          # Custom policy returns new_plan → executor continues, completed outputs preserved
  test_pattern.py         # PlanAndExecute.run end-to-end with TestModel planner:
                          #   - happy path (linear plan)
                          #   - planner output rejected (ValidationError from Plan validator)
                          #   - replay-safety (DBOS workflow restart skips completed _execute_step calls)
                          # Uses tests/coala/conftest.py-style DBOS fixture
```

## Integration с существующими primitives

- **`MapReduce`** — registered as `WorkflowStep` (wrap `mr.run` as an async fn) or `CallableStep`. Pattern compose-able.
- **`Reflection` / `DivergentConvergent`** — same: registered as callable or workflow.
- **`CoALAUnit`** — native via `UnitStep`. Apps register unit by name; planner references it.
- **`@Durable.workflow`** functions — native via `WorkflowStep`.
- **`BallastCapability`s** (BudgetGuard, SemanticLoopDetector, GoalDriftDetector) — applied to agents WITHIN steps (e.g. to `agent` in `LLMStep`). Pattern itself doesn't attach capabilities; apps manage at agent construction.
- **HITL** — apps may register a `Step` impl that calls `HITLChannel.request` (e.g., "approve this draft before continuing"). Custom step.

**Recursive Plan-and-Execute as a step:** A `PlanAndExecute` instance's `run` method is `@Durable.workflow`-wrapped. Apps can register it as a `WorkflowStep` in another `PlanAndExecute` — hierarchical planning via composition, no special recursive-planner mode.

## Demo (notes-app, optional follow-up)

A `ResearchAndPublishFlow` could exercise the full pattern:
1. `research` (LLMStep with researcher agent)
2. `summarize` (UnitStep with `ResearchSummarize` CoALAUnit, deps=[research])
3. `propose_note` (WorkflowStep with HITL approval, deps=[summarize])
4. `publish` (CallableStep, deps=[propose_note])

This is **out of scope** for the framework spec; mentioned for clarity. Implementation plan focuses on framework + framework tests only.

## Open questions for review

None — all design decisions resolved during brainstorm. User pre-approved defaults:
1. Plan shape: DAG with `depends_on`
2. Step protocol: pluggable via `Step` interface + `StepRegistry`
3. Executor: framework dispatcher (no executor-agent middleman)
4. Re-plan: `RePlanPolicy` Protocol with `FailLoud` only in first cut
5. Dep passing: `execute(plan_input, dep_outputs, ctx)` — apps type internally
6. Pattern shape: single `PlanAndExecute(DBOSConfiguredInstance)` mirroring `MapReduce`
7. DAG semantics: wave-by-wave with `asyncio.gather` + semaphore

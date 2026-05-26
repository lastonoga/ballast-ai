# Plan-and-Execute Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Plan-and-Execute pattern: a `PlanAndExecute(DBOSConfiguredInstance)` class that takes a `planner: Agent[None, Plan]` and a `StepRegistry`, then on `.run(input)` calls planner → executes the resulting DAG of steps via framework dispatcher (no executor-agent middleman). 4 built-in step kinds (LLM / callable / unit / workflow) + custom kinds via Protocol.

**Architecture:** Same `DBOSConfiguredInstance` pattern as `MapReduce` — `@Durable.dbos_class()` wrapper class with `@Durable.workflow` `run()` method and `@Durable.step` per-step memoised methods. Wave-by-wave DAG traversal with `asyncio.gather(return_exceptions=True)` + semaphore. `Step` Protocol + `StepRegistry` for pluggable execution kinds. `RePlanPolicy` Protocol for failure routing (first cut: `FailLoud` only).

**Tech Stack:** Python 3.11+, pydantic v2 (Plan/PlannedStep models + validator), pydantic-ai (`Agent[None, Plan]` typed planner), DBOS (`Durable.workflow / step / dbos_class`), existing `BallastError` / `CoALAUnit` / async patterns.

**Spec:** `docs/superpowers/specs/2026-05-26-plan-and-execute-design.md`

---

## File Structure (reference for all tasks)

```
src/ballast/patterns/plan_execute/
  __init__.py             # public re-exports (Task 11)
  _protocols.py           # Step + RePlanPolicy Protocols + StepContext (Task 2)
  _plan.py                # Plan + PlannedStep + validator (Task 1)
  _registry.py            # StepRegistry (Task 5)
  _steps.py               # 4 built-in Step impls (Tasks 6-9)
  _policies.py            # FailLoud (Task 4)
  _errors.py              # PlanExecutionError (Task 3)
  _pattern.py             # PlanAndExecute(DBOSConfiguredInstance) (Task 10)

tests/patterns/plan_execute/
  __init__.py
  conftest.py             # reuse tests/coala/conftest.py-style DBOS fixture (Task 10)
  test_plan.py            # Plan validator (Task 1)
  test_protocols.py       # Step + RePlanPolicy runtime_checkable (Task 2)
  test_errors.py          # PlanExecutionError (Task 3)
  test_policies.py        # FailLoud (Task 4)
  test_registry.py        # StepRegistry (Task 5)
  test_steps_llm.py       # LLMStep + prompt rendering (Task 6)
  test_steps_callable.py  # CallableStep (Task 7)
  test_steps_unit.py      # UnitStep (Task 8)
  test_steps_workflow.py  # WorkflowStep (Task 9)
  test_pattern.py         # PlanAndExecute.run end-to-end (Task 10)
```

---

## Task 1: `Plan` + `PlannedStep` + DAG validator

**Files:**
- Create: `src/ballast/patterns/plan_execute/__init__.py` (empty for now)
- Create: `src/ballast/patterns/plan_execute/_plan.py`
- Create: `tests/patterns/plan_execute/__init__.py` (empty)
- Create: `tests/patterns/plan_execute/test_plan.py`

- [ ] **Step 1: Write the failing test (`tests/patterns/plan_execute/test_plan.py`)**

```python
"""Plan + PlannedStep + DAG validator."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from ballast.patterns.plan_execute._plan import Plan, PlannedStep


def _step(id: str, deps: list[str] = ()) -> PlannedStep:
    return PlannedStep(id=id, kind="llm", params={}, depends_on=list(deps))


def test_empty_plan_is_valid() -> None:
    p = Plan(steps=[])
    assert p.steps == []
    assert p.rationale == ""


def test_linear_chain_is_valid() -> None:
    p = Plan(steps=[_step("a"), _step("b", ["a"]), _step("c", ["b"])])
    assert len(p.steps) == 3


def test_diamond_dag_is_valid() -> None:
    p = Plan(steps=[
        _step("a"), _step("b", ["a"]), _step("c", ["a"]), _step("d", ["b", "c"]),
    ])
    assert len(p.steps) == 4


def test_duplicate_step_id_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        Plan(steps=[_step("a"), _step("a")])


def test_dangling_dep_rejected() -> None:
    with pytest.raises(ValidationError, match="dangling"):
        Plan(steps=[_step("a", ["nonexistent"])])


def test_cycle_detected() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        Plan(steps=[_step("a", ["b"]), _step("b", ["a"])])


def test_self_loop_detected() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        Plan(steps=[_step("a", ["a"])])


def test_rationale_field_optional() -> None:
    p = Plan(steps=[], rationale="initial plan")
    assert p.rationale == "initial plan"


def test_planned_step_required_fields() -> None:
    s = PlannedStep(id="a", kind="llm", params={"x": 1})
    assert s.id == "a"
    assert s.depends_on == []
    assert s.description == ""
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/patterns/plan_execute/test_plan.py -v`
Expected: `ModuleNotFoundError: No module named 'ballast.patterns.plan_execute'`.

- [ ] **Step 3: Implement `src/ballast/patterns/plan_execute/_plan.py`**

```python
"""``Plan`` + ``PlannedStep`` — DAG of planner-emitted execution nodes.

``Plan.__init__`` validates the DAG: unique step ids, no dangling
dependencies, no cycles. Apps construct plans either via a typed
planner agent (``Agent[None, Plan]``) or manually for testing.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator


class PlannedStep(BaseModel):
    """One node in the DAG. Planner emits these; executor consumes."""

    id: str
    """Unique within plan; planner picks."""

    kind: str
    """Registry key — ``"llm"`` / ``"callable"`` / ``"unit"`` / ``"workflow"`` / custom."""

    params: dict[str, Any] = {}
    """Kind-specific config — e.g. ``{"agent_name": "...", "prompt_template": "..."}``."""

    depends_on: list[str] = []
    """Other ``PlannedStep.id`` values this step depends on. Empty = root."""

    description: str = ""
    """Human-readable rationale from planner — surfaces in logs / observability."""


class Plan(BaseModel):
    """Full DAG. Validated at construction."""

    steps: list[PlannedStep] = []
    rationale: str = ""

    @model_validator(mode="after")
    def _validate_dag(self) -> "Plan":
        ids = [s.id for s in self.steps]
        seen: set[str] = set()
        for sid in ids:
            if sid in seen:
                raise ValueError(f"Plan has duplicate step id: {sid!r}")
            seen.add(sid)

        # Dangling dep check
        for s in self.steps:
            for dep in s.depends_on:
                if dep not in seen:
                    raise ValueError(
                        f"Step {s.id!r} has dangling dependency: "
                        f"{dep!r} (not in plan)"
                    )

        # Cycle detection via DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {sid: WHITE for sid in seen}
        adj: dict[str, list[str]] = {s.id: list(s.depends_on) for s in self.steps}

        def _dfs(node: str) -> None:
            color[node] = GRAY
            for dep in adj[node]:
                if color[dep] == GRAY:
                    raise ValueError(
                        f"Plan has cycle involving step {node!r} → {dep!r}"
                    )
                if color[dep] == WHITE:
                    _dfs(dep)
            color[node] = BLACK

        for sid in seen:
            if color[sid] == WHITE:
                _dfs(sid)

        return self


__all__ = ["Plan", "PlannedStep"]
```

- [ ] **Step 4: Create empty `__init__.py` package markers**

`src/ballast/patterns/plan_execute/__init__.py` (empty), `tests/patterns/plan_execute/__init__.py` (empty).

- [ ] **Step 5: Run — confirm pass**

Run: `uv run pytest tests/patterns/plan_execute/test_plan.py -v`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/patterns/plan_execute/__init__.py src/ballast/patterns/plan_execute/_plan.py tests/patterns/plan_execute/__init__.py tests/patterns/plan_execute/test_plan.py
git commit -m "feat(plan-execute): Plan + PlannedStep data models with DAG validator"
```

---

## Task 2: Protocols + `StepContext` vehicle

**Files:**
- Create: `src/ballast/patterns/plan_execute/_protocols.py`
- Create: `tests/patterns/plan_execute/test_protocols.py`

- [ ] **Step 1: Failing test (`tests/patterns/plan_execute/test_protocols.py`)**

```python
"""Step + RePlanPolicy Protocols + StepContext vehicle."""
from __future__ import annotations

from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._protocols import (
    RePlanPolicy, Step, StepContext,
)


def test_step_runtime_checkable() -> None:
    class _Stub:
        async def execute(self, plan_input, dep_outputs, ctx): return None
    assert isinstance(_Stub(), Step)

    class _Missing:
        pass
    assert not isinstance(_Missing(), Step)


def test_replan_policy_runtime_checkable() -> None:
    class _Stub:
        async def on_step_failure(self, plan, failed_step, error, partial_outputs):
            return None
    assert isinstance(_Stub(), RePlanPolicy)


def test_step_context_holds_plan_step_registry_workflow_id() -> None:
    plan = Plan(steps=[PlannedStep(id="a", kind="llm")])
    step = plan.steps[0]
    ctx = StepContext(plan=plan, step=step, step_registry=None, workflow_id="wf-1")
    assert ctx.plan is plan
    assert ctx.step is step
    assert ctx.step_registry is None
    assert ctx.workflow_id == "wf-1"


def test_step_context_workflow_id_optional() -> None:
    plan = Plan(steps=[PlannedStep(id="a", kind="llm")])
    ctx = StepContext(plan=plan, step=plan.steps[0], step_registry=None)
    assert ctx.workflow_id is None
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement `src/ballast/patterns/plan_execute/_protocols.py`**

```python
"""Step + RePlanPolicy Protocols + StepContext vehicle.

Apps implement custom step kinds by writing a ``Step``-compatible class
and registering it under a name (see ``StepRegistry``). Apps implement
custom failure handling by writing a ``RePlanPolicy``-compatible class
and passing it to ``PlanAndExecute(replan_policy=...)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ballast.patterns.plan_execute._plan import Plan, PlannedStep


@dataclass
class StepContext:
    """Read-only context passed to ``Step.execute``."""

    plan: Plan
    """Full DAG being executed."""

    step: PlannedStep
    """The specific step being executed."""

    step_registry: Any
    """The ``StepRegistry`` — typed Any to avoid circular import; runtime ducktyped."""

    workflow_id: str | None = None
    """DBOS workflow id (None when running outside a workflow)."""


@runtime_checkable
class Step(Protocol):
    """How to execute one planned step.

    Instances are stateless; the framework calls ``execute()`` with the
    resolved inputs. Apps register a Step class per ``kind`` value the
    planner can emit; framework looks up by kind name.
    """

    async def execute(
        self,
        plan_input: Any,
        dep_outputs: dict[str, Any],
        ctx: StepContext,
    ) -> Any: ...


@runtime_checkable
class RePlanPolicy(Protocol):
    """When/whether to invoke planner again after a step failure.

    Returns:
      ``None`` — fail loud (raise ``PlanExecutionError`` with failed_step + partial_outputs).
      ``Plan`` — new DAG to continue with. Executor preserves completed-step outputs;
                 ``new_plan`` may reference them by step.id as dependencies.
    """

    async def on_step_failure(
        self,
        plan: Plan,
        failed_step: PlannedStep,
        error: Exception,
        partial_outputs: dict[str, Any],
    ) -> Plan | None: ...


__all__ = ["RePlanPolicy", "Step", "StepContext"]
```

- [ ] **Step 4: Run — confirm pass**

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/patterns/plan_execute/_protocols.py tests/patterns/plan_execute/test_protocols.py
git commit -m "feat(plan-execute): Step + RePlanPolicy Protocols + StepContext vehicle"
```

---

## Task 3: `PlanExecutionError`

**Files:**
- Create: `src/ballast/patterns/plan_execute/_errors.py`
- Create: `tests/patterns/plan_execute/test_errors.py`

- [ ] **Step 1: Failing test (`tests/patterns/plan_execute/test_errors.py`)**

```python
"""PlanExecutionError — raised under FailLoud."""
from __future__ import annotations

import pytest

from ballast.errors import BallastError
from ballast.patterns.plan_execute._errors import PlanExecutionError
from ballast.patterns.plan_execute._plan import PlannedStep


def test_plan_execution_error_subclass_of_ballast_error() -> None:
    assert issubclass(PlanExecutionError, BallastError)


def test_plan_execution_error_has_code() -> None:
    assert PlanExecutionError.code == "BALLAST_PLAN_EXECUTION"


def test_plan_execution_error_carries_failed_step_and_partial_outputs() -> None:
    step = PlannedStep(id="x", kind="llm")
    exc = PlanExecutionError(
        "step failed",
        failed_step=step,
        partial_outputs={"a": "done", "b": 42},
    )
    assert exc.failed_step is step
    assert exc.partial_outputs == {"a": "done", "b": 42}
    assert "step failed" in str(exc)


def test_plan_execution_error_chain_via_from() -> None:
    step = PlannedStep(id="x", kind="llm")
    original = RuntimeError("network down")
    try:
        try:
            raise original
        except RuntimeError as cause:
            raise PlanExecutionError(
                "step x failed",
                failed_step=step,
                partial_outputs={},
            ) from cause
    except PlanExecutionError as exc:
        assert exc.__cause__ is original
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement `src/ballast/patterns/plan_execute/_errors.py`**

```python
"""``PlanExecutionError`` — raised when a step fails and ``RePlanPolicy``
returns ``None`` (fail-loud)."""
from __future__ import annotations

from typing import Any

from ballast.errors import BallastError
from ballast.patterns.plan_execute._plan import PlannedStep


class PlanExecutionError(BallastError):  # noqa: N818
    """A step's execution failed and the configured ``RePlanPolicy``
    declined to provide a new plan.

    Carries ``failed_step`` and ``partial_outputs`` for debugging /
    higher-level recovery in calling workflows.
    """

    code = "BALLAST_PLAN_EXECUTION"
    status_code = 422

    def __init__(
        self,
        message: str,
        *,
        failed_step: PlannedStep,
        partial_outputs: dict[str, Any],
    ) -> None:
        self.failed_step = failed_step
        self.partial_outputs = partial_outputs
        super().__init__(
            message,
            hint=(
                "A planned step failed and no replan policy supplied a recovery plan. "
                "Inspect failed_step + partial_outputs to decide whether to retry, "
                "expand the planner's instructions, or wire a custom RePlanPolicy."
            ),
            context={
                "failed_step_id": failed_step.id,
                "failed_step_kind": failed_step.kind,
                "completed_step_ids": sorted(partial_outputs),
            },
        )


__all__ = ["PlanExecutionError"]
```

- [ ] **Step 4: Run — confirm pass**

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/patterns/plan_execute/_errors.py tests/patterns/plan_execute/test_errors.py
git commit -m "feat(plan-execute): PlanExecutionError (subclass of BallastError)"
```

---

## Task 4: `FailLoud` RePlanPolicy

**Files:**
- Create: `src/ballast/patterns/plan_execute/_policies.py`
- Create: `tests/patterns/plan_execute/test_policies.py`

- [ ] **Step 1: Failing test (`tests/patterns/plan_execute/test_policies.py`)**

```python
"""FailLoud RePlanPolicy — only built-in in first cut."""
from __future__ import annotations

import pytest

from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._policies import FailLoud
from ballast.patterns.plan_execute._protocols import RePlanPolicy


def test_fail_loud_satisfies_replan_policy_protocol() -> None:
    assert isinstance(FailLoud(), RePlanPolicy)


@pytest.mark.asyncio
async def test_fail_loud_returns_none() -> None:
    plan = Plan(steps=[PlannedStep(id="a", kind="llm")])
    out = await FailLoud().on_step_failure(
        plan=plan, failed_step=plan.steps[0],
        error=RuntimeError("oops"), partial_outputs={},
    )
    assert out is None
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement `src/ballast/patterns/plan_execute/_policies.py`**

```python
"""Built-in ``RePlanPolicy`` implementations.

First cut ships only ``FailLoud``. Future ``OnFailure(planner, max_replans=N)``
will allow adaptive recovery without infinite-replan risk.
"""
from __future__ import annotations

from typing import Any

from ballast.patterns.plan_execute._plan import Plan, PlannedStep


class FailLoud:
    """No re-planning. Step failure → ``PlanExecutionError`` raised by executor."""

    async def on_step_failure(
        self,
        plan: Plan,
        failed_step: PlannedStep,
        error: Exception,
        partial_outputs: dict[str, Any],
    ) -> Plan | None:
        return None


__all__ = ["FailLoud"]
```

- [ ] **Step 4: Run — confirm pass**

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/patterns/plan_execute/_policies.py tests/patterns/plan_execute/test_policies.py
git commit -m "feat(plan-execute): FailLoud RePlanPolicy"
```

---

## Task 5: `StepRegistry`

**Files:**
- Create: `src/ballast/patterns/plan_execute/_registry.py`
- Create: `tests/patterns/plan_execute/test_registry.py`

- [ ] **Step 1: Failing test (`tests/patterns/plan_execute/test_registry.py`)**

```python
"""StepRegistry — apps populate, framework dispatches."""
from __future__ import annotations

import pytest

from ballast.patterns.plan_execute._registry import StepRegistry


class _FakeStep:
    async def execute(self, plan_input, dep_outputs, ctx): return "out"


def test_register_get_step() -> None:
    r = StepRegistry()
    s = _FakeStep()
    r.register_step("custom", s)
    assert r.get_step("custom") is s


def test_get_unknown_step_raises_helpful_keyerror() -> None:
    r = StepRegistry()
    r.register_step("foo", _FakeStep())
    with pytest.raises(KeyError, match="bar") as exc:
        r.get_step("bar")
    assert "foo" in str(exc.value)
    assert "available" in str(exc.value)


def test_register_get_agent_callable_unit_workflow() -> None:
    r = StepRegistry()
    obj_a, obj_b, obj_c, obj_d = object(), object(), object(), object()
    r.register_agent("ag", obj_a)
    r.register_callable("cb", obj_b)
    r.register_unit("un", obj_c)
    r.register_workflow("wf", obj_d)
    assert r.get_agent("ag") is obj_a
    assert r.get_callable("cb") is obj_b
    assert r.get_unit("un") is obj_c
    assert r.get_workflow("wf") is obj_d


def test_get_unknown_agent_callable_unit_workflow_raises_keyerror() -> None:
    r = StepRegistry()
    r.register_agent("foo", object())
    with pytest.raises(KeyError, match="bar"):
        r.get_agent("bar")
    with pytest.raises(KeyError):
        r.get_callable("nope")
    with pytest.raises(KeyError):
        r.get_unit("nope")
    with pytest.raises(KeyError):
        r.get_workflow("nope")


def test_with_defaults_preregisters_four_step_kinds() -> None:
    r = StepRegistry.with_defaults()
    assert r.get_step("llm") is not None
    assert r.get_step("callable") is not None
    assert r.get_step("unit") is not None
    assert r.get_step("workflow") is not None
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement `src/ballast/patterns/plan_execute/_registry.py`**

```python
"""``StepRegistry`` — apps register agents / callables / units / workflows
under names; planner references them in ``PlannedStep.params``; framework
dispatches via this registry.

``with_defaults()`` pre-registers the four built-in step kinds (``llm``,
``callable``, ``unit``, ``workflow``) so apps only need to register their
own agents / callables / units / workflows by name.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ballast.patterns.plan_execute._protocols import Step

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_ai import Agent

    from ballast.coala import CoALAUnit


def _err(kind: str, key: str, available: list[str]) -> KeyError:
    return KeyError(
        f"{kind} {key!r} not registered; available: {sorted(available)}"
    )


class StepRegistry:
    """Apps populate this before ``PlanAndExecute.run()``.

    Registration is name-keyed; planner emits ``step.kind`` + ``step.params``
    referencing those names. Framework dispatches via this registry without
    any reflection or magic.
    """

    def __init__(self) -> None:
        self._steps:     dict[str, Step]                = {}
        self._agents:    dict[str, "Agent[Any, Any]"]   = {}
        self._callables: dict[str, "Callable[..., Any]"] = {}
        self._units:     dict[str, "CoALAUnit"]         = {}
        self._workflows: dict[str, "Callable[..., Any]"] = {}

    # ---- register --------------------------------------------------------

    def register_step(self, kind: str, impl: Step) -> None:
        self._steps[kind] = impl

    def register_agent(self, name: str, agent: "Agent[Any, Any]") -> None:
        self._agents[name] = agent

    def register_callable(self, name: str, fn: "Callable[..., Any]") -> None:
        self._callables[name] = fn

    def register_unit(self, name: str, unit: "CoALAUnit") -> None:
        self._units[name] = unit

    def register_workflow(self, name: str, wf: "Callable[..., Any]") -> None:
        self._workflows[name] = wf

    # ---- get -------------------------------------------------------------

    def get_step(self, kind: str) -> Step:
        if kind not in self._steps:
            raise _err("step kind", kind, list(self._steps))
        return self._steps[kind]

    def get_agent(self, name: str) -> "Agent[Any, Any]":
        if name not in self._agents:
            raise _err("agent", name, list(self._agents))
        return self._agents[name]

    def get_callable(self, name: str) -> "Callable[..., Any]":
        if name not in self._callables:
            raise _err("callable", name, list(self._callables))
        return self._callables[name]

    def get_unit(self, name: str) -> "CoALAUnit":
        if name not in self._units:
            raise _err("unit", name, list(self._units))
        return self._units[name]

    def get_workflow(self, name: str) -> "Callable[..., Any]":
        if name not in self._workflows:
            raise _err("workflow", name, list(self._workflows))
        return self._workflows[name]

    # ---- factory ---------------------------------------------------------

    @classmethod
    def with_defaults(cls) -> "StepRegistry":
        """Pre-register the four built-in step kinds with this registry."""
        from ballast.patterns.plan_execute._steps import (
            CallableStep, LLMStep, UnitStep, WorkflowStep,
        )
        r = cls()
        r.register_step("llm",      LLMStep(r))
        r.register_step("callable", CallableStep(r))
        r.register_step("unit",     UnitStep(r))
        r.register_step("workflow", WorkflowStep(r))
        return r


__all__ = ["StepRegistry"]
```

NOTE: `with_defaults()` does a lazy import of `_steps.py` to avoid a circular import (the four Step impls reference `StepRegistry`). This is intentional.

- [ ] **Step 4: Stub `_steps.py` so `with_defaults()` import resolves**

Create `src/ballast/patterns/plan_execute/_steps.py` with minimal stubs (will be implemented in Tasks 6-9):

```python
"""Placeholder — full impls landed in Tasks 6-9."""
from __future__ import annotations

from typing import Any

from ballast.patterns.plan_execute._registry import StepRegistry


class LLMStep:
    def __init__(self, registry: StepRegistry): self._registry = registry
    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        raise NotImplementedError("LLMStep — implemented in Task 6")


class CallableStep:
    def __init__(self, registry: StepRegistry): self._registry = registry
    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        raise NotImplementedError("CallableStep — implemented in Task 7")


class UnitStep:
    def __init__(self, registry: StepRegistry): self._registry = registry
    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        raise NotImplementedError("UnitStep — implemented in Task 8")


class WorkflowStep:
    def __init__(self, registry: StepRegistry): self._registry = registry
    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        raise NotImplementedError("WorkflowStep — implemented in Task 9")


__all__ = ["CallableStep", "LLMStep", "UnitStep", "WorkflowStep"]
```

- [ ] **Step 5: Run — confirm pass**

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/patterns/plan_execute/_registry.py src/ballast/patterns/plan_execute/_steps.py tests/patterns/plan_execute/test_registry.py
git commit -m "feat(plan-execute): StepRegistry + step impl stubs"
```

---

## Task 6: `LLMStep` (real impl + prompt template renderer)

**Files:**
- Modify: `src/ballast/patterns/plan_execute/_steps.py` (replace `LLMStep` stub)
- Create: `tests/patterns/plan_execute/test_steps_llm.py`

- [ ] **Step 1: Failing test (`tests/patterns/plan_execute/test_steps_llm.py`)**

```python
"""LLMStep — agent invocation with templated prompt."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._protocols import StepContext
from ballast.patterns.plan_execute._registry import StepRegistry
from ballast.patterns.plan_execute._steps import LLMStep


@dataclass
class _FakeResult:
    output: Any


class _RecordingAgent:
    """Mimics pydantic-ai Agent.run."""
    def __init__(self, output): self.output = output; self.prompts = []
    async def run(self, prompt):
        self.prompts.append(prompt)
        return _FakeResult(self.output)


def _ctx(step: PlannedStep, registry: StepRegistry) -> StepContext:
    return StepContext(
        plan=Plan(steps=[step]), step=step, step_registry=registry,
    )


@pytest.mark.asyncio
async def test_llm_step_invokes_agent_with_rendered_prompt() -> None:
    registry = StepRegistry()
    agent = _RecordingAgent(output="summary")
    registry.register_agent("summarizer", agent)
    step = PlannedStep(
        id="s1", kind="llm",
        params={"agent_name": "summarizer", "prompt_template": "Summarize: {plan_input}"},
    )
    out = await LLMStep(registry).execute(
        plan_input="raw text", dep_outputs={}, ctx=_ctx(step, registry),
    )
    assert out == "summary"
    assert agent.prompts == ["Summarize: raw text"]


@pytest.mark.asyncio
async def test_llm_step_substitutes_plan_input_attr() -> None:
    @dataclass
    class _Input:
        topic: str

    registry = StepRegistry()
    agent = _RecordingAgent(output="ok")
    registry.register_agent("ag", agent)
    step = PlannedStep(
        id="s1", kind="llm",
        params={"agent_name": "ag", "prompt_template": "Topic={plan_input.topic}"},
    )
    await LLMStep(registry).execute(
        plan_input=_Input(topic="X"), dep_outputs={}, ctx=_ctx(step, registry),
    )
    assert agent.prompts == ["Topic=X"]


@pytest.mark.asyncio
async def test_llm_step_substitutes_dep_output_whole_and_field() -> None:
    @dataclass
    class _D:
        title: str

    registry = StepRegistry()
    agent = _RecordingAgent(output="ok")
    registry.register_agent("ag", agent)
    step = PlannedStep(
        id="s1", kind="llm",
        params={
            "agent_name": "ag",
            "prompt_template": "Whole={dep_a} Title={dep_b.title}",
        },
    )
    await LLMStep(registry).execute(
        plan_input=None,
        dep_outputs={"dep_a": "ALPHA", "dep_b": _D(title="BETA")},
        ctx=_ctx(step, registry),
    )
    assert agent.prompts == ["Whole=ALPHA Title=BETA"]


@pytest.mark.asyncio
async def test_llm_step_extracts_output_field_when_specified() -> None:
    @dataclass
    class _Result:
        summary: str
        debug: str

    registry = StepRegistry()
    agent = _RecordingAgent(output=_Result(summary="S", debug="D"))
    registry.register_agent("ag", agent)
    step = PlannedStep(
        id="s1", kind="llm",
        params={
            "agent_name": "ag",
            "prompt_template": "x",
            "output_field": "summary",
        },
    )
    out = await LLMStep(registry).execute(
        plan_input=None, dep_outputs={}, ctx=_ctx(step, registry),
    )
    assert out == "S"


@pytest.mark.asyncio
async def test_llm_step_unknown_agent_raises_keyerror() -> None:
    registry = StepRegistry()
    step = PlannedStep(
        id="s1", kind="llm",
        params={"agent_name": "missing", "prompt_template": "x"},
    )
    with pytest.raises(KeyError, match="missing"):
        await LLMStep(registry).execute(
            plan_input=None, dep_outputs={}, ctx=_ctx(step, registry),
        )
```

- [ ] **Step 2: Run — confirm fail**

Expected: `NotImplementedError` (stub from Task 5).

- [ ] **Step 3: Replace stub in `src/ballast/patterns/plan_execute/_steps.py` with real `LLMStep`**

Read the file, replace ONLY the `LLMStep` class:

```python
"""Built-in ``Step`` implementations: ``LLMStep``, ``CallableStep``,
``UnitStep``, ``WorkflowStep``.

Each dispatches via ``StepRegistry`` to the actual agent / function /
unit / workflow the app registered under a name. Planner emits
``PlannedStep(kind=..., params={...})``; framework wires it together.
"""
from __future__ import annotations

import re
from typing import Any

from ballast.patterns.plan_execute._registry import StepRegistry


_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(\.[a-zA-Z_][a-zA-Z0-9_]*)?\}")


def _render_prompt(
    template: str, plan_input: Any, dep_outputs: dict[str, Any],
) -> str:
    """f-string-like substitution. Supports:
      {plan_input}            — whole plan_input stringified
      {plan_input.field}      — attribute or dict-key access on plan_input
      {dep_id}                — whole dep output stringified
      {dep_id.field}          — attribute or dict-key access on dep output
    """
    def _resolve(name: str, attr: str | None) -> str:
        if name == "plan_input":
            value = plan_input
        elif name in dep_outputs:
            value = dep_outputs[name]
        else:
            return f"{{{name}{attr or ''}}}"  # leave unresolved literal
        if attr is None:
            return str(value)
        field = attr[1:]  # drop leading '.'
        if hasattr(value, field):
            return str(getattr(value, field))
        if isinstance(value, dict):
            return str(value.get(field, f"{{{name}{attr}}}"))
        return f"{{{name}{attr}}}"

    return _PLACEHOLDER.sub(
        lambda m: _resolve(m.group(1), m.group(2)),
        template,
    )


class LLMStep:
    """Run a registered pydantic-ai Agent with a templated prompt.

    Planner emits:
        PlannedStep(kind="llm", params={
            "agent_name": "<name>",
            "prompt_template": "<text with {plan_input.x} / {dep_id.field}>",
            "output_field": "<optional field name>",
        })
    """

    def __init__(self, registry: StepRegistry) -> None:
        self._registry = registry

    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        params = ctx.step.params
        agent = self._registry.get_agent(params["agent_name"])
        prompt = _render_prompt(
            params["prompt_template"], plan_input, dep_outputs,
        )
        result = await agent.run(prompt)
        output = result.output
        if "output_field" in params:
            field = params["output_field"]
            if hasattr(output, field):
                output = getattr(output, field)
            elif isinstance(output, dict) and field in output:
                output = output[field]
        return output


class CallableStep:
    def __init__(self, registry: StepRegistry): self._registry = registry
    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        raise NotImplementedError("CallableStep — implemented in Task 7")


class UnitStep:
    def __init__(self, registry: StepRegistry): self._registry = registry
    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        raise NotImplementedError("UnitStep — implemented in Task 8")


class WorkflowStep:
    def __init__(self, registry: StepRegistry): self._registry = registry
    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        raise NotImplementedError("WorkflowStep — implemented in Task 9")


__all__ = ["CallableStep", "LLMStep", "UnitStep", "WorkflowStep"]
```

- [ ] **Step 4: Run — confirm pass**

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/patterns/plan_execute/_steps.py tests/patterns/plan_execute/test_steps_llm.py
git commit -m "feat(plan-execute): LLMStep with prompt template renderer"
```

---

## Task 7: `CallableStep`

**Files:**
- Modify: `src/ballast/patterns/plan_execute/_steps.py` (replace `CallableStep` stub)
- Create: `tests/patterns/plan_execute/test_steps_callable.py`

- [ ] **Step 1: Failing test**

```python
"""CallableStep — dispatches to a registered async function."""
from __future__ import annotations

import pytest

from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._protocols import StepContext
from ballast.patterns.plan_execute._registry import StepRegistry
from ballast.patterns.plan_execute._steps import CallableStep


def _ctx(step: PlannedStep, registry: StepRegistry) -> StepContext:
    return StepContext(
        plan=Plan(steps=[step]), step=step, step_registry=registry,
    )


@pytest.mark.asyncio
async def test_callable_step_invokes_function_with_plan_input_and_dep_outputs() -> None:
    captured = {}

    async def my_fn(*, plan_input, dep_outputs, extra=None):
        captured["plan_input"] = plan_input
        captured["dep_outputs"] = dep_outputs
        captured["extra"] = extra
        return "result"

    registry = StepRegistry()
    registry.register_callable("my_fn", my_fn)
    step = PlannedStep(
        id="s1", kind="callable",
        params={"fn_name": "my_fn", "args": {"extra": "hello"}},
    )
    out = await CallableStep(registry).execute(
        plan_input={"x": 1}, dep_outputs={"a": "out_a"},
        ctx=_ctx(step, registry),
    )
    assert out == "result"
    assert captured["plan_input"] == {"x": 1}
    assert captured["dep_outputs"] == {"a": "out_a"}
    assert captured["extra"] == "hello"


@pytest.mark.asyncio
async def test_callable_step_args_optional() -> None:
    called = []

    async def my_fn(*, plan_input, dep_outputs):
        called.append((plan_input, dep_outputs))
        return None

    registry = StepRegistry()
    registry.register_callable("my_fn", my_fn)
    step = PlannedStep(id="s1", kind="callable", params={"fn_name": "my_fn"})
    await CallableStep(registry).execute(
        plan_input=42, dep_outputs={}, ctx=_ctx(step, registry),
    )
    assert called == [(42, {})]


@pytest.mark.asyncio
async def test_callable_step_unknown_function_raises_keyerror() -> None:
    registry = StepRegistry()
    step = PlannedStep(
        id="s1", kind="callable", params={"fn_name": "missing"},
    )
    with pytest.raises(KeyError, match="missing"):
        await CallableStep(registry).execute(
            plan_input=None, dep_outputs={}, ctx=_ctx(step, registry),
        )
```

- [ ] **Step 2: Run — confirm fail**

Expected: `NotImplementedError`.

- [ ] **Step 3: Replace `CallableStep` stub in `_steps.py`**

```python
class CallableStep:
    """Run a registered async function.

    Planner emits:
        PlannedStep(kind="callable", params={
            "fn_name": "<name>",
            "args": {"k": v, ...},  # optional extra kwargs
        })

    The function is invoked as ``fn(plan_input=..., dep_outputs=..., **args)``.
    """

    def __init__(self, registry: StepRegistry) -> None:
        self._registry = registry

    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        params = ctx.step.params
        fn = self._registry.get_callable(params["fn_name"])
        extra = params.get("args", {})
        return await fn(
            plan_input=plan_input, dep_outputs=dep_outputs, **extra,
        )
```

- [ ] **Step 4: Run — confirm pass**

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/patterns/plan_execute/_steps.py tests/patterns/plan_execute/test_steps_callable.py
git commit -m "feat(plan-execute): CallableStep"
```

---

## Task 8: `UnitStep`

**Files:**
- Modify: `src/ballast/patterns/plan_execute/_steps.py` (replace `UnitStep` stub)
- Create: `tests/patterns/plan_execute/test_steps_unit.py`

- [ ] **Step 1: Failing test**

```python
"""UnitStep — dispatches to a registered CoALAUnit via its 4-phase lifecycle."""
from __future__ import annotations

import pytest

from ballast.coala import CoALABase
from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._protocols import StepContext
from ballast.patterns.plan_execute._registry import StepRegistry
from ballast.patterns.plan_execute._steps import UnitStep


class _RecordingUnit(CoALABase[str, str, dict, str]):
    """Records each phase invocation order."""
    calls: list[str] = []

    async def observe(self, input):
        self.calls.append(f"observe({input})")
        return input.upper()

    async def retrieve(self, observation):
        self.calls.append(f"retrieve({observation})")
        return {"ctx": observation}

    async def act(self, observation, context):
        self.calls.append(f"act({observation},{context})")
        return f"acted-{observation}"

    async def learn(self, observation, context, output):
        self.calls.append(f"learn({output})")


def _ctx(step: PlannedStep, registry: StepRegistry) -> StepContext:
    return StepContext(
        plan=Plan(steps=[step]), step=step, step_registry=registry,
    )


@pytest.mark.asyncio
async def test_unit_step_runs_four_phases_in_order_with_plan_input() -> None:
    unit = _RecordingUnit()
    unit.calls = []
    registry = StepRegistry()
    registry.register_unit("u", unit)
    step = PlannedStep(id="s1", kind="unit", params={"unit_name": "u"})
    out = await UnitStep(registry).execute(
        plan_input="hello", dep_outputs={}, ctx=_ctx(step, registry),
    )
    assert out == "acted-HELLO"
    assert unit.calls == [
        "observe(hello)",
        "retrieve(HELLO)",
        "act(HELLO,{'ctx': 'HELLO'})",
        "learn(acted-HELLO)",
    ]


@pytest.mark.asyncio
async def test_unit_step_uses_dep_output_when_input_from_set() -> None:
    unit = _RecordingUnit()
    unit.calls = []
    registry = StepRegistry()
    registry.register_unit("u", unit)
    step = PlannedStep(
        id="s1", kind="unit",
        params={"unit_name": "u", "input_from": "dep_a"},
    )
    await UnitStep(registry).execute(
        plan_input="ignored",
        dep_outputs={"dep_a": "from_dep"},
        ctx=_ctx(step, registry),
    )
    assert unit.calls[0] == "observe(from_dep)"


@pytest.mark.asyncio
async def test_unit_step_unknown_unit_raises_keyerror() -> None:
    registry = StepRegistry()
    step = PlannedStep(
        id="s1", kind="unit", params={"unit_name": "missing"},
    )
    with pytest.raises(KeyError, match="missing"):
        await UnitStep(registry).execute(
            plan_input=None, dep_outputs={}, ctx=_ctx(step, registry),
        )
```

- [ ] **Step 2: Run — confirm fail**

Expected: `NotImplementedError`.

- [ ] **Step 3: Replace `UnitStep` stub in `_steps.py`**

```python
class UnitStep:
    """Run a registered ``CoALAUnit`` through its 4-phase lifecycle.

    Planner emits:
        PlannedStep(kind="unit", params={
            "unit_name": "<name>",
            "input_from": "<dep_id>",  # optional — use dep output instead of plan_input
        })
    """

    def __init__(self, registry: StepRegistry) -> None:
        self._registry = registry

    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        params = ctx.step.params
        unit = self._registry.get_unit(params["unit_name"])
        unit_input = (
            dep_outputs[params["input_from"]]
            if "input_from" in params
            else plan_input
        )
        observation = await unit.observe(unit_input)
        retrieved = await unit.retrieve(observation)
        out = await unit.act(observation, retrieved)
        await unit.learn(observation, retrieved, out)
        return out
```

- [ ] **Step 4: Run — confirm pass**

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/patterns/plan_execute/_steps.py tests/patterns/plan_execute/test_steps_unit.py
git commit -m "feat(plan-execute): UnitStep (CoALAUnit 4-phase dispatch)"
```

---

## Task 9: `WorkflowStep`

**Files:**
- Modify: `src/ballast/patterns/plan_execute/_steps.py` (replace `WorkflowStep` stub)
- Create: `tests/patterns/plan_execute/test_steps_workflow.py`

- [ ] **Step 1: Failing test**

```python
"""WorkflowStep — dispatches to a registered async workflow callable."""
from __future__ import annotations

import pytest

from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._protocols import StepContext
from ballast.patterns.plan_execute._registry import StepRegistry
from ballast.patterns.plan_execute._steps import WorkflowStep


def _ctx(step: PlannedStep, registry: StepRegistry) -> StepContext:
    return StepContext(
        plan=Plan(steps=[step]), step=step, step_registry=registry,
    )


@pytest.mark.asyncio
async def test_workflow_step_invokes_callable_with_plan_input() -> None:
    captured = []

    async def my_wf(input):
        captured.append(input)
        return f"done-{input}"

    registry = StepRegistry()
    registry.register_workflow("my_wf", my_wf)
    step = PlannedStep(
        id="s1", kind="workflow", params={"workflow_name": "my_wf"},
    )
    out = await WorkflowStep(registry).execute(
        plan_input="payload", dep_outputs={}, ctx=_ctx(step, registry),
    )
    assert out == "done-payload"
    assert captured == ["payload"]


@pytest.mark.asyncio
async def test_workflow_step_uses_dep_output_when_input_from_set() -> None:
    captured = []

    async def my_wf(input):
        captured.append(input)
        return None

    registry = StepRegistry()
    registry.register_workflow("my_wf", my_wf)
    step = PlannedStep(
        id="s1", kind="workflow",
        params={"workflow_name": "my_wf", "input_from": "dep_a"},
    )
    await WorkflowStep(registry).execute(
        plan_input="ignored",
        dep_outputs={"dep_a": {"k": 1}},
        ctx=_ctx(step, registry),
    )
    assert captured == [{"k": 1}]


@pytest.mark.asyncio
async def test_workflow_step_unknown_workflow_raises_keyerror() -> None:
    registry = StepRegistry()
    step = PlannedStep(
        id="s1", kind="workflow", params={"workflow_name": "missing"},
    )
    with pytest.raises(KeyError, match="missing"):
        await WorkflowStep(registry).execute(
            plan_input=None, dep_outputs={}, ctx=_ctx(step, registry),
        )
```

- [ ] **Step 2: Run — confirm fail**

Expected: `NotImplementedError`.

- [ ] **Step 3: Replace `WorkflowStep` stub in `_steps.py`**

```python
class WorkflowStep:
    """Run a registered async workflow callable as a sub-step.

    The workflow callable is invoked with a single positional argument:
    either the original ``plan_input`` or the named dep output if
    ``input_from`` is set.

    Planner emits:
        PlannedStep(kind="workflow", params={
            "workflow_name": "<name>",
            "input_from": "<dep_id>",  # optional
        })
    """

    def __init__(self, registry: StepRegistry) -> None:
        self._registry = registry

    async def execute(self, plan_input, dep_outputs, ctx) -> Any:
        params = ctx.step.params
        workflow = self._registry.get_workflow(params["workflow_name"])
        wf_input = (
            dep_outputs[params["input_from"]]
            if "input_from" in params
            else plan_input
        )
        return await workflow(wf_input)
```

- [ ] **Step 4: Run — confirm pass**

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/patterns/plan_execute/_steps.py tests/patterns/plan_execute/test_steps_workflow.py
git commit -m "feat(plan-execute): WorkflowStep"
```

---

## Task 10: `PlanAndExecute` pattern (the entry point)

**Files:**
- Create: `src/ballast/patterns/plan_execute/_pattern.py`
- Create: `tests/patterns/plan_execute/conftest.py` — DBOS fixture
- Create: `tests/patterns/plan_execute/test_pattern.py`

- [ ] **Step 1: DBOS fixture (`tests/patterns/plan_execute/conftest.py`)**

Same pattern as `tests/coala/conftest.py`:

```python
"""DBOS bootstrap for plan-execute pattern tests."""
from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import pytest_asyncio
from dbos import DBOS, DBOSConfig


@pytest.fixture(scope="module")
def dbos_runtime() -> Iterator[type[DBOS]]:
    tmp = tempfile.mkdtemp(prefix="dbos-plan-execute-")
    DBOS(config=DBOSConfig(
        name="plan-execute-test",
        system_database_url=f"sqlite:///{Path(tmp)/'dbos.sqlite'}",
    ))
    DBOS.launch()
    try:
        yield DBOS
    finally:
        DBOS.destroy(destroy_registry=False)


@pytest_asyncio.fixture
async def fresh_dbos_executor(dbos_runtime: type[DBOS]) -> AsyncIterator[None]:
    from dbos._dbos import _get_dbos_instance
    _get_dbos_instance()._executor_field = ThreadPoolExecutor(
        max_workers=8, thread_name_prefix="dbos-test-",
    )
    yield
```

- [ ] **Step 2: Failing tests (`tests/patterns/plan_execute/test_pattern.py`)**

```python
"""PlanAndExecute.run end-to-end."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ballast.patterns.plan_execute._errors import PlanExecutionError
from ballast.patterns.plan_execute._pattern import PlanAndExecute
from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._policies import FailLoud
from ballast.patterns.plan_execute._registry import StepRegistry


# ---- Fakes -----------------------------------------------------------------

@dataclass
class _FakeRes:
    output: Any


class _FakePlanner:
    """Mimics pydantic-ai Agent[None, Plan].run."""
    def __init__(self, plan: Plan): self.plan = plan
    async def run(self, input: Any):
        return _FakeRes(self.plan)


# ---- Tests -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_linear_plan_returns_dict_of_step_outputs(fresh_dbos_executor) -> None:
    plan = Plan(steps=[
        PlannedStep(id="a", kind="callable", params={"fn_name": "fn_a"}),
        PlannedStep(id="b", kind="callable", params={"fn_name": "fn_b"}, depends_on=["a"]),
    ])
    registry = StepRegistry.with_defaults()

    async def fn_a(*, plan_input, dep_outputs):
        return f"A({plan_input})"

    async def fn_b(*, plan_input, dep_outputs):
        return f"B({dep_outputs['a']})"

    registry.register_callable("fn_a", fn_a)
    registry.register_callable("fn_b", fn_b)

    pattern = PlanAndExecute(planner=_FakePlanner(plan), registry=registry)
    outputs = await pattern.run("INPUT")

    assert outputs == {"a": "A(INPUT)", "b": "B(A(INPUT))"}


@pytest.mark.asyncio
async def test_run_diamond_plan_executes_parallel_branches(fresh_dbos_executor) -> None:
    import asyncio

    plan = Plan(steps=[
        PlannedStep(id="root", kind="callable", params={"fn_name": "fn_root"}),
        PlannedStep(id="left", kind="callable", params={"fn_name": "fn_branch"}, depends_on=["root"]),
        PlannedStep(id="right", kind="callable", params={"fn_name": "fn_branch"}, depends_on=["root"]),
        PlannedStep(id="join", kind="callable", params={"fn_name": "fn_join"}, depends_on=["left", "right"]),
    ])
    registry = StepRegistry.with_defaults()

    async def fn_root(*, plan_input, dep_outputs): return "R"
    async def fn_branch(*, plan_input, dep_outputs):
        await asyncio.sleep(0.01)
        return f"B({dep_outputs['root']})"
    async def fn_join(*, plan_input, dep_outputs):
        return f"J({dep_outputs['left']}+{dep_outputs['right']})"

    registry.register_callable("fn_root",   fn_root)
    registry.register_callable("fn_branch", fn_branch)
    registry.register_callable("fn_join",   fn_join)

    pattern = PlanAndExecute(planner=_FakePlanner(plan), registry=registry)
    outputs = await pattern.run(None)

    assert outputs["root"] == "R"
    assert outputs["left"] == "B(R)"
    assert outputs["right"] == "B(R)"
    assert outputs["join"] == "J(B(R)+B(R))"


@pytest.mark.asyncio
async def test_run_empty_plan_returns_empty_dict(fresh_dbos_executor) -> None:
    pattern = PlanAndExecute(
        planner=_FakePlanner(Plan(steps=[])),
        registry=StepRegistry.with_defaults(),
    )
    out = await pattern.run("x")
    assert out == {}


@pytest.mark.asyncio
async def test_fail_loud_raises_plan_execution_error(fresh_dbos_executor) -> None:
    plan = Plan(steps=[
        PlannedStep(id="bad", kind="callable", params={"fn_name": "boom"}),
    ])
    registry = StepRegistry.with_defaults()

    async def boom(*, plan_input, dep_outputs):
        raise RuntimeError("kaboom")

    registry.register_callable("boom", boom)

    pattern = PlanAndExecute(
        planner=_FakePlanner(plan), registry=registry,
        replan_policy=FailLoud(),
    )
    with pytest.raises(PlanExecutionError) as exc:
        await pattern.run(None)

    assert exc.value.failed_step.id == "bad"
    assert exc.value.partial_outputs == {}
    assert "kaboom" in str(exc.value.__cause__)


@pytest.mark.asyncio
async def test_custom_replan_policy_continues_after_failure(fresh_dbos_executor) -> None:
    plan_v1 = Plan(steps=[
        PlannedStep(id="a", kind="callable", params={"fn_name": "ok"}),
        PlannedStep(id="b", kind="callable", params={"fn_name": "boom"}, depends_on=["a"]),
    ])
    plan_v2 = Plan(steps=[
        PlannedStep(id="a", kind="callable", params={"fn_name": "ok"}),
        PlannedStep(id="b_recovery", kind="callable", params={"fn_name": "ok"}, depends_on=["a"]),
    ])
    registry = StepRegistry.with_defaults()

    async def ok(*, plan_input, dep_outputs):
        return "OK"

    async def boom(*, plan_input, dep_outputs):
        raise RuntimeError("fail")

    registry.register_callable("ok", ok)
    registry.register_callable("boom", boom)

    class _SwapPlan:
        def __init__(self): self.calls = 0
        async def on_step_failure(self, plan, failed_step, error, partial_outputs):
            self.calls += 1
            return plan_v2 if self.calls == 1 else None

    pattern = PlanAndExecute(
        planner=_FakePlanner(plan_v1), registry=registry,
        replan_policy=_SwapPlan(),
    )
    outputs = await pattern.run(None)
    assert outputs == {"a": "OK", "b_recovery": "OK"}


@pytest.mark.asyncio
async def test_max_parallel_caps_concurrency(fresh_dbos_executor) -> None:
    import asyncio

    plan = Plan(steps=[
        PlannedStep(id=f"s{i}", kind="callable", params={"fn_name": "slow"})
        for i in range(5)
    ])
    registry = StepRegistry.with_defaults()

    in_flight = {"n": 0, "max": 0}

    async def slow(*, plan_input, dep_outputs):
        in_flight["n"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["n"])
        await asyncio.sleep(0.01)
        in_flight["n"] -= 1
        return None

    registry.register_callable("slow", slow)

    pattern = PlanAndExecute(
        planner=_FakePlanner(plan), registry=registry, max_parallel=2,
    )
    await pattern.run(None)
    assert in_flight["max"] <= 2
```

- [ ] **Step 3: Run — confirm fail**

Expected: ImportError for `PlanAndExecute`.

- [ ] **Step 4: Implement `src/ballast/patterns/plan_execute/_pattern.py`**

```python
"""``PlanAndExecute`` — pattern entry point.

Same ``DBOSConfiguredInstance`` shape as ``MapReduce``. The unit stored
on ``self`` (planner + registry + replan policy) is never pickled per
step call because we live on a configured instance with a unique
``config_name``.
"""
from __future__ import annotations

import asyncio
import itertools
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from dbos import DBOSConfiguredInstance

from ballast.durable import Durable
from ballast.patterns.plan_execute._errors import PlanExecutionError
from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._policies import FailLoud
from ballast.patterns.plan_execute._protocols import (
    RePlanPolicy, StepContext,
)
from ballast.patterns.plan_execute._registry import StepRegistry

if TYPE_CHECKING:
    from pydantic_ai import Agent


InT = TypeVar("InT")
OutT = TypeVar("OutT")

_instance_counter = itertools.count()


@Durable.dbos_class()
class PlanAndExecute(DBOSConfiguredInstance, Generic[InT, OutT]):
    """Plan-and-Execute pattern: planner emits DAG, framework executes nodes.

    Two-phase durable workflow:
      1. ``_plan_step`` — call planner.run(input) → ``Plan``.
      2. ``_execute_dag`` (orchestrator, not a step itself) — wave-by-wave
         traversal with ``asyncio.gather`` + semaphore; each step dispatch
         goes through ``_execute_step`` which IS a ``@Durable.step``.

    On replay, DBOS memoises completed steps; only the unfinished tail
    re-runs.
    """

    def __init__(
        self, *,
        planner: "Agent[None, Plan]",
        registry: StepRegistry,
        replan_policy: RePlanPolicy | None = None,
        max_parallel: int = 8,
    ) -> None:
        super().__init__(
            config_name=f"{type(self).__qualname__}-{next(_instance_counter)}",
        )
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        self._planner = planner
        self._registry = registry
        self._replan_policy: RePlanPolicy = replan_policy or FailLoud()
        self._max_parallel = max_parallel

    @Durable.workflow()
    async def run(self, input: InT) -> dict[str, Any]:
        """Returns ``{step_id: output}`` for every completed step."""
        plan = await self._plan_step(input)
        outputs = await self._execute_dag(input, plan)
        return outputs

    @Durable.step()
    async def _plan_step(self, input: InT) -> Plan:
        result = await self._planner.run(_serialize_for_planner(input))
        return result.output

    async def _execute_dag(self, plan_input: InT, plan: Plan) -> dict[str, Any]:
        """Wave-by-wave DAG traversal. Not a @Durable.step itself."""
        outputs: dict[str, Any] = {}
        pending: dict[str, PlannedStep] = {s.id: s for s in plan.steps}
        sem = asyncio.Semaphore(self._max_parallel)

        while pending:
            ready: list[PlannedStep] = [
                s for s in pending.values()
                if all(dep in outputs for dep in s.depends_on)
            ]
            if not ready:
                raise RuntimeError(
                    f"plan deadlock: {len(pending)} steps remain, none ready. "
                    f"Pending: {sorted(pending)}"
                )

            async def _run_one(step: PlannedStep) -> tuple[str, Any]:
                async with sem:
                    return step.id, await self._execute_step(
                        plan, step, plan_input, outputs,
                    )

            batch_results = await asyncio.gather(
                *(_run_one(s) for s in ready),
                return_exceptions=True,
            )

            # Process results; on first exception, hand off to replan policy.
            new_plan: Plan | None = None
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    failed_step = ready[i]
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
                    break  # rebuild pending from new_plan
                step_id, output = result  # type: ignore[misc]
                outputs[step_id] = output
                del pending[step_id]

            if new_plan is not None:
                plan = new_plan
                pending = {
                    s.id: s for s in plan.steps if s.id not in outputs
                }

        return outputs

    @Durable.step()
    async def _execute_step(
        self,
        plan: Plan,
        planned: PlannedStep,
        plan_input: InT,
        outputs: dict[str, Any],
    ) -> Any:
        """Execute one planned step — memoised per ``(config_name, args)``."""
        step_impl = self._registry.get_step(planned.kind)
        dep_outputs = {
            dep_id: outputs[dep_id] for dep_id in planned.depends_on
        }
        ctx = StepContext(
            plan=plan,
            step=planned,
            step_registry=self._registry,
            workflow_id=_current_workflow_id(),
        )
        return await step_impl.execute(plan_input, dep_outputs, ctx)


def _serialize_for_planner(input: Any) -> str:
    """Render the user-supplied plan input as a prompt string for the planner."""
    if isinstance(input, str):
        return input
    return repr(input)


def _current_workflow_id() -> str | None:
    """Best-effort fetch of current DBOS workflow id; None if unavailable."""
    try:
        return Durable.current_workflow_id()
    except Exception:  # noqa: BLE001
        return None


__all__ = ["PlanAndExecute"]
```

- [ ] **Step 5: Run — confirm pass**

Run: `uv run pytest tests/patterns/plan_execute/test_pattern.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/patterns/plan_execute/_pattern.py tests/patterns/plan_execute/conftest.py tests/patterns/plan_execute/test_pattern.py
git commit -m "feat(plan-execute): PlanAndExecute pattern (DBOSConfiguredInstance + per-step memoisation)"
```

---

## Task 11: Public API re-exports

**Files:**
- Modify: `src/ballast/patterns/plan_execute/__init__.py`
- Modify: `src/ballast/patterns/__init__.py` (re-export `PlanAndExecute`)
- Modify: `src/ballast/__init__.py` (top-level `PlanAndExecute`)

- [ ] **Step 1: Edit `src/ballast/patterns/plan_execute/__init__.py`**

```python
"""Plan-and-Execute pattern — planner-driven DAG with framework dispatcher."""
from ballast.patterns.plan_execute._errors import PlanExecutionError
from ballast.patterns.plan_execute._pattern import PlanAndExecute
from ballast.patterns.plan_execute._plan import Plan, PlannedStep
from ballast.patterns.plan_execute._policies import FailLoud
from ballast.patterns.plan_execute._protocols import (
    RePlanPolicy, Step, StepContext,
)
from ballast.patterns.plan_execute._registry import StepRegistry
from ballast.patterns.plan_execute._steps import (
    CallableStep, LLMStep, UnitStep, WorkflowStep,
)

__all__ = [
    "CallableStep",
    "FailLoud",
    "LLMStep",
    "Plan",
    "PlanAndExecute",
    "PlanExecutionError",
    "PlannedStep",
    "RePlanPolicy",
    "Step",
    "StepContext",
    "StepRegistry",
    "UnitStep",
    "WorkflowStep",
]
```

- [ ] **Step 2: Edit `src/ballast/patterns/__init__.py`**

Find the existing pattern re-exports (e.g., `MapReduce`, `Reflection`). Add:
```python
from ballast.patterns.plan_execute import PlanAndExecute
```

Add `"PlanAndExecute"` to `__all__` (alphabetical).

- [ ] **Step 3: Edit `src/ballast/__init__.py`**

Find where `MapReduce` is re-exported from `ballast.patterns`. Add `PlanAndExecute` import next to it:
```python
from ballast.patterns import (
    ...,
    PlanAndExecute,
    ...,
)
```

Add `"PlanAndExecute"` to `__all__` (alphabetical).

- [ ] **Step 4: Smoke import**

Run:
```
uv run python -c "from ballast import PlanAndExecute; print('ok')"
```
Expected: `ok`.

Then verify the full subpackage:
```
uv run python -c "
from ballast.patterns.plan_execute import (
    PlanAndExecute, Plan, PlannedStep,
    Step, RePlanPolicy, StepContext, StepRegistry,
    LLMStep, CallableStep, UnitStep, WorkflowStep,
    FailLoud, PlanExecutionError,
)
print('plan_execute subpackage ok')
"
```
Expected: `plan_execute subpackage ok`.

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/__init__.py src/ballast/patterns/__init__.py src/ballast/patterns/plan_execute/__init__.py
git commit -m "feat(ballast): re-export PlanAndExecute at top level + patterns subpackage"
```

---

## Task 12: Final smoke

- [ ] **Step 1: Run framework suite**

Run: `uv run pytest tests/ --tb=short -q`
Expected: green. All new tests (~35-40 new) plus existing 538+ still passing.

- [ ] **Step 2: Run plan-execute suite specifically**

Run: `uv run pytest tests/patterns/plan_execute/ -v`
Expected: all green.

- [ ] **Step 3: Smoke import the whole framework + new pattern**

Run:
```
uv run python -c "
from ballast import (
    Ballast, BallastSettings,
    PlanAndExecute,
    CoALABase, CoALAUnit, as_workflow, as_tool, as_capability,
    GoalDriftDetector, with_drift_monitor,
)
from ballast.patterns.plan_execute import (
    Plan, PlannedStep, StepRegistry, FailLoud, PlanExecutionError,
    LLMStep, CallableStep, UnitStep, WorkflowStep,
)
print('all imports ok')
"
```
Expected: `all imports ok`.

- [ ] **Step 4: Commit any cleanup**

```bash
git status
git add -u && git commit -m "chore(plan-execute): final smoke cleanup" || echo "nothing to commit"
```

---

## Self-Review (against the spec)

**Spec coverage:**

| Spec section | Tasks |
|---|---|
| File structure | Tasks 1-11 (each task creates one file) |
| Public API | Tasks 11 |
| `Plan` + `PlannedStep` + DAG validator | Task 1 |
| `Step`, `RePlanPolicy` Protocols + `StepContext` | Task 2 |
| `PlanExecutionError` | Task 3 |
| `FailLoud` policy | Task 4 |
| `StepRegistry` + `with_defaults()` | Task 5 |
| `LLMStep` (+ prompt renderer) | Task 6 |
| `CallableStep` | Task 7 |
| `UnitStep` | Task 8 |
| `WorkflowStep` | Task 9 |
| `PlanAndExecute` pattern + DAG executor | Task 10 |
| Top-level re-exports | Task 11 |
| Final smoke | Task 12 |

**Placeholder scan:** No TBDs/TODOs/vague-step-without-code. Each step has complete code or exact command + expected output. `_steps.py` shipped in stub form in Task 5 and replaced piece-by-piece in Tasks 6-9 (intentional incremental approach — keeps each task small enough to TDD cleanly).

**Type consistency:**
- `Plan.steps` / `PlannedStep.id` / `PlannedStep.kind` / `PlannedStep.params` / `PlannedStep.depends_on` consistent across all tasks.
- `Step.execute(plan_input, dep_outputs, ctx)` signature consistent in Task 2 (Protocol), Tasks 6-9 (impls), Task 10 (caller).
- `StepContext(plan, step, step_registry, workflow_id)` consistent in Tasks 2, 6-9, 10.
- `RePlanPolicy.on_step_failure(plan, failed_step, error, partial_outputs) -> Plan | None` consistent in Tasks 2, 4, 10.
- `StepRegistry.get_step(kind)` / `get_agent(name)` etc. consistent in Tasks 5, 6-9.
- `PlanAndExecute` constructor `(planner, registry, replan_policy=None, max_parallel=8)` consistent in Task 10, smoke (Task 12).
- `_execute_step` args (`plan, planned, plan_input, outputs`) consistent inside Task 10's `_execute_dag` vs `_execute_step` definitions.

**Known plan-vs-spec gap:** None. Spec's "Open questions" section was empty (all decisions resolved during brainstorm). Plan implements every spec section.

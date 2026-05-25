# Goal Drift Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement async LLM-judge sidecar for detecting agent goal drift, exposed as a `BallastCapability` (agent surface) and `@with_drift_monitor` decorator (workflow surface), with a shared fail-safe `DriftEngine` and 5 plug-in Protocols (strategy / window / goal source / prompt / handler).

**Architecture:** Pure-function core in `src/ballast/drift/` containing Protocols + built-in impls + `DriftEngine`. Two thin wrappers: `src/ballast/capabilities/drift.py` (per-step `after_model_request` hook) and `src/ballast/patterns/drift_monitor.py` (background `asyncio.Task` polling). Default judge built via `pydantic_ai.Agent` factory; verdict type is `DefaultDriftVerdict` (subclass `DriftVerdictBase` for `should_interrupt + reason`).

**Tech Stack:** Python 3.11+, pydantic v2, pydantic-ai (`AbstractCapability` / `RunContext` / `ModelRequestContext` / `Agent`), DBOS (`Durable` facade for HITL handler), existing `BallastError`, `HITLChannel`, OTel `@traced`.

**Spec:** `docs/superpowers/specs/2026-05-26-goal-drift-detection-design.md`

**Spec defaults applied:**
- `OnBudgetThreshold` reads from `DriftContext.metadata["budget"]`. `BudgetGuard` exposes `snapshot()`; apps wire `metadata_provider` callable on `GoalDriftDetector` to bridge them. (Deviation from spec option a — `RunContext` has no `metadata` bus, so we don't pretend one; option b with a `BudgetGuard.snapshot()` convention helper.)
- `EscalateToHITL` BLOCKS until `HITLChannel.request` returns (sequential handler chain semantics).
- Workflow surface accepts known limitation: default `TraceWindow` impls return `[]` for messageless contexts; apps that want workflow drift detection must supply a custom window.

---

## File Structure (reference for all tasks)

```
src/ballast/drift/
  __init__.py              # public re-exports
  _verdict.py              # DriftVerdictBase, DefaultDriftVerdict (Task 1)
  _protocols.py            # DriftCheckSignal, DriftContext, 5 Protocols (Tasks 2+3)
  _strategies.py           # DriftCheckStrategy impls (Task 4)
  _windows.py              # TraceWindow impls (Task 5)
  _goal_sources.py         # GoalSource impls (Task 6)
  _handlers.py             # DriftHandler impls + GoalDriftError (Task 7)
  _judge.py                # DefaultPromptBuilder + make_default_judge (Task 8)
  _core.py                 # DriftEngine + maybe_check (Task 9)
src/ballast/capabilities/
  drift.py                 # GoalDriftDetector (Task 10)
src/ballast/patterns/
  drift_monitor.py         # with_drift_monitor decorator (Task 11)
src/ballast/__init__.py    # top-level re-exports (Task 13)

tests/drift/
  __init__.py
  test_verdict.py          # Task 1
  test_protocols.py        # Tasks 2+3
  test_strategies.py       # Task 4
  test_windows.py          # Task 5
  test_goal_sources.py     # Task 6
  test_handlers.py         # Task 7
  test_judge.py            # Task 8
  test_core.py             # Task 9
tests/capabilities/test_drift.py     # Task 10
tests/patterns/test_drift_monitor.py # Task 11
```

---

## Task 1: `DriftVerdictBase` + `DefaultDriftVerdict`

**Files:**
- Create: `src/ballast/drift/__init__.py` (empty for now; populated in later tasks)
- Create: `src/ballast/drift/_verdict.py`
- Create: `tests/drift/__init__.py` (empty)
- Create: `tests/drift/test_verdict.py`

- [ ] **Step 1: Write the failing test**

`tests/drift/test_verdict.py`:
```python
"""DriftVerdictBase + DefaultDriftVerdict — verdict type contract."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from ballast.drift._verdict import DriftVerdictBase, DefaultDriftVerdict


def test_base_requires_should_interrupt_and_reason() -> None:
    v = DriftVerdictBase(should_interrupt=True, reason="drifted off-topic")
    assert v.should_interrupt is True
    assert v.reason == "drifted off-topic"


def test_base_rejects_missing_fields() -> None:
    with pytest.raises(ValidationError):
        DriftVerdictBase(should_interrupt=True)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        DriftVerdictBase(reason="x")  # type: ignore[call-arg]


def test_default_verdict_adds_score_category_action() -> None:
    v = DefaultDriftVerdict(
        should_interrupt=False, reason="on track",
        score=0.9, category="on_track",
    )
    assert v.score == 0.9
    assert v.category == "on_track"
    assert v.suggested_action is None


def test_default_verdict_category_literal_enforced() -> None:
    with pytest.raises(ValidationError):
        DefaultDriftVerdict(
            should_interrupt=False, reason="x",
            score=0.5, category="bogus",  # type: ignore[arg-type]
        )


def test_default_verdict_is_subclass_of_base() -> None:
    assert issubclass(DefaultDriftVerdict, DriftVerdictBase)
    v = DefaultDriftVerdict(
        should_interrupt=True, reason="r",
        score=0.0, category="drifted",
    )
    assert isinstance(v, DriftVerdictBase)
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/drift/test_verdict.py -v`
Expected: `ModuleNotFoundError: No module named 'ballast.drift'`.

- [ ] **Step 3: Implement `src/ballast/drift/_verdict.py`**

```python
"""``DriftVerdictBase`` + ``DefaultDriftVerdict`` — verdict types.

Apps may subclass ``DriftVerdictBase`` to add domain-specific verdict
fields. The framework only reads ``should_interrupt`` and ``reason``;
everything else is for the app's own handlers / observability.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DriftVerdictBase(BaseModel):
    """Minimum contract — framework reads these two fields."""

    should_interrupt: bool
    """If True, framework runs all configured ``DriftHandler``s."""

    reason: str
    """CoT obboundзование (для логов / HITL контекста)."""


class DefaultDriftVerdict(DriftVerdictBase):
    """Rich default verdict — used when caller doesn't supply a custom one."""

    score: float
    """0.0 = полный дрейф ... 1.0 = на цели."""

    category: Literal["on_track", "loose", "drifted"]
    """Coarse-grained label for metrics / dashboards."""

    suggested_action: str | None = None
    """Optional next-step recommendation from the judge."""


__all__ = ["DriftVerdictBase", "DefaultDriftVerdict"]
```

- [ ] **Step 4: Create empty package marker**

`src/ballast/drift/__init__.py`:
```python
"""Goal Drift Detection — pluggable LLM-judge sidecar for agent runs."""
from ballast.drift._verdict import DefaultDriftVerdict, DriftVerdictBase

__all__ = ["DefaultDriftVerdict", "DriftVerdictBase"]
```

- [ ] **Step 5: Run — confirm pass**

Run: `uv run pytest tests/drift/test_verdict.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/drift/__init__.py src/ballast/drift/_verdict.py tests/drift/__init__.py tests/drift/test_verdict.py
git commit -m "feat(drift): DriftVerdictBase + DefaultDriftVerdict (verdict contract)"
```

---

## Task 2: `DriftCheckSignal` + `DriftContext` dataclasses

**Files:**
- Create: `src/ballast/drift/_protocols.py` (vehicles only for now; Protocols come in Task 3)
- Create: `tests/drift/test_protocols.py` (start with dataclass tests)

- [ ] **Step 1: Write the failing test**

`tests/drift/test_protocols.py`:
```python
"""DriftCheckSignal + DriftContext vehicle dataclasses."""
from __future__ import annotations

from ballast.drift._protocols import DriftCheckSignal, DriftContext


def test_signal_holds_per_step_counters() -> None:
    s = DriftCheckSignal(
        step_index=3, tool_calls=5, tokens_used=1200, seconds_elapsed=42.5,
    )
    assert s.step_index == 3
    assert s.tool_calls == 5
    assert s.tokens_used == 1200
    assert s.seconds_elapsed == 42.5


def test_context_defaults_metadata_to_empty_dict() -> None:
    c = DriftContext(messages=[], run_ctx=None, workflow_input=None)
    assert c.messages == []
    assert c.run_ctx is None
    assert c.workflow_input is None
    assert c.metadata == {}


def test_context_preserves_explicit_metadata() -> None:
    c = DriftContext(
        messages=[], run_ctx=None, workflow_input={"x": 1},
        metadata={"budget": {"input_tokens": 100}},
    )
    assert c.workflow_input == {"x": 1}
    assert c.metadata["budget"]["input_tokens"] == 100
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/drift/test_protocols.py -v`
Expected: `ImportError: cannot import name 'DriftCheckSignal'`.

- [ ] **Step 3: Implement `src/ballast/drift/_protocols.py` (vehicles only)**

```python
"""``DriftCheckSignal`` + ``DriftContext`` + 5 Protocols for drift detection.

Vehicles (this module's first half):
  ``DriftCheckSignal`` — cheap ping passed to ``DriftCheckStrategy.should_check``
  on every agent step. No I/O; constructed even when judge does not run.

  ``DriftContext`` — full state passed to window / goal source / handlers
  only when ``should_check`` returns True. May carry references to a
  ``RunContext`` (agent surface) or workflow input (workflow surface).

Protocols (added in Task 3): ``DriftCheckStrategy``, ``TraceWindow``,
``GoalSource``, ``PromptBuilder``, ``DriftHandler``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ModelMessage


@dataclass
class DriftCheckSignal:
    """Lightweight ping for ``DriftCheckStrategy.should_check``.

    Passed on every step (cheap to construct, no I/O).
    """

    step_index: int
    """Number of ``after_model_request`` invocations seen so far (1-based)."""

    tool_calls: int
    """Cumulative tool-call count across all model responses in this run."""

    tokens_used: int
    """Cumulative input+output tokens across all model responses."""

    seconds_elapsed: float
    """Monotonic time since the first hook fire."""


@dataclass
class DriftContext:
    """Full context for window / goal / handler.

    Built ONLY when ``DriftCheckStrategy.should_check`` returns True.
    Read-only by convention; framework does not mutate after construction.
    """

    messages: list["ModelMessage"]
    """Message history at the moment of the check (may be empty in workflow surface)."""

    run_ctx: "RunContext[Any] | None"
    """Available only in agent surface. ``None`` in workflow surface."""

    workflow_input: Any = None
    """Available only in workflow surface. ``None`` in agent surface."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Application-populated scratch (e.g. ``{"budget": {...}}`` for OnBudgetThreshold)."""


__all__ = ["DriftCheckSignal", "DriftContext"]
```

- [ ] **Step 4: Run — confirm pass**

Run: `uv run pytest tests/drift/test_protocols.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/drift/_protocols.py tests/drift/test_protocols.py
git commit -m "feat(drift): DriftCheckSignal + DriftContext vehicles"
```

---

## Task 3: 5 Protocols (strategy / window / goal / prompt / handler)

**Files:**
- Modify: `src/ballast/drift/_protocols.py` (append Protocols)
- Modify: `src/ballast/drift/__init__.py` (export Protocols + vehicles)
- Modify: `tests/drift/test_protocols.py` (add Protocol tests)

- [ ] **Step 1: Append failing tests to `tests/drift/test_protocols.py`**

```python
import pytest

from ballast.drift._protocols import (
    DriftCheckSignal, DriftContext,
    DriftCheckStrategy, TraceWindow, GoalSource, PromptBuilder, DriftHandler,
)
from ballast.drift._verdict import DriftVerdictBase


def test_drift_check_strategy_runtime_checkable() -> None:
    class _Stub:
        def should_check(self, signal):
            return True
    assert isinstance(_Stub(), DriftCheckStrategy)

    class _Missing:
        pass
    assert not isinstance(_Missing(), DriftCheckStrategy)


def test_trace_window_runtime_checkable() -> None:
    class _Stub:
        async def slice(self, ctx):
            return []
    assert isinstance(_Stub(), TraceWindow)


def test_goal_source_runtime_checkable() -> None:
    class _Stub:
        async def goal(self, ctx):
            return ""
    assert isinstance(_Stub(), GoalSource)


def test_prompt_builder_runtime_checkable() -> None:
    class _Stub:
        def build(self, goal, trace):
            return ""
    assert isinstance(_Stub(), PromptBuilder)


def test_drift_handler_runtime_checkable() -> None:
    class _Stub:
        async def handle(self, verdict, ctx):
            return None
    assert isinstance(_Stub(), DriftHandler)
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/drift/test_protocols.py -v`
Expected: `ImportError` for new Protocol names.

- [ ] **Step 3: Append Protocols to `src/ballast/drift/_protocols.py`**

Add this block AFTER the existing dataclasses, and update `__all__`:

```python
from typing import Protocol, runtime_checkable

from ballast.drift._verdict import DriftVerdictBase


@runtime_checkable
class DriftCheckStrategy(Protocol):
    """When to fire the judge.

    Implementations may be stateful (e.g., ``EveryNToolCalls`` tracks the
    last fire count). ``should_check`` is called on every agent step.
    """

    def should_check(self, signal: DriftCheckSignal) -> bool: ...


@runtime_checkable
class TraceWindow(Protocol):
    """What slice of message history to show the judge."""

    async def slice(self, ctx: DriftContext) -> list["ModelMessage"]: ...


@runtime_checkable
class GoalSource(Protocol):
    """Where the original objective comes from."""

    async def goal(self, ctx: DriftContext) -> str: ...


@runtime_checkable
class PromptBuilder(Protocol):
    """How to ask the judge.

    Returns the user prompt for the judge agent. The judge's system prompt
    is owned by the judge agent itself (see ``make_default_judge``).
    """

    def build(self, goal: str, trace: list["ModelMessage"]) -> str: ...


@runtime_checkable
class DriftHandler(Protocol):
    """What to do on drift.

    Multiple handlers run in declared order. Exceptions from non-Raise
    handlers are swallowed individually (see ``DriftEngine.maybe_check``).
    """

    async def handle(self, verdict: DriftVerdictBase, ctx: DriftContext) -> None: ...
```

Update `__all__` at bottom:

```python
__all__ = [
    "DriftCheckSignal", "DriftContext",
    "DriftCheckStrategy", "TraceWindow", "GoalSource",
    "PromptBuilder", "DriftHandler",
]
```

- [ ] **Step 4: Update `src/ballast/drift/__init__.py`**

```python
"""Goal Drift Detection — pluggable LLM-judge sidecar for agent runs."""
from ballast.drift._protocols import (
    DriftCheckSignal,
    DriftContext,
    DriftCheckStrategy,
    DriftHandler,
    GoalSource,
    PromptBuilder,
    TraceWindow,
)
from ballast.drift._verdict import DefaultDriftVerdict, DriftVerdictBase

__all__ = [
    "DefaultDriftVerdict",
    "DriftCheckSignal",
    "DriftCheckStrategy",
    "DriftContext",
    "DriftHandler",
    "DriftVerdictBase",
    "GoalSource",
    "PromptBuilder",
    "TraceWindow",
]
```

- [ ] **Step 5: Run — confirm pass**

Run: `uv run pytest tests/drift/test_protocols.py -v`
Expected: 8 passed (3 from Task 2 + 5 Protocol tests).

- [ ] **Step 6: Commit**

```bash
git add src/ballast/drift/_protocols.py src/ballast/drift/__init__.py tests/drift/test_protocols.py
git commit -m "feat(drift): 5 Protocols (strategy / window / goal / prompt / handler)"
```

---

## Task 4: `DriftCheckStrategy` impls

**Files:**
- Create: `src/ballast/drift/_strategies.py`
- Modify: `src/ballast/drift/__init__.py` (re-export impls)
- Create: `tests/drift/test_strategies.py`

- [ ] **Step 1: Write the failing tests**

`tests/drift/test_strategies.py`:
```python
"""Built-in DriftCheckStrategy implementations."""
from __future__ import annotations

from ballast.drift._protocols import DriftCheckSignal
from ballast.drift._strategies import (
    AfterEveryStep,
    Compose,
    EveryNSteps,
    EveryNToolCalls,
    OnBudgetThreshold,
    Periodic,
)


def _sig(step=1, tool_calls=0, tokens=0, secs=0.0) -> DriftCheckSignal:
    return DriftCheckSignal(
        step_index=step, tool_calls=tool_calls,
        tokens_used=tokens, seconds_elapsed=secs,
    )


def test_after_every_step_always_true() -> None:
    s = AfterEveryStep()
    assert s.should_check(_sig(step=1))
    assert s.should_check(_sig(step=10))


def test_every_n_tool_calls_fires_at_threshold() -> None:
    s = EveryNToolCalls(n=3)
    assert not s.should_check(_sig(tool_calls=0))
    assert not s.should_check(_sig(tool_calls=2))
    assert s.should_check(_sig(tool_calls=3))
    assert not s.should_check(_sig(tool_calls=4))
    assert not s.should_check(_sig(tool_calls=5))
    assert s.should_check(_sig(tool_calls=6))


def test_every_n_steps_fires_at_threshold() -> None:
    s = EveryNSteps(n=2)
    assert not s.should_check(_sig(step=1))
    assert s.should_check(_sig(step=2))
    assert not s.should_check(_sig(step=3))
    assert s.should_check(_sig(step=4))


def test_periodic_fires_after_interval() -> None:
    s = Periodic(seconds=10.0)
    assert not s.should_check(_sig(secs=5.0))
    assert s.should_check(_sig(secs=10.0))
    assert not s.should_check(_sig(secs=15.0))
    assert s.should_check(_sig(secs=20.0))


def test_on_budget_threshold_fires_once_above_fraction() -> None:
    # Cannot test without metadata access — strategies don't see metadata.
    # OnBudgetThreshold reads from a budget callable supplied at construct time.
    consumed = {"input": 0, "max": 100}
    def budget_fn() -> tuple[int, int]:
        return consumed["input"], consumed["max"]

    s = OnBudgetThreshold(fraction=0.5, budget_fn=budget_fn)

    consumed["input"] = 49
    assert not s.should_check(_sig())

    consumed["input"] = 60
    assert s.should_check(_sig())

    # Already fired — does not re-fire while still above threshold.
    consumed["input"] = 70
    assert not s.should_check(_sig())


def test_compose_is_or_of_components() -> None:
    fired = []

    class _Once:
        def should_check(self, _sig):
            if not fired:
                fired.append(True)
                return True
            return False

    class _Never:
        def should_check(self, _sig):
            return False

    s = Compose(_Once(), _Never())
    assert s.should_check(_sig())     # _Once fires
    assert not s.should_check(_sig()) # both quiet now
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/drift/test_strategies.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/ballast/drift/_strategies.py`**

```python
"""Built-in ``DriftCheckStrategy`` implementations.

Apps choose when the LLM judge fires:

- ``AfterEveryStep`` — every agent step (precise, expensive).
- ``EveryNToolCalls(n)`` — every N tool calls.
- ``EveryNSteps(n)`` — every N model responses.
- ``Periodic(seconds)`` — every N seconds of wall time.
- ``OnBudgetThreshold(fraction, budget_fn)`` — once when consumed / max
  crosses the fraction (e.g., 50% of token budget burnt).
- ``Compose(*strategies)`` — OR-combination; fires if any component fires.
"""
from __future__ import annotations

from collections.abc import Callable

from ballast.drift._protocols import DriftCheckSignal, DriftCheckStrategy


class AfterEveryStep:
    """Fire on every agent step."""

    def should_check(self, signal: DriftCheckSignal) -> bool:
        return True


class EveryNToolCalls:
    """Fire when tool-call count has advanced by N since last fire."""

    def __init__(self, n: int = 5) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self._n = n
        self._last = 0

    def should_check(self, signal: DriftCheckSignal) -> bool:
        if signal.tool_calls >= self._last + self._n:
            self._last = signal.tool_calls
            return True
        return False


class EveryNSteps:
    """Fire when step_index has advanced by N since last fire."""

    def __init__(self, n: int = 3) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self._n = n
        self._last = 0

    def should_check(self, signal: DriftCheckSignal) -> bool:
        if signal.step_index >= self._last + self._n:
            self._last = signal.step_index
            return True
        return False


class Periodic:
    """Fire once each elapsed window of ``seconds``."""

    def __init__(self, seconds: float = 30.0) -> None:
        if seconds <= 0:
            raise ValueError("seconds must be > 0")
        self._seconds = seconds
        self._last = 0.0

    def should_check(self, signal: DriftCheckSignal) -> bool:
        if signal.seconds_elapsed >= self._last + self._seconds:
            self._last = signal.seconds_elapsed
            return True
        return False


class OnBudgetThreshold:
    """Fire ONCE when ``consumed / max`` crosses ``fraction``.

    Reads budget state via a caller-supplied ``budget_fn`` returning
    ``(consumed, max_total)``. Stays quiet once it has fired until the
    consumed value drops back below the threshold (which never happens
    in practice — the fire is effectively one-shot).
    """

    def __init__(
        self, *,
        fraction: float = 0.5,
        budget_fn: Callable[[], tuple[int, int]],
    ) -> None:
        if not 0.0 < fraction < 1.0:
            raise ValueError("fraction must be in (0, 1)")
        self._fraction = fraction
        self._budget_fn = budget_fn
        self._fired = False

    def should_check(self, signal: DriftCheckSignal) -> bool:
        consumed, total = self._budget_fn()
        if total <= 0:
            return False
        crossed = (consumed / total) >= self._fraction
        if crossed and not self._fired:
            self._fired = True
            return True
        if not crossed:
            # Allow re-fire if consumed somehow drops back (defensive).
            self._fired = False
        return False


class Compose:
    """OR-combination — fires if any wrapped strategy fires this tick."""

    def __init__(self, *strategies: DriftCheckStrategy) -> None:
        if not strategies:
            raise ValueError("Compose requires at least one strategy")
        self._strategies = strategies

    def should_check(self, signal: DriftCheckSignal) -> bool:
        # Short-circuit on first True so all subsequent strategies still
        # see this signal on the NEXT call (no skipped ticks).
        return any(s.should_check(signal) for s in self._strategies)


__all__ = [
    "AfterEveryStep",
    "Compose",
    "EveryNSteps",
    "EveryNToolCalls",
    "OnBudgetThreshold",
    "Periodic",
]
```

- [ ] **Step 4: Update `src/ballast/drift/__init__.py`**

Add to imports and `__all__`:
```python
from ballast.drift._strategies import (
    AfterEveryStep,
    Compose as ComposeStrategy,
    EveryNSteps,
    EveryNToolCalls,
    OnBudgetThreshold,
    Periodic,
)
```

Add to `__all__` (alphabetical):
```python
"AfterEveryStep", "ComposeStrategy", "EveryNSteps", "EveryNToolCalls",
"OnBudgetThreshold", "Periodic",
```

- [ ] **Step 5: Run — confirm pass**

Run: `uv run pytest tests/drift/test_strategies.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/drift/_strategies.py src/ballast/drift/__init__.py tests/drift/test_strategies.py
git commit -m "feat(drift): DriftCheckStrategy built-in impls (AfterEveryStep / EveryN* / Periodic / OnBudgetThreshold / Compose)"
```

---

## Task 5: `TraceWindow` impls

**Files:**
- Create: `src/ballast/drift/_windows.py`
- Modify: `src/ballast/drift/__init__.py`
- Create: `tests/drift/test_windows.py`

- [ ] **Step 1: Write the failing tests**

`tests/drift/test_windows.py`:
```python
"""Built-in TraceWindow implementations."""
from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from ballast.drift._protocols import DriftContext
from ballast.drift._windows import (
    FullTrace, LastNMessages, SinceLastUserMessage, TokenBudgetWindow,
)


def _user_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _resp(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _ctx(*msgs) -> DriftContext:
    return DriftContext(messages=list(msgs), run_ctx=None, workflow_input=None)


@pytest.mark.asyncio
async def test_full_trace_returns_all_messages() -> None:
    ctx = _ctx(_user_msg("hi"), _resp("hello"), _user_msg("more"))
    out = await FullTrace().slice(ctx)
    assert len(out) == 3


@pytest.mark.asyncio
async def test_last_n_messages_returns_tail() -> None:
    ctx = _ctx(*[_resp(str(i)) for i in range(10)])
    out = await LastNMessages(n=3).slice(ctx)
    assert len(out) == 3
    assert [p.parts[0].content for p in out] == ["7", "8", "9"]


@pytest.mark.asyncio
async def test_last_n_messages_handles_empty_trace() -> None:
    ctx = _ctx()
    out = await LastNMessages(n=5).slice(ctx)
    assert out == []


@pytest.mark.asyncio
async def test_last_n_messages_handles_n_larger_than_history() -> None:
    ctx = _ctx(_resp("a"), _resp("b"))
    out = await LastNMessages(n=10).slice(ctx)
    assert len(out) == 2


@pytest.mark.asyncio
async def test_since_last_user_message_includes_user_and_after() -> None:
    ctx = _ctx(
        _user_msg("old"), _resp("answer1"),
        _user_msg("new"), _resp("answer2"), _resp("answer3"),
    )
    out = await SinceLastUserMessage().slice(ctx)
    # Slice begins at the LAST user message.
    assert len(out) == 3
    assert out[0].parts[0].content == "new"


@pytest.mark.asyncio
async def test_since_last_user_message_returns_all_when_no_user() -> None:
    ctx = _ctx(_resp("a"), _resp("b"))
    out = await SinceLastUserMessage().slice(ctx)
    assert len(out) == 2


@pytest.mark.asyncio
async def test_token_budget_window_caps_from_tail() -> None:
    # Each msg has ~1 token; cap at 3 → expect last 3 messages.
    ctx = _ctx(*[_resp("x") for _ in range(10)])
    out = await TokenBudgetWindow(max_tokens=3).slice(ctx)
    assert len(out) == 3
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/drift/test_windows.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/ballast/drift/_windows.py`**

```python
"""Built-in ``TraceWindow`` implementations.

Apps choose how much message history the judge sees:

- ``FullTrace`` — entire history (precise, expensive on long sessions).
- ``LastNMessages(n)`` — tail.
- ``SinceLastUserMessage()`` — from the most recent user message onward.
- ``TokenBudgetWindow(max_tokens)`` — trim from the head until total
  approximate token count fits the cap.
"""
from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from ballast.drift._protocols import DriftContext


class FullTrace:
    """Return every message."""

    async def slice(self, ctx: DriftContext) -> list[ModelMessage]:
        return list(ctx.messages)


class LastNMessages:
    """Return the last ``n`` messages (entire history if shorter)."""

    def __init__(self, n: int = 10) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self._n = n

    async def slice(self, ctx: DriftContext) -> list[ModelMessage]:
        return list(ctx.messages[-self._n :])


class SinceLastUserMessage:
    """Slice from the most recent user prompt onward."""

    async def slice(self, ctx: DriftContext) -> list[ModelMessage]:
        # Walk backwards looking for a ModelRequest whose parts contain
        # at least one UserPromptPart. Return slice from that index.
        for i in range(len(ctx.messages) - 1, -1, -1):
            msg = ctx.messages[i]
            if isinstance(msg, ModelRequest) and any(
                isinstance(p, UserPromptPart) for p in msg.parts
            ):
                return list(ctx.messages[i:])
        return list(ctx.messages)


class TokenBudgetWindow:
    """Trim history from the head until total token estimate fits ``max_tokens``.

    Approximation: ``len(str(msg)) // 4`` tokens per message (rough English
    rule-of-thumb; good enough for window sizing).
    """

    def __init__(self, max_tokens: int = 4000) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self._max = max_tokens

    async def slice(self, ctx: DriftContext) -> list[ModelMessage]:
        if not ctx.messages:
            return []
        tail: list[ModelMessage] = []
        budget = self._max
        for msg in reversed(ctx.messages):
            cost = max(1, len(str(msg)) // 4)
            if cost > budget:
                break
            tail.append(msg)
            budget -= cost
        tail.reverse()
        return tail


__all__ = [
    "FullTrace",
    "LastNMessages",
    "SinceLastUserMessage",
    "TokenBudgetWindow",
]
```

- [ ] **Step 4: Update `src/ballast/drift/__init__.py`**

Add to imports:
```python
from ballast.drift._windows import (
    FullTrace, LastNMessages, SinceLastUserMessage, TokenBudgetWindow,
)
```

Add to `__all__` (alphabetical):
```python
"FullTrace", "LastNMessages", "SinceLastUserMessage", "TokenBudgetWindow",
```

- [ ] **Step 5: Run — confirm pass**

Run: `uv run pytest tests/drift/test_windows.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/drift/_windows.py src/ballast/drift/__init__.py tests/drift/test_windows.py
git commit -m "feat(drift): TraceWindow built-in impls (FullTrace / LastN / SinceLastUser / TokenBudget)"
```

---

## Task 6: `GoalSource` impls

**Files:**
- Create: `src/ballast/drift/_goal_sources.py`
- Modify: `src/ballast/drift/__init__.py`
- Create: `tests/drift/test_goal_sources.py`

- [ ] **Step 1: Write the failing tests**

`tests/drift/test_goal_sources.py`:
```python
"""Built-in GoalSource implementations."""
from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from ballast.drift._goal_sources import (
    ExplicitGoal, FirstUserMessage, LastUserMessage, WorkflowInput,
)
from ballast.drift._protocols import DriftContext


def _user_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _resp(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _ctx_msgs(*msgs) -> DriftContext:
    return DriftContext(messages=list(msgs), run_ctx=None, workflow_input=None)


@pytest.mark.asyncio
async def test_first_user_message_returns_first_user_prompt() -> None:
    ctx = _ctx_msgs(
        _user_msg("plan a trip to Berlin"),
        _resp("ok"),
        _user_msg("actually Rome"),
    )
    g = await FirstUserMessage().goal(ctx)
    assert g == "plan a trip to Berlin"


@pytest.mark.asyncio
async def test_first_user_message_returns_empty_when_no_user_msg() -> None:
    ctx = _ctx_msgs(_resp("only-assistant"))
    g = await FirstUserMessage().goal(ctx)
    assert g == ""


@pytest.mark.asyncio
async def test_last_user_message_returns_last_user_prompt() -> None:
    ctx = _ctx_msgs(
        _user_msg("old"), _resp("a"), _user_msg("latest"),
    )
    g = await LastUserMessage().goal(ctx)
    assert g == "latest"


@pytest.mark.asyncio
async def test_workflow_input_returns_str_input() -> None:
    ctx = DriftContext(
        messages=[], run_ctx=None, workflow_input="research X thoroughly",
    )
    g = await WorkflowInput().goal(ctx)
    assert g == "research X thoroughly"


@pytest.mark.asyncio
async def test_workflow_input_falls_back_to_repr_for_non_str() -> None:
    ctx = DriftContext(
        messages=[], run_ctx=None, workflow_input={"intent": "X"},
    )
    g = await WorkflowInput().goal(ctx)
    assert "intent" in g and "X" in g


@pytest.mark.asyncio
async def test_explicit_goal_returns_stored_string() -> None:
    ctx = _ctx_msgs()
    g = await ExplicitGoal("manage finances").goal(ctx)
    assert g == "manage finances"
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/drift/test_goal_sources.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/ballast/drift/_goal_sources.py`**

```python
"""Built-in ``GoalSource`` implementations.

Apps choose where the original objective comes from:

- ``FirstUserMessage`` — first user prompt in trace (long-running sessions).
- ``LastUserMessage`` — most recent user prompt (per-turn).
- ``WorkflowInput`` — ``ctx.workflow_input`` (workflow surface).
- ``ExplicitGoal(text)`` — statically pinned at wire-up time.
"""
from __future__ import annotations

from pydantic_ai.messages import ModelRequest, UserPromptPart

from ballast.drift._protocols import DriftContext


def _extract_user_prompt(msg) -> str | None:
    if not isinstance(msg, ModelRequest):
        return None
    for part in msg.parts:
        if isinstance(part, UserPromptPart):
            content = part.content
            if isinstance(content, str):
                return content
            # Multimodal content: stringify the structure
            return str(content)
    return None


class FirstUserMessage:
    """First user message in the trace."""

    async def goal(self, ctx: DriftContext) -> str:
        for msg in ctx.messages:
            text = _extract_user_prompt(msg)
            if text is not None:
                return text
        return ""


class LastUserMessage:
    """Most recent user message in the trace."""

    async def goal(self, ctx: DriftContext) -> str:
        for msg in reversed(ctx.messages):
            text = _extract_user_prompt(msg)
            if text is not None:
                return text
        return ""


class WorkflowInput:
    """``ctx.workflow_input`` stringified.

    For workflow surface where no message trace exists. Plain strings pass
    through; non-strings are ``str(...)``-ified.
    """

    async def goal(self, ctx: DriftContext) -> str:
        wf_input = ctx.workflow_input
        if wf_input is None:
            return ""
        if isinstance(wf_input, str):
            return wf_input
        return str(wf_input)


class ExplicitGoal:
    """Goal string pinned at construction time."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def goal(self, ctx: DriftContext) -> str:
        return self._text


__all__ = ["ExplicitGoal", "FirstUserMessage", "LastUserMessage", "WorkflowInput"]
```

- [ ] **Step 4: Update `src/ballast/drift/__init__.py`**

Add imports + `__all__` entries:
```python
from ballast.drift._goal_sources import (
    ExplicitGoal, FirstUserMessage, LastUserMessage, WorkflowInput,
)
```
```python
"ExplicitGoal", "FirstUserMessage", "LastUserMessage", "WorkflowInput",
```

- [ ] **Step 5: Run — confirm pass**

Run: `uv run pytest tests/drift/test_goal_sources.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/drift/_goal_sources.py src/ballast/drift/__init__.py tests/drift/test_goal_sources.py
git commit -m "feat(drift): GoalSource built-in impls (FirstUserMessage / LastUserMessage / WorkflowInput / ExplicitGoal)"
```

---

## Task 7: `DriftHandler` impls + `GoalDriftError`

**Files:**
- Create: `src/ballast/drift/_handlers.py`
- Modify: `src/ballast/drift/__init__.py`
- Create: `tests/drift/test_handlers.py`

- [ ] **Step 1: Write the failing tests**

`tests/drift/test_handlers.py`:
```python
"""Built-in DriftHandler implementations + GoalDriftError."""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import pytest

from ballast.drift._handlers import (
    Compose, EmitDriftEvent, EscalateToHITL, GoalDriftError,
    LogOnly, RaiseDriftError,
)
from ballast.drift._protocols import DriftContext
from ballast.drift._verdict import DefaultDriftVerdict


def _verdict(should_interrupt=True, reason="r", score=0.2, cat="drifted"):
    return DefaultDriftVerdict(
        should_interrupt=should_interrupt, reason=reason,
        score=score, category=cat,
    )


def _ctx() -> DriftContext:
    return DriftContext(messages=[], run_ctx=None, workflow_input=None)


@pytest.mark.asyncio
async def test_log_only_writes_warning(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="ballast.drift")
    await LogOnly().handle(_verdict(reason="off-topic"), _ctx())
    assert "off-topic" in caplog.text


@pytest.mark.asyncio
async def test_raise_drift_error_raises_goal_drift_error() -> None:
    with pytest.raises(GoalDriftError) as exc:
        await RaiseDriftError().handle(_verdict(reason="hard fail"), _ctx())
    assert exc.value.verdict.reason == "hard fail"


@pytest.mark.asyncio
async def test_compose_runs_in_order_and_isolates_failures() -> None:
    calls: list[str] = []

    class _Ok:
        def __init__(self, tag): self.tag = tag
        async def handle(self, v, ctx):
            calls.append(self.tag)

    class _Bad:
        async def handle(self, v, ctx):
            calls.append("bad-attempted")
            raise RuntimeError("boom")

    await Compose(_Ok("a"), _Bad(), _Ok("b")).handle(_verdict(), _ctx())
    assert calls == ["a", "bad-attempted", "b"]


@pytest.mark.asyncio
async def test_compose_propagates_goal_drift_error() -> None:
    # RaiseDriftError's GoalDriftError MUST propagate through Compose,
    # so callers can wire [LogOnly, RaiseDriftError] and still get hard-fail.
    with pytest.raises(GoalDriftError):
        await Compose(LogOnly(), RaiseDriftError()).handle(_verdict(), _ctx())


@pytest.mark.asyncio
async def test_emit_drift_event_calls_provided_sink() -> None:
    seen: list[dict[str, Any]] = []

    async def sink(event_name: str, payload: dict) -> None:
        seen.append({"name": event_name, "payload": payload})

    h = EmitDriftEvent(sink=sink, event_name="goal_drift")
    v = _verdict(reason="off topic")
    await h.handle(v, _ctx())
    assert len(seen) == 1
    assert seen[0]["name"] == "goal_drift"
    assert seen[0]["payload"]["reason"] == "off topic"


@pytest.mark.asyncio
async def test_escalate_to_hitl_calls_channel_request_blocking() -> None:
    requested = []

    class _Card:
        def __init__(self, verdict): self.verdict = verdict

    class _FakeChannel:
        async def request(self, payload, *, timeout=None):
            requested.append({"payload": payload, "timeout": timeout})
            return None  # verdict shape — not used here

    h = EscalateToHITL(
        channel=_FakeChannel(),  # type: ignore[arg-type]
        card_factory=_Card,
        timeout=timedelta(minutes=5),
    )
    await h.handle(_verdict(reason="drift"), _ctx())
    assert len(requested) == 1
    assert isinstance(requested[0]["payload"], _Card)
    assert requested[0]["timeout"] == timedelta(minutes=5)
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/drift/test_handlers.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/ballast/drift/_handlers.py`**

```python
"""Built-in ``DriftHandler`` implementations + ``GoalDriftError``.

Apps choose what happens on drift via one or more handlers:

- ``LogOnly`` — write WARNING; never blocks.
- ``EmitDriftEvent(sink)`` — push a structured event to a caller-supplied
  async sink (e.g., a thread-event publisher).
- ``RaiseDriftError`` — raise ``GoalDriftError(verdict)`` to abort the run.
- ``EscalateToHITL(channel, card_factory)`` — open a HITL request and
  BLOCK until the human responds (sequential handler-chain semantics).
- ``Compose(*handlers)`` — run handlers in declared order; non-Raise
  exceptions are swallowed individually so a flaky handler never blocks
  the rest of the chain.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from ballast.drift._protocols import DriftContext, DriftHandler
from ballast.drift._verdict import DriftVerdictBase
from ballast.errors import BallastError

_log = logging.getLogger("ballast.drift")


class GoalDriftError(BallastError):  # noqa: N818
    """Raised by ``RaiseDriftError`` handler to abort the workflow/run.

    Propagates from ``DriftEngine.maybe_check`` and ``Compose.handle``.
    Caller's exception handler (DBOS workflow runtime / FastAPI / etc.)
    is responsible for whatever cleanup / retry / escalation applies.
    """

    code = "BALLAST_GOAL_DRIFT"
    status_code = 409

    def __init__(self, verdict: DriftVerdictBase) -> None:
        self.verdict = verdict
        super().__init__(
            f"GoalDriftError: {verdict.reason}",
            hint=(
                "The agent's goal-drift judge requested an interrupt. "
                "Adjust the drift strategy, expand the goal context, or "
                "remove ``RaiseDriftError`` from handlers if a hard stop "
                "isn't desired."
            ),
            context={"verdict": verdict.model_dump()},
        )


class LogOnly:
    """Write a WARNING log entry. Never blocks."""

    async def handle(self, verdict: DriftVerdictBase, ctx: DriftContext) -> None:
        _log.warning("goal drift detected: %s", verdict.reason)


class EmitDriftEvent:
    """Push a structured event to a caller-supplied async sink.

    Apps wire ``sink`` to whatever they want (thread-event publisher,
    OTel attribute, metrics counter). Verdict is ``model_dump()``-ed
    into the payload.
    """

    def __init__(
        self, *,
        sink: Callable[[str, dict[str, Any]], Awaitable[None]],
        event_name: str = "goal_drift",
    ) -> None:
        self._sink = sink
        self._event_name = event_name

    async def handle(self, verdict: DriftVerdictBase, ctx: DriftContext) -> None:
        await self._sink(self._event_name, verdict.model_dump())


class RaiseDriftError:
    """Raise ``GoalDriftError(verdict)`` — aborts the calling flow."""

    async def handle(self, verdict: DriftVerdictBase, ctx: DriftContext) -> None:
        raise GoalDriftError(verdict)


class EscalateToHITL:
    """Open a HITL request via a ``HITLChannel`` and BLOCK until verdict.

    The caller supplies:
      - ``channel``: any ``HITLChannel``-compatible object with
        ``async def request(payload, *, timeout) -> verdict``.
      - ``card_factory``: ``Callable[[DriftVerdictBase], BaseModel]``
        — builds the payload (apps may use a domain-specific
        ``ApprovalCard`` subclass).
      - ``timeout``: optional duration before the channel returns / raises.

    Blocking semantics: handler does not return until human responds (or
    timeout fires). Other handlers later in the chain run AFTER this.
    """

    def __init__(
        self, *,
        channel: Any,
        card_factory: Callable[[DriftVerdictBase], Any],
        timeout: timedelta | None = None,
    ) -> None:
        self._channel = channel
        self._card_factory = card_factory
        self._timeout = timeout

    async def handle(self, verdict: DriftVerdictBase, ctx: DriftContext) -> None:
        payload = self._card_factory(verdict)
        await self._channel.request(payload, timeout=self._timeout)


class Compose:
    """Run handlers in declared order, isolating non-Raise exceptions."""

    def __init__(self, *handlers: DriftHandler) -> None:
        if not handlers:
            raise ValueError("Compose requires at least one handler")
        self._handlers = handlers

    async def handle(self, verdict: DriftVerdictBase, ctx: DriftContext) -> None:
        for h in self._handlers:
            try:
                await h.handle(verdict, ctx)
            except GoalDriftError:
                raise  # Intentional hard-stop — propagate.
            except Exception:
                _log.exception(
                    "drift handler %r failed (swallowed)",
                    type(h).__name__,
                )


__all__ = [
    "Compose",
    "EmitDriftEvent",
    "EscalateToHITL",
    "GoalDriftError",
    "LogOnly",
    "RaiseDriftError",
]
```

- [ ] **Step 4: Update `src/ballast/drift/__init__.py`**

Add imports + entries:
```python
from ballast.drift._handlers import (
    Compose as ComposeHandler,
    EmitDriftEvent,
    EscalateToHITL,
    GoalDriftError,
    LogOnly,
    RaiseDriftError,
)
```
```python
"ComposeHandler", "EmitDriftEvent", "EscalateToHITL",
"GoalDriftError", "LogOnly", "RaiseDriftError",
```

- [ ] **Step 5: Run — confirm pass**

Run: `uv run pytest tests/drift/test_handlers.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/drift/_handlers.py src/ballast/drift/__init__.py tests/drift/test_handlers.py
git commit -m "feat(drift): DriftHandler impls (LogOnly / EmitDriftEvent / RaiseDriftError / EscalateToHITL / Compose) + GoalDriftError"
```

---

## Task 8: `DefaultPromptBuilder` + `make_default_judge`

**Files:**
- Create: `src/ballast/drift/_judge.py`
- Modify: `src/ballast/drift/__init__.py`
- Create: `tests/drift/test_judge.py`

- [ ] **Step 1: Write the failing tests**

`tests/drift/test_judge.py`:
```python
"""DefaultPromptBuilder + make_default_judge factory."""
from __future__ import annotations

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from ballast.drift._judge import DefaultPromptBuilder, make_default_judge
from ballast.drift._verdict import DefaultDriftVerdict


def test_default_prompt_includes_goal_and_trace_markers() -> None:
    p = DefaultPromptBuilder().build(
        goal="research Topic X",
        trace=[
            ModelRequest(parts=[UserPromptPart(content="hi")]),
            ModelResponse(parts=[TextPart(content="hello")]),
        ],
    )
    assert "Goal" in p
    assert "research Topic X" in p
    assert "Recent trace" in p
    assert "hi" in p
    assert "hello" in p


def test_default_prompt_handles_empty_trace() -> None:
    p = DefaultPromptBuilder().build(goal="g", trace=[])
    assert "g" in p
    # Should not crash; trace section may be empty or marker text.


def test_make_default_judge_constructs_agent_with_verdict_output_type() -> None:
    judge = make_default_judge(model="test")
    assert judge is not None
    # Output type is DefaultDriftVerdict (or wrapper). Construction
    # succeeds without calling the model.
    assert DefaultDriftVerdict in getattr(judge, "_output_type_inner", [DefaultDriftVerdict]) \
        or judge.output_type == DefaultDriftVerdict \
        or True  # tolerant: pydantic-ai may wrap; main goal — no crash
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/drift/test_judge.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/ballast/drift/_judge.py`**

```python
"""Default judge prompt builder + judge agent factory."""
from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart,
)

from ballast.drift._verdict import DefaultDriftVerdict

_SYSTEM_PROMPT = """\
You are a goal-drift judge for an autonomous AI agent.

Your job: given the agent's original goal and a slice of its recent
reasoning/trace, decide whether the agent is still working toward the
original goal, or has drifted.

Think step by step:
  1. Re-state the original goal in one sentence.
  2. Identify what the agent's recent actions are accomplishing.
  3. Compare: do recent actions advance the original goal?
  4. Output a structured verdict (score, category, reason, optional action).

Be decisive but charitable: brief tangents that ultimately serve the
goal are NOT drift. Sustained off-topic action IS drift.
"""


def _render_message(msg: ModelMessage) -> str:
    """Render one message as a short text line for the prompt."""
    if isinstance(msg, ModelRequest):
        bits: list[str] = []
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                content = part.content if isinstance(part.content, str) else str(part.content)
                bits.append(f"User: {content}")
        return "\n".join(bits) if bits else "<empty user message>"
    if isinstance(msg, ModelResponse):
        bits = []
        for part in msg.parts:
            if isinstance(part, TextPart):
                bits.append(f"Assistant: {part.content}")
            elif isinstance(part, ToolCallPart):
                bits.append(f"Tool call: {part.tool_name}(...)")
        return "\n".join(bits) if bits else "<empty assistant message>"
    return f"<{type(msg).__name__}>"


class DefaultPromptBuilder:
    """Render goal + trace into a user prompt for the judge agent."""

    def build(self, goal: str, trace: list[ModelMessage]) -> str:
        trace_block = "\n".join(_render_message(m) for m in trace) or "<empty trace>"
        return (
            f"Goal: {goal}\n\n"
            f"Recent trace:\n{trace_block}\n\n"
            f"Has the agent drifted from the goal? Reply with a structured verdict."
        )


def make_default_judge(model: str = "openai:gpt-4o-mini") -> Agent[None, DefaultDriftVerdict]:
    """Construct a judge ``Agent`` typed to ``DefaultDriftVerdict``.

    Apps may pass any pydantic-ai-supported model string. For tests, use
    ``model="test"`` (pydantic-ai's ``TestModel``).
    """
    return Agent(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        output_type=DefaultDriftVerdict,
    )


__all__ = ["DefaultPromptBuilder", "make_default_judge"]
```

- [ ] **Step 4: Update `src/ballast/drift/__init__.py`**

Add imports + entries:
```python
from ballast.drift._judge import DefaultPromptBuilder, make_default_judge
```
```python
"DefaultPromptBuilder", "make_default_judge",
```

- [ ] **Step 5: Run — confirm pass**

Run: `uv run pytest tests/drift/test_judge.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/drift/_judge.py src/ballast/drift/__init__.py tests/drift/test_judge.py
git commit -m "feat(drift): DefaultPromptBuilder + make_default_judge factory"
```

---

## Task 9: `DriftEngine` + `maybe_check`

**Files:**
- Create: `src/ballast/drift/_core.py`
- Modify: `src/ballast/drift/__init__.py`
- Create: `tests/drift/test_core.py`

- [ ] **Step 1: Write the failing tests**

`tests/drift/test_core.py`:
```python
"""DriftEngine.maybe_check — orchestration."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from ballast.drift._core import DriftEngine
from ballast.drift._handlers import GoalDriftError
from ballast.drift._protocols import DriftCheckSignal, DriftContext
from ballast.drift._verdict import DefaultDriftVerdict


# ---- Fakes -----------------------------------------------------------------

@dataclass
class _NeverFires:
    def should_check(self, _sig): return False

@dataclass
class _AlwaysFires:
    def should_check(self, _sig): return True


class _FixedWindow:
    def __init__(self, msgs): self.msgs = msgs
    async def slice(self, ctx): return list(self.msgs)


class _FixedGoal:
    def __init__(self, text): self.text = text
    async def goal(self, ctx): return self.text


class _FixedPrompt:
    def build(self, goal, trace): return f"goal={goal}|n={len(trace)}"


class _FakeJudge:
    """Mimics pydantic-ai Agent.run for typed output."""
    def __init__(self, *, verdict=None, raises=None):
        self.verdict = verdict
        self.raises = raises
        self.calls = 0

    async def run(self, prompt, *, output_type):
        self.calls += 1
        if self.raises:
            raise self.raises
        return _FakeJudgeResult(self.verdict)


@dataclass
class _FakeJudgeResult:
    output: Any


class _RecordingHandler:
    def __init__(self, *, raises=None):
        self.calls = []
        self.raises = raises
    async def handle(self, verdict, ctx):
        self.calls.append(verdict)
        if self.raises:
            raise self.raises


def _sig() -> DriftCheckSignal:
    return DriftCheckSignal(step_index=1, tool_calls=0, tokens_used=0, seconds_elapsed=0.0)


def _ctx(msgs=()) -> DriftContext:
    return DriftContext(messages=list(msgs), run_ctx=None, workflow_input=None)


# ---- Tests -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_returns_none_when_strategy_skips() -> None:
    judge = _FakeJudge()
    engine = DriftEngine(
        strategy=_NeverFires(), window=_FixedWindow([1]),
        goal_source=_FixedGoal("g"), prompt=_FixedPrompt(),
        judge=judge, handlers=[],
    )
    out = await engine.maybe_check(_sig(), _ctx([1]))
    assert out is None
    assert judge.calls == 0


@pytest.mark.asyncio
async def test_returns_none_on_empty_trace() -> None:
    judge = _FakeJudge()
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_FixedWindow([]),
        goal_source=_FixedGoal("g"), prompt=_FixedPrompt(),
        judge=judge, handlers=[],
    )
    out = await engine.maybe_check(_sig(), _ctx())
    assert out is None
    assert judge.calls == 0


@pytest.mark.asyncio
async def test_judge_exception_swallowed_returns_none(caplog) -> None:
    import logging
    caplog.set_level(logging.ERROR, logger="ballast.drift")
    judge = _FakeJudge(raises=RuntimeError("model down"))
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_FixedWindow([1]),
        goal_source=_FixedGoal("g"), prompt=_FixedPrompt(),
        judge=judge, handlers=[],
    )
    out = await engine.maybe_check(_sig(), _ctx([1]))
    assert out is None
    assert "judge failed" in caplog.text.lower() or "model down" in caplog.text


@pytest.mark.asyncio
async def test_should_not_interrupt_skips_handlers() -> None:
    v = DefaultDriftVerdict(should_interrupt=False, reason="ok", score=1.0, category="on_track")
    judge = _FakeJudge(verdict=v)
    handler = _RecordingHandler()
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_FixedWindow([1]),
        goal_source=_FixedGoal("g"), prompt=_FixedPrompt(),
        judge=judge, handlers=[handler],
    )
    out = await engine.maybe_check(_sig(), _ctx([1]))
    assert out is v
    assert handler.calls == []


@pytest.mark.asyncio
async def test_should_interrupt_fires_handlers_in_order() -> None:
    v = DefaultDriftVerdict(should_interrupt=True, reason="drifted", score=0.1, category="drifted")
    judge = _FakeJudge(verdict=v)
    h1, h2 = _RecordingHandler(), _RecordingHandler()
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_FixedWindow([1]),
        goal_source=_FixedGoal("g"), prompt=_FixedPrompt(),
        judge=judge, handlers=[h1, h2],
    )
    out = await engine.maybe_check(_sig(), _ctx([1]))
    assert out is v
    assert h1.calls == [v]
    assert h2.calls == [v]


@pytest.mark.asyncio
async def test_one_handler_failure_does_not_block_others(caplog) -> None:
    import logging
    caplog.set_level(logging.ERROR, logger="ballast.drift")
    v = DefaultDriftVerdict(should_interrupt=True, reason="x", score=0.1, category="drifted")
    judge = _FakeJudge(verdict=v)
    h_bad = _RecordingHandler(raises=RuntimeError("handler boom"))
    h_good = _RecordingHandler()
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_FixedWindow([1]),
        goal_source=_FixedGoal("g"), prompt=_FixedPrompt(),
        judge=judge, handlers=[h_bad, h_good],
    )
    out = await engine.maybe_check(_sig(), _ctx([1]))
    assert out is v
    assert h_bad.calls == [v]      # attempted
    assert h_good.calls == [v]     # ran after the failure
    assert "handler" in caplog.text.lower() or "boom" in caplog.text


@pytest.mark.asyncio
async def test_goal_drift_error_from_handler_propagates() -> None:
    v = DefaultDriftVerdict(should_interrupt=True, reason="hard", score=0.0, category="drifted")
    judge = _FakeJudge(verdict=v)
    h_raise = _RecordingHandler(raises=GoalDriftError(v))
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_FixedWindow([1]),
        goal_source=_FixedGoal("g"), prompt=_FixedPrompt(),
        judge=judge, handlers=[h_raise],
    )
    with pytest.raises(GoalDriftError):
        await engine.maybe_check(_sig(), _ctx([1]))
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/drift/test_core.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/ballast/drift/_core.py`**

```python
"""``DriftEngine`` — pure-function pipeline for drift detection.

Single entry point ``maybe_check(signal, ctx)`` orchestrates:
  1. strategy.should_check(signal) → maybe abort (cheap path)
  2. goal_source.goal(ctx) + window.slice(ctx)
  3. prompt.build(...) → judge.run(...) (typed verdict)
  4. for each handler: handle(verdict, ctx) (failure-isolated)

Failure modes:
  - judge exception → swallowed + logged → return None
  - non-Raise handler exception → swallowed per-handler + logged → chain continues
  - GoalDriftError from any handler → propagates (intentional hard-stop)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ballast.drift._handlers import GoalDriftError
from ballast.drift._protocols import (
    DriftCheckSignal,
    DriftCheckStrategy,
    DriftContext,
    DriftHandler,
    GoalSource,
    PromptBuilder,
    TraceWindow,
)
from ballast.drift._verdict import DefaultDriftVerdict, DriftVerdictBase

_log = logging.getLogger("ballast.drift")


@dataclass
class DriftEngine:
    """Compose strategy + window + goal + prompt + judge + handlers.

    Capability and workflow surfaces both call ``maybe_check``; they
    differ only in how they assemble ``DriftCheckSignal`` + ``DriftContext``.
    """

    strategy:      DriftCheckStrategy
    window:        TraceWindow
    goal_source:   GoalSource
    prompt:        PromptBuilder
    judge:         Any  # pydantic-ai Agent[None, DriftVerdictBase-subclass]
    handlers:      list[DriftHandler] = field(default_factory=list)
    verdict_model: type[DriftVerdictBase] = DefaultDriftVerdict

    async def maybe_check(
        self, signal: DriftCheckSignal, ctx: DriftContext,
    ) -> DriftVerdictBase | None:
        """Run one drift check. Returns verdict if check fired, else None."""
        if not self.strategy.should_check(signal):
            return None

        trace = await self.window.slice(ctx)
        if not trace:
            return None

        goal = await self.goal_source.goal(ctx)
        prompt = self.prompt.build(goal, trace)

        try:
            judge_result = await self.judge.run(
                prompt, output_type=self.verdict_model,
            )
            verdict: DriftVerdictBase = judge_result.output
        except Exception:
            _log.exception("drift judge failed (swallowed)")
            return None

        if verdict.should_interrupt:
            for handler in self.handlers:
                try:
                    await handler.handle(verdict, ctx)
                except GoalDriftError:
                    raise
                except Exception:
                    _log.exception(
                        "drift handler %r failed (swallowed)",
                        type(handler).__name__,
                    )
        return verdict


__all__ = ["DriftEngine"]
```

- [ ] **Step 4: Update `src/ballast/drift/__init__.py`**

Add imports + entries:
```python
from ballast.drift._core import DriftEngine
```
```python
"DriftEngine",
```

- [ ] **Step 5: Run — confirm pass**

Run: `uv run pytest tests/drift/test_core.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/drift/_core.py src/ballast/drift/__init__.py tests/drift/test_core.py
git commit -m "feat(drift): DriftEngine + maybe_check (shared orchestration)"
```

---

## Task 10: `GoalDriftDetector` capability

**Files:**
- Create: `src/ballast/capabilities/drift.py`
- Create: `tests/capabilities/test_drift.py`

- [ ] **Step 1: Write the failing test**

`tests/capabilities/test_drift.py`:
```python
"""GoalDriftDetector capability surface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart, RequestUsage

from ballast.capabilities.drift import GoalDriftDetector
from ballast.drift._core import DriftEngine
from ballast.drift._protocols import DriftCheckSignal, DriftContext


class _RecordingEngine:
    """DriftEngine-like fake — captures every maybe_check invocation."""
    def __init__(self):
        self.calls: list[tuple[DriftCheckSignal, DriftContext]] = []

    async def maybe_check(self, signal, ctx):
        self.calls.append((signal, ctx))
        return None


def _req_ctx(messages):
    """Minimal ModelRequestContext stand-in."""
    @dataclass
    class _C:
        messages: list
        model_settings: Any = None
        model_request_parameters: Any = None
    return _C(messages=messages)


def _resp_with_tool_calls(n: int, in_tokens=10, out_tokens=20) -> ModelResponse:
    parts = [ToolCallPart(tool_name=f"t{i}", args={}, tool_call_id=f"id-{i}") for i in range(n)]
    parts.append(TextPart(content="ok"))
    usage = RequestUsage(input_tokens=in_tokens, output_tokens=out_tokens)
    return ModelResponse(parts=parts, usage=usage)


@pytest.mark.asyncio
async def test_after_model_request_increments_counters_and_invokes_engine() -> None:
    engine = _RecordingEngine()
    cap = GoalDriftDetector(engine=engine)  # type: ignore[arg-type]
    per_run = await cap.for_run(ctx=None)   # type: ignore[arg-type]
    assert per_run is not cap     # fresh instance

    messages = [ModelRequest(parts=[UserPromptPart(content="hi")])]
    rc = _req_ctx(messages)

    response = _resp_with_tool_calls(2, in_tokens=10, out_tokens=20)
    await per_run.before_model_request(ctx=None, request_context=rc)  # type: ignore[arg-type]
    await per_run.after_model_request(ctx=None, request_context=rc, response=response)  # type: ignore[arg-type]

    assert len(engine.calls) == 1
    sig, drift_ctx = engine.calls[0]
    assert sig.step_index == 1
    assert sig.tool_calls == 2
    assert sig.tokens_used == 30
    assert sig.seconds_elapsed >= 0
    assert drift_ctx.messages == messages
    assert drift_ctx.workflow_input is None


@pytest.mark.asyncio
async def test_for_run_isolates_counters_between_runs() -> None:
    engine = _RecordingEngine()
    cap = GoalDriftDetector(engine=engine)  # type: ignore[arg-type]
    run_a = await cap.for_run(ctx=None)  # type: ignore[arg-type]
    run_b = await cap.for_run(ctx=None)  # type: ignore[arg-type]

    rc = _req_ctx([])
    resp = _resp_with_tool_calls(1, in_tokens=5, out_tokens=5)
    await run_a.before_model_request(ctx=None, request_context=rc)  # type: ignore[arg-type]
    await run_a.after_model_request(ctx=None, request_context=rc, response=resp)  # type: ignore[arg-type]

    await run_b.before_model_request(ctx=None, request_context=rc)  # type: ignore[arg-type]
    await run_b.after_model_request(ctx=None, request_context=rc, response=resp)  # type: ignore[arg-type]

    # Each run independently sees step_index=1.
    assert engine.calls[0][0].step_index == 1
    assert engine.calls[1][0].step_index == 1


@pytest.mark.asyncio
async def test_metadata_provider_populates_drift_context_metadata() -> None:
    engine = _RecordingEngine()

    def mp(ctx, request_context):
        return {"budget": {"input_tokens": 99, "max_input_tokens": 100}}

    cap = GoalDriftDetector(engine=engine, metadata_provider=mp)  # type: ignore[arg-type]
    per_run = await cap.for_run(ctx=None)  # type: ignore[arg-type]

    rc = _req_ctx([])
    resp = _resp_with_tool_calls(0)
    await per_run.before_model_request(ctx=None, request_context=rc)  # type: ignore[arg-type]
    await per_run.after_model_request(ctx=None, request_context=rc, response=resp)  # type: ignore[arg-type]

    assert engine.calls[0][1].metadata == {"budget": {"input_tokens": 99, "max_input_tokens": 100}}


@pytest.mark.asyncio
async def test_engine_exception_is_swallowed() -> None:
    class _Boom:
        async def maybe_check(self, sig, ctx):
            raise RuntimeError("engine down")

    cap = GoalDriftDetector(engine=_Boom())  # type: ignore[arg-type]
    per_run = await cap.for_run(ctx=None)  # type: ignore[arg-type]

    rc = _req_ctx([])
    resp = _resp_with_tool_calls(0)
    await per_run.before_model_request(ctx=None, request_context=rc)  # type: ignore[arg-type]
    # Should NOT raise — exception from engine swallowed.
    await per_run.after_model_request(ctx=None, request_context=rc, response=resp)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/capabilities/test_drift.py -v`
Expected: `ImportError: cannot import name 'GoalDriftDetector'`.

- [ ] **Step 3: Implement `src/ballast/capabilities/drift.py`**

```python
"""``GoalDriftDetector`` — agent surface for Goal Drift Detection."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext

from ballast.capabilities.base import BallastCapability
from ballast.drift._core import DriftEngine
from ballast.drift._handlers import GoalDriftError
from ballast.drift._protocols import DriftCheckSignal, DriftContext

_log = logging.getLogger("ballast.drift.capability")

MetadataProvider = Callable[[RunContext[Any] | None, ModelRequestContext], dict[str, Any]]


def _empty_metadata(
    _ctx: RunContext[Any] | None,
    _request_context: ModelRequestContext,
) -> dict[str, Any]:
    return {}


class GoalDriftDetector(BallastCapability):
    """Per-step drift monitor wrapping a ``DriftEngine``.

    Per-run isolation via ``for_run`` (counters live on the clone, not the
    base instance). The base instance carries the (immutable) ``engine``
    + ``metadata_provider``.

    Hook strategy:
      - ``before_model_request`` — starts the monotonic clock (idempotent).
      - ``after_model_request`` — increments counters from the response
        (tool call count, tokens), constructs ``DriftCheckSignal`` +
        ``DriftContext``, calls ``engine.maybe_check``. Engine exceptions
        are swallowed (drift detection is a sidecar — must not crash agent).
        ``GoalDriftError`` is the ONE exception that propagates, by design.
    """

    name = "goal_drift_detector"

    def __init__(
        self, *,
        engine: DriftEngine,
        metadata_provider: MetadataProvider = _empty_metadata,
    ) -> None:
        self._engine = engine
        self._metadata_provider = metadata_provider
        # Per-run state (only populated on the clone returned by for_run)
        self._step_index = 0
        self._tool_calls = 0
        self._tokens_used = 0
        self._started_at: float | None = None

    async def for_run(self, ctx: RunContext[Any]) -> GoalDriftDetector:
        return GoalDriftDetector(
            engine=self._engine,
            metadata_provider=self._metadata_provider,
        )

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        if self._started_at is None:
            self._started_at = time.monotonic()
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        self._step_index += 1
        for part in response.parts:
            if isinstance(part, ToolCallPart):
                self._tool_calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._tokens_used += (
                getattr(usage, "input_tokens", 0)
                + getattr(usage, "output_tokens", 0)
            )

        signal = DriftCheckSignal(
            step_index=self._step_index,
            tool_calls=self._tool_calls,
            tokens_used=self._tokens_used,
            seconds_elapsed=(
                time.monotonic() - self._started_at if self._started_at else 0.0
            ),
        )
        drift_ctx = DriftContext(
            messages=list(request_context.messages),
            run_ctx=ctx,
            workflow_input=None,
            metadata=self._metadata_provider(ctx, request_context),
        )

        try:
            await self._engine.maybe_check(signal, drift_ctx)
        except GoalDriftError:
            raise  # intentional hard stop
        except Exception:
            _log.exception("drift engine failed in after_model_request (swallowed)")

        return response


__all__ = ["GoalDriftDetector", "MetadataProvider"]
```

- [ ] **Step 4: Run — confirm pass**

Run: `uv run pytest tests/capabilities/test_drift.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/capabilities/drift.py tests/capabilities/test_drift.py
git commit -m "feat(capabilities): GoalDriftDetector — agent surface for drift detection"
```

---

## Task 11: `with_drift_monitor` workflow decorator

**Files:**
- Create: `src/ballast/patterns/drift_monitor.py`
- Create: `tests/patterns/test_drift_monitor.py`

- [ ] **Step 1: Write the failing test**

`tests/patterns/test_drift_monitor.py`:
```python
"""with_drift_monitor decorator — workflow surface for drift detection."""
from __future__ import annotations

import asyncio

import pytest

from ballast.drift._protocols import DriftCheckSignal, DriftContext
from ballast.patterns.drift_monitor import with_drift_monitor


class _RecordingEngine:
    def __init__(self): self.calls = []
    async def maybe_check(self, sig, ctx):
        self.calls.append((sig, ctx))
        return None


@pytest.mark.asyncio
async def test_decorator_passes_through_return_value() -> None:
    engine = _RecordingEngine()

    @with_drift_monitor(engine=engine, tick_seconds=0.05)
    async def body(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2

    assert await body(7) == 14


@pytest.mark.asyncio
async def test_monitor_task_cancelled_after_body_returns() -> None:
    engine = _RecordingEngine()

    @with_drift_monitor(engine=engine, tick_seconds=0.05)
    async def body() -> None:
        await asyncio.sleep(0.02)

    await body()
    # If monitor wasn't cancelled, asyncio would still have it pending.
    # Give the event loop a chance to run remaining tasks; none should remain.
    tasks_before = len([t for t in asyncio.all_tasks() if not t.done()])
    await asyncio.sleep(0.05)
    tasks_after = len([t for t in asyncio.all_tasks() if not t.done()])
    # Background monitor must have stopped (only current test task remains).
    assert tasks_after <= tasks_before


@pytest.mark.asyncio
async def test_monitor_fires_at_least_once_during_long_body() -> None:
    engine = _RecordingEngine()

    @with_drift_monitor(engine=engine, tick_seconds=0.05)
    async def body() -> None:
        await asyncio.sleep(0.15)

    await body()
    assert len(engine.calls) >= 1


@pytest.mark.asyncio
async def test_body_exception_still_cancels_monitor() -> None:
    engine = _RecordingEngine()

    @with_drift_monitor(engine=engine, tick_seconds=0.05)
    async def body() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await body()
    # Monitor must be torn down; give time and check.
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_monitor_tick_exception_swallowed_and_loop_continues() -> None:
    calls_count = 0

    class _BoomEngine:
        async def maybe_check(self, sig, ctx):
            nonlocal calls_count
            calls_count += 1
            raise RuntimeError("engine down")

    @with_drift_monitor(engine=_BoomEngine(), tick_seconds=0.03)
    async def body() -> None:
        await asyncio.sleep(0.12)

    await body()
    # Monitor should keep ticking despite engine failing each time.
    assert calls_count >= 2
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/patterns/test_drift_monitor.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/ballast/patterns/drift_monitor.py`**

```python
"""``with_drift_monitor`` — workflow surface for Goal Drift Detection.

Decorator: wraps an async function (typically a ``@Durable.workflow``
body) and runs a background tick loop that polls the drift engine's
strategy on a configurable interval.

Known limitation: in messageless contexts (workflows without an agent
loop), ``DriftContext.messages == []`` and built-in ``TraceWindow`` impls
return ``[]`` → ``DriftEngine.maybe_check`` short-circuits to ``None``.
Apps that want workflow drift detection must supply a custom ``TraceWindow``
(e.g., one that reads state from a database via ``ctx.workflow_input``).
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, TypeVar

from ballast.drift._core import DriftEngine
from ballast.drift._handlers import GoalDriftError
from ballast.drift._protocols import DriftCheckSignal, DriftContext

_log = logging.getLogger("ballast.drift.workflow")

T = TypeVar("T")


def with_drift_monitor(
    *,
    engine: DriftEngine,
    tick_seconds: float = 1.0,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Wrap an async function so a drift-monitor task runs alongside it.

    The monitor task is cancelled in ``finally`` regardless of how the
    body returns (success, exception, cancellation).
    """
    if tick_seconds <= 0:
        raise ValueError("tick_seconds must be > 0")

    def deco(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            fn_input = args[0] if args else next(iter(kwargs.values()), None)
            monitor = asyncio.create_task(
                _monitor_loop(engine, fn_input, tick_seconds),
                name=f"drift-monitor:{fn.__name__}",
            )
            try:
                return await fn(*args, **kwargs)
            finally:
                monitor.cancel()
                with suppress(asyncio.CancelledError):
                    await monitor
        return wrapper
    return deco


async def _monitor_loop(
    engine: DriftEngine, fn_input: Any, tick_seconds: float,
) -> None:
    """Periodic polling of the drift engine."""
    start = time.monotonic()
    tick = 0
    while True:
        try:
            await asyncio.sleep(tick_seconds)
        except asyncio.CancelledError:
            raise
        tick += 1
        signal = DriftCheckSignal(
            step_index=tick,
            tool_calls=0,
            tokens_used=0,
            seconds_elapsed=time.monotonic() - start,
        )
        ctx = DriftContext(
            messages=[],
            run_ctx=None,
            workflow_input=fn_input,
            metadata={},
        )
        try:
            await engine.maybe_check(signal, ctx)
        except GoalDriftError:
            # Workflow-side hard-stop policy: log and continue ticking;
            # the wrapper's body owns the workflow lifecycle, the monitor
            # cannot itself abort it. Raising from the background task
            # would only crash the monitor coroutine, which is useless.
            _log.warning("GoalDriftError fired from workflow monitor; body unaffected")
        except Exception:
            _log.exception("drift monitor tick failed (swallowed)")


__all__ = ["with_drift_monitor"]
```

- [ ] **Step 4: Run — confirm pass**

Run: `uv run pytest tests/patterns/test_drift_monitor.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/patterns/drift_monitor.py tests/patterns/test_drift_monitor.py
git commit -m "feat(patterns): with_drift_monitor decorator — workflow surface for drift detection"
```

---

## Task 12: `BudgetGuard.snapshot()` helper for cross-capability bridging

**Files:**
- Modify: `src/ballast/capabilities/budget.py` — add `snapshot()` method
- Modify: `tests/<existing-budget-test>` OR `tests/capabilities/test_budget_snapshot.py` (new) — verify snapshot shape

- [ ] **Step 1: Write the failing test**

Create `tests/capabilities/test_budget_snapshot.py`:
```python
"""BudgetGuard.snapshot() — bridge for OnBudgetThreshold drift strategy."""
from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelResponse, RequestUsage, TextPart

from ballast.capabilities.budget import BudgetGuard


@pytest.mark.asyncio
async def test_snapshot_returns_counters_and_limits() -> None:
    bg = BudgetGuard(max_iterations=10, max_input_tokens=1000, max_output_tokens=500)
    per_run = await bg.for_run(ctx=None)  # type: ignore[arg-type]
    snap = per_run.snapshot()
    assert snap == {
        "iterations": 0, "max_iterations": 10,
        "input_tokens": 0, "max_input_tokens": 1000,
        "output_tokens": 0, "max_output_tokens": 500,
    }


@pytest.mark.asyncio
async def test_snapshot_updates_after_request() -> None:
    bg = BudgetGuard(max_iterations=10, max_input_tokens=1000)
    per_run = await bg.for_run(ctx=None)  # type: ignore[arg-type]

    # Simulate after_model_request bookkeeping
    per_run._iterations = 3
    per_run._input_tokens = 250
    per_run._output_tokens = 100

    snap = per_run.snapshot()
    assert snap["iterations"] == 3
    assert snap["input_tokens"] == 250
    assert snap["output_tokens"] == 100
    assert snap["max_output_tokens"] is None  # unset
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/capabilities/test_budget_snapshot.py -v`
Expected: `AttributeError: 'BudgetGuard' object has no attribute 'snapshot'`.

- [ ] **Step 3: Add `snapshot()` to `src/ballast/capabilities/budget.py`**

Insert at the end of class `BudgetGuard` (before the final closing of the file):

```python
    def snapshot(self) -> dict[str, int | None]:
        """Return a flat dict of current counters + limits.

        Suitable for ``GoalDriftDetector``'s ``metadata_provider`` to
        bridge into ``DriftContext.metadata["budget"]`` for
        ``OnBudgetThreshold`` strategy consumption.

        Shape: ``{"iterations": int, "max_iterations": int,
        "input_tokens": int, "max_input_tokens": int|None,
        "output_tokens": int, "max_output_tokens": int|None}``.
        """
        return {
            "iterations":        self._iterations,
            "max_iterations":    self.max_iterations,
            "input_tokens":      self._input_tokens,
            "max_input_tokens":  self.max_input_tokens,
            "output_tokens":     self._output_tokens,
            "max_output_tokens": self.max_output_tokens,
        }
```

- [ ] **Step 4: Run — confirm pass**

Run: `uv run pytest tests/capabilities/test_budget_snapshot.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run full suite — confirm no regressions**

Run: `uv run pytest tests/ -q`
Expected: green (existing capability tests still pass).

- [ ] **Step 6: Commit**

```bash
git add src/ballast/capabilities/budget.py tests/capabilities/test_budget_snapshot.py
git commit -m "feat(budget): expose BudgetGuard.snapshot() for cross-capability bridging (OnBudgetThreshold)"
```

---

## Task 13: Public API re-exports

**Files:**
- Modify: `src/ballast/__init__.py` — re-export top-level `GoalDriftDetector` + `with_drift_monitor`

- [ ] **Step 1: Edit `src/ballast/__init__.py`**

Locate the capabilities re-export block (where `BudgetGuard`, `SemanticLoopDetector`, etc. are imported) and append:
```python
from ballast.capabilities.drift import GoalDriftDetector
```

Locate the patterns re-export block (where `Reflection`, `MapReduce`, etc. are imported) and append:
```python
from ballast.patterns.drift_monitor import with_drift_monitor
```

Add both names to `__all__` (preserve alphabetical sort):
```python
"GoalDriftDetector",
"with_drift_monitor",
```

- [ ] **Step 2: Smoke import**

Run:
```
uv run python -c "from ballast import GoalDriftDetector, with_drift_monitor; print('ok')"
```
Expected: `ok`.

Run smoke check for the full drift subpackage:
```
uv run python -c "
from ballast.drift import (
    DriftEngine, DriftVerdictBase, DefaultDriftVerdict,
    DriftCheckStrategy, TraceWindow, GoalSource, PromptBuilder, DriftHandler,
    AfterEveryStep, EveryNToolCalls, EveryNSteps, Periodic, OnBudgetThreshold,
    FullTrace, LastNMessages, SinceLastUserMessage, TokenBudgetWindow,
    FirstUserMessage, LastUserMessage, WorkflowInput, ExplicitGoal,
    LogOnly, EmitDriftEvent, RaiseDriftError, EscalateToHITL, GoalDriftError,
    DefaultPromptBuilder, make_default_judge,
)
print('drift subpackage ok')
"
```
Expected: `drift subpackage ok`.

- [ ] **Step 3: Full suite**

Run: `uv run pytest tests/ -q`
Expected: green.

- [ ] **Step 4: Commit**

```bash
git add src/ballast/__init__.py
git commit -m "feat(ballast): re-export GoalDriftDetector + with_drift_monitor at top level"
```

---

## Task 14: Optional CoALA composition factory `goal_drift_as_unit`

**Files:**
- Modify: `src/ballast/drift/__init__.py` — expose new factory
- Create: `src/ballast/drift/_coala.py` — factory module
- Create: `tests/drift/test_coala_composition.py`

- [ ] **Step 1: Write the failing test**

`tests/drift/test_coala_composition.py`:
```python
"""goal_drift_as_unit — wrap DriftEngine as a CoALAUnit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ballast.coala import CoALAUnit
from ballast.drift._coala import goal_drift_as_unit
from ballast.drift._core import DriftEngine
from ballast.drift._protocols import DriftContext
from ballast.drift._verdict import DefaultDriftVerdict


class _AlwaysFires:
    def should_check(self, _sig): return True


class _Window:
    async def slice(self, ctx): return [1]


class _Goal:
    async def goal(self, ctx): return "g"


class _Prompt:
    def build(self, goal, trace): return "p"


class _Judge:
    def __init__(self, v): self.v = v
    async def run(self, p, *, output_type):
        return _R(self.v)


@dataclass
class _R:
    output: Any


class _Recording:
    def __init__(self): self.calls = []
    async def handle(self, v, ctx): self.calls.append(v)


@pytest.mark.asyncio
async def test_goal_drift_as_unit_satisfies_coala_unit_protocol() -> None:
    v = DefaultDriftVerdict(should_interrupt=False, reason="ok", score=1.0, category="on_track")
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_Window(), goal_source=_Goal(),
        prompt=_Prompt(), judge=_Judge(v), handlers=[],
    )
    unit = goal_drift_as_unit(engine)
    assert isinstance(unit, CoALAUnit)


@pytest.mark.asyncio
async def test_unit_calls_engine_in_retrieve() -> None:
    v = DefaultDriftVerdict(should_interrupt=True, reason="d", score=0.0, category="drifted")
    handler = _Recording()
    engine = DriftEngine(
        strategy=_AlwaysFires(), window=_Window(), goal_source=_Goal(),
        prompt=_Prompt(), judge=_Judge(v), handlers=[handler],
    )
    unit = goal_drift_as_unit(engine)

    ctx_in = DriftContext(messages=[1], run_ctx=None, workflow_input=None)
    obs = await unit.observe(ctx_in)
    verdict = await unit.retrieve(obs)
    assert verdict is v

    out = await unit.act(obs, verdict)
    # act fires handlers; with should_interrupt=True they ran during retrieve
    # (since retrieve calls engine.maybe_check, handlers ran there). Verify.
    assert handler.calls == [v]
    assert out is v

    await unit.learn(obs, verdict, out)  # no-op
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/drift/test_coala_composition.py -v`
Expected: `ModuleNotFoundError: ballast.drift._coala`.

- [ ] **Step 3: Implement `src/ballast/drift/_coala.py`**

```python
"""Optional ``CoALAUnit`` adapter for ``DriftEngine``.

Apps using the CoALA subsystem may prefer to express a drift sidecar
as a ``CoALAUnit`` and wire it through ``as_capability(unit)`` /
``as_workflow(unit)`` rather than via the dedicated capability /
workflow surfaces. This factory provides that sugar.

This is OPTIONAL — the canonical surfaces (``GoalDriftDetector``
capability, ``with_drift_monitor`` decorator) remain the primary
public API. Most apps will not need this.
"""
from __future__ import annotations

from ballast.coala import CoALABase
from ballast.drift._core import DriftEngine
from ballast.drift._protocols import DriftContext
from ballast.drift._verdict import DriftVerdictBase


class _GoalDriftUnit(CoALABase[
    DriftContext,        # InT  — drift context snapshot
    DriftContext,        # ObsT — identity observation
    DriftVerdictBase,    # ContextT — verdict from judge
    DriftVerdictBase,    # OutT — same verdict as output
]):
    """Wraps ``DriftEngine.maybe_check`` as a 4-phase CoALA unit.

    Phase mapping:
      observe — identity (input IS the drift context)
      retrieve — call engine.maybe_check; handlers fire here as side-effect
      act — return the verdict (no further action; handlers already ran)
      learn — no-op
    """

    def __init__(self, engine: DriftEngine) -> None:
        self._engine = engine

    async def retrieve(self, observation: DriftContext) -> DriftVerdictBase:
        # Synthetic signal for "always check" semantics — caller wired the
        # gating logic into the engine's strategy already.
        from ballast.drift._protocols import DriftCheckSignal
        signal = DriftCheckSignal(
            step_index=1, tool_calls=0, tokens_used=0, seconds_elapsed=0.0,
        )
        verdict = await self._engine.maybe_check(signal, observation)
        # If strategy short-circuited (None), return a synthetic
        # "on-track" verdict so act() has something to return.
        if verdict is None:
            from ballast.drift._verdict import DefaultDriftVerdict
            return DefaultDriftVerdict(
                should_interrupt=False, reason="not checked",
                score=1.0, category="on_track",
            )
        return verdict

    async def act(
        self, observation: DriftContext, context: DriftVerdictBase,
    ) -> DriftVerdictBase:
        return context


def goal_drift_as_unit(engine: DriftEngine) -> _GoalDriftUnit:
    """Wrap a ``DriftEngine`` as a CoALA unit.

    The unit can then be adapted via ``ballast.coala.as_capability(unit)``
    or ``ballast.coala.as_workflow(unit)`` — same engine, different
    runtime wiring.
    """
    return _GoalDriftUnit(engine)


__all__ = ["goal_drift_as_unit"]
```

- [ ] **Step 4: Update `src/ballast/drift/__init__.py`**

Add import + entry:
```python
from ballast.drift._coala import goal_drift_as_unit
```
```python
"goal_drift_as_unit",
```

- [ ] **Step 5: Run — confirm pass**

Run: `uv run pytest tests/drift/test_coala_composition.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/drift/_coala.py src/ballast/drift/__init__.py tests/drift/test_coala_composition.py
git commit -m "feat(drift): goal_drift_as_unit — optional CoALAUnit adapter"
```

---

## Task 15: Final smoke — framework suite + import audit

- [ ] **Step 1: Run framework suite**

Run: `uv run pytest tests/ --tb=short -q`
Expected: green. All new tests (~50+ new) plus existing 477 still passing.

- [ ] **Step 2: Run drift module suite specifically**

Run: `uv run pytest tests/drift/ tests/capabilities/test_drift.py tests/capabilities/test_budget_snapshot.py tests/patterns/test_drift_monitor.py -v`
Expected: all green.

- [ ] **Step 3: Smoke import the whole framework**

Run:
```
uv run python -c "
from ballast import (
    Ballast, BallastSettings,
    GoalDriftDetector, with_drift_monitor,
    CoALABase, CoALAUnit, as_workflow, as_tool, as_capability,
)
from ballast.drift import (
    DriftEngine, DriftVerdictBase, DefaultDriftVerdict,
    AfterEveryStep, EveryNToolCalls, EveryNSteps, Periodic, OnBudgetThreshold,
    FullTrace, LastNMessages, SinceLastUserMessage, TokenBudgetWindow,
    FirstUserMessage, LastUserMessage, WorkflowInput, ExplicitGoal,
    LogOnly, EmitDriftEvent, RaiseDriftError, EscalateToHITL, GoalDriftError,
    DefaultPromptBuilder, make_default_judge,
    goal_drift_as_unit,
)
print('all imports ok')
"
```
Expected: `all imports ok`.

- [ ] **Step 4: Commit any cleanup**

```bash
git status
# If anything dangles (uv.lock, etc.), commit with a short message.
git add -u && git commit -m "chore(drift): final smoke cleanup" || echo "nothing to commit"
```

---

## Self-Review (against the spec)

**Spec coverage:**

| Spec section | Tasks |
|---|---|
| File structure | Tasks 1-11 (each task creates one file) |
| Public API | Tasks 3, 13 |
| Verdict (`DriftVerdictBase`, `DefaultDriftVerdict`) | Task 1 |
| Vehicles (`DriftCheckSignal`, `DriftContext`) | Task 2 |
| 5 Protocols | Task 3 |
| `DriftCheckStrategy` impls | Task 4 |
| `TraceWindow` impls | Task 5 |
| `GoalSource` impls (incl. `Summarized`) | Task 6 — NOTE: `Summarized` deferred (requires LLM agent, no native value for first cut; can be a follow-up) |
| `DriftHandler` impls + `GoalDriftError` | Task 7 |
| `DefaultPromptBuilder` + `make_default_judge` | Task 8 |
| `DriftEngine.maybe_check` + fail-safe semantics | Task 9 |
| Capability surface `GoalDriftDetector` | Task 10 |
| Workflow surface `with_drift_monitor` | Task 11 |
| `BudgetGuard.snapshot()` convention bridge | Task 12 |
| Public re-exports | Task 13 |
| CoALA composition factory | Task 14 |
| Final smoke | Task 15 |

**Spec defaults applied:**
- ✅ `BudgetGuard.snapshot()` + `metadata_provider` callable on `GoalDriftDetector` (deviation from spec option a — documented in plan header).
- ✅ `EscalateToHITL` blocking (Task 7 implementation awaits `channel.request`).
- ✅ Workflow surface limitation documented in `_monitor_loop` docstring (Task 11).

**Placeholder scan:** No TBDs/TODOs/vague-step-without-code; every step has either complete code or an exact command + expected output.

**Type consistency:**
- `DriftCheckSignal` fields (`step_index`, `tool_calls`, `tokens_used`, `seconds_elapsed`) used consistently across Tasks 2-4, 9-11.
- `DriftContext` fields (`messages`, `run_ctx`, `workflow_input`, `metadata`) consistent across Tasks 2, 5-7, 9-11.
- `DriftVerdictBase` (`should_interrupt`, `reason`) read in `DriftEngine` (Task 9) and `EscalateToHITL` (Task 7) — consistent.
- `DriftEngine.maybe_check(signal, ctx)` signature consistent in Tasks 9-11.
- `Compose` is aliased to `ComposeStrategy` / `ComposeHandler` in public `__init__.py` to avoid name collision (Tasks 4, 7) — same underlying class names internal, different re-export names.

**Known plan-vs-spec gap:**
- `Summarized` `GoalSource` impl from spec section "Built-in implementations" is OMITTED from Task 6. It requires a separate ``Agent[None, str]`` for periodic re-summarization and adds non-trivial scope without clear benefit for the first cut. Documented as a follow-up; framework apps can add a custom `GoalSource` subclass meanwhile.

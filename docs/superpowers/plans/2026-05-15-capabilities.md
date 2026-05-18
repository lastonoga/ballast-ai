# Capabilities (Sub-project #4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the L1 Capabilities layer of `pydantic-ai-stateflow`: production-grade middleware that wraps `pydantic_ai.Agent` runs. Six capabilities total in this sub-project — `BudgetGuard`, `SemanticLoopDetector`, `PIIGuard`, `GroundedRetry`, plus the shared helpers `SemanticDeduper`, `TypedLoopGuard`, `as_critique`. Two capabilities from the spec catalog (`GoalDriftDetector`, `LLMJudgeHook`) are deferred to later sub-projects because they depend on judge composition (SP5) and EvalStore (SP7) respectively.

**Architecture:** Each capability extends a thin `StateflowCapability` base over `pydantic_ai.capabilities.AbstractCapability`. Capabilities mount lifecycle hooks (`before_model_request`, `after_model_request`, `wrap_run`, `on_output_validate_error`) and use `ctx.state` for state that persists across the run (and survives DBOS replay when the run is inside a workflow). The composition of multiple capabilities follows pydantic-ai's wrap semantics: `before_*` fires top-to-bottom, `after_*` reverse, `wrap_*` nests outermost-first.

**Tech Stack:** `pydantic-ai` capabilities + hooks API (already a dependency from SP1), `pydantic_ai.Embedder` for semantic similarity helpers, the existing `Det` runtime from SP3.

**Spec sections covered:** 2B (full capability catalog and base class), 4F (MVP L1 scope), code-review notes from 2B.3 (split raw-response `SemanticLoopDetector` vs typed `TypedLoopGuard`).

**Scope vs deferred:**
- v1: `BudgetGuard`, `SemanticLoopDetector`, `PIIGuard`, `GroundedRetry`, `SemanticDeduper`, `TypedLoopGuard`, `as_critique` + an `Embedder` Protocol with a thin pydantic-ai default impl.
- Deferred to SP5: `GoalDriftDetector` (requires a judge-agent composition pattern that becomes natural with Patterns).
- Deferred to SP7: `LLMJudgeHook` (requires the EvalStore Repository protocol that the evals subsystem owns).

---

## File Structure

```
src/pydantic_ai_stateflow/
├── capabilities/
│   ├── __init__.py                # public exports
│   ├── base.py                    # StateflowCapability base class
│   ├── budget.py                  # BudgetGuard
│   ├── semantic_loop.py           # SemanticLoopDetector (raw response)
│   ├── pii.py                     # PIIGuard
│   ├── grounded_retry.py          # GroundedRetry
│   └── helpers/
│       ├── __init__.py
│       ├── embedder.py            # Embedder Protocol + DefaultEmbedder
│       ├── semantic_deduper.py    # SemanticDeduper utility
│       ├── typed_loop_guard.py    # TypedLoopGuard (Pattern-side)
│       └── as_critique.py         # Adapter for non-LLM critics
└── (existing modules unchanged)

tests/
└── capabilities/
    ├── __init__.py
    ├── test_base.py
    ├── test_budget_guard.py
    ├── test_embedder.py
    ├── test_semantic_deduper.py
    ├── test_semantic_loop_detector.py
    ├── test_typed_loop_guard.py
    ├── test_pii_guard.py
    ├── test_grounded_retry.py
    ├── test_as_critique.py
    └── test_public_api.py
```

---

## Task 1: `StateflowCapability` base class

**Files:**
- Create: `src/pydantic_ai_stateflow/capabilities/__init__.py`
- Create: `src/pydantic_ai_stateflow/capabilities/base.py`
- Create: `tests/capabilities/__init__.py`
- Create: `tests/capabilities/test_base.py`

- [ ] **Step 1: Failing test**

`tests/capabilities/test_base.py`:

```python
import pytest

from pydantic_ai_stateflow.capabilities import StateflowCapability


def test_stateflow_capability_has_name_classvar():
    """Each capability subclass declares its own .name (ClassVar) for tracing."""

    class FakeCap(StateflowCapability):
        name = "fake"

    assert FakeCap.name == "fake"


def test_stateflow_capability_is_abstract_capability():
    """StateflowCapability must inherit from pydantic_ai's AbstractCapability."""
    from pydantic_ai.capabilities import AbstractCapability

    assert issubclass(StateflowCapability, AbstractCapability)


def test_stateflow_capability_requires_name_attribute_in_subclass():
    """Subclasses should declare a name; this is enforced via ClassVar pattern.

    We don't strictly raise on missing name (Python class attrs are dynamic),
    but we do default to the class name for debuggability when name is omitted.
    """
    class NamelessCap(StateflowCapability):
        pass

    # Fallback: the class name itself when `name` not explicitly set
    assert NamelessCap.name == "NamelessCap"
```

- [ ] **Step 2: Run → fail (ImportError)**

```bash
uv run pytest tests/capabilities/test_base.py -v
```

- [ ] **Step 3: Implement**

`src/pydantic_ai_stateflow/capabilities/__init__.py`:

```python
from pydantic_ai_stateflow.capabilities.base import StateflowCapability

__all__ = ["StateflowCapability"]
```

`src/pydantic_ai_stateflow/capabilities/base.py`:

```python
from __future__ import annotations

from typing import Any, ClassVar

from pydantic_ai.capabilities import AbstractCapability


class StateflowCapability(AbstractCapability[Any]):
    """Base class for all framework capabilities.

    Provides:
    - `name: ClassVar[str]` — defaults to the subclass `__name__` if not set.
      Used in observability spans, ctx.state keys, and error messages.

    Future (Sub-project #6 / #7): adds a default `wrap_run` that opens a
    logfire span for the capability — placeholder for now.
    """

    name: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Default name to class __name__ if subclass didn't override.
        # Use a unique sentinel to detect missing override across the MRO.
        if "name" not in cls.__dict__ or cls.__dict__["name"] == "":
            cls.name = cls.__name__
```

- [ ] **Step 4: Tests pass (3 new)**

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/capabilities/__init__.py src/pydantic_ai_stateflow/capabilities/base.py tests/capabilities
git commit -m "feat(capabilities): StateflowCapability base class"
```

---

## Task 2: `BudgetGuard` (token + iteration limit)

Outermost capability. Tracks input/output tokens consumed across all `before/after_model_request` hooks. Raises `BudgetExhausted` on overflow.

**Files:**
- Create: `src/pydantic_ai_stateflow/capabilities/budget.py`
- Modify: `src/pydantic_ai_stateflow/capabilities/__init__.py`
- Create: `tests/capabilities/test_budget_guard.py`

- [ ] **Step 1: Failing tests**

`tests/capabilities/test_budget_guard.py`:

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, RequestUsage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_stateflow.capabilities import BudgetExhausted, BudgetGuard


def make_fn_model_returning(text: str, *, input_tokens: int = 10, output_tokens: int = 5) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # FunctionModel doesn't natively report usage; we use what's available.
        return ModelResponse(parts=[TextPart(content=text)])
    return FunctionModel(fn)


@pytest.mark.asyncio
async def test_budget_guard_allows_run_within_iteration_limit():
    agent = Agent(model=make_fn_model_returning("ok"), capabilities=[BudgetGuard(max_iterations=10)])
    result = await agent.run("hi")
    assert "ok" in str(result.output).lower() or result.output == "ok"


@pytest.mark.asyncio
async def test_budget_guard_raises_when_max_iterations_zero():
    """A zero iteration budget refuses the first model call."""
    agent = Agent(model=make_fn_model_returning("ok"), capabilities=[BudgetGuard(max_iterations=0)])
    with pytest.raises(BudgetExhausted):
        await agent.run("hi")


def test_budget_guard_defaults_are_unlimited_for_tokens():
    """Without max_input_tokens / max_output_tokens, only iteration matters."""
    guard = BudgetGuard(max_iterations=5)
    assert guard.max_input_tokens is None
    assert guard.max_output_tokens is None
    assert guard.max_iterations == 5
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement**

`src/pydantic_ai_stateflow/capabilities/budget.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import CapabilityOrdering
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext

from pydantic_ai_stateflow.capabilities.base import StateflowCapability


class BudgetExhausted(Exception):
    """Raised by BudgetGuard when the run exceeds the configured budget."""

    def __init__(self, reason: str, **details: Any) -> None:
        self.reason = reason
        self.details = details
        super().__init__(f"BudgetExhausted: {reason} ({details})")


@dataclass
class _BudgetState:
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class BudgetGuard(StateflowCapability):
    """Outermost capability: refuses runs that exceed iteration / token budget.

    State is kept under `ctx.state['budget_guard']` so it survives DBOS replay
    (because @DBOS.workflow ctx.state is part of the workflow event log).
    """

    name = "budget_guard"

    def __init__(
        self,
        *,
        max_iterations: int = 20,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost")

    def _state(self, ctx: RunContext[Any]) -> _BudgetState:
        if "budget_guard" not in ctx.state:
            ctx.state["budget_guard"] = _BudgetState()
        return ctx.state["budget_guard"]

    async def before_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext
    ) -> ModelRequestContext:
        state = self._state(ctx)
        if state.iterations >= self.max_iterations:
            raise BudgetExhausted(
                reason="max_iterations",
                at_step=state.iterations,
                limit=self.max_iterations,
            )
        state.iterations += 1
        return request_context

    async def after_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        usage = getattr(response, "usage", None)
        if usage is not None:
            state = self._state(ctx)
            state.input_tokens += getattr(usage, "input_tokens", 0) or 0
            state.output_tokens += getattr(usage, "output_tokens", 0) or 0
            if self.max_input_tokens is not None and state.input_tokens > self.max_input_tokens:
                raise BudgetExhausted(
                    reason="max_input_tokens",
                    consumed=state.input_tokens,
                    limit=self.max_input_tokens,
                )
            if self.max_output_tokens is not None and state.output_tokens > self.max_output_tokens:
                raise BudgetExhausted(
                    reason="max_output_tokens",
                    consumed=state.output_tokens,
                    limit=self.max_output_tokens,
                )
        return response
```

Update `__init__.py`:

```python
from pydantic_ai_stateflow.capabilities.base import StateflowCapability
from pydantic_ai_stateflow.capabilities.budget import BudgetExhausted, BudgetGuard

__all__ = ["BudgetExhausted", "BudgetGuard", "StateflowCapability"]
```

- [ ] **Step 4: Tests pass (3 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/capabilities/budget.py src/pydantic_ai_stateflow/capabilities/__init__.py tests/capabilities/test_budget_guard.py
git commit -m "feat(capabilities): BudgetGuard (iteration + token limits)"
```

---

## Task 3: `Embedder` Protocol + DefaultEmbedder

The framework needs embeddings for semantic similarity. We define a Protocol so users can swap implementations (OpenAI / local / cached). Default uses `pydantic_ai.Embedder`.

**Files:**
- Create: `src/pydantic_ai_stateflow/capabilities/helpers/__init__.py`
- Create: `src/pydantic_ai_stateflow/capabilities/helpers/embedder.py`
- Create: `tests/capabilities/test_embedder.py`

- [ ] **Step 1: Failing tests**

`tests/capabilities/test_embedder.py`:

```python
import pytest

from pydantic_ai_stateflow.capabilities.helpers import Embedder


class _FakeEmbedder:
    """Structural impl for testing Protocol satisfaction."""
    async def embed(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def test_fake_embedder_satisfies_protocol():
    assert isinstance(_FakeEmbedder(), Embedder)


@pytest.mark.asyncio
async def test_fake_embedder_embed_returns_vector():
    e = _FakeEmbedder()
    v = await e.embed("hello")
    assert v == [5.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_fake_embedder_embed_batch():
    e = _FakeEmbedder()
    vs = await e.embed_batch(["a", "bb"])
    assert vs == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement**

`src/pydantic_ai_stateflow/capabilities/helpers/__init__.py`:

```python
from pydantic_ai_stateflow.capabilities.helpers.embedder import Embedder

__all__ = ["Embedder"]
```

`src/pydantic_ai_stateflow/capabilities/helpers/embedder.py`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Structural type for an async text embedding service.

    Frameworks consume `Embedder` rather than `pydantic_ai.Embedder` directly
    so users can plug in their own (cached, local, batched, etc.) without
    depending on the pydantic-ai concrete class.
    """

    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

- [ ] **Step 4: Tests pass (3 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/capabilities/helpers tests/capabilities/test_embedder.py
git commit -m "feat(capabilities): Embedder Protocol"
```

---

## Task 4: `SemanticDeduper` helper

Sliding-window embedder + cosine similarity. Used by both `SemanticLoopDetector` (Task 5) and `TypedLoopGuard` (Task 6).

**Files:**
- Create: `src/pydantic_ai_stateflow/capabilities/helpers/semantic_deduper.py`
- Modify: `src/pydantic_ai_stateflow/capabilities/helpers/__init__.py`
- Create: `tests/capabilities/test_semantic_deduper.py`

- [ ] **Step 1: Failing tests**

`tests/capabilities/test_semantic_deduper.py`:

```python
import pytest

from pydantic_ai_stateflow.capabilities.helpers import (
    SemanticDeduper,
    SemanticLoopDetected,
)


class _IdentityEmbedder:
    """Returns a deterministic 'embedding' so we can craft exact-match scenarios."""
    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = mapping

    async def embed(self, text: str) -> list[float]:
        return self._mapping[text]

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


@pytest.mark.asyncio
async def test_deduper_does_not_fire_below_window():
    """First few snapshots fill the window; no detection yet."""
    e = _IdentityEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    d = SemanticDeduper(e)
    await d.add_and_check("a", threshold=0.95, window=3)
    await d.add_and_check("b", threshold=0.95, window=3)
    # No exception — fewer than `window` snapshots seen


@pytest.mark.asyncio
async def test_deduper_fires_when_window_filled_with_similar():
    """Three near-identical embeddings exceed cosine threshold."""
    e = _IdentityEmbedder({
        "x1": [1.0, 0.0],
        "x2": [1.0, 0.0],
        "x3": [1.0, 0.0],
    })
    d = SemanticDeduper(e)
    await d.add_and_check("x1", threshold=0.95, window=3)
    await d.add_and_check("x2", threshold=0.95, window=3)
    with pytest.raises(SemanticLoopDetected):
        await d.add_and_check("x3", threshold=0.95, window=3)


@pytest.mark.asyncio
async def test_deduper_does_not_fire_when_diverse():
    """Window of dissimilar embeddings is allowed."""
    e = _IdentityEmbedder({
        "a": [1.0, 0.0, 0.0],
        "b": [0.0, 1.0, 0.0],
        "c": [0.0, 0.0, 1.0],
        "d": [-1.0, 0.0, 0.0],
    })
    d = SemanticDeduper(e)
    for s in ["a", "b", "c", "d"]:
        await d.add_and_check(s, threshold=0.95, window=3)
    # Never raises


@pytest.mark.asyncio
async def test_deduper_sliding_window_drops_old_entries():
    """After window is full, oldest entry is dropped on insert."""
    e = _IdentityEmbedder({
        "old": [1.0, 0.0],
        "n1": [0.0, 1.0],
        "n2": [0.0, 1.0],
        "n3": [0.0, 1.0],
    })
    d = SemanticDeduper(e)
    await d.add_and_check("old", threshold=0.95, window=2)
    await d.add_and_check("n1", threshold=0.95, window=2)
    # n2 + n1 == similar; old should have been dropped
    with pytest.raises(SemanticLoopDetected):
        await d.add_and_check("n2", threshold=0.95, window=2)
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement**

Update `src/pydantic_ai_stateflow/capabilities/helpers/__init__.py`:

```python
from pydantic_ai_stateflow.capabilities.helpers.embedder import Embedder
from pydantic_ai_stateflow.capabilities.helpers.semantic_deduper import (
    SemanticDeduper,
    SemanticLoopDetected,
)

__all__ = ["Embedder", "SemanticDeduper", "SemanticLoopDetected"]
```

`src/pydantic_ai_stateflow/capabilities/helpers/semantic_deduper.py`:

```python
from __future__ import annotations

import math
from collections import deque

from pydantic_ai_stateflow.capabilities.helpers.embedder import Embedder


class SemanticLoopDetected(Exception):
    """Raised when a sliding window of snapshots is too similar (loop / repeat)."""

    def __init__(self, snapshot: str, similarities: list[float] | None = None) -> None:
        self.snapshot = snapshot
        self.similarities = similarities or []
        super().__init__(
            f"SemanticLoopDetected: snapshot={snapshot!r} similarities={self.similarities}"
        )


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors. Returns 0 if either is zero."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticDeduper:
    """Sliding-window cosine-similarity loop detector.

    Used by SemanticLoopDetector (L1, raw model response) and by
    TypedLoopGuard (L2, typed output between Pattern iterations).
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._history: deque[list[float]] = deque()

    async def add_and_check(self, snapshot: str, *, threshold: float, window: int) -> None:
        """Embed `snapshot`, slide window, raise SemanticLoopDetected on match."""
        emb = await self._embedder.embed(snapshot)
        # Slide window FIRST so we never compare beyond `window` entries
        while len(self._history) >= window:
            self._history.popleft()
        # Detection: if appending this embedding fills the window AND
        # all existing entries are sufficiently similar to this one, fire.
        if len(self._history) >= window - 1 and self._history:
            sims = [_cosine(emb, prev) for prev in self._history]
            if all(s >= threshold for s in sims):
                self._history.append(emb)
                raise SemanticLoopDetected(snapshot=snapshot[:200], similarities=sims)
        self._history.append(emb)
```

- [ ] **Step 4: Tests pass (4 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/capabilities/helpers/semantic_deduper.py src/pydantic_ai_stateflow/capabilities/helpers/__init__.py tests/capabilities/test_semantic_deduper.py
git commit -m "feat(capabilities): SemanticDeduper sliding-window cosine detector"
```

---

## Task 5: `SemanticLoopDetector` capability

L1 capability that monitors raw model responses (text + tool-call args) for repetition within a single `agent.run()`.

**Files:**
- Create: `src/pydantic_ai_stateflow/capabilities/semantic_loop.py`
- Modify: `src/pydantic_ai_stateflow/capabilities/__init__.py`
- Create: `tests/capabilities/test_semantic_loop_detector.py`

- [ ] **Step 1: Failing tests**

`tests/capabilities/test_semantic_loop_detector.py`:

```python
import json

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_stateflow.capabilities import SemanticLoopDetector
from pydantic_ai_stateflow.capabilities.helpers import SemanticLoopDetected


class _IdentityEmbedder:
    """Returns same vector for identical input."""
    async def embed(self, text: str) -> list[float]:
        # Deterministic: hash → 3D vector
        h = abs(hash(text)) % 1_000_000
        return [float(h % 100), float((h // 100) % 100), float((h // 10000) % 100)]

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


@pytest.mark.asyncio
async def test_default_selector_extracts_text_and_toolcalls():
    """Default selector serialises TextPart + ToolCallPart args stably."""
    from pydantic_ai_stateflow.capabilities.semantic_loop import _default_response_text

    resp = ModelResponse(parts=[
        TextPart(content="hello"),
        ToolCallPart(tool_name="do", args={"k": 1}),
    ])
    snap = _default_response_text(resp)
    assert "hello" in snap
    assert "do" in snap
    # args serialised as stable JSON
    assert json.dumps({"k": 1}, sort_keys=True) in snap


@pytest.mark.asyncio
async def test_loop_detector_allows_diverse_responses():
    """Successive different responses are fine."""
    counter = {"i": 0}
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        counter["i"] += 1
        return ModelResponse(parts=[TextPart(content=f"answer_{counter['i']}")])

    detector = SemanticLoopDetector(
        embedder=_IdentityEmbedder(), threshold=0.99, window=2,
    )
    agent = Agent(model=FunctionModel(fn), capabilities=[detector])
    # Run completes — single iteration, no loop possible inside one run.
    await agent.run("hi")
```

(Real-world loop scenarios appear across multiple iterations within one `agent.run()` when tools are involved. A simpler self-contained test of the deduper logic is covered in Task 4; here we just verify the capability *integrates cleanly* with `Agent.run()` and doesn't crash.)

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement**

`src/pydantic_ai_stateflow/capabilities/semantic_loop.py`:

```python
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestContext

from pydantic_ai_stateflow.capabilities.base import StateflowCapability
from pydantic_ai_stateflow.capabilities.helpers import Embedder, SemanticDeduper


def _default_response_text(response: ModelResponse) -> str:
    """Concatenate TextParts and serialise ToolCallPart args as stable JSON.

    Stable JSON keys ensure that identical tool calls — regardless of arg dict
    iteration order — produce identical snapshot strings.
    """
    bits: list[str] = []
    for p in response.parts:
        if isinstance(p, TextPart):
            bits.append(p.content)
        elif isinstance(p, ToolCallPart):
            args = p.args if isinstance(p.args, dict) else {"_raw": p.args}
            bits.append(f"{p.tool_name}({json.dumps(args, sort_keys=True, default=str)})")
    return "\n".join(bits)


class SemanticLoopDetector(StateflowCapability):
    """Detects repeated model responses within a single agent.run().

    Works at the model-response level (raw text + tool-call args). For
    detecting loops between Pattern iterations on TYPED output, see
    `TypedLoopGuard` (Task 6).
    """

    name = "semantic_loop_detector"

    def __init__(
        self,
        *,
        embedder: Embedder,
        threshold: float = 0.95,
        window: int = 3,
        selector: Callable[[ModelResponse], str] = _default_response_text,
    ) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.window = window
        self.selector = selector

    def _deduper(self, ctx: RunContext[Any]) -> SemanticDeduper:
        if "semantic_loop_detector" not in ctx.state:
            ctx.state["semantic_loop_detector"] = SemanticDeduper(self.embedder)
        return ctx.state["semantic_loop_detector"]

    async def after_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        snapshot = self.selector(response)
        await self._deduper(ctx).add_and_check(
            snapshot, threshold=self.threshold, window=self.window
        )
        return response
```

Update `__init__.py`:

```python
from pydantic_ai_stateflow.capabilities.base import StateflowCapability
from pydantic_ai_stateflow.capabilities.budget import BudgetExhausted, BudgetGuard
from pydantic_ai_stateflow.capabilities.semantic_loop import SemanticLoopDetector

__all__ = [
    "BudgetExhausted",
    "BudgetGuard",
    "SemanticLoopDetector",
    "StateflowCapability",
]
```

- [ ] **Step 4: Tests pass (2 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/capabilities/semantic_loop.py src/pydantic_ai_stateflow/capabilities/__init__.py tests/capabilities/test_semantic_loop_detector.py
git commit -m "feat(capabilities): SemanticLoopDetector (raw model-response level)"
```

---

## Task 6: `TypedLoopGuard` Pattern-side helper

Used by `Reflection` (SP5) and similar Patterns to detect typed-output loops across iterations.

**Files:**
- Create: `src/pydantic_ai_stateflow/capabilities/helpers/typed_loop_guard.py`
- Modify: `src/pydantic_ai_stateflow/capabilities/helpers/__init__.py`
- Create: `tests/capabilities/test_typed_loop_guard.py`

- [ ] **Step 1: Failing tests**

`tests/capabilities/test_typed_loop_guard.py`:

```python
import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.capabilities.helpers import (
    SemanticLoopDetected,
    TypedLoopGuard,
)


class _IdentityEmbedder:
    async def embed(self, text: str) -> list[float]:
        h = abs(hash(text)) % 1_000_000
        return [float(h % 100), float((h // 100) % 100)]

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


class Draft(BaseModel):
    rationale: str
    score: int


@pytest.mark.asyncio
async def test_guard_fires_when_same_field_repeats():
    guard = TypedLoopGuard[Draft](
        embedder=_IdentityEmbedder(),
        selector=lambda d: d.rationale,
        threshold=0.99,
        window=2,
    )
    await guard.check(Draft(rationale="same reason", score=1))
    with pytest.raises(SemanticLoopDetected):
        await guard.check(Draft(rationale="same reason", score=2))


@pytest.mark.asyncio
async def test_guard_allows_diverse_field_values():
    guard = TypedLoopGuard[Draft](
        embedder=_IdentityEmbedder(),
        selector=lambda d: d.rationale,
        threshold=0.99, window=2,
    )
    await guard.check(Draft(rationale="first", score=1))
    await guard.check(Draft(rationale="completely different", score=2))


@pytest.mark.asyncio
async def test_guard_supports_list_selector():
    """Selector may return a list of strings (multiple fields to check)."""
    guard = TypedLoopGuard[Draft](
        embedder=_IdentityEmbedder(),
        selector=lambda d: [d.rationale, str(d.score)],
        threshold=0.99, window=2,
    )
    await guard.check(Draft(rationale="A", score=1))
    # New selector returns two snapshots: rationale changed but score same.
    # Since both must be similar for the loop to fire (in this impl: any
    # single one matching across the window triggers), we choose a softer
    # contract: "if ANY of the selected snapshots is loop-detected, raise".
    # Two iterations is too short; need a third to truly test list semantics.
    await guard.check(Draft(rationale="B", score=1))
    with pytest.raises(SemanticLoopDetected):
        await guard.check(Draft(rationale="C", score=1))
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement**

Update `helpers/__init__.py`:

```python
from pydantic_ai_stateflow.capabilities.helpers.embedder import Embedder
from pydantic_ai_stateflow.capabilities.helpers.semantic_deduper import (
    SemanticDeduper,
    SemanticLoopDetected,
)
from pydantic_ai_stateflow.capabilities.helpers.typed_loop_guard import TypedLoopGuard

__all__ = [
    "Embedder",
    "SemanticDeduper",
    "SemanticLoopDetected",
    "TypedLoopGuard",
]
```

`src/pydantic_ai_stateflow/capabilities/helpers/typed_loop_guard.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from pydantic_ai_stateflow.capabilities.helpers.embedder import Embedder
from pydantic_ai_stateflow.capabilities.helpers.semantic_deduper import SemanticDeduper

OutT = TypeVar("OutT")


class TypedLoopGuard(Generic[OutT]):
    """Loop detector for typed Pattern outputs (SP5).

    Pattern code calls `.check(output)` between iterations. Each selector
    field is checked through its OWN deduper instance, so a list of fields
    detects loops independently per field. Loop in ANY field raises.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        selector: Callable[[OutT], str | list[str]],
        threshold: float = 0.95,
        window: int = 3,
    ) -> None:
        self.embedder = embedder
        self.selector = selector
        self.threshold = threshold
        self.window = window
        self._dedupers: dict[int, SemanticDeduper] = {}

    def _deduper_for(self, index: int) -> SemanticDeduper:
        if index not in self._dedupers:
            self._dedupers[index] = SemanticDeduper(self.embedder)
        return self._dedupers[index]

    async def check(self, output: OutT) -> None:
        snapshots = self.selector(output)
        if isinstance(snapshots, str):
            snapshots = [snapshots]
        for i, snap in enumerate(snapshots):
            await self._deduper_for(i).add_and_check(
                snap, threshold=self.threshold, window=self.window
            )
```

- [ ] **Step 4: Tests pass (3 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/capabilities/helpers/typed_loop_guard.py src/pydantic_ai_stateflow/capabilities/helpers/__init__.py tests/capabilities/test_typed_loop_guard.py
git commit -m "feat(capabilities): TypedLoopGuard (Pattern-side typed field loop check)"
```

---

## Task 7: `PIIGuard` capability

Innermost capability: redacts PII patterns from `TextPart` responses before any downstream processing.

**Files:**
- Create: `src/pydantic_ai_stateflow/capabilities/pii.py`
- Modify: `src/pydantic_ai_stateflow/capabilities/__init__.py`
- Create: `tests/capabilities/test_pii_guard.py`

- [ ] **Step 1: Failing tests**

`tests/capabilities/test_pii_guard.py`:

```python
import re

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_stateflow.capabilities import PIIGuard


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d{10,15}")


def make_fn_model_returning(text: str) -> FunctionModel:
    def fn(messages, info: AgentInfo):
        return ModelResponse(parts=[TextPart(content=text)])
    return FunctionModel(fn)


@pytest.mark.asyncio
async def test_pii_guard_redacts_email():
    agent = Agent(
        model=make_fn_model_returning("Contact me at alice@example.com soon."),
        capabilities=[PIIGuard(patterns=[EMAIL_RE])],
    )
    result = await agent.run("ignored")
    text = str(result.output) if result.output else ""
    assert "alice@example.com" not in text
    assert "[REDACTED]" in text


@pytest.mark.asyncio
async def test_pii_guard_redacts_phone_with_custom_replacement():
    agent = Agent(
        model=make_fn_model_returning("Call +1234567890 now."),
        capabilities=[PIIGuard(patterns=[PHONE_RE], replacement="[PHONE]")],
    )
    result = await agent.run("ignored")
    text = str(result.output) if result.output else ""
    assert "+1234567890" not in text
    assert "[PHONE]" in text


@pytest.mark.asyncio
async def test_pii_guard_passes_through_clean_text():
    agent = Agent(
        model=make_fn_model_returning("Nothing to see here."),
        capabilities=[PIIGuard(patterns=[EMAIL_RE])],
    )
    result = await agent.run("ignored")
    assert "Nothing to see here" in str(result.output)
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement**

`src/pydantic_ai_stateflow/capabilities/pii.py`:

```python
from __future__ import annotations

import re
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import CapabilityOrdering
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext

from pydantic_ai_stateflow.capabilities.base import StateflowCapability


class PIIGuard(StateflowCapability):
    """Innermost capability: regex-redacts PII from text responses.

    Applied AFTER all other after_model_request hooks (innermost in the
    wrap chain), so other capabilities see the raw text and downstream
    output validation / persistence sees the redacted form.

    For richer detection (NER), users can subclass and override
    `redact(text)` — the regex layer is just a sensible default.
    """

    name = "pii_guard"

    def __init__(
        self,
        *,
        patterns: list[re.Pattern[str]],
        replacement: str = "[REDACTED]",
    ) -> None:
        self.patterns = patterns
        self.replacement = replacement

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    def redact(self, text: str) -> str:
        for pat in self.patterns:
            text = pat.sub(self.replacement, text)
        return text

    async def after_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        for part in response.parts:
            if isinstance(part, TextPart):
                part.content = self.redact(part.content)
        return response
```

Update `__init__.py`:

```python
from pydantic_ai_stateflow.capabilities.base import StateflowCapability
from pydantic_ai_stateflow.capabilities.budget import BudgetExhausted, BudgetGuard
from pydantic_ai_stateflow.capabilities.pii import PIIGuard
from pydantic_ai_stateflow.capabilities.semantic_loop import SemanticLoopDetector

__all__ = [
    "BudgetExhausted",
    "BudgetGuard",
    "PIIGuard",
    "SemanticLoopDetector",
    "StateflowCapability",
]
```

- [ ] **Step 4: Tests pass (3 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/capabilities/pii.py src/pydantic_ai_stateflow/capabilities/__init__.py tests/capabilities/test_pii_guard.py
git commit -m "feat(capabilities): PIIGuard (regex-based redaction)"
```

---

## Task 8: `GroundedRetry` capability

Transforms `ValidationError` on output into a structured `ModelRetry` with field-specific feedback.

**Files:**
- Create: `src/pydantic_ai_stateflow/capabilities/grounded_retry.py`
- Modify: `src/pydantic_ai_stateflow/capabilities/__init__.py`
- Create: `tests/capabilities/test_grounded_retry.py`

- [ ] **Step 1: Failing tests**

`tests/capabilities/test_grounded_retry.py`:

```python
import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_stateflow.capabilities.grounded_retry import (
    GroundedRetry,
    _build_feedback,
)


class _Out(BaseModel):
    choice: str
    score: int


def test_build_feedback_for_missing_field():
    try:
        _Out.model_validate({"choice": "a"})
    except ValidationError as err:
        feedback = _build_feedback(err, raw_output={"choice": "a"})
    assert "score" in feedback
    assert "missing" in feedback.lower()


def test_build_feedback_for_literal_violation():
    """Literal-type errors should mention the allowed values."""
    from typing import Literal

    class L(BaseModel):
        status: Literal["a", "b", "c"]

    try:
        L.model_validate({"status": "z"})
    except ValidationError as err:
        feedback = _build_feedback(err, raw_output={"status": "z"})
    # Pydantic literal_error includes ctx.expected
    assert "status" in feedback
    # Field name + actual rejected value should appear
    assert "z" in feedback or "'z'" in feedback


def test_grounded_retry_has_max_retries_default():
    cap = GroundedRetry()
    assert cap.max_retries == 3


def test_grounded_retry_accepts_custom_max_retries():
    cap = GroundedRetry(max_retries=5)
    assert cap.max_retries == 5
```

(Integration test for the actual hook fire is hard without a model that controllably produces invalid output; the unit tests above pin the feedback-building logic which is the meaty part. Integration coverage will arrive when SP5 patterns use this in a Reflection loop.)

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement**

`src/pydantic_ai_stateflow/capabilities/grounded_retry.py`:

```python
from __future__ import annotations

from typing import Any

from pydantic import ValidationError
from pydantic_ai import ModelRetry, RunContext

from pydantic_ai_stateflow.capabilities.base import StateflowCapability


def _build_feedback(error: ValidationError, raw_output: Any) -> str:
    """Translate a Pydantic ValidationError into a model-friendly retry hint.

    Special-cases:
    - `literal_error`: list allowed values + actual bad value
    - `missing`: name the required field
    - default: pass through Pydantic's message with the field path
    """
    msgs: list[str] = []
    for err in error.errors():
        loc = ".".join(str(part) for part in err["loc"])
        etype = err["type"]
        if etype == "literal_error":
            allowed = err.get("ctx", {}).get("expected", "")
            actual = err.get("input")
            msgs.append(
                f"Field '{loc}' must be one of: {allowed}. You returned: {actual!r}."
            )
        elif etype == "missing":
            msgs.append(f"Required field '{loc}' is missing.")
        else:
            msgs.append(f"{loc}: {err['msg']}")
    return "Output validation failed:\n" + "\n".join(f"- {m}" for m in msgs)


class GroundedRetry(StateflowCapability):
    """Converts Pydantic validation errors on output into structured ModelRetry.

    Gives the model precise feedback (which field, what was expected, what
    it returned) instead of a generic "JSON invalid" message. From the
    spec: this lifts F1 from ~0.84 to ~0.96 in structured-output tasks.

    Each retry attempt is counted in ctx.state; once max_retries is hit the
    original ValidationError bubbles out.
    """

    name = "grounded_retry"

    def __init__(self, *, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    def _attempts(self, ctx: RunContext[Any]) -> int:
        return ctx.state.get("grounded_retry.attempts", 0)

    def _bump(self, ctx: RunContext[Any]) -> None:
        ctx.state["grounded_retry.attempts"] = self._attempts(ctx) + 1

    async def on_output_validate_error(
        self, ctx: RunContext[Any], *, raw_output: Any, error: ValidationError
    ) -> None:
        if self._attempts(ctx) >= self.max_retries:
            raise error
        self._bump(ctx)
        raise ModelRetry(_build_feedback(error, raw_output))
```

Update `__init__.py`:

```python
from pydantic_ai_stateflow.capabilities.base import StateflowCapability
from pydantic_ai_stateflow.capabilities.budget import BudgetExhausted, BudgetGuard
from pydantic_ai_stateflow.capabilities.grounded_retry import GroundedRetry
from pydantic_ai_stateflow.capabilities.pii import PIIGuard
from pydantic_ai_stateflow.capabilities.semantic_loop import SemanticLoopDetector

__all__ = [
    "BudgetExhausted",
    "BudgetGuard",
    "GroundedRetry",
    "PIIGuard",
    "SemanticLoopDetector",
    "StateflowCapability",
]
```

- [ ] **Step 4: Tests pass (4 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/capabilities/grounded_retry.py src/pydantic_ai_stateflow/capabilities/__init__.py tests/capabilities/test_grounded_retry.py
git commit -m "feat(capabilities): GroundedRetry (structured ValidationError feedback)"
```

---

## Task 9: `as_critique` adapter (non-LLM critics)

Lets Pattern code (Reflection in SP5) accept either an `Agent[..., Critique]` or a plain Python `async def critic(input) -> X` for the critic.

**Files:**
- Create: `src/pydantic_ai_stateflow/capabilities/helpers/as_critique.py`
- Modify: `src/pydantic_ai_stateflow/capabilities/helpers/__init__.py`
- Create: `tests/capabilities/test_as_critique.py`

- [ ] **Step 1: Failing tests**

`tests/capabilities/test_as_critique.py`:

```python
import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.capabilities.helpers import Critique, as_critique


class CustomVerdict(BaseModel):
    passed: bool
    issues: list[str] = []


@pytest.mark.asyncio
async def test_as_critique_wraps_async_function():
    async def fn(payload):
        return Critique(passed=True, confidence=1.0)

    agent = as_critique(fn)
    result = await agent.run("anything")
    assert result.output.passed is True


@pytest.mark.asyncio
async def test_as_critique_wraps_object_with_check_method():
    class C:
        async def check(self, payload):
            return Critique(passed=False, issues=["bad"])

    agent = as_critique(C())
    result = await agent.run("anything")
    assert result.output.passed is False
    assert result.output.issues == ["bad"]


@pytest.mark.asyncio
async def test_as_critique_coerces_custom_pass_object():
    """A return object with .passed coerces into Critique."""
    async def fn(payload):
        return CustomVerdict(passed=True, issues=["minor"])

    agent = as_critique(fn)
    result = await agent.run("anything")
    assert result.output.passed is True
    assert result.output.issues == ["minor"]


@pytest.mark.asyncio
async def test_as_critique_coerces_bool():
    async def fn(payload):
        return True

    agent = as_critique(fn)
    result = await agent.run("anything")
    assert result.output.passed is True
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement**

Add `Critique` to helpers `__init__.py` exports + new module.

Update `helpers/__init__.py`:

```python
from pydantic_ai_stateflow.capabilities.helpers.as_critique import Critique, as_critique
from pydantic_ai_stateflow.capabilities.helpers.embedder import Embedder
from pydantic_ai_stateflow.capabilities.helpers.semantic_deduper import (
    SemanticDeduper,
    SemanticLoopDetected,
)
from pydantic_ai_stateflow.capabilities.helpers.typed_loop_guard import TypedLoopGuard

__all__ = [
    "Critique",
    "Embedder",
    "SemanticDeduper",
    "SemanticLoopDetected",
    "TypedLoopGuard",
    "as_critique",
]
```

`src/pydantic_ai_stateflow/capabilities/helpers/as_critique.py`:

```python
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


class Critique(BaseModel):
    """Canonical critique shape used by Reflection (SP5)."""
    passed: bool
    issues: list[str] = []
    suggestions: list[str] = []
    confidence: float = 1.0


def _coerce_to_critique(value: Any) -> Critique:
    if isinstance(value, Critique):
        return value
    if isinstance(value, bool):
        return Critique(passed=value, confidence=1.0)
    # Duck-type: object with .passed (and possibly .issues / .suggestions)
    passed = getattr(value, "passed", None)
    if isinstance(passed, bool):
        return Critique(
            passed=passed,
            issues=list(getattr(value, "issues", []) or []),
            suggestions=list(getattr(value, "suggestions", []) or []),
            confidence=float(getattr(value, "confidence", 1.0)),
        )
    raise TypeError(f"Cannot coerce {type(value).__name__} to Critique")


def as_critique(fn: Callable[[Any], Awaitable[Any]] | Any) -> Agent[Any, Critique]:
    """Wrap a non-LLM critic (callable or object with .check) as a pydantic-ai Agent.

    Lets Reflection (SP5) accept any critic — LLM agent, plain Python
    function, or stateful object with a `check()` method — through a
    single uniform interface (`Agent.run(...)`).

    Internally uses FunctionModel so no real LLM is invoked.
    """
    callable_fn: Callable[[Any], Awaitable[Any]]
    if hasattr(fn, "check") and callable(fn.check):
        callable_fn = fn.check  # type: ignore[assignment]
    else:
        callable_fn = fn  # type: ignore[assignment]

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # The "input" we hand to the critic is the last user message body.
        # Real Patterns will marshal a structured payload in via the Agent;
        # for arbitrary inputs, the user prompt text is the simplest contract.
        payload = messages[-1].parts[0].content if messages else None
        verdict = await callable_fn(payload)
        critique = _coerce_to_critique(verdict)
        return ModelResponse(parts=[ToolCallPart(
            tool_name="final_result",
            args=critique.model_dump(),
        )])

    return Agent(model=FunctionModel(model_fn), output_type=Critique)
```

- [ ] **Step 4: Tests pass (4 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/capabilities/helpers/as_critique.py src/pydantic_ai_stateflow/capabilities/helpers/__init__.py tests/capabilities/test_as_critique.py
git commit -m "feat(capabilities): as_critique adapter (non-LLM critics as Agent)"
```

---

## Task 10: Public API + integration smoke

**Files:**
- Modify: `src/pydantic_ai_stateflow/__init__.py`
- Create: `tests/capabilities/test_public_api.py`

- [ ] **Step 1: Update top-level exports**

In `src/pydantic_ai_stateflow/__init__.py`, ADD:

```python
from pydantic_ai_stateflow.capabilities import (
    BudgetExhausted,
    BudgetGuard,
    GroundedRetry,
    PIIGuard,
    SemanticLoopDetector,
    StateflowCapability,
)
from pydantic_ai_stateflow.capabilities.helpers import (
    Critique,
    Embedder,
    SemanticDeduper,
    SemanticLoopDetected,
    TypedLoopGuard,
    as_critique,
)
```

And to `__all__` (preserve existing):

```python
__all__ = [
    # ... existing entries ...
    "BudgetExhausted",
    "BudgetGuard",
    "Critique",
    "Embedder",
    "GroundedRetry",
    "PIIGuard",
    "SemanticDeduper",
    "SemanticLoopDetected",
    "SemanticLoopDetector",
    "StateflowCapability",
    "TypedLoopGuard",
    "as_critique",
]
```

- [ ] **Step 2: Integration smoke**

`tests/capabilities/test_public_api.py`:

```python
def test_capabilities_visible_at_top_level():
    from pydantic_ai_stateflow import (
        BudgetExhausted,
        BudgetGuard,
        Critique,
        Embedder,
        GroundedRetry,
        PIIGuard,
        SemanticDeduper,
        SemanticLoopDetected,
        SemanticLoopDetector,
        StateflowCapability,
        TypedLoopGuard,
        as_critique,
    )

    assert BudgetGuard is not None
    assert SemanticLoopDetector is not None
    assert PIIGuard is not None
    assert GroundedRetry is not None
    assert callable(as_critique)


import re

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_stateflow import BudgetGuard, PIIGuard


@pytest.mark.asyncio
async def test_two_capabilities_compose_in_one_agent():
    """Stacking BudgetGuard (outermost) + PIIGuard (innermost) — both fire."""
    def fn(messages, info: AgentInfo):
        return ModelResponse(parts=[TextPart(content="contact alice@example.com")])

    agent = Agent(
        model=FunctionModel(fn),
        capabilities=[
            BudgetGuard(max_iterations=5),
            PIIGuard(patterns=[re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")]),
        ],
    )
    result = await agent.run("hi")
    text = str(result.output) if result.output else ""
    assert "alice@example.com" not in text
    assert "[REDACTED]" in text
```

- [ ] **Step 3: Tests pass (2 new)**
- [ ] **Step 4: Full suite + mypy + ruff**
- [ ] **Step 5: Commit**

```bash
git add src/pydantic_ai_stateflow/__init__.py tests/capabilities/test_public_api.py
git commit -m "feat: Sub-project #4 public API (capabilities + helpers)"
```

---

## Sub-project #4 acceptance criteria

After all 10 tasks:

- ✅ `from pydantic_ai_stateflow import BudgetGuard, SemanticLoopDetector, PIIGuard, GroundedRetry, as_critique, TypedLoopGuard, Embedder, Critique, ...` works
- ✅ `BudgetGuard(max_iterations=N, max_input_tokens=N, max_output_tokens=N)` enforces during `agent.run()` via `before_model_request` / `after_model_request` hooks
- ✅ `PIIGuard(patterns=[regex])` redacts text in `TextPart`s via `after_model_request` (innermost)
- ✅ `SemanticLoopDetector(embedder=..., threshold=..., window=...)` detects repeated raw responses via `SemanticDeduper`
- ✅ `TypedLoopGuard(selector=lambda d: d.field)` gives Pattern code (SP5) per-field loop detection
- ✅ `GroundedRetry()` translates `ValidationError` to structured `ModelRetry` with field-specific hints (lifts F1 on schema-constrained outputs)
- ✅ `as_critique(callable_or_object)` wraps non-LLM critics into Agent shape for SP5 Reflection
- ✅ Two capabilities can be stacked (BudgetGuard outermost + PIIGuard innermost) — composition smoke test passes
- ✅ All Sub-project #1 + #2 + #3 tests still pass
- ✅ mypy strict + ruff clean

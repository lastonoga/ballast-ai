# DivergentConvergent: own envelope→hypotheses mapping — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `diverge` / `synthesize` adapter methods from app-side agents — let `DivergentConvergent` own the envelope→hypotheses mapping via two app-provided callables, leaving agents as pure `StateflowAgent` instances.

**Architecture:** Framework `DivergentConvergent` accepts agents whose `.run(...)` returns a `result.output` envelope; the pattern instance carries one `hypotheses: Callable[[EnvT], list[HypT]]` projector for diverge phase and one `format_synth_prompt: Callable[[InT, list[HypT]], str]` for the synthesis step. App-side agents go back to being neutral `StateflowAgent`s with no pattern-specific methods.

**Tech Stack:** Python 3.11+, pydantic v2, pydantic-ai (via `StateflowAgent`), DBOS (`Durable.workflow` / `Durable.step`), pytest.

---

## File Map

- **Modify** `src/pydantic_ai_stateflow/patterns/divergent_convergent/primitives.py` — replace `DivergentAgent` / `Synthesizer` Protocols with structural shapes mirroring pydantic-ai's `Agent.run`.
- **Modify** `src/pydantic_ai_stateflow/patterns/divergent_convergent/pattern.py` — add `EnvT` TypeVar, two new constructor params, rewrite `_diverge_one` and `_converge` bodies.
- **Create** `tests/patterns/test_divergent_convergent.py` — minimal new TDD test exercising the new API with mock agents.
- **Modify** `examples/notes-app/backend/src/notes_app/brainstorm_agents.py` — delete `diverge()`, `synthesize()`, and `_format_synth_prompt`.
- **Modify** `examples/notes-app/backend/src/notes_app/brainstorm_flow.py` — add module-level `_format_synth_prompt`, pass `hypotheses=` + `format_synth_prompt=` to `DivergentConvergent` in the factory.

---

## Task 1: Failing unit test for new pattern API

**Files:**
- Create: `tests/patterns/test_divergent_convergent.py`

- [ ] **Step 1: Create the failing test**

Write `tests/patterns/test_divergent_convergent.py`:

```python
"""Unit tests for ``DivergentConvergent`` covering the
agent + projector contract (no ``diverge``/``synthesize`` methods on
agents — the pattern owns envelope→hypotheses mapping).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.patterns.divergent_convergent import (
    DivergentBranch,
    DivergentConvergent,
)


class _Idea(BaseModel):
    title: str


class _Ideas(BaseModel):
    """Envelope returned by mock divergent agents — mirrors the
    ``TodoIdeas { ideas: list[TodoIdea] }`` shape used by the notes-app."""
    ideas: list[_Idea]


@dataclass
class _AgentResult:
    """Structural stand-in for pydantic-ai's ``AgentRunResult``."""
    output: object


class _MockDivergentAgent:
    """Returns a fixed envelope per call. The pattern projects via
    the ``hypotheses`` callable supplied to ``DivergentConvergent``."""

    def __init__(self, ideas: list[_Idea]) -> None:
        self._ideas = ideas
        self.calls = 0

    async def run(self, task: str) -> _AgentResult:
        del task
        self.calls += 1
        return _AgentResult(output=_Ideas(ideas=list(self._ideas)))


class _MockSynthesizer:
    """Returns the first idea unchanged. Receives a string prompt
    (built by the pattern via ``format_synth_prompt``)."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    async def run(self, prompt: str) -> _AgentResult:
        self.last_prompt = prompt
        # Pretend we picked the first candidate from the rendered prompt.
        return _AgentResult(output=_Idea(title="winner"))


@pytest.mark.asyncio
async def test_pattern_invokes_hypotheses_projector_per_branch() -> None:
    """``DivergentConvergent`` should call ``hypotheses(env)`` exactly
    once per successful branch and feed the merged pool into the
    synthesizer via ``format_synth_prompt``."""
    agent_a = _MockDivergentAgent([_Idea(title="a1"), _Idea(title="a2")])
    agent_b = _MockDivergentAgent([_Idea(title="b1")])
    synth = _MockSynthesizer()

    projector_calls: list[_Ideas] = []

    def hypotheses(env: _Ideas) -> list[_Idea]:
        projector_calls.append(env)
        return env.ideas

    prompt_calls: list[tuple[str, list[_Idea]]] = []

    def format_synth_prompt(task: str, candidates: list[_Idea]) -> str:
        prompt_calls.append((task, list(candidates)))
        return f"task={task};n={len(candidates)}"

    dc = DivergentConvergent[str, _Ideas, _Idea, _Idea](
        branches=(
            DivergentBranch(label="a", agent=agent_a),
            DivergentBranch(label="b", agent=agent_b),
        ),
        synthesizer=synth,
        hypotheses=hypotheses,
        format_synth_prompt=format_synth_prompt,
        min_hypotheses=2,
        divergent_concurrency=2,
        config_name=f"test-dc-{uuid4()}",
    )

    result = await dc.run("topic")

    assert result == _Idea(title="winner")
    assert agent_a.calls == 1
    assert agent_b.calls == 1
    assert len(projector_calls) == 2
    assert len(prompt_calls) == 1
    task, candidates = prompt_calls[0]
    assert task == "topic"
    # Merged pool: agent_a's two ideas + agent_b's one idea, in order.
    assert [c.title for c in candidates] == ["a1", "a2", "b1"]
    assert synth.last_prompt == "task=topic;n=3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/patterns/test_divergent_convergent.py -v`

Expected: FAIL with `TypeError: DivergentConvergent.__init__() got an unexpected keyword argument 'hypotheses'` (or similar — the constructor signature does not yet accept the new kwargs).

- [ ] **Step 3: Commit**

```bash
git add tests/patterns/test_divergent_convergent.py
git commit -m "test(patterns): failing TDD test for DivergentConvergent projector API"
```

---

## Task 2: Update `primitives.py` Protocols

**Files:**
- Modify: `src/pydantic_ai_stateflow/patterns/divergent_convergent/primitives.py`

- [ ] **Step 1: Replace `DivergentAgent` and `Synthesizer` Protocols**

Overwrite the entire content of `src/pydantic_ai_stateflow/patterns/divergent_convergent/primitives.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

InT_contra = TypeVar("InT_contra", contravariant=True)
EnvT_co = TypeVar("EnvT_co", covariant=True)
OutT_co = TypeVar("OutT_co", covariant=True)
HypothesisT_contra = TypeVar("HypothesisT_contra", contravariant=True)


@runtime_checkable
class _AgentRunResult(Protocol[OutT_co]):
    """Anything with a typed ``.output`` property.

    Structural mirror of pydantic-ai's ``AgentRunResult`` — declared
    here so the framework doesn't import pydantic-ai at module level.
    """

    @property
    def output(self) -> OutT_co: ...


@runtime_checkable
class DivergentAgent(Protocol[InT_contra, EnvT_co]):
    """One ``branch`` in the divergent phase.

    Returns an *envelope* (``EnvT``) per ``.run(task)`` call. The
    pattern then applies the app-supplied ``hypotheses`` projector to
    extract the ``list[Hypothesis]`` it reduces over.

    The signature deliberately matches pydantic-ai's ``Agent.run`` so
    pydantic-ai agents satisfy it natively — but the framework doesn't
    import pydantic-ai, so apps can substitute mocks or pure-Python
    heuristics.
    """

    async def run(self, task: InT_contra) -> _AgentRunResult[EnvT_co]: ...


@runtime_checkable
class Synthesizer(Protocol[OutT_co]):
    """Convergent reducer over the surviving pool.

    Receives a string prompt rendered by the pattern (which used the
    app-supplied ``format_synth_prompt`` to project ``(task, candidates)``
    into text). ``.output`` IS the final ``OutT`` — no projector
    needed for synthesis since the synthesizer's output type is the
    same as the pattern's result type.
    """

    async def run(self, prompt: str) -> _AgentRunResult[OutT_co]: ...


@runtime_checkable
class Verifier(Protocol[HypothesisT_contra]):
    """Optional scorer applied between dedup and synthesis.

    Returns a float per hypothesis (higher = better). The pattern then
    sorts descending and (if ``top_k`` is set) slices to the top-K
    before handing them to the synthesizer. Common backings:

    * a small reward model / classifier,
    * an LLM judge with a structured rubric,
    * a heuristic over hypothesis fields.

    Per BoN-MAV ("Best-of-N Multi-Agent Verification", 2025), keeping
    the verifier SEPARATE from the synthesizer beats letting one model
    do both — the synthesizer biases toward the first plausible
    candidate it sees, while a dedicated verifier scores them on
    explicit criteria.
    """

    async def score(
        self,
        *,
        task: Any,
        hypothesis: HypothesisT_contra,
    ) -> float: ...


HypothesisT = TypeVar("HypothesisT")
InT = TypeVar("InT")
EnvT = TypeVar("EnvT")


@dataclass(frozen=True)
class DivergentBranch(Generic[InT, EnvT]):
    """One labelled branch in the divergent fan-out.

    ``label`` lands in traces / queue task names so you can tell whose
    pool dominated after convergence. ``agent`` is anything satisfying
    ``DivergentAgent[InT, EnvT]`` — typically a ``StateflowAgent`` /
    ``pydantic_ai.Agent`` whose output type is ``EnvT``.
    """

    label: str
    agent: DivergentAgent[InT, EnvT]
```

- [ ] **Step 2: Commit**

```bash
git add src/pydantic_ai_stateflow/patterns/divergent_convergent/primitives.py
git commit -m "refactor(patterns): structural agent Protocols for DivergentConvergent"
```

---

## Task 3: Update `pattern.py` — add `EnvT`, projector params, rewrite step bodies

**Files:**
- Modify: `src/pydantic_ai_stateflow/patterns/divergent_convergent/pattern.py`

- [ ] **Step 1: Update imports and TypeVars**

In `src/pydantic_ai_stateflow/patterns/divergent_convergent/pattern.py`, locate the imports block and replace the line:

```python
import itertools
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Literal, TypeVar
```

with (note the added `Callable` import):

```python
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Literal, TypeVar
```

Then locate:

```python
InT = TypeVar("InT")
HypothesisT = TypeVar("HypothesisT")
OutT = TypeVar("OutT")
```

and replace with:

```python
InT = TypeVar("InT")
EnvT = TypeVar("EnvT")
HypothesisT = TypeVar("HypothesisT")
OutT = TypeVar("OutT")
```

- [ ] **Step 2: Update class Generic parameters**

Locate the class declaration:

```python
@Durable.dbos_class()
class DivergentConvergent(
    DBOSConfiguredInstance, Generic[InT, HypothesisT, OutT],
):
```

Replace with:

```python
@Durable.dbos_class()
class DivergentConvergent(
    DBOSConfiguredInstance, Generic[InT, EnvT, HypothesisT, OutT],
):
```

- [ ] **Step 3: Add `hypotheses` and `format_synth_prompt` constructor parameters**

Locate the `__init__` signature:

```python
def __init__(
    self,
    branches: tuple[DivergentBranch[InT, HypothesisT], ...],
    synthesizer: Synthesizer[InT, HypothesisT, OutT],
    *,
    deduper: Deduper | None = None,
    verifier: Verifier[HypothesisT] | None = None,
    top_k: int | None = None,
    best_of_n: int = 1,
    min_hypotheses: int = 2,
    per_branch_failure: Literal["strict", "skip"] = "skip",
    divergent_concurrency: int = 4,
    config_name: str | None = None,
) -> None:
```

Replace with:

```python
def __init__(
    self,
    branches: tuple[DivergentBranch[InT, EnvT], ...],
    synthesizer: Synthesizer[OutT],
    *,
    hypotheses: Callable[[EnvT], list[HypothesisT]],
    format_synth_prompt: Callable[[InT, list[HypothesisT]], str],
    deduper: Deduper | None = None,
    verifier: Verifier[HypothesisT] | None = None,
    top_k: int | None = None,
    best_of_n: int = 1,
    min_hypotheses: int = 2,
    per_branch_failure: Literal["strict", "skip"] = "skip",
    divergent_concurrency: int = 4,
    config_name: str | None = None,
) -> None:
```

- [ ] **Step 4: Store the new callables on the instance**

In the same `__init__`, locate:

```python
self._synthesizer = synthesizer
self._deduper = deduper
```

and replace with:

```python
self._synthesizer = synthesizer
self._hypotheses = hypotheses
self._format_synth_prompt = format_synth_prompt
self._deduper = deduper
```

- [ ] **Step 5: Update `_diverge_one` step body**

Locate:

```python
@Durable.step()
async def _diverge_one(
    self, label: str, sample_idx: int, task: InT,
) -> list[HypothesisT]:
    # ``sample_idx`` is unused in the body — it's there so each
    # best-of-N sample becomes a DISTINCT step invocation (DBOS
    # caches step results by name + args; without it, K samples
    # of the same branch would share the cached first result on
    # workflow replay).
    #
    # OTel context propagation is handled by ``@Durable.step`` —
    # the carrier travels in a magic kwarg from ``Durable.enqueue``
    # and is attached to this fiber before the body runs.
    del sample_idx
    branch = self._branches[label]
    return await branch.agent.diverge(task)
```

Replace with:

```python
@Durable.step()
async def _diverge_one(
    self, label: str, sample_idx: int, task: InT,
) -> list[HypothesisT]:
    # ``sample_idx`` is unused in the body — it's there so each
    # best-of-N sample becomes a DISTINCT step invocation (DBOS
    # caches step results by name + args; without it, K samples
    # of the same branch would share the cached first result on
    # workflow replay).
    #
    # OTel context propagation is handled by ``@Durable.step`` —
    # the carrier travels in a magic kwarg from ``Durable.enqueue``
    # and is attached to this fiber before the body runs.
    del sample_idx
    branch = self._branches[label]
    result = await branch.agent.run(task)
    return self._hypotheses(result.output)
```

- [ ] **Step 6: Update `_converge` step body**

Locate:

```python
@Durable.step()
async def _converge(
    self, task: InT, candidates: list[HypothesisT],
) -> OutT:
    return await self._synthesizer.synthesize(
        task=task, candidates=candidates,
    )
```

Replace with:

```python
@Durable.step()
async def _converge(
    self, task: InT, candidates: list[HypothesisT],
) -> OutT:
    prompt = self._format_synth_prompt(task, candidates)
    result = await self._synthesizer.run(prompt)
    return result.output
```

- [ ] **Step 7: Run the unit test from Task 1 to verify it passes**

Run: `uv run pytest tests/patterns/test_divergent_convergent.py -v`

Expected: PASS (the new API matches the test).

- [ ] **Step 8: Run the full framework test suite to check no regressions**

Run: `uv run pytest tests/ -x -q`

Expected: all tests pass (473+ passed previously) plus the new test.

- [ ] **Step 9: Commit**

```bash
git add src/pydantic_ai_stateflow/patterns/divergent_convergent/pattern.py
git commit -m "refactor(patterns): DivergentConvergent owns envelope→hypotheses mapping"
```

---

## Task 4: Strip adapter methods from notes-app agents

**Files:**
- Modify: `examples/notes-app/backend/src/notes_app/brainstorm_agents.py`

- [ ] **Step 1: Delete the `diverge` method from `BrainstormDivergentAgent`**

In `examples/notes-app/backend/src/notes_app/brainstorm_agents.py`, locate and delete this block (it was appended after `model_settings`):

```python
    async def diverge(self, task: str) -> list[TodoIdea]:
        """``DivergentAgent[str, TodoIdea]`` Protocol implementation —
        called by ``DivergentConvergent`` for each branch sample.

        The agent's ``output_type=TodoIdeas`` shapes the LLM response;
        we unpack ``.ideas`` so the framework gets the flat hypothesis
        list it expects.
        """
        result = await self.agent.run(task, model_settings=self.model_settings())
        ideas: TodoIdeas = result.output
        return ideas.ideas
```

- [ ] **Step 2: Delete the `synthesize` method and `_format_synth_prompt` helper**

In the same file, locate and delete this block (appended after `BrainstormSynthesizerAgent.model_settings`):

```python
    async def synthesize(
        self, *, task: str, candidates: list[TodoIdea],
    ) -> TodoIdea:
        """``Synthesizer[str, TodoIdea, TodoIdea]`` Protocol —
        called by ``DivergentConvergent`` with the surviving pool.

        Renders the candidates into the prompt; the agent's
        ``output_type=TodoIdea`` constrains the response to a single
        chosen idea (lightly edited / blended is fine)."""
        prompt = _format_synth_prompt(task, candidates)
        result = await self.agent.run(prompt, model_settings=self.model_settings())
        chosen: TodoIdea = result.output
        return chosen


def _format_synth_prompt(task: str, candidates: list[TodoIdea]) -> str:
    lines = [f"Тема: {task}", "", "Кандидаты:"]
    for i, idea in enumerate(candidates, 1):
        lines.append(f"{i}. {idea.title} — {idea.body}")
    return "\n".join(lines)
```

After deletion, the file should end with the closing of `BrainstormSynthesizerAgent.model_settings`, followed by the existing `__all__` block:

```python
__all__ = ["BrainstormDivergentAgent", "BrainstormSynthesizerAgent"]
```

- [ ] **Step 3: Verify the file parses**

Run: `uv run python -c "from notes_app import brainstorm_agents; print(brainstorm_agents.__all__)"` from `examples/notes-app/backend`.

Expected: `['BrainstormDivergentAgent', 'BrainstormSynthesizerAgent']`.

- [ ] **Step 4: Commit**

```bash
git add examples/notes-app/backend/src/notes_app/brainstorm_agents.py
git commit -m "refactor(notes-app): drop diverge/synthesize adapter methods from agents"
```

---

## Task 5: Wire projectors in `brainstorm_flow.py` factory

**Files:**
- Modify: `examples/notes-app/backend/src/notes_app/brainstorm_flow.py`

- [ ] **Step 1: Add a module-level `_format_synth_prompt` helper**

In `examples/notes-app/backend/src/notes_app/brainstorm_flow.py`, immediately after the `CONVERGENT_PROMPT` block (search for `CONVERGENT_PROMPT = (`), add this helper:

```python
def _format_synth_prompt(task: str, candidates: list[TodoIdea]) -> str:
    """Render the candidate pool into a synthesis prompt.

    Lives in the factory module (not on the agent) — it's part of how
    THIS app wires the pattern, not part of the synthesizer's own
    behaviour. The pattern receives it as ``format_synth_prompt`` so
    the unwrap (envelope → list) and the prompt-rendering both live
    at the same boundary."""
    lines = [f"Тема: {task}", "", "Кандидаты:"]
    for i, idea in enumerate(candidates, 1):
        lines.append(f"{i}. {idea.title} — {idea.body}")
    return "\n".join(lines)
```

- [ ] **Step 2: Pass projectors when constructing `DivergentConvergent`**

Locate this block inside `build_brainstorm_flow`:

```python
    divergent = DivergentConvergent[str, TodoIdea, TodoIdea](
        branches=branches,
        synthesizer=synthesizer,
        deduper=deduper,
        best_of_n=best_of_n,
        min_hypotheses=min_hypotheses,
        top_k=top_k,
        divergent_concurrency=divergent_concurrency,
        config_name=f"{config_name}-divergent",
    )
```

Replace with:

```python
    divergent = DivergentConvergent[str, TodoIdeas, TodoIdea, TodoIdea](
        branches=branches,
        synthesizer=synthesizer,
        hypotheses=lambda env: env.ideas,
        format_synth_prompt=_format_synth_prompt,
        deduper=deduper,
        best_of_n=best_of_n,
        min_hypotheses=min_hypotheses,
        top_k=top_k,
        divergent_concurrency=divergent_concurrency,
        config_name=f"{config_name}-divergent",
    )
```

Note: the generic parameter list now has **four** types (`InT=str`, `EnvT=TodoIdeas`, `HypT=TodoIdea`, `OutT=TodoIdea`).

- [ ] **Step 3: Run notes-app backend tests**

From repo root:

Run: `cd examples/notes-app/backend && uv run pytest -x -q`

Expected: 17 passed, 2 skipped (same baseline as before).

- [ ] **Step 4: Run framework tests to confirm no regressions**

From repo root:

Run: `uv run pytest tests/ -x -q`

Expected: all previously-passing tests still pass + the new
`test_pattern_invokes_hypotheses_projector_per_branch` test.

- [ ] **Step 5: Commit**

```bash
git add examples/notes-app/backend/src/notes_app/brainstorm_flow.py
git commit -m "refactor(notes-app): supply hypotheses/format_synth_prompt to pattern"
```

---

## Task 6: Live smoke (manual)

**Files:** none — manual verification only.

- [ ] **Step 1: Start backend**

```bash
cd examples/notes-app/backend
uv run uvicorn notes_app.main:app --reload --port 8000
```

Expected: starts without errors. DBOS launch logs should include
`notes-brainstorm-flow-divergent` queue registration.

- [ ] **Step 2: Start frontend (separate terminal)**

```bash
cd examples/notes-app/frontend
bun run dev
```

- [ ] **Step 3: Trigger brainstorm flow in the browser**

Open `http://localhost:3000`, type any message into a fresh thread, then click the **Brainstorm todo** button. Watch for:

- `data-brainstorm-progress` row mutating through diverge → converge → hitl
- per-branch `data-brainstorm-branch` rows (practical/creative/analyst) ticking off
- approval thread spawning in the sidebar with the proposed title/body

If anything fails: revert the last commit and inspect.

- [ ] **Step 4: No commit (manual step).**

---

## Self-Review

**Spec coverage** — every section of the design doc has a task:

- "Framework: structural agent Protocol" → Task 2.
- "Framework: `DivergentConvergent` accepts two callables" → Task 3.
- "App: agents lose their adapter methods" → Task 4.
- "App: factory wires the agents and the two callables" → Task 5.
- "Determinism / replay" note → step bodies in Task 3 already run inside `@Durable.step`; the spec's correctness claim is preserved by the refactor.
- "Migration" → Tasks 4 + 5 + 6 in order; single in-repo caller, BC-break allowed.
- "Testing — framework unit tests" → Task 1 + the assertion inside it covers "the `hypotheses` projector is invoked exactly once per successful branch result".
- "Testing — notes-app smoke" → Task 5 Step 3.
- "Testing — live verification" → Task 6.

**Placeholders** — none. Every step contains exact code or commands.

**Type consistency**

- `DivergentConvergent` generics: `[InT, EnvT, HypothesisT, OutT]` everywhere (Task 1 test, Task 3 class declaration, Task 5 factory).
- Constructor kwargs: `hypotheses`, `format_synth_prompt`, `synthesizer` — same names across Tasks 1, 3, 5.
- Synthesizer Protocol: single type param `[OutT]` — Task 2's Protocol matches the Task 1 mock's `async def run(self, prompt: str) -> _AgentResult` and Task 3's `await self._synthesizer.run(prompt)`.
- `DivergentBranch(label, agent)` — same shape in Tasks 1, 2, 5.
- `EnvT` TypeVar — declared in Task 2 (primitives) and Task 3 (pattern); not used in app code (Task 5 just substitutes `TodoIdeas` for it as a concrete type parameter).

No issues found.

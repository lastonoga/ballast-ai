# CoALA Unit Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Phase 1 (Episodic) + Phase 2 (Semantic) memory facades with a single `CoALAUnit` Protocol + three runtime adapters (`as_workflow`, `as_tool`, `as_capability`). One contract for memory-aware computation; framework owns runtime wiring; apps own all storage + retrieval logic.

**Architecture:** A `CoALAUnit` exposes four async methods — `observe(input) → observation`, `retrieve(observation) → context`, `act(observation, context) → output`, `learn(observation, context, output) → None`. Three framework-provided adapters wrap a unit into a `@Durable.workflow`, a `pydantic_ai.Tool`, or a `BallastCapability`. Apps write one class, pick adapter(s). No framework-owned `Episode` schema, no `EpisodicMemory` facade, no `SemanticMemory` decorator zoo — those phases are deleted entirely.

**Tech Stack:** Python 3.11+, pydantic v2, DBOS (`@Durable.workflow` / `@Durable.step`), pydantic-ai (`Tool`, `BallastCapability`), existing `current_user_id` ContextVar.

**Spec:** `docs/superpowers/specs/2026-05-25-coala-unit-architecture-design.md`

---

## Phase A — Cleanup (delete Phase 1+2)

**Ordering principle:** strip notes-app consumers first → strip Ballast setters → strip public-API re-exports → strip `scan_context` special-cases → delete memory subpackages → move `Scope` → delete Alembic migration. Each task leaves the test suite green.

---

### Task A1: Strip notes-app memory consumers

**Files:**
- Modify: `examples/notes-app/backend/src/notes_app/agents/notes.py` — drop `RememberTurn` from `default_notes_capabilities`; drop semantic-memory tool aggregation from `build_agent`
- Modify: `examples/notes-app/backend/src/notes_app/main.py` — drop `_build_episodic_memory()`, `with_episodic_memory(...)`, `with_semantic_memory(...)` calls + their imports
- Modify: `examples/notes-app/backend/src/notes_app/workflows/create_note.py` — drop the episodic recall block at the top of the workflow
- Delete: `examples/notes-app/backend/src/notes_app/memory/` (whole subpackage — `__init__.py` + `semantic_sources.py`)
- Delete: `examples/notes-app/backend/tests/test_notes_semantic.py`

- [ ] **Step 1: Strip `default_notes_capabilities` (RememberTurn)**

Open `examples/notes-app/backend/src/notes_app/agents/notes.py`. Find the `default_notes_capabilities()` function. Remove:

```python
# REMOVE these imports near the top:
from ballast import get_ballast
from ballast.memory.episodic import RememberTurn

# REMOVE the try/except block inside default_notes_capabilities():
    try:
        memory = getattr(get_ballast(), "_episodic_memory", None)
        if memory is not None:
            caps.append(RememberTurn(memory=memory))
    except Exception:  # noqa: BLE001
        pass
```

Replace with just `return caps`.

- [ ] **Step 2: Strip `build_agent` semantic tools**

In the same file, find `NotesAgent.build_agent`. Remove the semantic-tool aggregation:

```python
# REMOVE:
semantic = getattr(get_ballast(), "_semantic_memory", None)
extra_tools = semantic.as_tools() if semantic is not None else []

# CHANGE:
return Agent(
    model=build_openrouter_model(),
    output_type=[str, DeferredToolRequests],
    deps_type=NoteToolDeps,
    system_prompt=SYSTEM_PROMPT,
    capabilities=default_notes_capabilities(),
    tools=extra_tools,    # ← REMOVE this line
)
```

Keep `get_ballast` import only if still used elsewhere in the file — otherwise drop it.

- [ ] **Step 3: Strip `main.py` memory wiring**

Open `examples/notes-app/backend/src/notes_app/main.py`. Remove:

```python
# REMOVE imports:
from ballast.auth.context import current_user_id
from ballast.memory import Scope
from ballast.memory.episodic import EpisodicMemory, ThreadEpisodicSource
from ballast.memory.semantic import SemanticMemory
from notes_app.memory.semantic_sources import notes_semantic

# REMOVE the _build_episodic_memory function (whole def + body).
# REMOVE the _LazyThreadRepo helper class.
# REMOVE _openai_embedder = None constant.

# In the builder chain, REMOVE:
.with_episodic_memory(_build_episodic_memory())
.with_semantic_memory(SemanticMemory(sources=[notes_semantic]))
```

Keep `current_user_id` import ONLY if used elsewhere in main.py (verify with grep).

- [ ] **Step 4: Strip `create_note_flow` recall block**

Open `examples/notes-app/backend/src/notes_app/workflows/create_note.py`. Remove:

```python
# REMOVE imports:
import logging
from ballast import get_ballast
from ballast.memory.episodic import DetailLevel
from ballast.memory.episodic.strategies import TopK

_log = logging.getLogger(__name__)

# REMOVE the try/except block at the top of create_note_flow:
    try:
        memory = getattr(get_ballast(), "_episodic_memory", None)
        if memory is not None:
            recall = await memory.episodic_for(
                intent=f"prior notes about {draft.title}",
                strategy=TopK(k=3, detail=DetailLevel.PREVIEW),
            )
            _log.info(...)
    except Exception:
        _log.exception(...)
```

The workflow body should now go directly to its existing logic (build draft → channel.request → save).

- [ ] **Step 5: Delete `notes_app/memory/` subpackage + test**

```
git rm -r examples/notes-app/backend/src/notes_app/memory/
git rm examples/notes-app/backend/tests/test_notes_semantic.py
```

- [ ] **Step 6: Run notes-app suite — confirm green**

```
cd examples/notes-app/backend && uv run pytest -q
```

Expected: green (some test_notes_semantic was the only test specifically using NotesSemantic; remaining tests unaffected).

- [ ] **Step 7: Commit**

```bash
git add examples/notes-app/backend
git commit -m "refactor(notes-app): strip Phase 1+2 memory consumers (RememberTurn, NotesSemantic, recall in create_note_flow)"
```

---

### Task A2: Strip `Ballast` memory setters + attrs

**Files:**
- Modify: `src/ballast/app.py`
- Modify: `tests/app/test_with_memory.py` — delete or rewrite
- Modify: `tests/app/test_with_semantic_memory.py` — delete or rewrite

- [ ] **Step 1: Delete the memory tests**

```
git rm tests/app/test_with_memory.py
git rm tests/app/test_with_semantic_memory.py
```

- [ ] **Step 2: Remove setters + attrs from `Ballast`**

Open `src/ballast/app.py`. Remove:

```python
# REMOVE from TYPE_CHECKING imports:
from ballast.memory._scope import Scope
from ballast.memory.episodic._facade import EpisodicMemory
from ballast.memory.semantic._facade import SemanticMemory

# REMOVE from Ballast.__init__:
self._memory: "EpisodicMemory | None" = None
self._episodic_memory: "EpisodicMemory | None" = None
self._semantic_memory: "SemanticMemory | None" = None

# REMOVE the three methods:
def with_episodic_memory(...): ...
def with_semantic_memory(...): ...
def with_memory(...): ...
```

Drop any `Callable` import that becomes unused. Run mypy/pyright if applicable to spot dangling imports.

- [ ] **Step 3: Run framework + app suites — confirm green**

```
uv run pytest tests/ -q
```

Expected: green (the tests we deleted in Step 1 were the only consumers of the setters).

- [ ] **Step 4: Commit**

```bash
git add src/ballast/app.py tests/app/
git commit -m "refactor(app): drop with_memory/with_episodic_memory/with_semantic_memory setters + memory attrs"
```

---

### Task A3: Strip top-level public API re-exports

**Files:**
- Modify: `src/ballast/__init__.py`
- Modify: `src/ballast/memory/__init__.py` — strip semantic re-exports (leaves only `Scope` for now; A6 moves it)

- [ ] **Step 1: Edit `src/ballast/__init__.py`**

Remove ALL memory-related imports + `__all__` entries:

```python
# REMOVE imports (or find and delete this block):
from ballast.memory import Scope
from ballast.memory.episodic import (
    DetailLevel,
    Episode,
    EpisodicMemory,
    EpisodicSource,
    RecallResult,
    RememberTurn,
    ScoredEpisode,
)
from ballast.memory.episodic.sources import (
    EpisodeRow,
    ThreadEpisodicSource,
    VectorEpisodicSource,
)
from ballast.memory.episodic.strategies import (
    AllRelevant, Cluster, MapReduce as MapReduceStrategy, Recency, RecallStrategy, TopK,
)
from ballast.memory.semantic import (
    DomainSemanticSource, SemanticMemory, SemanticSource, VectorSemanticSource, memory_tool,
)
from ballast.patterns.map_reduce import map_reduce_llm   # KEEP this one — it's a pattern
```

Keep `map_reduce_llm` (it's an orthogonal pattern, not memory). Also keep the `Scope` import temporarily — A6 moves it to `ballast.auth.scope` and we'll re-import from there.

Remove all the memory names from `__all__` (alphabetical insertion — find + delete).

- [ ] **Step 2: Edit `src/ballast/memory/__init__.py`**

Remove all semantic re-exports; leave only `Scope` for now (moves in A6):

```python
"""CoALA-inspired memory subsystem. (Phase 1+2 deleted; CoALAUnit replaces.)"""
from ballast.memory._scope import Scope

__all__ = ["Scope"]
```

- [ ] **Step 3: Run suite — confirm green**

```
uv run pytest tests/ -q
```

Expected: green (nothing uses the dropped exports anymore — notes-app stripped in A1, app tests stripped in A2).

- [ ] **Step 4: Commit**

```bash
git add src/ballast/__init__.py src/ballast/memory/__init__.py
git commit -m "refactor(ballast): drop Phase 1+2 memory re-exports from public API"
```

---

### Task A4: Revert `scan_context` Episode/RecallResult special-casing (keep Ref recognition)

**Files:**
- Modify: `src/ballast/grounded/_scan.py`
- Modify: `tests/memory/test_grounded_integration.py` — delete (memory tests dir gets fully removed in A5)

- [ ] **Step 1: Inspect current state**

```
grep -n "Episode\|RecallResult\|Ref" src/ballast/grounded/_scan.py
```

Phase 1 added two things to `_walk`:
1. Recognition of `Ref` instances (`isinstance(obj, Ref)`) — this is **GENERAL-PURPOSE**, keep
2. Special-casing for `Episode` / `RecallResult` (or descent that unwrapped their `.references` field) — this is **memory-specific**, delete

- [ ] **Step 2: Strip Episode/RecallResult handling**

In `src/ballast/grounded/_scan.py`, find and remove any branches like:

```python
# REMOVE:
from ballast.memory.episodic._models import Episode, RecallResult
# OR lazy imports of the same.

# REMOVE any isinstance(obj, Episode) / isinstance(obj, RecallResult) branches.

# KEEP the Ref recognition branch:
if isinstance(obj, Ref):
    if obj.entity_type in targets:
        sources.by_entity_type.setdefault(obj.entity_type, []).append(obj.id)
    return
```

If the Episode/RecallResult handling came VIA the Ref recognition (i.e. the `_walk` already descends BaseModel.references list-of-Refs naturally), there may be nothing explicit to remove — just verify by reading the file. The point is: no symbol from `ballast.memory.*` should be imported by `_scan.py`.

- [ ] **Step 3: Delete the integration test**

```
git rm tests/memory/test_grounded_integration.py
```

(Whole `tests/memory/` directory is removed in A5; this is a head-start.)

- [ ] **Step 4: Run grounded tests — confirm green**

```
uv run pytest tests/grounded/ -q
```

Expected: green — `Ref` recognition keeps working.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/grounded/_scan.py tests/memory/test_grounded_integration.py
git commit -m "refactor(grounded): drop Episode/RecallResult special-cases from scan_context (Ref recognition kept)"
```

---

### Task A5: Delete `src/ballast/memory/episodic/` + `src/ballast/memory/semantic/` subpackages + tests

**Files:**
- Delete: `src/ballast/memory/episodic/` (whole subpackage)
- Delete: `src/ballast/memory/semantic/` (whole subpackage)
- Delete: `tests/memory/episodic/` (whole)
- Delete: `tests/memory/semantic/` (whole)
- Delete: `tests/memory/test_models.py`, `test_scope.py`, `test_protocol.py`, `test_mergers.py`, `test_facade.py`, `test_tools.py`, `test_remember_turn.py` (whatever remains in `tests/memory/`)

- [ ] **Step 1: git rm both source subpackages**

```
git rm -r src/ballast/memory/episodic/
git rm -r src/ballast/memory/semantic/
```

- [ ] **Step 2: git rm all remaining memory tests**

```
git rm -r tests/memory/
```

(Whole directory — `tests/memory/test_scope.py` content moves to `tests/auth/` in A6.)

- [ ] **Step 3: Also drop the persistence test for semantic vector + episodic VectorEpisodicSource**

```
git rm tests/persistence/test_semantic_vector.py
git rm tests/persistence/test_vector_source.py    # if it exists
```

Also drop the model-registration imports in `tests/persistence/conftest.py`:

```python
# REMOVE these lines from the import block at top:
import ballast.memory.episodic.sources._vector  # noqa: F401
import tests.persistence.test_semantic_vector   # noqa: F401
```

- [ ] **Step 4: Run framework suite — confirm green**

```
uv run pytest tests/ -q
```

Expected: green. No dangling imports (we stripped all consumers in A1-A4).

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory tests/memory tests/persistence/
git commit -m "refactor(memory): delete Phase 1+2 subpackages (episodic, semantic) entirely"
```

---

### Task A6: Move `Scope` → `ballast.auth.scope`

**Files:**
- Create: `src/ballast/auth/scope.py`
- Delete: `src/ballast/memory/_scope.py`
- Modify: `src/ballast/memory/__init__.py` — empty out (or delete dir if nothing remains)
- Modify: `src/ballast/auth/__init__.py` — re-export `Scope`
- Modify: `src/ballast/__init__.py` — import `Scope` from `ballast.auth`
- Create: `tests/auth/test_scope.py` — copy from former `tests/memory/test_scope.py` content (already deleted; rewrite below)

- [ ] **Step 1: Create the new location**

`src/ballast/auth/scope.py`:

```python
"""``Scope`` — base scope for memory queries / repository filters.

Apps subclass to add domain-specific dimensions (project_id, org_id,
team_id, …). ``extra="allow"`` so consumers can graceful-read
app-custom fields via getattr without requiring a subclass.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Scope(BaseModel):
    """Base scope. Subclass to add app-specific dimensions."""

    model_config = ConfigDict(extra="allow")

    user_id:   str | None = None
    tenant_id: str | None = None
    thread_id: str | None = None


__all__ = ["Scope"]
```

- [ ] **Step 2: Update `ballast.auth.__init__`**

Open `src/ballast/auth/__init__.py`. Add:

```python
from ballast.auth.scope import Scope

# extend __all__:
__all__ = ["Scope", "acting_as", "current_user_id"]   # add "Scope" alphabetical
```

- [ ] **Step 3: Update `ballast/__init__.py`**

Change the import for `Scope`:

```python
# WAS:
from ballast.memory import Scope

# CHANGE TO:
from ballast.auth import Scope
```

- [ ] **Step 4: Delete the old memory subpackage entirely**

```
git rm -r src/ballast/memory/
```

(After A5 deleted the subpackages, only `_scope.py` + `__init__.py` remained. Now delete the whole dir.)

- [ ] **Step 5: Write the failing scope test**

Create `tests/auth/test_scope.py`:

```python
"""``Scope`` — app-subclassable scope BaseModel with extra=allow."""
from __future__ import annotations

from ballast.auth import Scope


def test_default_scope_has_optional_user_tenant_thread() -> None:
    s = Scope()
    assert s.user_id is None
    assert s.tenant_id is None
    assert s.thread_id is None


def test_explicit_scope_fields() -> None:
    s = Scope(user_id="u-1", tenant_id="t-1", thread_id="th-1")
    assert (s.user_id, s.tenant_id, s.thread_id) == ("u-1", "t-1", "th-1")


def test_extra_fields_allowed_for_app_dimensions() -> None:
    s = Scope(user_id="u-1", project_id="p-99")  # type: ignore[call-arg]
    assert getattr(s, "project_id", None) == "p-99"


def test_subclass_adds_typed_dimensions() -> None:
    class ProjectScope(Scope):
        project_id: str | None = None

    s = ProjectScope(user_id="u-1", project_id="p-9")
    assert s.project_id == "p-9"
    assert isinstance(s, Scope)
```

- [ ] **Step 6: Run suite — confirm green**

```
uv run pytest tests/ -q
```

Expected: green (Scope tests pass at new location; nothing else broke).

- [ ] **Step 7: Commit**

```bash
git add src/ballast/auth src/ballast/__init__.py tests/auth/test_scope.py src/ballast/memory
git commit -m "refactor(scope): move Scope from ballast.memory → ballast.auth.scope (final memory/ cleanup)"
```

---

### Task A7: Drop Alembic 0003 episodes migration

**Files:**
- Delete: `src/ballast/alembic/versions/0003_episodes.py`
- Modify: `tests/persistence/test_alembic_migration.py` — remove `"episodes"` from expected tables (if Phase 1 added it)
- Modify: `pyproject.toml` — consider keeping `pgvector` optional dep (apps may still use it) — DON'T drop

- [ ] **Step 1: Delete the migration**

```
git rm src/ballast/alembic/versions/0003_episodes.py
```

- [ ] **Step 2: Update Alembic migration smoke test**

```
grep -n "episodes" tests/persistence/test_alembic_migration.py
```

If the test asserts `"episodes"` is in the expected tables set, remove that entry.

- [ ] **Step 3: Keep pgvector in pyproject.toml**

Apps may still use pgvector for their own custom storage. Verify it's listed under `[project.optional-dependencies]` and leave as-is.

- [ ] **Step 4: Run suite — confirm green**

```
uv run pytest tests/ -q
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/alembic tests/persistence/test_alembic_migration.py
git commit -m "refactor(persistence): drop Alembic 0003 episodes migration (Phase 1 table no longer used)"
```

---

## Phase B — Build CoALA Unit Architecture

---

### Task B1: `CoALAUnit` Protocol

**Files:**
- Create: `src/ballast/coala/__init__.py`
- Create: `src/ballast/coala/_protocol.py`
- Create: `tests/coala/__init__.py` (empty)
- Create: `tests/coala/test_protocol.py`

- [ ] **Step 1: Write the failing test**

Create `tests/coala/__init__.py` (empty). Then `tests/coala/test_protocol.py`:

```python
"""``CoALAUnit`` Protocol — structural type for memory-aware units."""
from __future__ import annotations

from ballast.coala import CoALAUnit


def test_runtime_checkable_protocol() -> None:
    class _Stub:
        async def observe(self, input): return input
        async def retrieve(self, observation): return {}
        async def act(self, observation, context): return None
        async def learn(self, observation, context, output): return None

    assert isinstance(_Stub(), CoALAUnit)


def test_protocol_rejects_missing_phase() -> None:
    class _NoLearn:
        async def observe(self, input): return input
        async def retrieve(self, observation): return {}
        async def act(self, observation, context): return None
        # learn missing

    assert not isinstance(_NoLearn(), CoALAUnit)
```

- [ ] **Step 2: Run — confirm fail**

```
uv run pytest tests/coala/test_protocol.py -v
```

Expected: `ModuleNotFoundError: No module named 'ballast.coala'`.

- [ ] **Step 3: Implement Protocol**

Create `src/ballast/coala/_protocol.py`:

```python
"""``CoALAUnit`` Protocol — single contract for memory-aware computation."""
from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

InT      = TypeVar("InT")
ObsT     = TypeVar("ObsT")
ContextT = TypeVar("ContextT")
OutT     = TypeVar("OutT")


@runtime_checkable
class CoALAUnit(Protocol[InT, ObsT, ContextT, OutT]):
    """Unit of memory-aware computation following CoALA's 4-phase
    decision procedure.

    Same contract regardless of runtime — a workflow, an agent tool, an
    agent capability — any can be wrapped via the corresponding adapter
    (``as_workflow``, ``as_tool``, ``as_capability``).

    Phase semantics (from Sumers et al., "Cognitive Architectures for
    Language Agents"):
      observe  — parse raw input into structured working-memory state
      retrieve — pull relevant long-term memory based on observation
      act      — reason + ground + execute; produces output
      learn    — persist insights back into long-term memory
    """

    async def observe(self, input: InT) -> ObsT: ...
    async def retrieve(self, observation: ObsT) -> ContextT: ...
    async def act(self, observation: ObsT, context: ContextT) -> OutT: ...
    async def learn(self, observation: ObsT, context: ContextT, output: OutT) -> None: ...


__all__ = ["CoALAUnit"]
```

- [ ] **Step 4: Implement package __init__**

Create `src/ballast/coala/__init__.py`:

```python
"""CoALA Unit Architecture — single Protocol + multiple runtime adapters."""
from ballast.coala._protocol import CoALAUnit

__all__ = ["CoALAUnit"]
```

- [ ] **Step 5: Run — confirm pass**

```
uv run pytest tests/coala/test_protocol.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/coala tests/coala
git commit -m "feat(coala): CoALAUnit Protocol (observe / retrieve / act / learn)"
```

---

### Task B2: `CoALABase` ABC with sensible defaults

**Files:**
- Create: `src/ballast/coala/_base.py`
- Modify: `src/ballast/coala/__init__.py` (export `CoALABase`)
- Create: `tests/coala/test_base.py`

- [ ] **Step 1: Write the failing test**

```python
"""``CoALABase`` ABC — default observe/learn; abstract retrieve/act."""
from __future__ import annotations

import pytest

from ballast.coala import CoALABase, CoALAUnit


class _Minimal(CoALABase[str, str, dict, str]):
    async def retrieve(self, observation): return {"data": "ctx"}
    async def act(self, observation, context): return f"acted on {observation}"


def test_default_observe_is_identity() -> None:
    import asyncio
    out = asyncio.run(_Minimal().observe("hello"))
    assert out == "hello"


def test_default_learn_is_no_op() -> None:
    import asyncio
    result = asyncio.run(_Minimal().learn("o", {}, "out"))
    assert result is None


def test_subclass_satisfies_coala_unit_protocol() -> None:
    assert isinstance(_Minimal(), CoALAUnit)


def test_abstract_retrieve_must_be_overridden() -> None:
    class _NoRetrieve(CoALABase):
        async def act(self, observation, context): return None

    with pytest.raises(TypeError, match="abstract"):
        _NoRetrieve()  # type: ignore[abstract]


def test_abstract_act_must_be_overridden() -> None:
    class _NoAct(CoALABase):
        async def retrieve(self, observation): return {}

    with pytest.raises(TypeError, match="abstract"):
        _NoAct()  # type: ignore[abstract]
```

- [ ] **Step 2: Run — confirm fail**

Expected: `ImportError` for `CoALABase`.

- [ ] **Step 3: Implement**

Create `src/ballast/coala/_base.py`:

```python
"""``CoALABase`` ABC — ergonomic base with default observe + learn."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InT      = TypeVar("InT")
ObsT     = TypeVar("ObsT")
ContextT = TypeVar("ContextT")
OutT     = TypeVar("OutT")


class CoALABase(Generic[InT, ObsT, ContextT, OutT], ABC):
    """Minimal-friction base. Apps override only the phases they need.

    ``observe`` defaults to identity (input passes through). Override
    when you need to extract intent / entities / signals before retrieve.

    ``learn`` defaults to no-op. Override to write episodes, facts,
    learned skills, etc. — anything the app wants to persist.

    ``retrieve`` and ``act`` are abstract — every meaningful unit has a
    retrieval step (even if it returns an empty Context) and an act
    step (the actual work).
    """

    async def observe(self, input: InT) -> ObsT:
        return input  # type: ignore[return-value]

    @abstractmethod
    async def retrieve(self, observation: ObsT) -> ContextT: ...

    @abstractmethod
    async def act(self, observation: ObsT, context: ContextT) -> OutT: ...

    async def learn(
        self, observation: ObsT, context: ContextT, output: OutT,
    ) -> None:
        return None


__all__ = ["CoALABase"]
```

- [ ] **Step 4: Update package __init__**

```python
"""CoALA Unit Architecture — single Protocol + multiple runtime adapters."""
from ballast.coala._base import CoALABase
from ballast.coala._protocol import CoALAUnit

__all__ = ["CoALABase", "CoALAUnit"]
```

- [ ] **Step 5: Run — confirm pass**

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/coala/_base.py src/ballast/coala/__init__.py tests/coala/test_base.py
git commit -m "feat(coala): CoALABase ABC with default observe + learn"
```

---

### Task B3: `as_workflow` adapter

**Files:**
- Create: `src/ballast/coala/adapters/__init__.py`
- Create: `src/ballast/coala/adapters/workflow.py`
- Modify: `src/ballast/coala/__init__.py` (export `as_workflow`)
- Create: `tests/coala/test_workflow_adapter.py`
- Create: `tests/coala/conftest.py` (DBOS bootstrap)

- [ ] **Step 1: Write DBOS fixture**

Create `tests/coala/conftest.py`:

```python
"""DBOS bootstrap for CoALA adapter tests."""
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
    tmp = tempfile.mkdtemp(prefix="dbos-coala-")
    DBOS(config=DBOSConfig(
        name="coala-test",
        system_database_url=f"sqlite:///{Path(tmp)/'dbos.sqlite'}",
    ))
    DBOS.launch()
    try: yield DBOS
    finally: DBOS.destroy(destroy_registry=False)


@pytest_asyncio.fixture
async def fresh_dbos_executor(dbos_runtime: type[DBOS]) -> AsyncIterator[None]:
    from dbos._dbos import _get_dbos_instance
    _get_dbos_instance()._executor_field = ThreadPoolExecutor(
        max_workers=8, thread_name_prefix="dbos-test-",
    )
    yield
```

- [ ] **Step 2: Write the failing test**

```python
"""``as_workflow`` adapter — wraps CoALAUnit as @Durable.workflow."""
from __future__ import annotations

import pytest

from ballast.coala import CoALABase, as_workflow


class _Recording(CoALABase[str, str, dict, str]):
    """Records each phase call for assertion."""
    calls: list[str] = []

    async def observe(self, input):
        self.calls.append(f"observe({input})")
        return input.upper()

    async def retrieve(self, observation):
        self.calls.append(f"retrieve({observation})")
        return {"ctx": observation + "_data"}

    async def act(self, observation, context):
        self.calls.append(f"act({observation}, {context})")
        return f"{observation}|{context['ctx']}"

    async def learn(self, observation, context, output):
        self.calls.append(f"learn(out={output})")


@pytest.mark.asyncio
async def test_workflow_runs_all_four_phases_in_order(
    fresh_dbos_executor: None,
) -> None:
    unit = _Recording()
    unit.calls = []     # reset (class-level list)
    runner = as_workflow(unit)
    out = await runner("hello")
    assert out == "HELLO|HELLO_data"
    assert unit.calls == [
        "observe(hello)",
        "retrieve(HELLO)",
        "act(HELLO, {'ctx': 'HELLO_data'})",
        "learn(out=HELLO|HELLO_data)",
    ]


@pytest.mark.asyncio
async def test_workflow_returns_act_output_not_learn(
    fresh_dbos_executor: None,
) -> None:
    class _Unit(CoALABase[str, str, dict, str]):
        async def retrieve(self, observation): return {}
        async def act(self, observation, context): return "from-act"
        async def learn(self, observation, context, output): return None

    runner = as_workflow(_Unit())
    out = await runner("x")
    assert out == "from-act"
```

- [ ] **Step 3: Run — confirm fail**

Expected: `ImportError` for `as_workflow`.

- [ ] **Step 4: Implement**

Create `src/ballast/coala/adapters/__init__.py`:

```python
"""Runtime adapters for CoALAUnit — workflow, tool, capability."""
from ballast.coala.adapters.workflow import as_workflow

__all__ = ["as_workflow"]
```

Create `src/ballast/coala/adapters/workflow.py`:

```python
"""``as_workflow`` — adapt CoALAUnit to @Durable.workflow."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from ballast.coala._protocol import CoALAUnit
from ballast.durable import Durable

InT  = TypeVar("InT")
OutT = TypeVar("OutT")


def as_workflow(
    unit: CoALAUnit[InT, Any, Any, OutT],
) -> Callable[[InT], Awaitable[OutT]]:
    """Wrap a CoALAUnit as a @Durable.workflow runner.

    Each phase becomes a @Durable.step — memoised on replay, retryable.
    Crash mid-lifecycle: already-completed phases skip; only the
    unfinished tail re-runs.

    Returns a plain async callable. The unit instance is captured via
    closure (NOT serialised by DBOS — callables can't be pickled as
    workflow args).
    """
    @Durable.workflow()
    async def runner(input: InT) -> OutT:
        observation = await _observe_step(unit, input)
        context     = await _retrieve_step(unit, observation)
        output      = await _act_step(unit, observation, context)
        await _learn_step(unit, observation, context, output)
        return output

    runner.__name__ = f"coala_workflow_{type(unit).__name__}"
    runner.__doc__  = (type(unit).__doc__ or "").strip() or None
    return runner


# Each phase wrapped as a step — memoised on replay.
@Durable.step()
async def _observe_step(unit: Any, input: Any) -> Any:
    return await unit.observe(input)


@Durable.step()
async def _retrieve_step(unit: Any, observation: Any) -> Any:
    return await unit.retrieve(observation)


@Durable.step()
async def _act_step(unit: Any, observation: Any, context: Any) -> Any:
    return await unit.act(observation, context)


@Durable.step()
async def _learn_step(unit: Any, observation: Any, context: Any, output: Any) -> None:
    return await unit.learn(observation, context, output)


__all__ = ["as_workflow"]
```

- [ ] **Step 5: Update package __init__**

Edit `src/ballast/coala/__init__.py`:

```python
"""CoALA Unit Architecture — single Protocol + multiple runtime adapters."""
from ballast.coala._base import CoALABase
from ballast.coala._protocol import CoALAUnit
from ballast.coala.adapters import as_workflow

__all__ = ["CoALABase", "CoALAUnit", "as_workflow"]
```

- [ ] **Step 6: Run — confirm pass**

```
uv run pytest tests/coala/test_workflow_adapter.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add src/ballast/coala tests/coala/test_workflow_adapter.py tests/coala/conftest.py
git commit -m "feat(coala): as_workflow adapter (CoALAUnit → @Durable.workflow with per-phase steps)"
```

---

### Task B4: `as_tool` adapter

**Files:**
- Create: `src/ballast/coala/adapters/tool.py`
- Modify: `src/ballast/coala/adapters/__init__.py` (export `as_tool`)
- Modify: `src/ballast/coala/__init__.py` (export `as_tool`)
- Create: `tests/coala/test_tool_adapter.py`

- [ ] **Step 1: Write the failing test**

```python
"""``as_tool`` adapter — wraps CoALAUnit as pydantic-ai Tool."""
from __future__ import annotations

import pytest

from ballast.coala import CoALABase, as_tool


class _Greeter(CoALABase[str, str, dict, str]):
    """Greets the user."""
    calls: list[str] = []

    async def observe(self, input):
        self.calls.append(f"observe({input})")
        return input

    async def retrieve(self, observation):
        self.calls.append(f"retrieve({observation})")
        return {"prefix": "Hi"}

    async def act(self, observation, context):
        self.calls.append(f"act({observation})")
        return f"{context['prefix']}, {observation}!"

    async def learn(self, observation, context, output):
        self.calls.append(f"learn({output})")


def test_tool_name_defaults_to_class_name() -> None:
    tool = as_tool(_Greeter())
    assert tool.name == "_Greeter"


def test_tool_description_defaults_to_class_docstring() -> None:
    tool = as_tool(_Greeter())
    assert (tool.description or "").strip() == "Greets the user."


def test_tool_overrides_take_precedence() -> None:
    tool = as_tool(_Greeter(), name="greet", description="custom desc")
    assert tool.name == "greet"
    assert tool.description == "custom desc"


@pytest.mark.asyncio
async def test_tool_invocation_runs_all_phases() -> None:
    unit = _Greeter()
    unit.calls = []
    tool = as_tool(unit)
    out = await tool.function(input="Alice")
    assert out == "Hi, Alice!"
    assert unit.calls == [
        "observe(Alice)",
        "retrieve(Alice)",
        "act(Alice)",
        "learn(Hi, Alice!)",
    ]
```

- [ ] **Step 2: Run — confirm fail**

Expected: `ImportError` for `as_tool`.

- [ ] **Step 3: Implement**

Create `src/ballast/coala/adapters/tool.py`:

```python
"""``as_tool`` — adapt CoALAUnit to pydantic-ai Tool."""
from __future__ import annotations

import inspect
from typing import Any

from pydantic_ai import Tool

from ballast.coala._protocol import CoALAUnit


def as_tool(
    unit: CoALAUnit,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    """Wrap a CoALAUnit as a pydantic-ai Tool.

    From the LLM's POV: one tool call. Internally framework runs all
    four CoALA phases — observe parses LLM-supplied args, retrieve
    fetches memory, act produces output, learn records. Output is
    returned to the agent for next-step reasoning.

    Tool name defaults to ``type(unit).__name__``; description defaults
    to the unit's class docstring. Both overridable via kwargs.

    The LLM-facing arg schema is derived from the unit's ``observe``
    signature (specifically, the parameter after ``self``). Apps choose
    the schema by typing ``observe``'s input: ``BaseModel`` for nested
    JSON, primitives for flat args.
    """
    unit_name = name or type(unit).__name__
    unit_desc = description or (type(unit).__doc__ or "").strip() or None

    observe_sig = inspect.signature(type(unit).observe)
    # Drop ``self`` — remaining is the InT parameter
    params_after_self = list(observe_sig.parameters.values())[1:]
    if len(params_after_self) != 1:
        raise ValueError(
            f"CoALAUnit.observe must take exactly one parameter after self; "
            f"{type(unit).__name__}.observe has {len(params_after_self)}",
        )
    input_param = params_after_self[0]

    async def _runner(**kwargs: Any) -> Any:
        input_value = kwargs[input_param.name]
        observation = await unit.observe(input_value)
        context     = await unit.retrieve(observation)
        output      = await unit.act(observation, context)
        await unit.learn(observation, context, output)
        return output

    _runner.__signature__ = inspect.Signature(
        parameters=[inspect.Parameter(
            input_param.name,
            kind=inspect.Parameter.KEYWORD_ONLY,
            annotation=input_param.annotation,
        )],
        return_annotation=observe_sig.return_annotation,
    )
    _runner.__name__ = unit_name
    _runner.__doc__  = unit_desc

    return Tool(_runner, name=unit_name, description=unit_desc, takes_ctx=False)


__all__ = ["as_tool"]
```

- [ ] **Step 4: Update adapter package + main __init__**

`src/ballast/coala/adapters/__init__.py`:

```python
"""Runtime adapters for CoALAUnit — workflow, tool, capability."""
from ballast.coala.adapters.tool import as_tool
from ballast.coala.adapters.workflow import as_workflow

__all__ = ["as_tool", "as_workflow"]
```

`src/ballast/coala/__init__.py`:

```python
"""CoALA Unit Architecture — single Protocol + multiple runtime adapters."""
from ballast.coala._base import CoALABase
from ballast.coala._protocol import CoALAUnit
from ballast.coala.adapters import as_tool, as_workflow

__all__ = ["CoALABase", "CoALAUnit", "as_tool", "as_workflow"]
```

- [ ] **Step 5: Run — confirm pass**

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/coala tests/coala/test_tool_adapter.py
git commit -m "feat(coala): as_tool adapter (CoALAUnit → pydantic-ai Tool with derived schema)"
```

---

### Task B5: `as_capability` adapter

**Files:**
- Create: `src/ballast/coala/adapters/capability.py`
- Modify: `src/ballast/coala/adapters/__init__.py` (export `as_capability`)
- Modify: `src/ballast/coala/__init__.py` (export `as_capability`)
- Create: `tests/coala/test_capability_adapter.py`

- [ ] **Step 1: Locate the BallastCapability before/after hook signatures**

```
grep -n "async def before_model_request\|async def after_run" src/ballast/capabilities/*.py src/ballast/capabilities/**/*.py 2>&1 | head -10
```

Confirm the exact signature `as_capability` must implement. Phase 1's `RememberTurn` used `async def after_run(self, ctx, *, result)`.

- [ ] **Step 2: Write the failing test**

```python
"""``as_capability`` adapter — wraps CoALAUnit as pydantic-ai capability."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ballast.coala import CoALABase, as_capability


class _Recording(CoALABase[str, str, dict, str]):
    """Records phase calls."""
    calls: list[str] = []

    async def observe(self, input):
        self.calls.append(f"observe({input})")
        return input

    async def retrieve(self, observation):
        self.calls.append(f"retrieve({observation})")
        return {"ctx": "data"}

    async def act(self, observation, context):
        self.calls.append("act-NOT-CALLED-BY-FRAMEWORK")
        return "should-not-run"

    async def learn(self, observation, context, output):
        self.calls.append(f"learn({output})")


@dataclass
class _FakeCtx:
    deps: dict = None
    def __post_init__(self): self.deps = self.deps or {}


@dataclass
class _FakeResult:
    output: str = "agent-output"


@pytest.mark.asyncio
async def test_capability_observe_and_retrieve_fire_before_request() -> None:
    unit = _Recording()
    unit.calls = []
    cap = as_capability(unit)
    ctx = _FakeCtx()
    out = await cap.before_model_request(ctx, "hello-msg")
    # observe + retrieve fired in order
    assert unit.calls == ["observe(hello-msg)", "retrieve(hello-msg)"]


@pytest.mark.asyncio
async def test_capability_learn_fires_after_run() -> None:
    unit = _Recording()
    unit.calls = []
    cap = as_capability(unit)
    ctx = _FakeCtx()
    await cap.before_model_request(ctx, "msg")
    unit.calls = []   # reset
    result = _FakeResult(output="final-output")
    await cap.after_run(ctx, result=result)
    assert unit.calls == ["learn(final-output)"]


@pytest.mark.asyncio
async def test_capability_gate_skips_learn() -> None:
    unit = _Recording()
    unit.calls = []
    cap = as_capability(unit, gate=lambda result: False)
    ctx = _FakeCtx()
    await cap.before_model_request(ctx, "msg")
    unit.calls = []
    await cap.after_run(ctx, result=_FakeResult())
    assert unit.calls == []     # learn skipped


@pytest.mark.asyncio
async def test_capability_swallows_learn_exceptions() -> None:
    class _BrokenLearn(CoALABase[str, str, dict, str]):
        async def retrieve(self, observation): return {}
        async def act(self, observation, context): return None
        async def learn(self, observation, context, output):
            raise RuntimeError("oops")

    cap = as_capability(_BrokenLearn())
    ctx = _FakeCtx()
    await cap.before_model_request(ctx, "msg")
    # No exception raised — learn failure swallowed
    result = _FakeResult()
    out = await cap.after_run(ctx, result=result)
    assert out is result    # result returned unchanged
```

- [ ] **Step 3: Run — confirm fail**

Expected: `ImportError` for `as_capability`.

- [ ] **Step 4: Implement**

Create `src/ballast/coala/adapters/capability.py`:

```python
"""``as_capability`` — adapt CoALAUnit to BallastCapability."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ballast.capabilities.base import BallastCapability
from ballast.coala._protocol import CoALAUnit

_log = logging.getLogger("ballast.coala.capability")

_OBSERVATION_KEY = "_coala_observation"
_CONTEXT_KEY     = "_coala_context"

GateFn = Callable[[Any], bool | Awaitable[bool]]


def as_capability(
    unit: CoALAUnit, *,
    gate: GateFn | None = None,
) -> BallastCapability:
    """Wrap a CoALAUnit as a pydantic-ai capability for an agent.

    Phase → hook mapping:
      observe + retrieve → ``before_model_request``. Observation +
        context cached on ``ctx.deps`` for later ``learn`` access.

      act → the agent's own ``.iter()`` loop. NOT framework-mediated.
        The agent reasons + calls tools naturally; CoALA's act phase
        IS the agent run from the framework's POV.

      learn → ``after_run``, gated by optional ``gate`` callback.
        Failures inside ``learn`` are swallowed + logged so memory-write
        bugs never block user-facing replies.
    """

    class _CoALACapability(BallastCapability):
        name = f"coala_{type(unit).__name__}"

        async def before_model_request(
            self, ctx: Any, message: Any,
        ) -> Any:
            observation = await unit.observe(message)
            context     = await unit.retrieve(observation)
            _stash(ctx, _OBSERVATION_KEY, observation)
            _stash(ctx, _CONTEXT_KEY, context)
            return message

        async def after_run(
            self, ctx: Any, *, result: Any,
        ) -> Any:
            try:
                if gate is not None:
                    g = gate(result)
                    passed = await g if asyncio.iscoroutine(g) else g
                    if not passed:
                        return result
                observation = _unstash(ctx, _OBSERVATION_KEY)
                context     = _unstash(ctx, _CONTEXT_KEY)
                output      = getattr(result, "output", result)
                await unit.learn(observation, context, output)
            except Exception:
                _log.exception("CoALA learn() failed (swallowed)")
            return result

    return _CoALACapability()


def _stash(ctx: Any, key: str, value: Any) -> None:
    """Store on ctx.deps (works for dict or dataclass-like dep objects)."""
    deps = getattr(ctx, "deps", None)
    if isinstance(deps, dict):
        deps[key] = value
    else:
        # For non-dict deps, fall back to setattr on ctx itself
        setattr(ctx, key, value)


def _unstash(ctx: Any, key: str) -> Any:
    deps = getattr(ctx, "deps", None)
    if isinstance(deps, dict):
        return deps.get(key)
    return getattr(ctx, key, None)


__all__ = ["as_capability"]
```

- [ ] **Step 5: Update package __inits__**

`src/ballast/coala/adapters/__init__.py`:

```python
"""Runtime adapters for CoALAUnit — workflow, tool, capability."""
from ballast.coala.adapters.capability import as_capability
from ballast.coala.adapters.tool import as_tool
from ballast.coala.adapters.workflow import as_workflow

__all__ = ["as_capability", "as_tool", "as_workflow"]
```

`src/ballast/coala/__init__.py`:

```python
"""CoALA Unit Architecture — single Protocol + multiple runtime adapters."""
from ballast.coala._base import CoALABase
from ballast.coala._protocol import CoALAUnit
from ballast.coala.adapters import as_capability, as_tool, as_workflow

__all__ = [
    "CoALABase", "CoALAUnit",
    "as_capability", "as_tool", "as_workflow",
]
```

- [ ] **Step 6: Run — confirm pass**

```
uv run pytest tests/coala/test_capability_adapter.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add src/ballast/coala tests/coala/test_capability_adapter.py
git commit -m "feat(coala): as_capability adapter (observe+retrieve before, learn after — gated, swallowed)"
```

---

### Task B6: Public API re-exports

**Files:**
- Modify: `src/ballast/__init__.py`

- [ ] **Step 1: Add CoALA re-exports**

Find a sensible spot in the imports block (e.g. after the patterns section). Add:

```python
from ballast.coala import (
    CoALABase,
    CoALAUnit,
    as_capability,
    as_tool,
    as_workflow,
)
```

Extend `__all__` with all five names (preserve alphabetical sort).

- [ ] **Step 2: Smoke import**

```
uv run python -c "from ballast import CoALABase, CoALAUnit, as_workflow, as_tool, as_capability; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Run full suite**

```
uv run pytest tests/ -q
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add src/ballast/__init__.py
git commit -m "feat(ballast): re-export CoALA public API"
```

---

### Task B7: notes-app `ResearchSummarize` unit (demo)

**Files:**
- Create: `examples/notes-app/backend/src/notes_app/coala/__init__.py` (empty)
- Create: `examples/notes-app/backend/src/notes_app/coala/research_summarize.py`
- Create: `examples/notes-app/backend/tests/test_research_summarize.py`

- [ ] **Step 1: Write the failing test**

```python
"""``ResearchSummarize`` CoALAUnit — notes-app demo."""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from ballast.coala import CoALAUnit
from notes_app.coala.research_summarize import (
    ResearchObservation, ResearchQuery, ResearchSummarize,
)
from notes_app.repositories.note import InMemoryNoteRepository


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemoryNoteRepository]:
    fresh = InMemoryNoteRepository()
    monkeypatch.setattr("notes_app.repositories.note.notes_repo", fresh)
    yield fresh


def test_satisfies_coala_unit_protocol() -> None:
    assert isinstance(ResearchSummarize(), CoALAUnit)


@pytest.mark.asyncio
async def test_observe_extracts_intent_and_user(
    repo: InMemoryNoteRepository,
) -> None:
    unit = ResearchSummarize()
    obs = await unit.observe(ResearchQuery(user_query="ML in prod"))
    assert isinstance(obs, ResearchObservation)
    assert obs.intent == "ML in prod"


@pytest.mark.asyncio
async def test_retrieve_returns_empty_when_no_matching_notes(
    repo: InMemoryNoteRepository,
) -> None:
    unit = ResearchSummarize()
    obs = await unit.observe(ResearchQuery(user_query="anything"))
    ctx = await unit.retrieve(obs)
    assert ctx.related_notes == []


@pytest.mark.asyncio
async def test_retrieve_returns_search_matches(
    repo: InMemoryNoteRepository,
) -> None:
    await repo.create(title="ml-deployment", body="machine learning in prod")
    await repo.create(title="fashion", body="trends")
    unit = ResearchSummarize()
    obs = await unit.observe(ResearchQuery(user_query="machine learning"))
    ctx = await unit.retrieve(obs)
    titles = {n.title for n in ctx.related_notes}
    assert "ml-deployment" in titles
    assert "fashion" not in titles
```

- [ ] **Step 2: Run — confirm fail**

Expected: `ModuleNotFoundError: notes_app.coala`.

- [ ] **Step 3: Implement the unit**

Create `examples/notes-app/backend/src/notes_app/coala/__init__.py` (empty).

Create `examples/notes-app/backend/src/notes_app/coala/research_summarize.py`:

```python
"""``ResearchSummarize`` CoALAUnit — demo for the CoALA architecture.

Mixed retrieval (search across notes via notes_repo) + custom act
(LLM-free placeholder summary) + custom learn (logs a learning record).

The unit demonstrates the 4-phase contract in a way that exercises:
  observe — parse user query into typed observation
  retrieve — query the relational notes repository
  act — synthesize a summary from retrieved corpus
  learn — record what was synthesized for future reference
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ballast.auth.context import current_user_id
from ballast.coala import CoALABase

from notes_app.models.note import Note


@dataclass
class ResearchQuery:
    user_query: str


@dataclass
class ResearchObservation:
    intent: str
    user_id: str | None


@dataclass
class ResearchContext:
    related_notes: list[Note] = field(default_factory=list)


@dataclass
class ResearchSummary:
    title: str
    body: str


class ResearchSummarize(CoALABase[
    ResearchQuery, ResearchObservation, ResearchContext, ResearchSummary,
]):
    """Summarize the user's recent research on a topic via notes_repo."""

    async def observe(self, q: ResearchQuery) -> ResearchObservation:
        return ResearchObservation(
            intent=q.user_query,
            user_id=current_user_id(),
        )

    async def retrieve(self, obs: ResearchObservation) -> ResearchContext:
        from notes_app.repositories.note import notes_repo  # noqa: PLC0415
        related = await notes_repo.search(obs.intent)
        return ResearchContext(related_notes=related[:10])

    async def act(
        self, obs: ResearchObservation, ctx: ResearchContext,
    ) -> ResearchSummary:
        if not ctx.related_notes:
            return ResearchSummary(
                title=f"No prior research on {obs.intent!r}",
                body="No matching notes found.",
            )
        bullets = "\n".join(
            f"- {n.title}: {n.body[:80]}" for n in ctx.related_notes
        )
        return ResearchSummary(
            title=f"Research: {obs.intent}",
            body=f"Found {len(ctx.related_notes)} prior notes:\n{bullets}",
        )

    async def learn(
        self,
        obs: ResearchObservation,
        ctx: ResearchContext,
        output: ResearchSummary,
    ) -> None:
        # Placeholder — apps would persist to their episode store / vector
        # index / metrics sink here. The demo just logs.
        import logging  # noqa: PLC0415
        logging.getLogger("notes_app.coala").info(
            "research_summarize.learn user=%s intent=%s notes=%d title=%r",
            obs.user_id, obs.intent, len(ctx.related_notes), output.title,
        )
```

- [ ] **Step 4: Run — confirm pass**

```
cd examples/notes-app/backend && uv run pytest tests/test_research_summarize.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/notes-app/backend/src/notes_app/coala examples/notes-app/backend/tests/test_research_summarize.py
git commit -m "feat(notes-app): ResearchSummarize CoALAUnit demo (observe / retrieve / act / learn)"
```

---

### Task B8: notes-app wire `ResearchSummarize` as tool on `NotesAgent`

**Files:**
- Modify: `examples/notes-app/backend/src/notes_app/agents/notes.py`

- [ ] **Step 1: Edit `NotesAgent.build_agent`**

Add the CoALA unit as a tool alongside the existing `@NotesAgent.tool`-decorated functions:

```python
from ballast.coala import as_tool
from notes_app.coala.research_summarize import ResearchSummarize

class NotesAgent(DurableAgent):
    name = "notes"
    metadata_model = None

    def build_agent(self) -> Agent[NoteToolDeps, Any]:
        return Agent(
            model=build_openrouter_model(),
            output_type=[str, DeferredToolRequests],
            deps_type=NoteToolDeps,
            system_prompt=SYSTEM_PROMPT,
            capabilities=default_notes_capabilities(),
            tools=[as_tool(ResearchSummarize())],     # ← CoALAUnit-derived tool
        )
```

- [ ] **Step 2: Smoke run**

```
cd examples/notes-app/backend && uv run pytest -q
```

Expected: green (smoke import works; ResearchSummarize tool registered; existing tools unaffected).

- [ ] **Step 3: Commit**

```bash
git add examples/notes-app/backend/src/notes_app/agents/notes.py
git commit -m "feat(notes-app): NotesAgent exposes ResearchSummarize as a tool via as_tool(unit)"
```

---

### Task B9: Final smoke — full framework + notes-app suites

- [ ] **Step 1: Run framework suite**

```
uv run pytest tests/ --tb=short -q
```

Expected: green. All Phase 1+2 tests gone; new CoALA tests present.

- [ ] **Step 2: Run notes-app suite**

```
cd examples/notes-app/backend && uv run pytest --tb=short -q
```

Expected: green. `test_notes_semantic` gone; `test_research_summarize` present.

- [ ] **Step 3: Smoke import the whole framework**

```
uv run python -c "
from ballast import (
    CoALABase, CoALAUnit, as_workflow, as_tool, as_capability,
    Ballast, BallastSettings,
)
from ballast.auth import Scope, current_user_id, acting_as
print('all imports ok')
"
```

Expected: `all imports ok`.

- [ ] **Step 4: Commit (any cleanup)**

```bash
git status && git diff
# commit any trailing tweaks
```

---

## Follow-up plan (out of scope here)

A separate spec / plan should cover:

1. **MCP-backed CoALA units** — `MCPRetrieveUnit(server_url)` that delegates retrieve() to an MCP tool call; demonstrates 3rd-party integration without touching the framework.
2. **Phase 4 (learning loop)** as a CoALA pattern — apps cluster outputs of `learn()` writes and HITL-suggest a new unit class for the discovered pattern.
3. **Sub-agent as CoALAUnit** — `class MyAgent(CoALABase)` where `act` invokes a sub-agent's `.run()`; full composability.
4. **Goal-drift detection** as a capability that wraps `act` with judge calls — sidecar pattern on top of CoALA.
5. **`scan_context` integration with `ResearchContext.related_notes`** — passing a CoALA `ResearchContext` to a `GroundedAgent` should auto-extract `Ref[Note]` from `.related_notes`. Verify and document.

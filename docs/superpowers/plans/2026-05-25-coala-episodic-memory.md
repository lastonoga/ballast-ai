# CoALA Phase 1 — Episodic Memory + MapReduce Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship CoALA-inspired episodic memory: federation of `EpisodicSource` impls (Thread, Vector) with pluggable `RecallStrategy` (TopK, AllRelevant, MapReduce, Recency, Cluster), dual access (workflow `episodic_for()` push + agent `as_tools()` pull), and tight integration with grounded `Ref[T]` so recall results auto-constrain agent output schemas. Restores `map_reduce_llm` as a reusable framework primitive.

**Architecture:** `EpisodicMemory(sources=[…])` facade fans recall out to each source in parallel; a `RecallStrategy` merges + reduces; `Episode.references: list[Ref[T]]` are aggregated into a `RecallResult` that `scan_context` recognizes — feeding directly into `GroundedAgent` output schemas. `RememberTurn` capability writes turn-summaries back into writable sources (`VectorEpisodicSource`) after `LLMJudge` passes. `MapReduce` is a generic `@Durable.workflow` reused by both `MapReduceStrategy` and (future) long-doc RAG.

**Tech Stack:** Python 3.11+, pydantic v2, SQLModel + Postgres (pgvector), DBOS (`@Durable.workflow`), pydantic-ai (agent tool exposure), existing `Embedder` Protocol.

**Spec:** `docs/superpowers/specs/2026-05-25-coala-episodic-memory-design.md`

---

## File Map

**Framework — new:**
- `src/ballast/patterns/map_reduce/__init__.py` — re-exports `map_reduce_llm`.
- `src/ballast/patterns/map_reduce/pattern.py` — `map_reduce_llm` durable workflow.
- `src/ballast/memory/__init__.py` — top-level re-exports.
- `src/ballast/memory/_scope.py` — `Scope` base BaseModel.
- `src/ballast/memory/episodic/__init__.py` — episodic re-exports.
- `src/ballast/memory/episodic/_models.py` — `DetailLevel`, `Episode`, `ScoredEpisode`, `RecallResult`.
- `src/ballast/memory/episodic/_protocol.py` — `EpisodicSource` Protocol.
- `src/ballast/memory/episodic/_mergers.py` — `ScoreMerger` + `RRFMerger`/`WeightedMerger`/`RawScoreMerger`.
- `src/ballast/memory/episodic/strategies/__init__.py`
- `src/ballast/memory/episodic/strategies/_protocol.py` — `RecallStrategy` Protocol.
- `src/ballast/memory/episodic/strategies/_topk.py` — `TopK` strategy.
- `src/ballast/memory/episodic/strategies/_all_relevant.py` — `AllRelevant` strategy.
- `src/ballast/memory/episodic/strategies/_recency.py` — `Recency` strategy.
- `src/ballast/memory/episodic/strategies/_cluster.py` — `Cluster` strategy.
- `src/ballast/memory/episodic/strategies/_map_reduce.py` — `MapReduce` strategy.
- `src/ballast/memory/episodic/sources/__init__.py`
- `src/ballast/memory/episodic/sources/_thread.py` — `ThreadEpisodicSource`.
- `src/ballast/memory/episodic/sources/_vector.py` — `VectorEpisodicSource` + `EpisodeRow` SQLModel.
- `src/ballast/memory/episodic/_facade.py` — `EpisodicMemory` facade.
- `src/ballast/memory/episodic/_tools.py` — agent-tool factory (`as_tools()`).
- `src/ballast/memory/episodic/_triggers.py` — `RememberTurn` capability.
- `src/ballast/alembic/versions/0003_episodes.py` — `episodes` table migration.

**Framework — modify:**
- `src/ballast/grounded/_scan_context.py` — recognize `RecallResult` + `Episode` during recursion.
- `src/ballast/app.py` — `Ballast.with_memory(memory, scope_builder=None)` setter.
- `src/ballast/__init__.py` — re-export `Memory`, `Episode`, `DetailLevel`, key strategies, `EpisodicMemory`, `Scope`, sources, `RememberTurn`.
- `pyproject.toml` — add `pgvector` dependency (optional extra `memory`).

**Notes-app — modify (smoke):**
- `examples/notes-app/backend/src/notes_app/main.py` — `with_memory(EpisodicMemory(sources=[Thread, Vector]))`.
- `examples/notes-app/backend/src/notes_app/agents/notes.py` — append `RememberTurn(...)` to `default_notes_capabilities()`.
- `examples/notes-app/backend/src/notes_app/workflows/create_note.py` — call `memory.episodic_for(...)` to enrich the approval card payload.

**Tests — new:** mirrored under `tests/memory/`, `tests/memory/strategies/`, `tests/memory/sources/`, `tests/patterns/map_reduce/`, `examples/notes-app/backend/tests/test_create_note_memory.py`.

---

## Task 1: `map_reduce_llm` primitive (restored)

**Files:**
- Create: `src/ballast/patterns/map_reduce/__init__.py`
- Create: `src/ballast/patterns/map_reduce/pattern.py`
- Create: `tests/patterns/map_reduce/__init__.py`
- Create: `tests/patterns/map_reduce/test_map_reduce_llm.py`

- [ ] **Step 1: Write the failing test**

Create `tests/patterns/map_reduce/__init__.py` (empty). Then `tests/patterns/map_reduce/test_map_reduce_llm.py`:

```python
"""``map_reduce_llm`` — parallel per-item map + reduce. Generic primitive."""
from __future__ import annotations

import pytest

from ballast.patterns.map_reduce import map_reduce_llm


@pytest.mark.asyncio
async def test_simple_map_reduce(fresh_dbos_executor: None) -> None:
    """Map doubles each int; reduce sums."""
    async def double(x: int) -> int: return x * 2

    async def sum_all(xs: list[int]) -> int: return sum(xs)

    out = await map_reduce_llm(
        items=[1, 2, 3, 4, 5],
        map_step=double,
        reduce_step=sum_all,
    )
    assert out == 30   # (1+2+3+4+5)*2


@pytest.mark.asyncio
async def test_empty_items_short_circuits(
    fresh_dbos_executor: None,
) -> None:
    """Zero items → reduce called once with []."""
    async def map_fn(_: int) -> int: raise AssertionError("should not run")

    async def reduce_fn(xs: list[int]) -> str: return f"got {len(xs)} items"

    out = await map_reduce_llm(
        items=[], map_step=map_fn, reduce_step=reduce_fn,
    )
    assert out == "got 0 items"


@pytest.mark.asyncio
async def test_collapse_threshold_triggers_recursive_reduce(
    fresh_dbos_executor: None,
) -> None:
    """When mapped output exceeds collapse_threshold, batches are
    reduced before the final reduce."""
    reduce_calls: list[int] = []

    async def passthrough(x: int) -> int: return x

    async def sum_reduce(xs: list[int]) -> int:
        reduce_calls.append(len(xs))
        return sum(xs)

    out = await map_reduce_llm(
        items=list(range(10)),
        map_step=passthrough,
        reduce_step=sum_reduce,
        collapse_threshold=3,
    )
    assert out == sum(range(10))   # final value
    # First batches of 3 reduced, then their results re-reduced.
    assert len(reduce_calls) > 1
```

Add `tests/patterns/map_reduce/conftest.py` mirroring `tests/patterns/conftest.py` for DBOS bootstrap:

```python
"""DBOS bootstrap for map_reduce tests."""
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
    tmp = tempfile.mkdtemp(prefix="dbos-mapreduce-")
    DBOS(config=DBOSConfig(
        name="map-reduce-test",
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

- [ ] **Step 2: Run test — confirm import failure**

```
uv run pytest tests/patterns/map_reduce/test_map_reduce_llm.py -v
```

Expected: `ModuleNotFoundError: No module named 'ballast.patterns.map_reduce'`.

- [ ] **Step 3: Implement**

Create `src/ballast/patterns/map_reduce/pattern.py`:

```python
"""``map_reduce_llm`` — generic parallel map + reduce as @Durable.workflow.

Reused by:
  - ``MapReduceStrategy`` in memory recall (large result sets)
  - Future long-document RAG (per-chunk extract + reduce)
  - Custom apps with embarrassingly-parallel LLM steps
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from ballast.durable import Durable

InT     = TypeVar("InT")
MapT    = TypeVar("MapT")
ReduceT = TypeVar("ReduceT")


@Durable.workflow()
async def map_reduce_llm(
    items: list[InT],
    *,
    map_step:    Callable[[InT], Awaitable[MapT]],
    reduce_step: Callable[[list[MapT]], Awaitable[ReduceT]],
    map_concurrency:    int = 8,
    collapse_threshold: int | None = None,
) -> ReduceT:
    """Map each item in parallel (bounded by ``map_concurrency``),
    then reduce. If ``collapse_threshold`` is set and mapped output
    exceeds it, perform recursive batch-reduce before the final reduce.

    Empty ``items`` short-circuits to ``reduce_step([])`` (single call).
    """
    if not items:
        return await reduce_step([])

    sem = asyncio.Semaphore(map_concurrency)

    async def _bounded(x: InT) -> MapT:
        async with sem:
            return await map_step(x)

    mapped: list[MapT] = await asyncio.gather(*(_bounded(x) for x in items))

    if collapse_threshold is not None and len(mapped) > collapse_threshold:
        batches: list[list[MapT]] = [
            mapped[i:i + collapse_threshold]
            for i in range(0, len(mapped), collapse_threshold)
        ]
        partial: list[MapT] = []
        for batch in batches:
            partial.append(await reduce_step(batch))   # type: ignore[arg-type]
        mapped = partial

    return await reduce_step(mapped)


__all__ = ["map_reduce_llm"]
```

Create `src/ballast/patterns/map_reduce/__init__.py`:

```python
"""``map_reduce_llm`` — generic parallel map+reduce primitive."""
from ballast.patterns.map_reduce.pattern import map_reduce_llm

__all__ = ["map_reduce_llm"]
```

- [ ] **Step 4: Run tests — confirm pass**

```
uv run pytest tests/patterns/map_reduce/ -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/patterns/map_reduce tests/patterns/map_reduce
git commit -m "feat(patterns): restore map_reduce_llm primitive"
```

---

## Task 2: `Scope` base model

**Files:**
- Create: `src/ballast/memory/__init__.py` (empty `__all__ = []` for now)
- Create: `src/ballast/memory/_scope.py`
- Create: `tests/memory/__init__.py` (empty)
- Create: `tests/memory/test_scope.py`

- [ ] **Step 1: Write the failing test**

```python
"""``Scope`` — app-subclassable scope BaseModel with extra=allow."""
from __future__ import annotations

import pytest

from ballast.memory import Scope


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
    # Extra fields readable via attr access (graceful for source consumers).
    assert getattr(s, "project_id", None) == "p-99"


def test_subclass_adds_typed_dimensions() -> None:
    class ProjectScope(Scope):
        project_id: str | None = None

    s = ProjectScope(user_id="u-1", project_id="p-9")
    assert s.project_id == "p-9"
    assert isinstance(s, Scope)
```

- [ ] **Step 2: Run — confirm fail**

```
uv run pytest tests/memory/test_scope.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `Scope`**

Create `src/ballast/memory/_scope.py`:

```python
"""``Scope`` — base scope for memory queries.

Apps subclass to add domain-specific dimensions (project_id, org_id,
team_id, …). ``extra="allow"`` so sources can graceful-read
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

Create `src/ballast/memory/__init__.py`:

```python
"""CoALA-inspired memory subsystem (Phase 1: episodic only)."""
from ballast.memory._scope import Scope

__all__ = ["Scope"]
```

- [ ] **Step 4: Run — confirm pass**

```
uv run pytest tests/memory/test_scope.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory tests/memory
git commit -m "feat(memory): Scope base model (app-subclassable)"
```

---

## Task 3: Episode data models

**Files:**
- Create: `src/ballast/memory/episodic/__init__.py` (re-exports)
- Create: `src/ballast/memory/episodic/_models.py`
- Create: `tests/memory/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
"""Episode wire-contract: DetailLevel, Episode, ScoredEpisode, RecallResult."""
from __future__ import annotations

from datetime import UTC, datetime

from ballast.grounded import Ref
from ballast.memory import Scope
from ballast.memory.episodic import (
    DetailLevel, Episode, RecallResult, ScoredEpisode,
)


def _now() -> datetime: return datetime(2026, 5, 25, tzinfo=UTC)


def test_detail_level_string_enum() -> None:
    assert DetailLevel.PREVIEW.value == "preview"
    assert DetailLevel.SUMMARY.value == "summary"
    assert DetailLevel.FULL.value == "full"
    # Ordering for comparison (>=) — implemented via int conversion.
    assert DetailLevel.SUMMARY >= DetailLevel.PREVIEW
    assert DetailLevel.FULL >= DetailLevel.SUMMARY


def test_episode_minimal() -> None:
    ep = Episode(
        id="thread:abc:turn:0", source="thread",
        occurred_at=_now(), scope=Scope(user_id="u-1"),
        preview="user asked about ML",
    )
    assert ep.preview == "user asked about ML"
    assert ep.summary is None
    assert ep.full is None
    assert ep.references == []


def test_episode_with_references() -> None:
    note_ref = Ref[str](id="n-1")
    ep = Episode(
        id="ep-1", source="vector", occurred_at=_now(),
        scope=Scope(user_id="u-1"), preview="...",
        references=[note_ref],
    )
    assert len(ep.references) == 1


def test_recall_result_references_aggregates() -> None:
    note1, note2, note3 = (Ref[str](id=f"n-{i}") for i in (1, 2, 3))
    ep_a = Episode(
        id="a", source="x", occurred_at=_now(),
        scope=Scope(), preview="p", references=[note1, note2],
    )
    ep_b = Episode(
        id="b", source="x", occurred_at=_now(),
        scope=Scope(), preview="p", references=[note3],
    )
    rr = RecallResult(episodes=[
        ScoredEpisode(episode=ep_a, score=0.9),
        ScoredEpisode(episode=ep_b, score=0.5),
    ])
    refs = rr.references
    assert len(refs) == 3
```

- [ ] **Step 2: Run — confirm fail**

```
uv run pytest tests/memory/test_models.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

Create `src/ballast/memory/episodic/_models.py`:

```python
"""Episode wire-contract: types every source / strategy / consumer speaks."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from ballast.grounded import Ref
from ballast.memory._scope import Scope


class DetailLevel(StrEnum):
    """Hydration level. Comparable: ``FULL >= SUMMARY >= PREVIEW``."""

    PREVIEW = "preview"   # 1-2 lines, cheap, always present
    SUMMARY = "summary"   # paragraph-level
    FULL    = "full"      # complete trajectory (messages + tool calls + outputs)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, DetailLevel):
            return NotImplemented
        order = (DetailLevel.PREVIEW, DetailLevel.SUMMARY, DetailLevel.FULL)
        return order.index(self) >= order.index(other)


class Episode(BaseModel):
    """One unit of recallable past activity, source-agnostic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    source: str
    occurred_at: datetime
    scope: Scope

    preview: str
    summary: str | None = None
    full:    dict[str, Any] | None = None

    references: list[Ref[Any]] = []
    metadata: dict[str, Any] = {}


class ScoredEpisode(BaseModel):
    """Episode + source-relative score (a merger normalizes across sources)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    episode: Episode
    score:   float


class RecallResult(BaseModel):
    """Output of ``EpisodicMemory.episodic_for(...)``.

    ``.references`` aggregates all Ref[T] across all episodes — ready-to-go
    ground-set for ``GroundedAgent`` when this is passed as ``context=[…]``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    episodes: list[ScoredEpisode] = []

    @property
    def references(self) -> list[Ref[Any]]:
        return [r for se in self.episodes for r in se.episode.references]


__all__ = [
    "DetailLevel", "Episode", "RecallResult", "ScoredEpisode",
]
```

Create `src/ballast/memory/episodic/__init__.py`:

```python
"""Episodic memory — federation of EpisodicSource impls."""
from ballast.memory.episodic._models import (
    DetailLevel, Episode, RecallResult, ScoredEpisode,
)

__all__ = ["DetailLevel", "Episode", "RecallResult", "ScoredEpisode"]
```

- [ ] **Step 4: Run — confirm pass**

```
uv run pytest tests/memory/test_models.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory/episodic tests/memory/test_models.py
git commit -m "feat(memory): DetailLevel + Episode + ScoredEpisode + RecallResult"
```

---

## Task 4: `EpisodicSource` Protocol

**Files:**
- Create: `src/ballast/memory/episodic/_protocol.py`
- Modify: `src/ballast/memory/episodic/__init__.py` (add export)
- Create: `tests/memory/test_protocol.py`

- [ ] **Step 1: Write the failing test**

```python
"""``EpisodicSource`` Protocol — structural type check."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ballast.memory import Scope
from ballast.memory.episodic import (
    DetailLevel, Episode, EpisodicSource, ScoredEpisode,
)


class _Stub:
    name = "stub"
    async def recall(self, *, intent, scope, k, detail) -> list[ScoredEpisode]:
        return []
    async def hydrate(self, episode, *, detail) -> Episode:
        return episode
    async def remember(self, episode) -> None:
        return None


def test_runtime_checkable_protocol() -> None:
    assert isinstance(_Stub(), EpisodicSource)


def test_protocol_requires_name_attr() -> None:
    class NoName:
        async def recall(self, *, intent, scope, k, detail): return []
        async def hydrate(self, episode, *, detail): return episode
        async def remember(self, episode) -> None: return None
    # name attribute MUST be present — runtime_checkable doesn't enforce
    # but we document via hasattr.
    assert not hasattr(NoName(), "name")
```

- [ ] **Step 2: Run — confirm fail**

```
uv run pytest tests/memory/test_protocol.py -v
```

Expected: ImportError for `EpisodicSource`.

- [ ] **Step 3: Implement**

Create `src/ballast/memory/episodic/_protocol.py`:

```python
"""``EpisodicSource`` Protocol — one source of episodic facts.

Apps register many sources in ``EpisodicMemory``. The facade fans out
recall in parallel; a ``RecallStrategy`` merges + reduces.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ballast.memory._scope import Scope
from ballast.memory.episodic._models import (
    DetailLevel, Episode, ScoredEpisode,
)


@runtime_checkable
class EpisodicSource(Protocol):
    """Owns ``recall`` / ``hydrate`` / ``remember`` for one backing."""

    name: str

    async def recall(
        self, *,
        intent: str,
        scope: Scope,
        k: int,
        detail: DetailLevel,
    ) -> list[ScoredEpisode]: ...

    async def hydrate(
        self, episode: Episode, *, detail: DetailLevel,
    ) -> Episode: ...

    async def remember(self, episode: Episode) -> None: ...


__all__ = ["EpisodicSource"]
```

Update `src/ballast/memory/episodic/__init__.py`:

```python
"""Episodic memory — federation of EpisodicSource impls."""
from ballast.memory.episodic._models import (
    DetailLevel, Episode, RecallResult, ScoredEpisode,
)
from ballast.memory.episodic._protocol import EpisodicSource

__all__ = [
    "DetailLevel", "Episode", "EpisodicSource", "RecallResult", "ScoredEpisode",
]
```

- [ ] **Step 4: Run — confirm pass**

```
uv run pytest tests/memory/test_protocol.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory/episodic/_protocol.py src/ballast/memory/episodic/__init__.py tests/memory/test_protocol.py
git commit -m "feat(memory): EpisodicSource Protocol"
```

---

## Task 5: `ScoreMerger` Protocol + `RRFMerger`

**Files:**
- Create: `src/ballast/memory/episodic/_mergers.py`
- Modify: `src/ballast/memory/episodic/__init__.py` (export `RRFMerger`)
- Create: `tests/memory/test_mergers.py`

- [ ] **Step 1: Write the failing test**

```python
"""``RRFMerger`` — Reciprocal Rank Fusion, IR-standard cross-source merge."""
from __future__ import annotations

from datetime import UTC, datetime

from ballast.memory import Scope
from ballast.memory.episodic import Episode, ScoredEpisode
from ballast.memory.episodic._mergers import RRFMerger


def _ep(id_: str) -> Episode:
    return Episode(
        id=id_, source="x", occurred_at=datetime.now(UTC),
        scope=Scope(), preview="p",
    )


def test_rrf_merges_two_sources_with_overlap() -> None:
    """Overlapping episode appears once with combined RRF score."""
    a, b, c = _ep("a"), _ep("b"), _ep("c")
    src1 = [ScoredEpisode(episode=a, score=0.9),
            ScoredEpisode(episode=b, score=0.6)]
    src2 = [ScoredEpisode(episode=a, score=0.8),
            ScoredEpisode(episode=c, score=0.4)]

    merged = RRFMerger(k=60).merge({"s1": src1, "s2": src2})
    ids = [se.episode.id for se in merged]
    assert "a" in ids
    assert len(set(ids)) == len(ids)   # dedup'd


def test_rrf_score_higher_for_higher_combined_rank() -> None:
    a = _ep("a"); b = _ep("b")
    src1 = [ScoredEpisode(episode=a, score=0.9),
            ScoredEpisode(episode=b, score=0.5)]
    src2 = [ScoredEpisode(episode=a, score=0.8),
            ScoredEpisode(episode=b, score=0.6)]
    merged = RRFMerger().merge({"s1": src1, "s2": src2})
    by_id = {se.episode.id: se.score for se in merged}
    assert by_id["a"] > by_id["b"]   # both rank-1 in each source > both rank-2


def test_rrf_empty_inputs() -> None:
    assert RRFMerger().merge({}) == []
    assert RRFMerger().merge({"s1": []}) == []
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError for `RRFMerger`.

- [ ] **Step 3: Implement**

Create `src/ballast/memory/episodic/_mergers.py`:

```python
"""Cross-source mergers for federated episodic recall.

Default: ``RRFMerger`` (Reciprocal Rank Fusion) — IR-standard, doesn't
require comparable scores across sources.
"""
from __future__ import annotations

from typing import Protocol

from ballast.memory.episodic._models import Episode, ScoredEpisode


class ScoreMerger(Protocol):
    """Merge per-source ScoredEpisode lists into a single ranked list."""

    def merge(
        self, results_by_source: dict[str, list[ScoredEpisode]],
    ) -> list[ScoredEpisode]: ...


class RRFMerger(ScoreMerger):
    """Reciprocal Rank Fusion: ``score(d) = sum_s 1 / (k + rank_s(d))``.

    Where ``rank_s(d)`` is the rank (1-indexed) of d in source s, or
    infinity if d isn't in source s. ``k=60`` is the canonical RRF
    constant. Dedupes by episode id; episode chosen is the first
    encounter (sources order-independent — we sum contributions).
    """

    def __init__(self, k: int = 60) -> None:
        self._k = k

    def merge(
        self, results_by_source: dict[str, list[ScoredEpisode]],
    ) -> list[ScoredEpisode]:
        agg: dict[str, tuple[Episode, float]] = {}
        for results in results_by_source.values():
            for rank, scored in enumerate(results, start=1):
                ep_id = scored.episode.id
                contribution = 1.0 / (self._k + rank)
                if ep_id in agg:
                    existing_ep, existing_score = agg[ep_id]
                    agg[ep_id] = (existing_ep, existing_score + contribution)
                else:
                    agg[ep_id] = (scored.episode, contribution)
        return sorted(
            (ScoredEpisode(episode=ep, score=sc) for ep, sc in agg.values()),
            key=lambda se: se.score,
            reverse=True,
        )


class WeightedMerger(ScoreMerger):
    """Per-source weighted scores. Apps hint relative trust."""

    def __init__(
        self, weights: dict[str, float], *, normalize: bool = True,
    ) -> None:
        if normalize:
            total = sum(weights.values()) or 1.0
            weights = {k: v / total for k, v in weights.items()}
        self._weights = weights

    def merge(
        self, results_by_source: dict[str, list[ScoredEpisode]],
    ) -> list[ScoredEpisode]:
        agg: dict[str, tuple[Episode, float]] = {}
        for src, results in results_by_source.items():
            w = self._weights.get(src, 0.0)
            for scored in results:
                ep_id = scored.episode.id
                contrib = scored.score * w
                if ep_id in agg:
                    existing_ep, existing_score = agg[ep_id]
                    agg[ep_id] = (existing_ep, existing_score + contrib)
                else:
                    agg[ep_id] = (scored.episode, contrib)
        return sorted(
            (ScoredEpisode(episode=ep, score=sc) for ep, sc in agg.values()),
            key=lambda se: se.score, reverse=True,
        )


class RawScoreMerger(ScoreMerger):
    """Simple union + sort by raw score. Requires score comparability
    across sources — use only when all sources produce calibrated scores."""

    def merge(
        self, results_by_source: dict[str, list[ScoredEpisode]],
    ) -> list[ScoredEpisode]:
        merged: list[ScoredEpisode] = []
        seen: set[str] = set()
        for results in results_by_source.values():
            for scored in results:
                if scored.episode.id not in seen:
                    merged.append(scored)
                    seen.add(scored.episode.id)
        return sorted(merged, key=lambda se: se.score, reverse=True)


__all__ = ["RRFMerger", "RawScoreMerger", "ScoreMerger", "WeightedMerger"]
```

Update `src/ballast/memory/episodic/__init__.py` to add: `from ballast.memory.episodic._mergers import RRFMerger, RawScoreMerger, ScoreMerger, WeightedMerger` and extend `__all__` accordingly.

- [ ] **Step 4: Run — confirm pass**

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory/episodic/_mergers.py src/ballast/memory/episodic/__init__.py tests/memory/test_mergers.py
git commit -m "feat(memory): RRF / Weighted / Raw score mergers"
```

---

## Task 6: `RecallStrategy` Protocol

**Files:**
- Create: `src/ballast/memory/episodic/strategies/__init__.py`
- Create: `src/ballast/memory/episodic/strategies/_protocol.py`
- Create: `tests/memory/strategies/__init__.py` (empty)
- Create: `tests/memory/strategies/test_protocol.py`

- [ ] **Step 1: Write the failing test**

```python
"""``RecallStrategy`` Protocol — structural type for strategy impls."""
from __future__ import annotations

from ballast.memory import Scope
from ballast.memory.episodic import EpisodicSource, RecallResult
from ballast.memory.episodic.strategies import RecallStrategy


class _Stub:
    requires_grounding = False
    async def execute(self, *, intent, sources, scope) -> RecallResult:
        return RecallResult(episodes=[])


def test_runtime_checkable() -> None:
    assert isinstance(_Stub(), RecallStrategy)
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement**

Create `src/ballast/memory/episodic/strategies/_protocol.py`:

```python
"""``RecallStrategy`` — pluggable strategy for recall reduction."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ballast.memory._scope import Scope
from ballast.memory.episodic._models import RecallResult
from ballast.memory.episodic._protocol import EpisodicSource


@runtime_checkable
class RecallStrategy(Protocol):
    """Federates per-source recall into one RecallResult.

    Set ``requires_grounding = True`` if the strategy expects every
    Episode to carry ``references`` (so an empty ``references`` set
    surfaces as a warning rather than silent grounding collapse).
    """

    requires_grounding: bool

    async def execute(
        self, *,
        intent: str,
        sources: list[EpisodicSource],
        scope: Scope,
    ) -> RecallResult: ...


__all__ = ["RecallStrategy"]
```

Create `src/ballast/memory/episodic/strategies/__init__.py`:

```python
"""Recall strategies — pluggable reduction of federated source results."""
from ballast.memory.episodic.strategies._protocol import RecallStrategy

__all__ = ["RecallStrategy"]
```

- [ ] **Step 4: Run — confirm pass**

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory/episodic/strategies tests/memory/strategies
git commit -m "feat(memory): RecallStrategy Protocol"
```

---

## Task 7: `TopK` strategy

**Files:**
- Create: `src/ballast/memory/episodic/strategies/_topk.py`
- Modify: `src/ballast/memory/episodic/strategies/__init__.py` (add export)
- Create: `tests/memory/strategies/test_topk.py`

- [ ] **Step 1: Write the failing test**

```python
"""``TopK`` strategy — parallel query → merge → first K."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import DetailLevel, Episode, ScoredEpisode
from ballast.memory.episodic.strategies import TopK


class _FakeSource:
    def __init__(self, name: str, returns: list[ScoredEpisode]) -> None:
        self.name = name
        self._returns = returns
    async def recall(self, *, intent, scope, k, detail):
        return self._returns
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


def _ep(i: str) -> Episode:
    return Episode(
        id=i, source="x", occurred_at=datetime.now(UTC),
        scope=Scope(), preview="p",
    )


@pytest.mark.asyncio
async def test_topk_returns_top_k_by_merged_score() -> None:
    src1 = _FakeSource("s1", [
        ScoredEpisode(episode=_ep("a"), score=0.9),
        ScoredEpisode(episode=_ep("b"), score=0.4),
    ])
    src2 = _FakeSource("s2", [
        ScoredEpisode(episode=_ep("c"), score=0.8),
    ])
    out = await TopK(k=2).execute(
        intent="x", sources=[src1, src2], scope=Scope(),
    )
    assert len(out.episodes) == 2


@pytest.mark.asyncio
async def test_topk_resilient_to_source_failure() -> None:
    class _Broken:
        name = "broken"
        async def recall(self, **_): raise RuntimeError("down")
        async def hydrate(self, episode, *, detail): return episode
        async def remember(self, episode) -> None: return None

    ok = _FakeSource("ok", [ScoredEpisode(episode=_ep("a"), score=0.9)])
    out = await TopK(k=5).execute(
        intent="x", sources=[_Broken(), ok], scope=Scope(),
    )
    assert len(out.episodes) == 1
    assert out.episodes[0].episode.id == "a"
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement**

Create `src/ballast/memory/episodic/strategies/_topk.py`:

```python
"""``TopK`` recall strategy — classic RAG default."""
from __future__ import annotations

import asyncio
import logging

from ballast.memory._scope import Scope
from ballast.memory.episodic._mergers import RRFMerger, ScoreMerger
from ballast.memory.episodic._models import DetailLevel, RecallResult
from ballast.memory.episodic._protocol import EpisodicSource

_log = logging.getLogger(__name__)


class TopK:
    """Query all sources in parallel, merge, return top K."""

    requires_grounding = False

    def __init__(
        self,
        *,
        k: int = 5,
        detail: DetailLevel = DetailLevel.SUMMARY,
        merger: ScoreMerger | None = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"TopK k must be >= 1, got {k!r}")
        self._k = k
        self._detail = detail
        self._merger = merger or RRFMerger()

    async def execute(
        self, *, intent: str, sources: list[EpisodicSource], scope: Scope,
    ) -> RecallResult:
        async def _safe_recall(src):
            try:
                return src.name, await src.recall(
                    intent=intent, scope=scope, k=self._k, detail=self._detail,
                )
            except Exception:
                _log.exception("episodic source %s recall failed", src.name)
                return src.name, []

        results = await asyncio.gather(*(_safe_recall(s) for s in sources))
        by_source = {name: episodes for name, episodes in results}
        merged = self._merger.merge(by_source)
        return RecallResult(episodes=merged[: self._k])


__all__ = ["TopK"]
```

Update `__init__.py` to add `TopK`.

- [ ] **Step 4: Run — confirm pass**

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory/episodic/strategies/_topk.py src/ballast/memory/episodic/strategies/__init__.py tests/memory/strategies/test_topk.py
git commit -m "feat(memory): TopK recall strategy + source-failure resilience"
```

---

## Task 8: `AllRelevant` + `Recency` strategies

**Files:**
- Create: `src/ballast/memory/episodic/strategies/_all_relevant.py`
- Create: `src/ballast/memory/episodic/strategies/_recency.py`
- Modify: `__init__.py` (exports)
- Create: `tests/memory/strategies/test_all_relevant.py`
- Create: `tests/memory/strategies/test_recency.py`

- [ ] **Step 1: Write failing tests**

`tests/memory/strategies/test_all_relevant.py`:

```python
"""``AllRelevant`` — return everything above a threshold."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import Episode, ScoredEpisode
from ballast.memory.episodic.strategies import AllRelevant


class _FakeSource:
    def __init__(self, returns): self.name = "x"; self._r = returns
    async def recall(self, **_): return self._r
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


def _se(id_: str, score: float) -> ScoredEpisode:
    return ScoredEpisode(
        episode=Episode(id=id_, source="x", occurred_at=datetime.now(UTC),
                        scope=Scope(), preview="p"),
        score=score,
    )


@pytest.mark.asyncio
async def test_all_relevant_filters_by_threshold() -> None:
    src = _FakeSource([_se("a", 0.9), _se("b", 0.4), _se("c", 0.7)])
    out = await AllRelevant(threshold=0.5).execute(
        intent="x", sources=[src], scope=Scope(),
    )
    ids = {se.episode.id for se in out.episodes}
    assert ids == {"a", "c"}
```

`tests/memory/strategies/test_recency.py`:

```python
"""``Recency`` — sort by occurred_at desc; scores ignored."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import Episode, ScoredEpisode
from ballast.memory.episodic.strategies import Recency


class _FakeSource:
    def __init__(self, returns): self.name = "x"; self._r = returns
    async def recall(self, **_): return self._r
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


def _se(id_: str, dt: datetime, score: float) -> ScoredEpisode:
    return ScoredEpisode(
        episode=Episode(id=id_, source="x", occurred_at=dt,
                        scope=Scope(), preview="p"),
        score=score,
    )


@pytest.mark.asyncio
async def test_recency_orders_by_occurred_at_desc() -> None:
    now = datetime.now(UTC)
    src = _FakeSource([
        _se("old", now - timedelta(days=7), 0.9),
        _se("new", now,                     0.1),
        _se("mid", now - timedelta(days=2), 0.5),
    ])
    out = await Recency(n=3).execute(
        intent="x", sources=[src], scope=Scope(),
    )
    assert [se.episode.id for se in out.episodes] == ["new", "mid", "old"]


@pytest.mark.asyncio
async def test_recency_n_caps_results() -> None:
    now = datetime.now(UTC)
    src = _FakeSource([
        _se(f"e-{i}", now - timedelta(days=i), 0.0) for i in range(10)
    ])
    out = await Recency(n=3).execute(intent="x", sources=[src], scope=Scope())
    assert len(out.episodes) == 3
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError both.

- [ ] **Step 3: Implement `AllRelevant`**

`src/ballast/memory/episodic/strategies/_all_relevant.py`:

```python
"""``AllRelevant`` — return all matches above a score threshold."""
from __future__ import annotations

import asyncio
import logging

from ballast.memory._scope import Scope
from ballast.memory.episodic._mergers import RRFMerger, ScoreMerger
from ballast.memory.episodic._models import DetailLevel, RecallResult
from ballast.memory.episodic._protocol import EpisodicSource

_log = logging.getLogger(__name__)


class AllRelevant:
    """Return everything above a threshold — for when context fits."""

    requires_grounding = False

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        detail: DetailLevel = DetailLevel.PREVIEW,
        merger: ScoreMerger | None = None,
        per_source_limit: int = 100,
    ) -> None:
        self._threshold = threshold
        self._detail = detail
        self._merger = merger or RRFMerger()
        self._per_source = per_source_limit

    async def execute(
        self, *, intent: str, sources: list[EpisodicSource], scope: Scope,
    ) -> RecallResult:
        async def _safe(src):
            try:
                return src.name, await src.recall(
                    intent=intent, scope=scope, k=self._per_source,
                    detail=self._detail,
                )
            except Exception:
                _log.exception("episodic source %s recall failed", src.name)
                return src.name, []
        results = await asyncio.gather(*(_safe(s) for s in sources))
        merged = self._merger.merge({n: r for n, r in results})
        return RecallResult(
            episodes=[se for se in merged if se.score >= self._threshold],
        )


__all__ = ["AllRelevant"]
```

- [ ] **Step 4: Implement `Recency`**

`src/ballast/memory/episodic/strategies/_recency.py`:

```python
"""``Recency`` — most-recent N episodes; scores ignored."""
from __future__ import annotations

import asyncio
import logging

from ballast.memory._scope import Scope
from ballast.memory.episodic._models import DetailLevel, RecallResult
from ballast.memory.episodic._protocol import EpisodicSource

_log = logging.getLogger(__name__)


class Recency:
    """Sort federated results by ``occurred_at`` desc; return first N."""

    requires_grounding = False

    def __init__(
        self,
        *,
        n: int = 10,
        detail: DetailLevel = DetailLevel.PREVIEW,
        per_source_limit: int = 50,
    ) -> None:
        self._n = n
        self._detail = detail
        self._per_source = per_source_limit

    async def execute(
        self, *, intent: str, sources: list[EpisodicSource], scope: Scope,
    ) -> RecallResult:
        async def _safe(src):
            try:
                return await src.recall(
                    intent=intent, scope=scope, k=self._per_source,
                    detail=self._detail,
                )
            except Exception:
                _log.exception("episodic source %s recall failed", src.name)
                return []
        per_source = await asyncio.gather(*(_safe(s) for s in sources))
        flat = [se for batch in per_source for se in batch]
        # Dedup by episode id (first encounter wins for ordering stability).
        seen, dedup = set(), []
        for se in flat:
            if se.episode.id in seen: continue
            seen.add(se.episode.id); dedup.append(se)
        dedup.sort(key=lambda se: se.episode.occurred_at, reverse=True)
        return RecallResult(episodes=dedup[: self._n])


__all__ = ["Recency"]
```

Update `__init__.py` exports.

- [ ] **Step 5: Run — confirm pass**

Expected: both green.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/memory/episodic/strategies tests/memory/strategies/test_all_relevant.py tests/memory/strategies/test_recency.py
git commit -m "feat(memory): AllRelevant + Recency strategies"
```

---

## Task 9: `Cluster` strategy

**Files:**
- Create: `src/ballast/memory/episodic/strategies/_cluster.py`
- Modify: `__init__.py` (export)
- Create: `tests/memory/strategies/test_cluster.py`

- [ ] **Step 1: Write the failing test**

```python
"""``Cluster`` — semantic dedup: one medoid per cluster."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import Episode, ScoredEpisode
from ballast.memory.episodic.strategies import Cluster


class _FakeEmbedder:
    """Returns fixed embeddings keyed by episode preview content."""
    _table = {
        "alpha":  [1.0, 0.0],
        "alpha2": [0.95, 0.05],   # very close to alpha
        "beta":   [0.0, 1.0],
        "beta2":  [0.05, 0.95],   # very close to beta
    }
    async def embed(self, text): return self._table[text]
    async def embed_batch(self, texts): return [self._table[t] for t in texts]


class _FakeSource:
    def __init__(self, returns): self.name = "x"; self._r = returns
    async def recall(self, **_): return self._r
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


def _se(preview: str) -> ScoredEpisode:
    return ScoredEpisode(
        episode=Episode(id=preview, source="x", occurred_at=datetime.now(UTC),
                        scope=Scope(), preview=preview),
        score=0.5,
    )


@pytest.mark.asyncio
async def test_cluster_returns_one_per_cluster() -> None:
    src = _FakeSource([_se("alpha"), _se("alpha2"), _se("beta"), _se("beta2")])
    out = await Cluster(n_clusters=2, embedder=_FakeEmbedder()).execute(
        intent="x", sources=[src], scope=Scope(),
    )
    assert len(out.episodes) == 2
    ids = {se.episode.id for se in out.episodes}
    # One representative from each cluster
    assert (("alpha" in ids or "alpha2" in ids)
            and ("beta" in ids or "beta2" in ids))
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement**

`src/ballast/memory/episodic/strategies/_cluster.py`:

```python
"""``Cluster`` strategy — k-means dedup; one medoid per cluster.

Uses ``Embedder`` (existing framework Protocol) to vectorize episode
previews; minimal k-means without external numpy dep — we only need
correctness for small N, not speed.
"""
from __future__ import annotations

import asyncio
import logging
import math
import random

from ballast.capabilities.helpers.embedder import Embedder
from ballast.memory._scope import Scope
from ballast.memory.episodic._models import (
    DetailLevel, RecallResult, ScoredEpisode,
)
from ballast.memory.episodic._protocol import EpisodicSource

_log = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _kmeans_assign(
    points: list[list[float]], k: int, *, max_iter: int = 20, seed: int = 0,
) -> list[int]:
    """Returns cluster-id per point. Distance = 1 - cosine_sim."""
    if k <= 0: return [0] * len(points)
    rng = random.Random(seed)
    centroids = [points[i] for i in rng.sample(range(len(points)), k=min(k, len(points)))]
    assignments = [0] * len(points)
    for _ in range(max_iter):
        new_assignments = [
            min(range(len(centroids)),
                key=lambda c: 1.0 - _cosine(p, centroids[c]))
            for p in points
        ]
        if new_assignments == assignments: break
        assignments = new_assignments
        for c in range(len(centroids)):
            cluster_pts = [points[i] for i, a in enumerate(assignments) if a == c]
            if not cluster_pts: continue
            dim = len(cluster_pts[0])
            centroids[c] = [
                sum(p[d] for p in cluster_pts) / len(cluster_pts) for d in range(dim)
            ]
    return assignments


class Cluster:
    """Semantic dedup — one episode per cluster (medoid)."""

    requires_grounding = False

    def __init__(
        self,
        *,
        n_clusters: int = 5,
        embedder: Embedder,
        detail: DetailLevel = DetailLevel.SUMMARY,
        per_source_limit: int = 50,
    ) -> None:
        self._n = n_clusters
        self._embedder = embedder
        self._detail = detail
        self._per_source = per_source_limit

    async def execute(
        self, *, intent: str, sources: list[EpisodicSource], scope: Scope,
    ) -> RecallResult:
        async def _safe(src):
            try:
                return await src.recall(
                    intent=intent, scope=scope, k=self._per_source,
                    detail=self._detail,
                )
            except Exception:
                _log.exception("episodic source %s recall failed", src.name)
                return []
        per_source = await asyncio.gather(*(_safe(s) for s in sources))
        flat: list[ScoredEpisode] = [se for batch in per_source for se in batch]
        # Dedup by id before clustering.
        seen, dedup = set(), []
        for se in flat:
            if se.episode.id in seen: continue
            seen.add(se.episode.id); dedup.append(se)
        if not dedup:
            return RecallResult(episodes=[])
        embs = await self._embedder.embed_batch([
            se.episode.summary or se.episode.preview for se in dedup
        ])
        assigns = _kmeans_assign(embs, self._n)
        # Pick highest-score representative per cluster.
        reps: dict[int, ScoredEpisode] = {}
        for se, cid in zip(dedup, assigns, strict=True):
            if cid not in reps or se.score > reps[cid].score:
                reps[cid] = se
        return RecallResult(
            episodes=sorted(reps.values(), key=lambda se: se.score, reverse=True),
        )


__all__ = ["Cluster"]
```

Update `__init__.py` to export `Cluster`.

- [ ] **Step 4: Run — confirm pass**

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory/episodic/strategies/_cluster.py src/ballast/memory/episodic/strategies/__init__.py tests/memory/strategies/test_cluster.py
git commit -m "feat(memory): Cluster strategy (k-means dedup)"
```

---

## Task 10: `MapReduce` strategy

**Files:**
- Create: `src/ballast/memory/episodic/strategies/_map_reduce.py`
- Modify: `__init__.py` (export)
- Create: `tests/memory/strategies/test_map_reduce.py`

- [ ] **Step 1: Write the failing test**

```python
"""``MapReduce`` strategy — LLM-driven digest for large result sets."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import Episode, ScoredEpisode
from ballast.memory.episodic.strategies import MapReduce


class _FakeSource:
    def __init__(self, returns): self.name = "x"; self._r = returns
    async def recall(self, **_): return self._r
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


def _se(id_: str) -> ScoredEpisode:
    return ScoredEpisode(
        episode=Episode(id=id_, source="x", occurred_at=datetime.now(UTC),
                        scope=Scope(), preview=f"preview {id_}"),
        score=0.5,
    )


@pytest.mark.asyncio
async def test_map_reduce_strategy_calls_map_and_reduce(
    fresh_dbos_executor: None,
) -> None:
    map_calls: list[str] = []
    reduce_calls: list[int] = []

    async def map_fn(ep: ScoredEpisode) -> str:
        map_calls.append(ep.episode.id)
        return f"M({ep.episode.id})"

    async def reduce_fn(items: list[str]) -> str:
        reduce_calls.append(len(items))
        return ", ".join(items)

    src = _FakeSource([_se(str(i)) for i in range(5)])
    out = await MapReduce(
        max_items=10,
        map_fn=map_fn,
        reduce_fn=reduce_fn,
    ).execute(intent="x", sources=[src], scope=Scope())

    assert sorted(map_calls) == ["0", "1", "2", "3", "4"]
    assert reduce_calls == [5]
    assert len(out.episodes) == 1                              # single synthesized episode
    assert out.episodes[0].episode.preview == "M(0), M(1), M(2), M(3), M(4)"
```

This test needs DBOS — add `tests/memory/strategies/conftest.py` mirroring `tests/patterns/map_reduce/conftest.py` (or import-share).

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement**

`src/ballast/memory/episodic/strategies/_map_reduce.py`:

```python
"""``MapReduce`` strategy — LLM-driven digest of large recall sets.

Builds on ``ballast.patterns.map_reduce.map_reduce_llm``: per-episode
``map_fn`` runs in parallel (typically an LLM call summarizing one
episode); ``reduce_fn`` synthesizes the final digest. The strategy
wraps the digest in a single synthetic Episode whose ``preview`` is
the digest text — preserving the RecallResult contract.

Apps that want structured digest output should make ``reduce_fn``
return a string and pass that to a follow-on agent run.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from ballast.memory._scope import Scope
from ballast.memory.episodic._models import (
    DetailLevel, Episode, RecallResult, ScoredEpisode,
)
from ballast.memory.episodic._protocol import EpisodicSource
from ballast.patterns.map_reduce import map_reduce_llm

_log = logging.getLogger(__name__)


class MapReduce:
    """Federate → top max_items → LLM map+reduce → synthetic Episode."""

    requires_grounding = False

    def __init__(
        self,
        *,
        max_items: int,
        map_fn: Callable[[ScoredEpisode], Awaitable[str]],
        reduce_fn: Callable[[list[str]], Awaitable[str]],
        detail: DetailLevel = DetailLevel.FULL,
        map_concurrency: int = 8,
    ) -> None:
        self._max = max_items
        self._map_fn = map_fn
        self._reduce_fn = reduce_fn
        self._detail = detail
        self._map_concurrency = map_concurrency

    async def execute(
        self, *, intent: str, sources: list[EpisodicSource], scope: Scope,
    ) -> RecallResult:
        async def _safe(src):
            try:
                return await src.recall(
                    intent=intent, scope=scope, k=self._max,
                    detail=self._detail,
                )
            except Exception:
                _log.exception("episodic source %s recall failed", src.name)
                return []
        per_source = await asyncio.gather(*(_safe(s) for s in sources))
        flat: list[ScoredEpisode] = [se for batch in per_source for se in batch][: self._max]
        if not flat:
            return RecallResult(episodes=[])
        digest = await map_reduce_llm(
            items=flat,
            map_step=self._map_fn,
            reduce_step=self._reduce_fn,
            map_concurrency=self._map_concurrency,
        )
        synthesized = Episode(
            id=f"digest:{intent[:32]}",
            source="map-reduce-strategy",
            occurred_at=datetime.now(timezone.utc),
            scope=scope,
            preview=digest,
            summary=digest,
            references=[r for se in flat for r in se.episode.references],
            metadata={"intent": intent, "source_episode_count": len(flat)},
        )
        return RecallResult(episodes=[ScoredEpisode(episode=synthesized, score=1.0)])


__all__ = ["MapReduce"]
```

Update `__init__.py` exports.

- [ ] **Step 4: Run — confirm pass**

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory/episodic/strategies/_map_reduce.py src/ballast/memory/episodic/strategies/__init__.py tests/memory/strategies/test_map_reduce.py tests/memory/strategies/conftest.py
git commit -m "feat(memory): MapReduce strategy (LLM digest of recall set)"
```

---

## Task 11: `ThreadEpisodicSource` (zero new infra)

**Files:**
- Create: `src/ballast/memory/episodic/sources/__init__.py`
- Create: `src/ballast/memory/episodic/sources/_thread.py`
- Create: `tests/memory/sources/__init__.py` (empty)
- Create: `tests/memory/sources/test_thread_source.py`

- [ ] **Step 1: Write the failing test**

```python
"""``ThreadEpisodicSource`` — turns existing thread_repo history into Episodes."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import DetailLevel
from ballast.memory.episodic.sources import ThreadEpisodicSource


class _FakeThreadRepo:
    """Minimal stub — only methods the source touches."""

    def __init__(self, threads, messages):
        self._threads, self._messages = threads, messages

    async def list_for_user(self, user_id, *, limit):
        return [t for t in self._threads if t["user_id"] == user_id][:limit]

    async def all_messages(self, thread_id):
        return [m for m in self._messages if m["thread_id"] == thread_id]


@pytest.mark.asyncio
async def test_thread_source_recall_returns_recent_first() -> None:
    now = datetime.now(UTC)
    repo = _FakeThreadRepo(
        threads=[
            {"id": "t-old", "user_id": "u-1", "created_at": now - timedelta(days=7)},
            {"id": "t-new", "user_id": "u-1", "created_at": now},
        ],
        messages=[
            {"thread_id": "t-old", "role": "user", "text": "old prompt",
             "created_at": now - timedelta(days=7)},
            {"thread_id": "t-old", "role": "assistant", "text": "old reply",
             "created_at": now - timedelta(days=7)},
            {"thread_id": "t-new", "role": "user", "text": "new prompt",
             "created_at": now},
            {"thread_id": "t-new", "role": "assistant", "text": "new reply",
             "created_at": now},
        ],
    )
    src = ThreadEpisodicSource(thread_repo=repo)
    out = await src.recall(
        intent="x", scope=Scope(user_id="u-1"), k=10, detail=DetailLevel.PREVIEW,
    )
    ids = [se.episode.id for se in out]
    assert ids[0].startswith("thread:t-new") and ids[-1].startswith("thread:t-old")


@pytest.mark.asyncio
async def test_thread_source_remember_not_supported() -> None:
    from ballast.memory.episodic import Episode

    src = ThreadEpisodicSource(thread_repo=_FakeThreadRepo([], []))
    ep = Episode(id="x", source="thread", occurred_at=datetime.now(UTC),
                 scope=Scope(), preview="p")
    with pytest.raises(NotImplementedError):
        await src.remember(ep)
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement**

`src/ballast/memory/episodic/sources/_thread.py`:

```python
"""``ThreadEpisodicSource`` — wraps existing thread_repo as episodic memory.

v1 returns recent threads/turns sorted by ``created_at`` desc (no
embedding index). Phase 1.5 adds a vector index over turn previews
if recency proves insufficient.
"""
from __future__ import annotations

from typing import Any, Protocol

from ballast.memory._scope import Scope
from ballast.memory.episodic._models import (
    DetailLevel, Episode, ScoredEpisode,
)


class _ThreadRepo(Protocol):
    async def list_for_user(self, user_id: str, *, limit: int) -> list[Any]: ...
    async def all_messages(self, thread_id: str) -> list[Any]: ...


def _get(o: Any, k: str, default: Any = None) -> Any:
    return getattr(o, k, None) if hasattr(o, k) else (o.get(k, default) if isinstance(o, dict) else default)


class ThreadEpisodicSource:
    """Episodic source backed by the thread repository."""

    name = "thread"

    def __init__(self, *, thread_repo: _ThreadRepo) -> None:
        self._repo = thread_repo

    async def recall(
        self, *, intent: str, scope: Scope, k: int, detail: DetailLevel,
    ) -> list[ScoredEpisode]:
        user = getattr(scope, "user_id", None)
        if user is None: return []
        threads = await self._repo.list_for_user(user, limit=k)
        # Sort newest first.
        threads = sorted(threads, key=lambda t: _get(t, "created_at"), reverse=True)
        out: list[ScoredEpisode] = []
        for t in threads:
            messages = await self._repo.all_messages(_get(t, "id"))
            if not messages: continue
            user_msg = next(
                (m for m in messages if _get(m, "role") == "user"),
                messages[0],
            )
            preview = (_get(user_msg, "text") or "")[:200]
            summary = None
            if detail >= DetailLevel.SUMMARY:
                # Lightweight summary = first user msg + first assistant reply.
                assistant_msg = next(
                    (m for m in messages if _get(m, "role") == "assistant"), None,
                )
                summary = preview + (
                    "\n→ " + (_get(assistant_msg, "text") or "")[:300]
                    if assistant_msg else ""
                )
            full = None
            if detail >= DetailLevel.FULL:
                full = {"messages": messages}
            out.append(ScoredEpisode(
                episode=Episode(
                    id=f"thread:{_get(t, 'id')}",
                    source=self.name,
                    occurred_at=_get(t, "created_at"),
                    scope=scope,
                    preview=preview,
                    summary=summary,
                    full=full,
                    references=[],   # v1: no ref extraction; Phase 1.5
                ),
                score=1.0,           # recency-only — uniform score
            ))
        return out

    async def hydrate(self, episode: Episode, *, detail: DetailLevel) -> Episode:
        if not episode.id.startswith("thread:"):
            raise ValueError(f"{self.name} cannot hydrate id={episode.id}")
        if detail < DetailLevel.SUMMARY: return episode
        thread_id = episode.id.removeprefix("thread:")
        messages = await self._repo.all_messages(thread_id)
        full = {"messages": messages} if detail >= DetailLevel.FULL else episode.full
        return episode.model_copy(update={
            "summary": episode.summary or (messages[0].get("text", "")[:300] if messages else ""),
            "full": full,
        })

    async def remember(self, episode: Episode) -> None:
        raise NotImplementedError(
            "ThreadEpisodicSource is read-only — thread_repo is source-of-truth",
        )


__all__ = ["ThreadEpisodicSource"]
```

Create `src/ballast/memory/episodic/sources/__init__.py`:

```python
"""Built-in episodic sources."""
from ballast.memory.episodic.sources._thread import ThreadEpisodicSource

__all__ = ["ThreadEpisodicSource"]
```

- [ ] **Step 4: Run — confirm pass**

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory/episodic/sources tests/memory/sources/test_thread_source.py
git commit -m "feat(memory): ThreadEpisodicSource (read-only wrapper over thread_repo)"
```

---

## Task 12: `VectorEpisodicSource` + `EpisodeRow` + Alembic 0003

**Files:**
- Create: `src/ballast/memory/episodic/sources/_vector.py` (`EpisodeRow` + `SqlEpisodeRepository` + `VectorEpisodicSource`)
- Create: `src/ballast/alembic/versions/0003_episodes.py`
- Modify: `src/ballast/memory/episodic/sources/__init__.py`
- Modify: `pyproject.toml` (add `pgvector` extra)
- Create: `tests/memory/sources/test_vector_source.py`
- Modify: `tests/persistence/conftest.py` (import `EpisodeRow` so create_all sees it)

- [ ] **Step 1: Write the failing test**

```python
"""``VectorEpisodicSource`` — pgvector-backed semantic recall."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import DetailLevel, Episode
from ballast.memory.episodic.sources import VectorEpisodicSource


class _FakeEmbedder:
    """Returns embeddings keyed by text."""
    _table = {
        "ML in production":  [1.0, 0.0],
        "fashion trends":    [0.0, 1.0],
        "ml deployment":     [0.9, 0.1],
    }
    async def embed(self, text): return self._table[text]
    async def embed_batch(self, texts): return [self._table[t] for t in texts]


@pytest.mark.asyncio
async def test_vector_source_remember_then_recall(
    session_factory,         # from tests/persistence/conftest.py (pg fixture)
) -> None:
    src = VectorEpisodicSource(
        sessionmaker=session_factory, embedder=_FakeEmbedder(),
    )
    ep_ml = Episode(
        id="ep-1", source="vector",
        occurred_at=datetime(2026, 5, 25, tzinfo=UTC),
        scope=Scope(user_id="u-1"),
        preview="ML in production", summary="ML in production",
    )
    ep_fashion = Episode(
        id="ep-2", source="vector",
        occurred_at=datetime(2026, 5, 24, tzinfo=UTC),
        scope=Scope(user_id="u-1"),
        preview="fashion trends", summary="fashion trends",
    )
    await src.remember(ep_ml)
    await src.remember(ep_fashion)

    out = await src.recall(
        intent="ml deployment",
        scope=Scope(user_id="u-1"),
        k=2, detail=DetailLevel.SUMMARY,
    )
    # ml deployment ≈ ML in production (cosine ~0.99), << fashion (cosine ~0)
    assert out[0].episode.id == "ep-1"
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError (auto-skipped if Docker missing — Postgres fixture skips).

- [ ] **Step 3: Add `pgvector` dependency**

In `pyproject.toml` extras (find existing `[tool.uv.dependencies]` or `[project.optional-dependencies]` block; mirror layout):

```toml
[project.optional-dependencies]
memory = [
  "pgvector >= 0.3.0",
]
```

Run `uv sync --extra memory` to install.

- [ ] **Step 4: Implement vector source + repo + model**

`src/ballast/memory/episodic/sources/_vector.py`:

```python
"""``VectorEpisodicSource`` — pgvector-backed semantic recall.

Stores Episode summaries with embedded vectors; cosine-search at recall.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, JSON, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import Field, SQLModel

from ballast.capabilities.helpers.embedder import Embedder
from ballast.memory._scope import Scope
from ballast.memory.episodic._models import (
    DetailLevel, Episode, ScoredEpisode,
)


_EMBED_DIM = 1536   # OpenAI text-embedding-3-small default; configurable per-app
_JSON_PORTABLE = JSONB().with_variant(JSON(), "sqlite")


class EpisodeRow(SQLModel, table=True):
    """SQL row for a stored episode."""

    __tablename__ = "episodes"

    id:          str = Field(primary_key=True)
    source:      str = Field(index=True)
    user_id:     str | None = Field(default=None, index=True)
    tenant_id:   str | None = Field(default=None, index=True)
    thread_id:   str | None = Field(default=None, index=True)
    preview:     str
    summary:     str | None = None
    full:        dict[str, Any] | None = Field(
        default=None, sa_column=Column(_JSON_PORTABLE, nullable=True),
    )
    references_json: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(_JSON_PORTABLE, nullable=False),
    )
    metadata_json: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(_JSON_PORTABLE, nullable=False),
    )
    embedding: list[float] = Field(
        sa_column=Column(Vector(_EMBED_DIM), nullable=False),
    )
    occurred_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )


class VectorEpisodicSource:
    """Episodic source backed by Postgres + pgvector."""

    name = "vector"

    def __init__(
        self, *, sessionmaker: async_sessionmaker[AsyncSession], embedder: Embedder,
    ) -> None:
        self._sm = sessionmaker
        self._embedder = embedder

    async def recall(
        self, *, intent: str, scope: Scope, k: int, detail: DetailLevel,
    ) -> list[ScoredEpisode]:
        query_vec = await self._embedder.embed(intent)
        async with self._sm() as session:
            stmt = (
                select(
                    EpisodeRow,
                    EpisodeRow.embedding.cosine_distance(query_vec).label("dist"),
                )
                .where(EpisodeRow.user_id == getattr(scope, "user_id", None))
            )
            if getattr(scope, "tenant_id", None) is not None:
                stmt = stmt.where(EpisodeRow.tenant_id == scope.tenant_id)
            stmt = stmt.order_by("dist").limit(k)
            rows = (await session.execute(stmt)).all()
        return [
            ScoredEpisode(
                episode=Episode(
                    id=row.id, source=self.name,
                    occurred_at=row.occurred_at,
                    scope=Scope(
                        user_id=row.user_id, tenant_id=row.tenant_id,
                        thread_id=row.thread_id,
                    ),
                    preview=row.preview,
                    summary=row.summary if detail >= DetailLevel.SUMMARY else None,
                    full=row.full if detail >= DetailLevel.FULL else None,
                    references=[],     # references-deserialization in Phase 1.5
                    metadata=row.metadata_json,
                ),
                score=1.0 - float(dist),    # cosine sim
            )
            for row, dist in rows
        ]

    async def hydrate(self, episode: Episode, *, detail: DetailLevel) -> Episode:
        async with self._sm() as session:
            row = await session.get(EpisodeRow, episode.id)
            if row is None:
                return episode
            return episode.model_copy(update={
                "summary": row.summary if detail >= DetailLevel.SUMMARY else episode.summary,
                "full":    row.full    if detail >= DetailLevel.FULL    else episode.full,
            })

    async def remember(self, episode: Episode) -> None:
        embed_text = episode.summary or episode.preview
        vec = await self._embedder.embed(embed_text)
        async with self._sm() as session:
            async with session.begin():
                row = EpisodeRow(
                    id=episode.id, source=episode.source,
                    user_id=getattr(episode.scope, "user_id", None),
                    tenant_id=getattr(episode.scope, "tenant_id", None),
                    thread_id=getattr(episode.scope, "thread_id", None),
                    preview=episode.preview, summary=episode.summary,
                    full=episode.full,
                    references_json=[r.model_dump(mode="json") for r in episode.references],
                    metadata_json=episode.metadata,
                    embedding=vec,
                    occurred_at=episode.occurred_at,
                )
                session.add(row)


__all__ = ["EpisodeRow", "VectorEpisodicSource"]
```

Update `src/ballast/memory/episodic/sources/__init__.py`:

```python
"""Built-in episodic sources."""
from ballast.memory.episodic.sources._thread import ThreadEpisodicSource
from ballast.memory.episodic.sources._vector import (
    EpisodeRow, VectorEpisodicSource,
)

__all__ = ["EpisodeRow", "ThreadEpisodicSource", "VectorEpisodicSource"]
```

- [ ] **Step 5: Write Alembic migration**

`src/ballast/alembic/versions/0003_episodes.py`:

```python
"""create episodes table

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-25
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision      = "0003"
down_revision = "0002"
branch_labels = None
depends_on    = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "episodes",
        sa.Column("id", sa.String(), nullable=False, primary_key=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("tenant_id", sa.String(), nullable=True),
        sa.Column("thread_id", sa.String(), nullable=True),
        sa.Column("preview", sa.String(), nullable=False),
        sa.Column("summary", sa.String(), nullable=True),
        sa.Column("full", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("references_json", sa.dialects.postgresql.JSONB(),
                  nullable=False, server_default="[]"),
        sa.Column("metadata_json", sa.dialects.postgresql.JSONB(),
                  nullable=False, server_default="{}"),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True),
                  nullable=False, index=True),
    )
    op.create_index("ix_episodes_source",    "episodes", ["source"])
    op.create_index("ix_episodes_user_id",   "episodes", ["user_id"])
    op.create_index("ix_episodes_tenant_id", "episodes", ["tenant_id"])
    op.create_index("ix_episodes_thread_id", "episodes", ["thread_id"])
    # IVFFlat index for fast cosine search (created with default lists=100).
    op.execute(
        "CREATE INDEX ix_episodes_embedding_cos ON episodes "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
    )


def downgrade() -> None:
    op.drop_index("ix_episodes_embedding_cos", table_name="episodes")
    op.drop_index("ix_episodes_thread_id", table_name="episodes")
    op.drop_index("ix_episodes_tenant_id", table_name="episodes")
    op.drop_index("ix_episodes_user_id", table_name="episodes")
    op.drop_index("ix_episodes_source", table_name="episodes")
    op.drop_table("episodes")
```

- [ ] **Step 6: Register the model in tests/persistence/conftest.py**

Add: `import ballast.memory.episodic.sources._vector  # noqa: F401` to the conftest's `_register_models()` block (or wherever the imports for `SQLModel.metadata.create_all` live).

Also: pgvector needs the extension. In the conftest's `pg_container` fixture, ensure `init_db` runs `CREATE EXTENSION IF NOT EXISTS vector` before `create_all`. If `pg_container` uses standard postgres image, swap to `pgvector/pgvector:pg16`.

- [ ] **Step 7: Run — confirm pass**

```
uv run pytest tests/memory/sources/test_vector_source.py -v
```

Expected: 1 passed (or skipped if Docker missing).

- [ ] **Step 8: Commit**

```bash
git add src/ballast/memory/episodic/sources/_vector.py src/ballast/memory/episodic/sources/__init__.py src/ballast/alembic/versions/0003_episodes.py tests/memory/sources/test_vector_source.py tests/persistence/conftest.py pyproject.toml
git commit -m "feat(memory): VectorEpisodicSource + pgvector + Alembic 0003"
```

---

## Task 13: `EpisodicMemory` facade

**Files:**
- Create: `src/ballast/memory/episodic/_facade.py`
- Modify: `__init__.py` (export `EpisodicMemory`)
- Create: `tests/memory/test_facade.py`

- [ ] **Step 1: Write the failing test**

```python
"""``EpisodicMemory`` facade — dispatches recall via strategy."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import (
    DetailLevel, Episode, EpisodicMemory, RecallResult, ScoredEpisode,
)
from ballast.memory.episodic.strategies import TopK


class _FakeSource:
    def __init__(self, returns): self.name = "fake"; self._r = returns
    async def recall(self, **_): return self._r
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: self.last_remembered = episode


def _se(id_: str) -> ScoredEpisode:
    return ScoredEpisode(
        episode=Episode(id=id_, source="fake",
                        occurred_at=datetime.now(UTC),
                        scope=Scope(), preview="p"),
        score=0.9,
    )


@pytest.mark.asyncio
async def test_episodic_for_runs_strategy() -> None:
    src = _FakeSource([_se("a"), _se("b")])
    mem = EpisodicMemory(sources=[src], default_strategy=TopK(k=1))
    out = await mem.episodic_for(intent="x")
    assert isinstance(out, RecallResult)
    assert len(out.episodes) == 1


@pytest.mark.asyncio
async def test_default_scope_builder_called_if_no_scope_passed() -> None:
    src = _FakeSource([])
    captured: list[Scope] = []
    class _SpySrc:
        name = "spy"
        async def recall(self, *, intent, scope, k, detail):
            captured.append(scope); return []
        async def hydrate(self, episode, *, detail): return episode
        async def remember(self, episode) -> None: return None
    spy = _SpySrc()
    mem = EpisodicMemory(
        sources=[spy],
        default_scope_builder=lambda: Scope(user_id="from-builder"),
    )
    await mem.episodic_for(intent="x")
    assert captured[0].user_id == "from-builder"


@pytest.mark.asyncio
async def test_remember_fans_out_to_writable_sources() -> None:
    writable = _FakeSource([])
    class _ReadOnly:
        name = "ro"
        async def recall(self, **_): return []
        async def hydrate(self, episode, *, detail): return episode
        async def remember(self, episode) -> None:
            raise NotImplementedError()
    mem = EpisodicMemory(sources=[writable, _ReadOnly()])
    ep = Episode(id="x", source="fake",
                 occurred_at=datetime.now(UTC), scope=Scope(), preview="p")
    await mem.remember(ep)
    assert writable.last_remembered.id == "x"
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError for `EpisodicMemory`.

- [ ] **Step 3: Implement**

`src/ballast/memory/episodic/_facade.py`:

```python
"""``EpisodicMemory`` facade — federation + dual surface (push/pull)."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from ballast.memory._scope import Scope
from ballast.memory.episodic._models import Episode, RecallResult
from ballast.memory.episodic._protocol import EpisodicSource
from ballast.memory.episodic.strategies._protocol import RecallStrategy
from ballast.memory.episodic.strategies._topk import TopK

_log = logging.getLogger(__name__)


class EpisodicMemory:
    """Federation of EpisodicSource impls. Two surfaces:

    - Direct (workflow push):  ``await memory.episodic_for(intent=…)``
    - Tool (agent pull):       ``memory.as_tools()`` returns pydantic-ai tools
    """

    def __init__(
        self,
        sources: list[EpisodicSource],
        *,
        default_strategy: RecallStrategy | None = None,
        default_scope_builder: Callable[[], Scope] | None = None,
    ) -> None:
        if not sources:
            raise ValueError("EpisodicMemory requires at least one source")
        self._sources = sources
        self._default_strategy = default_strategy or TopK()
        self._default_scope_builder = default_scope_builder

    async def episodic_for(
        self,
        *,
        intent: str,
        strategy: RecallStrategy | None = None,
        scope: Scope | None = None,
    ) -> RecallResult:
        used_strategy = strategy or self._default_strategy
        used_scope = (
            scope if scope is not None
            else (
                self._default_scope_builder() if self._default_scope_builder
                else Scope()
            )
        )
        result = await used_strategy.execute(
            intent=intent, sources=self._sources, scope=used_scope,
        )
        if getattr(used_strategy, "requires_grounding", False) and not result.references:
            _log.warning(
                "episodic recall(intent=%r) returned 0 references but strategy "
                "requires_grounding=True — output_type with Ref[T] will fail",
                intent,
            )
        return result

    async def remember(self, episode: Episode) -> None:
        async def _safe(src):
            try:
                await src.remember(episode)
            except NotImplementedError:
                pass     # read-only source — silent
            except Exception:
                _log.exception("episodic source %s remember failed", src.name)
        await asyncio.gather(*(_safe(s) for s in self._sources))

    def as_tools(self) -> list:
        from ballast.memory.episodic._tools import build_recall_tool  # noqa: PLC0415
        return [build_recall_tool(self)]


__all__ = ["EpisodicMemory"]
```

Update `__init__.py` exports to add `EpisodicMemory`.

- [ ] **Step 4: Run — confirm pass**

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory/episodic/_facade.py src/ballast/memory/episodic/__init__.py tests/memory/test_facade.py
git commit -m "feat(memory): EpisodicMemory facade with dual surface (push/pull)"
```

---

## Task 14: `as_tools()` — agent pull surface

**Files:**
- Create: `src/ballast/memory/episodic/_tools.py`
- Create: `tests/memory/test_tools.py`

- [ ] **Step 1: Write the failing test**

```python
"""``EpisodicMemory.as_tools()`` — pydantic-ai-compatible recall tool."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import Episode, EpisodicMemory, ScoredEpisode


class _FakeSource:
    name = "fake"
    def __init__(self, returns): self._r = returns
    async def recall(self, **_): return self._r
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


@pytest.mark.asyncio
async def test_as_tools_returns_one_recall_tool() -> None:
    src = _FakeSource([ScoredEpisode(
        episode=Episode(id="a", source="fake",
                        occurred_at=datetime.now(UTC),
                        scope=Scope(), preview="hello"),
        score=0.9,
    )])
    mem = EpisodicMemory(sources=[src])
    tools = mem.as_tools()
    assert len(tools) == 1
    # The tool function should be async and accept (intent, k)
    out = await tools[0].function(intent="x", k=3)
    # Returns a list of dicts (JSON-serializable summary) for agent inspection.
    assert isinstance(out, list) and out[0]["preview"] == "hello"
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement**

`src/ballast/memory/episodic/_tools.py`:

```python
"""Agent pull surface — exposes ``EpisodicMemory`` as a pydantic-ai tool."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import Tool

if TYPE_CHECKING:
    from ballast.memory.episodic._facade import EpisodicMemory


def build_recall_tool(memory: "EpisodicMemory") -> Tool:
    """Return a pydantic-ai Tool wrapping ``memory.episodic_for``."""

    async def recall_episodes(
        intent: str,
        k: int = 5,
    ) -> list[dict]:
        """Recall episodes from your past activity that are relevant to
        the given intent. Returns up to ``k`` episodes with id / preview /
        occurred_at. Use this when the user references prior work or
        when you suspect you've handled a similar task before."""
        from ballast.memory.episodic.strategies._topk import TopK  # noqa: PLC0415
        result = await memory.episodic_for(intent=intent, strategy=TopK(k=k))
        return [
            {
                "id":          se.episode.id,
                "source":      se.episode.source,
                "preview":     se.episode.preview,
                "summary":     se.episode.summary,
                "occurred_at": se.episode.occurred_at.isoformat(),
                "score":       se.score,
            }
            for se in result.episodes
        ]

    return Tool(recall_episodes, takes_ctx=False)


__all__ = ["build_recall_tool"]
```

- [ ] **Step 4: Run — confirm pass**

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory/episodic/_tools.py tests/memory/test_tools.py
git commit -m "feat(memory): as_tools() — pydantic-ai recall tool for agent pull surface"
```

---

## Task 15: `RememberTurn` capability

**Files:**
- Create: `src/ballast/memory/episodic/_triggers.py`
- Modify: `src/ballast/memory/episodic/__init__.py` (export)
- Create: `tests/memory/test_remember_turn.py`

- [ ] **Step 1: Find the existing capability base + after_run hook**

```
grep -n "class BallastCapability\|async def after_run" src/ballast/capabilities/base.py
```

Mirror an existing capability (e.g. `BudgetGuard` or `JudgeAfterRun`) for hook shape.

- [ ] **Step 2: Write the failing test**

```python
"""``RememberTurn`` — capability that writes episodes after successful turns."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ballast.memory import Scope
from ballast.memory.episodic import (
    Episode, EpisodicMemory, RememberTurn, ScoredEpisode,
)


class _FakeSource:
    name = "fake"
    def __init__(self): self.remembered: list[Episode] = []
    async def recall(self, **_): return []
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode): self.remembered.append(episode)


class _RunResult:
    """Minimal stub for pydantic-ai RunResult / RunContext."""
    def __init__(self, user="hi", assistant="hello"): 
        self.user, self.assistant = user, assistant


class _RunCtx:
    def __init__(self, user_id="u-1", thread_id="t-1"):
        class _Deps: parent_thread_id = thread_id
        self.deps = _Deps()


@pytest.mark.asyncio
async def test_remember_turn_writes_on_pass() -> None:
    src = _FakeSource()
    mem = EpisodicMemory(sources=[src])
    cap = RememberTurn(memory=mem, gate=lambda *_: True)
    await cap.after_run(_RunCtx(), _RunResult())
    assert len(src.remembered) == 1
    assert src.remembered[0].preview != ""


@pytest.mark.asyncio
async def test_remember_turn_skips_when_gate_fails() -> None:
    src = _FakeSource()
    mem = EpisodicMemory(sources=[src])
    cap = RememberTurn(memory=mem, gate=lambda *_: False)
    await cap.after_run(_RunCtx(), _RunResult())
    assert src.remembered == []
```

- [ ] **Step 3: Implement**

`src/ballast/memory/episodic/_triggers.py`:

```python
"""``RememberTurn`` — capability that writes episodes after successful turns.

Default gate: always-True (apps wire a callable returning ``False`` to
skip, e.g. ``gate=lambda ctx, result: judge_passed(result)`` — typical
integration with LLMJudge).
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ballast.capabilities.base import BallastCapability
from ballast.memory._scope import Scope
from ballast.memory.episodic._facade import EpisodicMemory
from ballast.memory.episodic._models import Episode

_log = logging.getLogger(__name__)


GateFn = Callable[[Any, Any], bool | Awaitable[bool]]


class RememberTurn(BallastCapability):
    """After each agent run, if the gate passes, summarize + persist."""

    def __init__(
        self,
        *,
        memory: EpisodicMemory,
        gate: GateFn | None = None,
        summarizer: Callable[[Any, Any], Awaitable[str]] | None = None,
    ) -> None:
        super().__init__()
        self._memory = memory
        self._gate = gate or (lambda *_: True)
        self._summarizer = summarizer or self._default_summarizer

    @staticmethod
    async def _default_summarizer(ctx: Any, result: Any) -> str:
        u = getattr(result, "user", "") or ""
        a = getattr(result, "assistant", "") or ""
        return f"User: {u[:200]}\nAssistant: {a[:300]}"

    async def after_run(self, ctx: Any, result: Any) -> None:
        try:
            gate_out = self._gate(ctx, result)
            passed = await gate_out if hasattr(gate_out, "__await__") else gate_out
            if not passed:
                return
            summary = await self._summarizer(ctx, result)
            ep = Episode(
                id=str(uuid4()),
                source="remember-turn",
                occurred_at=datetime.now(timezone.utc),
                scope=Scope(
                    user_id=getattr(getattr(ctx, "deps", None), "user_id", None),
                    thread_id=getattr(getattr(ctx, "deps", None), "parent_thread_id", None),
                ),
                preview=summary[:200],
                summary=summary,
            )
            await self._memory.remember(ep)
        except Exception:
            _log.exception("RememberTurn after_run failed (swallowed)")


__all__ = ["RememberTurn"]
```

If `BallastCapability` doesn't have an `after_run` slot, copy the closest sibling pattern (e.g. `JudgeAfterRun`).

Update `__init__.py` exports.

- [ ] **Step 4: Run — confirm pass**

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory/episodic/_triggers.py src/ballast/memory/episodic/__init__.py tests/memory/test_remember_turn.py
git commit -m "feat(memory): RememberTurn capability (gated auto-write)"
```

---

## Task 16: `scan_context` integration with `Episode` / `RecallResult`

**Files:**
- Modify: `src/ballast/grounded/_scan_context.py`
- Create: `tests/memory/test_grounded_integration.py`

- [ ] **Step 1: Locate scan_context**

```
grep -n "def scan_context\|scan_context\b" src/ballast/grounded/
```

- [ ] **Step 2: Write the failing integration test**

```python
"""``scan_context`` recognizes ``RecallResult`` + ``Episode`` and unwraps refs."""
from __future__ import annotations

from datetime import UTC, datetime

from ballast.grounded import Ref
from ballast.grounded._scan_context import scan_context
from ballast.memory import Scope
from ballast.memory.episodic import Episode, RecallResult, ScoredEpisode


class _Note: pass     # marker type used as Ref[_Note] target


def _ep(refs):
    return Episode(
        id="ep", source="x", occurred_at=datetime.now(UTC),
        scope=Scope(), preview="p", references=refs,
    )


def test_scan_context_collects_refs_from_recall_result() -> None:
    n1, n2 = Ref[_Note](id="n-1"), Ref[_Note](id="n-2")
    rr = RecallResult(episodes=[
        ScoredEpisode(episode=_ep([n1, n2]), score=0.9),
    ])
    collected = scan_context([rr])
    # _Note → {"n-1", "n-2"}
    assert _Note in collected
    assert set(collected[_Note]) == {"n-1", "n-2"}


def test_scan_context_collects_refs_from_loose_episode() -> None:
    n1 = Ref[_Note](id="n-3")
    ep = _ep([n1])
    collected = scan_context([ep])
    assert _Note in collected and collected[_Note] == ["n-3"]
```

- [ ] **Step 3: Run — confirm fail**

Expected: assertion failures (scan_context doesn't yet know about Episode/RecallResult).

- [ ] **Step 4: Patch `scan_context`**

In `src/ballast/grounded/_scan_context.py`, find the recursion (a place that descends into `BaseModel` instances and collects `Ref[T]`). Add:

```python
# at the top of the descent function
from ballast.memory.episodic._models import Episode, RecallResult

# in the descent body, BEFORE generic BaseModel recursion:
if isinstance(value, RecallResult):
    for ref in value.references:
        _collect(ref)
    return

if isinstance(value, Episode):
    for ref in value.references:
        _collect(ref)
    return
```

The exact integration point depends on existing code shape — keep it minimal (~6-10 lines).

If there's a circular-import risk, do the imports lazily inside the descent function.

- [ ] **Step 5: Run — confirm pass**

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/grounded/_scan_context.py tests/memory/test_grounded_integration.py
git commit -m "feat(grounded): scan_context recognizes RecallResult + Episode"
```

---

## Task 17: `Ballast.with_memory(...)` builder

**Files:**
- Modify: `src/ballast/app.py`
- Create: `tests/app/test_with_memory.py`

- [ ] **Step 1: Mirror pattern from `with_approval_repo` / `with_judge_defaults`**

```
grep -n "def with_approval_repo\|def with_judge_defaults" src/ballast/app.py
```

- [ ] **Step 2: Write the failing test**

```python
"""``Ballast.with_memory`` — fluent setter that installs EpisodicMemory."""
from __future__ import annotations

from ballast.app import Ballast
from ballast.memory import Scope
from ballast.memory.episodic import EpisodicMemory


class _FakeSource:
    name = "f"
    async def recall(self, **_): return []
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


def test_with_memory_installs_facade() -> None:
    mem = EpisodicMemory(sources=[_FakeSource()])
    app = Ballast().with_memory(mem)
    # Verify the engine carries the memory facade
    assert app._memory is mem  # internal attr OK — test pins the contract


def test_with_memory_scope_builder() -> None:
    mem = EpisodicMemory(sources=[_FakeSource()])
    builder = lambda: Scope(user_id="from-test")
    app = Ballast().with_memory(mem, scope_builder=builder)
    assert app._scope_builder is builder
```

- [ ] **Step 3: Implement**

In `src/ballast/app.py`, add (mirroring `with_approval_repo` shape):

```python
from collections.abc import Callable

if TYPE_CHECKING:
    from ballast.memory._scope import Scope
    from ballast.memory.episodic._facade import EpisodicMemory

# In Ballast.__init__:
self._memory: "EpisodicMemory | None" = None
self._scope_builder: "Callable[[], Scope] | None" = None

def with_memory(
    self,
    memory: "EpisodicMemory",
    *,
    scope_builder: "Callable[[], Scope] | None" = None,
) -> "Ballast":
    """Wire an EpisodicMemory facade + optional default scope-builder.

    The scope_builder is called by ``memory.episodic_for(...)`` when
    no explicit scope is passed — typically reads ambient ContextVars
    (user_id, tenant_id, project_id, …).
    """
    self._memory = memory
    self._scope_builder = scope_builder
    # Re-wire the facade's default builder if provided here.
    if scope_builder is not None:
        memory._default_scope_builder = scope_builder
    return self
```

- [ ] **Step 4: Run — confirm pass**

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/app.py tests/app/test_with_memory.py
git commit -m "feat(app): Ballast.with_memory() fluent setter"
```

---

## Task 18: Public API exports

**Files:**
- Modify: `src/ballast/__init__.py`

- [ ] **Step 1: Add re-exports**

```python
# at the imports block
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
    AllRelevant,
    Cluster,
    MapReduce as MapReduceStrategy,    # disambiguate from patterns.map_reduce
    Recency,
    RecallStrategy,
    TopK,
)
from ballast.patterns.map_reduce import map_reduce_llm
```

Add all of these to `__all__`.

- [ ] **Step 2: Verify nothing broke**

```
uv run pytest tests/ -q
```

Expected: green (full framework suite).

- [ ] **Step 3: Commit**

```bash
git add src/ballast/__init__.py
git commit -m "feat(ballast): re-export memory + map_reduce public API"
```

---

## Task 19: Notes-app — wire memory in main.py

**Files:**
- Modify: `examples/notes-app/backend/src/notes_app/main.py`

- [ ] **Step 1: Edit `main.py`**

Add imports near the other ballast imports:

```python
from ballast.memory import Scope
from ballast.memory.episodic import (
    EpisodicMemory, RememberTurn, ThreadEpisodicSource,
    VectorEpisodicSource,
)
from ballast.auth.context import current_user_id
```

Wire memory inside the builder chain (after `with_approval_repo`):

```python
def _build_memory():
    # VectorEpisodicSource requires Postgres + sessionmaker; only enable
    # in the SQL path so test/in-memory mode degrades gracefully.
    sources = [ThreadEpisodicSource(thread_repo=ballast_thread_repo)]
    if _should_use_sql():
        sources.append(VectorEpisodicSource(
            sessionmaker=_pg_sessionmaker,
            embedder=_openai_embedder,
        ))
    return EpisodicMemory(
        sources=sources,
        default_scope_builder=lambda: Scope(user_id=current_user_id()),
    )

ballast = (
    Ballast()
    .with_judge_defaults(...)
    .with_approval_repo(...)
    .with_memory(_build_memory())
    ...
)
```

If `_should_use_sql()` / `_pg_sessionmaker` don't already exist with those names, mirror the pattern from `SqlApprovalCardRepository` wiring (Task PG T1 / commit `98f3f79c`). Either reuse or introduce the helper inline.

For embedder: instantiate a thin wrapper around OpenAI's embedding API if not already wired (e.g. `pydantic_ai.providers.openai.OpenAIEmbedder` or roll a 10-line wrapper around `openai.AsyncOpenAI().embeddings.create`). Document fallback to `None` if `OPENAI_API_KEY` missing → vector source skipped entirely.

- [ ] **Step 2: Smoke run**

```
cd examples/notes-app/backend && uv run pytest -q
```

Expected: green.

- [ ] **Step 3: Commit**

```bash
git add examples/notes-app/backend/src/notes_app/main.py
git commit -m "feat(notes-app): wire EpisodicMemory (Thread + optional Vector)"
```

---

## Task 20: Notes-app — `RememberTurn` capability

**Files:**
- Modify: `examples/notes-app/backend/src/notes_app/agents/notes.py`

- [ ] **Step 1: Edit `default_notes_capabilities()`**

In `examples/notes-app/backend/src/notes_app/agents/notes.py`, append `RememberTurn`:

```python
from ballast import get_ballast
from ballast.memory.episodic import RememberTurn

def default_notes_capabilities() -> list[BallastCapability]:
    memory = get_ballast()._memory   # facade installed via Ballast.with_memory
    caps: list[BallastCapability] = [
        BudgetGuard(...),
        PIIGuard(...),
        JudgeAfterRun(...),
    ]
    if memory is not None:
        caps.append(RememberTurn(
            memory=memory,
            gate=lambda ctx, result: True,   # write every turn for now;
                                              # later: judge-gated
        ))
    return caps
```

Touching `get_ballast()._memory` is reading a private attr — fine for the example app but worth a follow-up to add a public `get_ballast().memory` accessor (out of scope here; mark as TODO in a comment).

- [ ] **Step 2: Smoke run**

```
cd examples/notes-app/backend && uv run pytest -q
```

Expected: green.

- [ ] **Step 3: Commit**

```bash
git add examples/notes-app/backend/src/notes_app/agents/notes.py
git commit -m "feat(notes-app): RememberTurn capability on NotesAgent"
```

---

## Task 21: Notes-app — recall in `create_note_flow`

**Files:**
- Modify: `examples/notes-app/backend/src/notes_app/workflows/create_note.py`

- [ ] **Step 1: Use `memory.episodic_for(...)` to fetch similar past notes**

Edit `create_note_flow` to enrich the card payload:

```python
from ballast import get_ballast
from ballast.memory.episodic.strategies import TopK
from ballast.memory.episodic import DetailLevel

@Durable.workflow()
async def create_note_flow(draft: ProposedNote) -> Note | None:
    # Optional memory enrichment — if memory wired, fetch related episodes.
    memory = get_ballast()._memory
    similar_summaries: list[str] = []
    if memory is not None:
        try:
            recall = await memory.episodic_for(
                intent=f"prior notes about {draft.title}",
                strategy=TopK(k=3, detail=DetailLevel.PREVIEW),
            )
            similar_summaries = [
                se.episode.preview for se in recall.episodes
            ]
        except Exception:
            pass     # memory failures shouldn't block the save flow

    # ... existing channel.request(draft) flow ...
    verdict = await _channel.request(draft)
    if verdict.decision != "approve":
        return None
    final = verdict.modified or draft
    return await notes_repo.create(
        title=final.title, body=final.body,
    )
```

For Phase 1 we don't (yet) plumb `similar_summaries` into the card UI — that's a UX iteration. Captured locally + logged so a manual smoke can see them in the logs.

Add at least an `_log.info("recall returned %d similar", len(similar_summaries))` so manual smoke can confirm.

- [ ] **Step 2: Smoke run**

```
cd examples/notes-app/backend && uv run pytest -q
```

Expected: green.

- [ ] **Step 3: Commit**

```bash
git add examples/notes-app/backend/src/notes_app/workflows/create_note.py
git commit -m "feat(notes-app): recall similar past notes in create_note_flow"
```

---

## Task 22: Final smoke — full framework + notes-app suites

- [ ] **Step 1: Run framework suite**

```
uv run pytest tests/ --tb=short -q
```

Expected: green (all memory + map_reduce tests passing alongside existing).

- [ ] **Step 2: Run notes-app suite**

```
cd examples/notes-app/backend && uv run pytest --tb=short -q
```

Expected: green.

- [ ] **Step 3: Manual browser/CLI smoke**

If Postgres + OPENAI_API_KEY available:

```
cd examples/notes-app/backend && uv run uvicorn notes_app.main:app --reload &
cd examples/notes-app/frontend && pnpm dev &
```

In the chat:

1. Ask the agent to save 3 different notes (different topics).
2. Verify each saved.
3. Open a NEW thread.
4. Ask "сохрани заметку про X" where X matches one of the prior topics.
5. Inspect backend logs — should see `recall returned N similar` from `create_note_flow` mentioning the prior note's preview.

If Postgres NOT available — at least confirm Thread source recall works:

```
cd examples/notes-app/backend && uv run python -c "
import asyncio
from ballast import get_ballast
async def main():
    mem = get_ballast()._memory
    r = await mem.episodic_for(intent='test')
    print('recall episodes:', len(r.episodes))
asyncio.run(main())
"
```

- [ ] **Step 4: Commit (any cleanup)**

```bash
git status && git diff
# commit any tweaks
```

---

## Follow-up plan (out of scope here)

A separate spec covers:

1. **Phase 1.5** — embedding-indexed `ThreadEpisodicSource` (if recency proves insufficient in demos).
2. **Phase 2 — Semantic memory**: `SemanticSource` Protocol + `@memory_tool` decorator to expose domain repos; `DomainSemanticSource` ABC.
3. **Phase 3 — Procedural memory**: `WorkflowRegistry` with introspection; `as_tools()` exposes registered workflows as skills.
4. **Phase 4 — Learning loop**: clustering of recent episodes → HITL-suggested consolidation into a procedural skill via UICardChannel.
5. **MCP-backed sources**: `MCPEpisodicSource(mcp_server)` for Linear / GitHub / Notion past-activity recall.
6. **Public `get_ballast().memory` accessor** to replace `_memory` private access.
7. **Card UI enrichment**: surface `similar_summaries` from `create_note_flow` in the approval card.

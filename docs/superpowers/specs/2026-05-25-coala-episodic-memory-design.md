# CoALA Memory — Phase 1: Episodic Memory + MapReduce Primitive

**Date:** 2026-05-25
**Status:** Approved (brainstorming) → ready for implementation plan
**Author:** Kir + Claude
**Scope:** Phase 1 of a 4-phase CoALA-inspired memory subsystem. Phases 2-4 are deferred (see "Out of scope").

## Problem

Agents in Ballast today are stateless across thread boundaries. Each new
thread starts cold; no transfer of experience ("when did I solve a similar
task before?"). Long sessions hit context-window limits; old turns get
lost rather than summarized into recallable memory. There is no first-class
notion of *memory* — only `ThreadRepository.all_messages()` returning
linear history of the current thread.

This blocks:
- **Cross-thread recall** — "I helped this user with X last week — what did
  we settle on?"
- **Learning from outcomes** — successful trajectories are forgotten the
  moment the thread closes.
- **Bounded context growth** — older turns either bloat the prompt or
  silently drop off.

The article on production AI agent architecture (CoALA + supporting
patterns) prescribes a four-module memory architecture (working /
episodic / semantic / procedural) with explicit retrieval APIs. This
spec covers **episodic memory only**, as the foundation for the rest.

## Core insights (from brainstorming)

Five non-obvious decisions shape the design:

1. **Memory is a federation of sources, not a storage primitive.**
   Episodes can come from many places (raw thread history, vector store
   of summaries, business tables read as episodes, 3rd-party systems via
   MCP). Each source is an `EpisodicSource` impl; the framework merges
   results.

2. **Dual surface — push (workflow) AND pull (agent tool).** A workflow
   knows the current step's context and can pre-fetch relevant memory
   deterministically (`await memory.episodic_for(...)`); an agent in
   open-ended chat gets the same memory as a callable tool
   (`memory.as_tools()`). Same backing store; two access patterns.

3. **Detail level is per-strategy, not per-episode.** Some recalls want
   1-line previews; others want full message traces. `RecallStrategy`
   carries a `detail: DetailLevel` and asks each source for that level.

4. **Recall strategy is pluggable.** Top-K is the default for RAG-style
   agent prompting, but `AllRelevant`, `Recency`, `Cluster`, and
   `MapReduce` (LLM-driven reduction for large result sets) cover
   distinct use cases. Apps can write custom strategies.

5. **Memory recall feeds grounded reasoning.** Each `Episode` carries
   `references: list[Ref[T]]` to domain entities. Passing a
   `RecallResult` as context to a `GroundedAgent` constrains the model's
   output schema to those entities — the model physically cannot
   hallucinate UUIDs that weren't recalled.

## Design

### 1. Episode wire-contract

```python
# src/ballast/memory/episodic/_models.py
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from ballast.grounded import Ref


class DetailLevel(StrEnum):
    PREVIEW = "preview"   # 1-2 lines — always available
    SUMMARY = "summary"   # paragraph-level
    FULL    = "full"      # complete trajectory (messages + tool calls + outputs)


class Episode(BaseModel):
    """One unit of recallable past activity, source-agnostic."""

    id: str                                  # source-namespaced id
    source: str                              # "thread" / "vector" / "linear" / ...
    occurred_at: datetime
    scope: "Scope"

    preview: str                             # cheap, always present
    summary: str | None = None               # populated if detail >= SUMMARY
    full:    dict[str, Any] | None = None    # source-specific shape, present if detail == FULL

    references: list[Ref[Any]] = []          # typed pointers into domain entities
    metadata: dict[str, Any] = {}            # source-specific extras


class ScoredEpisode(BaseModel):
    episode: Episode
    score: float                             # source-relative; merger normalizes


class RecallResult(BaseModel):
    episodes: list[ScoredEpisode]

    @property
    def references(self) -> list[Ref[Any]]:
        """Union of all Ref[T] across all returned episodes — ready ground-set."""
        return [r for e in self.episodes for r in e.episode.references]
```

### 2. Scope — app-subclassable

```python
# src/ballast/memory/_scope.py
from pydantic import BaseModel, ConfigDict


class Scope(BaseModel):
    """Base scope. Apps subclass to add dimensions (project_id, org_id, …).

    `extra="allow"` so sources can read app-custom fields via getattr
    without ceremony.
    """

    model_config = ConfigDict(extra="allow")

    user_id:   str | None = None
    tenant_id: str | None = None
    thread_id: str | None = None
```

App example:

```python
# notes_app/memory/scope.py
class NotesScope(Scope):
    project_id: str | None = None
    folder_id:  str | None = None
```

ContextVars per dimension live alongside the existing
`ballast.auth.context.current_user_id`. Apps register their own
(`current_project_id`) and supply a `scope_builder` callable to the
Ballast builder:

```python
ballast = (
    Ballast()
    .with_memory(
        EpisodicMemory(sources=[...]),
        scope_builder=lambda: NotesScope(
            user_id=current_user_id(),
            tenant_id=current_tenant_id(),
            thread_id=current_thread_id(),
            project_id=current_project_id(),
        ),
    )
    ...
)
```

### 3. `EpisodicSource` Protocol

```python
# src/ballast/memory/episodic/_protocol.py
from typing import Protocol, runtime_checkable


@runtime_checkable
class EpisodicSource(Protocol):
    """One source of episodic facts. Apps register many in EpisodicMemory."""

    name: str                                # for logging / source attribution

    async def recall(
        self, *,
        intent: str,
        scope: Scope,
        k: int,
        detail: DetailLevel,
    ) -> list[ScoredEpisode]: ...

    async def hydrate(
        self, episode: Episode, *, detail: DetailLevel,
    ) -> Episode:
        """Upgrade an existing episode to higher detail level.
        Sources without native support for the requested level raise
        NotImplementedError."""
        ...

    async def remember(self, episode: Episode) -> None:
        """Optional write. Read-only sources (Linear, GitHub, etc.)
        raise NotImplementedError."""
        ...
```

### 4. `RecallStrategy` Protocol + built-in impls

```python
# src/ballast/memory/episodic/strategies/_protocol.py
class RecallStrategy(Protocol):
    """Turns federated source results into the final recall set."""

    requires_grounding: bool = False         # hint: empty references → warn

    async def execute(
        self, *,
        intent: str,
        sources: list[EpisodicSource],
        scope: Scope,
    ) -> RecallResult: ...
```

Built-in strategies in `src/ballast/memory/episodic/strategies/`:

| Strategy | Behavior | Detail default |
|---|---|---|
| `TopK(k, merger, detail)` | parallel query → merge → first K | `SUMMARY` |
| `AllRelevant(threshold, detail)` | parallel query → filter `score >= threshold` | `PREVIEW` |
| `MapReduce(max_items, map_prompt, reduce_prompt, llm, detail)` | parallel query → take top max_items → `map_reduce_llm` digest | `FULL` |
| `Recency(n, detail)` | sort by `occurred_at` desc → first N (scores ignored) | `PREVIEW` |
| `Cluster(n_clusters, embedder, detail)` | k-means on embeddings → one medoid per cluster | `SUMMARY` |

Custom strategies are first-class — apps implement the Protocol.

### 5. `ScoreMerger` — used inside score-based strategies

```python
# src/ballast/memory/episodic/_mergers.py
class ScoreMerger(Protocol):
    def merge(
        self, results_by_source: dict[str, list[ScoredEpisode]],
    ) -> list[ScoredEpisode]: ...


class RRFMerger(ScoreMerger):
    """Reciprocal Rank Fusion — score-agnostic, IR standard.
    Used as default by TopK / AllRelevant."""
    def __init__(self, k: int = 60) -> None: ...
    def merge(self, results_by_source): ...


class WeightedMerger(ScoreMerger):
    """Per-source weights. Apps hint 'trust memory.vector more than memory.notion'."""
    def __init__(self, weights: dict[str, float], normalize: bool = True) -> None: ...


class RawScoreMerger(ScoreMerger):
    """Simple union + sort. Requires score comparability — use only
    when all sources produce calibrated scores."""
    pass
```

Default: `RRFMerger()` — no calibration assumptions; works out of the box.

### 6. `EpisodicMemory` facade

```python
# src/ballast/memory/episodic/_facade.py
class EpisodicMemory:
    """Federation of EpisodicSource impls. Exposes recall via direct API
    (workflow push) and as tools (agent pull)."""

    def __init__(
        self,
        sources: list[EpisodicSource],
        *,
        default_strategy: RecallStrategy = TopK(k=5),
        default_scope_builder: Callable[[], Scope] | None = None,
    ) -> None: ...

    # ── direct API — workflow code calls this ──
    async def episodic_for(
        self, *,
        intent: str,
        strategy: RecallStrategy | None = None,
        scope: Scope | None = None,
    ) -> RecallResult: ...

    async def remember(self, episode: Episode) -> None:
        """Fan-out to all sources that support remember(). Read-only sources skip silently."""
        ...

    # ── tool factory — agent picks these up via Agent(tools=[...]) ──
    def as_tools(self) -> list[Tool]:
        """Returns tools like `recall_episodes(intent: str, k: int = 5)` —
        agent decides when to call."""
        ...
```

### 7. Built-in sources (v1)

#### `ThreadEpisodicSource` — zero new infra

Wraps the existing `thread_repo`. One Episode per "significant turn"
(heuristic: each user→assistant turn pair). `references` populated from
tool calls (any `Ref[T]` parameters / return values).

- `recall(intent, scope, k, detail=PREVIEW)` — for now, returns recent
  turns matching `scope.user_id` ordered by `occurred_at` desc. (Future:
  add embedding index over turn previews for semantic recall — falls into
  Phase 1.5 if needed.)
- `hydrate(episode, detail=FULL)` — pulls full message list from
  `thread_repo` for the episode's `thread_id`.
- `remember()` — `NotImplementedError`. Thread history is the
  source-of-truth; we don't write back.

#### `VectorEpisodicSource` — first-class learnable memory

New SQL table `episodes` with pgvector column. Embeddings over `summary`.

- `recall(intent, scope, k, detail)` — embed `intent`, cosine search
  filtered by `scope.user_id` + `scope.tenant_id` if set. Returns scored
  episodes.
- `hydrate(episode, detail)` — row already carries `summary`. For `FULL`,
  loads `full` JSON column (if `RememberTurn` was configured to capture
  it).
- `remember(episode)` — INSERT row with computed embedding.

Both sources use the existing `Embedder` Protocol at
`src/ballast/capabilities/helpers/embedder.py` (no new abstraction).

### 8. Write triggers

```python
# src/ballast/memory/episodic/_triggers.py

# Manual API
await memory.remember(Episode(...))

# Auto-after-turn capability
class RememberTurn(BallastCapability):
    """After every agent turn, if the gate passes, summarize and store."""
    def __init__(
        self,
        *,
        gate: Callable[[RunContext, RunResult], Awaitable[bool]] | None = None,
        summarizer: Agent | None = None,                       # cheap LLM
        detail_to_store: DetailLevel = DetailLevel.SUMMARY,
    ) -> None: ...

    async def after_run(self, ctx, result) -> None: ...
```

Default `gate` integrates with `LLMJudge`: stores only if the judge passes
(prevents the memory from being polluted by failed turns). Apps without
a judge can pass `gate=lambda *_: True` to always store.

Apps wire as a capability on their agent:

```python
class NotesAgent(DurableAgent):
    def build_agent(self) -> Agent:
        return Agent(
            ...,
            capabilities=[
                *default_notes_capabilities(),
                RememberTurn(summarizer=cheap_haiku),
            ],
        )
```

### 9. `MapReduce` primitive (prerequisite — restored)

We deleted the earlier MapReduce pattern; restoring as a reusable
primitive that both memory's `MapReduceStrategy` AND future
long-document RAG can use.

```python
# src/ballast/patterns/map_reduce/pattern.py
from ballast.durable import Durable


@Durable.workflow
async def map_reduce_llm[InT, MapT, ReduceT](
    items: list[InT],
    *,
    map_step: Callable[[InT], Awaitable[MapT]],
    reduce_step: Callable[[list[MapT]], Awaitable[ReduceT]],
    map_concurrency: int = 8,
    collapse_threshold: int | None = None,
) -> ReduceT:
    """Parallel map → reduce. If map-output exceeds collapse_threshold,
    apply reduce recursively (per-batch) before the final reduce."""
    ...
```

Used by `MapReduceStrategy` over recall hits. Also stands alone for
long-document extraction in future work.

### 10. Integration with `Ref[T]` / `scan_context`

This is the closing-the-loop part. `scan_context` already recurses
through Pydantic objects collecting `Ref[T]` for grounding. We extend
its recursion to recognize:

- `RecallResult` — unwrap `.references`
- `Episode` — unwrap `.references`

That's the entire integration — ~10 lines in `scan_context`. After:

```python
recall = await memory.episodic_for(intent="...", strategy=TopK(k=5))

result = await grounded_agent.run(
    prompt="Pick the most relevant past note to extend.",
    context=[recall],                            # ← memory feeds ground set
    output_type=ExtendDecision,                  # contains Ref[Note]
)
# result.output.target — typed Ref[Note], LLM physically cannot
# hallucinate an id not in recall.references
hydrated = await result.hydrate(notes=notes_repo)
# hydrated.target — real Note instance
```

### 11. App-level wiring

```python
# notes_app/main.py
from ballast import Ballast
from ballast.memory.episodic import (
    EpisodicMemory,
    ThreadEpisodicSource,
    VectorEpisodicSource,
    RememberTurn,
    TopK,
)

ballast = (
    Ballast()
    .with_judge_defaults(...)
    .with_approval_repo(InMemoryApprovalCardRepository())
    .with_memory(
        EpisodicMemory(
            sources=[
                ThreadEpisodicSource(thread_repo),
                VectorEpisodicSource(episode_repo, embedder=openai_embedder),
            ],
            default_strategy=TopK(k=5),
            default_scope_builder=build_notes_scope,
        ),
    )
    .build()
)
```

## Error handling

- `EpisodicSource.recall` raises → facade logs + skips that source for
  this call (other sources still contribute). Multi-source resilience.
- Empty `recall.references` with strategy `requires_grounding=True` →
  facade emits warning; agent gets empty enum → output validation fails
  loudly rather than silently dropping the constraint.
- `MapReduceStrategy` with 0 hits → returns empty `RecallResult` (don't
  invoke LLM with empty input).
- `Embedder` failure during `remember()` → log + raise; the calling
  capability decides whether to fail the turn (default: swallow + log).

## Testing

- **Unit per source**: `ThreadEpisodicSource.recall` against in-memory
  thread repo with seeded threads. `VectorEpisodicSource.recall` against
  PG fixture with seeded `episodes` rows.
- **Unit per strategy**: with a fake source returning fixed `ScoredEpisode`
  lists, exercise TopK/AllRelevant/Recency/Cluster/MapReduce reductions.
- **Merger tests**: RRFMerger correctness vs hand-computed values.
- **Facade integration**: federation across multiple fake sources,
  verify merging and detail-level propagation.
- **Grounded integration**: `scan_context` over `RecallResult` collects
  expected Refs; `GroundedAgent` with recall in context produces
  output_type with valid enum.
- **MapReduce primitive**: parallel map + reduce correctness + collapse
  recursion.
- **End-to-end** (notes-app): inject a `RememberTurn` capability into
  `NotesAgent`; run several create_note turns; assert episodes recallable
  in a later thread.

## What this design deliberately does NOT do (Phase 1)

- **No semantic memory (Phase 2).** Domain repos as memory primitives.
  Defer until Phase 1 lands.
- **No procedural memory (Phase 3).** Workflow registry with introspection.
- **No learning loop (Phase 4).** HITL-suggested skill consolidation.
- **No vector index over thread previews in `ThreadEpisodicSource` v1.**
  Recent-first fallback; embedding-indexed version is Phase 1.5 if
  user-visible quality demands it.
- **No automatic source registration / discovery.** Apps register sources
  explicitly via builder — no magic.
- **No cross-tenant recall.** `scope.tenant_id` (if set) is hard filter.

## Files touched

**Framework — new:**

  - `src/ballast/memory/__init__.py`
  - `src/ballast/memory/_scope.py`
  - `src/ballast/memory/episodic/__init__.py`
  - `src/ballast/memory/episodic/_protocol.py`        (EpisodicSource)
  - `src/ballast/memory/episodic/_models.py`          (Episode, DetailLevel, ScoredEpisode, RecallResult)
  - `src/ballast/memory/episodic/_facade.py`          (EpisodicMemory)
  - `src/ballast/memory/episodic/_mergers.py`         (RRFMerger, WeightedMerger, RawScoreMerger)
  - `src/ballast/memory/episodic/strategies/__init__.py`
  - `src/ballast/memory/episodic/strategies/_protocol.py`   (RecallStrategy)
  - `src/ballast/memory/episodic/strategies/_topk.py`
  - `src/ballast/memory/episodic/strategies/_all_relevant.py`
  - `src/ballast/memory/episodic/strategies/_recency.py`
  - `src/ballast/memory/episodic/strategies/_cluster.py`
  - `src/ballast/memory/episodic/strategies/_map_reduce.py`
  - `src/ballast/memory/episodic/sources/__init__.py`
  - `src/ballast/memory/episodic/sources/_thread.py`        (ThreadEpisodicSource)
  - `src/ballast/memory/episodic/sources/_vector.py`        (VectorEpisodicSource + SqlEpisodeRow + repo)
  - `src/ballast/memory/episodic/_triggers.py`              (RememberTurn capability)
  - `src/ballast/memory/episodic/_tools.py`                 (recall_episodes tool factory for agent pull surface)
  - `src/ballast/patterns/map_reduce/__init__.py`           (restored)
  - `src/ballast/patterns/map_reduce/pattern.py`            (map_reduce_llm)
  - `src/ballast/alembic/versions/0003_episodes.py`

**Framework — modify:**

  - `src/ballast/app.py` — `Ballast.with_memory(memory, scope_builder)` setter
  - `src/ballast/grounded/_scan_context.py` — recognize RecallResult + Episode in the recursion
  - `src/ballast/__init__.py` — re-export Memory / Episode / DetailLevel / strategies
  - `pyproject.toml` — add `pgvector` dependency (optional extra)

**Notes-app — modify (smoke):**

  - `examples/notes-app/backend/src/notes_app/main.py` — wire `EpisodicMemory(sources=[Thread, Vector])` + `with_memory(...)` + `RememberTurn` capability on NotesAgent
  - `examples/notes-app/backend/src/notes_app/agents/notes.py` — append `RememberTurn(summarizer=...)` to `default_notes_capabilities()`
  - `examples/notes-app/backend/src/notes_app/workflows/create_note.py` — use `memory.episodic_for(...)` to fetch similar past notes; enrich the approval card payload

**Tests — new:**

  - `tests/memory/test_episode_models.py`
  - `tests/memory/test_scope.py`
  - `tests/memory/test_mergers.py`
  - `tests/memory/strategies/test_topk.py`
  - `tests/memory/strategies/test_recency.py`
  - `tests/memory/strategies/test_cluster.py`
  - `tests/memory/strategies/test_map_reduce.py`
  - `tests/memory/sources/test_thread_source.py`
  - `tests/memory/sources/test_vector_source.py`
  - `tests/memory/test_facade.py`
  - `tests/memory/test_grounded_integration.py`
  - `tests/memory/test_remember_turn.py`
  - `tests/patterns/map_reduce/test_map_reduce_llm.py`

## Open follow-ups (Phase 2+)

  - **Phase 2 — Semantic memory:** `SemanticSource` Protocol + decorator
    to expose domain repos as memory primitives; `DomainSemanticSource`
    base + `VectorSemanticSource` for free-text fields.
  - **Phase 3 — Procedural memory:** `WorkflowRegistry` with introspection
    (list_skills returning JSON schemas); `as_tools()` exposes registered
    workflows as named agent skills.
  - **Phase 4 — Learning loop:** clustering of recent episodes →
    HITL-suggested consolidation into a new procedural skill via UICardChannel.
  - **Phase 1.5 (conditional):** embedding index over thread previews in
    `ThreadEpisodicSource` if recency-only proves insufficient in
    notes-app demos.
  - **MCP-backed sources:** `MCPEpisodicSource(mcp_server)` for Linear /
    GitHub / Notion past-activity recall. Builds on the MCP integration
    we discussed but didn't yet schedule.

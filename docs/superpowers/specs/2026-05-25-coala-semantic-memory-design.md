# CoALA Phase 2 — Semantic Memory

**Date:** 2026-05-25
**Status:** Approved (brainstorming) → ready for implementation plan
**Author:** Kir + Claude
**Scope:** Phase 2 of a 4-phase CoALA-inspired memory subsystem. Phase 1
(episodic) shipped. Phases 3 (procedural) and 4 (learning) follow.

## Problem

Phase 1 gave the agent **episodic memory** — what happened before, via
embedded turn-summaries. That's only one CoALA slot. The agent has no
first-class API for **factual** knowledge about the current world state:
"my notes tagged 'work'", "this user's recent orders", "the policy on
refunds". Today such queries either:

1. Are not accessible to the agent at all (no tool), or
2. Are wired ad-hoc per tool, with no unified semantics of "this is
   memory; here's how it's discovered, scoped, and exposed".

The result: every app re-invents how it exposes domain data to the
agent. No consistent "memory" mental model. CoALA's full benefit
requires explicit semantic memory.

## Core insights (from brainstorming)

1. **Semantic memory in domain-rich relational apps is NOT vector RAG.**
   Most facts are typed rows in a SQL database, queryable with joins +
   filters. Embeddings only matter for free-text fields. Exposing repos
   as memory primitives is the bulk of the work.

2. **Per-domain shapes are not uniform.** Unlike Episodic where every
   record fits the same `Episode` schema, semantic facts have wildly
   different shapes (Note vs Order vs User). There is no universal
   "Fact" type. So the protocol can't define `recall() → list[Fact]` —
   each source exposes its own typed methods.

3. **Workflows already have direct access to domain repos.** A workflow
   doesn't need a memory facade to query notes — it calls `notes_repo`
   directly. Semantic memory's value-add is **exposing those methods
   as agent tools**, not gatekeeping workflow access.

4. **Source registration ≠ source instantiation.** Sources are module-
   level singletons (like `notes_repo`); the `SemanticMemory(sources=)`
   facade collects them only to derive the agent's tool set.

5. **Scope (user/tenant) is enforced at the repo layer**, not in source
   methods, using the `current_user_id()` ContextVar that already exists
   from Phase 1. Source methods stay scope-free in their signatures.

## Design

### 1. `SemanticSource` Protocol

```python
# src/ballast/memory/semantic/_protocol.py
from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class SemanticSource(Protocol):
    """Source of typed structured facts about the domain.

    Implementations expose one or more ``@memory_tool``-decorated async
    methods. ``SemanticMemory.as_tools()`` introspects each registered
    source and builds pydantic-ai ``Tool`` instances from the marked
    methods. The framework knows nothing about the methods' signatures
    or return types — they flow through to the LLM as typed tools.
    """

    name: ClassVar[str]
```

Minimal — no abstract methods. Different from `EpisodicSource`
(`recall`/`hydrate`/`remember`) because semantic facts have no uniform
shape.

### 2. `@memory_tool` decorator

```python
# src/ballast/memory/semantic/_decorator.py
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")


@overload
def memory_tool(fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def memory_tool(*, name: str | None = None, description: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def memory_tool(
    fn: Callable[P, R] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Mark an async method on a ``SemanticSource`` as a tool exposed
    to the agent.

    Bare form::

        @memory_tool
        async def find_by_tag(self, tag: str, limit: int = 10) -> list[Note]:
            \"\"\"Find notes tagged with `tag`. Most recent first.\"\"\"
            ...

    With overrides (collision avoidance / explicit description)::

        @memory_tool(name="search_notes_by_tag",
                     description="Search notes by tag")
        async def find_by_tag(self, tag: str): ...
    """
    def decorate(f: Callable[P, R]) -> Callable[P, R]:
        f.__memory_tool__ = True                    # type: ignore[attr-defined]
        f.__memory_tool_name__ = name               # type: ignore[attr-defined]
        f.__memory_tool_description__ = description # type: ignore[attr-defined]
        return f

    if fn is not None:
        return decorate(fn)
    return decorate
```

The decorator only attaches marker attributes. Pydantic-ai's `Tool`
constructor reads `__doc__` + `inspect.signature(fn)` for description
and arg-schema; we let it.

### 3. `DomainSemanticSource` — repo-wrapping convenience base

```python
# src/ballast/memory/semantic/_domain.py
from abc import ABC
from typing import ClassVar

from ballast.memory.semantic._protocol import SemanticSource


class DomainSemanticSource(SemanticSource, ABC):
    """Base class for semantic sources that wrap domain repositories.

    Convention: subclass, set ``name``, add ``@memory_tool`` methods
    that delegate to repo singletons. Scope (user_id, tenant_id) is
    enforced by the underlying repo via ``current_user_id()``
    ContextVar — no scope parameter on the source methods.

    Pure convenience — DOES NOT enforce any structural shape beyond
    ``name``. Subclasses use ``@memory_tool`` freely.
    """

    name: ClassVar[str]
```

Apps use it like:

```python
# notes_app/memory/semantic_sources.py
from ballast.memory.semantic import DomainSemanticSource, memory_tool
from notes_app.models.note import Note


class NotesSemantic(DomainSemanticSource):
    name = "notes"

    @memory_tool
    async def find_by_tag(self, tag: str, limit: int = 10) -> list[Note]:
        """Return notes tagged with `tag`, most recent first."""
        from notes_app.repositories.note import notes_repo
        return await notes_repo.find_by_tag(tag, limit=limit)

    @memory_tool
    async def recent(self, days: int = 7) -> list[Note]:
        """Return notes created in the last `days` days."""
        from notes_app.repositories.note import notes_repo
        return await notes_repo.recent(days=days)


notes_semantic: NotesSemantic = NotesSemantic()   # module singleton
```

### 4. `VectorSemanticSource` — convenience base for free-text RAG

```python
# src/ballast/memory/semantic/_vector.py
from abc import ABC
from typing import Any, ClassVar

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import SQLModel

from ballast.capabilities.helpers.embedder import Embedder
from ballast.memory.semantic._protocol import SemanticSource


class VectorSemanticSource(SemanticSource, ABC):
    """Base class for semantic sources backed by embedded free-text fields.

    Provides typical wiring (``embedder`` + ``sessionmaker``) and a
    helper ``_vector_search`` for the common cosine-distance query.
    Subclasses decide what to expose via ``@memory_tool`` — one search
    method per indexed corpus, or one method total, app's choice.
    """

    name: ClassVar[str]

    def __init__(
        self,
        *,
        embedder: Embedder,
        sessionmaker: async_sessionmaker[AsyncSession],
    ) -> None:
        self._embedder = embedder
        self._sessionmaker = sessionmaker

    async def _vector_search(
        self,
        *,
        query: str,
        table: type[SQLModel],
        embedding_column: Any,         # e.g. MyEmbeddingRow.embedding
        k: int,
        where: Any | None = None,      # optional sqlalchemy where clause
    ) -> list[Any]:
        """Embed ``query`` and cosine-search ``table`` ordered by
        distance. Returns up to ``k`` rows."""
        from sqlmodel import select  # noqa: PLC0415

        query_vec = await self._embedder.embed(query)
        async with self._sessionmaker() as session:
            stmt = select(table).order_by(embedding_column.cosine_distance(query_vec))
            if where is not None:
                stmt = stmt.where(where)
            stmt = stmt.limit(k)
            result = await session.execute(stmt)
            return list(result.scalars().all())
```

App author writes:

```python
class NoteBodySemantic(VectorSemanticSource):
    name = "notes-vector"

    @memory_tool
    async def search_bodies(self, query: str, k: int = 5) -> list[Note]:
        """Find notes whose body is semantically similar to `query`."""
        rows = await self._vector_search(
            query=query,
            table=NoteBodyEmbeddingRow,
            embedding_column=NoteBodyEmbeddingRow.embedding,
            k=k,
        )
        return [_to_note(r) for r in rows]
```

App owns the embedding row schema + write-side indexing (typically a
post-save hook on the domain repo). Framework provides only the
read-side helper.

### 5. `SemanticMemory` facade — thin tool aggregator

```python
# src/ballast/memory/semantic/_facade.py
from pydantic_ai import Tool

from ballast.memory.semantic._protocol import SemanticSource
from ballast.memory.semantic._tools import extract_memory_tools


class SemanticMemory:
    """Federation of SemanticSource impls for agent-pull tool exposure.

    Workflow code imports + calls source singletons directly — no
    facade indirection. This object exists purely to:
      1. collect tools from all sources for ``Agent(tools=...)``,
      2. validate no tool-name collisions across sources at construction,
      3. provide introspection (``list_sources``) for admin / debug.
    """

    def __init__(self, sources: list[SemanticSource]) -> None:
        if not sources:
            raise ValueError("SemanticMemory requires at least one source")
        self._sources = sources
        self._validate_no_collisions()

    def as_tools(self) -> list[Tool]:
        """Return pydantic-ai ``Tool`` instances collected from all
        ``@memory_tool``-decorated methods across all sources."""
        return [
            tool for src in self._sources for tool in extract_memory_tools(src)
        ]

    def list_sources(self) -> list[SemanticSource]:
        return list(self._sources)

    def _validate_no_collisions(self) -> None:
        names: dict[str, str] = {}    # tool_name → owning_source_name
        for src in self._sources:
            for tool in extract_memory_tools(src):
                if tool.name in names:
                    raise ValueError(
                        f"SemanticMemory tool-name collision: {tool.name!r} "
                        f"defined by both {names[tool.name]!r} and {src.name!r}. "
                        "Use @memory_tool(name=...) to disambiguate."
                    )
                names[tool.name] = src.name
```

### 6. Tool extraction

```python
# src/ballast/memory/semantic/_tools.py
import inspect
from pydantic_ai import Tool

from ballast.memory.semantic._protocol import SemanticSource


def extract_memory_tools(source: SemanticSource) -> list[Tool]:
    """Find every ``@memory_tool``-marked method on ``source`` and wrap
    each in a pydantic-ai ``Tool``.

    Method name (or override via ``@memory_tool(name=...)``) becomes
    the tool name; docstring (or override) becomes the description;
    pydantic-ai derives the arg schema from ``inspect.signature`` +
    type hints.
    """
    tools: list[Tool] = []
    for attr_name, attr_value in inspect.getmembers(source, inspect.iscoroutinefunction):
        if not getattr(attr_value, "__memory_tool__", False):
            continue
        tool_name = (
            getattr(attr_value, "__memory_tool_name__", None) or attr_name
        )
        tool_description = (
            getattr(attr_value, "__memory_tool_description__", None)
            or (attr_value.__doc__ or "").strip()
            or None
        )
        tools.append(Tool(
            attr_value,
            name=tool_name,
            description=tool_description,
            takes_ctx=False,
        ))
    return tools
```

### 7. Ballast wiring — separate setters

Phase 1 used `Ballast.with_memory(EpisodicMemory(...))`. Phase 2
splits this into:

```python
# src/ballast/app.py — additions

def with_episodic_memory(
    self,
    memory: "EpisodicMemory",
    *,
    scope_builder: "Callable[[], Scope] | None" = None,
) -> "Ballast":
    """Wire an EpisodicMemory facade + optional default scope-builder.

    Replaces the deprecated ``with_memory`` (which is kept as a
    backward-compatible alias for one release).
    """
    self._episodic_memory = memory
    if scope_builder is not None:
        memory._default_scope_builder = scope_builder
    return self


def with_semantic_memory(
    self,
    memory: "SemanticMemory",
) -> "Ballast":
    """Wire a SemanticMemory facade for agent-pull tool exposure.

    Workflow code accesses semantic sources directly via their module
    singletons (e.g. ``from notes_app.memory.semantic_sources import
    notes_semantic``) — the facade is purely for agent tool collection.
    """
    self._semantic_memory = memory
    return self


# Deprecated alias kept for Phase 1 back-compat — one release window.
def with_memory(
    self,
    memory: "EpisodicMemory",
    *,
    scope_builder: "Callable[[], Scope] | None" = None,
) -> "Ballast":
    """Deprecated. Use ``with_episodic_memory(...)`` instead."""
    import warnings
    warnings.warn(
        "Ballast.with_memory is deprecated; use with_episodic_memory.",
        DeprecationWarning, stacklevel=2,
    )
    return self.with_episodic_memory(memory, scope_builder=scope_builder)
```

Internal attrs:

```python
# In Ballast.__init__:
self._episodic_memory: "EpisodicMemory | None" = None
self._semantic_memory: "SemanticMemory | None" = None
```

Phase 1's existing `self._memory` attr also stays (as an alias of
`self._episodic_memory`) until consumers migrate; the deprecated
`with_memory` keeps both in sync. Phase 1 consumers in `notes-app`
that read `get_ballast()._memory` will be updated to
`._episodic_memory` in this phase.

### 8. App integration

```python
# notes_app/memory/semantic_sources.py
from ballast.memory.semantic import DomainSemanticSource, memory_tool
from notes_app.models.note import Note
from notes_app.models.todo import Todo


class NotesSemantic(DomainSemanticSource):
    name = "notes"

    @memory_tool
    async def find_by_tag(self, tag: str, limit: int = 10) -> list[Note]:
        """Return notes tagged with `tag`."""
        from notes_app.repositories.note import notes_repo
        return await notes_repo.find_by_tag(tag, limit=limit)

    @memory_tool
    async def recent(self, days: int = 7) -> list[Note]:
        """Return notes created in the last `days` days."""
        from notes_app.repositories.note import notes_repo
        return await notes_repo.recent(days=days)


notes_semantic: NotesSemantic = NotesSemantic()


# notes_app/main.py
from ballast.memory.semantic import SemanticMemory
from notes_app.memory.semantic_sources import notes_semantic

ballast = (
    Ballast()
    ...
    .with_episodic_memory(EpisodicMemory(sources=[...]))
    .with_semantic_memory(SemanticMemory(sources=[notes_semantic]))
    ...
)


# notes_app/agents/notes.py — NotesAgent.build_agent additions
class NotesAgent(DurableAgent):
    def build_agent(self) -> Agent[NoteToolDeps, Any]:
        semantic = getattr(get_ballast(), "_semantic_memory", None)
        extra_tools = semantic.as_tools() if semantic is not None else []
        return Agent(
            model=build_openrouter_model(),
            output_type=[str, DeferredToolRequests],
            deps_type=NoteToolDeps,
            system_prompt=SYSTEM_PROMPT,
            capabilities=default_notes_capabilities(),
            tools=extra_tools,    # ← semantic-memory tools join the standard set
        )
```

The existing `@NotesAgent.tool`-decorated tools (create_note, list_notes,
etc.) continue to register at class-decoration time — `tools=` ctor
arg layers on top.

### 9. Workflow-push usage

```python
# notes_app/workflows/create_note.py — enrich approval card with semantic recall
from notes_app.memory.semantic_sources import notes_semantic

@Durable.workflow()
async def create_note_flow(draft: ProposedNote) -> Note | None:
    # Direct call to module singleton — no facade.
    similar = await notes_semantic.find_by_tag(
        tag=_extract_tag(draft.title) or "general",
        limit=3,
    )
    _log.info("create_note_flow: %d similar via semantic", len(similar))

    # ... existing flow ...
```

No imports from Ballast. Workflow code stays domain-specific.

### 10. Scope (user / tenant) — enforced at repo layer

No `scope` parameter on `SemanticSource` methods. Apps' underlying
repos already enforce scope via `current_user_id()` ContextVar
(installed by Phase 1's `ballast.auth.context`). E.g.
`notes_repo.find_by_tag(tag, limit=10)` internally does
`WHERE user_id = current_user_id() AND tag = ?`.

This keeps source methods simple (no boilerplate scope plumbing) and
authoritative (the repo is the single source of truth for who-sees-what).

### 11. Collision handling

`SemanticMemory.__init__` validates that no two `@memory_tool` methods
across all registered sources share the same exposed tool name. Raises
`ValueError` with a clear hint pointing to `@memory_tool(name=...)`
for explicit disambiguation. No auto-prefixing — name collisions are
bugs to fix, not noise to mask.

## Error handling

- **Tool call raises in the source method**: pydantic-ai propagates
  to the agent as a tool-error response; agent decides whether to
  retry / fall back. No special handling at the SemanticMemory layer.
- **Collision at construction**: hard fail, fast feedback.
- **Empty sources list**: `ValueError` at SemanticMemory construction.
- **Source with no `@memory_tool` methods**: warning logged at
  construction; source contributes zero tools but is otherwise valid
  (might be a draft).

## Testing

- **Unit — `@memory_tool` decorator**: marks the function with attrs;
  preserves callable behavior; supports name + description overrides.
- **Unit — `extract_memory_tools`**: skips unmarked methods; reads
  override name / description; uses docstring as fallback.
- **Unit — `SemanticMemory` collision detection**: raises on duplicate
  tool names across sources; clean construction with disambiguated
  names succeeds.
- **Unit — `DomainSemanticSource` / `VectorSemanticSource`**: minimal
  ABC tests; subclass + `@memory_tool` produces an introspectable tool.
- **Integration — `VectorSemanticSource._vector_search`**: against
  the existing pgvector test fixture, seed rows + assert cosine
  ordering.
- **Integration — Ballast wiring**: `with_semantic_memory` stores the
  facade; agent built with `tools=memory.as_tools()` can be invoked
  and the model-call routes to the source method.
- **Integration — notes-app**: a `NotesSemantic` example wired in;
  agent run invokes `find_by_tag` end-to-end.

## What this design deliberately does NOT do

- **No push-API facade** — `SemanticMemory.invoke(name, **args)` is
  not added. Workflow code imports source singletons and calls
  methods directly.
- **No scope parameter on source methods** — scope is enforced at the
  repo layer via the existing `current_user_id()` ContextVar.
- **No auto-attach via capability** — apps explicitly splat
  `*semantic.as_tools()` into `Agent(tools=...)`. Implicit wiring is
  a Phase 4 / follow-up.
- **No caching layer** — no use case yet; defer until requested.
- **No tool-name auto-prefixing** — collisions raise loudly; explicit
  rename via `@memory_tool(name=...)` is the fix.
- **No `memory_tool` support on standalone functions** — only methods
  of `SemanticSource` subclasses are introspected. Standalone tool
  exposure is what pydantic-ai's `@Agent.tool` already does.
- **No migration of existing `notes_repo` to a SemanticSource** — apps
  decide which subset of repo methods to expose via wrapping; the repo
  stays a pure domain primitive.

## Files touched

**Framework — new:**

- `src/ballast/memory/semantic/__init__.py` — re-exports
- `src/ballast/memory/semantic/_protocol.py` — `SemanticSource`
- `src/ballast/memory/semantic/_decorator.py` — `@memory_tool`
- `src/ballast/memory/semantic/_tools.py` — `extract_memory_tools`
- `src/ballast/memory/semantic/_facade.py` — `SemanticMemory`
- `src/ballast/memory/semantic/_domain.py` — `DomainSemanticSource` ABC
- `src/ballast/memory/semantic/_vector.py` — `VectorSemanticSource` ABC + `_vector_search`

**Framework — modify:**

- `src/ballast/app.py` — `with_episodic_memory` (rename + retain alias),
  `with_semantic_memory` (new). Internal `self._semantic_memory` attr.
- `src/ballast/__init__.py` — re-export `SemanticMemory`,
  `SemanticSource`, `DomainSemanticSource`, `VectorSemanticSource`,
  `memory_tool`.
- `src/ballast/memory/__init__.py` — re-export the semantic public API
  alongside the existing `Scope`.

**Notes-app — new:**

- `examples/notes-app/backend/src/notes_app/memory/semantic_sources.py`
  — `NotesSemantic(DomainSemanticSource)` + `notes_semantic` singleton.

**Notes-app — modify:**

- `examples/notes-app/backend/src/notes_app/main.py` — `.with_semantic_memory(...)`
  in builder chain; rename `.with_memory(...)` → `.with_episodic_memory(...)`.
- `examples/notes-app/backend/src/notes_app/agents/notes.py` —
  `NotesAgent.build_agent()` reads semantic memory and passes
  `tools=memory.as_tools()` to `Agent(...)`.
- `examples/notes-app/backend/src/notes_app/workflows/create_note.py` —
  swap the existing recall logic to use `notes_semantic.find_by_tag(...)`
  alongside (or instead of) episodic recall, demonstrating the workflow-push
  pattern.

**Tests — new (mirroring source layout):**

- `tests/memory/semantic/test_decorator.py`
- `tests/memory/semantic/test_tools.py`
- `tests/memory/semantic/test_facade.py`
- `tests/memory/semantic/test_domain.py`
- `tests/memory/semantic/test_vector.py`
- `tests/app/test_with_semantic_memory.py`
- `examples/notes-app/backend/tests/test_notes_semantic.py` — wired
  source + agent integration smoke.

## Scope estimate

~10-12 tasks at TDD granularity:

1. `SemanticSource` Protocol + `DomainSemanticSource` ABC
2. `@memory_tool` decorator
3. `extract_memory_tools` introspection helper
4. `SemanticMemory` facade + collision detection
5. `VectorSemanticSource` ABC + `_vector_search` helper
6. `Ballast.with_semantic_memory(...)` + rename `with_memory` →
   `with_episodic_memory` + deprecated alias
7. Phase 1 consumer migration (`notes_app/agents/notes.py` etc. swap
   `_memory` → `_episodic_memory`)
8. Public API exports
9. Notes-app: `NotesSemantic` source + module singleton
10. Notes-app: `main.py` wiring
11. Notes-app: `NotesAgent` reads semantic tools
12. Final smoke

~3-5 days execution time (much smaller than Phase 1 since no new
persistence layer, no embedding-store boilerplate, no strategy
hierarchy).

## Open follow-ups (Phase 3+ or sidecars)

- **Phase 3 — Procedural memory**: workflow registry + introspection;
  agent calls registered workflows by name as named "skills".
- **Phase 4 — Learning loop**: cluster recent episodes → HITL
  consolidation into a new procedural skill.
- **Auto-attach capability**: implicit injection of `as_tools()` into
  every `DurableAgent` build, removing the explicit splat. Phase 4 sugar.
- **Caching layer for semantic tool calls** — repo invalidation
  semantics make this nontrivial; defer until use case appears.
- **MCP-backed semantic sources** — `MCPSemanticSource(mcp_server)`
  bridges external MCP servers (Linear, GitHub, Notion) into the
  framework's semantic-tool surface. Naturally composable.

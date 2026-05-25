# CoALA Phase 2 — Semantic Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `SemanticMemory` — typed semantic-fact memory layer: `SemanticSource` Protocol + `@memory_tool` decorator + `DomainSemanticSource` (repo-wrapping convenience) + `VectorSemanticSource` (free-text RAG helper) + `SemanticMemory` facade (collects tools across sources for agent exposure). Workflow code uses source module singletons directly; the agent gets the union of all `@memory_tool`-decorated methods as pydantic-ai tools.

**Architecture:** Each source is a module-level singleton, typically subclassing `DomainSemanticSource` and wrapping a domain repo with `@memory_tool`-marked async methods. The framework introspects each source via `inspect.getmembers` + the decorator's marker attribute, builds `pydantic_ai.Tool` instances, and exposes them as a single list via `SemanticMemory.as_tools()`. No workflow-push facade — workflows import + call source singletons directly (same convention as `notes_repo`).

**Tech Stack:** Python 3.11+, pydantic v2, pydantic-ai (`Tool`), SQLAlchemy/SQLModel + pgvector (for `VectorSemanticSource`), existing `Embedder` Protocol.

**Spec:** `docs/superpowers/specs/2026-05-25-coala-semantic-memory-design.md`

---

## File Map

**Framework — new:**
- `src/ballast/memory/semantic/__init__.py` — public re-exports
- `src/ballast/memory/semantic/_protocol.py` — `SemanticSource` Protocol
- `src/ballast/memory/semantic/_decorator.py` — `@memory_tool` decorator
- `src/ballast/memory/semantic/_tools.py` — `extract_memory_tools` introspection helper
- `src/ballast/memory/semantic/_facade.py` — `SemanticMemory` facade
- `src/ballast/memory/semantic/_domain.py` — `DomainSemanticSource` ABC
- `src/ballast/memory/semantic/_vector.py` — `VectorSemanticSource` ABC + `_vector_search` helper

**Framework — modify:**
- `src/ballast/app.py` — rename `with_memory` → `with_episodic_memory` + deprecated alias, add `with_semantic_memory`, internal `_semantic_memory` attr
- `src/ballast/memory/__init__.py` — add semantic re-exports alongside `Scope`
- `src/ballast/__init__.py` — top-level re-exports

**Notes-app — new:**
- `examples/notes-app/backend/src/notes_app/memory/semantic_sources.py` — `NotesSemantic(DomainSemanticSource)` + `notes_semantic` singleton

**Notes-app — modify:**
- `examples/notes-app/backend/src/notes_app/main.py` — `.with_semantic_memory(...)` chain, rename `.with_memory(...)` → `.with_episodic_memory(...)`
- `examples/notes-app/backend/src/notes_app/agents/notes.py` — `NotesAgent.build_agent` passes `tools=memory.as_tools()`, switch private attr read from `_memory` → `_episodic_memory`
- `examples/notes-app/backend/src/notes_app/workflows/create_note.py` — switch `getattr(get_ballast(), "_memory", None)` → `"_episodic_memory"`
- `examples/notes-app/backend/src/notes_app/repositories/note.py` — add `find_by_tag` and `recent` methods (used by `NotesSemantic` example)

**Tests — new:**
- `tests/memory/semantic/__init__.py` (empty)
- `tests/memory/semantic/test_decorator.py`
- `tests/memory/semantic/test_tools.py`
- `tests/memory/semantic/test_facade.py`
- `tests/memory/semantic/test_domain.py`
- `tests/memory/semantic/test_vector.py`
- `tests/app/test_with_semantic_memory.py`
- `examples/notes-app/backend/tests/test_notes_semantic.py`

---

## Task 1: `SemanticSource` Protocol + `DomainSemanticSource` ABC

**Files:**
- Create: `src/ballast/memory/semantic/__init__.py`
- Create: `src/ballast/memory/semantic/_protocol.py`
- Create: `src/ballast/memory/semantic/_domain.py`
- Create: `tests/memory/semantic/__init__.py` (empty)
- Create: `tests/memory/semantic/test_domain.py`

- [ ] **Step 1: Write the failing test**

Create `tests/memory/semantic/__init__.py` (empty). Then `tests/memory/semantic/test_domain.py`:

```python
"""``SemanticSource`` Protocol + ``DomainSemanticSource`` ABC."""
from __future__ import annotations

from ballast.memory.semantic import DomainSemanticSource, SemanticSource


def test_runtime_checkable_protocol() -> None:
    class _Stub:
        name = "stub"
    assert isinstance(_Stub(), SemanticSource)


def test_protocol_rejects_missing_name() -> None:
    class _NoName:
        pass
    assert not isinstance(_NoName(), SemanticSource)


def test_domain_semantic_source_is_subclass_of_protocol() -> None:
    class _MySource(DomainSemanticSource):
        name = "my"
    assert isinstance(_MySource(), SemanticSource)
    assert _MySource().name == "my"


def test_domain_semantic_source_can_be_subclassed_without_methods() -> None:
    """ABC has no abstract methods — subclassing alone is sufficient."""
    class _Empty(DomainSemanticSource):
        name = "empty"
    instance = _Empty()
    assert instance.name == "empty"
```

- [ ] **Step 2: Run — confirm fail**

```
uv run pytest tests/memory/semantic/test_domain.py -v
```

Expected: `ModuleNotFoundError: No module named 'ballast.memory.semantic'`.

- [ ] **Step 3: Implement Protocol**

Create `src/ballast/memory/semantic/_protocol.py`:

```python
"""``SemanticSource`` Protocol — typed source of structured facts."""
from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class SemanticSource(Protocol):
    """Source of typed structured facts about the domain.

    Implementations expose one or more ``@memory_tool``-decorated async
    methods. ``SemanticMemory.as_tools()`` introspects each registered
    source and builds pydantic-ai ``Tool`` instances from the marked
    methods. The framework knows nothing about the methods' signatures
    or return types — they flow through to the LLM as typed tools.

    No abstract methods (unlike ``EpisodicSource``) — semantic facts
    have wildly different per-domain shapes; only the ``name`` attribute
    is required.
    """

    name: ClassVar[str]


__all__ = ["SemanticSource"]
```

- [ ] **Step 4: Implement DomainSemanticSource ABC**

Create `src/ballast/memory/semantic/_domain.py`:

```python
"""``DomainSemanticSource`` — convenience base for repo-wrapping sources."""
from __future__ import annotations

from abc import ABC
from typing import ClassVar

from ballast.memory.semantic._protocol import SemanticSource


class DomainSemanticSource(SemanticSource, ABC):
    """Base class for semantic sources that wrap domain repositories.

    Convention: subclass, set ``name``, add ``@memory_tool`` methods
    that delegate to repo singletons. Scope (user_id, tenant_id) is
    enforced by the underlying repo via ``current_user_id()``
    ContextVar (installed by Phase 1 ``ballast.auth.context``) — no
    scope parameter on the source methods.

    Pure convenience — DOES NOT enforce any structural shape beyond
    ``name``. Subclasses use ``@memory_tool`` freely on as few or as
    many methods as they want.
    """

    name: ClassVar[str]


__all__ = ["DomainSemanticSource"]
```

- [ ] **Step 5: Implement package __init__**

Create `src/ballast/memory/semantic/__init__.py`:

```python
"""Semantic memory — typed facts about the world via repo-wrappers."""
from ballast.memory.semantic._domain import DomainSemanticSource
from ballast.memory.semantic._protocol import SemanticSource

__all__ = ["DomainSemanticSource", "SemanticSource"]
```

- [ ] **Step 6: Run tests — confirm pass**

```
uv run pytest tests/memory/semantic/test_domain.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add src/ballast/memory/semantic tests/memory/semantic
git commit -m "feat(memory): SemanticSource Protocol + DomainSemanticSource ABC"
```

---

## Task 2: `@memory_tool` decorator

**Files:**
- Create: `src/ballast/memory/semantic/_decorator.py`
- Modify: `src/ballast/memory/semantic/__init__.py` (export `memory_tool`)
- Create: `tests/memory/semantic/test_decorator.py`

- [ ] **Step 1: Write the failing test**

```python
"""``@memory_tool`` decorator — marker for tool-exposable methods."""
from __future__ import annotations

import pytest

from ballast.memory.semantic import memory_tool


def test_bare_decorator_marks_function() -> None:
    @memory_tool
    async def my_method(self, x: int) -> str:
        """My docstring."""
        return str(x)

    assert getattr(my_method, "__memory_tool__", False) is True
    assert getattr(my_method, "__memory_tool_name__", None) is None
    assert getattr(my_method, "__memory_tool_description__", None) is None


def test_decorator_with_name_override() -> None:
    @memory_tool(name="custom_name")
    async def my_method(self, x: int) -> str:
        return str(x)

    assert my_method.__memory_tool__ is True
    assert my_method.__memory_tool_name__ == "custom_name"


def test_decorator_with_description_override() -> None:
    @memory_tool(description="Custom description")
    async def my_method(self) -> str:
        return "x"

    assert my_method.__memory_tool__ is True
    assert my_method.__memory_tool_description__ == "Custom description"


def test_decorator_preserves_callable_behavior() -> None:
    """Decorator doesn't wrap — calling the method works normally."""
    import asyncio

    @memory_tool
    async def doubler(x: int) -> int:
        return x * 2

    out = asyncio.run(doubler(5))
    assert out == 10


def test_decorator_with_both_overrides() -> None:
    @memory_tool(name="foo", description="bar")
    async def my_method(self) -> None: ...

    assert my_method.__memory_tool_name__ == "foo"
    assert my_method.__memory_tool_description__ == "bar"
```

- [ ] **Step 2: Run — confirm fail**

```
uv run pytest tests/memory/semantic/test_decorator.py -v
```

Expected: `ImportError: cannot import name 'memory_tool'`.

- [ ] **Step 3: Implement decorator**

Create `src/ballast/memory/semantic/_decorator.py`:

```python
"""``@memory_tool`` — marks async methods on a SemanticSource for
exposure to the agent via ``SemanticMemory.as_tools()``."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")


@overload
def memory_tool(fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def memory_tool(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


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

    With overrides::

        @memory_tool(name="search_notes_by_tag",
                     description="Search notes by tag")
        async def find_by_tag(self, tag: str): ...

    The decorator does not wrap the callable — it attaches marker
    attributes (``__memory_tool__`` + optional overrides) that
    ``extract_memory_tools`` reads during introspection.
    """
    def decorate(f: Callable[P, R]) -> Callable[P, R]:
        f.__memory_tool__ = True                    # type: ignore[attr-defined]
        f.__memory_tool_name__ = name               # type: ignore[attr-defined]
        f.__memory_tool_description__ = description # type: ignore[attr-defined]
        return f

    if fn is not None:
        return decorate(fn)
    return decorate


__all__ = ["memory_tool"]
```

- [ ] **Step 4: Update package __init__**

Edit `src/ballast/memory/semantic/__init__.py`:

```python
"""Semantic memory — typed facts about the world via repo-wrappers."""
from ballast.memory.semantic._decorator import memory_tool
from ballast.memory.semantic._domain import DomainSemanticSource
from ballast.memory.semantic._protocol import SemanticSource

__all__ = ["DomainSemanticSource", "SemanticSource", "memory_tool"]
```

- [ ] **Step 5: Run — confirm pass**

```
uv run pytest tests/memory/semantic/test_decorator.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/memory/semantic/_decorator.py src/ballast/memory/semantic/__init__.py tests/memory/semantic/test_decorator.py
git commit -m "feat(memory): @memory_tool decorator for semantic-source methods"
```

---

## Task 3: `extract_memory_tools` introspection helper

**Files:**
- Create: `src/ballast/memory/semantic/_tools.py`
- Create: `tests/memory/semantic/test_tools.py`

- [ ] **Step 1: Write the failing test**

```python
"""``extract_memory_tools`` — introspection from SemanticSource → pydantic-ai Tools."""
from __future__ import annotations

from ballast.memory.semantic import DomainSemanticSource, memory_tool
from ballast.memory.semantic._tools import extract_memory_tools


class _SampleSource(DomainSemanticSource):
    name = "sample"

    @memory_tool
    async def find_by_tag(self, tag: str, limit: int = 10) -> list[str]:
        """Find samples tagged with `tag`. Most recent first."""
        return [f"{tag}:{i}" for i in range(limit)]

    @memory_tool(name="recent_samples")
    async def recent(self, days: int = 7) -> list[str]:
        """Recent samples."""
        return [f"day{i}" for i in range(days)]

    async def _internal_helper(self) -> None:
        """Not decorated — must not be exposed."""
        return None

    async def public_undecorated(self) -> str:
        """Public method without decorator — must not be exposed."""
        return "x"


def test_extracts_only_decorated_methods() -> None:
    tools = extract_memory_tools(_SampleSource())
    tool_names = {t.name for t in tools}
    assert tool_names == {"find_by_tag", "recent_samples"}


def test_uses_override_name_when_set() -> None:
    tools = extract_memory_tools(_SampleSource())
    name_method = next(t for t in tools if t.name == "recent_samples")
    # `name=` override wins; method's bare attribute name is "recent".
    assert name_method.name == "recent_samples"


def test_uses_docstring_as_description_fallback() -> None:
    tools = extract_memory_tools(_SampleSource())
    find_tool = next(t for t in tools if t.name == "find_by_tag")
    assert "tagged with" in (find_tool.description or "").lower()


def test_returns_empty_for_source_with_no_decorated_methods() -> None:
    class _Empty(DomainSemanticSource):
        name = "empty"
        async def helper(self) -> None: ...
    assert extract_memory_tools(_Empty()) == []
```

- [ ] **Step 2: Run — confirm fail**

```
uv run pytest tests/memory/semantic/test_tools.py -v
```

Expected: `ImportError` for `extract_memory_tools`.

- [ ] **Step 3: Implement**

Create `src/ballast/memory/semantic/_tools.py`:

```python
"""Introspection helpers — turn ``@memory_tool``-decorated methods on a
``SemanticSource`` into pydantic-ai ``Tool`` instances."""
from __future__ import annotations

import inspect

from pydantic_ai import Tool

from ballast.memory.semantic._protocol import SemanticSource


def extract_memory_tools(source: SemanticSource) -> list[Tool]:
    """Find every ``@memory_tool``-marked async method on ``source``
    and wrap each in a pydantic-ai ``Tool``.

    Tool name resolution: ``@memory_tool(name=...)`` override wins;
    otherwise the method's attribute name is used.

    Description resolution: ``@memory_tool(description=...)`` override
    wins; otherwise the method's docstring (stripped) is used; if both
    are absent, ``None``.

    Argument schema: derived by pydantic-ai from ``inspect.signature``
    + the method's type hints. The framework adds no extra wrapping.
    """
    tools: list[Tool] = []
    for attr_name, attr_value in inspect.getmembers(source, inspect.iscoroutinefunction):
        if not getattr(attr_value, "__memory_tool__", False):
            continue
        tool_name = (
            getattr(attr_value, "__memory_tool_name__", None) or attr_name
        )
        override_description = getattr(attr_value, "__memory_tool_description__", None)
        if override_description:
            description = override_description
        elif attr_value.__doc__:
            description = attr_value.__doc__.strip()
        else:
            description = None
        tools.append(Tool(
            attr_value,
            name=tool_name,
            description=description,
            takes_ctx=False,
        ))
    return tools


__all__ = ["extract_memory_tools"]
```

- [ ] **Step 4: Run — confirm pass**

```
uv run pytest tests/memory/semantic/test_tools.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory/semantic/_tools.py tests/memory/semantic/test_tools.py
git commit -m "feat(memory): extract_memory_tools introspection helper"
```

---

## Task 4: `SemanticMemory` facade + collision detection

**Files:**
- Create: `src/ballast/memory/semantic/_facade.py`
- Modify: `src/ballast/memory/semantic/__init__.py` (export `SemanticMemory`)
- Create: `tests/memory/semantic/test_facade.py`

- [ ] **Step 1: Write the failing test**

```python
"""``SemanticMemory`` facade — thin tool aggregator + collision detection."""
from __future__ import annotations

import pytest

from ballast.memory.semantic import (
    DomainSemanticSource,
    SemanticMemory,
    memory_tool,
)


class _NotesSource(DomainSemanticSource):
    name = "notes"

    @memory_tool
    async def find_by_tag(self, tag: str) -> list[str]:
        """Find notes by tag."""
        return [tag]


class _OrdersSource(DomainSemanticSource):
    name = "orders"

    @memory_tool
    async def recent(self, days: int = 7) -> list[str]:
        """Recent orders."""
        return [f"day{i}" for i in range(days)]


class _CollidingSource(DomainSemanticSource):
    name = "colliding"

    @memory_tool
    async def find_by_tag(self, tag: str) -> list[str]:
        """Same tool name as _NotesSource — should collide."""
        return [tag]


def test_aggregates_tools_across_sources() -> None:
    memory = SemanticMemory(sources=[_NotesSource(), _OrdersSource()])
    tool_names = {t.name for t in memory.as_tools()}
    assert tool_names == {"find_by_tag", "recent"}


def test_collision_raises_on_construction() -> None:
    with pytest.raises(ValueError, match="collision"):
        SemanticMemory(sources=[_NotesSource(), _CollidingSource()])


def test_collision_message_mentions_both_sources() -> None:
    with pytest.raises(ValueError) as exc_info:
        SemanticMemory(sources=[_NotesSource(), _CollidingSource()])
    msg = str(exc_info.value)
    assert "notes" in msg and "colliding" in msg
    assert "find_by_tag" in msg


def test_collision_can_be_resolved_by_name_override() -> None:
    class _Resolved(DomainSemanticSource):
        name = "resolved"

        @memory_tool(name="alt_find_by_tag")
        async def find_by_tag(self, tag: str) -> list[str]:
            return [tag]

    memory = SemanticMemory(sources=[_NotesSource(), _Resolved()])
    tool_names = {t.name for t in memory.as_tools()}
    assert tool_names == {"find_by_tag", "alt_find_by_tag"}


def test_empty_sources_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SemanticMemory(sources=[])


def test_list_sources_returns_registered() -> None:
    src1, src2 = _NotesSource(), _OrdersSource()
    memory = SemanticMemory(sources=[src1, src2])
    assert memory.list_sources() == [src1, src2]
```

- [ ] **Step 2: Run — confirm fail**

```
uv run pytest tests/memory/semantic/test_facade.py -v
```

Expected: `ImportError` for `SemanticMemory`.

- [ ] **Step 3: Implement facade**

Create `src/ballast/memory/semantic/_facade.py`:

```python
"""``SemanticMemory`` — thin tool aggregator for agent-pull exposure."""
from __future__ import annotations

from pydantic_ai import Tool

from ballast.memory.semantic._protocol import SemanticSource
from ballast.memory.semantic._tools import extract_memory_tools


class SemanticMemory:
    """Federation of ``SemanticSource`` impls for agent-pull tool exposure.

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
            tool
            for src in self._sources
            for tool in extract_memory_tools(src)
        ]

    def list_sources(self) -> list[SemanticSource]:
        """Return the registered sources (for introspection / debug)."""
        return list(self._sources)

    def _validate_no_collisions(self) -> None:
        """Raise ``ValueError`` if two sources expose the same tool name.

        Apps fix this by passing ``@memory_tool(name=...)`` on the
        method whose tool name should change.
        """
        owner_by_tool: dict[str, str] = {}
        for src in self._sources:
            for tool in extract_memory_tools(src):
                if tool.name in owner_by_tool:
                    raise ValueError(
                        f"SemanticMemory tool-name collision: {tool.name!r} "
                        f"defined by both {owner_by_tool[tool.name]!r} and "
                        f"{src.name!r}. Use @memory_tool(name=...) to "
                        "disambiguate."
                    )
                owner_by_tool[tool.name] = src.name


__all__ = ["SemanticMemory"]
```

- [ ] **Step 4: Update package __init__**

Edit `src/ballast/memory/semantic/__init__.py`:

```python
"""Semantic memory — typed facts about the world via repo-wrappers."""
from ballast.memory.semantic._decorator import memory_tool
from ballast.memory.semantic._domain import DomainSemanticSource
from ballast.memory.semantic._facade import SemanticMemory
from ballast.memory.semantic._protocol import SemanticSource

__all__ = [
    "DomainSemanticSource",
    "SemanticMemory",
    "SemanticSource",
    "memory_tool",
]
```

- [ ] **Step 5: Run — confirm pass**

```
uv run pytest tests/memory/semantic/test_facade.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/memory/semantic/_facade.py src/ballast/memory/semantic/__init__.py tests/memory/semantic/test_facade.py
git commit -m "feat(memory): SemanticMemory facade + collision detection"
```

---

## Task 5: `VectorSemanticSource` ABC + `_vector_search` helper

**Files:**
- Create: `src/ballast/memory/semantic/_vector.py`
- Modify: `src/ballast/memory/semantic/__init__.py` (export `VectorSemanticSource`)
- Create: `tests/memory/semantic/test_vector.py`

- [ ] **Step 1: Write the failing test**

The test uses the existing `session_factory` PG fixture (from `tests/persistence/conftest.py`) — same pattern as `test_vector_source.py` for Phase 1's VectorEpisodicSource. Place this test in `tests/persistence/` so it inherits the fixture:

Create `tests/persistence/test_semantic_vector.py`:

```python
"""``VectorSemanticSource._vector_search`` — generic cosine-search helper."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import Field, SQLModel

from ballast.memory.semantic import VectorSemanticSource, memory_tool


class _DocRow(SQLModel, table=True):
    """Throwaway test table — pads vectors to 1536 (the standard dim)."""

    __tablename__ = "_test_doc_row"

    id:        str = Field(primary_key=True)
    text:      str
    embedding: list[float] = Field(sa_column=Column(Vector(1536), nullable=False))


def _pad(vec: list[float]) -> list[float]:
    return vec + [0.0] * (1536 - len(vec))


class _FakeEmbedder:
    _table = {
        "machine learning":   _pad([1.0, 0.0, 0.0]),
        "ml model":           _pad([0.95, 0.05, 0.0]),
        "fashion trends":     _pad([0.0, 1.0, 0.0]),
    }
    async def embed(self, text: str) -> list[float]:
        return self._table[text]
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._table[t] for t in texts]


class _DocsSemantic(VectorSemanticSource):
    name = "docs"

    @memory_tool
    async def search(self, query: str, k: int = 3) -> list[_DocRow]:
        """Find docs whose text is semantically similar to `query`."""
        return await self._vector_search(
            query=query,
            table=_DocRow,
            embedding_column=_DocRow.embedding,
            k=k,
        )


@pytest.mark.asyncio
async def test_vector_search_returns_cosine_ordered(
    session_factory,
) -> None:
    # Seed three docs
    async with session_factory() as session:
        async with session.begin():
            session.add_all([
                _DocRow(id="ml", text="machine learning",
                        embedding=_pad([1.0, 0.0, 0.0])),
                _DocRow(id="fashion", text="fashion trends",
                        embedding=_pad([0.0, 1.0, 0.0])),
            ])

    src = _DocsSemantic(embedder=_FakeEmbedder(), sessionmaker=session_factory)
    results = await src.search(query="ml model", k=2)
    # "ml model" embedding ≈ "machine learning" (cosine ~1.0) >>
    # "fashion trends" (cosine ~0). ML row must come first.
    assert results[0].id == "ml"


@pytest.mark.asyncio
async def test_vector_search_respects_k(
    session_factory,
) -> None:
    async with session_factory() as session:
        async with session.begin():
            session.add_all([
                _DocRow(id=f"row-{i}", text="machine learning",
                        embedding=_pad([1.0 - i * 0.1, 0.0, 0.0]))
                for i in range(5)
            ])

    src = _DocsSemantic(embedder=_FakeEmbedder(), sessionmaker=session_factory)
    results = await src.search(query="ml model", k=3)
    assert len(results) == 3
```

Also add a non-PG unit test for the ABC shape:

Create `tests/memory/semantic/test_vector.py`:

```python
"""``VectorSemanticSource`` ABC — instantiation + protocol conformance."""
from __future__ import annotations

from typing import Any

from ballast.memory.semantic import (
    SemanticSource,
    VectorSemanticSource,
    memory_tool,
)


class _DummyEmbedder:
    async def embed(self, text: str) -> list[float]: return [0.0]
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: return [[0.0] for _ in texts]


class _DummyMaker:
    """Just a sessionmaker placeholder for ABC-shape testing."""
    def __call__(self): raise NotImplementedError


class _SearchSource(VectorSemanticSource):
    name = "search"

    @memory_tool
    async def search(self, query: str) -> list[Any]:
        """Search."""
        return []


def test_subclass_satisfies_semantic_source_protocol() -> None:
    src = _SearchSource(embedder=_DummyEmbedder(), sessionmaker=_DummyMaker())
    assert isinstance(src, SemanticSource)
    assert src.name == "search"


def test_subclass_must_set_name() -> None:
    class _NoName(VectorSemanticSource):
        pass
    with pytest.raises(AttributeError):
        _NoName(embedder=_DummyEmbedder(), sessionmaker=_DummyMaker()).name  # type: ignore[misc]
```

`pytest` import — add at top: `import pytest`.

- [ ] **Step 2: Run — confirm fail**

```
uv run pytest tests/memory/semantic/test_vector.py tests/persistence/test_semantic_vector.py -v
```

Expected: ImportError for `VectorSemanticSource`.

- [ ] **Step 3: Implement**

Create `src/ballast/memory/semantic/_vector.py`:

```python
"""``VectorSemanticSource`` — convenience base for free-text RAG sources."""
from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import SQLModel, select

from ballast.capabilities.helpers.embedder import Embedder
from ballast.memory.semantic._protocol import SemanticSource


class VectorSemanticSource(SemanticSource, ABC):
    """Base class for semantic sources backed by embedded free-text fields.

    Provides typical wiring (``embedder`` + ``sessionmaker``) and a
    helper ``_vector_search`` for the common cosine-distance query.
    Subclasses decide what to expose via ``@memory_tool`` — one search
    method per indexed corpus, or one method total, app's choice.

    The framework provides only the read-side helper. Apps own the
    embedding row schema and write-side indexing (typically a post-save
    hook on the domain repo).
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
        embedding_column: Any,           # e.g. MyRow.embedding
        k: int,
        where: Any | None = None,        # optional SQLAlchemy WHERE clause
    ) -> list[Any]:
        """Embed ``query`` and cosine-search ``table`` ordered by distance.

        Returns up to ``k`` rows. Subclasses typically project the
        result into a domain type before returning to the caller.
        """
        query_vec = await self._embedder.embed(query)
        async with self._sessionmaker() as session:
            stmt = select(table).order_by(embedding_column.cosine_distance(query_vec))
            if where is not None:
                stmt = stmt.where(where)
            stmt = stmt.limit(k)
            result = await session.execute(stmt)
            return list(result.scalars().all())


__all__ = ["VectorSemanticSource"]
```

- [ ] **Step 4: Update package __init__**

Edit `src/ballast/memory/semantic/__init__.py`:

```python
"""Semantic memory — typed facts about the world via repo-wrappers."""
from ballast.memory.semantic._decorator import memory_tool
from ballast.memory.semantic._domain import DomainSemanticSource
from ballast.memory.semantic._facade import SemanticMemory
from ballast.memory.semantic._protocol import SemanticSource
from ballast.memory.semantic._vector import VectorSemanticSource

__all__ = [
    "DomainSemanticSource",
    "SemanticMemory",
    "SemanticSource",
    "VectorSemanticSource",
    "memory_tool",
]
```

- [ ] **Step 5: Register test table in pg conftest**

Edit `tests/persistence/conftest.py` — add to the top-level model-import block (so `SQLModel.metadata` includes the throwaway `_test_doc_row` table when running pgvector smoke):

```python
# In tests/persistence/conftest.py near the existing imports:
import tests.persistence.test_semantic_vector  # noqa: F401 — registers _DocRow
```

If this circular-imports awkwardly (test module imports the conftest), instead inline the `_DocRow` class into `tests/persistence/_test_models.py` and import THAT from both the conftest and the test. Pragmatic choice based on what works without import order issues — if the noqa-import works, prefer it (one file).

- [ ] **Step 6: Run — confirm pass**

```
uv run pytest tests/memory/semantic/test_vector.py tests/persistence/test_semantic_vector.py -v
```

Expected: unit tests pass; PG tests pass (or skip cleanly without Docker).

- [ ] **Step 7: Commit**

```bash
git add src/ballast/memory/semantic/_vector.py src/ballast/memory/semantic/__init__.py tests/memory/semantic/test_vector.py tests/persistence/test_semantic_vector.py tests/persistence/conftest.py
git commit -m "feat(memory): VectorSemanticSource ABC + _vector_search helper"
```

---

## Task 6: `Ballast.with_semantic_memory(...)` + rename `with_memory` → `with_episodic_memory`

**Files:**
- Modify: `src/ballast/app.py`
- Create: `tests/app/test_with_semantic_memory.py`

- [ ] **Step 1: Find existing builder pattern**

```
grep -n "def with_memory\|def with_episodic_memory\|self._memory\|self._episodic_memory" src/ballast/app.py
```

Phase 1 already has `with_memory(EpisodicMemory)` and `self._memory`. This task renames the method and attr for symmetry with the new semantic setter.

- [ ] **Step 2: Write the failing test**

```python
"""``Ballast.with_semantic_memory`` + ``with_episodic_memory`` (rename)."""
from __future__ import annotations

import warnings

import pytest

from ballast.app import Ballast
from ballast.memory.episodic import EpisodicMemory
from ballast.memory.semantic import (
    DomainSemanticSource,
    SemanticMemory,
    memory_tool,
)
from ballast.settings import BallastSettings


class _EpisodicSource:
    name = "ep"
    async def recall(self, **_): return []
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


class _NotesSemantic(DomainSemanticSource):
    name = "notes"
    @memory_tool
    async def find_by_tag(self, tag: str) -> list[str]: return [tag]


def test_with_semantic_memory_installs_facade() -> None:
    sm = SemanticMemory(sources=[_NotesSemantic()])
    app = Ballast(BallastSettings()).with_semantic_memory(sm)
    assert app._semantic_memory is sm


def test_with_episodic_memory_replaces_with_memory() -> None:
    em = EpisodicMemory(sources=[_EpisodicSource()])
    app = Ballast(BallastSettings()).with_episodic_memory(em)
    assert app._episodic_memory is em


def test_with_memory_alias_still_works_but_warns() -> None:
    """Backward-compat alias — emits DeprecationWarning."""
    em = EpisodicMemory(sources=[_EpisodicSource()])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        app = Ballast(BallastSettings()).with_memory(em)
        assert app._episodic_memory is em
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_both_setters_chain() -> None:
    em = EpisodicMemory(sources=[_EpisodicSource()])
    sm = SemanticMemory(sources=[_NotesSemantic()])
    app = (
        Ballast(BallastSettings())
        .with_episodic_memory(em)
        .with_semantic_memory(sm)
    )
    assert app._episodic_memory is em
    assert app._semantic_memory is sm
```

- [ ] **Step 3: Run — confirm fail**

```
uv run pytest tests/app/test_with_semantic_memory.py -v
```

Expected: `AttributeError` for `with_semantic_memory` and/or `_episodic_memory`.

- [ ] **Step 4: Patch `src/ballast/app.py`**

In the existing `TYPE_CHECKING` block, add:

```python
if TYPE_CHECKING:
    # ... existing ...
    from ballast.memory.semantic._facade import SemanticMemory
```

In `Ballast.__init__`, add (next to `self._memory: ... = None`):

```python
self._episodic_memory: "EpisodicMemory | None" = None
self._semantic_memory: "SemanticMemory | None" = None
```

Replace the existing `with_memory(...)` method body. New version:

```python
def with_episodic_memory(
    self,
    memory: "EpisodicMemory",
    *,
    scope_builder: "Callable[[], Scope] | None" = None,
) -> "Ballast":
    """Wire an EpisodicMemory facade + optional default scope-builder.

    Replaces the deprecated ``with_memory`` (kept as a backward-
    compatible alias for one release window).
    """
    self._episodic_memory = memory
    self._memory = memory   # back-compat shadow for any Phase 1 consumers
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


def with_memory(
    self,
    memory: "EpisodicMemory",
    *,
    scope_builder: "Callable[[], Scope] | None" = None,
) -> "Ballast":
    """Deprecated — use ``with_episodic_memory(...)`` instead.

    Kept as a backward-compatible alias for one release window so
    existing Phase 1 wiring continues to work.
    """
    import warnings
    warnings.warn(
        "Ballast.with_memory is deprecated; use with_episodic_memory.",
        DeprecationWarning, stacklevel=2,
    )
    return self.with_episodic_memory(memory, scope_builder=scope_builder)
```

- [ ] **Step 5: Run — confirm pass**

```
uv run pytest tests/app/test_with_semantic_memory.py tests/app/ -v
```

Expected: green (4 new tests + existing app tests still pass).

- [ ] **Step 6: Commit**

```bash
git add src/ballast/app.py tests/app/test_with_semantic_memory.py
git commit -m "feat(app): Ballast.with_semantic_memory + rename with_memory → with_episodic_memory (alias kept)"
```

---

## Task 7: Phase 1 consumer migration — `_memory` → `_episodic_memory`

**Files:**
- Modify: `examples/notes-app/backend/src/notes_app/agents/notes.py`
- Modify: `examples/notes-app/backend/src/notes_app/workflows/create_note.py`

- [ ] **Step 1: Grep for consumers**

```
grep -rn "_memory\b\|getattr(get_ballast(), \"_memory\"" examples/notes-app/backend/src/notes_app/ 2>&1 | grep -v __pycache__
```

Expected callsites (from Phase 1 commits):
- `notes_app/agents/notes.py` — `default_notes_capabilities()` reads `_memory`
- `notes_app/workflows/create_note.py` — recall logic reads `_memory`

- [ ] **Step 2: Patch `notes_app/agents/notes.py`**

Replace `getattr(get_ballast(), "_memory", None)` → `getattr(get_ballast(), "_episodic_memory", None)`. The local variable name `memory` stays (it's a local; the attr read is what changes).

- [ ] **Step 3: Patch `notes_app/workflows/create_note.py`**

Same substitution: `getattr(get_ballast(), "_memory", None)` → `getattr(get_ballast(), "_episodic_memory", None)`.

- [ ] **Step 4: Run notes-app suite — confirm no regression**

```
cd examples/notes-app/backend && uv run pytest -q
```

Expected: 18 passed, 2 skipped (or equivalent — same green count as before).

- [ ] **Step 5: Commit**

```bash
git add examples/notes-app/backend/src/notes_app/agents/notes.py examples/notes-app/backend/src/notes_app/workflows/create_note.py
git commit -m "refactor(notes-app): _memory → _episodic_memory after Ballast rename"
```

---

## Task 8: Public API exports

**Files:**
- Modify: `src/ballast/memory/__init__.py`
- Modify: `src/ballast/__init__.py`

- [ ] **Step 1: Update `src/ballast/memory/__init__.py`**

```python
"""CoALA-inspired memory subsystem."""
from ballast.memory._scope import Scope
from ballast.memory.semantic import (
    DomainSemanticSource,
    SemanticMemory,
    SemanticSource,
    VectorSemanticSource,
    memory_tool,
)

__all__ = [
    "DomainSemanticSource",
    "Scope",
    "SemanticMemory",
    "SemanticSource",
    "VectorSemanticSource",
    "memory_tool",
]
```

- [ ] **Step 2: Update `src/ballast/__init__.py`**

Find the existing memory re-exports block (Phase 1 added `Scope`, `Episode`, `EpisodicMemory`, etc.). Append the semantic exports:

```python
# Append in the memory re-exports section:
from ballast.memory.semantic import (
    DomainSemanticSource,
    SemanticMemory,
    SemanticSource,
    VectorSemanticSource,
    memory_tool,
)
```

Extend `__all__` to include these five names (alphabetical insertion into the existing list).

- [ ] **Step 3: Smoke import**

```
uv run python -c "from ballast import SemanticMemory, SemanticSource, DomainSemanticSource, VectorSemanticSource, memory_tool; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Verify nothing broke**

```
uv run pytest tests/ -q
```

Expected: full framework suite green.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/memory/__init__.py src/ballast/__init__.py
git commit -m "feat(ballast): re-export semantic memory public API"
```

---

## Task 9: Notes-app — `NotesSemantic` source + `notes_semantic` singleton

**Files:**
- Create: `examples/notes-app/backend/src/notes_app/memory/__init__.py` (empty)
- Create: `examples/notes-app/backend/src/notes_app/memory/semantic_sources.py`
- Modify: `examples/notes-app/backend/src/notes_app/repositories/note.py` (add `find_by_tag`, `recent` if missing)
- Create: `examples/notes-app/backend/tests/test_notes_semantic.py`

- [ ] **Step 1: Check existing `notes_repo` surface**

```
grep -n "async def" examples/notes-app/backend/src/notes_app/repositories/note.py | head
```

The example needs at minimum `find_by_tag(tag, limit)` and `recent(days)`. If the repo doesn't have them, add minimal impls. The notes-app's existing `Note` model may not have a `tag` field — if so, use whichever existing dimensions are queryable (e.g. `created_at` for `recent`; substring `search` instead of `find_by_tag`).

**Pragmatic adaptation**: if the existing Note model has no `tag` column, change `NotesSemantic` to expose `recent(days: int)` and `search(query: str)` (where `search` wraps the existing `search()` method on the in-memory repo, which we know exists from earlier code review). Don't add new columns / migrations to the notes-app domain in this phase — that's domain work, not memory work.

- [ ] **Step 2: Write the failing test**

```python
"""``NotesSemantic`` — notes-app semantic source over notes_repo."""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from ballast.memory.semantic import SemanticSource
from notes_app.memory.semantic_sources import NotesSemantic, notes_semantic
from notes_app.repositories.note import InMemoryNoteRepository


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemoryNoteRepository]:
    fresh = InMemoryNoteRepository()
    monkeypatch.setattr("notes_app.repositories.note.notes_repo", fresh)
    yield fresh


def test_module_singleton_exists_and_named() -> None:
    assert isinstance(notes_semantic, NotesSemantic)
    assert notes_semantic.name == "notes"


def test_satisfies_semantic_source_protocol() -> None:
    assert isinstance(notes_semantic, SemanticSource)


@pytest.mark.asyncio
async def test_recent_returns_recent_notes(repo: InMemoryNoteRepository) -> None:
    n1 = await repo.create(title="t1", body="b1")
    n2 = await repo.create(title="t2", body="b2")
    results = await notes_semantic.recent(days=30)
    ids = {n.id for n in results}
    assert {n1.id, n2.id} <= ids


@pytest.mark.asyncio
async def test_search_substring(repo: InMemoryNoteRepository) -> None:
    await repo.create(title="ml notes", body="machine learning content")
    await repo.create(title="fashion", body="trends")
    results = await notes_semantic.search(query="machine")
    titles = {n.title for n in results}
    assert "ml notes" in titles
    assert "fashion" not in titles
```

- [ ] **Step 3: Run — confirm fail**

```
cd examples/notes-app/backend && uv run pytest tests/test_notes_semantic.py -v
```

Expected: `ModuleNotFoundError` for `notes_app.memory`.

- [ ] **Step 4: Implement the source**

Create `examples/notes-app/backend/src/notes_app/memory/__init__.py` (empty).

Create `examples/notes-app/backend/src/notes_app/memory/semantic_sources.py`:

```python
"""Notes-app semantic memory sources — wrap ``notes_repo`` for agent exposure.

This module declares the typed accessors the LLM agent sees as tools
when ``SemanticMemory(sources=[notes_semantic])`` is wired into the
Ballast builder.

Each method is marked with ``@memory_tool``. Its docstring becomes the
tool description the LLM reads when deciding which tool to call. Keep
docstrings concrete + decision-oriented.
"""
from __future__ import annotations

from ballast.memory.semantic import DomainSemanticSource, memory_tool

from notes_app.models.note import Note


class NotesSemantic(DomainSemanticSource):
    """Read-only semantic view over the user's notes."""

    name = "notes"

    @memory_tool
    async def recent(self, days: int = 7) -> list[Note]:
        """Return notes the user has created or edited in the last
        `days` days. Most recent first. Use when the user references
        recent work without a specific identifier."""
        from notes_app.repositories.note import notes_repo  # noqa: PLC0415
        # InMemoryNoteRepository may not have a `recent()` method — fall back
        # to list_() and filter in-Python.
        all_notes = await notes_repo.list_()
        from datetime import datetime, timedelta, timezone  # noqa: PLC0415
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [
            n for n in all_notes
            if getattr(n, "created_at", cutoff) >= cutoff
        ]

    @memory_tool
    async def search(self, query: str, limit: int = 10) -> list[Note]:
        """Find notes whose title or body matches `query` (substring,
        case-insensitive). Use when the user references a note by topic
        or keyword rather than by id."""
        from notes_app.repositories.note import notes_repo  # noqa: PLC0415
        hits = await notes_repo.search(query)
        return hits[:limit]


notes_semantic: NotesSemantic = NotesSemantic()
```

The `recent()` impl handles the case where the repo doesn't have a native `recent()` method (filters in-Python). If the repo DOES have a `recent()` method, prefer it (test both paths during implementation; if only one works without modifying the repo, document the choice).

- [ ] **Step 5: Run — confirm pass**

```
cd examples/notes-app/backend && uv run pytest tests/test_notes_semantic.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add examples/notes-app/backend/src/notes_app/memory examples/notes-app/backend/tests/test_notes_semantic.py
git commit -m "feat(notes-app): NotesSemantic source + notes_semantic singleton"
```

---

## Task 10: Notes-app — wire `SemanticMemory` in main.py

**Files:**
- Modify: `examples/notes-app/backend/src/notes_app/main.py`

- [ ] **Step 1: Read current builder chain**

```
grep -n "with_episodic_memory\|with_memory\|with_approval_repo\|_build_episodic_memory" examples/notes-app/backend/src/notes_app/main.py
```

Phase 1 already calls `.with_memory(_build_episodic_memory())`. T6 of THIS plan introduces `with_episodic_memory` — update the call site.

- [ ] **Step 2: Edit `main.py`**

Add imports near the other ballast / memory imports:

```python
from ballast.memory.semantic import SemanticMemory
from notes_app.memory.semantic_sources import notes_semantic
```

In the builder chain:

1. Rename `.with_memory(_build_episodic_memory())` → `.with_episodic_memory(_build_episodic_memory())`.
2. Add a new line: `.with_semantic_memory(SemanticMemory(sources=[notes_semantic]))`.

Resulting chain shape:

```python
ballast = (
    Ballast(...)
    ...existing...
    .with_approval_repo(...)
    .with_episodic_memory(_build_episodic_memory())
    .with_semantic_memory(SemanticMemory(sources=[notes_semantic]))
    ...
)
```

- [ ] **Step 3: Smoke run**

```
cd examples/notes-app/backend && uv run pytest -q
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add examples/notes-app/backend/src/notes_app/main.py
git commit -m "feat(notes-app): wire SemanticMemory(sources=[notes_semantic]) via builder"
```

---

## Task 11: Notes-app — `NotesAgent` reads semantic tools

**Files:**
- Modify: `examples/notes-app/backend/src/notes_app/agents/notes.py`

- [ ] **Step 1: Locate `NotesAgent.build_agent`**

```
grep -n "def build_agent" examples/notes-app/backend/src/notes_app/agents/notes.py
```

- [ ] **Step 2: Edit `build_agent`**

Add `tools=` argument to the `Agent(...)` constructor, sourcing semantic tools from the wired `SemanticMemory`:

```python
class NotesAgent(DurableAgent):
    name = "notes"
    metadata_model = None

    def build_agent(self) -> Agent[NoteToolDeps, Any]:
        # Pull semantic-memory tools if the engine wired any.
        semantic = getattr(get_ballast(), "_semantic_memory", None)
        extra_tools = semantic.as_tools() if semantic is not None else []
        return Agent(
            model=build_openrouter_model(),
            output_type=[str, DeferredToolRequests],
            deps_type=NoteToolDeps,
            system_prompt=SYSTEM_PROMPT,
            capabilities=default_notes_capabilities(),
            tools=extra_tools,
        )
```

`get_ballast` is already imported in this file (Phase 1 RememberTurn wiring). If for some reason it's not, add `from ballast import get_ballast`.

- [ ] **Step 3: Smoke run**

```
cd examples/notes-app/backend && uv run pytest -q
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add examples/notes-app/backend/src/notes_app/agents/notes.py
git commit -m "feat(notes-app): NotesAgent reads semantic-memory tools at build"
```

---

## Task 12: Final smoke

- [ ] **Step 1: Run framework suite**

```
uv run pytest tests/ --tb=short -q
```

Expected: all green (Phase 1 tests + new semantic tests, no regressions).

- [ ] **Step 2: Run notes-app suite**

```
cd examples/notes-app/backend && uv run pytest --tb=short -q
```

Expected: all green (including new `test_notes_semantic.py`).

- [ ] **Step 3: Manual smoke (optional, requires running stack)**

If the user wants to see the agent actually use a semantic tool:

```
cd examples/notes-app/backend && uv run uvicorn notes_app.main:app --reload &
cd examples/notes-app/frontend && pnpm dev &
```

In the chat:

1. Ask the agent to "create three notes about different topics".
2. Verify each saved.
3. Ask "find my recent notes" — agent should call `recent(days=7)` via the semantic tool and return the list.
4. Ask "search my notes for X" — agent should call `search(query='X')`.

If steps 3-4 don't trigger the semantic tools, check that `NotesAgent.build_agent` is actually passing `tools=` and that `Agent(...)` accepts it (check pydantic-ai version — if `tools=` isn't a constructor arg in the installed pydantic-ai, adapt to whichever tool-registration API IS available, e.g. `agent.tool_plain(fn)` after construction).

- [ ] **Step 4: Commit (any cleanup)**

```bash
git status && git diff
# commit any trailing tweaks
```

---

## Follow-up plan (out of scope here)

A separate spec / plan should cover:

1. **Phase 3 — Procedural memory** — `WorkflowRegistry` with introspection; agent invokes named workflows as skills.
2. **Phase 4 — Learning loop** — cluster recent episodes → HITL-suggested skill consolidation.
3. **Auto-attach capability** — implicit injection of `as_tools()` into every `DurableAgent` build, removing the explicit splat.
4. **MCP-backed semantic sources** — `MCPSemanticSource(mcp_server)` bridges external MCP servers into the semantic-tool surface.
5. **Cache layer for semantic tool calls** — when use case appears.
6. **Public `get_ballast().memory` accessor** to replace the private `_episodic_memory` / `_semantic_memory` reads in notes-app.

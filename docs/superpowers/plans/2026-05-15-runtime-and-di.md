# Runtime + DI (Sub-project #3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the L3 runtime layer of `ballast-ai`: DBOS integration, `Det` helpers upgraded to durable `@DBOS.step`, type-keyed DI `Container`, `ServiceProvider` Protocol, `Engine` orchestrator with bootstrap-time invariants, and a custom ruff plugin enforcing the determinism boundary (`STATEFLOW001-013` rules from spec 2E.1 + 4G).

**Architecture:** DBOS owns the durable runtime — it persists workflow / step state into Postgres so crashes are recovered by replay. `Det.uuid_for` upgraded from plain async function (Sub-project #1 placeholder) to `@DBOS.step` so its result is durably recorded. `Container` is type-keyed DI (no string-key service-locator pattern per spec 4A.0.7). `ServiceProvider` is single-phase `register(container)` (per spec 4A.0.13 — reverted from initial two-phase design). `Engine` boots providers in user-declared order and runs bootstrap-time invariants (Tool coverage, Alembic pending check, etc). Custom ruff rules block forbidden patterns (Repository call outside `@DBOS.step`, agent.run outside `@DBOS.step`, etc) at lint time.

**Tech Stack:** `dbos-transact` (durable workflow runtime), the existing SQLAlchemy/Postgres infra from Sub-project #2, a custom `ruff` plugin module.

**Spec sections covered:** 1.4 #1 (Explicit DI), 1.4 #7 (Workflow/Step determinism), 1.4 #18 (DBOS as the no-globals exception), 2E.1 (Determinism boundary), 2E.2 (Bootstrap / ServiceProvider), 4A Delta 4 (Det.uuid_for safety — Critical Fix #1), 4A.0.7 (Container is explicit DI not service locator), 4A.0.13 (ServiceProvider single-phase), 4F (MVP L3 scope), 4G (STATEFLOW lint rules), 4H (bootstrap rules).

**Scope vs deferred:** v1 implements DBOS plugin integration + Container + ServiceProvider + Engine + Det upgrade + first 9 STATEFLOW lint rules (the determinism boundary ones — STATEFLOW001-007 + 013 + extension for Repository tenant_id). Deferred to later sub-projects: full `STATEFLOW008-012` (those require knowledge of higher-layer patterns), Logfire/observability instrumentation (Sub-project #7 or future #6), advanced DBOS features like queues with concurrency limits (#5 patterns).

---

## File Structure

```
src/ballast/
├── runtime/
│   ├── __init__.py                # public: Det, Container, ServiceProvider, Engine, ...
│   ├── det.py                     # upgraded — Det.* methods now @DBOS.step
│   ├── idempotency.py             # unchanged from Sub-project #1
│   ├── container.py               # Container Protocol + DefaultContainer
│   ├── provider.py                # ServiceProvider Protocol
│   ├── engine.py                  # Engine orchestrator
│   └── dbos_setup.py              # DBOS configuration + initialization helpers
├── providers/
│   ├── __init__.py                # CoreProvider, PersistenceProvider exports
│   ├── core.py                    # CoreProvider — binds Det, EventDispatcher (stub)
│   └── persistence.py             # PersistenceProvider — binds session_factory + Repos
├── ruff/
│   ├── __init__.py
│   └── stateflow_rules.py         # AST-based rule definitions
└── (existing modules untouched)

tests/
├── runtime/
│   ├── test_container.py
│   ├── test_provider.py
│   ├── test_engine.py
│   ├── test_dbos_det_step.py      # verifies Det.* is @DBOS.step
│   └── test_dbos_workflow_smoke.py
├── providers/
│   ├── test_core_provider.py
│   └── test_persistence_provider.py
└── lint/
    └── test_stateflow_rules.py
```

---

## Task 1: Add `dbos-transact` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add to `[project] dependencies`**

```toml
dependencies = [
    "pydantic>=2.7",
    "pydantic-ai>=0.0.13",
    "sqlmodel>=0.0.22",
    "sqlalchemy[asyncio]>=2.0",
    "alembic>=1.13",
    "asyncpg>=0.29",
    "dbos>=1.0",
]
```

- [ ] **Step 2: Sync and verify nothing broke**

```bash
uv sync --extra dev
uv run pytest && uv run mypy src && uv run ruff check
```

Expected: 117 passed + 9 skipped (Sub-project #2 state), all green.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add dbos-transact dependency"
```

---

## Task 2: Container Protocol + DefaultContainer

Type-keyed DI registry. No string keys (per spec 4A.0.7).

**Files:**
- Create: `src/ballast/runtime/container.py`
- Create: `tests/runtime/__init__.py` (if not present)
- Create: `tests/runtime/test_container.py`

- [ ] **Step 1: Write failing tests**

`tests/runtime/test_container.py`:

```python
from typing import Protocol

import pytest

from ballast.runtime.container import Container, DefaultContainer


class Greeter(Protocol):
    def hello(self) -> str: ...


class ConcreteGreeter:
    def hello(self) -> str:
        return "hi"


def test_bind_and_get_singleton():
    c = DefaultContainer()
    c.bind(Greeter, lambda _: ConcreteGreeter())
    g = c.get(Greeter)
    assert g.hello() == "hi"


def test_singleton_returns_same_instance():
    c = DefaultContainer()
    c.bind(Greeter, lambda _: ConcreteGreeter())
    assert c.get(Greeter) is c.get(Greeter)


def test_non_singleton_returns_fresh_instance():
    c = DefaultContainer()
    c.bind(Greeter, lambda _: ConcreteGreeter(), singleton=False)
    assert c.get(Greeter) is not c.get(Greeter)


def test_get_unknown_type_raises_key_error():
    c = DefaultContainer()
    with pytest.raises(KeyError, match="Greeter"):
        c.get(Greeter)


def test_factory_receives_container_for_dependencies():
    """Factory can resolve other deps via the container parameter."""
    class Foo:
        pass

    class Bar:
        def __init__(self, foo: Foo):
            self.foo = foo

    c = DefaultContainer()
    c.bind(Foo, lambda _: Foo())
    c.bind(Bar, lambda c: Bar(c.get(Foo)))

    assert isinstance(c.get(Bar).foo, Foo)


def test_default_container_satisfies_protocol():
    assert isinstance(DefaultContainer(), Container)
```

- [ ] **Step 2: Run — fail**

```bash
uv run pytest tests/runtime/test_container.py -v
```

- [ ] **Step 3: Implement**

`src/ballast/runtime/container.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Container(Protocol):
    """Type-keyed DI registry.

    Bindings are keyed by Protocol or concrete type. Factories receive the
    Container so they can resolve their own dependencies.

    No string keys (per spec 4A.0.7 — service-locator pattern forbidden).
    """

    def bind(
        self,
        protocol: type[T],
        factory: Callable[["Container"], T],
        *,
        singleton: bool = True,
    ) -> None: ...

    def get(self, protocol: type[T]) -> T: ...


class DefaultContainer:
    """Minimal type-keyed DI container.

    Singleton by default — `singleton=False` for fresh-per-resolution
    bindings (rare; only needed for stateful per-request services).
    """

    def __init__(self) -> None:
        self._factories: dict[type, tuple[Callable[[Container], Any], bool]] = {}
        self._instances: dict[type, Any] = {}

    def bind(
        self,
        protocol: type[T],
        factory: Callable[[Container], T],
        *,
        singleton: bool = True,
    ) -> None:
        self._factories[protocol] = (factory, singleton)
        # Drop any previously-cached singleton if rebinding
        self._instances.pop(protocol, None)

    def get(self, protocol: type[T]) -> T:
        if protocol not in self._factories:
            raise KeyError(f"No binding for {protocol.__name__}")
        factory, singleton = self._factories[protocol]
        if singleton:
            if protocol not in self._instances:
                self._instances[protocol] = factory(self)
            return self._instances[protocol]
        return factory(self)
```

- [ ] **Step 4: Tests pass (6 new)**

- [ ] **Step 5: Full suite + mypy + ruff**

- [ ] **Step 6: Commit**

```bash
git add src/ballast/runtime/container.py tests/runtime
git commit -m "feat(runtime): Container Protocol + DefaultContainer (type-keyed DI)"
```

---

## Task 3: ServiceProvider Protocol

Single-phase per spec 4A.0.13.

**Files:**
- Create: `src/ballast/runtime/provider.py`
- Create: `tests/runtime/test_provider.py`

- [ ] **Step 1: Failing tests**

`tests/runtime/test_provider.py`:

```python
import pytest

from ballast.runtime.container import Container, DefaultContainer
from ballast.runtime.provider import ServiceProvider


class Greeter:
    def hello(self) -> str:
        return "hello"


class GreeterProvider:
    async def register(self, container: Container) -> None:
        container.bind(Greeter, lambda _: Greeter())


@pytest.mark.asyncio
async def test_provider_protocol_register_binds_into_container():
    c = DefaultContainer()
    p: ServiceProvider = GreeterProvider()
    await p.register(c)
    assert isinstance(c.get(Greeter), Greeter)


@pytest.mark.asyncio
async def test_concrete_provider_satisfies_protocol():
    assert isinstance(GreeterProvider(), ServiceProvider)
```

- [ ] **Step 2: Run — fail**

- [ ] **Step 3: Implement**

`src/ballast/runtime/provider.py`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ballast.runtime.container import Container


@runtime_checkable
class ServiceProvider(Protocol):
    """Single-phase provider (per spec 4A.0.13).

    Replaces the original two-phase register/boot ceremony (which was
    enterprise overhead for 8 manually-ordered providers, per code-review).

    Providers are registered in user-declared order in Engine constructor.
    If provider B depends on provider A's bindings, B must come after A
    in the list.

    Engine runs post-registration invariants (Alembic check, Tool coverage,
    etc) AFTER all providers have registered.
    """

    async def register(self, container: Container) -> None:
        """Bind everything this provider owns + initialise as needed.

        Free to perform async I/O (e.g. preheating caches), but must not
        depend on other providers' instances mid-registration — only on
        their bindings (resolved lazily on `container.get`).
        """
```

- [ ] **Step 4: Tests pass (2 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/runtime/provider.py tests/runtime/test_provider.py
git commit -m "feat(runtime): ServiceProvider Protocol (single-phase per 4A.0.13)"
```

---

## Task 4: Engine orchestrator + bootstrap invariants framework

**Files:**
- Create: `src/ballast/runtime/engine.py`
- Create: `tests/runtime/test_engine.py`

- [ ] **Step 1: Failing tests**

`tests/runtime/test_engine.py`:

```python
import pytest

from ballast.runtime.container import Container, DefaultContainer
from ballast.runtime.engine import Engine, EngineInvariantViolation
from ballast.runtime.provider import ServiceProvider


class _Service:
    initialised: bool = False


class _Provider:
    async def register(self, container: Container) -> None:
        container.bind(_Service, lambda _: _Service())


@pytest.mark.asyncio
async def test_engine_boots_providers_in_order():
    order: list[str] = []

    class FirstProvider:
        async def register(self, c: Container) -> None:
            order.append("first")
            c.bind(int, lambda _: 1)

    class SecondProvider:
        async def register(self, c: Container) -> None:
            order.append("second")
            # Verifies it sees first's binding (no late-resolution needed)
            assert c.get(int) == 1
            c.bind(str, lambda _: "two")

    engine = Engine(providers=[FirstProvider(), SecondProvider()])
    await engine.boot()
    assert order == ["first", "second"]


@pytest.mark.asyncio
async def test_engine_container_accessible_after_boot():
    engine = Engine(providers=[_Provider()])
    await engine.boot()
    assert isinstance(engine.container.get(_Service), _Service)


@pytest.mark.asyncio
async def test_engine_runs_invariants_after_all_providers_registered():
    invariant_seen_int: int | None = None

    async def check_int_bound(c: Container) -> None:
        nonlocal invariant_seen_int
        invariant_seen_int = c.get(int)

    class IntProvider:
        async def register(self, c: Container) -> None:
            c.bind(int, lambda _: 42)

    engine = Engine(providers=[IntProvider()], invariants=[check_int_bound])
    await engine.boot()
    assert invariant_seen_int == 42


@pytest.mark.asyncio
async def test_invariant_violation_blocks_boot():
    async def always_fail(c: Container) -> None:
        raise EngineInvariantViolation("nope")

    engine = Engine(providers=[], invariants=[always_fail])
    with pytest.raises(EngineInvariantViolation):
        await engine.boot()


@pytest.mark.asyncio
async def test_boot_is_idempotent_via_same_engine_instance():
    """Calling boot twice on same Engine raises to prevent silent re-registration."""
    engine = Engine(providers=[_Provider()])
    await engine.boot()
    with pytest.raises(RuntimeError, match="already booted"):
        await engine.boot()
```

- [ ] **Step 2: Run — fail**

- [ ] **Step 3: Implement**

`src/ballast/runtime/engine.py`:

```python
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from ballast.runtime.container import Container, DefaultContainer
from ballast.runtime.provider import ServiceProvider

T = TypeVar("T")

Invariant = Callable[[Container], Awaitable[None]]


class EngineInvariantViolation(Exception):
    """Raised when a bootstrap-time invariant check fails.

    Engine.boot propagates this so the application start fails fast
    instead of running with a broken configuration (per spec 4H).
    """


class Engine:
    """Orchestrator: registers providers + runs bootstrap invariants.

    Per spec 4H:
    - Providers register in user-declared order (no auto-DAG)
    - All providers register before any invariants run
    - Invariants raise EngineInvariantViolation to abort startup
    - Container is exposed publicly so FastAPI / CLI callers can
      `Depends(get_container)` rather than reaching for a global.
    """

    def __init__(
        self,
        *,
        providers: list[ServiceProvider],
        invariants: list[Invariant] | None = None,
        container: Container | None = None,
    ) -> None:
        self.container: Container = container if container is not None else DefaultContainer()
        self._providers = list(providers)
        self._invariants = list(invariants or [])
        self._booted = False

    async def boot(self) -> None:
        if self._booted:
            raise RuntimeError("Engine already booted")
        for provider in self._providers:
            await provider.register(self.container)
        for invariant in self._invariants:
            await invariant(self.container)
        self._booted = True
```

- [ ] **Step 4: Tests pass (5 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/runtime/engine.py tests/runtime/test_engine.py
git commit -m "feat(runtime): Engine orchestrator with bootstrap-time invariants"
```

---

## Task 5: DBOS setup helper

**Files:**
- Create: `src/ballast/runtime/dbos_setup.py`
- Create: `tests/runtime/test_dbos_setup.py`

DBOS needs a Postgres connection string. We provide a helper that constructs a DBOS config from the existing pg_dsn convention used by Sub-project #2.

- [ ] **Step 1: Failing tests**

`tests/runtime/test_dbos_setup.py`:

```python
import pytest

from ballast.runtime.dbos_setup import (
    DBOSConfig,
    build_dbos_config,
)


def test_build_dbos_config_from_dsn():
    dsn = "postgresql+asyncpg://user:pass@host:5432/dbname"
    cfg = build_dbos_config(dsn)
    assert isinstance(cfg, DBOSConfig)
    assert cfg.database_url is not None
    # asyncpg-flavored URL → DBOS expects postgresql+psycopg or plain postgresql
    assert "+asyncpg" not in cfg.database_url


def test_build_dbos_config_strips_asyncpg_dialect():
    dsn = "postgresql+asyncpg://localhost/x"
    cfg = build_dbos_config(dsn)
    assert cfg.database_url == "postgresql://localhost/x"


def test_build_dbos_config_passes_app_name():
    cfg = build_dbos_config("postgresql://localhost/x", app_name="my-app")
    assert cfg.app_name == "my-app"
```

- [ ] **Step 2: Run — fail**

- [ ] **Step 3: Implement**

`src/ballast/runtime/dbos_setup.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DBOSConfig:
    """Framework wrapper around DBOS configuration.

    DBOS itself uses a TypedDict / dict-shape config — we keep a frozen
    dataclass at our boundary so the framework API is type-stable even if
    DBOS internals shift.
    """
    database_url: str
    app_name: str


def build_dbos_config(
    pg_dsn: str,
    *,
    app_name: str = "ballast-ai",
) -> DBOSConfig:
    """Translate a Sub-project #2 asyncpg DSN into a DBOS-friendly URL.

    DBOS uses synchronous psycopg under the hood for its own internal
    workflow_status table, so an asyncpg-flavored URL is stripped.
    """
    sync_url = re.sub(r"^postgresql\+asyncpg://", "postgresql://", pg_dsn)
    return DBOSConfig(database_url=sync_url, app_name=app_name)
```

- [ ] **Step 4: Tests pass (3 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/runtime/dbos_setup.py tests/runtime/test_dbos_setup.py
git commit -m "feat(runtime): build_dbos_config helper (translate asyncpg DSN)"
```

---

## Task 6: `Det.*` upgraded to `@DBOS.step`

Critical Fix #1 from code-review pass. Replaces plain async functions with `@DBOS.step` decorators so results are durably recorded across replay.

**Files:**
- Modify: `src/ballast/runtime/det.py`
- Create: `tests/runtime/test_dbos_det_step.py`

- [ ] **Step 1: Failing tests**

`tests/runtime/test_dbos_det_step.py`:

```python
"""Verify Det.* methods are registered as @DBOS.step.

We can't easily run them through a real DBOS workflow without a PG +
DBOS launch (covered by Task 11 smoke test). Instead this test inspects
the DBOS step registry to confirm registration.
"""

from ballast.runtime import Det


def test_det_now_is_dbos_step():
    fn = Det.now
    # DBOS wraps stepped functions; the wrapper exposes either a
    # dbos_internal attribute or has its __wrapped__ chain include DBOS.
    # We check for any of several known markers.
    assert (
        hasattr(fn, "__dbos_step__")
        or hasattr(fn, "__wrapped__")
        or getattr(fn, "_is_dbos_step", False)
    ), f"Det.now is not marked as a DBOS step: {fn!r}"


def test_det_uuid4_is_dbos_step():
    fn = Det.uuid4
    assert (
        hasattr(fn, "__dbos_step__")
        or hasattr(fn, "__wrapped__")
        or getattr(fn, "_is_dbos_step", False)
    )


def test_det_uuid_for_is_dbos_step():
    fn = Det.uuid_for
    assert (
        hasattr(fn, "__dbos_step__")
        or hasattr(fn, "__wrapped__")
        or getattr(fn, "_is_dbos_step", False)
    )


def test_det_random_choice_is_dbos_step():
    fn = Det.random_choice
    assert (
        hasattr(fn, "__dbos_step__")
        or hasattr(fn, "__wrapped__")
        or getattr(fn, "_is_dbos_step", False)
    )
```

(If these marker attributes don't all exist on DBOS-stepped functions, the implementer may need to adjust the test to check whatever DBOS actually exposes. The intent is: "verify these methods are decorated.")

- [ ] **Step 2: Run — fail (currently Det.* are plain async)**

- [ ] **Step 3: Implement**

Modify `src/ballast/runtime/det.py`. Apply `@DBOS.step()` to each method:

```python
from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TypeVar
from uuid import UUID, uuid5
from uuid import uuid4 as _uuid4

from dbos import DBOS

from ballast.runtime.idempotency import IdempotencyInput

T = TypeVar("T")

_UUID_NAMESPACE = UUID("ad9c8e22-1bc4-4a4f-9c40-d9c4f4ad7e10")


class Det:
    """Deterministic-recorded helpers wrapped as DBOS steps.

    Each method's result is recorded by DBOS on first invocation and
    replayed verbatim on recovery. This makes deterministic UUIDs and
    timestamps robust across crashes — replay produces identical values
    without re-running the function.

    Side-effects outside `@DBOS.step` boundaries are forbidden by lint
    rule STATEFLOW001-007 (Sub-project #3 Task 10).
    """

    @staticmethod
    @DBOS.step()
    async def now() -> datetime:
        return datetime.now(tz=UTC)

    @staticmethod
    @DBOS.step()
    async def uuid4() -> UUID:
        return _uuid4()

    @staticmethod
    @DBOS.step()
    async def random_choice(seq: Sequence[T]) -> T:
        return random.choice(seq)

    @staticmethod
    @DBOS.step()
    async def uuid_for(inputs: IdempotencyInput) -> UUID:
        """Deterministic UUID5 from a strict-typed input.

        Wrapped as @DBOS.step (Critical Fix #1, code-review pass): the
        result is durably recorded so replay returns the same UUID
        regardless of serialization-version drift across Pydantic /
        Python upgrades.
        """
        canonical = inputs.canonical_json()
        return uuid5(_UUID_NAMESPACE, canonical)
```

- [ ] **Step 4: Tests pass**

Run also Sub-project #1's Det test to confirm nothing broke (`tests/runtime/test_det.py` — its test functions just call `await Det.now()` etc, which DBOS supports outside a workflow context as a regular call):

```bash
uv run pytest tests/runtime/
```

- [ ] **Step 5: Full suite + mypy + ruff**

If DBOS imports cause type errors in tests that don't initialise DBOS, you may need `import dbos` at module-level only and configure DBOS lazily. Consult DBOS docs.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/runtime/det.py tests/runtime/test_dbos_det_step.py
git commit -m "feat(runtime): Det.* methods upgraded to @DBOS.step (Critical Fix #1)"
```

---

## Task 7: CoreProvider — minimal core bindings

**Files:**
- Create: `src/ballast/providers/__init__.py`
- Create: `src/ballast/providers/core.py`
- Create: `tests/providers/__init__.py`
- Create: `tests/providers/test_core_provider.py`

CoreProvider binds the minimal cross-cutting services: Det (just a class — no instance binding needed for static methods, but we bind a `DetProtocol`-style accessor for testability) and a placeholder for EventDispatcher (filled in later sub-project).

- [ ] **Step 1: Failing test**

`tests/providers/test_core_provider.py`:

```python
import pytest

from ballast.providers import CoreProvider
from ballast.runtime import DefaultContainer


@pytest.mark.asyncio
async def test_core_provider_binds_det():
    container = DefaultContainer()
    await CoreProvider().register(container)

    from ballast.runtime import Det
    assert container.get(type(Det)) is Det  # Det class binding
```

- [ ] **Step 2: Run — fail**

- [ ] **Step 3: Implement**

`src/ballast/providers/__init__.py`:

```python
from ballast.providers.core import CoreProvider

__all__ = ["CoreProvider"]
```

`src/ballast/providers/core.py`:

```python
from __future__ import annotations

from ballast.runtime import Container, Det


class CoreProvider:
    """Binds core framework primitives shared by all apps.

    Currently: `Det` (deterministic helpers). EventDispatcher binding
    will be added in a future sub-project.
    """

    async def register(self, container: Container) -> None:
        # Bind the Det class itself so callers can `container.get(type(Det))`
        # and get the static-method namespace. This is also a marker that
        # core bindings have been installed.
        container.bind(type(Det), lambda _: Det)
```

- [ ] **Step 4: Test passes**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/providers/__init__.py src/ballast/providers/core.py tests/providers/__init__.py tests/providers/test_core_provider.py
git commit -m "feat(providers): CoreProvider (binds Det)"
```

---

## Task 8: PersistenceProvider — binds session factory + Repos

**Files:**
- Create: `src/ballast/providers/persistence.py`
- Modify: `src/ballast/providers/__init__.py`
- Create: `tests/providers/test_persistence_provider.py`

- [ ] **Step 1: Failing test**

`tests/providers/test_persistence_provider.py`:

```python
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from ballast.persistence import (
    HITLRepository,
    OutboxRepository,
    SqlAlchemyUnitOfWork,
    ThreadRepository,
)
from ballast.providers import PersistenceProvider
from ballast.runtime import DefaultContainer


@pytest.mark.asyncio
async def test_persistence_provider_binds_session_factory_and_uow_factory():
    """PersistenceProvider binds an async_sessionmaker for downstream UoW.

    Uses in-memory SQLite for unit test to avoid Docker dep.
    """
    container = DefaultContainer()
    provider = PersistenceProvider(dsn="sqlite+aiosqlite:///:memory:")
    await provider.register(container)

    factory = container.get(async_sessionmaker)
    assert factory is not None


@pytest.mark.asyncio
async def test_persistence_provider_binds_uow_factory_callable():
    container = DefaultContainer()
    provider = PersistenceProvider(dsn="sqlite+aiosqlite:///:memory:")
    await provider.register(container)

    # UoW factory is a callable that returns a fresh SqlAlchemyUnitOfWork
    UoWFactory = type(lambda: None)  # placeholder; real test below
    # Instead, get the binding via SqlAlchemyUnitOfWork type
    uow_factory = container.get(SqlAlchemyUnitOfWork)
    # Calling factory returns a fresh UoW
    uow1 = uow_factory()
    uow2 = uow_factory()
    assert uow1 is not uow2
    assert isinstance(uow1, SqlAlchemyUnitOfWork)
```

- [ ] **Step 2: Run — fail**

- [ ] **Step 3: Implement**

`src/ballast/providers/persistence.py`:

```python
from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ballast.persistence import SqlAlchemyUnitOfWork
from ballast.runtime import Container


class PersistenceProvider:
    """Binds the async session factory and a UoW factory.

    Apps typically wire Repository factories themselves (in their own
    provider) because each app has its own Repository compositions.
    This provider only handles the framework infra: engine, sessionmaker,
    UoW factory.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    async def register(self, container: Container) -> None:
        engine = create_async_engine(self.dsn)
        sessionmaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        # Bind the sessionmaker itself (downstream repos take an AsyncSession)
        container.bind(async_sessionmaker, lambda _: sessionmaker)

        # Bind a UoW factory — callers call `uow_factory()` to get a fresh UoW.
        # We bind it under SqlAlchemyUnitOfWork (the type users will look up).
        def _uow_factory() -> SqlAlchemyUnitOfWork:
            return SqlAlchemyUnitOfWork(sessionmaker)

        container.bind(SqlAlchemyUnitOfWork, lambda _: _uow_factory, singleton=True)
```

Modify `src/ballast/providers/__init__.py`:

```python
from ballast.providers.core import CoreProvider
from ballast.providers.persistence import PersistenceProvider

__all__ = ["CoreProvider", "PersistenceProvider"]
```

- [ ] **Step 4: Tests pass (2 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/providers/persistence.py src/ballast/providers/__init__.py tests/providers/test_persistence_provider.py
git commit -m "feat(providers): PersistenceProvider (sessionmaker + UoW factory)"
```

---

## Task 9: `runtime` package public API

**Files:**
- Modify: `src/ballast/runtime/__init__.py`
- Create: `tests/runtime/test_public_api.py`

- [ ] **Step 1: Failing test**

`tests/runtime/test_public_api.py`:

```python
def test_runtime_public_api():
    from ballast.runtime import (
        Container,
        DefaultContainer,
        Det,
        Engine,
        EngineInvariantViolation,
        IdempotencyInput,
        IdempotencyValue,
        ServiceProvider,
    )

    assert Container is not None
    assert DefaultContainer is not None
    assert Det is not None
    assert Engine is not None
    assert EngineInvariantViolation is not None
    assert IdempotencyInput is not None
    assert IdempotencyValue is not None
    assert ServiceProvider is not None
```

- [ ] **Step 2: Run — fail (some exports missing)**

- [ ] **Step 3: Update `src/ballast/runtime/__init__.py`**

```python
from ballast.runtime.container import Container, DefaultContainer
from ballast.runtime.dbos_setup import DBOSConfig, build_dbos_config
from ballast.runtime.det import Det
from ballast.runtime.engine import Engine, EngineInvariantViolation
from ballast.runtime.idempotency import IdempotencyInput, IdempotencyValue
from ballast.runtime.provider import ServiceProvider

__all__ = [
    "Container",
    "DBOSConfig",
    "DefaultContainer",
    "Det",
    "Engine",
    "EngineInvariantViolation",
    "IdempotencyInput",
    "IdempotencyValue",
    "ServiceProvider",
    "build_dbos_config",
]
```

- [ ] **Step 4: Test passes**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/runtime/__init__.py tests/runtime/test_public_api.py
git commit -m "feat(runtime): top-level runtime public API"
```

---

## Task 10: DBOS workflow smoke test (end-to-end durable replay verification)

**Files:**
- Create: `tests/runtime/test_dbos_workflow_smoke.py`

This is the proof-of-life test — spawns a DBOS workflow, exercises Det.uuid_for inside it, verifies the result is durable across "replay" (simulated by re-executing same workflow_id).

- [ ] **Step 1: Write the smoke test**

`tests/runtime/test_dbos_workflow_smoke.py`:

```python
"""End-to-end DBOS smoke test: workflow uses Det.uuid_for, replay returns same value.

Requires Docker + Postgres (testcontainers). Skips cleanly otherwise.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from dbos import DBOS

from ballast.runtime import (
    DBOSConfig,
    Det,
    IdempotencyInput,
    build_dbos_config,
)

# Reuse persistence conftest for pg_dsn fixture
pytest_plugins = ["tests.persistence.conftest"]


@pytest.fixture
def dbos_config(pg_dsn: str) -> DBOSConfig:
    return build_dbos_config(pg_dsn, app_name="stateflow-test")


@pytest.fixture
def dbos_runtime(dbos_config: DBOSConfig):
    """Initialise + launch DBOS for the test session, teardown on exit."""
    DBOS(config={"name": dbos_config.app_name, "database_url": dbos_config.database_url})
    DBOS.launch()
    yield DBOS
    DBOS.destroy()


@DBOS.workflow()
async def _sample_workflow() -> UUID:
    """Workflow that calls Det.uuid_for — result should be durable."""
    return await Det.uuid_for(
        IdempotencyInput(namespace="smoke", parts={"x": 1})
    )


@pytest.mark.asyncio
async def test_dbos_workflow_uses_det_step(dbos_runtime):
    """Run the workflow once; expect a stable UUID5 result."""
    result = await _sample_workflow()
    assert isinstance(result, UUID)
    assert result.version == 5


@pytest.mark.asyncio
async def test_dbos_workflow_replay_returns_same_uuid(dbos_runtime):
    """Replay the workflow with the same idempotency key; result must match.

    Demonstrates the Critical Fix #1 guarantee: even if the underlying
    canonicalization changed, DBOS replays the cached step output.
    """
    workflow_id = "smoke-test-replay"
    a = await DBOS.start_workflow(_sample_workflow, workflow_id=workflow_id).get_result()
    b = await DBOS.start_workflow(_sample_workflow, workflow_id=workflow_id).get_result()
    assert a == b
```

- [ ] **Step 2: Run — depending on Docker availability**

Without Docker: skips. With Docker:

```bash
DOCKER_HOST=unix:///Users/kirunya/.docker/run/docker.sock \
TESTCONTAINERS_RYUK_DISABLED=true \
uv run pytest tests/runtime/test_dbos_workflow_smoke.py -v
```

If DBOS API differs from the calls used above, adjust per actual DBOS docs (e.g., DBOS may need `DBOS.set_config(...)` style init). The intent is: workflow runs, Det.uuid_for is called, result is durable, replay returns the same value.

- [ ] **Step 3: Verify test passes (with Docker) or skips cleanly (without)**

- [ ] **Step 4: Full suite + mypy + ruff**

- [ ] **Step 5: Commit**

```bash
git add tests/runtime/test_dbos_workflow_smoke.py
git commit -m "test: DBOS workflow smoke test (Det.uuid_for replay determinism)"
```

---

## Task 11: STATEFLOW lint rules (custom ruff plugin) — first 5 rules

Per spec 2E.1 + 4G. Implements rules `STATEFLOW001-005` (the most critical determinism boundary rules). Remaining rules (006-013) deferred to follow-up tasks once we have more pattern code to test them against.

**Files:**
- Create: `src/ballast/ruff/__init__.py`
- Create: `src/ballast/ruff/stateflow_rules.py`
- Create: `tests/lint/__init__.py`
- Create: `tests/lint/test_stateflow_rules.py`

> **Note:** ruff plugins (external rules) are not yet stably supported by the public ruff API. As an MVP, we implement these as a standalone AST-based linter that can be invoked from the test suite, with a CLI entrypoint planned for follow-up. The rules detect violations using the standard `ast` module.

- [ ] **Step 1: Write failing tests**

`tests/lint/test_stateflow_rules.py`:

```python
"""Tests for STATEFLOW lint rules — first 5 critical rules from spec 2E.1.

Each test feeds source-text to the rule engine and asserts whether a
violation was reported.
"""

import ast

from ballast.ruff.stateflow_rules import (
    check_source,
)


def _violations(code: str) -> list[str]:
    """Return list of rule IDs violated in the given source."""
    return [v.rule_id for v in check_source(code)]


def test_STATEFLOW001_datetime_now_in_workflow_body():  # noqa: N802
    code = """
from datetime import datetime
from dbos import DBOS

@DBOS.workflow()
async def bad():
    return datetime.now()
"""
    assert "STATEFLOW001" in _violations(code)


def test_STATEFLOW001_clean_when_datetime_in_step():  # noqa: N802
    code = """
from datetime import datetime
from dbos import DBOS

@DBOS.step()
async def ok():
    return datetime.now()
"""
    assert "STATEFLOW001" not in _violations(code)


def test_STATEFLOW002_time_time_in_workflow():  # noqa: N802
    code = """
import time
from dbos import DBOS

@DBOS.workflow()
async def bad():
    return time.time()
"""
    assert "STATEFLOW002" in _violations(code)


def test_STATEFLOW003_httpx_call_in_workflow():  # noqa: N802
    code = """
import httpx
from dbos import DBOS

@DBOS.workflow()
async def bad():
    return await httpx.get("https://example.com")
"""
    assert "STATEFLOW003" in _violations(code)


def test_STATEFLOW004_random_in_workflow():  # noqa: N802
    code = """
import random
from dbos import DBOS

@DBOS.workflow()
async def bad():
    return random.random()
"""
    assert "STATEFLOW004" in _violations(code)


def test_STATEFLOW005_asyncio_sleep_in_workflow():  # noqa: N802
    code = """
import asyncio
from dbos import DBOS

@DBOS.workflow()
async def bad():
    await asyncio.sleep(1)
"""
    assert "STATEFLOW005" in _violations(code)


def test_clean_workflow_with_no_violations():
    code = """
from dbos import DBOS

@DBOS.workflow()
async def good():
    x = 1 + 1
    return x
"""
    assert _violations(code) == []
```

- [ ] **Step 2: Run — fail**

- [ ] **Step 3: Implement**

`src/ballast/ruff/__init__.py`:

```python
from ballast.ruff.stateflow_rules import (
    Violation,
    check_path,
    check_source,
)

__all__ = ["Violation", "check_path", "check_source"]
```

`src/ballast/ruff/stateflow_rules.py`:

```python
"""STATEFLOW lint rules — AST-based detection of determinism-boundary violations.

Rules implemented in this sub-project (first 5; remaining 8 in follow-up):
- STATEFLOW001: `datetime.now()` / `datetime.utcnow()` inside @DBOS.workflow body
- STATEFLOW002: `time.time()` / `time.monotonic()` inside @DBOS.workflow body
- STATEFLOW003: `httpx.*` / `requests.*` calls inside @DBOS.workflow body
- STATEFLOW004: `random.*` inside @DBOS.workflow body
- STATEFLOW005: `asyncio.sleep(...)` inside @DBOS.workflow body

Each rule is a single AST visitor; aggregated by `check_source(code)` /
`check_path(path)`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Violation:
    rule_id: str
    line: int
    col: int
    message: str


def check_source(source: str) -> list[Violation]:
    """Parse `source` and return all STATEFLOW violations found."""
    tree = ast.parse(source)
    checker = _StateflowChecker()
    checker.visit(tree)
    return checker.violations


def check_path(path: Path | str) -> list[Violation]:
    """Read file at `path` and lint it."""
    p = Path(path)
    return check_source(p.read_text(encoding="utf-8"))


# ─── checker implementation ──────────────────────────────────────────────────


def _is_dbos_workflow_decorator(decorator: ast.expr) -> bool:
    """True iff decorator is @DBOS.workflow / @DBOS.workflow() (any args)."""
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Attribute):
        return (
            isinstance(target.value, ast.Name)
            and target.value.id == "DBOS"
            and target.attr == "workflow"
        )
    return False


def _is_dbos_step_decorator(decorator: ast.expr) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Attribute):
        return (
            isinstance(target.value, ast.Name)
            and target.value.id == "DBOS"
            and target.attr in ("step", "transaction")
        )
    return False


def _matches_module_call(call: ast.Call, modules: tuple[str, ...], attrs: set[str] | None = None) -> bool:
    """True iff call is `mod.attr(...)` where mod ∈ modules.

    If attrs is None, any attribute matches; otherwise only `attr ∈ attrs`.
    """
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    if not isinstance(func.value, ast.Name):
        return False
    if func.value.id not in modules:
        return False
    if attrs is not None and func.attr not in attrs:
        return False
    return True


class _StateflowChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[Violation] = []
        self._in_workflow: int = 0  # depth counter to handle nested defs

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_func(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_func(node)

    def _visit_func(self, node: ast.AsyncFunctionDef | ast.FunctionDef) -> None:
        in_workflow = any(_is_dbos_workflow_decorator(d) for d in node.decorator_list)
        in_step = any(_is_dbos_step_decorator(d) for d in node.decorator_list)
        if in_workflow and not in_step:
            self._in_workflow += 1
            try:
                self.generic_visit(node)
            finally:
                self._in_workflow -= 1
        else:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if self._in_workflow > 0:
            self._check_call(node)
        self.generic_visit(node)

    def _check_call(self, node: ast.Call) -> None:
        # STATEFLOW001 — datetime.now / datetime.utcnow
        if _matches_module_call(node, ("datetime",), {"now", "utcnow"}):
            self.violations.append(Violation(
                rule_id="STATEFLOW001",
                line=node.lineno, col=node.col_offset,
                message="datetime.now()/utcnow() outside @DBOS.step inside a workflow",
            ))
        # STATEFLOW002 — time.time / time.monotonic
        if _matches_module_call(node, ("time",), {"time", "monotonic", "perf_counter"}):
            self.violations.append(Violation(
                rule_id="STATEFLOW002",
                line=node.lineno, col=node.col_offset,
                message="time.time()/monotonic() outside @DBOS.step inside a workflow",
            ))
        # STATEFLOW003 — httpx.*/requests.*
        if _matches_module_call(node, ("httpx", "requests")):
            self.violations.append(Violation(
                rule_id="STATEFLOW003",
                line=node.lineno, col=node.col_offset,
                message="HTTP call outside @DBOS.step inside a workflow",
            ))
        # STATEFLOW004 — random.*
        if _matches_module_call(node, ("random",)):
            self.violations.append(Violation(
                rule_id="STATEFLOW004",
                line=node.lineno, col=node.col_offset,
                message="random.* outside @DBOS.step inside a workflow",
            ))
        # STATEFLOW005 — asyncio.sleep
        if _matches_module_call(node, ("asyncio",), {"sleep"}):
            self.violations.append(Violation(
                rule_id="STATEFLOW005",
                line=node.lineno, col=node.col_offset,
                message="asyncio.sleep() in workflow (use DBOS.sleep() instead)",
            ))
```

- [ ] **Step 4: Tests pass (7 new)**

- [ ] **Step 5: Full suite + mypy + ruff**

- [ ] **Step 6: Commit**

```bash
git add src/ballast/ruff tests/lint
git commit -m "feat(ruff): STATEFLOW001-005 lint rules (determinism boundary)"
```

---

## Task 12: Top-level public API + final smoke

**Files:**
- Modify: `src/ballast/__init__.py`
- Create: `tests/test_runtime_public.py`

- [ ] **Step 1: Update top-level exports**

In `src/ballast/__init__.py`, ADD to existing exports:

```python
from ballast.providers import CoreProvider, PersistenceProvider
from ballast.runtime import (
    Container,
    DBOSConfig,
    DefaultContainer,
    Engine,
    EngineInvariantViolation,
    ServiceProvider,
    build_dbos_config,
)
```

And to `__all__`:

```python
__all__ = [
    # ...existing entries from Sub-projects #1 + #2...
    "Container",
    "CoreProvider",
    "DBOSConfig",
    "DefaultContainer",
    "Engine",
    "EngineInvariantViolation",
    "PersistenceProvider",
    "ServiceProvider",
    "build_dbos_config",
]
```

- [ ] **Step 2: Verify integration test**

`tests/test_runtime_public.py`:

```python
def test_runtime_classes_visible_from_top_level():
    from ballast import (
        Container,
        CoreProvider,
        DBOSConfig,
        DefaultContainer,
        Engine,
        EngineInvariantViolation,
        PersistenceProvider,
        ServiceProvider,
        build_dbos_config,
    )

    assert Engine is not None
    assert Container is not None
    assert isinstance(DefaultContainer(), Container)
    assert callable(build_dbos_config)
```

- [ ] **Step 3: Tests pass**
- [ ] **Step 4: Full suite + mypy + ruff**
- [ ] **Step 5: Commit**

```bash
git add src/ballast/__init__.py tests/test_runtime_public.py
git commit -m "feat: Sub-project #3 public API (Container, Engine, providers)"
```

---

## Sub-project #3 acceptance criteria

After all 12 tasks:

- ✅ `from ballast import Engine, Container, ServiceProvider, CoreProvider, PersistenceProvider, build_dbos_config` works
- ✅ `Det.now / uuid4 / random_choice / uuid_for` are `@DBOS.step`-decorated (Critical Fix #1 closed)
- ✅ `Container` is type-keyed (no string keys); `DefaultContainer.bind(Protocol, factory, singleton=True/False)` works
- ✅ `ServiceProvider` is single-phase `async def register(container)` (per 4A.0.13)
- ✅ `Engine(providers=[...], invariants=[...])`:
    - boots providers in declared order
    - runs invariants after all providers register
    - raises `EngineInvariantViolation` on failure
    - raises `RuntimeError` on double-boot
- ✅ `CoreProvider` binds `Det`
- ✅ `PersistenceProvider(dsn=...)` binds async sessionmaker + UoW factory
- ✅ `build_dbos_config(pg_dsn)` produces a DBOS-friendly config (strips asyncpg dialect)
- ✅ STATEFLOW001-005 lint rules detect forbidden patterns in workflow bodies
- ✅ DBOS workflow smoke test passes with Docker, skips cleanly without
- ✅ Sub-project #1 + #2 tests all still pass
- ✅ mypy strict + ruff clean

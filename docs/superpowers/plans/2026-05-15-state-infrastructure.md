# State Infrastructure (Sub-project #2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the L4 state infrastructure layer of `pydantic-ai-stateflow`: SQLModel framework tables (Tenant, Thread, Message, Outbox, HITL family), Repository Protocols with InMemory + Postgres implementations, `UnitOfWork` Protocol with `SqlAlchemyUnitOfWork` concrete, Alembic migrations, and a testcontainers-based PG integration test setup. Multi-tenant first-class (every method requires `tenant_id`).

**Architecture:** SQLModel (Pydantic 2 + SQLAlchemy 2 async) for persistence rows. Pydantic domain models stay separate (per spec 1.4 #6: Persistence ≠ Domain). `UnitOfWork` Protocol hides SQLAlchemy `AsyncSession` from Pattern signatures (spec 4A.0.5). Repositories are Protocols + InMemory + Postgres concrete pairs. Alembic for migrations; framework migrations run first, app migrations after. `testcontainers[postgresql]` for integration test fixture.

**Tech Stack:** SQLModel, SQLAlchemy 2.x (async), asyncpg, Alembic, testcontainers, pytest, pytest-asyncio.

**Spec sections covered:** 1.4 #6, #11, #12 (Persistence ≠ Domain, Repository as port, thin API), 1.12 (multi-tenant), 4A.0.5 (UnitOfWork), 4B (L4 framework tables + Repository protocols).

**Scope vs deferred:** v1 implements Tenant + Thread + Message + Outbox + HITL family (BlockingRequirement / Decision / AuthzDenial). Deferred to future sub-projects: CheckpointRow (Sub-project #3 with DBOS), EvalRunRow (#7 evals), DriftMetricRow (#6 observability), ProposalAuditRow (#5 patterns), AdvisorCacheRow (autopilot).

---

## File Structure

```
src/pydantic_ai_stateflow/
├── persistence/
│   ├── __init__.py                  # public: UnitOfWork, Repos, ...
│   ├── uow.py                       # UnitOfWork Protocol + SqlAlchemyUnitOfWork
│   ├── tenant/
│   │   ├── __init__.py
│   │   ├── persistence.py           # TenantRow
│   │   └── domain.py                # Tenant (Pydantic)
│   ├── thread/
│   │   ├── __init__.py
│   │   ├── persistence.py           # ThreadRow, MessageRow, ThreadPurpose
│   │   ├── domain.py                # Thread, Message
│   │   └── repository.py            # ThreadRepository Protocol + InMemory + Postgres
│   ├── outbox/
│   │   ├── __init__.py
│   │   ├── persistence.py           # OutboxRow
│   │   ├── domain.py                # OutboxEvent
│   │   └── repository.py            # OutboxRepository Protocol + InMemory + Postgres
│   └── hitl/
│       ├── __init__.py
│       ├── persistence.py           # BlockingRequirementRow, DecisionRow, AuthzDenialRow
│       ├── domain.py                # BlockingRequirement, Decision, AuthzDenial
│       └── repository.py            # HITLRepository Protocol + InMemory + Postgres
├── testing/
│   ├── __init__.py                  # public: InMemoryThreadRepository, ...
│   └── _aliases.py                  # re-export InMemory* from persistence/* for ergonomics
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── 0001_framework_tables.py
└── (existing modules unchanged)

tests/
├── persistence/
│   ├── conftest.py                  # pg_dsn fixture via testcontainers
│   ├── test_uow.py
│   ├── test_thread_inmemory.py
│   ├── test_thread_postgres.py      # integration
│   ├── test_outbox_inmemory.py
│   ├── test_outbox_postgres.py
│   ├── test_hitl_inmemory.py
│   └── test_hitl_postgres.py
└── integration/
    └── test_state_smoke.py          # end-to-end across all repos with one PG instance
```

---

## Task 1: Add SQLModel / SQLAlchemy / Alembic / testcontainers dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update `pyproject.toml` dependencies**

In `[project] dependencies` ADD:

```toml
dependencies = [
    "pydantic>=2.7",
    "pydantic-ai>=0.0.13",
    "sqlmodel>=0.0.22",
    "sqlalchemy[asyncio]>=2.0",
    "alembic>=1.13",
    "asyncpg>=0.29",
]
```

In `[project.optional-dependencies.dev]` ADD:

```toml
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.5",
    "mypy>=1.10",
    "testcontainers[postgresql]>=4.7",
]
```

- [ ] **Step 2: Sync and verify**

```bash
uv sync --extra dev
uv run pytest && uv run mypy src && uv run ruff check
```

Expected: full Sub-project #1 suite (74 tests) still passes; mypy + ruff clean.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add SQLModel, SQLAlchemy async, Alembic, testcontainers deps"
```

---

## Task 2: `UnitOfWork` Protocol + `SqlAlchemyUnitOfWork`

**Files:**
- Create: `src/pydantic_ai_stateflow/persistence/__init__.py`
- Create: `src/pydantic_ai_stateflow/persistence/uow.py`
- Create: `tests/persistence/__init__.py`
- Create: `tests/persistence/test_uow.py`

- [ ] **Step 1: Write failing test**

`tests/persistence/test_uow.py`:

```python
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from pydantic_ai_stateflow.persistence import SqlAlchemyUnitOfWork, UnitOfWork


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """In-memory SQLite for UoW unit-tests; tables can be empty."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_uow_commits_on_clean_exit(session_factory):
    uow: UnitOfWork = SqlAlchemyUnitOfWork(session_factory)
    async with uow:
        # nothing to do — just exercise the lifecycle
        pass
    # No exception → committed cleanly


@pytest.mark.asyncio
async def test_uow_rollbacks_on_exception(session_factory):
    uow: UnitOfWork = SqlAlchemyUnitOfWork(session_factory)
    with pytest.raises(RuntimeError, match="boom"):
        async with uow:
            raise RuntimeError("boom")
    # Exception propagated; rollback happened (no assertion needed for the empty op)


@pytest.mark.asyncio
async def test_uow_explicit_commit_inside_context(session_factory):
    """Inside the context, manual commit() is allowed and triggers a fresh tx implicitly."""
    uow: UnitOfWork = SqlAlchemyUnitOfWork(session_factory)
    async with uow:
        await uow.commit()


@pytest.mark.asyncio
async def test_uow_protocol_signature_is_satisfied_by_concrete(session_factory):
    """SqlAlchemyUnitOfWork structurally satisfies UnitOfWork."""
    instance = SqlAlchemyUnitOfWork(session_factory)
    assert isinstance(instance, UnitOfWork)
```

Add `aiosqlite>=0.20` to `[project.optional-dependencies.dev]` if not present (it's needed by `sqlite+aiosqlite://` for tests that don't want a real PG). Run `uv sync --extra dev` after editing.

- [ ] **Step 2: Run test — verify failure (ImportError)**

```bash
uv run pytest tests/persistence/test_uow.py -v
```

- [ ] **Step 3: Implement**

Add `aiosqlite` to dev deps in `pyproject.toml` if needed:

```toml
dev = [
    ...,
    "aiosqlite>=0.20",
]
```

`src/pydantic_ai_stateflow/persistence/__init__.py`:

```python
from pydantic_ai_stateflow.persistence.uow import SqlAlchemyUnitOfWork, UnitOfWork

__all__ = ["SqlAlchemyUnitOfWork", "UnitOfWork"]
```

`src/pydantic_ai_stateflow/persistence/uow.py`:

```python
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@runtime_checkable
class UnitOfWork(Protocol):
    """Hides SQLAlchemy AsyncSession from Pattern signatures (per spec 4A.0.5).

    Use as an async context manager. On clean exit: commit. On exception: rollback.
    `commit()` is also exposed for callers that need to commit mid-transaction
    (e.g. transactional outbox).
    """

    async def __aenter__(self) -> "UnitOfWork": ...
    async def __aexit__(self, *exc_info: Any) -> None: ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...


class SqlAlchemyUnitOfWork:
    """Concrete UoW backed by a SQLAlchemy async sessionmaker.

    Internally manages an AsyncSession. The session itself is exposed as
    `self.session` only inside `persistence/*` modules — Patterns / Capabilities
    must depend on the UnitOfWork Protocol, not on this concrete class.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None

    @property
    def session(self) -> AsyncSession:
        if self._session is None:
            raise RuntimeError("UnitOfWork must be entered before accessing session")
        return self._session

    async def __aenter__(self) -> "SqlAlchemyUnitOfWork":
        self._session = self._session_factory()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session is None:
            return
        try:
            if exc_type is None:
                await self._session.commit()
            else:
                await self._session.rollback()
        finally:
            await self._session.close()
            self._session = None

    async def commit(self) -> None:
        if self._session is not None:
            await self._session.commit()

    async def rollback(self) -> None:
        if self._session is not None:
            await self._session.rollback()
```

- [ ] **Step 4: Verify tests pass**

```bash
uv run pytest tests/persistence/test_uow.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/persistence tests/persistence pyproject.toml uv.lock
git commit -m "feat(persistence): UnitOfWork Protocol + SqlAlchemyUnitOfWork"
```

---

## Task 3: Alembic skeleton + env.py

**Files:**
- Create: `src/pydantic_ai_stateflow/alembic/env.py`
- Create: `src/pydantic_ai_stateflow/alembic/script.py.mako`
- Create: `src/pydantic_ai_stateflow/alembic/versions/__init__.py`
- Create: `src/pydantic_ai_stateflow/alembic.ini`
- Create: `tests/persistence/test_alembic_metadata.py`

- [ ] **Step 1: Write failing test**

`tests/persistence/test_alembic_metadata.py`:

```python
"""Smoke test: Alembic metadata is loadable and consistent with SQLModel registry.

Real migrations are exercised in integration tests (Task 5+ via testcontainers PG).
"""

from sqlmodel import SQLModel


def test_metadata_is_empty_in_isolation():
    """Before any framework tables are imported, SQLModel.metadata has no tables.

    This test guards against accidentally loading framework persistence modules
    at import time (which would pollute metadata in unrelated tests).
    """
    # Just confirm metadata exists and is a MetaData
    assert SQLModel.metadata is not None
    # Don't assert count == 0 because dev environment may have other models cached;
    # just confirm metadata is accessible.


def test_alembic_ini_exists():
    """Alembic config file exists and is loadable."""
    from pathlib import Path

    import pydantic_ai_stateflow

    pkg_dir = Path(pydantic_ai_stateflow.__file__).parent
    assert (pkg_dir / "alembic.ini").exists()
    assert (pkg_dir / "alembic" / "env.py").exists()
    assert (pkg_dir / "alembic" / "script.py.mako").exists()
```

- [ ] **Step 2: Run — fails (alembic.ini not found)**

- [ ] **Step 3: Implement**

`src/pydantic_ai_stateflow/alembic.ini`:

```ini
[alembic]
script_location = %(here)s/alembic
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql+asyncpg://localhost/placeholder

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

`src/pydantic_ai_stateflow/alembic/env.py`:

```python
"""Alembic environment for pydantic-ai-stateflow framework tables.

Importing this module imports every persistence module under
`pydantic_ai_stateflow.persistence.*` so SQLModel.metadata is populated.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlmodel import SQLModel

# Import all framework persistence modules so their tables register with SQLModel.metadata.
# Each new persistence module added below must also be imported here.
import pydantic_ai_stateflow.persistence.tenant.persistence  # noqa: F401
import pydantic_ai_stateflow.persistence.thread.persistence  # noqa: F401
import pydantic_ai_stateflow.persistence.outbox.persistence  # noqa: F401
import pydantic_ai_stateflow.persistence.hitl.persistence    # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
```

`src/pydantic_ai_stateflow/alembic/script.py.mako`:

```
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

`src/pydantic_ai_stateflow/alembic/versions/__init__.py`: empty file.

The persistence modules don't exist yet (Tasks 4-9 create them). For env.py to import without error, create placeholder package init files now:

```bash
mkdir -p src/pydantic_ai_stateflow/persistence/tenant
mkdir -p src/pydantic_ai_stateflow/persistence/thread
mkdir -p src/pydantic_ai_stateflow/persistence/outbox
mkdir -p src/pydantic_ai_stateflow/persistence/hitl
touch src/pydantic_ai_stateflow/persistence/tenant/__init__.py
touch src/pydantic_ai_stateflow/persistence/tenant/persistence.py
touch src/pydantic_ai_stateflow/persistence/thread/__init__.py
touch src/pydantic_ai_stateflow/persistence/thread/persistence.py
touch src/pydantic_ai_stateflow/persistence/outbox/__init__.py
touch src/pydantic_ai_stateflow/persistence/outbox/persistence.py
touch src/pydantic_ai_stateflow/persistence/hitl/__init__.py
touch src/pydantic_ai_stateflow/persistence/hitl/persistence.py
```

Update `pyproject.toml` to ship `alembic.ini` and the `alembic/` folder as package data:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/pydantic_ai_stateflow"]
include = ["src/pydantic_ai_stateflow/alembic.ini"]

[tool.hatch.build.targets.wheel.force-include]
"src/pydantic_ai_stateflow/alembic" = "pydantic_ai_stateflow/alembic"
```

- [ ] **Step 4: Verify tests pass**

```bash
uv sync --extra dev
uv run pytest tests/persistence/test_alembic_metadata.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/alembic.ini src/pydantic_ai_stateflow/alembic src/pydantic_ai_stateflow/persistence pyproject.toml tests/persistence/test_alembic_metadata.py
git commit -m "feat(persistence): Alembic skeleton (env.py, script.py.mako, alembic.ini)"
```

---

## Task 4: TenantRow + Tenant domain

**Files:**
- Modify: `src/pydantic_ai_stateflow/persistence/tenant/persistence.py`
- Create: `src/pydantic_ai_stateflow/persistence/tenant/domain.py`
- Create: `src/pydantic_ai_stateflow/persistence/tenant/__init__.py`
- Create: `tests/persistence/test_tenant.py`

- [ ] **Step 1: Failing test**

`tests/persistence/test_tenant.py`:

```python
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlmodel import SQLModel

from pydantic_ai_stateflow.persistence.tenant.domain import Tenant
from pydantic_ai_stateflow.persistence.tenant.persistence import TenantRow


def test_tenant_row_registered_with_metadata():
    """TenantRow's table must be in SQLModel.metadata."""
    assert "tenants" in SQLModel.metadata.tables


def test_tenant_row_fields():
    row = TenantRow(id=uuid4(), name="acme")
    assert isinstance(row.id, UUID)
    assert row.name == "acme"
    assert isinstance(row.created_at, datetime)


def test_tenant_domain_roundtrip_from_row():
    row = TenantRow(id=uuid4(), name="acme")
    domain = Tenant.from_row(row)
    assert domain.id == row.id
    assert domain.name == row.name
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement**

`src/pydantic_ai_stateflow/persistence/tenant/__init__.py`:

```python
from pydantic_ai_stateflow.persistence.tenant.domain import Tenant
from pydantic_ai_stateflow.persistence.tenant.persistence import TenantRow

__all__ = ["Tenant", "TenantRow"]
```

`src/pydantic_ai_stateflow/persistence/tenant/persistence.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


class TenantRow(SQLModel, table=True):
    __tablename__ = "tenants"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=_now_utc)
```

`src/pydantic_ai_stateflow/persistence/tenant/domain.py`:

```python
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from pydantic_ai_stateflow.persistence.tenant.persistence import TenantRow


class Tenant(BaseModel):
    """Pydantic domain representation of a tenant."""
    model_config = ConfigDict(frozen=True)

    id: UUID
    name: str
    created_at: datetime

    @classmethod
    def from_row(cls, row: TenantRow) -> "Tenant":
        return cls(id=row.id, name=row.name, created_at=row.created_at)
```

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Full suite + mypy + ruff**

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/persistence/tenant tests/persistence/test_tenant.py
git commit -m "feat(persistence): TenantRow table + Tenant domain"
```

---

## Task 5: ThreadRow + MessageRow tables + Thread/Message domain

**Files:**
- Modify: `src/pydantic_ai_stateflow/persistence/thread/persistence.py`
- Create: `src/pydantic_ai_stateflow/persistence/thread/domain.py`
- Create: `src/pydantic_ai_stateflow/persistence/thread/__init__.py`
- Create: `tests/persistence/test_thread_models.py`

- [ ] **Step 1: Failing test**

`tests/persistence/test_thread_models.py`:

```python
from uuid import UUID, uuid4

from sqlmodel import SQLModel

from pydantic_ai_stateflow.persistence.thread.domain import Message, Thread, ThreadPurpose
from pydantic_ai_stateflow.persistence.thread.persistence import MessageRow, ThreadRow


def test_thread_table_registered():
    assert "threads" in SQLModel.metadata.tables
    assert "messages" in SQLModel.metadata.tables


def test_thread_purpose_enum_values():
    assert ThreadPurpose.ONBOARDING == "onboarding"
    assert ThreadPurpose.CONVERSATION == "conversation"
    assert ThreadPurpose.HITL == "hitl"


def test_thread_row_minimal_fields():
    row = ThreadRow(
        tenant_id=uuid4(),
        purpose=ThreadPurpose.CONVERSATION.value,
        actor_id="user-1",
    )
    assert isinstance(row.id, UUID)
    assert row.purpose == "conversation"
    assert row.actor_id == "user-1"
    assert row.purpose_metadata == {}


def test_thread_row_with_purpose_metadata():
    row = ThreadRow(
        tenant_id=uuid4(),
        purpose=ThreadPurpose.HITL.value,
        actor_id="founder-x",
        purpose_metadata={"gate_kind": "strategy_review", "wave_id": "abc"},
    )
    assert row.purpose_metadata["gate_kind"] == "strategy_review"


def test_message_row_fields():
    thread_id = uuid4()
    tenant_id = uuid4()
    row = MessageRow(
        tenant_id=tenant_id,
        thread_id=thread_id,
        role="user",
        parts=[{"kind": "text", "content": "hello"}],
    )
    assert row.role == "user"
    assert row.parts == [{"kind": "text", "content": "hello"}]


def test_thread_domain_from_row():
    row = ThreadRow(
        tenant_id=uuid4(),
        purpose=ThreadPurpose.CONVERSATION.value,
        actor_id="a",
    )
    domain = Thread.from_row(row)
    assert domain.id == row.id
    assert domain.purpose == ThreadPurpose.CONVERSATION
    assert domain.purpose_metadata == {}


def test_message_domain_from_row():
    row = MessageRow(
        tenant_id=uuid4(),
        thread_id=uuid4(),
        role="assistant",
        parts=[{"kind": "text", "content": "hi"}],
    )
    domain = Message.from_row(row)
    assert domain.role == "assistant"
    assert domain.parts == [{"kind": "text", "content": "hi"}]
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement**

`src/pydantic_ai_stateflow/persistence/thread/persistence.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


class ThreadRow(SQLModel, table=True):
    __tablename__ = "threads"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = Field(foreign_key="tenants.id", index=True)
    purpose: str                                              # ThreadPurpose enum value or domain-specific str
    purpose_metadata: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    workflow_id: UUID | None = Field(default=None, index=True)
    actor_id: str
    created_at: datetime = Field(default_factory=_now_utc)


class MessageRow(SQLModel, table=True):
    __tablename__ = "messages"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = Field(foreign_key="tenants.id", index=True)
    thread_id: UUID = Field(foreign_key="threads.id", index=True)
    role: str                                                # "system" / "user" / "assistant" / "tool"
    parts: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, server_default="[]"),
    )
    created_at: datetime = Field(default_factory=_now_utc)
```

`src/pydantic_ai_stateflow/persistence/thread/domain.py`:

```python
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from pydantic_ai_stateflow.persistence.thread.persistence import MessageRow, ThreadRow


class ThreadPurpose(StrEnum):
    ONBOARDING = "onboarding"
    CONVERSATION = "conversation"
    HITL = "hitl"


class Thread(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID
    tenant_id: UUID
    purpose: ThreadPurpose | str                              # may be domain-specific str
    purpose_metadata: dict[str, Any]
    workflow_id: UUID | None
    actor_id: str
    created_at: datetime

    @classmethod
    def from_row(cls, row: ThreadRow) -> "Thread":
        # Coerce known purposes to enum; unknown stays as str
        try:
            purpose = ThreadPurpose(row.purpose)
        except ValueError:
            purpose = row.purpose
        return cls(
            id=row.id, tenant_id=row.tenant_id, purpose=purpose,
            purpose_metadata=row.purpose_metadata, workflow_id=row.workflow_id,
            actor_id=row.actor_id, created_at=row.created_at,
        )


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID
    tenant_id: UUID
    thread_id: UUID
    role: str
    parts: list[dict[str, Any]]
    created_at: datetime

    @classmethod
    def from_row(cls, row: MessageRow) -> "Message":
        return cls(
            id=row.id, tenant_id=row.tenant_id, thread_id=row.thread_id,
            role=row.role, parts=row.parts, created_at=row.created_at,
        )
```

`src/pydantic_ai_stateflow/persistence/thread/__init__.py`:

```python
from pydantic_ai_stateflow.persistence.thread.domain import Message, Thread, ThreadPurpose
from pydantic_ai_stateflow.persistence.thread.persistence import MessageRow, ThreadRow

__all__ = ["Message", "MessageRow", "Thread", "ThreadPurpose", "ThreadRow"]
```

- [ ] **Step 4: Tests pass (7 new)**

- [ ] **Step 5: Full suite + mypy + ruff**

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/persistence/thread tests/persistence/test_thread_models.py
git commit -m "feat(persistence): ThreadRow + MessageRow tables + Thread/Message domain"
```

---

## Task 6: `ThreadRepository` Protocol + `InMemoryThreadRepository`

**Files:**
- Create: `src/pydantic_ai_stateflow/persistence/thread/repository.py`
- Modify: `src/pydantic_ai_stateflow/persistence/thread/__init__.py`
- Create: `tests/persistence/test_thread_inmemory.py`

- [ ] **Step 1: Failing tests**

`tests/persistence/test_thread_inmemory.py`:

```python
from uuid import uuid4

import pytest

from pydantic_ai_stateflow.persistence.thread import (
    InMemoryThreadRepository,
    ThreadPurpose,
    ThreadRepository,
)


@pytest.fixture
def tenant_id():
    return uuid4()


@pytest.fixture
def other_tenant_id():
    return uuid4()


@pytest.fixture
def repo() -> ThreadRepository:
    return InMemoryThreadRepository()


@pytest.mark.asyncio
async def test_create_and_load_thread(repo, tenant_id):
    thread = await repo.create(
        purpose=ThreadPurpose.CONVERSATION.value,
        purpose_metadata={},
        actor_id="founder-1",
        tenant_id=tenant_id,
    )
    loaded = await repo.load(thread.id, tenant_id=tenant_id)
    assert loaded.id == thread.id
    assert loaded.actor_id == "founder-1"


@pytest.mark.asyncio
async def test_load_returns_none_for_wrong_tenant(repo, tenant_id, other_tenant_id):
    thread = await repo.create(
        purpose=ThreadPurpose.HITL.value,
        purpose_metadata={"gate_kind": "x"},
        actor_id="a",
        tenant_id=tenant_id,
    )
    result = await repo.load(thread.id, tenant_id=other_tenant_id)
    assert result is None


@pytest.mark.asyncio
async def test_add_message_and_read_history(repo, tenant_id):
    thread = await repo.create(
        purpose=ThreadPurpose.CONVERSATION.value,
        purpose_metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    await repo.add_message(
        thread.id, role="user", parts=[{"kind": "text", "content": "hi"}], tenant_id=tenant_id
    )
    await repo.add_message(
        thread.id, role="assistant", parts=[{"kind": "text", "content": "hello"}], tenant_id=tenant_id
    )
    history = await repo.history(thread.id, tenant_id=tenant_id, limit=10)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


@pytest.mark.asyncio
async def test_history_respects_limit_oldest_first(repo, tenant_id):
    thread = await repo.create(
        purpose=ThreadPurpose.CONVERSATION.value,
        purpose_metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    for i in range(5):
        await repo.add_message(
            thread.id, role="user", parts=[{"kind": "text", "content": f"m{i}"}], tenant_id=tenant_id
        )
    history = await repo.history(thread.id, tenant_id=tenant_id, limit=3)
    assert len(history) == 3
    # Oldest first
    assert history[0].parts[0]["content"] == "m0"


@pytest.mark.asyncio
async def test_history_cross_tenant_isolation(repo, tenant_id, other_tenant_id):
    """Adding a message to another tenant's thread must fail safely."""
    thread = await repo.create(
        purpose=ThreadPurpose.CONVERSATION.value,
        purpose_metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    with pytest.raises(KeyError):
        await repo.add_message(
            thread.id, role="user", parts=[], tenant_id=other_tenant_id
        )
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement**

`src/pydantic_ai_stateflow/persistence/thread/repository.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic_ai_stateflow.persistence.thread.domain import Message, Thread


@runtime_checkable
class ThreadRepository(Protocol):
    """Port for thread + message persistence.

    All methods require `tenant_id` — multi-tenant first-class per spec 1.12.
    """

    async def create(
        self, *, purpose: str, purpose_metadata: dict[str, Any], actor_id: str, tenant_id: UUID
    ) -> Thread: ...
    async def load(self, id: UUID, *, tenant_id: UUID) -> Thread | None: ...
    async def add_message(
        self, thread_id: UUID, *, role: str, parts: list[dict[str, Any]], tenant_id: UUID
    ) -> Message: ...
    async def history(
        self, thread_id: UUID, *, tenant_id: UUID, limit: int = 100
    ) -> list[Message]: ...


class InMemoryThreadRepository:
    """In-memory implementation for unit tests."""

    def __init__(self) -> None:
        self._threads: dict[UUID, Thread] = {}
        self._messages: dict[UUID, list[Message]] = {}

    async def create(
        self, *, purpose: str, purpose_metadata: dict[str, Any], actor_id: str, tenant_id: UUID
    ) -> Thread:
        thread = Thread(
            id=uuid4(), tenant_id=tenant_id, purpose=purpose,
            purpose_metadata=dict(purpose_metadata), workflow_id=None,
            actor_id=actor_id, created_at=datetime.now(tz=UTC),
        )
        self._threads[thread.id] = thread
        self._messages[thread.id] = []
        return thread

    async def load(self, id: UUID, *, tenant_id: UUID) -> Thread | None:
        thread = self._threads.get(id)
        if thread is None or thread.tenant_id != tenant_id:
            return None
        return thread

    async def add_message(
        self, thread_id: UUID, *, role: str, parts: list[dict[str, Any]], tenant_id: UUID
    ) -> Message:
        thread = self._threads.get(thread_id)
        if thread is None or thread.tenant_id != tenant_id:
            raise KeyError(f"Thread {thread_id} not found for tenant {tenant_id}")
        msg = Message(
            id=uuid4(), tenant_id=tenant_id, thread_id=thread_id,
            role=role, parts=list(parts), created_at=datetime.now(tz=UTC),
        )
        self._messages[thread_id].append(msg)
        return msg

    async def history(
        self, thread_id: UUID, *, tenant_id: UUID, limit: int = 100
    ) -> list[Message]:
        thread = self._threads.get(thread_id)
        if thread is None or thread.tenant_id != tenant_id:
            return []
        return self._messages[thread_id][:limit]
```

Modify `src/pydantic_ai_stateflow/persistence/thread/__init__.py` to add the new exports:

```python
from pydantic_ai_stateflow.persistence.thread.domain import Message, Thread, ThreadPurpose
from pydantic_ai_stateflow.persistence.thread.persistence import MessageRow, ThreadRow
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
    ThreadRepository,
)

__all__ = [
    "InMemoryThreadRepository",
    "Message",
    "MessageRow",
    "Thread",
    "ThreadPurpose",
    "ThreadRepository",
    "ThreadRow",
]
```

- [ ] **Step 4: Tests pass (5 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/persistence/thread tests/persistence/test_thread_inmemory.py
git commit -m "feat(persistence): ThreadRepository Protocol + InMemoryThreadRepository"
```

---

## Task 7: testcontainers PG fixture + PostgresThreadRepository

**Files:**
- Create: `tests/persistence/conftest.py`
- Create: `src/pydantic_ai_stateflow/persistence/thread/postgres.py`
- Modify: `src/pydantic_ai_stateflow/persistence/thread/__init__.py`
- Create: `tests/persistence/test_thread_postgres.py`

- [ ] **Step 1: PG fixture in conftest**

`tests/persistence/conftest.py`:

```python
"""Shared testcontainers Postgres fixture for persistence integration tests."""

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

import pydantic_ai_stateflow


@pytest.fixture(scope="session")
def pg_container() -> PostgresContainer:
    """One PG instance per test session."""
    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@pytest.fixture(scope="session")
def pg_dsn(pg_container: PostgresContainer) -> str:
    """asyncpg-compatible URL."""
    url = pg_container.get_connection_url()
    # testcontainers default URL uses psycopg driver; swap for asyncpg
    return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://").replace(
        "postgresql://", "postgresql+asyncpg://"
    )


@pytest.fixture(scope="session")
def pg_dsn_sync(pg_container: PostgresContainer) -> str:
    """psycopg2-compatible URL for Alembic offline / sync ops."""
    url = pg_container.get_connection_url()
    return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")


@pytest.fixture(scope="session", autouse=True)
def apply_alembic_migrations(pg_dsn_sync: str) -> None:
    """Run Alembic upgrade once per session, before any test runs."""
    pkg_dir = Path(pydantic_ai_stateflow.__file__).parent
    cfg = Config(str(pkg_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(pkg_dir / "alembic"))
    cfg.set_main_option("sqlalchemy.url", pg_dsn_sync)
    command.upgrade(cfg, "head")


@pytest.fixture
async def session_factory(pg_dsn: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(pg_dsn)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()
```

(The 0001 migration is created in Task 13 — for now Alembic upgrade will be a no-op until that task. Tests in Task 7 will skip the integration tests if migration is empty; we'll un-skip in Task 13.)

For Task 7, the `PostgresThreadRepository` test depends on migrations existing. Two choices:
- (A) Use `SQLModel.metadata.create_all()` in a fixture instead of Alembic for early tasks → tests work immediately.
- (B) Defer all Postgres tests until Task 13 lands the migration.

Pick (A) — pragmatic and matches what most projects do for test setup:

Replace `apply_alembic_migrations` fixture with:

```python
@pytest.fixture(scope="session", autouse=True)
def create_all_tables(pg_dsn_sync: str) -> None:
    """Create all framework tables via SQLModel.metadata for tests.

    Production uses Alembic (see Task 13); tests use the live metadata to avoid
    needing migration files to be in lock-step with table definitions during dev.
    """
    from sqlalchemy import create_engine
    from sqlmodel import SQLModel

    # Import persistence modules to populate metadata
    import pydantic_ai_stateflow.persistence.tenant.persistence  # noqa: F401
    import pydantic_ai_stateflow.persistence.thread.persistence  # noqa: F401
    import pydantic_ai_stateflow.persistence.outbox.persistence  # noqa: F401
    import pydantic_ai_stateflow.persistence.hitl.persistence    # noqa: F401

    engine = create_engine(pg_dsn_sync)
    SQLModel.metadata.create_all(engine)
    engine.dispose()
```

- [ ] **Step 2: Failing test for PostgresThreadRepository**

`tests/persistence/test_thread_postgres.py`:

```python
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pydantic_ai_stateflow.persistence import SqlAlchemyUnitOfWork
from pydantic_ai_stateflow.persistence.tenant.persistence import TenantRow
from pydantic_ai_stateflow.persistence.thread import (
    PostgresThreadRepository,
    ThreadPurpose,
)


@pytest.fixture
async def tenant_id(session_factory: async_sessionmaker[AsyncSession]):
    tid = uuid4()
    async with session_factory() as s:
        s.add(TenantRow(id=tid, name="t1"))
        await s.commit()
    return tid


@pytest.mark.asyncio
async def test_create_and_load_thread(session_factory, tenant_id):
    uow = SqlAlchemyUnitOfWork(session_factory)
    async with uow:
        repo = PostgresThreadRepository(uow.session)
        thread = await repo.create(
            purpose=ThreadPurpose.CONVERSATION.value,
            purpose_metadata={"k": "v"},
            actor_id="founder-1",
            tenant_id=tenant_id,
        )

    async with session_factory() as s:
        repo2 = PostgresThreadRepository(s)
        loaded = await repo2.load(thread.id, tenant_id=tenant_id)
        assert loaded is not None
        assert loaded.purpose_metadata == {"k": "v"}


@pytest.mark.asyncio
async def test_load_cross_tenant_returns_none(session_factory, tenant_id):
    other_tid = uuid4()
    async with session_factory() as s:
        s.add(TenantRow(id=other_tid, name="t2"))
        await s.commit()

    uow = SqlAlchemyUnitOfWork(session_factory)
    async with uow:
        repo = PostgresThreadRepository(uow.session)
        thread = await repo.create(
            purpose=ThreadPurpose.CONVERSATION.value,
            purpose_metadata={},
            actor_id="a",
            tenant_id=tenant_id,
        )

    async with session_factory() as s:
        repo2 = PostgresThreadRepository(s)
        assert await repo2.load(thread.id, tenant_id=other_tid) is None


@pytest.mark.asyncio
async def test_add_message_and_history(session_factory, tenant_id):
    uow = SqlAlchemyUnitOfWork(session_factory)
    async with uow:
        repo = PostgresThreadRepository(uow.session)
        thread = await repo.create(
            purpose=ThreadPurpose.CONVERSATION.value,
            purpose_metadata={},
            actor_id="a",
            tenant_id=tenant_id,
        )
        await repo.add_message(thread.id, role="user", parts=[{"kind": "text", "content": "hi"}], tenant_id=tenant_id)
        await repo.add_message(thread.id, role="assistant", parts=[{"kind": "text", "content": "hey"}], tenant_id=tenant_id)

    async with session_factory() as s:
        repo2 = PostgresThreadRepository(s)
        history = await repo2.history(thread.id, tenant_id=tenant_id, limit=10)
        assert [m.role for m in history] == ["user", "assistant"]
```

- [ ] **Step 3: Run — fails**

- [ ] **Step 4: Implement PostgresThreadRepository**

`src/pydantic_ai_stateflow/persistence/thread/postgres.py`:

```python
from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic_ai_stateflow.persistence.thread.domain import Message, Thread
from pydantic_ai_stateflow.persistence.thread.persistence import MessageRow, ThreadRow


class PostgresThreadRepository:
    """SQLAlchemy-backed implementation of ThreadRepository.

    Lives inside `persistence/`. Callers should obtain a session via UnitOfWork
    rather than importing this class directly outside `persistence/` and
    `providers/`.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(
        self, *, purpose: str, purpose_metadata: dict[str, Any], actor_id: str, tenant_id: UUID
    ) -> Thread:
        row = ThreadRow(
            tenant_id=tenant_id, purpose=purpose,
            purpose_metadata=dict(purpose_metadata), actor_id=actor_id,
        )
        self._s.add(row)
        await self._s.flush()                                       # populate row.id without commit
        await self._s.refresh(row)
        return Thread.from_row(row)

    async def load(self, id: UUID, *, tenant_id: UUID) -> Thread | None:
        stmt = select(ThreadRow).where(ThreadRow.id == id, ThreadRow.tenant_id == tenant_id)
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        return Thread.from_row(row) if row is not None else None

    async def add_message(
        self, thread_id: UUID, *, role: str, parts: list[dict[str, Any]], tenant_id: UUID
    ) -> Message:
        # Defensive tenant check
        check = select(ThreadRow).where(ThreadRow.id == thread_id, ThreadRow.tenant_id == tenant_id)
        if (await self._s.execute(check)).scalar_one_or_none() is None:
            raise KeyError(f"Thread {thread_id} not found for tenant {tenant_id}")
        row = MessageRow(
            tenant_id=tenant_id, thread_id=thread_id,
            role=role, parts=list(parts),
        )
        self._s.add(row)
        await self._s.flush()
        await self._s.refresh(row)
        return Message.from_row(row)

    async def history(
        self, thread_id: UUID, *, tenant_id: UUID, limit: int = 100
    ) -> list[Message]:
        stmt = (
            select(MessageRow)
            .where(MessageRow.thread_id == thread_id, MessageRow.tenant_id == tenant_id)
            .order_by(MessageRow.created_at.asc())
            .limit(limit)
        )
        rows = (await self._s.execute(stmt)).scalars().all()
        return [Message.from_row(r) for r in rows]
```

Update `src/pydantic_ai_stateflow/persistence/thread/__init__.py`:

```python
from pydantic_ai_stateflow.persistence.thread.domain import Message, Thread, ThreadPurpose
from pydantic_ai_stateflow.persistence.thread.persistence import MessageRow, ThreadRow
from pydantic_ai_stateflow.persistence.thread.postgres import PostgresThreadRepository
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
    ThreadRepository,
)

__all__ = [
    "InMemoryThreadRepository",
    "Message",
    "MessageRow",
    "PostgresThreadRepository",
    "Thread",
    "ThreadPurpose",
    "ThreadRepository",
    "ThreadRow",
]
```

- [ ] **Step 5: Tests pass (3 new integration tests; testcontainers takes ~10s on first run)**

- [ ] **Step 6: Full suite + mypy + ruff**

- [ ] **Step 7: Commit**

```bash
git add src/pydantic_ai_stateflow/persistence/thread/postgres.py src/pydantic_ai_stateflow/persistence/thread/__init__.py tests/persistence/conftest.py tests/persistence/test_thread_postgres.py
git commit -m "feat(persistence): PostgresThreadRepository + testcontainers PG fixture"
```

---

## Task 8: `OutboxRepository` (Row + Protocol + InMemory + Postgres)

Follows the same shape as Tasks 5–7 for Thread but for the Outbox. Outbox enables transactional outbox pattern (Apply + EmitEvent in one transaction) used by MutationPipeline in Sub-project #5.

**Files:**
- Modify: `src/pydantic_ai_stateflow/persistence/outbox/persistence.py`
- Create: `src/pydantic_ai_stateflow/persistence/outbox/domain.py`
- Create: `src/pydantic_ai_stateflow/persistence/outbox/repository.py`
- Create: `src/pydantic_ai_stateflow/persistence/outbox/postgres.py`
- Create: `src/pydantic_ai_stateflow/persistence/outbox/__init__.py`
- Create: `tests/persistence/test_outbox_inmemory.py`
- Create: `tests/persistence/test_outbox_postgres.py`

- [ ] **Step 1: Write failing tests (both unit and integration)**

`tests/persistence/test_outbox_inmemory.py`:

```python
from uuid import uuid4

import pytest

from pydantic_ai_stateflow.persistence.outbox import (
    InMemoryOutboxRepository,
    OutboxRepository,
)


@pytest.mark.asyncio
async def test_enqueue_and_list_undelivered():
    repo: OutboxRepository = InMemoryOutboxRepository()
    tenant_id = uuid4()
    await repo.enqueue(
        event_type="OrderCreated",
        payload={"order_id": "abc", "amount": 100},
        tenant_id=tenant_id,
    )
    rows = await repo.list_undelivered(tenant_id=tenant_id, limit=10)
    assert len(rows) == 1
    assert rows[0].event_type == "OrderCreated"
    assert rows[0].payload == {"order_id": "abc", "amount": 100}


@pytest.mark.asyncio
async def test_mark_delivered_removes_from_undelivered_list():
    repo: OutboxRepository = InMemoryOutboxRepository()
    tenant_id = uuid4()
    await repo.enqueue(event_type="E", payload={}, tenant_id=tenant_id)
    [row] = await repo.list_undelivered(tenant_id=tenant_id, limit=10)
    await repo.mark_delivered(row.id, tenant_id=tenant_id)
    assert await repo.list_undelivered(tenant_id=tenant_id, limit=10) == []


@pytest.mark.asyncio
async def test_undelivered_is_per_tenant():
    repo: OutboxRepository = InMemoryOutboxRepository()
    t1, t2 = uuid4(), uuid4()
    await repo.enqueue(event_type="E1", payload={}, tenant_id=t1)
    await repo.enqueue(event_type="E2", payload={}, tenant_id=t2)
    rows_t1 = await repo.list_undelivered(tenant_id=t1, limit=10)
    assert len(rows_t1) == 1
    assert rows_t1[0].event_type == "E1"
```

`tests/persistence/test_outbox_postgres.py`:

```python
from uuid import uuid4

import pytest

from pydantic_ai_stateflow.persistence import SqlAlchemyUnitOfWork
from pydantic_ai_stateflow.persistence.outbox import PostgresOutboxRepository
from pydantic_ai_stateflow.persistence.tenant.persistence import TenantRow


@pytest.fixture
async def tenant_id(session_factory):
    tid = uuid4()
    async with session_factory() as s:
        s.add(TenantRow(id=tid, name="t-outbox"))
        await s.commit()
    return tid


@pytest.mark.asyncio
async def test_enqueue_and_list_undelivered_postgres(session_factory, tenant_id):
    uow = SqlAlchemyUnitOfWork(session_factory)
    async with uow:
        repo = PostgresOutboxRepository(uow.session)
        await repo.enqueue(event_type="OrderCreated", payload={"x": 1}, tenant_id=tenant_id)

    async with session_factory() as s:
        repo2 = PostgresOutboxRepository(s)
        rows = await repo2.list_undelivered(tenant_id=tenant_id, limit=10)
        assert any(r.event_type == "OrderCreated" for r in rows)


@pytest.mark.asyncio
async def test_mark_delivered_postgres(session_factory, tenant_id):
    uow = SqlAlchemyUnitOfWork(session_factory)
    async with uow:
        repo = PostgresOutboxRepository(uow.session)
        await repo.enqueue(event_type="E", payload={}, tenant_id=tenant_id)

    async with session_factory() as s:
        repo2 = PostgresOutboxRepository(s)
        [row] = await repo2.list_undelivered(tenant_id=tenant_id, limit=10)
        await repo2.mark_delivered(row.id, tenant_id=tenant_id)
        await s.commit()

    async with session_factory() as s:
        repo3 = PostgresOutboxRepository(s)
        assert await repo3.list_undelivered(tenant_id=tenant_id, limit=10) == []
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement**

`src/pydantic_ai_stateflow/persistence/outbox/persistence.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


class OutboxRow(SQLModel, table=True):
    __tablename__ = "outbox"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = Field(foreign_key="tenants.id", index=True)
    event_type: str
    payload: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    workflow_id: UUID | None = Field(default=None)
    delivered_at: datetime | None = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=_now_utc, index=True)
```

`src/pydantic_ai_stateflow/persistence/outbox/domain.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from pydantic_ai_stateflow.persistence.outbox.persistence import OutboxRow


class OutboxEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    tenant_id: UUID
    event_type: str
    payload: dict[str, Any]
    workflow_id: UUID | None
    delivered_at: datetime | None
    created_at: datetime

    @classmethod
    def from_row(cls, row: OutboxRow) -> "OutboxEvent":
        return cls(
            id=row.id, tenant_id=row.tenant_id, event_type=row.event_type,
            payload=row.payload, workflow_id=row.workflow_id,
            delivered_at=row.delivered_at, created_at=row.created_at,
        )
```

`src/pydantic_ai_stateflow/persistence/outbox/repository.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic_ai_stateflow.persistence.outbox.domain import OutboxEvent


@runtime_checkable
class OutboxRepository(Protocol):
    async def enqueue(
        self, *, event_type: str, payload: dict[str, Any], tenant_id: UUID,
        workflow_id: UUID | None = None,
    ) -> OutboxEvent: ...
    async def list_undelivered(
        self, *, tenant_id: UUID, limit: int = 100
    ) -> list[OutboxEvent]: ...
    async def mark_delivered(self, id: UUID, *, tenant_id: UUID) -> None: ...


class InMemoryOutboxRepository:
    def __init__(self) -> None:
        self._rows: list[OutboxEvent] = []

    async def enqueue(
        self, *, event_type: str, payload: dict[str, Any], tenant_id: UUID,
        workflow_id: UUID | None = None,
    ) -> OutboxEvent:
        event = OutboxEvent(
            id=uuid4(), tenant_id=tenant_id, event_type=event_type,
            payload=dict(payload), workflow_id=workflow_id,
            delivered_at=None, created_at=datetime.now(tz=UTC),
        )
        self._rows.append(event)
        return event

    async def list_undelivered(
        self, *, tenant_id: UUID, limit: int = 100
    ) -> list[OutboxEvent]:
        out = [r for r in self._rows if r.tenant_id == tenant_id and r.delivered_at is None]
        return out[:limit]

    async def mark_delivered(self, id: UUID, *, tenant_id: UUID) -> None:
        for i, r in enumerate(self._rows):
            if r.id == id and r.tenant_id == tenant_id:
                self._rows[i] = r.model_copy(update={"delivered_at": datetime.now(tz=UTC)})
                return
```

`src/pydantic_ai_stateflow/persistence/outbox/postgres.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic_ai_stateflow.persistence.outbox.domain import OutboxEvent
from pydantic_ai_stateflow.persistence.outbox.persistence import OutboxRow


class PostgresOutboxRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def enqueue(
        self, *, event_type: str, payload: dict[str, Any], tenant_id: UUID,
        workflow_id: UUID | None = None,
    ) -> OutboxEvent:
        row = OutboxRow(
            tenant_id=tenant_id, event_type=event_type,
            payload=dict(payload), workflow_id=workflow_id,
        )
        self._s.add(row)
        await self._s.flush()
        await self._s.refresh(row)
        return OutboxEvent.from_row(row)

    async def list_undelivered(
        self, *, tenant_id: UUID, limit: int = 100
    ) -> list[OutboxEvent]:
        stmt = (
            select(OutboxRow)
            .where(OutboxRow.tenant_id == tenant_id, OutboxRow.delivered_at.is_(None))
            .order_by(OutboxRow.created_at.asc())
            .limit(limit)
        )
        rows = (await self._s.execute(stmt)).scalars().all()
        return [OutboxEvent.from_row(r) for r in rows]

    async def mark_delivered(self, id: UUID, *, tenant_id: UUID) -> None:
        stmt = (
            update(OutboxRow)
            .where(OutboxRow.id == id, OutboxRow.tenant_id == tenant_id)
            .values(delivered_at=datetime.now(tz=UTC))
        )
        await self._s.execute(stmt)
```

`src/pydantic_ai_stateflow/persistence/outbox/__init__.py`:

```python
from pydantic_ai_stateflow.persistence.outbox.domain import OutboxEvent
from pydantic_ai_stateflow.persistence.outbox.persistence import OutboxRow
from pydantic_ai_stateflow.persistence.outbox.postgres import PostgresOutboxRepository
from pydantic_ai_stateflow.persistence.outbox.repository import (
    InMemoryOutboxRepository,
    OutboxRepository,
)

__all__ = [
    "InMemoryOutboxRepository",
    "OutboxEvent",
    "OutboxRepository",
    "OutboxRow",
    "PostgresOutboxRepository",
]
```

- [ ] **Step 4: Tests pass (3 InMemory + 2 Postgres)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/persistence/outbox tests/persistence/test_outbox_inmemory.py tests/persistence/test_outbox_postgres.py
git commit -m "feat(persistence): OutboxRepository (Row + Protocol + InMemory + Postgres)"
```

---

## Task 9: HITL persistence (3 tables: BlockingRequirement, Decision, AuthzDenial)

**Files:**
- Modify: `src/pydantic_ai_stateflow/persistence/hitl/persistence.py`
- Create: `src/pydantic_ai_stateflow/persistence/hitl/domain.py`
- Create: `src/pydantic_ai_stateflow/persistence/hitl/__init__.py`
- Create: `tests/persistence/test_hitl_models.py`

- [ ] **Step 1: Failing tests**

`tests/persistence/test_hitl_models.py`:

```python
from datetime import UTC, datetime
from uuid import uuid4

from sqlmodel import SQLModel

from pydantic_ai_stateflow.persistence.hitl import (
    AuthzDenialRow,
    BlockingRequirement,
    BlockingRequirementRow,
    BlockingRequirementStatus,
    Decision,
    DecisionRow,
    DecisionVerdict,
    HITLPurpose,
)


def test_hitl_tables_registered():
    for name in ("hitl_blocking_requirements", "hitl_decisions", "hitl_authz_denials"):
        assert name in SQLModel.metadata.tables


def test_blocking_requirement_row_minimal():
    row = BlockingRequirementRow(
        tenant_id=uuid4(),
        gate_kind="strategy_review",
        workflow_id=uuid4(),
        payload={"prompt": "approve?"},
        purpose=HITLPurpose.APPROVAL.value,
        status=BlockingRequirementStatus.PENDING.value,
    )
    assert row.gate_kind == "strategy_review"
    assert row.payload == {"prompt": "approve?"}


def test_decision_row_minimal():
    row = DecisionRow(
        tenant_id=uuid4(),
        blocking_requirement_id=uuid4(),
        actor_id="founder-1",
        verdict=DecisionVerdict.APPROVE.value,
        payload={"feedback": "ok"},
    )
    assert row.verdict == "approve"
    assert row.helper_verdict_payload is None
    assert row.helper_verdict_context_type is None


def test_authz_denial_row_minimal():
    row = AuthzDenialRow(
        tenant_id=uuid4(),
        request_id=uuid4(),
        actor_id="intruder",
        voter_votes={"voter1": "DENY"},
    )
    assert row.actor_id == "intruder"


def test_domain_models_from_rows():
    req_row = BlockingRequirementRow(
        tenant_id=uuid4(), gate_kind="x", workflow_id=uuid4(),
        payload={}, purpose=HITLPurpose.APPROVAL.value,
        status=BlockingRequirementStatus.PENDING.value,
    )
    domain_req = BlockingRequirement.from_row(req_row)
    assert domain_req.status == BlockingRequirementStatus.PENDING
    assert domain_req.purpose == HITLPurpose.APPROVAL

    dec_row = DecisionRow(
        tenant_id=uuid4(), blocking_requirement_id=uuid4(),
        actor_id="a", verdict=DecisionVerdict.REJECT.value, payload={"reason": "no"},
    )
    domain_dec = Decision.from_row(dec_row)
    assert domain_dec.verdict == DecisionVerdict.REJECT
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement**

`src/pydantic_ai_stateflow/persistence/hitl/persistence.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


class BlockingRequirementRow(SQLModel, table=True):
    __tablename__ = "hitl_blocking_requirements"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = Field(foreign_key="tenants.id", index=True)
    gate_kind: str
    workflow_id: UUID = Field(index=True)
    payload: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    purpose: str                                              # HITLPurpose value
    status: str                                               # BlockingRequirementStatus value
    timeout_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=_now_utc, index=True)
    resolved_at: datetime | None = Field(default=None)


class DecisionRow(SQLModel, table=True):
    __tablename__ = "hitl_decisions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = Field(foreign_key="tenants.id", index=True)
    blocking_requirement_id: UUID = Field(
        foreign_key="hitl_blocking_requirements.id", index=True
    )
    actor_id: str
    verdict: str                                              # DecisionVerdict value
    payload: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    helper_verdict_payload: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
    )
    helper_verdict_context_type: str | None = Field(default=None)
    helper_thread_id: UUID | None = Field(default=None, foreign_key="threads.id")
    created_at: datetime = Field(default_factory=_now_utc, index=True)


class AuthzDenialRow(SQLModel, table=True):
    __tablename__ = "hitl_authz_denials"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = Field(foreign_key="tenants.id", index=True)
    request_id: UUID = Field(foreign_key="hitl_blocking_requirements.id", index=True)
    actor_id: str
    voter_votes: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    attempted_at: datetime = Field(default_factory=_now_utc)
```

`src/pydantic_ai_stateflow/persistence/hitl/domain.py`:

```python
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from pydantic_ai_stateflow.persistence.hitl.persistence import (
    AuthzDenialRow,
    BlockingRequirementRow,
    DecisionRow,
)


class HITLPurpose(StrEnum):
    APPROVAL = "approval"
    REJECT_RECOVERY = "reject_recovery"
    AMBIGUITY = "ambiguity"
    POLICY_CONFLICT = "policy_conflict"


class BlockingRequirementStatus(StrEnum):
    PENDING = "pending"
    RESOLVED = "resolved"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class DecisionVerdict(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    REVISE = "revise"
    OVERRIDE = "override"


class BlockingRequirement(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID
    tenant_id: UUID
    gate_kind: str
    workflow_id: UUID
    payload: dict[str, Any]
    purpose: HITLPurpose
    status: BlockingRequirementStatus
    timeout_at: datetime | None
    created_at: datetime
    resolved_at: datetime | None

    @classmethod
    def from_row(cls, row: BlockingRequirementRow) -> "BlockingRequirement":
        return cls(
            id=row.id, tenant_id=row.tenant_id, gate_kind=row.gate_kind,
            workflow_id=row.workflow_id, payload=row.payload,
            purpose=HITLPurpose(row.purpose),
            status=BlockingRequirementStatus(row.status),
            timeout_at=row.timeout_at, created_at=row.created_at,
            resolved_at=row.resolved_at,
        )


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID
    tenant_id: UUID
    blocking_requirement_id: UUID
    actor_id: str
    verdict: DecisionVerdict
    payload: dict[str, Any]
    helper_verdict_payload: dict[str, Any] | None
    helper_verdict_context_type: str | None
    helper_thread_id: UUID | None
    created_at: datetime

    @classmethod
    def from_row(cls, row: DecisionRow) -> "Decision":
        return cls(
            id=row.id, tenant_id=row.tenant_id,
            blocking_requirement_id=row.blocking_requirement_id,
            actor_id=row.actor_id, verdict=DecisionVerdict(row.verdict),
            payload=row.payload,
            helper_verdict_payload=row.helper_verdict_payload,
            helper_verdict_context_type=row.helper_verdict_context_type,
            helper_thread_id=row.helper_thread_id,
            created_at=row.created_at,
        )


class AuthzDenial(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID
    tenant_id: UUID
    request_id: UUID
    actor_id: str
    voter_votes: dict[str, Any]
    attempted_at: datetime

    @classmethod
    def from_row(cls, row: AuthzDenialRow) -> "AuthzDenial":
        return cls(
            id=row.id, tenant_id=row.tenant_id, request_id=row.request_id,
            actor_id=row.actor_id, voter_votes=row.voter_votes,
            attempted_at=row.attempted_at,
        )
```

`src/pydantic_ai_stateflow/persistence/hitl/__init__.py`:

```python
from pydantic_ai_stateflow.persistence.hitl.domain import (
    AuthzDenial,
    BlockingRequirement,
    BlockingRequirementStatus,
    Decision,
    DecisionVerdict,
    HITLPurpose,
)
from pydantic_ai_stateflow.persistence.hitl.persistence import (
    AuthzDenialRow,
    BlockingRequirementRow,
    DecisionRow,
)

__all__ = [
    "AuthzDenial",
    "AuthzDenialRow",
    "BlockingRequirement",
    "BlockingRequirementRow",
    "BlockingRequirementStatus",
    "Decision",
    "DecisionRow",
    "DecisionVerdict",
    "HITLPurpose",
]
```

- [ ] **Step 4: Tests pass**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/persistence/hitl tests/persistence/test_hitl_models.py
git commit -m "feat(persistence): HITL tables (BlockingRequirement, Decision, AuthzDenial) + domain"
```

---

## Task 10: `HITLRepository` Protocol + InMemory + Postgres

**Files:**
- Create: `src/pydantic_ai_stateflow/persistence/hitl/repository.py`
- Create: `src/pydantic_ai_stateflow/persistence/hitl/postgres.py`
- Modify: `src/pydantic_ai_stateflow/persistence/hitl/__init__.py`
- Create: `tests/persistence/test_hitl_inmemory.py`
- Create: `tests/persistence/test_hitl_postgres.py`

- [ ] **Step 1: Failing tests**

`tests/persistence/test_hitl_inmemory.py`:

```python
from uuid import uuid4

import pytest

from pydantic_ai_stateflow.persistence.hitl import (
    BlockingRequirementStatus,
    DecisionVerdict,
    HITLPurpose,
    InMemoryHITLRepository,
)


@pytest.fixture
def repo() -> InMemoryHITLRepository:
    return InMemoryHITLRepository()


@pytest.fixture
def tenant_id():
    return uuid4()


@pytest.mark.asyncio
async def test_persist_request_creates_pending_record(repo, tenant_id):
    workflow_id = uuid4()
    req = await repo.persist_request(
        prompt={"title": "approve?"},
        workflow_id=workflow_id, gate_kind="strategy_review",
        purpose=HITLPurpose.APPROVAL.value, tenant_id=tenant_id,
    )
    assert req.status == BlockingRequirementStatus.PENDING
    assert req.gate_kind == "strategy_review"


@pytest.mark.asyncio
async def test_persist_response_resolves_request(repo, tenant_id):
    req = await repo.persist_request(
        prompt={}, workflow_id=uuid4(), gate_kind="g",
        purpose=HITLPurpose.APPROVAL.value, tenant_id=tenant_id,
    )
    dec = await repo.persist_response(
        request_id=req.id, actor_id="founder-1",
        verdict=DecisionVerdict.APPROVE.value, payload={},
        tenant_id=tenant_id,
    )
    assert dec.verdict == DecisionVerdict.APPROVE
    # Request should now be resolved
    loaded = await repo.load_request(req.id, tenant_id=tenant_id)
    assert loaded.status == BlockingRequirementStatus.RESOLVED


@pytest.mark.asyncio
async def test_persist_timeout_marks_status(repo, tenant_id):
    req = await repo.persist_request(
        prompt={}, workflow_id=uuid4(), gate_kind="g",
        purpose=HITLPurpose.APPROVAL.value, tenant_id=tenant_id,
    )
    await repo.persist_timeout(req.id, tenant_id=tenant_id)
    loaded = await repo.load_request(req.id, tenant_id=tenant_id)
    assert loaded.status == BlockingRequirementStatus.TIMED_OUT


@pytest.mark.asyncio
async def test_persist_authz_denied_records_attempt(repo, tenant_id):
    req = await repo.persist_request(
        prompt={}, workflow_id=uuid4(), gate_kind="g",
        purpose=HITLPurpose.APPROVAL.value, tenant_id=tenant_id,
    )
    await repo.persist_authz_denied(
        request_id=req.id, actor_id="intruder",
        voter_votes={"v1": "DENY"}, tenant_id=tenant_id,
    )
    # Request stays pending
    loaded = await repo.load_request(req.id, tenant_id=tenant_id)
    assert loaded.status == BlockingRequirementStatus.PENDING


@pytest.mark.asyncio
async def test_list_pending_for_tenant(repo, tenant_id):
    other = uuid4()
    await repo.persist_request(prompt={}, workflow_id=uuid4(), gate_kind="g",
                                purpose=HITLPurpose.APPROVAL.value, tenant_id=tenant_id)
    await repo.persist_request(prompt={}, workflow_id=uuid4(), gate_kind="g",
                                purpose=HITLPurpose.APPROVAL.value, tenant_id=other)
    pending = await repo.list_pending(tenant_id=tenant_id)
    assert all(p.tenant_id == tenant_id for p in pending)
    assert len(pending) == 1
```

`tests/persistence/test_hitl_postgres.py`:

```python
from uuid import uuid4

import pytest

from pydantic_ai_stateflow.persistence import SqlAlchemyUnitOfWork
from pydantic_ai_stateflow.persistence.hitl import (
    BlockingRequirementStatus,
    DecisionVerdict,
    HITLPurpose,
    PostgresHITLRepository,
)
from pydantic_ai_stateflow.persistence.tenant.persistence import TenantRow


@pytest.fixture
async def tenant_id(session_factory):
    tid = uuid4()
    async with session_factory() as s:
        s.add(TenantRow(id=tid, name="t-hitl"))
        await s.commit()
    return tid


@pytest.mark.asyncio
async def test_request_response_postgres(session_factory, tenant_id):
    uow = SqlAlchemyUnitOfWork(session_factory)
    async with uow:
        repo = PostgresHITLRepository(uow.session)
        req = await repo.persist_request(
            prompt={"title": "go?"}, workflow_id=uuid4(), gate_kind="g",
            purpose=HITLPurpose.APPROVAL.value, tenant_id=tenant_id,
        )

    async with session_factory() as s:
        repo2 = PostgresHITLRepository(s)
        loaded = await repo2.load_request(req.id, tenant_id=tenant_id)
        assert loaded.status == BlockingRequirementStatus.PENDING

    uow2 = SqlAlchemyUnitOfWork(session_factory)
    async with uow2:
        repo3 = PostgresHITLRepository(uow2.session)
        await repo3.persist_response(
            request_id=req.id, actor_id="founder",
            verdict=DecisionVerdict.APPROVE.value, payload={}, tenant_id=tenant_id,
        )

    async with session_factory() as s:
        repo4 = PostgresHITLRepository(s)
        final = await repo4.load_request(req.id, tenant_id=tenant_id)
        assert final.status == BlockingRequirementStatus.RESOLVED
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement**

`src/pydantic_ai_stateflow/persistence/hitl/repository.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic_ai_stateflow.persistence.hitl.domain import (
    AuthzDenial,
    BlockingRequirement,
    BlockingRequirementStatus,
    Decision,
    DecisionVerdict,
    HITLPurpose,
)


@runtime_checkable
class HITLRepository(Protocol):
    async def persist_request(
        self, *, prompt: dict[str, Any], workflow_id: UUID,
        gate_kind: str, purpose: str, tenant_id: UUID,
        timeout_at: datetime | None = None,
    ) -> BlockingRequirement: ...

    async def load_request(self, request_id: UUID, *, tenant_id: UUID) -> BlockingRequirement | None: ...

    async def persist_response(
        self, *, request_id: UUID, actor_id: str, verdict: str,
        payload: dict[str, Any], tenant_id: UUID,
        helper_verdict_payload: dict[str, Any] | None = None,
        helper_verdict_context_type: str | None = None,
        helper_thread_id: UUID | None = None,
    ) -> Decision: ...

    async def persist_timeout(self, request_id: UUID, *, tenant_id: UUID) -> None: ...

    async def persist_authz_denied(
        self, *, request_id: UUID, actor_id: str,
        voter_votes: dict[str, Any], tenant_id: UUID,
    ) -> AuthzDenial: ...

    async def list_pending(self, *, tenant_id: UUID, limit: int = 100) -> list[BlockingRequirement]: ...


class InMemoryHITLRepository:
    def __init__(self) -> None:
        self._requests: dict[UUID, BlockingRequirement] = {}
        self._decisions: dict[UUID, Decision] = {}
        self._denials: list[AuthzDenial] = []

    async def persist_request(
        self, *, prompt, workflow_id, gate_kind, purpose, tenant_id, timeout_at=None,
    ):
        req = BlockingRequirement(
            id=uuid4(), tenant_id=tenant_id, gate_kind=gate_kind,
            workflow_id=workflow_id, payload=prompt,
            purpose=HITLPurpose(purpose),
            status=BlockingRequirementStatus.PENDING,
            timeout_at=timeout_at, created_at=datetime.now(tz=UTC), resolved_at=None,
        )
        self._requests[req.id] = req
        return req

    async def load_request(self, request_id, *, tenant_id):
        req = self._requests.get(request_id)
        if req is None or req.tenant_id != tenant_id:
            return None
        return req

    async def persist_response(
        self, *, request_id, actor_id, verdict, payload, tenant_id,
        helper_verdict_payload=None, helper_verdict_context_type=None,
        helper_thread_id=None,
    ):
        req = self._requests.get(request_id)
        if req is None or req.tenant_id != tenant_id:
            raise KeyError(f"Request {request_id} not found")
        dec = Decision(
            id=uuid4(), tenant_id=tenant_id,
            blocking_requirement_id=request_id, actor_id=actor_id,
            verdict=DecisionVerdict(verdict), payload=payload,
            helper_verdict_payload=helper_verdict_payload,
            helper_verdict_context_type=helper_verdict_context_type,
            helper_thread_id=helper_thread_id,
            created_at=datetime.now(tz=UTC),
        )
        self._decisions[dec.id] = dec
        self._requests[request_id] = req.model_copy(update={
            "status": BlockingRequirementStatus.RESOLVED,
            "resolved_at": datetime.now(tz=UTC),
        })
        return dec

    async def persist_timeout(self, request_id, *, tenant_id):
        req = self._requests.get(request_id)
        if req is None or req.tenant_id != tenant_id:
            return
        self._requests[request_id] = req.model_copy(update={
            "status": BlockingRequirementStatus.TIMED_OUT,
            "resolved_at": datetime.now(tz=UTC),
        })

    async def persist_authz_denied(
        self, *, request_id, actor_id, voter_votes, tenant_id,
    ):
        denial = AuthzDenial(
            id=uuid4(), tenant_id=tenant_id, request_id=request_id,
            actor_id=actor_id, voter_votes=dict(voter_votes),
            attempted_at=datetime.now(tz=UTC),
        )
        self._denials.append(denial)
        return denial

    async def list_pending(self, *, tenant_id, limit=100):
        return [r for r in self._requests.values()
                if r.tenant_id == tenant_id
                and r.status == BlockingRequirementStatus.PENDING][:limit]
```

`src/pydantic_ai_stateflow/persistence/hitl/postgres.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic_ai_stateflow.persistence.hitl.domain import (
    AuthzDenial,
    BlockingRequirement,
    BlockingRequirementStatus,
    Decision,
)
from pydantic_ai_stateflow.persistence.hitl.persistence import (
    AuthzDenialRow,
    BlockingRequirementRow,
    DecisionRow,
)


class PostgresHITLRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def persist_request(
        self, *, prompt: dict[str, Any], workflow_id: UUID,
        gate_kind: str, purpose: str, tenant_id: UUID,
        timeout_at: datetime | None = None,
    ) -> BlockingRequirement:
        row = BlockingRequirementRow(
            tenant_id=tenant_id, gate_kind=gate_kind,
            workflow_id=workflow_id, payload=dict(prompt),
            purpose=purpose, status=BlockingRequirementStatus.PENDING.value,
            timeout_at=timeout_at,
        )
        self._s.add(row)
        await self._s.flush()
        await self._s.refresh(row)
        return BlockingRequirement.from_row(row)

    async def load_request(self, request_id: UUID, *, tenant_id: UUID) -> BlockingRequirement | None:
        stmt = select(BlockingRequirementRow).where(
            BlockingRequirementRow.id == request_id,
            BlockingRequirementRow.tenant_id == tenant_id,
        )
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        return BlockingRequirement.from_row(row) if row is not None else None

    async def persist_response(
        self, *, request_id: UUID, actor_id: str, verdict: str,
        payload: dict[str, Any], tenant_id: UUID,
        helper_verdict_payload: dict[str, Any] | None = None,
        helper_verdict_context_type: str | None = None,
        helper_thread_id: UUID | None = None,
    ) -> Decision:
        row = DecisionRow(
            tenant_id=tenant_id, blocking_requirement_id=request_id,
            actor_id=actor_id, verdict=verdict, payload=dict(payload),
            helper_verdict_payload=helper_verdict_payload,
            helper_verdict_context_type=helper_verdict_context_type,
            helper_thread_id=helper_thread_id,
        )
        self._s.add(row)
        now = datetime.now(tz=UTC)
        await self._s.execute(
            update(BlockingRequirementRow)
            .where(BlockingRequirementRow.id == request_id,
                   BlockingRequirementRow.tenant_id == tenant_id)
            .values(status=BlockingRequirementStatus.RESOLVED.value, resolved_at=now)
        )
        await self._s.flush()
        await self._s.refresh(row)
        return Decision.from_row(row)

    async def persist_timeout(self, request_id: UUID, *, tenant_id: UUID) -> None:
        now = datetime.now(tz=UTC)
        await self._s.execute(
            update(BlockingRequirementRow)
            .where(BlockingRequirementRow.id == request_id,
                   BlockingRequirementRow.tenant_id == tenant_id)
            .values(status=BlockingRequirementStatus.TIMED_OUT.value, resolved_at=now)
        )

    async def persist_authz_denied(
        self, *, request_id: UUID, actor_id: str,
        voter_votes: dict[str, Any], tenant_id: UUID,
    ) -> AuthzDenial:
        row = AuthzDenialRow(
            tenant_id=tenant_id, request_id=request_id,
            actor_id=actor_id, voter_votes=dict(voter_votes),
        )
        self._s.add(row)
        await self._s.flush()
        await self._s.refresh(row)
        return AuthzDenial.from_row(row)

    async def list_pending(
        self, *, tenant_id: UUID, limit: int = 100
    ) -> list[BlockingRequirement]:
        stmt = (
            select(BlockingRequirementRow)
            .where(BlockingRequirementRow.tenant_id == tenant_id,
                   BlockingRequirementRow.status == BlockingRequirementStatus.PENDING.value)
            .order_by(BlockingRequirementRow.created_at.asc())
            .limit(limit)
        )
        rows = (await self._s.execute(stmt)).scalars().all()
        return [BlockingRequirement.from_row(r) for r in rows]
```

Update `src/pydantic_ai_stateflow/persistence/hitl/__init__.py` adding the new exports:

```python
from pydantic_ai_stateflow.persistence.hitl.domain import (
    AuthzDenial,
    BlockingRequirement,
    BlockingRequirementStatus,
    Decision,
    DecisionVerdict,
    HITLPurpose,
)
from pydantic_ai_stateflow.persistence.hitl.persistence import (
    AuthzDenialRow,
    BlockingRequirementRow,
    DecisionRow,
)
from pydantic_ai_stateflow.persistence.hitl.postgres import PostgresHITLRepository
from pydantic_ai_stateflow.persistence.hitl.repository import (
    HITLRepository,
    InMemoryHITLRepository,
)

__all__ = [
    "AuthzDenial",
    "AuthzDenialRow",
    "BlockingRequirement",
    "BlockingRequirementRow",
    "BlockingRequirementStatus",
    "Decision",
    "DecisionRow",
    "DecisionVerdict",
    "HITLPurpose",
    "HITLRepository",
    "InMemoryHITLRepository",
    "PostgresHITLRepository",
]
```

- [ ] **Step 4: Tests pass (5 InMemory + 1 Postgres)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/persistence/hitl tests/persistence/test_hitl_inmemory.py tests/persistence/test_hitl_postgres.py
git commit -m "feat(persistence): HITLRepository Protocol + InMemory + Postgres"
```

---

## Task 11: `testing` package — public InMemory aliases for downstream consumers

**Files:**
- Create: `src/pydantic_ai_stateflow/testing/__init__.py`
- Create: `tests/test_testing_package.py`

- [ ] **Step 1: Failing test**

`tests/test_testing_package.py`:

```python
from pydantic_ai_stateflow.testing import (
    InMemoryHITLRepository,
    InMemoryOutboxRepository,
    InMemoryThreadRepository,
)


def test_testing_exports_inmemory_repos():
    assert InMemoryThreadRepository is not None
    assert InMemoryOutboxRepository is not None
    assert InMemoryHITLRepository is not None
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement**

`src/pydantic_ai_stateflow/testing/__init__.py`:

```python
"""Test doubles and helpers for downstream consumers.

Re-exports all `InMemory*` repository implementations so test code can
import everything from one place:

    from pydantic_ai_stateflow.testing import InMemoryThreadRepository
"""

from pydantic_ai_stateflow.persistence.hitl import InMemoryHITLRepository
from pydantic_ai_stateflow.persistence.outbox import InMemoryOutboxRepository
from pydantic_ai_stateflow.persistence.thread import InMemoryThreadRepository

__all__ = [
    "InMemoryHITLRepository",
    "InMemoryOutboxRepository",
    "InMemoryThreadRepository",
]
```

- [ ] **Step 4: Tests pass**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/testing tests/test_testing_package.py
git commit -m "feat(testing): public InMemory* repository aliases"
```

---

## Task 12: Top-level `persistence` public API

**Files:**
- Modify: `src/pydantic_ai_stateflow/persistence/__init__.py`
- Modify: `src/pydantic_ai_stateflow/__init__.py`
- Create: `tests/persistence/test_public_api.py`

- [ ] **Step 1: Failing test**

`tests/persistence/test_public_api.py`:

```python
def test_persistence_public_api():
    """Persistence-layer Protocols are importable from top-level package."""
    from pydantic_ai_stateflow.persistence import (
        HITLRepository,
        OutboxRepository,
        SqlAlchemyUnitOfWork,
        ThreadRepository,
        UnitOfWork,
    )

    assert UnitOfWork is not None
    assert SqlAlchemyUnitOfWork is not None
    assert ThreadRepository is not None
    assert OutboxRepository is not None
    assert HITLRepository is not None
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Update `src/pydantic_ai_stateflow/persistence/__init__.py`**

```python
from pydantic_ai_stateflow.persistence.hitl import (
    HITLRepository,
    InMemoryHITLRepository,
    PostgresHITLRepository,
)
from pydantic_ai_stateflow.persistence.outbox import (
    InMemoryOutboxRepository,
    OutboxRepository,
    PostgresOutboxRepository,
)
from pydantic_ai_stateflow.persistence.thread import (
    InMemoryThreadRepository,
    PostgresThreadRepository,
    ThreadRepository,
)
from pydantic_ai_stateflow.persistence.uow import SqlAlchemyUnitOfWork, UnitOfWork

__all__ = [
    "HITLRepository",
    "InMemoryHITLRepository",
    "InMemoryOutboxRepository",
    "InMemoryThreadRepository",
    "OutboxRepository",
    "PostgresHITLRepository",
    "PostgresOutboxRepository",
    "PostgresThreadRepository",
    "SqlAlchemyUnitOfWork",
    "ThreadRepository",
    "UnitOfWork",
]
```

- [ ] **Step 4: Tests pass**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/persistence/__init__.py tests/persistence/test_public_api.py
git commit -m "feat(persistence): top-level persistence public API"
```

---

## Task 13: Alembic 0001 — generate the framework migration

**Files:**
- Create: `src/pydantic_ai_stateflow/alembic/versions/0001_framework_tables.py`
- Create: `tests/persistence/test_alembic_migration.py`

- [ ] **Step 1: Failing test**

`tests/persistence/test_alembic_migration.py`:

```python
"""Verify Alembic upgrade to head creates all framework tables."""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

import pydantic_ai_stateflow


@pytest.fixture
def fresh_engine(pg_dsn_sync: str):
    """Create a separate fresh engine pointing to a unique schema so this
    test does not clobber the shared `create_all_tables` fixture's state."""
    engine = create_engine(pg_dsn_sync)
    yield engine
    engine.dispose()


def test_alembic_upgrade_creates_framework_tables(fresh_engine, pg_dsn_sync):
    """Run alembic upgrade head and inspect that all expected tables exist."""
    pkg_dir = Path(pydantic_ai_stateflow.__file__).parent
    cfg = Config(str(pkg_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(pkg_dir / "alembic"))
    cfg.set_main_option("sqlalchemy.url", pg_dsn_sync)

    # The fixture above already created all tables via SQLModel.metadata;
    # Alembic upgrade head should be idempotent on top of that schema.
    command.upgrade(cfg, "head")

    inspector = inspect(fresh_engine)
    table_names = set(inspector.get_table_names())
    expected = {
        "tenants",
        "threads",
        "messages",
        "outbox",
        "hitl_blocking_requirements",
        "hitl_decisions",
        "hitl_authz_denials",
        "alembic_version",
    }
    missing = expected - table_names
    assert not missing, f"Missing tables after Alembic upgrade: {missing}"
```

- [ ] **Step 2: Run — fails (no migration file)**

- [ ] **Step 3: Generate the migration**

Since `apply_alembic_migrations` doesn't run autogenerate as part of the test, write the migration by hand. The migration is the explicit DDL for all framework tables; you can either author it from scratch or run `alembic revision --autogenerate` against an empty DB once and commit the result.

Recommended approach for this task: write the migration by hand from the row definitions.

`src/pydantic_ai_stateflow/alembic/versions/0001_framework_tables.py`:

```python
"""framework tables: tenants, threads, messages, outbox, hitl_*

Revision ID: 0001
Revises:
Create Date: 2026-05-15 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "threads",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("tenants.id"), nullable=False, index=True),
        sa.Column("purpose", sa.String(), nullable=False),
        sa.Column("purpose_metadata", postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("workflow_id", postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column("actor_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_threads_tenant_id", "threads", ["tenant_id"])

    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("tenants.id"), nullable=False, index=True),
        sa.Column("thread_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("threads.id"), nullable=False, index=True),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("parts", postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "outbox",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("tenants.id"), nullable=False, index=True),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("workflow_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )

    op.create_table(
        "hitl_blocking_requirements",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("tenants.id"), nullable=False, index=True),
        sa.Column("gate_kind", sa.String(), nullable=False),
        sa.Column("workflow_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("purpose", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("timeout_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "hitl_decisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("tenants.id"), nullable=False, index=True),
        sa.Column("blocking_requirement_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("hitl_blocking_requirements.id"), nullable=False, index=True),
        sa.Column("actor_id", sa.String(), nullable=False),
        sa.Column("verdict", sa.String(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("helper_verdict_payload",
                  postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("helper_verdict_context_type", sa.String(), nullable=True),
        sa.Column("helper_thread_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("threads.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )

    op.create_table(
        "hitl_authz_denials",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("tenants.id"), nullable=False, index=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("hitl_blocking_requirements.id"), nullable=False, index=True),
        sa.Column("actor_id", sa.String(), nullable=False),
        sa.Column("voter_votes", postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("attempted_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("hitl_authz_denials")
    op.drop_table("hitl_decisions")
    op.drop_table("hitl_blocking_requirements")
    op.drop_table("outbox")
    op.drop_table("messages")
    op.drop_table("threads")
    op.drop_table("tenants")
```

- [ ] **Step 4: Test passes (Alembic upgrade head is idempotent over existing tables on the test PG)**

```bash
uv run pytest tests/persistence/test_alembic_migration.py -v
```

- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/alembic/versions/0001_framework_tables.py tests/persistence/test_alembic_migration.py
git commit -m "feat(persistence): Alembic 0001 — framework tables migration"
```

---

## Task 14: End-to-end smoke test across all repos

**Files:**
- Create: `tests/integration/test_state_smoke.py`

- [ ] **Step 1: Write the smoke test**

`tests/integration/test_state_smoke.py`:

```python
"""End-to-end smoke test exercising tenant + thread + message + outbox + HITL in one PG session."""

from uuid import uuid4

import pytest

from pydantic_ai_stateflow.persistence import (
    PostgresHITLRepository,
    PostgresOutboxRepository,
    PostgresThreadRepository,
    SqlAlchemyUnitOfWork,
)
from pydantic_ai_stateflow.persistence.hitl import (
    BlockingRequirementStatus,
    DecisionVerdict,
    HITLPurpose,
)
from pydantic_ai_stateflow.persistence.tenant.persistence import TenantRow
from pydantic_ai_stateflow.persistence.thread import ThreadPurpose


@pytest.mark.asyncio
async def test_full_state_round_trip(session_factory):
    """Single tenant, single thread, two messages, one outbox event, one HITL approval — all in PG."""
    tenant_id = uuid4()
    workflow_id = uuid4()

    # 1. Create tenant
    async with session_factory() as s:
        s.add(TenantRow(id=tenant_id, name="smoke-tenant"))
        await s.commit()

    # 2. Create thread + add two messages
    uow1 = SqlAlchemyUnitOfWork(session_factory)
    async with uow1:
        threads = PostgresThreadRepository(uow1.session)
        thread = await threads.create(
            purpose=ThreadPurpose.HITL.value,
            purpose_metadata={"gate_kind": "strategy_review"},
            actor_id="founder-1",
            tenant_id=tenant_id,
        )
        await threads.add_message(
            thread.id, role="user", parts=[{"kind": "text", "content": "approve please"}], tenant_id=tenant_id
        )
        await threads.add_message(
            thread.id, role="assistant", parts=[{"kind": "text", "content": "context follows..."}], tenant_id=tenant_id
        )

    # 3. Enqueue an outbox event + persist a HITL request in the SAME tx (transactional outbox)
    uow2 = SqlAlchemyUnitOfWork(session_factory)
    async with uow2:
        outbox = PostgresOutboxRepository(uow2.session)
        hitl = PostgresHITLRepository(uow2.session)

        req = await hitl.persist_request(
            prompt={"title": "approve order?", "amount": 100},
            workflow_id=workflow_id,
            gate_kind="strategy_review",
            purpose=HITLPurpose.APPROVAL.value,
            tenant_id=tenant_id,
        )
        await outbox.enqueue(
            event_type="HITLRequested",
            payload={"request_id": str(req.id), "gate_kind": "strategy_review"},
            tenant_id=tenant_id,
            workflow_id=workflow_id,
        )

    # 4. Founder responds with approve
    uow3 = SqlAlchemyUnitOfWork(session_factory)
    async with uow3:
        hitl = PostgresHITLRepository(uow3.session)
        await hitl.persist_response(
            request_id=req.id,
            actor_id="founder-1",
            verdict=DecisionVerdict.APPROVE.value,
            payload={"feedback": "looks good"},
            tenant_id=tenant_id,
            helper_verdict_payload={"rationale": "all checks passed", "confidence": 0.92},
            helper_verdict_context_type="waves_app.hitl.contexts.StrategyReviewContext",
            helper_thread_id=thread.id,
        )

    # 5. Verify final state
    async with session_factory() as s:
        threads = PostgresThreadRepository(s)
        history = await threads.history(thread.id, tenant_id=tenant_id, limit=10)
        assert [m.role for m in history] == ["user", "assistant"]

        hitl = PostgresHITLRepository(s)
        resolved = await hitl.load_request(req.id, tenant_id=tenant_id)
        assert resolved.status == BlockingRequirementStatus.RESOLVED

        outbox = PostgresOutboxRepository(s)
        events = await outbox.list_undelivered(tenant_id=tenant_id, limit=10)
        assert any(e.event_type == "HITLRequested" for e in events)
```

- [ ] **Step 2: Run — must pass**

```bash
uv run pytest tests/integration/test_state_smoke.py -v
```

- [ ] **Step 3: Full suite + mypy + ruff**
- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_state_smoke.py
git commit -m "test: end-to-end state smoke test (tenant + thread + messages + outbox + HITL)"
```

---

## Sub-project #2 acceptance criteria

After all 14 tasks:

- ✅ `from pydantic_ai_stateflow.persistence import UnitOfWork, ThreadRepository, OutboxRepository, HITLRepository` works
- ✅ `from pydantic_ai_stateflow.testing import InMemoryThreadRepository, InMemoryOutboxRepository, InMemoryHITLRepository` works for unit tests
- ✅ 4 framework SQLModel rows: TenantRow, ThreadRow, MessageRow, OutboxRow + 3 HITL rows = 7 tables
- ✅ Alembic env.py + 0001 migration runs `upgrade head` cleanly on empty PG
- ✅ Each Repository has Protocol + InMemory + Postgres implementations
- ✅ `SqlAlchemyUnitOfWork` implements `UnitOfWork` Protocol; AsyncSession does not leak into Pattern signatures
- ✅ End-to-end smoke test (Task 14) exercises tenant → thread → messages → transactional outbox → HITL request/response in real PG via testcontainers
- ✅ All Repository methods require `tenant_id`; cross-tenant operations return None / raise KeyError
- ✅ Sub-project #1's 74 tests still pass; new tests bring total to ~110
- ✅ mypy strict clean, ruff clean

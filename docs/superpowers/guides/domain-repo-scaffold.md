# Domain repository scaffold

This guide walks through adding a new domain entity (e.g. `Task`) to a
`ballast-ai` app: domain object → `Repository(Protocol)` →
in-memory and Postgres implementations → wiring through the framework
`Container` so tools can resolve it without module-level singletons.

It mirrors the shape the framework itself uses for
`ballast.persistence.thread.repository.ThreadRepository` — read
that file alongside this guide for the canonical reference.

## 1. Domain object (frozen Pydantic)

Keep the in-process domain model separate from the SQL row. The domain
model is what tools and HTTP handlers traffic in; the row is an
implementation detail of the Postgres impl.

```python
# myapp/tasks/domain.py
from __future__ import annotations
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class Task(BaseModel):
    """Immutable domain object — copy-with-update to mutate."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    tenant_id: UUID
    title: str
    done: bool = False
    created_at: datetime
    updated_at: datetime
```

## 2. Repository protocol

Tenant-scope every method. `KeyError` for cross-tenant or unknown ids on
mutating ops; `delete` is idempotent (silent no-op on unknown ids).

```python
# myapp/tasks/repository.py
from __future__ import annotations
from typing import Protocol, runtime_checkable
from uuid import UUID
from myapp.tasks.domain import Task


@runtime_checkable
class TaskRepository(Protocol):
    async def create(self, *, title: str, tenant_id: UUID) -> Task: ...
    async def get(self, task_id: UUID, *, tenant_id: UUID) -> Task | None: ...
    async def list_(self, *, tenant_id: UUID, limit: int = 100) -> list[Task]: ...
    async def mark_done(self, task_id: UUID, *, tenant_id: UUID) -> Task: ...
    async def delete(self, task_id: UUID, *, tenant_id: UUID) -> None: ...
```

## 3. In-memory implementation

Mirror the Protocol method-for-method. Useful for unit tests and dev
servers without a database.

```python
# myapp/tasks/repository.py (continued)
from datetime import UTC, datetime
from uuid import uuid4


class InMemoryTaskRepository:
    def __init__(self) -> None:
        self._tasks: dict[UUID, Task] = {}

    @staticmethod
    def _now() -> datetime:
        return datetime.now(tz=UTC)

    async def create(self, *, title: str, tenant_id: UUID) -> Task:
        now = self._now()
        task = Task(
            id=uuid4(), tenant_id=tenant_id, title=title,
            created_at=now, updated_at=now,
        )
        self._tasks[task.id] = task
        return task

    async def get(self, task_id: UUID, *, tenant_id: UUID) -> Task | None:
        t = self._tasks.get(task_id)
        return t if t is not None and t.tenant_id == tenant_id else None

    async def list_(self, *, tenant_id: UUID, limit: int = 100) -> list[Task]:
        rows = [t for t in self._tasks.values() if t.tenant_id == tenant_id]
        rows.sort(key=lambda t: t.created_at, reverse=True)
        return rows[:limit]

    async def mark_done(self, task_id: UUID, *, tenant_id: UUID) -> Task:
        existing = self._tasks.get(task_id)
        if existing is None or existing.tenant_id != tenant_id:
            raise KeyError(f"task {task_id} not found for tenant {tenant_id}")
        updated = existing.model_copy(update={"done": True, "updated_at": self._now()})
        self._tasks[task_id] = updated
        return updated

    async def delete(self, task_id: UUID, *, tenant_id: UUID) -> None:
        existing = self._tasks.get(task_id)
        if existing is not None and existing.tenant_id == tenant_id:
            del self._tasks[task_id]
        # Idempotent: silent no-op on unknown / wrong-tenant ids.
```

## 4. Postgres implementation skeleton

The SQL row is a `SQLModel`. Use the framework's `SqlAlchemyUnitOfWork`
(bound by `PersistenceProvider`) to scope sessions per HTTP request.

```python
# myapp/tasks/row.py
from datetime import datetime
from uuid import UUID
from sqlmodel import Field, SQLModel


class TaskRow(SQLModel, table=True):
    __tablename__ = "tasks"
    id: UUID = Field(primary_key=True)
    tenant_id: UUID = Field(index=True)
    title: str
    done: bool = False
    created_at: datetime
    updated_at: datetime
```

```python
# myapp/tasks/postgres.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from myapp.tasks.domain import Task
from myapp.tasks.row import TaskRow


class PostgresTaskRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, task_id, *, tenant_id):
        stmt = select(TaskRow).where(
            TaskRow.id == task_id, TaskRow.tenant_id == tenant_id,
        )
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return None if row is None else Task.model_validate(row, from_attributes=True)

    # ... create / list_ / mark_done / delete mirror the in-memory impl.
```

Alembic migration goes here — copy
`src/ballast/alembic/versions/0001_framework_tables.py`
as the template and add a `tasks` table with `tenant_id` indexed.

## 5. Bind via the Container

Apps should never hold a module-level repo singleton. Bind the repo on
startup so it lives on `app.state.container`:

```python
# myapp/main.py
from fastapi import FastAPI
from ballast.runtime import Engine
from myapp.tasks.repository import InMemoryTaskRepository, TaskRepository


async def _bind_domain_repos(app: FastAPI) -> None:
    app.state.container.bind(TaskRepository, InMemoryTaskRepository())


engine = Engine(providers=[...])
app = engine.fastapi_app(
    extra_routers=[...],
    on_startup=[_bind_domain_repos],
)
```

For Postgres-backed bindings (per-request session scope), bind a factory
that resolves the active session from the request-scoped UoW instead of
a pre-built instance — see `PersistenceProvider` for the pattern.

## 6. Consume from tools

Tools should pull from the container via their deps — never import a
module-level repo.

```python
# myapp/tasks/tools.py
from dataclasses import dataclass
from uuid import UUID
from pydantic_ai import Agent, RunContext
from ballast.runtime.container import Container
from myapp.tasks.repository import TaskRepository


@dataclass
class TaskToolDeps:
    container: Container
    tenant_id: UUID


def register_task_tools(agent: Agent[TaskToolDeps, ...]) -> None:
    @agent.tool
    async def create_task(ctx: RunContext[TaskToolDeps], title: str) -> str:
        repo = ctx.deps.container.get(TaskRepository)
        task = await repo.create(title=title, tenant_id=ctx.deps.tenant_id)
        return f"created task {task.id}"
```

Build the `deps_factory` closure over the app so each request gets a
fresh `TaskToolDeps` with the resolved repo:

```python
def deps_factory(*, tenant_id: UUID, **_) -> TaskToolDeps:
    return TaskToolDeps(
        container=app.state.container,
        tenant_id=tenant_id,
    )
```

That's the whole pattern — domain object, Protocol, two impls, one bind
call, one `container.get` in the tool. No module-level globals; the
repo's lifetime is the app's lifetime, owned by the container.

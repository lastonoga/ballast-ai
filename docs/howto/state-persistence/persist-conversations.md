# How to wire Postgres for thread + message persistence

**Problem:** In-memory `ThreadRepository` is fine for dev but loses everything on restart. For production, you need a Postgres-backed thread/message store. Alembic migrations included.

**Solution:** Wire `SqlThreadRepository` via `Ballast.with_thread_repo(...)`. Run included Alembic migrations to create the schema. Apps that need different repos (Mongo, DynamoDB, etc.) implement the `ThreadRepository` Protocol.

## Minimum

### 1. Install + configure

`pyproject.toml`:
```toml
dependencies = [
    "ballast",
    "sqlmodel",          # already a transitive of ballast.persistence
    "asyncpg",           # async Postgres driver
]
```

`.env` (or your config):
```
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/myapp
```

### 2. Run migrations

The framework ships Alembic migrations under `src/ballast/alembic/versions/` for the base schema (tenants, threads, messages, outbox) + add-on tables (approval_cards).

```bash
# Use the framework's Alembic config
uv run alembic -c $(python -c "import ballast.alembic; import os; print(os.path.join(os.path.dirname(ballast.alembic.__file__), 'alembic.ini'))") upgrade head
```

Or copy `src/ballast/alembic/alembic.ini` into your repo and point `script_location` to `ballast.alembic`.

Migrations create:
- `tenants` (multi-tenant scoping)
- `threads` (id, agent name, metadata JSONB, timestamps)
- `messages` (thread_id, parent_id, role, content, timestamps)
- `outbox` (transactional outbox for downstream publishing)
- `approval_cards` (HITL persistence — see [audit-trail-of-approvals.md](../trust-and-safety/audit-trail-of-approvals.md))

### 3. Wire the repo

```python
from sqlalchemy.ext.asyncio import create_async_engine
from ballast import Ballast
from ballast.persistence import SqlThreadRepository


engine = create_async_engine(
    settings.database_url,
    pool_size=20,
    max_overflow=10,
)

ballast = (
    Ballast()
    .with_thread_repo(SqlThreadRepository(engine))
    .with_agents([NotesAgent(), TodoApprovalAgent()])
)

app = ballast.fastapi_app()
```

That's it. All thread/message reads/writes now go to Postgres.

## Customize the engine

Standard SQLAlchemy async engine — apply any options you need:

```python
engine = create_async_engine(
    settings.database_url,
    pool_size=20,
    max_overflow=10,
    pool_recycle=300,                  # recycle connections after 5min
    echo=settings.sql_echo,            # log all SQL for debugging
    connect_args={"command_timeout": 60},  # asyncpg-specific
)
```

## Per-request session vs shared engine

`SqlThreadRepository` opens a fresh session per call via `engine.begin()` context manager. Connection pooling is at the engine level. You DON'T need a per-request session middleware — the repo handles its own transactional scope.

If you want to share a session across multiple repo calls in one request (e.g. atomic across thread + approval card):

```python
from sqlalchemy.ext.asyncio import AsyncSession


async with AsyncSession(engine) as session:
    async with session.begin():
        await thread_repo._save_in_session(session, thread)   # private; see source
        await approval_repo._save_in_session(session, card)
        # both committed together
```

The repos expose private helpers for this case — see `src/ballast/persistence/thread/sql.py` and `src/ballast/patterns/hitl/sql_repo.py` for the exact methods.

## Multi-tenant scoping

The schema includes a `tenants` table; threads have `tenant_id`. To enforce per-tenant isolation:

```python
from ballast.auth.context import current_tenant_id

class TenantScopedThreadRepo(SqlThreadRepository):
    async def get(self, thread_id: str):
        tenant_id = current_tenant_id()
        async with self._engine.begin() as conn:
            row = await conn.execute(
                select(ThreadRow).where(
                    ThreadRow.id == thread_id,
                    ThreadRow.tenant_id == tenant_id,    # scope filter
                )
            )
            return row.scalar_one_or_none()
```

Or wrap `SqlThreadRepository` and apply scoping in `_apply_filters` hook.

## Migrations for your own tables

The framework owns `threads / messages / approval_cards / outbox / tenants`. Your domain tables (`notes`, `users`, `projects`) are YOUR migration responsibility:

```python
# alembic/versions/0001_add_notes.py
def upgrade():
    op.create_table(
        "notes",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("user_id", sa.String, nullable=False),
        sa.Column("title", sa.String, nullable=False),
        sa.Column("body", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_notes_user_id", "notes", ["user_id"])
```

Run your migrations independently of framework migrations.

## Custom repo implementations

`ThreadRepository` is a Protocol. Implement it for any backend:

```python
from ballast.persistence import ThreadRepository, Thread


class MongoThreadRepo(ThreadRepository):
    def __init__(self, client):
        self._db = client.myapp

    async def get(self, thread_id: str) -> Thread | None:
        doc = await self._db.threads.find_one({"_id": thread_id})
        return Thread.model_validate(doc) if doc else None

    async def save(self, thread: Thread) -> None:
        await self._db.threads.update_one(
            {"_id": thread.id},
            {"$set": thread.model_dump()},
            upsert=True,
        )

    # ... implement other methods per Protocol
```

Wire it: `ballast.with_thread_repo(MongoThreadRepo(mongo_client))`. Framework doesn't care about backend.

## Testing

```python
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from ballast.persistence import SqlThreadRepository


@pytest_asyncio.fixture
async def pg_thread_repo():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    # Apply migrations
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    repo = SqlThreadRepository(engine)
    yield repo
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_thread_roundtrip(pg_thread_repo) -> None:
    thread = Thread(id="t-1", agent="notes", metadata={})
    await pg_thread_repo.save(thread)
    loaded = await pg_thread_repo.get("t-1")
    assert loaded == thread
```

For unit tests of higher-level logic, use the in-memory variant:

```python
from ballast.persistence import InMemoryThreadRepository

repo = InMemoryThreadRepository()
# Same Protocol; no DB needed.
```

## Caveats

- **`asyncpg` only.** SqlThreadRepository uses async SQLAlchemy with asyncpg driver. Sync `psycopg2` won't work.
- **Don't import the SqlModel ORM models directly in app code.** They're internal. App-facing types are `Thread`, `Message` pydantic models.
- **Migrations are append-only.** Never edit a published migration. Add a new one.
- **`outbox` table writes are committed in the same transaction as thread/message writes.** This gives transactional consistency for downstream publishing. You don't need a separate `OutboxPublisher` for messages — the framework handles it via the outbox poller.

## Related

- [audit-trail-of-approvals.md](../trust-and-safety/audit-trail-of-approvals.md) — same pattern for HITL cards
- [swap-thread-repo-for-mongo.md](swap-thread-repo-for-mongo.md) — custom repo recipe (planned)
- Reference: `reference/persistence/thread-repository.md`
- Reference: `reference/persistence/sql-repositories.md`

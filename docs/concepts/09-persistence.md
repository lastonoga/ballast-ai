# 9. Persistence

**Prerequisites:** [08-running-an-app.md](08-running-an-app.md).

## Introduction

The default `Ballast()` app uses in-memory repositories for everything: threads, messages, approval cards. Restart the process, lose all the state. That's correct for development and tests — fast, no setup, no migrations. It is not correct for anything a user will rely on.

Production needs three things: data survives restarts, multiple processes share a consistent view, and the schema evolves under control. The framework's answer is a small set of repository Protocols (so apps can swap implementations) plus a PostgreSQL-backed default for each, plus Alembic migrations that own the framework's tables.

This chapter walks through the `ThreadRepository` and `ApprovalCardRepository` Protocols, the framework's Alembic migrations, how to run them, the tenant-scoping pattern via `current_user_id` ContextVar, and how to plug in your own repository when the SQL default doesn't fit.

## The mental model

Two things to keep separate:

- **Repositories are the interface.** Your code (and the framework's routers) only ever talks to a `ThreadRepository` or `ApprovalCardRepository` Protocol. Everything stays decoupled from the storage backend.
- **Storage backends are swappable.** The framework ships in-memory and Postgres implementations. You can swap to anything that satisfies the Protocol — Mongo, DynamoDB, SQLite, your existing legacy DB — without touching any code that consumes the repository.

The framework owns a few tables (`threads`, `messages`, `thread_events`, `approval_cards`) via Alembic migrations. Your app's own tables live alongside, in the same database, managed by your own migrations. The framework's migrations are namespaced by file prefix so they don't collide with yours.

## The `ThreadRepository` Protocol

The full surface, lifted from `ballast.persistence.thread.repository`:

```python
class ThreadRepository(Protocol):
    async def create(self, *, agent: str, metadata: dict | None = None) -> Thread: ...
    async def load(self, id: UUID) -> Thread | None: ...

    async def add_message(self, thread_id: UUID, *, role: str,
                          parts: list[dict], id: str | None = None,
                          silent: bool = False) -> Message: ...
    async def upsert_message(self, thread_id: UUID, *, id: str,
                             role: str, parts: list[dict],
                             silent: bool = False) -> Message: ...
    async def history(self, thread_id: UUID, *, limit: int = 1000) -> list[Message]: ...
    async def delete_messages(self, thread_id: UUID, *, ids: list[str]) -> None: ...

    async def list_(self, *, include_archived: bool = False,
                    limit: int = 100, offset: int = 0) -> list[Thread]: ...
    async def update_metadata(self, thread_id: UUID, *, metadata: dict) -> Thread: ...
    async def archive(self, thread_id: UUID) -> Thread: ...
    async def unarchive(self, thread_id: UUID) -> Thread: ...
    async def close(self, thread_id: UUID) -> Thread: ...
    async def delete(self, thread_id: UUID) -> None: ...
```

Small but complete. Two things worth knowing:

- **`add_message` is idempotent when `id` is supplied.** The frontend can post the same message twice (network retry) and you'll get one row, not two. This is what lets the assistant-ui frontend optimistically append before the server confirms.
- **`silent=True` skips event signaling.** Normally adding a message fires a thread event so SSE subscribers see it; pass `silent=True` for backfill / migration code where you don't want to wake up listeners.

## In-memory vs SQL

```python
# Default — used if you never call .with_thread_repo(...)
from ballast.persistence.thread import InMemoryThreadRepository
repo = InMemoryThreadRepository()

# Production
from ballast.persistence.thread import SqlThreadRepository
repo = SqlThreadRepository(session_factory=async_session_factory)

app = Ballast(settings).with_thread_repo(repo).fastapi(cors="dev")
```

`SqlThreadRepository` takes an `async_sessionmaker[AsyncSession]` — standard SQLAlchemy 2.x async session factory. Construct it however you normally would:

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

engine = create_async_engine(settings.dbos.database_url, pool_size=20, pool_pre_ping=True)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)
```

The framework's `SqlThreadRepository` opens a session per call, so connection pool sizing matters. For a typical 4-worker uvicorn deployment, `pool_size=20` is comfortable.

## The `ApprovalCardRepository` Protocol

```python
class ApprovalCardRepository(Protocol):
    async def add(self, card: ApprovalCard) -> None: ...
    async def get(self, card_id: str) -> ApprovalCard | None: ...
    async def list_pending(self, *, limit: int = 50) -> list[ApprovalCard]: ...
    async def resolve(self, card_id: str, *, verdict: BaseModel) -> ApprovalCard: ...
```

Smaller surface — approval cards don't need archival, metadata patching, or pagination beyond `limit`. The key method is `list_pending`: **it automatically filters by `current_user_id()`** if that ContextVar is set. So your HITL queue endpoint can be:

```python
from ballast.auth.context import current_user_id
from ballast.persistence.approval_card import get_approval_card_repo

@router.get("/my-cards")
async def my_cards(user: User = Depends(current_authenticated_user)):
    current_user_id.set(user.id)   # any subsequent repo call scopes to this user
    repo = get_approval_card_repo()
    return await repo.list_pending(limit=20)
```

The current user sees only their own pending cards. No app code needs to remember to filter — the repo does it.

## The framework's Alembic migrations

Two migrations ship with the framework:

- **`0001_framework_tables.py`** — creates `threads`, `messages`, `thread_events`. The thread_events table is the event log; it backs both the SSE multiplexer and the message-history reconstruction story.
- **`0002_approval_cards.py`** — creates `approval_cards`. Brought in when HITL was added.

They live at `src/ballast/alembic/versions/`. The migration config is at `src/ballast/alembic.ini`.

There are deliberately *no* `tenants` or `users` tables. The framework doesn't take a stance on your auth model — apps own their user/tenant tables and just reference them by ID in `metadata` dicts. This is the same trade-off Django REST framework makes vs. fully baking in `auth_user`: less convenient out of the box, much less assumption-violating in the long run.

## Running migrations

Three options, in increasing order of seriousness:

### Auto-migrate at startup (dev only)

```
BALLAST_AUTO_MIGRATE=true
```

The framework runs `alembic upgrade head` during FastAPI lifespan startup. Fine for local dev, terrible for production (race condition between replicas, no rollback story).

### Manual via CLI

```bash
alembic -c $(python -c 'import ballast; print(ballast.__path__[0])')/alembic.ini upgrade head
```

This is what you run in a one-off deploy step (or a CI hook before the new app version takes traffic).

### Wrap in your app's own migrations

You probably have your own `alembic/` directory for your app's tables. Add the framework's migrations as a dependency: in your `env.py`, register the framework's `MetaData` alongside yours, or run them in sequence:

```python
# In your release script
subprocess.run(["alembic", "-c", "ballast_alembic.ini", "upgrade", "head"], check=True)
subprocess.run(["alembic", "upgrade", "head"], check=True)   # your own
```

Either way: framework migrations run first; your app's migrations come after.

## Tenant scoping via ContextVar

The framework's pattern for multi-tenant data is the `current_user_id` ContextVar (chapter 4 covered ContextVars in the deps context):

```python
from ballast.auth.context import current_user_id

# In your auth middleware:
@app.middleware("http")
async def auth(request: Request, call_next):
    user = await authenticate(request)
    token = current_user_id.set(user.id)
    try:
        return await call_next(request)
    finally:
        current_user_id.reset(token)
```

Once set, any repository call that scopes by user reads it. `ApprovalCardRepository.list_pending` is the built-in example; your own resolvers and repos can do the same:

```python
class MyNotesRepository:
    async def list_(self) -> list[Note]:
        user_id = current_user_id()
        if user_id is None:
            return []
        async with self._session_factory() as session:
            stmt = select(Note).where(Note.owner_id == user_id)
            return list((await session.execute(stmt)).scalars())
```

This composes with FastAPI's request-scoped middleware, with DBOS workflows (the ContextVar propagates through `asyncio.create_task` correctly), and with pytest fixtures (set/reset in a `pytest.fixture(autouse=True)`).

## Outbox-style event log

The `thread_events` table is the framework's nearest thing to an outbox. Every time `add_message` (or any other thread-mutating call) runs, an event row is inserted in the same transaction as the data change. Subscribers — the SSE multiplexer, custom consumers — read from this table to project the state.

Two consequences:

- **No "I committed the data but the message never reached subscribers" race.** Either the whole thing commits or nothing does.
- **You can replay.** If a subscriber crashed and missed events, it can resume by reading from the last position it acknowledged.

For most apps the in-process event stream is fine. For multi-process deployments, swap to a Redis or NATS-backed event stream via `.with_events(...)` and the same outbox table feeds it.

## Implementing your own repository

Two scenarios.

### You already have a database / ORM

You have an existing Postgres database with your own models, and you want to keep threads there too. Implement the `ThreadRepository` Protocol using your existing session/ORM:

```python
class MyOrmThreadRepository:
    def __init__(self, session_factory):
        self._sf = session_factory

    async def create(self, *, agent: str, metadata=None) -> Thread:
        async with self._sf() as s:
            row = ThreadOrm(agent=agent, metadata=metadata or {})
            s.add(row)
            await s.commit()
            return Thread(id=row.id, agent=row.agent, metadata=row.metadata, ...)

    async def add_message(self, thread_id, *, role, parts, id=None, silent=False) -> Message:
        ...

    # ... and so on
```

Then: `Ballast(settings).with_thread_repo(MyOrmThreadRepository(...))`.

### Non-SQL store

The Protocol doesn't care about SQL. Implement `ThreadRepository` against Mongo, DynamoDB, Redis, whatever. The framework's routers will use it through the Protocol; nothing else changes.

The only constraint: it must be async. Synchronous repos would block the event loop. If your client library is sync-only, wrap calls in `asyncio.to_thread(...)`.

## When to use the SQL default

The decision tree:

- **In-memory:** dev, tests, throwaway demos. Anything that doesn't need to survive a process restart.
- **`SqlThreadRepository`:** anything serious. Postgres is the standard answer; the framework has been tested against Postgres 14+.
- **Custom Protocol implementation:** when you already have a different store and don't want to add Postgres just for the framework.

Don't try to "make in-memory work in production by persisting to a JSON file." The in-memory repo isn't crash-safe, isn't multi-process-safe, and doesn't have transactional semantics. Use the SQL one or build a real adapter.

## Common mistakes

- **Forgetting to apply migrations.** App boots, first request crashes with "relation 'threads' does not exist." Apply 0001 and 0002 before serving traffic.
- **Using a fresh in-memory repo per request.** `Ballast().with_thread_repo(InMemoryThreadRepository())` instantiates *one* repo at app build time. If you accidentally construct a new repo inside a route handler, each request gets a fresh empty repo — and your data "vanishes."
- **Not setting `current_user_id`.** Without it set, `list_pending` returns *all* pending cards across all users. In a multi-tenant app this is a privacy leak.
- **Mixing sync and async sessions.** SQLAlchemy session types are not interchangeable. The framework expects `AsyncSession` factories; passing a sync sessionmaker will fail at first call.

## What this chapter did NOT cover

- The `Message.parts` schema (it's pydantic-ai's `ModelMessage` part shape; see pydantic-ai docs for details).
- Multi-tenant *deep* scoping (`org_id` + `user_id`); the framework's ContextVar covers user but you'd need a parallel `current_tenant_id` ContextVar of your own.
- Schema migrations for app-level tables — that's your own Alembic config, not a framework concern.
- How approval-card persistence works end-to-end — chapter 21.

## Where to go next

→ [10-testing.md](10-testing.md) — testing agents and workflows without real LLM calls.

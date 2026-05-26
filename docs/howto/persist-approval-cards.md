# How to persist HITL approval cards in Postgres

**Problem:** `UICardChannel` writes approval cards to memory by default. Page refresh or process restart loses pending cards. For production HITL — where cards may sit waiting days for human action — you need persistent storage.

**Solution:** Wire `SqlApprovalCardRepository` via `Ballast.with_approval_repo(...)`. Alembic migration 0002 (shipped with framework) creates the `approval_cards` table.

## Minimum

### 1. Migration

The framework ships migration `0002_approval_cards.py` alongside the base migrations. If you ran `alembic upgrade head` per [wire-postgres-thread-repo.md](wire-postgres-thread-repo.md), this is already applied.

Table schema:
- `id` (UUID primary key)
- `kind` (string — `__hitl_kind__` from card payload class)
- `payload` (JSONB — serialized payload)
- `verdict` (JSONB — serialized CardVerdict; nullable until decided)
- `requested_at`, `decided_at` (timestamps)
- `workflow_id`, `respond_topic` (for DBOS-correlated resume)
- `user_id`, `tenant_id` (optional scoping)

### 2. Wire the repo

```python
from ballast import Ballast
from ballast.patterns.hitl import SqlApprovalCardRepository


ballast = (
    Ballast()
    .with_thread_repo(SqlThreadRepository(engine))
    .with_approval_repo(SqlApprovalCardRepository(engine))     # ← add this
    .with_agents([NotesAgent()])
)
```

Now every `UICardChannel.request(...)` persists the card; verdict resolution updates it; the framework's `/approvals` router queries from this repo.

### 3. Use as before

```python
from ballast.patterns.hitl import UICardChannel, register_card_kind, CardVerdict
from typing import ClassVar
from pydantic import BaseModel


@register_card_kind
class PublishCard(BaseModel):
    __hitl_kind__: ClassVar[str] = "publish-post"
    title: str
    body: str


class PublishVerdict(CardVerdict[PublishCard]):
    __hitl_kind__: ClassVar[str] = "publish-post"


channel = UICardChannel(payload_type=PublishCard)

# In a workflow / agent tool:
verdict = await channel.request(
    PublishCard(title="Hello", body="World"),
    timeout=timedelta(hours=4),
)
# Card persists immediately; if process restarts, the user can still find/decide it.
# Workflow resumes after verdict via Durable.recv_async.
```

No API change — only the persistence layer changed.

## Query pending cards

The framework's `/approvals` router exposes:
- `GET /approvals` — list (filter by kind / user / status)
- `GET /approvals/{id}` — single card
- `POST /approvals/{id}/decide` — submit verdict
- `GET /approvals/stream` — SSE multiplexer for real-time updates

Apps can also query directly:

```python
from ballast.patterns.hitl import approval_card_repo

pending = await approval_card_repo.list_pending(
    user_id="user-123",
    kind="publish-post",
    limit=50,
)
```

`approval_card_repo` is a module-level singleton bound to whatever repo you registered via `with_approval_repo`.

## Per-tenant / per-user scoping

The `approval_cards` schema includes `tenant_id` + `user_id` columns. `SqlApprovalCardRepository.create(...)` auto-populates them from `current_tenant_id()` + `current_user_id()` ContextVars.

Set the user context in your FastAPI dependency:

```python
from ballast.auth.context import acting_as


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    user_id = await extract_user_from_request(request)  # your auth logic
    with acting_as(user_id=user_id, tenant_id=user_id_to_tenant(user_id)):
        return await call_next(request)
```

Now `card.user_id == "user-123"` is set automatically. Repo's `list_pending(user_id=...)` filters server-side.

## Custom approval card kinds

Apps define their own card payloads (already covered in [auto-bridge-requires-approval.md](auto-bridge-requires-approval.md)):

```python
@register_card_kind
class TransferFundsCard(BaseModel):
    __hitl_kind__: ClassVar[str] = "transfer-funds"
    amount_usd: int
    recipient: str
    note: str


class TransferVerdict(CardVerdict[TransferFundsCard]):
    __hitl_kind__: ClassVar[str] = "transfer-funds"


# Wire its own channel + the framework finds it via __hitl_kind__ registry
transfer_channel = UICardChannel(payload_type=TransferFundsCard)
```

Each card kind gets its own row in `approval_cards.kind`. Filter dashboards by kind.

## Frontend integration

The included assistant-ui frontend in `examples/notes-app/frontend/` has:
- `useApprovals()` hook — subscribes to `/approvals/stream` SSE, renders pending cards
- `ApprovalsPanel` component — list + decide UI
- Per-kind card renderers — apps register their own per-payload UI (e.g. `note.create` renderer)

You can build your own UI calling the REST endpoints.

## When the channel ISN'T UICardChannel

`ThreadChannel` (in-chat marker) and `HelperAgent`-bound `ConversationalChannel` don't use `ApprovalCardRepository` directly — they have their own state model (verdicts arrive via DBOS topics + thread events). `with_approval_repo` only affects `UICardChannel`.

If you wire custom channels (Slack, email, Telegram), those will need their own persistence — or share `SqlApprovalCardRepository` by writing to the same table.

## Cleanup of stale cards

The framework doesn't auto-delete cards. Implement a cleanup job:

```python
async def cleanup_old_cards():
    cutoff = datetime.now(UTC) - timedelta(days=30)
    await approval_card_repo.delete_decided_before(cutoff)
```

Run on cron / DBOS scheduled workflow / cleanup endpoint. Pending cards should NOT be cleaned up — they're waiting on humans.

## Testing

```python
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from ballast.patterns.hitl import SqlApprovalCardRepository


@pytest_asyncio.fixture
async def pg_approval_repo():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    repo = SqlApprovalCardRepository(engine)
    yield repo
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_card_roundtrip(pg_approval_repo) -> None:
    payload = PublishCard(title="T", body="B")
    card = await pg_approval_repo.create(payload=payload, ...)
    loaded = await pg_approval_repo.get(card.id)
    assert loaded == card
```

For unit tests, use `InMemoryApprovalCardRepository` (same Protocol).

## Caveats

- **JSON-Schema-only persistence.** Payloads stored as JSONB. Don't include unserializable types in cards (use `Ref[T]` instead of actual entity instances).
- **Idempotency.** `create()` is not idempotent — calling twice creates two cards. Wrap in DBOS step if you need replay-safety.
- **Card deletion ≠ verdict update.** Once a card is decided, the verdict is recorded but the card row stays. Run cleanup if disk fills up.

## Related

- [auto-bridge-requires-approval.md](auto-bridge-requires-approval.md) — automated tool-call → card flow via ApprovalCapability
- [wire-postgres-thread-repo.md](wire-postgres-thread-repo.md) — thread/message persistence (same pattern)
- Reference: `reference/persistence/approval-card-repository.md`
- Reference: `reference/hitl/ui-card-channel.md`

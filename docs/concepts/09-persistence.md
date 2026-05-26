# 9. Persistence

**Prerequisites:** [08-running-an-app.md](08-running-an-app.md).

**What you'll learn:** how the in-memory `ThreadRepository` defaults work in development; how to switch to Postgres for production; what the framework's Alembic migrations create; how to wire your own repository implementation.

## Sections

1. The default `InMemoryThreadRepository` and when it's enough
2. `SqlThreadRepository` for production; SQLModel + asyncpg
3. The framework's Alembic migrations: `tenants`, `threads`, `messages`, `outbox`, `approval_cards`
4. Running migrations against your database
5. Multi-tenant scoping via `current_tenant_id` ContextVar
6. The outbox table: transactional consistency for downstream publishers
7. Implementing a custom `ThreadRepository` (Mongo / DynamoDB / etc.)
8. Approval card persistence: `SqlApprovalCardRepository` + migration 0002
9. Per-user / per-tenant filtering in repository methods
10. Cleanup of stale state
11. Where to go next

## Next

[10-testing.md](10-testing.md) — testing agents and workflows without real LLM calls.

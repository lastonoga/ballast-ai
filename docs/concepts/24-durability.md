# 24. Durability — DBOS in depth

**Prerequisites:** [08-running-an-app.md](08-running-an-app.md), [21-human-in-the-loop.md](21-human-in-the-loop.md).

**What you'll learn:** what DBOS provides under the hood; the `Durable` facade re-exports; how step memoisation makes replay safe; how `recv_async` and `send_async` enable durable HITL waits; the `DBOSConfiguredInstance` pattern for stateful patterns like `MapReduce` and `as_workflow`-wrapped CoALA units.

## Sections

1. What "durable" means: workflow state is persisted; crashes resume from the last step
2. The `Durable` facade: `workflow`, `step`, `dbos_class`, `current_workflow_id`, `recv_async`, `send_async`
3. `@Durable.workflow()` lifecycle and replay semantics
4. `@Durable.step()` memoisation: same args → cached result
5. `DBOSConfiguredInstance` for stateful patterns (why MapReduce uses it)
6. The HITL pattern: `recv_async(topic)` suspends durably
7. Workflow IDs, idempotency, and the outbox table
8. SQLite for development, Postgres for production
9. Workflow cancellation: `cancel_thread_workflows`
10. Inspecting workflows: DBOS inspector tree view
11. When to use `@Durable.workflow` vs plain async functions
12. Where to go next

## Next

[25-custom-extensions.md](25-custom-extensions.md) — extending the framework.

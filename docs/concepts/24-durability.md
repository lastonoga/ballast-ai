# 24. Durability — DBOS in depth

**Prerequisites:** [08-running-an-app.md](08-running-an-app.md), [21-human-in-the-loop.md](21-human-in-the-loop.md).

## Introduction

You've been using `@Durable.workflow` and `@Durable.step` throughout the previous chapters without ever needing to look under the hood. That's the right state most of the time — the patterns and HITL channels handle the wiring. But there are moments where you'll want to know exactly what DBOS does: when you write your own workflow, when you need to cancel a long-running flow, when you're debugging why a step isn't memoising, or when you need to wait for an external signal.

This chapter is the under-the-hood tour. It covers what "durable" means in the DBOS sense, the `Durable` facade and every attribute it exposes, the replay semantics for `@Durable.workflow` and `@Durable.step`, the `DBOSConfiguredInstance` pattern that makes stateful patterns work, the `recv_async` / `send_async` primitives that power HITL, and the control-plane API for inspecting and managing live workflows.

## The mental model

DBOS persists workflow state to a database. Every workflow call gets an ID and a row; every step inside a workflow is logged with its inputs and outputs as it runs. On a crash, when the process comes back up, DBOS scans for in-flight workflows, finds the last completed step, and resumes the workflow from there.

What "durable" gives you in practice:

- **Crash mid-step:** the step runs again from the start (steps must be idempotent or at least safe to repeat).
- **Crash between steps:** the workflow resumes from after the last completed step.
- **Multi-day waits:** a workflow can suspend on `recv_async` for hours/days; the suspension is durable.
- **Cancellation:** any in-flight workflow can be cancelled by ID from any process.
- **Inspection:** any workflow's history (which steps ran, what they returned) is queryable.

The cost is a few milliseconds per step (DB write) and a few extra DB tables. The benefit is "your agent that takes 30 minutes can survive a deploy in the middle."

## The `Durable` facade

The framework wraps DBOS in a `Durable` facade so you don't import DBOS directly. Two reasons: the facade adds OTel context propagation automatically (so trace spans stitch across workflow boundaries), and it keeps the framework's public API stable if DBOS evolves.

```python
from ballast import Durable
```

What's on it:

### Worker-side decorators

- **`Durable.workflow(**kwargs)`** — wraps `@DBOS.workflow()` and attaches the OTel carrier.
- **`Durable.step(**kwargs)`** — wraps `@DBOS.step()` and attaches the OTel carrier.
- **`Durable.dbos_class(**kwargs)`** — pass-through to `@DBOS.dbos_class()`; used for stateful pattern classes.

### Caller-side helpers

- **`Durable.enqueue(queue, fn, *args, **kwargs)`** — queue a workflow for execution; injects OTel carrier.
- **`Durable.start_workflow(fn, *args, **kwargs)`** — launch a workflow without waiting; returns the workflow ID; injects OTel carrier.

### Inter-workflow messaging

- **`Durable.send_async(destination_id, message, topic=None)`** — send a message to a workflow.
- **`Durable.recv_async(topic=None, timeout_seconds=None)`** — receive a message inside a workflow; durable suspension.
- Synchronous variants exist (`Durable.send`, `Durable.recv`) but you almost never use them.

### Control plane

- **`Durable.list_workflows(**kwargs)`** — query live + completed workflows.
- **`Durable.list_workflow_steps(workflow_id, **kwargs)`** — query steps inside a workflow.
- **`Durable.cancel_workflow(workflow_id)`** — cancel an in-flight workflow.
- **`Durable.resume_workflow(workflow_id)`** — resume a previously suspended/cancelled workflow.
- **`Durable.fork_workflow(workflow_id, start_step, **kwargs)`** — replay from a specific step (debugging).
- **`Durable.retrieve_workflow(workflow_id)`** — get a handle to inspect a running workflow.

### Context / lifecycle

- **`Durable.current_workflow_id() -> str`** — get the active workflow's ID inside a workflow.
- **`Durable.init(config)`** — initialize DBOS (you usually don't call this; `Ballast.with_dbos()` does).
- **`Durable.launch()`** / **`Durable.destroy(destroy_registry=False)`** — lifecycle (called by FastAPI lifespan).
- **`Durable.is_launched() -> bool`** — check if DBOS is up.

## `@Durable.workflow` — replay semantics

```python
@Durable.workflow
async def my_workflow(input: dict) -> dict:
    a = await step_one(input)
    b = await step_two(a)
    c = await step_three(b)
    return {"result": c}
```

What happens on a normal run:

1. `my_workflow(input)` is called; DBOS assigns a workflow ID and persists "workflow X started with input I."
2. Each `await step_X(...)` runs; DBOS persists "step Y completed with output Z."
3. Workflow returns; DBOS marks it complete.

What happens on a crash between step_two and step_three:

1. Process restarts; FastAPI lifespan calls `Durable.launch()`.
2. DBOS scans for in-flight workflows; finds workflow X mid-execution.
3. DBOS calls `my_workflow(input)` again with the same workflow ID.
4. First `await step_one(input)` — DBOS sees this step already completed; returns the cached `a` immediately (no re-execution).
5. Second `await step_two(a)` — also cached; returns `b` immediately.
6. Third `await step_three(b)` — not yet completed; runs for real.
7. Workflow returns; DBOS marks complete.

The replay is *deterministic from the inputs* because completed steps return their cached outputs. The whole pattern hinges on: **steps must be idempotent or at least safe to repeat** — a crashed step might run again. Inside a step, that's true; between steps, it isn't.

## `@Durable.step` — memoisation

```python
@Durable.step
async def step_one(input: dict) -> Any:
    return await external_api.call(input)
```

DBOS memoises steps by (workflow_id, step_name, arguments). The first time a workflow calls `step_one(input)`, it runs and the result is cached. If the workflow crashes and resumes, the second call to `step_one(input)` with the same arguments returns the cached result — no re-execution.

Two consequences:

- **Side effects in steps run at most once per workflow.** Send an email in a step, crash, resume — the email isn't re-sent.
- **Non-determinism inside steps is fine.** A step that calls a flaky external API and retries internally is one logical operation; from the workflow's perspective, it returned exactly once.

The contract: steps wrap individual logical operations. Workflows orchestrate steps.

## `DBOSConfiguredInstance` — stateful patterns

When a pattern instance has state (`MapReduce` has its concurrency semaphore, `Reflection` has its critic/writer references), DBOS needs to know which instance is which on replay. Otherwise, `_map_one`'s cache could mix outputs from different `MapReduce` instances.

The solution: inherit from `DBOSConfiguredInstance` and pass a `config_name`:

```python
@Durable.dbos_class()
class MapReduce(DBOSConfiguredInstance, Generic[InT, OutT]):
    def __init__(self, *, map_step, reduce_step, ..., config_name=None):
        super().__init__(config_name=config_name or "default")
        ...
```

DBOS uses the `config_name` to namespace the per-instance state. Two `MapReduce` instances with different `config_name` get separate caches.

The shipped patterns (`Reflection`, `MapReduce`, `DivergentConvergent`, `PlanAndExecute`, the CoALA `as_workflow` adapter) all use this pattern correctly. When writing your own pattern, follow the same convention.

## `recv_async` and `send_async` — durable HITL

The HITL channels (chapter 21) work because of `recv_async`:

```python
@Durable.workflow
async def my_workflow(...):
    request_id = uuid4().hex
    topic = f"hitl:{request_id}"

    # 1. Deliver the request (persist a card, post to Slack, etc.)
    await deliver_request(workflow_id=Durable.current_workflow_id(), respond_topic=topic)

    # 2. Suspend until verdict arrives
    raw_verdict = await Durable.recv_async(topic=topic, timeout_seconds=3600)

    # 3. Continue with the verdict
    return process_verdict(raw_verdict)
```

`recv_async` is durable: when the workflow suspends, DBOS persists "workflow X waiting on topic T." A crash during the suspension → on restart, DBOS sees the workflow is waiting; it stays waiting. When the verdict arrives via `send_async`, DBOS wakes the workflow.

```python
# From the verdict-receiving endpoint:
await Durable.send_async(destination_id=workflow_id, message=verdict_dict, topic=topic)
```

The sender doesn't need to know whether the recipient workflow is in-process or on another replica or even running. DBOS handles the delivery.

## Workflow IDs and idempotency

Every workflow has an ID. By default DBOS generates a UUID; you can pass your own via the `idempotency_key` decorator argument:

```python
@Durable.workflow
async def process_payment(payment_id: str, amount: int) -> dict:
    ...

# Pass idempotency_key based on payment_id so two calls with the same ID
# resolve to the same workflow run.
```

Idempotent workflow IDs are how you make "user clicked submit twice" not result in two charges — both clicks resolve to the same workflow ID; the second one finds the workflow already running (or completed) and returns the existing result.

## SQLite for dev, Postgres for prod

DBOS supports both backends. For dev:

```
BALLAST_DBOS__DATABASE_URL=sqlite:///./dbos.db
```

For prod:

```
BALLAST_DBOS__DATABASE_URL=postgresql+psycopg://user:pw@host/db
```

The shipped patterns work identically against both. SQLite has limitations (no concurrent writes, no multi-process) so it's only suitable for dev and tests.

The testing fixtures (chapter 10) use in-memory SQLite for fast workflow tests; production uses Postgres for concurrency and durability.

## Workflow cancellation

```python
await Durable.cancel_workflow(workflow_id)
```

Cancels an in-flight workflow. The next time the workflow's coroutine wakes up (between steps, on a `recv_async` timeout), it gets a `WorkflowCancelled` exception. Use this for user-initiated stop ("user clicked Cancel"):

```python
@router.post("/threads/{thread_id}/cancel")
async def cancel_thread(thread_id: str):
    workflow_ids = await find_active_workflows_for_thread(thread_id)
    for wid in workflow_ids:
        await Durable.cancel_workflow(wid)
    return {"status": "cancelled"}
```

The framework also provides `cancel_thread_workflows(thread_id)` that wraps this for thread-scoped cancellation.

## Inspecting workflows: the `/dbos` route

The framework's `/dbos` router (mounted by `Ballast(...).fastapi(...)`) exposes a tree view of in-flight and completed workflows. Useful for debugging:

- Which workflows are stuck on `recv_async`?
- Which steps ran how many times?
- What were the inputs/outputs of each step?

The view reads from `Durable.list_workflows()` and `Durable.list_workflow_steps()`. You can build your own admin UI on top of these primitives if `/dbos` doesn't fit your needs.

## When to use `@Durable.workflow` vs plain async functions

Use `@Durable.workflow` when:

- The function takes more than a few seconds (so a crash-during-execution is meaningfully bad)
- The function has side effects you don't want to re-trigger on retry
- The function waits for external signals (HITL, scheduled events)
- The function calls multiple LLM/external services and you want per-step memoisation

Don't use it for:

- Pure functions (no side effects, fast) — overkill
- Functions that need to *fail* atomically (the partial-step replay model can be wrong here)
- Functions called inside an outer workflow (use `@Durable.step` instead)

The rule of thumb: workflows are *orchestrators*; steps are *operations*; plain async functions are *helpers*. Plenty of code stays in the "plain async function" category.

## Common mistakes

- **Side effects directly in the workflow body (not in a step).** The workflow body re-runs from the top on replay; side effects there happen multiple times. Always wrap side effects in `@Durable.step`.
- **Non-deterministic decisions in the workflow body.** `if random() < 0.5:` in a workflow body might go one way on first run and the other way on replay. Use steps for non-deterministic operations and cache the decision.
- **Forgetting `config_name` on parallel stateful patterns.** Two `MapReduce` instances without unique `config_name` corrupt each other's step cache.
- **Calling `recv_async` outside a workflow.** It only works inside a `@Durable.workflow`. Use a regular asyncio queue elsewhere.
- **Not handling `WorkflowCancelled` in long-running workflows.** Best practice: catch it, do any cleanup (close handles, mark thread "cancelled"), then re-raise.
- **Treating the workflow body like a regular async function.** It isn't — it can run multiple times. Write defensively: assume any line could be re-executed on the next replay, so guard side effects via steps.

## What this chapter did NOT cover

- DBOS internals (how the workflow log table is structured) — see DBOS's documentation.
- Multi-region Postgres setups — out of scope; depends on your infrastructure.
- The exact behavior of the `/dbos` inspector UI — run it and see.
- Performance tuning DBOS for very high workflow throughput — talk to the DBOS folks.

## Where to go next

→ [25-custom-extensions.md](25-custom-extensions.md) — extending the framework's surfaces.

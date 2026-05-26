# Architecture Overview

> **TL;DR:** Ballast is a thin opinionated layer on three industrial-grade libraries (`pydantic-ai`, `DBOS`, `FastAPI`). It doesn't reinvent any of them. It wires them into a coherent composition surface for agentic apps.

## Layered stack

```
┌─────────────────────────────────────────────────────────────────┐
│  APPLICATION (e.g. examples/notes-app)                           │
│  ─ domain models, agents, tools, workflows, UI                  │
└─────────────────────────────────────────────────────────────────┘
                              ▼ depends on
┌─────────────────────────────────────────────────────────────────┐
│  BALLAST FRAMEWORK                                               │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Patterns                                                  │    │
│  │  Reflection · MapReduce · DivergentConvergent ·          │    │
│  │  PlanAndExecute · HITLGate · MutationPipeline             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Capabilities (BallastCapability)                          │    │
│  │  BudgetGuard · SemanticLoopDetector · TypedLoopGuard ·    │    │
│  │  PIIGuard · GroundedRetry · GoalDriftDetector ·           │    │
│  │  ApprovalCapability · JudgeAfterRun · CB.as_capability    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Resilience                                                │    │
│  │  CircuitBreaker (+ as_step / as_workflow_decorator /      │    │
│  │  as_capability) · ThresholdPolicy · FallbackPolicy        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Cognitive (CoALA)                                         │    │
│  │  CoALAUnit Protocol · CoALABase · adapters                │    │
│  │  (as_workflow / as_tool / as_capability)                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Quality                                                   │    │
│  │  Scored[T] · Confidence labels · aggregate/filter/rank    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Grounded                                                  │    │
│  │  Ref[T] · Selector · scan_output · GroundedAgent          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ HITL                                                      │    │
│  │  HITLChannel · UICardChannel · ThreadChannel ·            │    │
│  │  ApprovalCard · CardVerdict · HelperAgent                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Runtime / Persistence / API                                │    │
│  │  Ballast · Engine · Durable facade · ThreadRepo ·         │    │
│  │  FastAPI router (streaming SSE / A2A / approvals)         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ▼ depends on
┌─────────────────────────────────────────────────────────────────┐
│  THIRD-PARTY STACK                                                │
│                                                                   │
│  pydantic-ai        Agent loop, structured outputs, tool calls,  │
│                     capability hooks, model providers             │
│                                                                   │
│  DBOS               Durable workflows, replay-safe @step,        │
│                     signals, recv_async, dbos_class               │
│                                                                   │
│  FastAPI            HTTP routes, SSE streaming, dependency-inj    │
│                                                                   │
│  pydantic v2        Data contracts, validators, JSON Schema       │
│                                                                   │
│  SQLModel + Alembic Postgres persistence (optional)               │
│                                                                   │
│  logfire / OTel     Tracing, spans, attributes                    │
└─────────────────────────────────────────────────────────────────┘
```

## Layer responsibilities

### Patterns (workflow-level compositions)

Pre-built, durable, replay-safe workflow shapes. Each `@Durable.workflow` or `DBOSConfiguredInstance`. Apps **compose** patterns; they don't subclass them. Examples:
- **Reflection** — writer-critic-refiner loop with iteration cap
- **MapReduce** — sharded extraction + global reduce (handles "Lost in the Middle")
- **DivergentConvergent** — divergent exploration + convergent synthesis (CREATIVEDC; anti-hivemind)
- **PlanAndExecute** — planner emits typed DAG, executor dispatches per `Step` Protocol

### Capabilities (agent-level hooks)

`BallastCapability` subclasses with `for_run` / `before_model_request` / `after_model_request` / `after_run` / `wrap_run` hooks. They modify or observe a single `agent.run()`. Stateless (or per-run via `for_run`). Stack any combination:

```python
agent = Agent(model=..., capabilities=[
    BudgetGuard(...),
    SemanticLoopDetector(...),
    GoalDriftDetector(...),
    ApprovalCapability(...),
])
```

### Resilience (cross-cutting reliability)

`CircuitBreaker` + future fellows (`Retry`, `Bulkhead`, `RateLimiter`). Lives at `ballast.resilience`. Single primitive with **three adapters** (`as_capability`, `as_workflow_decorator`, `as_step`) so the same breaker can protect any layer.

### Cognitive (CoALA)

Single contract `CoALAUnit` for memory-aware computation (observe / retrieve / act / learn). Three adapters wrap any unit into the matching runtime surface. Apps write one class; framework deploys it anywhere.

### Quality (typed return wrappers)

`Scored[T, ConfidenceT]` — generic wrapper carrying `value + rationale + confidence`. Composes with patterns (`MapReduce`'s reduce filters by confidence) and resilience (`CircuitBreaker.is_success` predicate).

### Grounded (typed entity references)

`Ref[T]` is a typed UUID-wrapper that hydrates against an app-supplied repository. `scan_output` narrows JSON Schema to `Literal` enums of valid IDs — anti-hallucination for entity references. `GroundedAgent` glues all of this on top of plain `pydantic-ai.Agent`.

### HITL (human-in-the-loop)

`HITLChannel` is a Protocol with one method (`request`). Built-in implementations: `UICardChannel` (UI panel via `/approvals/*` REST + SSE), `ThreadChannel` (in-conversation prompt). All built on `DBOSHITLChannel` ABC which uses `Durable.recv_async` for crash-safe waiting.

### Runtime / Persistence / API

The plumbing: `Ballast` app builder (fluent setters), `Engine` (DI + lifecycle), `Durable` facade (clean re-exports of DBOS workflow/step decorators), `ThreadRepository` (Postgres or InMemory), FastAPI router factory (`build_streaming_router`, `/approvals` router, `/a2a` endpoints).

## Compositional flow (typical agent step)

For a single `agent.run()` invocation, the lifecycle through Ballast's layers:

```
1. App calls agent.run(input)
   ▼
2. pydantic-ai's Agent dispatches `before_model_request` to each capability
   (BudgetGuard checks remaining budget, GoalDriftDetector starts clock,
    ApprovalCapability is a no-op here)
   ▼
3. LLM call — pydantic-ai handles model API + structured output validation
   ▼
4. pydantic-ai dispatches `after_model_request` to each capability
   (counters advance, drift judge may fire, semantic-loop detector adds embedding)
   ▼
5. If tools are called: pydantic-ai routes; if requires_approval=True:
   ApprovalCapability.wrap_run intercepts DeferredToolRequests,
   opens UICardChannel cards, awaits verdicts (durable via Durable.recv_async),
   maps to ToolApproved/Denied, re-runs agent.
   ▼
6. Loop returns when model emits final output_type-validated payload.
   ▼
7. `after_run` hooks fire (JudgeAfterRun grades the output, persists).
   ▼
8. Final AgentRunResult returned.
```

For workflow-level flows (`@Durable.workflow`), the same agent.run() happens inside a DBOS workflow, so:
- The workflow has a stable `workflow_id` for tracing
- Each `@Durable.step` inside is memoised — crash + replay skips completed work
- `Durable.recv_async` calls (HITL waits) are crash-safe — workflow resumes from waiting state when verdict arrives

## App entry point

```python
from ballast import Ballast, get_ballast

ballast = (
    Ballast()
    .with_thread_repo(my_thread_repo)
    .with_approval_repo(my_approval_repo)
    .with_agents([NotesAgent(), TodoApprovalAgent()])
    .with_capabilities([BudgetGuard(...), SemanticLoopDetector(...)])
)
app = ballast.fastapi_app()    # FastAPI app with streaming + HITL + A2A routes
```

That's it. The `Ballast` builder is a fluent surface; everything else (CB, Drift, CoALA, Scored, Patterns) is composed inside agents/workflows by the app code.

## Persistence (optional)

In-memory by default for dev. For production:

```python
from ballast.persistence import SqlThreadRepository, SqlApprovalCardRepository

ballast = (
    Ballast()
    .with_thread_repo(SqlThreadRepository(engine))
    .with_approval_repo(SqlApprovalCardRepository(engine))
    ...
)
```

Alembic migrations under `src/ballast/alembic/versions/` set up the schema. Run via standard Alembic CLI.

## Observability

`logfire.configure()` + `traced` decorator wrap everything. DBOS workflows produce per-step spans automatically. App-level OTel attributes propagate through `pattern.run` calls and capability hooks.

## See also

- [why-ballast.md](why-ballast.md) — mission + production-pain motivation
- [article-pain-points.md](article-pain-points.md) — concrete mapping article → solution
- [customization-everywhere.md](customization-everywhere.md) — Protocol-first design rationale

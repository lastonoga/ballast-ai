# SP4: Drop framework agent/workflow concepts — Implementation Plan

**Goal:** Framework stops knowing about "agent" and "workflow" as concepts. Drop `@sf.workflow`, `@sf.stateflow_agent`, workflow registry, agent class registry, auto-HTTP routes for workflows, `app.state.workflows/agents`. Add `Infra` + `RunContext` for cross-cutting; apps write their own routes via `sf.stream_response` primitive.

**Architectural shift:** Framework provides primitives (`Durable`, `Infra`, `RunContext`, patterns, persistence, event broadcaster, `stream_response`, threads CRUD router). Apps compose into agents/workflows + write FastAPI routes explicitly.

**Confirmed decisions (from brainstorm):**
- Drop `@sf.workflow` + `runtime/workflows.py` + `api/workflow_router.py` entirely
- Drop `@sf.stateflow_agent` + class registry
- Keep `StateflowAgent` ABC as **convenience** (tool/system-prompt helpers; no framework registry)
- Keep `StateflowDurableAgent` shape — abstraction "DBOS workflow over pydantic-ai Agent run loop"
- Keep `DurableHITLWorkflow` as pattern
- `RunContext` delivered as **explicit first arg** to flow/agent methods (`run(self, ctx, input)`)
- `Infra` is frozen dataclass holding repos+stream; `broadcaster` is `cached_property`; `infra.context(**per_call) -> RunContext`
- `sf.stream_response(agent, thread, body, ctx)` — primitive accepting any object with `.stream(...)` protocol
- `Thread.agent` is opaque app-owned string; framework just stores it
- `StateflowDurableAgent.__init__` loses `thread_repo / event_log / event_stream` params; only `config_name`

---

## Task list

- **T1**: New `runtime/infra.py` — `Infra` dataclass + `RunContext` + `Infra.context()` + tests
- **T2**: Drop `runtime/workflows.py`, `api/workflow_router.py`, related tests; drop decorator + class registry from `runtime/agents.py`; drop registry-based Depends helpers from `api/deps.py`
- **T3**: Migrate `StateflowDurableAgent` / `StateflowAgent` ABCs — drop infra params from `__init__`; methods take `ctx: RunContext`; framework calls use `ctx.thread_repo` / `ctx.event_log` / `ctx.event_stream` / `ctx.broadcaster`
- **T4**: Migrate `DurableHITLWorkflow` to ctx-based methods (`open(ctx, ...)`, `on_decision(ctx, ...)`)
- **T5**: Replace framework streaming router with `sf.stream_response(agent, thread, body, ctx)` primitive; drop module-level `streaming_router`; cancel endpoint as primitive too
- **T6**: Update `runtime/app.py:create_app` — drop `workflows=` / `agents=` / `extra_routers=` workflow-router wiring; take `infra=` param; populate `app.state.infra`
- **T7**: Update top-level `__init__.py` — drop removed exports; add `Infra`, `RunContext`, `stream_response`
- **T8**: Migrate notes-app — flow/agent constructors lose infra args; `main.py` writes own brainstorm route + streaming route; `brainstorm_flow.py` drops `@sf.workflow`; `agent.py` / `todo_approval_agent.py` drop `@sf.stateflow_agent`
- **T9**: Simplify `TestEngine` — drop workflow/agent override semantics; tests use `app.dependency_overrides` for repos, direct construction for flows/agents
- **T10**: Update test_smoke.py + framework tests; final sweep

After all tasks: framework + notes-app green.

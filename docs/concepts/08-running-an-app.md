# 8. Running an app — the `Ballast()` builder

**Prerequisites:** [04-dependencies-and-state.md](04-dependencies-and-state.md), [07-capabilities.md](07-capabilities.md).

## Introduction

So far you've built agents in isolation: declare an `Agent`, attach tools, set `output_type`, stack capabilities. Run it from a script with `await agent.run(...)` and the model talks back. That works for a demo. It doesn't work for a product.

A real product needs: an HTTP entrypoint that browsers and clients can hit; a streaming protocol so the UI shows tokens as they arrive; thread persistence so a user's conversation survives a page refresh; the HITL approval surface so `requires_approval=True` tools surface as cards; durable workflows so a crash mid-tool-call doesn't lose work; observability so you know what happened; and the lifecycle plumbing to start it all and shut it down cleanly.

`Ballast()` is the builder that wires these together. You hand it the pieces — repositories, providers, lifecycle hooks — and it returns a `FastAPI` app you can serve with `uvicorn`. Everything you've learned about agents, tools, capabilities, and outputs slots in unchanged; this chapter just covers how to *deploy* them.

## The mental model

Think of `Ballast` as a *fluent configuration object*. Each `.with_X(...)` call returns `self`, so you chain. When you finally call `.fastapi(...)`, it:

1. Materializes an `Engine` (the framework's DI container) from everything you registered.
2. Installs that engine as a process-wide singleton.
3. Builds a `FastAPI` app, mounts the built-in routers (`/threads`, `/dbos`, `/approvals`, `/health`), runs your startup hooks, and hands the app back.

The engine is the glue. Routers use it to find the thread repo, the approval repo, the event stream. Your own routes can grab it off `app.state.engine` if you need to.

Everything else is your code — your own routers for chat streaming, your own agents constructed however you like. The builder doesn't try to own your agents; it owns the *cross-cutting infrastructure* (persistence, approvals, lifecycle, observability) so your routes can focus on the agent logic.

## The minimal app

```python
from ballast import Ballast, BallastSettings

settings = BallastSettings()  # reads BALLAST_* env vars

app = (
    Ballast(settings)
    .with_dbos()
    .fastapi(cors="dev")
)
```

That's a runnable FastAPI app. No agents yet — but it has `/health`, `/threads`, `/dbos`, `/approvals` mounted, an in-memory thread repository, DBOS lifecycle wired, and CORS open for local development. Serve with `uvicorn app:app --reload` and you can hit it.

To actually expose an agent, mount your own router on top:

```python
from fastapi import APIRouter
from pydantic_ai import Agent
from ballast import Ballast, BallastSettings

settings = BallastSettings()

notes_agent = Agent(model="openai:gpt-4o-mini", system_prompt="You take notes.")

router = APIRouter()

@router.post("/notes/{thread_id}")
async def notes_endpoint(thread_id: str, body: dict):
    result = await notes_agent.run(body["text"])
    return {"output": result.output}

app = (
    Ballast(settings)
    .with_dbos()
    .fastapi(cors="dev", routers=[router])
)
```

The router is passed to `.fastapi(routers=[...])` and gets mounted *after* the built-in routers but *before* CORS middleware wraps everything.

## Step-by-step: the fluent setters

Each setter is independent. Call the ones you need.

### `.with_dbos(config: DBOSConfig | None = None)`

Wires DBOS lifecycle into the app's startup/shutdown hooks. Without this, `@Durable.workflow` decorated functions won't run. Omit `config` and it uses `settings.dbos.database_url`. For local dev with SQLite, set `BALLAST_DBOS__DATABASE_URL=sqlite:///./dbos.db`.

### `.with_thread_repo(repo)`

Installs a custom `ThreadRepository`. Default is `InMemoryThreadRepository`. For production, pass `SqlThreadRepository(session_factory=...)`. Chapter 9 covers the repo shape.

### `.with_approval_repo(repo)`

Installs the approval-card repository. Default is in-memory. Production uses `SqlApprovalCardRepository`. Chapter 21 covers HITL end-to-end.

### `.with_events(event_log, event_stream)`

Custom event-log + event-stream pair. The default is in-process; for multi-process deployments you'd swap to Redis-backed. The `event_stream` is what powers the `/approvals/stream` SSE multiplexer and any custom thread-event consumers.

### `.with_observability(config=None)`

Initializes Logfire + auto-instrumentation. With no config it reads `settings.observability` (`LOGFIRE_TOKEN`, service name, environment, instrument flags). Once installed, every agent run, DBOS workflow, and FastAPI request emits spans automatically.

### `.with_judge_defaults(model, *, model_settings=None)`

Sets the process-wide default model used by `LLMJudge` instances when they don't pin one themselves. Useful when you have a dozen judges in different capabilities and want to swap them all by changing one line. Chapter 23 covers judges.

### `.use(*providers)`

Plug in a third-party `Provider`. A provider is a small object that gets to register lifespan hooks, routers, and dependencies. This is how integrations (Stripe, Slack, etc.) plug in without the framework knowing about them. Most apps don't need this.

### `.add_on_startup(hook)` / `.add_on_shutdown(hook)`

Async callables that run during FastAPI lifespan. Use these for one-time setup (warm a cache, validate config) and teardown (flush logs, close pools). The framework's own startup tasks (DBOS launch, migrations) are already wired; you only add your own.

## `.fastapi(...)` — what you actually get

```python
app = ballast.fastapi(
    cors="dev",                    # or CORSConfig(...) or None
    routers=[my_router],            # your APIRouters
    health_checks={"db": check_db}, # custom liveness probes
    # **fastapi_kwargs              # forwarded to FastAPI() (title, docs_url, etc.)
)
```

What's mounted out of the box:

- **`/health`** — liveness/readiness; runs any `health_checks` you passed
- **`/threads`** — CRUD: create, list, get, archive, delete, get history, post message
- **`/dbos`** — DBOS workflow inspector (if `.with_dbos()` was called)
- **`/approvals`** — list pending cards, get one, decide, SSE stream of new cards

What's *not* mounted (you build these):

- **Chat streaming endpoints** — these are app-specific (the agent, the prompt construction, the event encoding). The framework provides primitives in `ballast.api.streaming` you use to build them. See [chapter 22](22-observability.md) and the demo notes-app for an idiomatic example.
- **Auth middleware** — bring your own; set `current_user_id` ContextVar at the request boundary so downstream repos and resolvers scope correctly.

The `Engine` is stashed at `app.state.engine`. Your routes pull it like:

```python
@router.post("/something")
async def handler(request: Request):
    engine = request.app.state.engine
    thread_repo = engine.thread_repo
    ...
```

But for routes that need a repository, prefer FastAPI dependency injection so testing can override:

```python
from ballast.persistence.thread import get_thread_repo
from fastapi import Depends

@router.post("/something")
async def handler(repo = Depends(get_thread_repo)):
    ...
```

Then in tests: `app.dependency_overrides[get_thread_repo] = lambda: my_fake_repo`.

## `BallastSettings` and environment variables

```python
from ballast import BallastSettings

settings = BallastSettings()    # reads env vars; raises if required ones missing
```

The full env-var schema (prefix `BALLAST_`, nested delimiter `__`):

```
BALLAST_DBOS__DATABASE_URL=postgresql+psycopg://user:pw@host/db
BALLAST_DBOS__APP_NAME=my-app
BALLAST_OBSERVABILITY__LOGFIRE_TOKEN=...
BALLAST_OBSERVABILITY__SERVICE_NAME=my-app
BALLAST_OBSERVABILITY__ENVIRONMENT=production
BALLAST_OBSERVABILITY__INSTRUMENT_FASTAPI=true
BALLAST_API__INSTALL_ERROR_MIDDLEWARE=true
BALLAST_API__EXPOSE_TRACEBACKS=false
BALLAST_LOGGING__LEVEL=INFO
BALLAST_AUTO_MIGRATE=true
```

A few standalone aliases are kept for backward compat: `DBOS_DATABASE_URL`, `BALLAST_LOG_LEVEL`.

`BALLAST_AUTO_MIGRATE=true` runs `alembic upgrade head` at startup. Convenient in dev, hazardous in prod (you usually want migrations to be a deploy step, not a startup race). Default is `false`.

For local dev a `.env` file works (pydantic-settings reads it automatically). For production, inject env vars via your platform's secret manager.

## Where the agent lives

The framework deliberately does *not* own your agent instances. There's no `Ballast.with_agents([...])` registry. The reason: agents are app code. They depend on your deps types, your tools, your prompt logic. Forcing them through a framework registry would either constrain you (one canonical shape) or be a thin pass-through (no value added).

Instead, build agents as module-level globals or factory functions, import them into your routers, and call `agent.run(...)` directly:

```python
# app/agents.py
from pydantic_ai import Agent
from app.deps import NotesDeps

notes_agent = Agent(
    model="openai:gpt-4o",
    deps_type=NotesDeps,
    system_prompt="...",
    tools=[search_notes, create_note],
)

# app/routes.py
from app.agents import notes_agent

@router.post("/chat/{thread_id}")
async def chat(thread_id: str, body: ChatBody, ...):
    deps = NotesDeps(notes_repo=..., user_id=current_user_id())
    result = await notes_agent.run(body.text, deps=deps)
    return {"output": result.output}
```

This is the same shape as Flask/FastAPI conventions for handler-level service objects. No magic.

## Dependency overrides for testing

Because routers grab repos via `Depends(get_thread_repo)`, tests can swap them with one line:

```python
from ballast.persistence.thread import get_thread_repo

def test_something(client):
    fake_repo = InMemoryThreadRepository()
    client.app.dependency_overrides[get_thread_repo] = lambda: fake_repo
    ...
```

The `client` fixture from `ballast.testing.pytest_plugin` (chapter 10) wraps app construction + lifespan and exposes a `TestClient`. The dependency-override pattern works through it unchanged.

## Lifecycle in one diagram

```
uvicorn starts
  ├─ FastAPI lifespan __aenter__
  │     ├─ user on_startup hooks run
  │     ├─ DBOS launch (if .with_dbos())
  │     ├─ optional Alembic upgrade head (if BALLAST_AUTO_MIGRATE=true)
  │     └─ observability initialized (if .with_observability())
  │
  ├─ serving HTTP requests
  │     ├─ POST /threads → engine.thread_repo.create(...)
  │     ├─ POST /chat (your route) → agent.run(..., deps=...)
  │     ├─ GET /approvals/stream → SSE multiplexer over event_stream
  │     └─ ...
  │
  └─ FastAPI lifespan __aexit__
        ├─ user on_shutdown hooks run
        ├─ DBOS destroy
        └─ observability flush
```

The framework's contract: startup is idempotent across reloads; shutdown is best-effort but won't lose persisted state (it's already in the DB by the time you call shutdown).

## Common mistakes

A few things that bite the first time:

- **Forgetting `.with_dbos()`** when using `@Durable.workflow`. The workflow decorator will silently fail (or worse, raise an uninformative error at first call). Always wire DBOS if any agent/pattern uses durability.
- **Building agents inside route handlers.** Don't. Agents do a lot of setup work (schema extraction, capability cloning). Build them once at import time, share across requests.
- **Leaking `deps` between requests.** Construct `deps` per-request inside the handler. Chapter 4 covers this in depth.
- **Setting `BALLAST_AUTO_MIGRATE=true` in production.** Two replicas booting at once will race the migration. Use a separate migration step in your deploy pipeline.

## What this chapter did NOT cover

- The exact shape of `ThreadRepository` / `ApprovalCardRepository` and how to swap them — chapter 9.
- Testing the assembled app — chapter 10.
- Building the chat-streaming endpoint that powers the assistant-ui frontend — chapter 22.
- The `/approvals` HITL surface — chapter 21.
- Observability setup details — chapter 22.

## Where to go next

→ [09-persistence.md](09-persistence.md) — making thread / message / approval state survive restarts.

# 4. Dependencies and state

**Prerequisites:** [01-agents.md](01-agents.md), [02-tools.md](02-tools.md).

## Introduction

Tools and capabilities almost always need *context*: a database connection, the current user's ID, request metadata, repository handles. None of this can come from the LLM — the model doesn't (and shouldn't) know who the user is or how to authenticate to the database. The framework needs a way to thread this state into tools and capabilities at call time, scoped per agent run.

pydantic-ai solves this with `deps_type` and `RunContext`. You declare what shape your dependencies have; the framework injects an instance per `agent.run(...)` call; tools and capabilities read it through `ctx.deps`. The same machinery scales from a single field (`user_id`) to a rich dataclass with multiple repositories, settings, and request data.

This chapter covers how to declare deps, where to construct them, what `RunContext` exposes, the `current_user_id` ContextVar for auth scoping, the `for_run` capability hook that gives stateful capabilities per-run isolation, and one critical rule: **do not put long-lived state on Agent or Capability instance attrs.**

## The mental model

Picture a single HTTP request hitting your FastAPI app. The request carries a user identity, possibly some session state, and references to long-lived resources (a database engine, an HTTP client, a thread repository). When the request triggers an agent run, all of that needs to be available to tools and capabilities without leaking across requests.

The framework gives you two complementary mechanisms:

1. **`deps_type` + `RunContext.deps`** — explicit, typed, per-run state passed into tools. You construct the deps object in your route handler; the framework hands it to every tool call.
2. **ContextVars** (specifically `current_user_id`) — implicit, ambient state that capabilities and repositories can read without an explicit handoff. Useful for cross-cutting concerns that don't belong in the deps signature.

Use deps for everything tool-specific. Use ContextVars sparingly — they're for things that *every* layer needs (auth identity is the canonical case).

## Declaring deps

The pattern is: define a dataclass or pydantic model for your deps, pass it as `deps_type` to the Agent, and use `RunContext[YourDeps]` in your tool signatures.

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class NoteToolDeps:
    user_id: str
    notes_repo: "NotesRepository"

agent = Agent(
    model="openai:gpt-4o-mini",
    deps_type=NoteToolDeps,
)

@agent.tool
async def search_notes(ctx: RunContext[NoteToolDeps], query: str) -> list[dict]:
    """Search the user's notes."""
    notes = await ctx.deps.notes_repo.search(ctx.deps.user_id, query)
    return [n.model_dump(mode="json") for n in notes]

@agent.tool
async def create_note(ctx: RunContext[NoteToolDeps], title: str, body: str) -> str:
    """Create a new note for the user."""
    note = await ctx.deps.notes_repo.create(
        user_id=ctx.deps.user_id, title=title, body=body,
    )
    return f"Created note {note.id}"
```

What you've declared: every tool on this agent receives a `RunContext[NoteToolDeps]` as its first parameter. The `deps` field on that context is a `NoteToolDeps` instance you supply per call. Type-checking works end to end — your IDE knows `ctx.deps.user_id` is a string and `ctx.deps.notes_repo` is your repo type.

The dataclass is the most common choice (cheap, mutable, no validation overhead). A pydantic `BaseModel` works too if you want validation on the deps themselves.

## Constructing deps in your route handler

Deps are constructed *per request* — they live as long as the agent run. Typical pattern in a FastAPI route:

```python
from fastapi import Depends, Request
from ballast.persistence import get_thread_repo
from notes_app.repositories.note import notes_repo

@app.post("/notes/agent/run")
async def notes_run(
    body: dict,
    request: Request,
    thread_repo = Depends(get_thread_repo),
):
    user_id = extract_user(request)
    deps = NoteToolDeps(user_id=user_id, notes_repo=notes_repo)

    result = await notes_agent.run(body["query"], deps=deps)
    return {"output": result.output}
```

Three things to notice:

1. **`deps=deps`** is the kwarg pydantic-ai uses on `agent.run`. The framework hands `deps` to every tool's `ctx.deps`.
2. **Construction happens at request time.** Don't construct deps once at module import — that defeats the per-request isolation.
3. **Repositories are typically module-level singletons** (`notes_repo`) but the per-user fields (`user_id`) are computed per request. Mix-and-match is fine.

In Ballast's higher-level routing (the framework's `build_streaming_router`), this happens for you — you declare a `make_deps` factory and the router calls it per request:

```python
def make_deps(thread, request) -> NoteToolDeps:
    return NoteToolDeps(user_id=extract_user(request), notes_repo=notes_repo)

router = build_streaming_router(agents=[notes_agent], make_deps=make_deps)
```

Chapter 8 covers the routing layer end-to-end.

## What `RunContext` exposes besides `deps`

`RunContext` is more than just a deps holder. Inside a tool, you can also access:

- **`ctx.usage`** — the cumulative usage (input/output tokens) so far in this run. Useful for tools that want to short-circuit when budget is tight.
- **`ctx.model`** — the model being used. Lets a tool log which provider is active or behave differently for streaming vs. non-streaming models.
- **`ctx.retry`** — how many times pydantic-ai has retried this run due to validation errors. A high retry count is signal that the schema or prompt needs revision.
- **`ctx.run_step`** — which iteration of the agent loop you're in. Tool that should only run once can check `if ctx.run_step > 1: return "already ran"`.

You won't use most of these often — but they're there when you need to write a tool that reasons about the agent's run state.

## `current_user_id` ContextVar for ambient auth

Some pieces of state are cross-cutting — every repository, every capability, every tool wants to know who the current user is. Threading `user_id` through every signature gets noisy fast. The framework ships a `current_user_id` ContextVar for this:

```python
from ballast.auth.context import current_user_id, acting_as

# Inside an auth middleware or FastAPI dependency:
async def auth_middleware(request: Request, call_next):
    user_id = await extract_user_from_jwt(request)
    with acting_as(user_id=user_id, tenant_id=jwt.tenant):
        return await call_next(request)
```

Now anywhere — inside a tool, a capability, a repository, a workflow — you can:

```python
from ballast.auth.context import current_user_id

async def some_function():
    user_id = current_user_id()    # gets it from the ContextVar
    ...
```

The framework's `SqlApprovalCardRepository`, `SqlThreadRepository`, and `goal_drift_as_unit` all read `current_user_id` internally. You set it once at the request boundary; everything else inherits it.

Use ContextVars *sparingly*. The rule of thumb: if every layer needs this information (auth identity), use a ContextVar. If only specific tools need it (a particular repo handle), put it on `deps`. ContextVars are global state in disguise; their cost is opacity.

## The `for_run` capability hook

Capabilities can be stateful — counters, embeddings, lock state. Different agent runs need to see *independent* state, even if they all use the same capability instance. The framework's solution is the `for_run` hook.

When you do:

```python
agent = Agent(
    model=...,
    capabilities=[BudgetGuard(max_iterations=10)],
)
```

You're holding a *config object*, not a stateful instance. When `agent.run(...)` starts, the framework calls `capability.for_run(ctx)` on each capability. The default returns `self` (stateless capabilities); stateful ones return a *fresh clone*:

```python
class BudgetGuard(BallastCapability):
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations
        self._iterations = 0    # this counter belongs to THIS clone, not the shared instance

    async def for_run(self, ctx):
        # Return a fresh instance per run — counters start at zero each time
        return BudgetGuard(max_iterations=self.max_iterations)
```

The cloned capability is what receives `before_model_request` / `after_model_request` / `after_run` callbacks for *this run only*. Two concurrent `agent.run` calls each get their own clone; counters don't bleed across runs.

This is why you can construct a `BudgetGuard` once and pass the same instance to every agent — there's no shared mutable state on the original. Same for `SemanticLoopDetector`, `GoalDriftDetector`, every framework-shipped capability. Chapter 7 covers the capability protocol in depth.

## The rule: don't put long-lived state on Agent or Capability instance attrs

Two mistakes worth calling out explicitly because they're easy to make:

### Mistake 1: caching results on the Agent

```python
# WRONG
class MyAgent(BallastAgent):
    name = "my"
    _cache: dict = {}    # <-- DON'T

    def build_agent(self):
        agent = Agent(...)
        @agent.tool
        async def lookup(ctx, key: str):
            if key in self._cache:
                return self._cache[key]
            value = await db.lookup(key)
            self._cache[key] = value
            return value
        return agent
```

`self._cache` is module-level. Concurrent requests share it. Cross-request data leak is one bad day away. Move the cache to a per-request store (Redis, in-memory dict scoped via deps, or actual cache library with explicit TTL).

### Mistake 2: storing per-run state on a capability without `for_run`

```python
# WRONG
class MyGuard(BallastCapability):
    def __init__(self):
        self._counter = 0       # <-- DON'T

    async def after_model_request(self, ctx, *, request_context, response):
        self._counter += 1
        if self._counter > 10:
            raise BudgetExhausted(...)
```

`self._counter` is shared across all agent runs that use this guard. After 10 *total* runs across the lifetime of the process, every subsequent run starts at +1 and fails immediately. Override `for_run` to return a fresh clone (`return MyGuard()`).

The rule is simple: **if it changes over time and is run-specific, it does not belong on `self` of an Agent or Capability that's instantiated once and shared.** Use `for_run` for capabilities; use `deps` for tools.

## When `RunContext.deps` is `None`

Sometimes an agent has no deps — it's a pure chat agent, no tools that need state. In that case `deps_type` is unset and `ctx.deps` is `None`. Your tools either don't take `RunContext` at all (`@agent.tool_plain`) or accept it but don't read `.deps`:

```python
agent = Agent(model="openai:gpt-4o-mini")   # no deps_type

@agent.tool_plain
async def get_time() -> str:
    """Return the current UTC time."""
    return datetime.now(UTC).isoformat()
```

Reach for deps the moment a tool needs *anything* request-specific. Don't try to be clever about avoiding it.

## Common deps patterns

A few shapes that come up across most apps:

### Deps with one repo

```python
@dataclass
class NoteDeps:
    user_id: str
    notes_repo: NotesRepository
```

Smallest useful deps. Single user, single repo. Most "simple" agents have this shape.

### Deps with multiple repos and request data

```python
@dataclass
class WorkflowDeps:
    user_id: str
    tenant_id: str
    notes_repo: NotesRepository
    project_repo: ProjectRepository
    settings: AppSettings
    request_id: str          # for tracing / idempotency
```

Larger agents that touch multiple domains. Pass everything they could conceivably need; tools pick what they want.

### Deps with a session

```python
@dataclass
class TransactionalDeps:
    user_id: str
    db_session: AsyncSession   # single session for the whole agent run

    async def __aenter__(self):
        await self.db_session.begin()
        return self

    async def __aexit__(self, *exc):
        if exc[0]:
            await self.db_session.rollback()
        else:
            await self.db_session.commit()
        await self.db_session.close()
```

Pattern for agents that perform multi-step transactions. The session is opened once per run; tools share it; transaction commits on success, rolls back on failure. The route handler enters the context manager around the `agent.run(...)` call.

## Testing with deps

For tests, construct deps manually with fakes:

```python
from pydantic_ai.models.test import TestModel

@pytest.mark.asyncio
async def test_notes_search_tool() -> None:
    fake_repo = InMemoryNotesRepository()
    await fake_repo.create(user_id="u-1", title="hello", body="world")

    agent = Agent(
        model=TestModel(call_tools=["search_notes"]),
        deps_type=NoteToolDeps,
    )

    @agent.tool
    async def search_notes(ctx: RunContext[NoteToolDeps], query: str) -> list[dict]:
        notes = await ctx.deps.notes_repo.search(ctx.deps.user_id, query)
        return [n.model_dump(mode="json") for n in notes]

    deps = NoteToolDeps(user_id="u-1", notes_repo=fake_repo)
    result = await agent.run("find hello", deps=deps)
    # assertions on tool calls + result
```

In-memory repos are usually available alongside the SQL ones (`InMemoryNotesRepository`, `InMemoryThreadRepository`, etc.). Chapter 10 covers testing patterns in depth.

## What this chapter did NOT cover

- The `BallastCapability` protocol that uses `for_run` formally — chapter 7.
- The `Ballast()` app builder that ties deps construction to FastAPI routes — chapter 8.
- ThreadRepository / ApprovalCardRepository persistence backends — chapter 9.

## Where to go next

→ [05-grounded-references.md](05-grounded-references.md) — typed entity references that prevent the agent from hallucinating IDs.

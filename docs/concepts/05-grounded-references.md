# 5. Grounded references — `Ref[T]`

**Prerequisites:** [03-structured-output.md](03-structured-output.md), [04-dependencies-and-state.md](04-dependencies-and-state.md).

## Introduction

LLMs invent things. Ask a model to pick an existing note from your database and it will cheerfully return `Note(id="abc-123-doesnt-exist")` with full confidence. The hallucination problem isn't a quirk — it's structural. The model has no native concept of "real ID in this database"; it just generates plausible-looking strings.

The brute-force solution is to give the model a list of all valid IDs in the prompt: "Pick one of these UUIDs: ...". This works at small scale and breaks fast. With 100 candidates, your prompt explodes. With 10,000, you can't fit them at all. With 100,000, you're spending more on prompt tokens than on actual reasoning.

The framework's solution is `Ref[T]` — a typed entity reference that hooks into pydantic-ai's structured-output machinery. When you declare an output field as `Ref[Project]`, the framework knows the field's *valid values* aren't free text; they're a discrete set determined at runtime by your app. At the moment the agent runs, the framework walks the output schema, finds the `Ref` fields, queries your app for the list of candidates, and *narrows the JSON Schema sent to the LLM* into a `Literal` enum of just those IDs. The model is structurally incapable of producing an invalid ID — JSON-Schema-valid output IS valid-in-your-database output.

This chapter covers how `Ref[T]` works, how to use it in agent outputs, how `GroundedAgent` wires the narrowing logic, how to hydrate refs into full entities, and the current limitations (input-side hydration is not yet automatic).

## The mental model

Think of `Ref[T]` as a *typed pointer*. The model produces an ID; you carry that ID around in your code as a `Ref[T]`; when you need the full entity, you ask your repository to hydrate it. The framework handles two pieces:

1. **Schema narrowing at agent-run time** — the JSON Schema sent to the LLM lists the valid IDs as an enum.
2. **Hydration after the run** — `result.hydrate(some_repo=...)` walks the result, finds every `Ref[T]`, fetches the entity by ID, and returns a copy with refs replaced by entities.

Together, these guarantee two things: the model can't hallucinate; your downstream code gets typed entities, not raw IDs that may not even exist.

## The simplest case

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from ballast import Ref

class Project(BaseModel):
    id: str
    name: str

class ResearchSummary(BaseModel):
    summary: str
    project: Ref[Project]       # typed reference

agent = Agent(
    model="openai:gpt-4o-mini",
    output_type=ResearchSummary,
)

result = await agent.run("Summarize the ML deployment project")
print(result.output.project)     # Ref[Project] — just an ID, no further data yet
```

This works at the *type* layer but not yet at the *narrowing* layer. The JSON Schema sent to the LLM still says `project: {type: string, format: uuid}` — meaning the model could produce any UUID it wants. To enable narrowing, wrap the agent with `GroundedAgent`:

```python
from ballast import GroundedAgent, GroundedResolver

class ProjectResolver(GroundedResolver[Project]):
    async def candidates(self) -> list[Project]:
        return await project_repo.list_all()

grounded = GroundedAgent(
    inner=agent,
    resolvers={Project: ProjectResolver()},
)

result = await grounded.run("Summarize the ML deployment project")
# Schema narrowing happened automatically; result.output.project.id is guaranteed-real.
```

Now when `grounded.run(...)` is called, the framework:

1. Walks `ResearchSummary` schema, identifies `project: Ref[Project]`.
2. Calls `ProjectResolver().candidates()` to get the live list of valid projects.
3. Rewrites the JSON Schema: `project: {enum: [<id-1>, <id-2>, ...]}`.
4. Sends the narrowed schema to the LLM.
5. The model picks an ID from the enum (any other ID would be a JSON Schema violation).
6. pydantic validates the response.

Your output is guaranteed to reference a real project.

## Hydrating refs to full entities

`Ref[T]` carries just the ID. To get the actual `Project` model:

```python
hydrated = await result.hydrate(project=project_repo)
print(hydrated.project)              # Now a real Project, not a Ref
print(hydrated.project.name)         # 'ML Deployment'
print(hydrated.project.description)  # full details
```

`result.hydrate(**repos)` walks the output, matches each `Ref[T]` to a repo by keyword (`project=project_repo` matches `Ref[Project]`), and calls `repo.get(ref.id)` for each. The result is the same shape as the input but with refs replaced by entities.

The repo only needs to implement one method: `async def get(self, id: str) -> T | None`. Whatever your repo type, satisfy this and `hydrate` works.

## Lists, optionals, nested

`Ref[T]` composes naturally:

```python
class ResearchOutput(BaseModel):
    summary: str
    primary_project: Ref[Project]              # single ref
    related_projects: list[Ref[Project]]        # list of refs
    parent_project: Ref[Project] | None         # optional ref

grounded = GroundedAgent(
    inner=Agent(model=..., output_type=ResearchOutput),
    resolvers={Project: ProjectResolver()},
)

result = await grounded.run(query)
hydrated = await result.hydrate(
    primary_project=project_repo,
    related_projects=project_repo,    # same repo handles both single and list
    parent_project=project_repo,
)
print(hydrated.primary_project.name)
print([p.name for p in hydrated.related_projects])
```

For nested models:

```python
class Snippet(BaseModel):
    text: str
    source: Ref[Source]

class ResearchOutput(BaseModel):
    summary: str
    snippets: list[Snippet]    # framework walks into Snippet to find the Ref
```

`scan_output` (the framework's schema walker) recurses through BaseModel fields and list/optional containers naturally — refs at any depth are discovered without special configuration.

This is also why `Scored[Note]` (chapter 6) composes seamlessly with grounded outputs: `scan_output` walks into `Scored.value` and finds any refs inside.

## The resolver pattern

`GroundedResolver[T]` is the contract that tells the framework "here's how to get the list of valid T candidates." The minimum implementation:

```python
class ProjectResolver(GroundedResolver[Project]):
    async def candidates(self) -> list[Project]:
        return await project_repo.list_all()
```

Three variations you'll likely write:

### Per-user / per-tenant scoping

The agent should only be able to reference projects the *current user* can see:

```python
from ballast.auth.context import current_user_id

class UserProjectResolver(GroundedResolver[Project]):
    async def candidates(self) -> list[Project]:
        return await project_repo.list_for_user(current_user_id())
```

The `current_user_id()` ContextVar (chapter 4) lets the resolver scope candidates without needing it as a constructor argument. Set the ContextVar at the request boundary; the resolver inherits it.

### Top-N narrowing

For repos with thousands of candidates, send only the most relevant:

```python
class RecentProjectResolver(GroundedResolver[Project]):
    def __init__(self, limit: int = 50):
        self._limit = limit

    async def candidates(self) -> list[Project]:
        return await project_repo.list_recent(
            user_id=current_user_id(), limit=self._limit,
        )
```

The model only sees the top-50 — manageable token cost, still grounded. You're trading off recall (older projects aren't selectable) for prompt size. Tune the number based on your data distribution.

### Search-narrowed candidates

If you know the agent will look for projects matching the user's query, prefilter:

```python
class QueryNarrowingResolver(GroundedResolver[Project]):
    def __init__(self, query: str):
        self._query = query

    async def candidates(self) -> list[Project]:
        return await project_repo.search(
            user_id=current_user_id(), query=self._query, limit=20,
        )
```

You'd construct this resolver per request, injecting the query from the user's input. This is the lowest-token-cost approach and the highest-precision — but it requires you to know the query upfront, which isn't always possible.

## Large-candidate-set warnings

If your resolver returns more than ~500 candidates, the framework emits a warning. The narrowed schema gets expensive at scale: 500 UUIDs is ~9,000 prompt tokens just for the enum. At 5,000 candidates, you're spending more on the enum than on the rest of the prompt.

Three things to do when you hit this:

1. **Narrow at the resolver layer** (above patterns). If the candidates aren't all relevant to *this* request, filter them out before they reach the schema.
2. **Question whether `Ref[T]` is the right tool**. If your "entity" has thousands of equally-relevant instances, maybe what you actually need is a search tool (`@agent.tool def search_projects(...)`) rather than a grounded reference. Refs work best when the candidate set is bounded and curated.
3. **Wait for `Selector`** (planned, not yet shipped). The future `Annotated[Ref[T], Selector(...)]` will let the agent describe what it wants ("the most recent project tagged 'ml'") and the framework will resolve to a specific ID without sending the full candidate list. Until then, custom narrowing is the workaround.

## When hydration fails

`result.hydrate(...)` calls `repo.get(ref.id)` for each ref. If a repo returns `None` (entity deleted between schema-narrowing and hydration, race condition, etc.), the framework raises `GroundedHydrationError`. Catch at app boundary:

```python
from ballast import GroundedHydrationError

try:
    hydrated = await result.hydrate(project=project_repo)
except GroundedHydrationError as exc:
    logger.warning("hydration failed: %s", exc)
    return "The entity I referenced has been deleted. Please re-run."
```

In practice this is rare — narrowing happens at run start, hydration happens immediately after, so the window for deletion is small. But it's not impossible, and silently substituting `None` would be worse than raising.

## What `Ref[T]` is, technically

`Ref[T]` is a generic class with a custom pydantic core schema. At validation time it accepts:

- A UUID string (the common case — what the LLM produces)
- An existing `Ref[T]` instance (passthrough)
- A `T` instance (auto-extracts `.id`)

It carries `id: str` and a hidden `entity_type: type[T]` derived from the generic parameter. Equality and hashing are based on the ID. JSON serialization emits the bare ID string. This makes `Ref[T]` interoperable with any code that treats IDs as strings — your existing repos, your database, your URL routes.

The `scan_output` walker (lives in `ballast.grounded`) is the schema introspector. It walks pydantic model fields, classifies each as `VALUE` / `REF` / `LIST_REF` / `OPTIONAL_REF` / `NESTED`, and feeds the narrowing engine. You don't interact with it directly unless you're building custom grounded-aware tooling.

## Current limitation: tool-input hydration is manual

`Ref[T]` works automatically for *outputs*. For *tool inputs* (the model passing an ID into a tool call), the framework narrows the JSON Schema (you get the enum constraint), but it doesn't yet *hydrate the entity before calling the tool body*. The tool receives a `Ref[T]` and must hydrate manually:

```python
@agent.tool
async def update_note(ctx: RunContext, note: Ref[Note], new_title: str) -> str:
    full_note = await note.hydrate(notes_repo)   # manual
    if full_note is None:
        return "Note not found."
    full_note.title = new_title
    await notes_repo.save(full_note)
    return f"Updated {note.id}"
```

Auto-hydration on tool inputs requires a pre-call hook in pydantic-ai that doesn't yet exist (this is the open task #108). Until then, manual hydration is the pattern. It's three lines of boilerplate per tool; not the end of the world.

The narrowing still works — the LLM can only call the tool with a real `note.id`. So even with manual hydration, you're protected from hallucinated IDs.

## Discipline checklist

- **Use `Ref[T]` whenever the agent must reference a real entity.** Strings invite hallucination.
- **Always wrap with `GroundedAgent` when you want narrowing.** Without it, refs are just typed strings — no schema-level protection.
- **Scope resolvers by user / tenant.** Don't expose entities the current user can't see.
- **Narrow the candidate set if it's large.** 50-100 is comfortable; 500+ starts hurting; 5,000+ is broken.
- **Always handle `GroundedHydrationError`** at app boundary — even if rare.
- **Don't use `Ref` for primitive values.** `Ref[str]` is meaningless. Refs are for `BaseModel` subclasses.

## What this chapter did NOT cover

- `Scored[T]` for confidence + rationale wrapping — chapter 6.
- Custom `Selector` for query-based ref resolution — out of scope for now (planned task).
- How `GroundedAgent` composes with capabilities like `GroundedRetry` — chapter 7 + chapter 11.

## Where to go next

→ [06-confidence-and-quality.md](06-confidence-and-quality.md) — adding quality signals to outputs.

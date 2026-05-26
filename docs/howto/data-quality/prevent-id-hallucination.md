# How to use `Ref[T]` to prevent UUID hallucination

**Problem:** Your agent must reference an entity (a Note, a User, a Project) — but the LLM tends to hallucinate UUIDs / IDs / slugs that don't exist. The agent says "I updated Note abc-123" when that note never existed. Downstream code crashes (or worse, silently misfires).

**Solution:** `Ref[T]` — a typed UUID wrapper with custom pydantic core schema. Used in agent outputs OR tool inputs. The framework's `scan_output` narrows the JSON Schema sent to the LLM to a `Literal` enum of REAL IDs at runtime, so the LLM literally can't emit an invalid one.

## Minimum: typed output with grounded reference

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from ballast import Ref


class Project(BaseModel):
    id: str
    name: str
    description: str


class ResearchSummary(BaseModel):
    summary: str
    project: Ref[Project]      # ← typed reference to a Project entity


agent = Agent(
    model="openai:gpt-4o",
    output_type=ResearchSummary,
)
```

Without further setup, this validates the LLM's response shape but doesn't constrain WHICH project IDs the LLM can pick. To constrain, wrap with `GroundedAgent`:

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
print(result.output.project)         # Ref[Project](id="real-uuid")
```

Now the JSON Schema sent to the LLM has `project: {enum: [<real-uuid-1>, <real-uuid-2>, ...]}`. The model is structurally incapable of emitting a fake ID.

## Hydrate the ref to a full entity

The output still contains a `Ref[Project]` (just an ID). To get the full `Project` object:

```python
hydrated = await result.hydrate(project=project_repo)
print(hydrated.project.name)         # the real Project model
print(hydrated.project.description)
```

`result.hydrate(**repos)` looks up each `Ref` in the corresponding repo (`project=...` matches `Ref[Project]`).

## List of refs

```python
class ResearchOutput(BaseModel):
    summary: str
    related_projects: list[Ref[Project]]   # framework handles list-of-refs


grounded = GroundedAgent(
    inner=Agent(model=..., output_type=ResearchOutput),
    resolvers={Project: ProjectResolver()},
)

result = await grounded.run("Find projects related to ML")
hydrated = await result.hydrate(related_projects=project_repo)
for p in hydrated.related_projects:
    print(p.name)
```

## Optional refs

```python
class ResearchOutput(BaseModel):
    summary: str
    parent_project: Ref[Project] | None    # OPTIONAL_REF role
```

LLM may omit / null-out; hydrated value is `None` or a real `Project`.

## Nested refs

```python
class Snippet(BaseModel):
    text: str
    source: Ref[Source]

class ResearchOutput(BaseModel):
    summary: str
    snippets: list[Snippet]    # framework walks into Snippet to find the Ref


grounded = GroundedAgent(
    inner=Agent(model=..., output_type=ResearchOutput),
    resolvers={Source: SourceResolver()},
)
```

`scan_output` walks into nested BaseModels and into list/optional containers — finds Refs at any depth.

## Use with `Scored[T]`

`Scored[T].value: T` — if T contains refs, they're found via natural recursion:

```python
from ballast import Scored

class FactWithSource(BaseModel):
    text: str
    source: Ref[Source]


agent = Agent(model=..., output_type=Scored[FactWithSource])
grounded = GroundedAgent(inner=agent, resolvers={Source: SourceResolver()})

result = await grounded.run(query)
print(result.output.confidence)              # "low" | "medium" | "high"
print(result.output.value.source)            # Ref[Source]
# Hydrate as usual; .value is just a nested model.
```

## Handle large candidate sets

If a resolver returns thousands of candidates, the LLM context blows up. Framework emits a warning. Recommended: implement narrowing in your resolver:

```python
class TopRecentProjectsResolver(GroundedResolver[Project]):
    def __init__(self, repo, limit: int = 50):
        self._repo = repo
        self._limit = limit

    async def candidates(self) -> list[Project]:
        # Only show top-50 most recent — LLM picks from these
        return await self._repo.list_recent(limit=self._limit)
```

For really large sets (millions), this is where `Selector` (planned, deferred) will help — DSL for app-side narrowing per-tool-call.

## Limitations

- **Tool inputs auto-hydration not yet implemented** (`#108` task). When LLM calls a tool with `Ref[Note]` param, the tool body receives a `Ref` (with `.id`) — must call `note.hydrate(notes_repo)` manually. Output hydration works (above example); input hydration is a future iteration that requires pydantic-ai's `before_tool_call` hook.
- **Resolver runs on every agent run.** If `candidates()` is expensive, cache inside the resolver impl.
- **The narrowed enum is a snapshot.** If the repo changes between the schema send and the LLM response, the LLM might pick an ID that no longer exists. Add a defensive `if not await note.hydrate(repo)` check in your downstream code.

## Errors

- `GroundedHydrationError` — `Ref.hydrate(repo)` returned `None`. The ID was valid at schema-narrowing time but the entity was deleted before hydration.
- `GroundedBuildError` — resolver returned an invalid candidate list (e.g. duplicates, missing IDs).

Both subclass `BallastError`; catch at app boundary.

## Caveats

- **Resolvers are tenant/user-scoped on YOUR side.** The framework doesn't know which user owns which project. Filter candidates in your resolver impl using `current_user_id()` if needed.
- **Don't `Ref` primitives.** `Ref[str]` is meaningless. Use `Ref[BaseModel-subclass]` only.
- **Repository must have `async def get(id) -> T | None`.** This is the only requirement for hydration. Your repo can do whatever auth / scoping logic inside.

## Related

- [add-confidence-to-outputs.md](add-confidence-to-outputs.md) — `Scored[T]` composition
- [build-cognitive-units.md](../multi-step-orchestration/build-cognitive-units.md) — CoALA units that return entities by `Ref`
- Reference: `reference/grounded/ref-t.md`
- Reference: `reference/grounded/grounded-agent.md`
- Explanation: [article-pain-points.md](../../explanation/article-pain-points.md) #11

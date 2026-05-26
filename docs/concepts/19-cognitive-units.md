# 19. Cognitive Units — CoALA

**Prerequisites:** [07-capabilities.md](07-capabilities.md), [14-patterns-intro.md](14-patterns-intro.md), [18-plan-and-execute.md](18-plan-and-execute.md).

## Introduction

A typical agent function blurs everything into one chunk: "given input, do whatever, return output." For simple tools that's fine. For computations that have a *memory* component — looking up prior context, deciding based on both input and memory, then writing something back — the blur becomes a problem. It's hard to test the parts independently, hard to swap in a different retrieval strategy, hard to reason about what gets persisted when.

The CoALA cognitive architecture (Sumers et al.) gives that structure a name: every cognitively-meaningful computation has four phases — *observe* what came in, *retrieve* relevant prior context, *act* to produce an output, *learn* what to persist. The framework's `CoALAUnit` Protocol makes this explicit so each phase is independently testable and replaceable, and ships three adapters (`as_tool`, `as_workflow`, `as_capability`) that let the same unit deploy as a pydantic-ai Tool, a durable workflow, or an agent capability.

This chapter walks through the four phases, the `CoALABase` ABC that gives you sensible defaults, the three adapters and what each does, and how `UnitStep` in `PlanAndExecute` bridges CoALA units into typed DAGs.

## The mental model

```
input
  │
  ├── observe(input) ─────► observation         # structure the raw input
  │
  ├── retrieve(observation) ─► context           # fetch from app's memory
  │
  ├── act(observation, context) ─► output       # do the work
  │
  └── learn(observation, context, output) ─► None  # persist what should outlast this call
```

Four phases, four pure methods. Each phase has clear inputs and outputs; you can test any one in isolation. Storage decisions are entirely *yours* — the framework calls `retrieve()` and `learn()` but doesn't prescribe what they read or write.

The framework's contribution is the protocol + the adapters. The app owns:

- The semantics of each phase (what counts as a "good" retrieval, what to write in `learn`).
- The storage backend (Postgres, vector DB, Redis, in-memory dict, whatever).

This is deliberate: cognitive computations are domain-specific. The framework's job is the contract, not the implementation.

## The `CoALAUnit` Protocol

```python
@runtime_checkable
class CoALAUnit(Protocol[InT, ObsT, ContextT, OutT]):
    async def observe(self, input: InT) -> ObsT: ...
    async def retrieve(self, observation: ObsT) -> ContextT: ...
    async def act(self, observation: ObsT, context: ContextT) -> OutT: ...
    async def learn(self, observation: ObsT, context: ContextT, output: OutT) -> None: ...
```

Four generic parameters: input type, observation type, context type, output type. You can pick any of them to be `dict`, a pydantic model, a primitive — whatever fits your domain.

## `CoALABase` — defaults that make the protocol cheap

Implementing all four phases for every unit is overkill. `CoALABase` provides sensible defaults so you only implement what's *different*:

```python
class CoALABase(Generic[InT, ObsT, ContextT, OutT], ABC):
    async def observe(self, input: InT) -> ObsT:
        return input    # default: identity

    @abstractmethod
    async def retrieve(self, observation: ObsT) -> ContextT: ...

    @abstractmethod
    async def act(self, observation: ObsT, context: ContextT) -> OutT: ...

    async def learn(self, observation: ObsT, context: ContextT, output: OutT) -> None:
        return None     # default: no-op
```

The two phases you almost always implement: `retrieve` (where the context comes from) and `act` (what the unit actually does). `observe` defaults to identity (most inputs don't need pre-processing). `learn` defaults to no-op (many units are read-only).

## A concrete unit

```python
from ballast.coala import CoALABase
from pydantic import BaseModel

class ResearchInput(BaseModel):
    topic: str

class ResearchObs(BaseModel):
    topic: str
    keywords: list[str]

class ResearchContext(BaseModel):
    relevant_notes: list[Note]

class ResearchSummary(BaseModel):
    summary: str
    sources: list[str]

class ResearchSummarize(CoALABase[ResearchInput, ResearchObs, ResearchContext, ResearchSummary]):

    def __init__(self, *, notes_repo, summarizer_agent):
        self._notes = notes_repo
        self._summarizer = summarizer_agent

    async def observe(self, input: ResearchInput) -> ResearchObs:
        keywords = extract_keywords(input.topic)
        return ResearchObs(topic=input.topic, keywords=keywords)

    async def retrieve(self, observation: ResearchObs) -> ResearchContext:
        notes = await self._notes.search_by_keywords(observation.keywords, limit=10)
        return ResearchContext(relevant_notes=notes)

    async def act(self, observation: ResearchObs, context: ResearchContext) -> ResearchSummary:
        prompt = format_research_prompt(observation, context.relevant_notes)
        result = await self._summarizer.run(prompt)
        return result.output

    async def learn(self, observation: ResearchObs, context: ResearchContext, output: ResearchSummary) -> None:
        await self._notes.tag_used_for_summary(
            note_ids=[n.id for n in context.relevant_notes],
            summary_id=output.id,
        )
```

Four methods, four clear responsibilities. Each one is testable on its own (pass a fake `notes_repo`, assert on the call).

## The three adapters

A unit is just a class. The three adapters expose it on different deployment surfaces.

### `as_tool(unit, *, name=None, description=None)`

Wraps the unit as a pydantic-ai `Tool`. The LLM sees one tool call; the framework runs all four phases when called:

```python
from ballast.coala import as_tool

agent = Agent(
    model="openai:gpt-4o",
    tools=[
        as_tool(research_unit, name="research_topic"),
        ...,
    ],
)
```

The LLM calls `research_topic(input)` and gets the `ResearchSummary` back. Internally, the four phases run; the LLM doesn't see the structure.

### `as_workflow(unit)`

Wraps the unit as a `@Durable.workflow` where each phase is a `@Durable.step`:

```python
from ballast.coala import as_workflow

research_workflow = as_workflow(research_unit)

result = await research_workflow(input)
```

Each phase is memoised independently. If `act` (the LLM call) crashes, on replay `observe` and `retrieve` return their cached results; only `act` re-runs. This is the most expensive-failure-friendly deployment.

Implementation detail: this uses the `DBOSConfiguredInstance` pattern under the hood so per-instance state survives replays. You don't have to think about it; `as_workflow` handles the wiring.

### `as_capability(unit, *, gate=None)`

Wraps the unit as a `BallastCapability` where:

- `observe` + `retrieve` run in `before_model_request` (so the model sees the context),
- `act` is the model call itself (the capability doesn't replace it),
- `learn` runs in `after_run` (so the unit persists based on the final output).

```python
from ballast.coala import as_capability

agent = Agent(
    model="openai:gpt-4o",
    capabilities=[as_capability(research_unit)],
)
```

This is the right deployment when you want the unit to auto-fire for *every* run — e.g., automatically retrieve relevant prior context and append it to the prompt before any agent call.

`gate` is an optional async callable that returns `True`/`False` to decide whether the capability should fire this run. Useful for conditionally enabling retrieval.

The `learn` phase runs in `after_run` regardless of agent success — exceptions in `learn` are caught and swallowed so a failing learn step doesn't break the user-facing run.

## `UnitStep` in PlanAndExecute

`PlanAndExecute` has a built-in `unit` step kind that calls a registered CoALA unit through its full 4-phase lifecycle:

```python
registry.register_unit("research_summarize", research_unit)

# Planner emits:
PlannedStep(
    id="research",
    kind="unit",
    params={"unit_name": "research_summarize"},
    depends_on=["extract_topic"],
)
```

When the executor reaches this step, it dispatches `observe(plan_input)` → `retrieve(obs)` → `act(obs, ctx)` → `learn(obs, ctx, out)` on the registered unit. The step's output (passed to downstream dependencies) is the `act` return value.

## Apps own storage; framework owns the contract

This is the single most important design choice in CoALA. The framework deliberately doesn't ship "memory backends" because memory is intrinsically domain-specific. A research-summarization unit's memory is annotated notes; an account-management unit's memory is customer records; a code-review unit's memory is past review verdicts. There's no single "memory layer" that fits all three.

What the framework provides:

- The **Protocol** (so any unit looks the same from outside).
- The **adapters** (so units deploy on multiple surfaces).
- The **observability** (each phase is a span; you can see retrieval latency vs. act latency separately).

What you provide:

- The actual repositories that back `retrieve` and `learn`.
- The semantics of "context" — what you fetch, how you score relevance, what you return.

This is the lesson from Phase 1+2 of the CoALA work in this framework: trying to ship a generic `EpisodicMemory` / `SemanticMemory` layer led to leaky abstractions that didn't fit any actual app. The current design — Protocol + adapters, no shipped backends — is what apps actually want.

## DBOSConfiguredInstance under the hood

`as_workflow` wraps the unit in a `_CoALAWorkflow(DBOSConfiguredInstance)` so the per-phase memoisation works correctly. The pattern: the unit instance is stable across calls (you create one and reuse it); DBOS uses the instance's `config_name` to namespace step-cache keys; per-phase steps cache their outputs against the workflow ID.

You don't need to know the details unless you're writing a custom adapter. For the standard `as_workflow` usage, just pass your unit and the framework handles it.

## Common mistakes

- **Putting LLM calls in `retrieve`.** Retrieval should be fast and deterministic; if it needs an LLM, that's a sign the unit should be split into two units.
- **Putting database writes in `act`.** That's what `learn` is for. Keeping `act` write-free makes it safer to retry on replay.
- **`learn` that raises.** The framework catches `learn` exceptions to protect the user-facing run, but if your `learn` raises every time, you have a silent data-loss bug. Test the `learn` phase directly.
- **One unit doing too much.** If `act` is calling three different agents and doing complex logic, the unit is doing the work of a `PlanAndExecute`. Split into multiple units composed via PlanAndExecute.
- **Using `as_capability` for write-heavy units.** Capabilities fire on every run. If `learn` is heavy, you'll do that work for every call. Use `as_tool` (run only when the model calls it) or `as_workflow` (run explicitly) instead.

## What this chapter did NOT cover

- The CoALA paper itself — Sumers et al., 2024.
- Composing units with patterns and capabilities into a full pipeline — chapter 20.
- The framework's earlier (now-removed) `EpisodicMemory` / `SemanticMemory` — that work is in git history; the design decision to remove it is what produced today's "apps own storage" rule.
- DBOSConfiguredInstance internals — chapter 24.

## Where to go next

→ [20-composition.md](20-composition.md) — combining patterns + units + capabilities into one pipeline.

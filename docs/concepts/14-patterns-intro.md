# 14. Patterns — when one Agent isn't enough

**Prerequisites:** [01-agents.md](01-agents.md), [07-capabilities.md](07-capabilities.md).

## Introduction

A single agent run handles a lot. The model decides what tools to call, when to stop, and what to return; capabilities govern the run; structured output guarantees the shape. Most "ask a question, get an answer" use cases fit cleanly into one `agent.run(...)`.

But some shapes don't. A research task that needs to extract facts from fifty document chunks doesn't fit in one prompt. A writing task that needs revision passes can't be done in one model call without skipping critic feedback. A multi-step plan with dependencies between steps is *structurally* not what a single agent does well. For these shapes, the framework ships **patterns** — reusable workflow scaffolds where each pattern composes one or more agents (and callables, and sub-workflows) into a higher-level computation.

This chapter introduces what a pattern is, the `Pattern` Protocol that every shipped pattern implements, how they get their durability for free via `@Durable.workflow`, and the three families (refinement, fan-out, planning) that the next four chapters drill into one at a time.

## The mental model

A pattern is a *typed, replay-safe composition of agents*. It takes typed input, runs a structured computation that involves one or more model calls, and produces typed output. From the outside, it looks just like an agent: `await pattern.run(input)`. From the inside, it's an orchestrated set of steps.

Three properties make a pattern more than "a function that calls some agents":

- **Typed at both ends.** Pattern Protocol is `Pattern[InT, OutT]`; you know what goes in, what comes out.
- **Durable by default.** The pattern's `run()` is wrapped in `@Durable.workflow` so a crash mid-execution resumes from the last completed step.
- **Per-step memoisation.** Internal steps (a map call, a critic invocation) are wrapped in `@Durable.step` so on replay, completed steps return cached results instead of re-running the LLM.

Together: patterns are the unit of "non-trivial agentic computation that needs to be production-safe."

## The `Pattern` Protocol

The minimal contract from `ballast.patterns.protocol`:

```python
@runtime_checkable
class Pattern(Protocol[InT, OutT]):
    name: ClassVar[str]
    async def run(self, input: InT) -> OutT: ...
```

That's it. One `name` (used by observability and the DBOS inspector) and one `run()` method. Everything else is up to the implementation.

The shipped patterns (`Reflection`, `MapReduce`, `DivergentConvergent`, `PlanAndExecute`) all satisfy this. Custom patterns just need to match the shape.

## Durability via `@Durable.workflow` + `DBOSConfiguredInstance`

The patterns all follow the same recipe:

```python
@Durable.dbos_class()
class Reflection(DBOSConfiguredInstance, Generic[InT, OutT]):
    name = "reflection"

    def __init__(self, *, writer, critic, max_iter=3, config_name=None):
        ...

    @Durable.workflow()
    async def run(self, task: InT) -> OutT:
        ...
```

Two pieces in concert:

- **`@Durable.dbos_class()`** + inheriting `DBOSConfiguredInstance` lets the pattern instance be durable-state-aware. DBOS needs a stable identifier per instance to memoise steps correctly across replays; `config_name` provides it.
- **`@Durable.workflow()` on `run()`** marks the whole pattern execution as a DBOS workflow. Mid-execution crash → DBOS replays from the last completed step.

The internal steps (`_map_one`, `_reduce`, etc.) are wrapped in `@Durable.step()` so their results are cached. On replay, those calls return the previous result instead of re-running the LLM. This is what makes durability cheap — only the *tail* re-executes, not the whole computation.

You don't have to think about this when *using* a pattern. You just `await pattern.run(input)` and the durability happens. You only need to know the mechanics when writing your own pattern (chapter 25).

## The three families

The patterns split into three families based on what kind of orchestration they do.

### Refinement (Reflection)

One agent produces a draft, a critic evaluates, a refiner revises. Loop until the critic accepts or you hit a budget. The agent equivalent of "write, then iterate."

Right when: output quality matters more than latency, and you can specify "good" via a critic.

### Fan-out (MapReduce, DivergentConvergent)

Multiple parallel agent runs followed by aggregation.

- **MapReduce** — one agent per chunk of input, then a reduce agent combines. Right for documents that don't fit in one context window.
- **DivergentConvergent** — multiple independent attempts (different prompts / models) followed by deduplication and synthesis. Right when you want *variety* before committing to an answer.

Both run their map / divergent phase concurrently (bounded by a semaphore), then collapse to a single output.

### Planning (PlanAndExecute)

One planner agent emits a typed DAG of steps; the framework dispatches each step (LLM, callable, CoALA unit, sub-workflow) according to its dependencies.

Right when: the task has clear sub-tasks with dependencies, and you'd rather pay one expensive planning call upfront than re-decide at every step.

## Apps usually compose, not subclass

The shipped patterns are *enough* for the vast majority of cases. You almost never need to subclass them — you instantiate, configure, and run. Composition is the dominant pattern:

```python
# Reflection inside MapReduce's reduce — refine the final summary
reflection_reduce = Reflection(
    writer=summarize_writer,
    critic=quality_critic,
    max_iter=3,
)

mr = MapReduce(
    map_agent=fact_extractor,
    reduce_step=lambda items: reflection_reduce.run(items),
    map_concurrency=8,
)

result = await mr.run(document_chunks)
```

That's the composability story — chapter 20 covers it in depth. Subclassing patterns is a chapter-25 topic (you write your own pattern only when none of the shipped shapes fit).

## Patterns vs Capabilities

They sound similar but they're different layers:

- **Capabilities** govern a single agent run. They observe / count / gate model requests within `agent.run(...)`.
- **Patterns** orchestrate *multiple* agent runs. They are the thing that *calls* `agent.run(...)`, potentially many times.

Capabilities go *inside* the agents that patterns use. A `Reflection` instance can have a `BudgetGuard` capability attached to its writer agent, its critic agent, and its refiner agent independently. The pattern doesn't know or care.

The simplest way to remember: capabilities are "in-flight"; patterns are "around the flight."

## Patterns inside a higher-level workflow

A `PlanAndExecute` step can itself be a `MapReduce` pattern. A `Reflection`'s critic could call a `DivergentConvergent` to brainstorm critique angles. Patterns nest cleanly because every pattern is itself a `@Durable.workflow` — and DBOS handles nested workflows correctly.

The constraint: nested workflows must have unique IDs. The patterns provide a `config_name` parameter that becomes part of the identifier, so two `Reflection` instances in the same outer workflow don't collide.

## Common mistakes

- **Treating a pattern as a one-time service object.** Patterns can be instantiated once at import time and reused across requests — they're stateless after construction. Don't construct one per request.
- **Wrapping a pattern in your own `@Durable.workflow`.** Patterns are already workflows. Wrapping them in another workflow creates a nested-workflow situation; not always wrong but rarely what you want. If your top-level entry needs to be a workflow, call patterns from inside it as plain `await pattern.run(...)`.
- **Forgetting `config_name` when running multiple instances of the same pattern.** Two `Reflection()` instances without distinct `config_name` will collide on DBOS state. Set `config_name="research_summary"` for one, `config_name="email_draft"` for the other.
- **Reaching for a pattern when one agent run would do.** Patterns add latency (multiple LLM calls) and complexity. If a single agent run produces an acceptable answer, don't add a pattern.

## What this chapter did NOT cover

- The specific behavior of each pattern — chapters 15-18.
- How CoALA units fit into patterns (`UnitStep` in PlanAndExecute) — chapter 19.
- Writing your own pattern — chapter 25.
- DBOS internals — chapter 24.

## Where to go next

→ [15-reflection.md](15-reflection.md) — the refinement family, starting with Writer-Critic-Refiner.

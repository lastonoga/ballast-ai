# How to build a CoALA Unit

**Problem:** You have a piece of memory-aware logic — extract relevant context from a knowledge base, reason over it, optionally persist what you learned. You want to deploy it as **a tool** for an agent, OR **a workflow** for a durable pipeline, OR **a capability** that auto-fires on every agent run. Same logic, three runtimes.

**Solution:** Subclass `CoALABase`. Implement `retrieve` and `act`. Get `observe` (identity) and `learn` (no-op) for free. Choose your deployment via `as_tool` / `as_workflow` / `as_capability` adapter.

## Minimum: a research-summarize unit

```python
from dataclasses import dataclass, field
from ballast.coala import CoALABase, CoALAUnit


@dataclass
class ResearchQuery:
    user_query: str


@dataclass
class ResearchObservation:
    intent: str


@dataclass
class ResearchContext:
    related_notes: list = field(default_factory=list)


@dataclass
class ResearchSummary:
    title: str
    body: str


class ResearchSummarize(CoALABase[
    ResearchQuery,         # InT
    ResearchObservation,   # ObsT
    ResearchContext,       # ContextT
    ResearchSummary,       # OutT
]):
    """Find related notes, summarize them."""

    async def observe(self, q: ResearchQuery) -> ResearchObservation:
        return ResearchObservation(intent=q.user_query)

    async def retrieve(self, obs: ResearchObservation) -> ResearchContext:
        from notes_app.repositories.note import notes_repo
        related = await notes_repo.search(obs.intent)
        return ResearchContext(related_notes=related[:10])

    async def act(
        self, obs: ResearchObservation, ctx: ResearchContext,
    ) -> ResearchSummary:
        if not ctx.related_notes:
            return ResearchSummary(
                title=f"No prior research on {obs.intent!r}",
                body="No matching notes found.",
            )
        bullets = "\n".join(f"- {n.title}" for n in ctx.related_notes)
        return ResearchSummary(
            title=f"Research: {obs.intent}",
            body=f"Found {len(ctx.related_notes)} prior notes:\n{bullets}",
        )

    # learn() defaults to no-op — override if you want to persist insights
```

That's the whole unit. Four phases, two overridden, two default. Now pick a runtime.

## Deploy as an agent tool

```python
from ballast.coala import as_tool

agent = Agent(
    model="openai:gpt-4o",
    tools=[as_tool(ResearchSummarize())],
)
# LLM sees one tool named "ResearchSummarize" whose input schema is
# derived from observe()'s parameter (ResearchQuery in this case).
# When the LLM calls it, all four phases run in sequence.
```

## Deploy as a durable workflow

```python
from ballast.coala import as_workflow

flow = as_workflow(ResearchSummarize())
summary = await flow(ResearchQuery(user_query="ML deployment patterns"))

# Internally: @Durable.workflow wraps it; each phase is a @Durable.step.
# On replay, completed phases skip; only the unfinished tail re-runs.
```

## Deploy as an agent capability (auto-fires per request)

```python
from ballast.coala import as_capability

agent = Agent(
    model=...,
    capabilities=[as_capability(ResearchSummarize())],
)
# observe + retrieve fire in before_model_request — context is added to ctx.deps
# act = the agent's normal run (NOT the unit's act — capability skips that)
# learn fires in after_run with the agent's result
```

> Note: as_capability uses observe + retrieve as a context-enrichment step before the agent's own run, then `learn` after. The unit's `act` is bypassed because the AGENT is now the actor.

## Add a learn phase (persist what happened)

```python
import logging

class ResearchSummarize(CoALABase[...]):
    async def observe(self, q): ...
    async def retrieve(self, obs): ...
    async def act(self, obs, ctx): ...

    async def learn(
        self,
        obs: ResearchObservation,
        ctx: ResearchContext,
        output: ResearchSummary,
    ) -> None:
        # Apps wire their own storage:
        await my_episodes.write({
            "intent": obs.intent,
            "found_n": len(ctx.related_notes),
            "title": output.title,
            "timestamp": datetime.now(UTC),
        })
        logging.info("research_summarize.learn intent=%s", obs.intent)
```

Apps own all storage. Framework just calls `learn(observation, context, output)` after every successful act.

## Compose with PlanAndExecute

CoALA units plug into `PlanAndExecute` as `UnitStep`:

```python
from ballast import PlanAndExecute
from ballast.patterns.plan_execute import StepRegistry

registry = StepRegistry.with_defaults()
registry.register_unit("research", ResearchSummarize())
registry.register_unit("publish", PublishUnit())

planner = Agent(
    model="openai:gpt-4o",
    output_type=Plan,
    system_prompt="Available unit kinds: research, publish.",
)

pattern = PlanAndExecute(planner=planner, registry=registry)
outputs = await pattern.run({"topic": "ML safety"})
# Planner emits a DAG; PlanAndExecute dispatches each UnitStep through your registered unit
```

## Why this abstraction matters

Without CoALA, you'd write the same logic three times — once as a tool body, once as a workflow function, once inside a capability hook. With CoALA, **write the logic once; pick the runtime via adapter.**

The four phases also force a clean separation:
- `observe` — parse raw input into typed observation
- `retrieve` — pull relevant context (your storage, your call)
- `act` — reason + produce output
- `learn` — persist for next time

If a unit doesn't need one phase, default it. If it needs custom storage, override `retrieve` + `learn`. The framework doesn't prescribe what storage to use.

## Caveats

- **`act` is bypassed in `as_capability`** — there the agent IS the actor; the unit only contributes context (observe + retrieve) and aftermath (learn).
- **Don't put expensive blocking I/O in `observe`** — it runs every time the unit fires. Cheap parsing only.
- **Generics are advisory** — `CoALABase[InT, ObsT, ContextT, OutT]` is for IDE help; not enforced at runtime. Apps can use `Any` if the types are dynamic.
- **`as_workflow` uses DBOSConfiguredInstance + per-phase steps.** Each phase is memoised on workflow replay. The unit instance is stored on the configured instance — it's never pickled per call. So units can close over agents / HTTP clients / locks safely.

## Related

- [deploy-coala-unit-as-tool.md](deploy-coala-unit-as-tool.md) — `as_tool` deep dive
- [deploy-coala-unit-as-workflow.md](deploy-coala-unit-as-workflow.md) — `as_workflow` + DBOS replay semantics
- [compose-coala-units-in-plan.md](compose-coala-units-in-plan.md) — `UnitStep` in PlanAndExecute
- Reference: `reference/coala/coala-unit-protocol.md`
- Explanation: [article-pain-points.md](../../explanation/article-pain-points.md) #25

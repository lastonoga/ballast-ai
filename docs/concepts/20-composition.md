# 20. Composition — combining everything

**Prerequisites:** chapters 14–19.

## Introduction

You've now met every primitive the framework ships: agents with tools and typed outputs, capabilities that govern runs, patterns (refinement, fan-out, planning), cognitive units, and the resilience layer. Each one solves a single concern. The question this chapter answers is: how do they actually fit together into one coherent pipeline?

The answer is "they compose naturally because each layer has a clean boundary." But composition isn't free — nest too deep and your stack trace becomes a maze; flatten too eagerly and you lose the per-layer guarantees that made the primitives valuable in the first place. This chapter walks through the common composition shapes, the replay semantics for nested workflows, and a worked end-to-end example.

## The mental model

Three layers of composition, from outside in:

```
┌──────────────────────────────────────────────────────────┐
│  Outer @Durable.workflow (your top-level entry)          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Pattern (PlanAndExecute / MapReduce / ...)        │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  Agent run (agent.run inside a step)         │  │  │
│  │  │  ┌────────────────────────────────────────┐  │  │  │
│  │  │  │  Capabilities (BudgetGuard, drift...)  │  │  │  │
│  │  │  └────────────────────────────────────────┘  │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

- The outermost workflow owns the durability — your top-level entry point gets a workflow ID and crash recovery.
- Patterns are themselves workflows, nested inside the outer one. They give you the orchestration shape (planning, fan-out, refinement).
- Agents run inside patterns (or directly in your workflow). They give you the LLM call.
- Capabilities govern each agent run. They give you per-run safety.
- Resilience (CircuitBreaker) wraps any of the above layers — you choose where it goes.

The layers are decoupled. A capability doesn't know which pattern uses the agent it's attached to. A pattern doesn't know whether it's the top level or nested. The outer workflow doesn't know the inner patterns' internals.

## Pattern inside pattern

The simplest composition: a `Reflection` as the reduce step of a `MapReduce`.

```python
refine_summary = Reflection(
    writer=summary_writer,
    critic=quality_critic,
    max_iter=3,
    config_name="summary_refinement",
)

mr = MapReduce(
    map_agent=fact_extractor,
    reduce_step=lambda items: refine_summary.run(items),
    map_concurrency=8,
    config_name="fact_to_summary",
)

summary = await mr.run(document_chunks)
```

What this gives you: parallel extraction across chunks (the map), iterative refinement of the final summary (the reduce). Each is a `@Durable.workflow`; DBOS handles the nesting and gives each its own ID.

The unique `config_name` per pattern is what keeps their DBOS state separate. Without it, two instances would collide.

## DivergentConvergent inside MapReduce

```python
dc = DivergentConvergent(
    branches=brainstorm_branches,
    synthesizer=synthesizer_agent,
    hypotheses=lambda env: env.ideas,
    config_name="per_chunk_brainstorm",
)

mr = MapReduce(
    map_step=lambda chunk: dc.run(chunk),
    reduce_step=aggregate_ideas,
    map_concurrency=4,         # lower — DC is heavier than a single agent call
    config_name="brainstorm_aggregation",
)
```

For each chunk, run a full divergent-convergent brainstorm; aggregate the synthesized outputs. Useful for "give me ten distinct ideas per section" pipelines.

## CoALA unit as a PlanAndExecute step

```python
research_unit = ResearchSummarize(notes_repo=..., summarizer_agent=...)

registry = StepRegistry.with_defaults()
registry.register_unit("research", research_unit)
registry.register_agent("publish", publish_agent)

pe = PlanAndExecute(planner=planner, registry=registry)

result = await pe.run({"topic": "ML deployment"})
```

The planner emits a `Plan` where one of the steps is `kind="unit", unit_name="research"`. The framework dispatches the unit's full 4-phase lifecycle (`observe → retrieve → act → learn`) when that step fires. Downstream steps see the unit's `act` output via `dep_outputs`.

This is why CoALA units are first-class citizens: they fit into the planning shape without translation.

## CoALA unit as a capability on a pattern's agent

```python
# Make every model call in this agent auto-retrieve relevant prior context
researcher_with_memory = Agent(
    model="openai:gpt-4o",
    system_prompt="...",
    capabilities=[
        BudgetGuard(max_iterations=15),
        as_capability(memory_lookup_unit),  # observe + retrieve before; learn after
    ],
)

# Then use this agent inside a pattern
mr = MapReduce(
    map_agent=researcher_with_memory,
    reduce_agent=synthesizer,
    map_concurrency=8,
)
```

The capability fires on *every* model call inside the agent — including each step of the map. So a 100-chunk map run does 100 retrievals + 100 learns. Make sure your unit is cheap enough for that.

## Resilience wrapping at every layer

`CircuitBreaker` adapts to whatever layer you want to protect.

```python
# Around an agent (cross-run signal for that agent)
agent_breaker = CircuitBreaker(threshold_factory=lambda: Consecutive(5))
robust_agent = Agent(..., capabilities=[as_capability(agent_breaker)])

# Around a PlanAndExecute step (per-step bucket)
step_breaker = CircuitBreaker(threshold_factory=lambda: Consecutive(3))
guarded_step = as_step(step_breaker, LLMStep(...))
registry.register_step("guarded_llm", guarded_step)

# Around an entire workflow
workflow_breaker = CircuitBreaker(threshold_factory=lambda: WindowedRate(0.3, window=...))

@as_workflow_decorator(workflow_breaker)
@Durable.workflow
async def my_workflow(...): ...
```

Multiple breakers at multiple layers don't conflict — they catch different things at different scopes.

## The outer workflow as orchestrator

Your top-level entry is often a thin `@Durable.workflow` that just sequences calls:

```python
@Durable.workflow
async def research_and_publish(topic: str) -> dict:
    # Step 1: Plan + execute research
    plan_result = await pe.run({"topic": topic})
    research = plan_result["research"]

    # Step 2: Refine via Reflection
    refined = await refine_summary.run(research)

    # Step 3: HITL approval
    approved = await approval_channel.request(refined)
    if approved.decision != "approve":
        return {"status": "rejected", "feedback": approved.feedback}

    # Step 4: Publish (external API call)
    publication = await publish_via_circuit_breaker(refined)

    return {"status": "published", "url": publication.url}
```

This is the most common shape for non-trivial flows. The outer workflow is short — just orchestration. The work happens inside patterns and units that are themselves workflows.

## Replay semantics for nested workflows

DBOS handles nested workflows correctly: each call to `@Durable.workflow` gets its own ID, the parent records the child completion in its state, and on parent replay the child results come from cache instead of re-running.

What that means in practice:

- Outer workflow crashes at step 4 (publish): on resume, steps 1-3 (plan, refine, approval) are from cache. Only the publish step re-runs.
- A pattern (say `MapReduce`) inside step 1 crashes halfway: that pattern's own per-step memoisation kicks in. The outer workflow doesn't re-run anything before step 1.

Two layers of crash recovery: workflow-level (the outer) and step-level (the pattern's internal `@Durable.step` results).

## When to flatten, when to nest

Two heuristics:

**Flatten when:** the two layers don't add independent value. If you have `@Durable.workflow` outer that just calls `MapReduce.run(...)` and nothing else, the outer adds no information — replace the outer entirely; let your route handler call `mr.run(...)` directly.

**Nest when:** each layer is doing genuine work. A research pipeline that plans (PlanAndExecute) and then refines (Reflection) and then publishes (HITL) is three layers because each layer is essential — flattening would force you to manually write the orchestration that PlanAndExecute already gives you for free.

The smell test: if you can't justify each level in one sentence ("PlanAndExecute decomposes into steps; Reflection improves the final summary; HITL gates publication"), some of them are noise.

## End-to-end example: research → critique → publish-with-approval

The full pipeline:

```python
# === Building blocks ===

# Tools registered in StepRegistry
registry.register_agent("extractor", fact_extractor_agent)   # extracts facts from chunks
registry.register_unit("research", research_unit)            # CoALA unit: research + retrieve prior
registry.register_callable("aggregate", aggregate_fn)        # plain function

# Reflection for the writeup
write_with_review = Reflection(
    writer=writer_agent,
    critic=quality_critic_judge,
    max_iter=3,
    config_name="writeup",
)

# CircuitBreaker around the external publish API
publish_breaker = CircuitBreaker(
    threshold_factory=lambda: Consecutive(5),
    fallback=EscalateToHITL(
        channel=publish_failure_channel,
        card_factory=publish_failure_card,
    ),
)

# HITL approval channel
approval_channel = UICardChannel(payload_type=PublishApprovalPayload)

# PlanAndExecute over the research + aggregate phase
pe = PlanAndExecute(planner=planner_agent, registry=registry, max_parallel=4)


# === Top-level orchestrator ===

@Durable.workflow
async def research_and_publish(topic: str, user_id: str) -> dict:
    current_user_id.set(user_id)

    # 1. Plan + execute research
    plan_result = await pe.run({"topic": topic})

    # 2. Refine the synthesized writeup
    refined = await write_with_review.run(plan_result["aggregate"])

    # 3. HITL approval
    verdict = await approval_channel.request(
        PublishApprovalPayload(draft=refined, topic=topic),
        timeout=timedelta(minutes=10),
    )

    if verdict.decision != "approve":
        return {"status": "rejected", "feedback": verdict.feedback}

    # 4. Publish via circuit breaker
    final = verdict.modified or refined   # use human edits if any
    publication = await publish_breaker.call(lambda: publish_api.create(final))

    return {"status": "published", "url": publication.url}
```

Five primitives composed: CoALA unit, plan-and-execute, reflection, HITL, circuit breaker. Each owns one concern; the outer workflow strings them together.

What this gives you:

- Durable. Crash at step 3 (waiting for human) → resume after restart with the approval card still open.
- Replay-safe. Crash at step 4 → steps 1-3 are cached; only step 4 re-runs.
- Observable. Each layer is its own span in logfire. You see "PlanAndExecute took 12s, Reflection took 8s, HITL waited 4 minutes, publish took 0.3s."
- Resilient. Publish API outage → breaker opens → fallback escalates to a human instead of silently failing.

## Common mistakes

- **Wrapping a pattern in another `@Durable.workflow` for no reason.** The pattern is already a workflow. Adding another layer means more nesting, more state to track, no functional gain.
- **Capabilities on the *wrong* agent.** If you want the budget cap to bound the *whole pipeline*, that's not what capabilities do — capabilities bound *agent runs*. Use a custom check inside your outer workflow, or stack budgets at every agent.
- **CircuitBreaker around `Reflection.run()` with `is_failure_exc=()`.** Default `is_failure_exc=(Exception,)` will count `ReflectionExhausted` as a failure — usually fine. But if you specifically don't want exhausted runs to count, narrow the tuple.
- **Nested workflows without `config_name`.** Two `Reflection` instances in the same outer workflow without unique `config_name` collide on DBOS state.
- **Too-deep nesting.** Five layers of pattern nesting becomes unreadable. If you're at four, ask whether two of them are doing the same thing and could merge.

## What this chapter did NOT cover

- The specifics of HITL channels — chapter 21.
- Observability per-layer span attributes — chapter 22.
- Eval-driven CI gates that run the whole pipeline — chapter 23.
- Custom step / pattern / channel implementations — chapter 25.

## Where to go next

→ [21-human-in-the-loop.md](21-human-in-the-loop.md) — the human surfaces.

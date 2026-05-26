# 15. Reflection

**Prerequisites:** [14-patterns-intro.md](14-patterns-intro.md), [06-confidence-and-quality.md](06-confidence-and-quality.md).

## Introduction

A single LLM call gives you whatever the model produces in one pass. Sometimes that's fine. Sometimes it's a first draft that would benefit from another look — the model wrote something plausible but missed an obvious issue, or used a worse phrasing than it could have, or included a detail that's subtly wrong. Humans iterate; an agentic system that doesn't iterate ships every first draft.

`Reflection` is the framework's refinement pattern: a writer produces a draft, a critic evaluates it, and (if the critic isn't satisfied) the writer revises with the critic's feedback in mind. Loop until the critic accepts or you hit `max_iter`. The output is the draft that passed.

This pattern is high-leverage when output quality matters, and high-cost when it doesn't — every iteration is at least two LLM calls. This chapter covers when to use it, how the `Reflection` class is constructed, the `as_critique` adapter for using non-LLM critics, and how it composes with `Scored[T]` and other patterns.

## The mental model

Three roles, two of which can be the same agent:

- **Writer** — produces a draft (and revisions based on critique).
- **Critic** — evaluates the latest draft. Returns a `Critique` (`passed: bool`, `issues`, `suggestions`).
- **Refiner** — usually the writer, called again with the critique attached. The framework's `Reflection` collapses this into the writer role.

The loop:

```
draft = writer(task, critiques=[])
for i in range(max_iter):
    critique = critic(draft)
    if critique.passed:
        return draft
    draft = writer(task, critiques=[..., critique])
return draft  # or raise ReflectionExhausted
```

Three iterations is the typical sweet spot. Going higher rarely helps (the critic and writer settle into a stable disagreement); going lower means you might ship an unrevised first draft.

## The simplest case

```python
from ballast import Reflection
from ballast.capabilities.helpers import Critique

async def write(task: str, critiques: list[Critique]) -> str:
    # ... call writer agent with task + any prior critiques
    return draft

async def critique(draft: str) -> Critique:
    # ... call critic agent
    return Critique(passed=True, issues=[], suggestions=[])

reflection = Reflection(
    writer=write,
    critic=critique,
    max_iter=3,
    config_name="research_summary",
)

final_draft = await reflection.run("Summarize the ML deployment project")
```

`config_name` matters when you run multiple `Reflection` instances in the same outer workflow — DBOS uses it to keep their state separate. For standalone use you can omit it.

## Using an `LLMJudge` as critic

The cleanest pattern: the critic is an `LLMJudge` (chapter 23). The framework auto-adapts:

```python
from ballast.evals import LLMJudge

critic = LLMJudge(
    model="openai:gpt-4o-mini",
    rubric="The summary should be factually accurate, under 100 words, and well-structured.",
    output_type=Critique,
)

reflection = Reflection(writer=write, critic=critic, max_iter=3)
```

`Reflection` detects that `critic` is an `LLMJudge`-shaped object and wraps it to fit the critic interface. You don't have to write the bridge.

## Using `as_critique` for non-LLM critics

Sometimes the "critic" is just a deterministic check — does the output match a regex, pass a schema validation, score above a threshold? You don't need an LLM for that. `as_critique` lifts a plain function into the critic interface:

```python
from ballast.capabilities.helpers import as_critique

async def check_length(draft: str) -> Critique:
    if len(draft) > 1000:
        return Critique(
            passed=False,
            issues=["Draft is too long"],
            suggestions=["Cut to under 1000 characters"],
        )
    return Critique(passed=True)

reflection = Reflection(
    writer=write,
    critic=as_critique(check_length),
    max_iter=3,
)
```

Fast, free, deterministic. Mix and match: you can chain checks (one for length, one for tone, one for grounding) and have your writer iterate against them. For mixed deterministic + LLM critique, build a custom critic function that calls both and merges the verdicts.

## The iteration cap is non-negotiable

`max_iter=3` is the default and it's there for a reason. Without it, two bad scenarios:

- **Critic always fails.** Your writer keeps revising forever. Each iteration is two LLM calls; ten iterations is twenty calls and you've lost an order of magnitude in cost.
- **Writer and critic disagree.** The critic wants A; the writer keeps producing B. The loop never converges. This is more common than you'd think — the two models often have subtly different priors.

When the cap is hit, `Reflection` raises `ReflectionExhausted(iterations, last_critique, last_draft)`. Catch it and decide:

```python
from ballast.patterns.reflection import ReflectionExhausted

try:
    final = await reflection.run(task)
except ReflectionExhausted as exc:
    # The critic never approved. Two choices:
    #  - Ship the last draft anyway (warn + log)
    #  - Hard-fail to the user
    logger.warning("reflection exhausted after %d", exc.iterations)
    return exc.last_draft   # or raise
```

The second pattern (hard-fail) is right for high-stakes outputs (financial advice, medical summaries). The first is right for "this is going to be reviewed by a human anyway."

## Composing with `Scored[T]`

`Scored[Critique]` is a sweet spot for `accept_if`-style early termination:

```python
class ScoredCritique(BaseModel):
    critique: Critique
    confidence: Confidence

critic = LLMJudge(
    model=...,
    rubric="...",
    output_type=Scored[Critique],
)

# Custom acceptance: pass if critic isn't confident in finding issues
def is_acceptable(critique: Scored[Critique]) -> bool:
    return critique.value.passed or critique.confidence == "low"
```

When the critic says "I see issues but I'm not sure," accept and move on. Three iterations of a high-confidence "fix X" is productive; five iterations of low-confidence nitpicking is the model thrashing.

## Cost trade-off

Per iteration you pay:

- 1 writer call (initial or refined)
- 1 critic call

For 3 iterations: up to 6 LLM calls. The minimum: 2 calls (write → critic accepts on first pass).

Compare against the alternative — one really careful single agent run that includes a self-check prompt. Sometimes that's enough. Reflection is for when self-check isn't reliable enough and you want a separate-model critic.

## Embedding inside MapReduce / PlanAndExecute

`Reflection` is just a `Pattern`. It composes naturally:

```python
# A MapReduce where the reduce step is a Reflection
mr = MapReduce(
    map_agent=fact_extractor,
    reduce_step=lambda items: reflection.run(items),  # Reflection over the items
    map_concurrency=8,
)

# A PlanAndExecute step that's a Reflection
registry.register_callable("refine", lambda x: reflection.run(x))
```

The outer pattern doesn't see the iteration; from its perspective, "refine" is just an async function that returns the final draft. Durability nests cleanly — the outer pattern's `@Durable.workflow` contains the inner Reflection's `@Durable.workflow`, and DBOS handles the lifecycle.

## Progress events

`Reflection` emits typed events to the `reflection_progress` signal so a UI can show iteration-by-iteration progress:

- `ReflectionEvent(type="draft", iteration=n, draft=...)` — initial or revised draft
- `ReflectionEvent(type="critique", iteration=n, critique=...)` — critic verdict
- `ReflectionEvent(type="passed", iteration=n, draft=...)` — critic accepted, loop done
- `ReflectionEvent(type="refine", iteration=n)` — entering revision
- `ReflectionEvent(type="exhausted", iteration=n)` — hit max_iter

Subscribe to the signal from your streaming router to surface the iteration to the user. ("Refining draft 2 of 3...")

## Common mistakes

- **Setting `max_iter` very high.** Past 5 you're rarely getting better results, just paying more. If you genuinely need more, the writer + critic combination is mis-tuned.
- **Critic too strict.** A critic that *always* finds something to improve makes `Reflection` an iteration toilet bowl. Calibrate: run the critic on 20 "definitely good" drafts; it should pass them.
- **Writer ignoring critique.** If the writer agent's prompt doesn't actually reference the prior critiques, you're paying for iterations that produce identical drafts. Make sure the writer prompt template includes the critique feedback.
- **Same model for writer and critic.** Sometimes fine. Often bad — the critic shares the writer's blind spots. Different model families (or even different temperatures) help.
- **Forgetting `config_name` for parallel reflections.** Two `Reflection` instances in the same workflow without unique names collide.

## What this chapter did NOT cover

- The `LLMJudge` constructor — chapter 23.
- `MapReduce` / `DivergentConvergent` — chapters 16, 17.
- Writing a custom pattern from scratch — chapter 25.
- Frontend rendering of `ReflectionEvent` — chapter 22.

## Where to go next

→ [16-mapreduce.md](16-mapreduce.md) — the fan-out family for documents bigger than context.

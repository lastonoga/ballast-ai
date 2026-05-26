# How to use Reflection (Writer-Critic-Refiner)

**Problem:** Your agent's first draft is OK but not great. You want a feedback loop: critic spots issues → refiner fixes them → repeat until quality threshold OR iteration cap. Without a cap, the loop can rage forever (loop-happiness) burning tokens.

**Solution:** `Reflection` pattern — three roles (writer / critic / refiner), capped iterations, optional `accept_if` predicate to short-circuit when good-enough.

## Minimum

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from ballast import Reflection


class Critique(BaseModel):
    issues: list[str]
    severity: int    # 0..3


class Article(BaseModel):
    title: str
    body: str


writer = Agent(
    model="openai:gpt-4o",
    system_prompt="Write a 500-word article on the given topic.",
    output_type=Article,
)

critic = Agent(
    model="openai:gpt-4o-mini",
    system_prompt="Critique the article. List concrete issues; assign severity 0..3.",
    output_type=Critique,
)

refiner = Agent(
    model="openai:gpt-4o",
    system_prompt="Revise the article addressing each issue in the critique.",
    output_type=Article,
)


reflection = Reflection(
    writer=writer,
    critic=critic,
    refiner=refiner,
    max_iterations=3,
    output_type=Article,
)

final = await reflection.run("Future of long-context LLMs")
print(final.title)
print(final.body)
```

The flow: writer → critique → refiner → critique → refiner → critique → final. Three refiner rounds max.

## Short-circuit when good-enough

```python
reflection = Reflection(
    writer=writer,
    critic=critic,
    refiner=refiner,
    max_iterations=5,
    accept_if=lambda critique: critique.severity == 0,    # no issues → stop
    output_type=Article,
)
```

`accept_if` runs on each critic verdict. When `True`, the loop exits and current draft is returned. Otherwise refiner runs and another critique iterates.

## Use `Scored[T]` for confidence-aware acceptance

```python
critic = Agent(model=..., output_type=Scored[Critique])

reflection = Reflection(
    writer=writer,
    critic=critic,
    refiner=refiner,
    accept_if=lambda critique: critique.confidence == "low",
    # accept the draft if critic isn't even confident in finding issues
)
```

When the critic is uncertain, that's signal the draft is at-least-OK. Combined with `max_iterations` it gives "fail-open after N rounds".

## TypedLoopGuard protects against silent convergence

Reflection bakes in automatic loop detection via `TypedLoopGuard` (built-in capability) — if the refiner stops making meaningful changes between iterations, the loop short-circuits even before `max_iterations`. No config needed.

For extra protection, add `BudgetGuard` to the underlying agents:

```python
budget = BudgetGuard(max_iterations=15, max_input_tokens=30_000)

writer = Agent(model=..., output_type=Article, capabilities=[budget])
critic = Agent(model=..., output_type=Critique, capabilities=[budget])
refiner = Agent(model=..., output_type=Article, capabilities=[budget])
```

Now if any single LLM call inside the reflection loop loops by itself, BudgetGuard catches it.

## Use as a critique adapter for other patterns

`as_critique` lets a Reflection-style judge become a single function call:

```python
from ballast import as_critique

critique_fn = as_critique(critic=critic, refiner=refiner)
verdict = await critique_fn(draft)
# verdict.passed: bool, verdict.refined: Article, verdict.issues: list[str]
```

Useful for embedding a critique-loop inside a `MapReduce.reduce_step` or `PlanAndExecute.Step`.

## Persist intermediate drafts

The reflection loop doesn't auto-persist intermediate drafts. If you want them for debugging / human review:

```python
class TrackedReflection(Reflection):
    def __init__(self, *args, persist: Callable, **kwargs):
        super().__init__(*args, **kwargs)
        self._persist = persist

    async def run(self, brief: str) -> Article:
        draft = await self._writer.run(brief)
        await self._persist({"iteration": 0, "draft": draft.output})
        for i in range(1, self._max_iterations + 1):
            critique = await self._critic.run(draft.output.model_dump_json())
            if self._accept_if and self._accept_if(critique.output):
                return draft.output
            draft = await self._refiner.run(...)
            await self._persist({"iteration": i, "draft": draft.output, "critique": critique.output})
        return draft.output
```

Or simpler: wire `JudgeAfterRun` on each underlying agent so logfire traces every step.

## Caveats

- **Don't use Reflection on tool-calling agents.** It's designed for typed-output refinement, not multi-step reasoning chains. For those, use `PlanAndExecute`.
- **Three agents = 3× the prompts to maintain.** Keep system prompts short and focused: writer = "do the thing", critic = "find issues", refiner = "fix issues per feedback".
- **Cost adds up fast.** Each iteration is writer + critic + refiner = 3 LLM calls. Set `max_iterations` deliberately — usually 2-3 is enough.
- **`accept_if` is sync.** Async predicates aren't supported in first cut; if you need an async accept-check, wrap it in `asyncio.run_coroutine_threadsafe` or move the check inside the critic prompt itself.

## Related

- [add-budget-guard.md](add-budget-guard.md) — protect each underlying agent
- [add-confidence-to-tool-outputs.md](add-confidence-to-tool-outputs.md) — `Scored[Critique]` for confidence-aware acceptance
- [use-mapreduce-for-rag.md](use-mapreduce-for-rag.md) — when you need fan-out, not iteration
- Reference: `reference/patterns/reflection.md`
- Explanation: [article-pain-points.md](../explanation/article-pain-points.md) #12

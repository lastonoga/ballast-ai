# 6. Confidence and quality — `Scored[T]`

**Prerequisites:** [03-structured-output.md](03-structured-output.md).

## Introduction

A model that extracts facts from a document, or summarizes search results, or proposes actions, has no built-in way to tell you how *confident* it is. The same response shape comes back whether the model is sure or guessing. Downstream code can't decide whether to ship the response to the user, escalate to a human, retry, or just log and move on.

The fix is structural: wrap every quality-sensitive output in a generic that carries `value + rationale + confidence`. The model is forced (by the schema) to think before committing to a confidence label — that single requirement is what dramatically reduces the mean-reversion problem that plagues numeric quality scores.

The framework ships this as `Scored[T, ConfidenceT]`. Defaults are friendly: `Confidence = Literal["low", "medium", "high"]` (named labels, not numbers, because models regress to the middle on numeric scales). Apps that need a different shape (binary safe/unsafe, numeric 1-5) override `ConfidenceT`.

This chapter covers how `Scored[T]` works, how to use it as an agent output, how it composes with `MapReduce` and `CircuitBreaker`, and the discipline of treating `rationale` as mandatory.

## The mental model

`Scored[T]` is a small pydantic model with three fields:

- **`value: T`** — the actual payload (a fact, a summary, a tool call result)
- **`rationale: str`** — one-sentence justification of the confidence label
- **`confidence: ConfidenceT`** — the label itself (default: `"low" | "medium" | "high"`)

That's it. No magic. The power isn't in the wrapper — it's in *requiring* the rationale field, which forces the LLM into a chain-of-thought step before assigning the label. Skip the rationale field and the model trends to "medium" for everything; require it and the label tracks the model's actual epistemic state.

`frozen=True` is set on the model, so instances are immutable after construction. Mutating quality signals in downstream code would mask the original model's assessment; we don't want that.

## The simplest case

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from ballast import Scored

class Fact(BaseModel):
    text: str
    source_url: str

extractor = Agent(
    model="openai:gpt-4o-mini",
    system_prompt=(
        "Extract one key fact from the user-provided document. "
        "Provide a one-sentence rationale and a confidence label "
        "('high' if explicitly stated, 'medium' if implied, 'low' if guessed)."
    ),
    output_type=Scored[Fact],
)

result = await extractor.run(document_text)
print(result.output.value.text)         # the fact
print(result.output.rationale)          # the justification
print(result.output.confidence)         # 'low' | 'medium' | 'high'
```

The JSON Schema sent to the model has three fields and a constrained `confidence` enum. The model produces something like:

```json
{
  "value": {"text": "Q3 revenue grew 12%", "source_url": "https://..."},
  "rationale": "Explicitly stated in the document's Q3 financial summary section.",
  "confidence": "high"
}
```

pydantic validates, you get a typed `Scored[Fact]` back.

## Why named labels, not numbers

The first instinct is to use a 1-10 scale: "more granular, easier to threshold." Don't. LLMs systematically regress to the middle on numeric scales: average score lands around 5-7 regardless of actual variability in input quality. You end up with a number that looks objective but isn't.

Named labels work better for two reasons. First, the discrete choices (`"low" / "medium" / "high"`) force the model to commit to one of three buckets rather than splitting hairs on a continuous scale. Second, the labels carry semantic meaning the model recognizes from training — "high confidence" has thousands of training examples; "score: 8.3" doesn't.

The default `Literal["low", "medium", "high"]` is what you should use unless you have a specific reason not to. Three buckets are enough for almost any downstream filtering decision.

## Filtering and ranking helpers

Three small functions cover the common downstream uses:

```python
from ballast.quality.scored import (
    aggregate_by_confidence,
    filter_by_min_confidence,
    rank_by_confidence,
)

items: list[Scored[Fact]] = await asyncio.gather(*(
    extractor.run(doc).then(lambda r: r.output)
    for doc in documents
))

# Drop low-confidence items
kept = filter_by_min_confidence(items, "medium")

# Sort high → low; tie-break by source_url
ranked = rank_by_confidence(kept, secondary_key=lambda it: it.value.source_url)

# Or bucket for separate downstream handling
buckets = aggregate_by_confidence(items)
print(f"{len(buckets['high'])} high-confidence facts")
print(f"{len(buckets['medium'])} medium-confidence facts")
print(f"{len(buckets['low'])} low-confidence facts")
```

These helpers are hardcoded to the default `Confidence` Literal — they use `label_rank("low") == 0 < label_rank("medium") == 1 < label_rank("high") == 2`. If you override `ConfidenceT` (see below), you'll write your own helpers.

There's also `label_to_float(label) -> float` that maps `"low" → 0.33`, `"medium" → 0.66`, `"high" → 1.0` for metrics dashboards that want a numeric proxy.

## Composing with `MapReduce`

The article's "Map-phase pattern" is the canonical use case for `Scored[T]`. Map workers extract facts from chunks, each tagged with rationale + confidence. The reducer filters low-confidence noise *before* synthesizing — high-quality output, lower token cost.

```python
from ballast import MapReduce, Scored
from ballast.quality.scored import filter_by_min_confidence, rank_by_confidence

extractor = Agent(model=..., output_type=Scored[Fact])
synthesizer = Agent(model=..., system_prompt="Synthesize a summary from facts.")

async def map_chunk(chunk: str) -> Scored[Fact]:
    return (await extractor.run(chunk)).output

async def reduce_facts(items: list[Scored[Fact]]) -> str:
    high_or_med = filter_by_min_confidence(items, "medium")
    ranked = rank_by_confidence(high_or_med)
    prompt = "Facts:\n" + "\n".join(f"- {it.value.text}" for it in ranked)
    return (await synthesizer.run(prompt)).output

mr = MapReduce(map_step=map_chunk, reduce_step=reduce_facts, map_concurrency=8)
summary = await mr.run(document_chunks)
```

Reading what this does end to end: 8 chunks processed in parallel, each producing one `Scored[Fact]`; the reducer drops "low" facts, ranks the rest, synthesizes. If 50% of chunks produced low-confidence noise, the reducer never sees it. Both quality and cost improve.

Chapter 16 (MapReduce deep dive) covers the pattern in detail.

## Composing with `CircuitBreaker`

If your agent repeatedly produces low-confidence outputs, that's a signal the system is degrading — bad inputs, prompt drift, model regression. `CircuitBreaker.is_success` is a predicate that lets you treat low-confidence outputs as failures:

```python
from ballast.resilience.circuit_breaker import CircuitBreaker, Consecutive

cb = CircuitBreaker(
    threshold_factory=lambda: Consecutive(3),
    is_success=lambda r: isinstance(r, Scored) and r.confidence != "low",
)

# After 3 consecutive low-confidence outputs, the breaker opens.
# Subsequent calls hit your fallback policy instead of the (degrading) extractor.
result = await cb.call(lambda: extractor.run(chunk))
```

This is the article's "mandatory final state" pattern realized: don't pretend silent low-confidence outputs are fine. Treat them as a failure signal and degrade gracefully. Chapter 13 covers `CircuitBreaker`.

## Composing with `Reflection`

`Scored[Critique]` is a particularly useful pattern inside a `Reflection` loop: when the critic isn't confident in its critique, accept the draft instead of looping forever:

```python
from ballast import Reflection

critic = Agent(model=..., output_type=Scored[Critique])

reflection = Reflection(
    writer=draft_agent,
    critic=critic,
    refiner=refiner_agent,
    max_iterations=5,
    accept_if=lambda critique: critique.confidence == "low",
    # Accept the draft when the critic isn't even confident in finding issues —
    # the critic is signaling "I think this is fine"
)
```

Three iterations of a critic confidently calling out problems is a productive loop. Five iterations of a critic with low confidence is the model thrashing. The `accept_if` predicate stops the thrashing.

## Composing with `scan_output`

`scan_output` (the schema walker that powers `Ref[T]` narrowing — chapter 5) recurses into `Scored.value` naturally. This means you can wrap a grounded output with `Scored` and get both behaviors:

```python
class FactWithSource(BaseModel):
    text: str
    project: Ref[Project]      # grounded reference

agent = Agent(model=..., output_type=Scored[FactWithSource])

# scan_output finds Ref[Project] inside Scored.value automatically.
# The schema sent to the LLM has both the narrowed project enum AND the Scored fields.
grounded = GroundedAgent(inner=agent, resolvers={Project: ProjectResolver()})
result = await grounded.run(query)

# Hydrate the value's ref:
hydrated_value = await result.output.value.project.hydrate(project_repo)
```

This is the framework's contract-stacking story: each typed wrapper handles one concern, and they compose without coordination. `Scored` adds quality signal; `Ref` adds grounded entity references; the inner model holds your business data.

## Custom `ConfidenceT`

The default `Literal["low", "medium", "high"]` covers almost everything. When you need something else:

```python
from typing import Literal
from ballast import Scored

# Binary safety classification
SafetyLabel = Literal["safe", "uncertain"]
agent = Agent(model=..., output_type=Scored[ContentItem, SafetyLabel])

# Numeric 1-5 (against the article's recommendation, but supported)
agent = Agent(model=..., output_type=Scored[Fact, int])
```

When you override `ConfidenceT`, the built-in helpers (`filter_by_min_confidence`, `rank_by_confidence`) don't work — they're hardcoded to the default `Confidence` Literal. You'll write your own helpers using the same shape:

```python
def filter_safe(items: list[Scored[ContentItem, SafetyLabel]]):
    return [it for it in items if it.confidence == "safe"]
```

For the int case, the trade-offs from "Why named labels, not numbers" still apply. If you find yourself building elaborate bucketing logic on top of `Scored[T, int]`, that's signal to rethink — usually labels would work just as well with less work.

## The `rationale` field discipline

The single most important habit when using `Scored[T]`: do **not** make `rationale` optional. The framework defines it as required for a reason — the model produces calibrated confidence labels *because* it has to write the rationale first.

If you find yourself wanting to "skip the rationale to save tokens," what you actually want is a different output type (just `Fact`, no scoring). The rationale is the work; the label is the byproduct. Removing the rationale gives you a label without the work — and the label loses meaning.

You can validate the rationale length if you want non-trivial reasoning:

```python
from typing import Annotated, ClassVar
from pydantic import Field
from ballast.quality.scored import DriftVerdictBase   # not a real import; illustrative

class StrictScored(BaseModel):
    value: Fact
    rationale: Annotated[str, Field(min_length=20)]
    confidence: Confidence
```

But typically just requiring the field (which is the default) is enough.

## Discipline checklist

- **Use `Scored[T]` for any tool / agent output where downstream code needs to know how reliable the value is.**
- **Use named labels.** The default Literal is right.
- **Don't bypass the rationale.** Required field for a reason.
- **Wire `is_success` predicates** that treat low confidence as a failure signal, not noise.
- **Compose, don't custom-build.** `Scored[Note]`, `Scored[list[Ref[Project]]]`, `Scored[Critique]` all work — you don't need to invent app-specific scoring schemas.
- **Don't double-wrap.** `Scored[Scored[T]]` is meaningless. Pick one level of granularity.

## What this chapter did NOT cover

- `BallastCapability` and `JudgeAfterRun` for *external* grading of outputs — chapter 23.
- `LLMJudge` for online quality evaluation — chapter 23.
- The `MapReduce` pattern in detail — chapter 16.
- The `Reflection` pattern — chapter 15.

## Where to go next

→ [07-capabilities.md](07-capabilities.md) — cross-cutting concerns hooked into the agent run.

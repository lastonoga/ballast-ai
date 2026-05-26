# How to add confidence + rationale to tool/agent outputs

**Problem:** Your agent extracts facts / writes summaries / emits structured data. Downstream code can't tell which outputs are high-quality vs guesses. You want a uniform `(value, rationale, confidence)` wrapper so reduce / filter / route by quality is possible.

**Solution:** Use `Scored[T]` as the output type. Default confidence is `Literal["low", "medium", "high"]` — labeled buckets that avoid the mean-reversion problem that hits numeric scales.

## Minimum: agent that returns `Scored[T]`

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
        "Provide a rationale (1 sentence) and confidence "
        "('high' if explicitly stated, 'medium' if implied, 'low' if guessed)."
    ),
    output_type=Scored[Fact],
)

result = await extractor.run(document_text)
print(result.output.value.text)        # the fact
print(result.output.rationale)         # the LLM's justification
print(result.output.confidence)        # "low" | "medium" | "high"
```

That's it. Pydantic validates the response; rationale + confidence are required fields.

## Filter and rank a batch

The framework ships small helpers for the default `Confidence` Literal:

```python
from ballast.quality.scored import (
    aggregate_by_confidence,
    filter_by_min_confidence,
    rank_by_confidence,
)

items: list[Scored[Fact]] = [
    await extractor.run(doc) for doc in documents  # parallel-friendly
]
items = [r.output for r in items]

# Drop low-confidence ones
kept = filter_by_min_confidence(items, "medium")

# Sort high → low; tie-break by source URL
ranked = rank_by_confidence(kept, secondary_key=lambda it: it.value.source_url)

# Or bucket for separate downstream handling
buckets = aggregate_by_confidence(items)
high_quality_facts = buckets["high"]
```

## Use inside MapReduce

The `Scored[T]` + MapReduce combo is the article's recommended Map-phase pattern:

```python
from ballast import MapReduce


async def map_extract(chunk: str) -> Scored[Fact]:
    result = await extractor.run(chunk)
    return result.output


async def reduce_synthesize(items: list[Scored[Fact]]) -> Summary:
    high_or_med = filter_by_min_confidence(items, "medium")
    ranked = rank_by_confidence(high_or_med)
    return await summarizer.run(prompt_with(ranked))


mr = MapReduce(map_step=map_extract, reduce_step=reduce_synthesize, map_concurrency=8)
summary = await mr.run(document_chunks)
```

Map workers run in parallel, each emits a `Scored[Fact]`. Reduce filters by confidence before synthesizing. Low-confidence noise is dropped before it pollutes the final answer.

## Bridge to Circuit Breaker

Treat repeated low-confidence outputs as a CB failure — useful when a model degrades or a document is junk:

```python
from ballast.resilience.circuit_breaker import CircuitBreaker, Consecutive

cb = CircuitBreaker(
    threshold_factory=lambda: Consecutive(3),
    is_success=lambda r: isinstance(r, Scored) and r.confidence != "low",
)

# CB opens after 3 consecutive low-confidence outputs
result = await cb.call(lambda: extractor.run(chunk))
```

## Use as agent output for confidence-aware reflection

```python
from ballast import Reflection

critique_agent = Agent(
    model=...,
    output_type=Scored[Critique],
)

reflection = Reflection(
    writer=draft_agent,
    critic=critique_agent,
    refiner=refiner_agent,
    # In your refiner step: only act on critique if critic.confidence == "high"
    accept_if=lambda critique: critique.confidence == "low",   # accept draft if critic isn't confident
)
```

## Custom confidence shape

The default `Confidence = Literal["low", "medium", "high"]` works for most cases. If you need different:

```python
from typing import Literal
from ballast import Scored

# Numeric 1-5 (if your downstream system requires it)
agent = Agent(model=..., output_type=Scored[Fact, int])

# Binary
Binary = Literal["safe", "uncertain"]
agent = Agent(model=..., output_type=Scored[ToolCall, Binary])
```

**Built-in helpers** (`filter_by_min_confidence` etc.) only work with the default 3-bin Literal. For custom shapes, you write your own — usually one-liners.

## Required `rationale` field

`Scored[T]` requires `rationale: str` — empty strings are allowed but the field IS required. This forces the LLM to think (CoT) before committing to a confidence label, dramatically reducing mean-reversion. Don't make it optional.

## Native composition with `scan_output` and `Ref[T]`

`scan_output` recurses into `Scored.value` automatically — `Ref[T]` fields inside the wrapped value are still found and JSON-Schema-narrowed:

```python
from ballast import Ref

class FactWithSource(BaseModel):
    text: str
    project: Ref[Project]

agent = Agent(model=..., output_type=Scored[FactWithSource])
# The agent's tool schema includes the dynamically-narrowed enum of valid project IDs
# (the Ref is found inside Scored.value via natural recursion).
```

## Caveats

- **`frozen=True`** — `Scored[T]` instances are immutable. Mutate by constructing a new instance.
- **Empty rationale** — pydantic allows `rationale=""` by default. If you want non-empty, use `Annotated[str, MinLen(1)]` in a custom subclass.
- **Don't double-wrap** — `Scored[Scored[T]]` works syntactically but is meaningless. Pick the right level of granularity.

## Related

- [add-circuit-breaker-to-tool.md](add-circuit-breaker-to-tool.md) — `is_success` predicate bridge
- [use-mapreduce-for-rag.md](use-mapreduce-for-rag.md) — Map-phase pattern with confidence-aware reduce
- Reference: `reference/quality/scored.md`
- Explanation: [article-pain-points.md](../explanation/article-pain-points.md) #6

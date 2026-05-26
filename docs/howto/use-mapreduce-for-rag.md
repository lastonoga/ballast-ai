# How to use MapReduce for documents larger than the context window

**Problem:** You need to extract structured facts from a document (PDF, transcript, codebase, legal contract) that's bigger than the LLM context window (or close to it — "lost in the middle" hits well before the hard limit). A single-call extraction misses facts in the middle of the text and produces low-recall summaries.

**Solution:** `MapReduce` pattern — shard the document into chunks, run an extractor LLM per chunk in parallel (Map), then aggregate via a reducer LLM (Reduce). Combine with `Scored[T]` so the reducer can filter low-confidence noise.

## Minimum

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from ballast import MapReduce, Scored
from ballast.quality.scored import filter_by_min_confidence, rank_by_confidence


class Fact(BaseModel):
    text: str
    citation: str


extractor = Agent(
    model="openai:gpt-4o-mini",
    system_prompt=(
        "Extract one key fact from the chunk. Provide a citation (exact "
        "substring from the chunk) and confidence label "
        "('high' / 'medium' / 'low')."
    ),
    output_type=Scored[Fact],
)

synthesizer = Agent(
    model="openai:gpt-4o",
    system_prompt="Given a list of high-quality facts, write a 3-paragraph summary.",
)


async def map_chunk(chunk: str) -> Scored[Fact]:
    result = await extractor.run(chunk)
    return result.output


async def reduce_facts(items: list[Scored[Fact]]) -> str:
    kept = filter_by_min_confidence(items, "medium")
    ranked = rank_by_confidence(kept)
    prompt = "Facts:\n" + "\n".join(
        f"- [{it.confidence}] {it.value.text} ({it.value.citation})"
        for it in ranked
    )
    result = await synthesizer.run(prompt)
    return result.output


chunks = split_into_chunks(big_document, max_tokens=2_000)
mr = MapReduce(
    map_step=map_chunk,
    reduce_step=reduce_facts,
    map_concurrency=8,
)
summary = await mr.run(chunks)
```

That's the full pattern: chunked extraction, confidence-aware filter, ranked synthesis. The Map phase runs 8 chunks in parallel (`map_concurrency=8`).

## Use agents directly (no custom map/reduce functions)

If your map + reduce are pure LLM calls, skip the wrapper functions:

```python
mr = MapReduce(
    map_agent=extractor,            # Agent.run called per chunk; output unwrapped to .output
    reduce_agent=synthesizer,       # Agent.run called once with all map outputs
    map_concurrency=8,
)
summary = await mr.run(chunks)
```

XOR-validated: provide either `map_step` (callable) OR `map_agent` (Agent), not both. Same for reduce.

## Hierarchical collapse for very long documents

For documents producing hundreds of map outputs (e.g. 200-chunk legal contract), even the reducer hits its context limit. Use `collapse_threshold`:

```python
mr = MapReduce(
    map_agent=extractor,
    reduce_agent=synthesizer,
    map_concurrency=10,
    collapse_threshold=20,    # if > 20 map outputs, recurse: reduce batches of 20 first
)
```

Now if Map produces 200 items, MapReduce reduces them in batches of 20 → produces 10 intermediate digests → final reduce over those 10.

## Resilient against per-chunk failures

`MapReduce` supports per-call retries:

```python
mr = MapReduce(
    map_agent=extractor,
    reduce_agent=synthesizer,
    retries=2,                          # retry each map / reduce step up to 2x
    retry_backoff_seconds=0.5,          # exponential
)
```

Per-call retry happens inside the `@Durable.step` — DBOS memoises successful results, retries failures.

For burst protection (one flaky chunk doesn't crash the whole map), pair with `CircuitBreaker`:

```python
from ballast.resilience.circuit_breaker import CircuitBreaker, Consecutive

cb = CircuitBreaker(threshold_factory=lambda: Consecutive(5))

async def safe_map(chunk: str) -> Scored[Fact]:
    return await cb.call(lambda: extractor.run(chunk).output)

mr = MapReduce(map_step=safe_map, reduce_agent=synthesizer)
```

After 5 consecutive chunk-failures, CB opens — subsequent chunks fall back per your fallback policy (e.g. return `Scored[Fact](value=..., rationale="skipped", confidence="low")`) rather than crashing the whole reduce.

## Use inside a DBOS workflow

`MapReduce.run` is `@Durable.workflow`-decorated. Wrap a higher-level workflow around it:

```python
from ballast import Durable

@Durable.workflow()
async def process_legal_contract(contract_text: str) -> str:
    chunks = split_into_chunks(contract_text, max_tokens=2_000)
    summary = await mr.run(chunks)
    await store_summary(contract_id, summary)
    return summary
```

On crash mid-MapReduce, DBOS replay skips completed map calls + the final reduce if it succeeded.

## Caveats

- **`map_concurrency` is a semaphore, not a thread count.** All Map calls run on the same event loop. If your extractor LLM calls are CPU-bound (rare), use a process pool separately.
- **Per-chunk failures don't crash MapReduce by default** — they propagate as exceptions. Use `retries=N` to make them robust, OR wrap in CircuitBreaker for graceful skipping.
- **Reducer sees ALL map outputs.** Don't skip the `Scored[T]` + filter pattern — high-volume noise will pollute the reducer's context window.
- **Map outputs must be picklable** — they're passed through DBOS step boundaries.

## Related

- [add-confidence-to-tool-outputs.md](add-confidence-to-tool-outputs.md) — `Scored[T]` wrapper for Map-phase quality
- [add-circuit-breaker-to-tool.md](add-circuit-breaker-to-tool.md) — protect Map calls from flaky chunks
- [build-plan-execute-pipeline.md](build-plan-execute-pipeline.md) — when you need a DAG, not a fan-out
- Reference: `reference/patterns/mapreduce.md`
- Explanation: [article-pain-points.md](../explanation/article-pain-points.md) #5

# 16. MapReduce — for documents bigger than context

**Prerequisites:** [14-patterns-intro.md](14-patterns-intro.md), [06-confidence-and-quality.md](06-confidence-and-quality.md).

## Introduction

Long-context models have made it tempting to just shove the whole document into one prompt and ask. Tempting and wrong: research on the "Lost in the Middle" effect shows that even when input fits, the model attends well to the start and end but loses information in the middle. A 300-page PDF crammed into a single 200K-context call produces a worse summary than the same document chunked and processed in pieces.

`MapReduce` is the framework's answer: shard the input into bounded chunks, run an agent (or a callable) per chunk in parallel, then run a reduce agent (or callable) to synthesize the per-chunk outputs into a final answer. Each model call sees something it can actually attend to; the framework handles concurrency, retries, hierarchical collapse for very large fan-outs, and replay-safety via DBOS.

This chapter walks through the `MapReduce` constructor, the difference between callable and agent forms, how `Scored[T]` makes the reduce phase smarter, and the knobs (`map_concurrency`, `collapse_threshold`, retries) you'll actually tune.

## The mental model

Two phases:

```
input: list[T_in]
   │
   ├── map (concurrent, bounded) ──► [u_1, u_2, ..., u_n]   # T_intermediate per chunk
   │
   └── reduce ──────────────────────► T_out
```

The map phase is parallel; bound it with `map_concurrency` so you don't overwhelm the model provider. The reduce phase is sequential by default but can be made *hierarchical* via `collapse_threshold` — useful when the map phase produces hundreds of intermediate items.

Both phases are wrapped in `@Durable.step` so on replay, already-completed work is cached. A crash at item 47 of 100 resumes from 47, not from 0.

## The simplest case

```python
from pydantic_ai import Agent
from ballast import MapReduce

extractor = Agent(model="openai:gpt-4o-mini", system_prompt="Extract one key fact.")
synthesizer = Agent(model="openai:gpt-4o", system_prompt="Synthesize a summary from facts.")

mr = MapReduce(
    map_agent=extractor,
    reduce_agent=synthesizer,
    map_concurrency=8,
)

summary = await mr.run(document_chunks)
```

Each chunk is sent to `extractor` independently; up to 8 run concurrently. When all are done, `synthesizer` receives the list of extracted facts and produces a final summary. One call to `mr.run(...)`; everything else is the framework.

## Agent form vs callable form

You can configure each phase as either:

- **An agent** (`map_agent=...` / `reduce_agent=...`) — the framework calls `agent.run(item)` and reads `.output`.
- **A callable** (`map_step=...` / `reduce_step=...`) — you pass an async function with full control.

The XOR is enforced: exactly one of agent-form or callable-form per phase.

Use the callable form when you need pre/post-processing around the LLM call:

```python
async def map_chunk(chunk: str) -> Scored[Fact]:
    if len(chunk) < 50:
        return Scored(value=Fact(text=""), rationale="too short", confidence="low")
    return (await extractor.run(chunk)).output

async def reduce_facts(items: list[Scored[Fact]]) -> str:
    kept = filter_by_min_confidence(items, "medium")
    if not kept:
        return "No reliable facts found."
    prompt = "Facts:\n" + "\n".join(f"- {it.value.text}" for it in kept)
    return (await synthesizer.run(prompt)).output

mr = MapReduce(map_step=map_chunk, reduce_step=reduce_facts, map_concurrency=8)
```

The callable form is also the only way to call a Pattern (e.g., `Reflection`) as your reduce phase.

## Concurrency

`map_concurrency` is the size of the asyncio semaphore that bounds map calls. The right value is provider- and tier-specific. For most OpenAI / Anthropic accounts:

- **8** is a safe default; you'll rarely hit rate limits.
- **16** if you have a tier-2+ account and the chunks are small.
- **32+** if you're on a high-throughput tier and the model is fast (e.g., gpt-4o-mini).

Going higher than the provider can handle leads to 429s. Those are caught by the `retries` mechanism, but the right move is to size for the provider rather than rely on retries.

## Retries on flaky chunks

```python
mr = MapReduce(
    map_agent=extractor,
    reduce_agent=synthesizer,
    map_concurrency=8,
    retries=2,                  # up to 2 retries per chunk
    retry_backoff_seconds=0.5,  # exponential: 0.5, 1.0, 2.0
)
```

A chunk that fails (network blip, transient 5xx) is retried up to `retries` times with exponential backoff before being given up on. Per-chunk failures don't fail the whole map phase — but check the retry telemetry; a high retry rate usually means `map_concurrency` is too high.

## Hierarchical collapse

`collapse_threshold` controls what happens when the map phase produces too many intermediate items. With 100 chunks and a synthesizer that can handle 10 at a time, you don't want to ship 100 facts to one reduce call (defeats the point). Setting `collapse_threshold=10` makes the framework do a *recursive* reduce: collapse groups of 10 first, then collapse the results, and so on.

```python
mr = MapReduce(
    map_agent=extractor,
    reduce_agent=synthesizer,
    map_concurrency=8,
    collapse_threshold=10,   # reduce in batches of 10
)
```

This is what makes MapReduce viable for documents in the hundreds-of-chunks range. Without it, the reduce call balloons.

## Composing with `Scored[T]`

This is the canonical use case for `Scored[T]` (chapter 6):

```python
from ballast.quality.scored import filter_by_min_confidence

extractor = Agent(model=..., output_type=Scored[Fact])

async def reduce_facts(items: list[Scored[Fact]]) -> str:
    high_or_med = filter_by_min_confidence(items, "medium")
    if not high_or_med:
        return "Insufficient reliable facts."
    prompt = "\n".join(f"- {it.value.text} ({it.confidence})" for it in high_or_med)
    return (await synthesizer.run(prompt)).output

mr = MapReduce(
    map_agent=extractor,         # produces Scored[Fact]
    reduce_step=reduce_facts,    # filters low-confidence then synthesizes
    map_concurrency=8,
)
```

What this buys you: the reduce never sees low-confidence noise. Two wins — better synthesis output, lower token cost in the reduce call.

## Wrapping map with `CircuitBreaker`

If you're calling an external service from inside the map function (e.g., a search API per chunk), wrap with a circuit breaker so one bad backend doesn't cascade:

```python
from ballast.resilience.circuit_breaker import CircuitBreaker, ReturnValue

cb = CircuitBreaker(
    threshold_factory=lambda: Consecutive(5),
    fallback=ReturnValue(Scored(value=Fact(text=""), rationale="degraded", confidence="low")),
)

async def map_chunk(chunk: str) -> Scored[Fact]:
    return await cb.call(lambda: extractor.run(chunk).then(lambda r: r.output))
```

After 5 consecutive failures, the breaker opens and subsequent chunks short-circuit to the degraded fallback. The map phase finishes; the reduce sees mostly real data plus a few low-confidence placeholders (which `filter_by_min_confidence` drops).

## Replay-safety in practice

`MapReduce.run()` is a `@Durable.workflow`. `_map_one` and `_reduce` are `@Durable.step`. What this means for crash recovery:

- Crash during map phase, item 47 of 100: on replay, items 1-46 are returned from cache; item 47 re-runs; items 48-100 then run.
- Crash during reduce phase: on replay, the entire map output is from cache (no re-extractions); only the reduce re-runs.
- Crash during hierarchical collapse: same — only the unfinished sub-reduce re-runs.

The replay model is what makes MapReduce safe to run on multi-hour jobs. Without it, a crash 80% through a 100-chunk job would mean re-running all 100 chunks.

## Common mistakes

- **`map_concurrency` too high.** Faster initially, then 429s and retries kick in and you net out slower (and your provider account gets flagged). Start at 8, measure.
- **No `Scored[T]` filter in reduce.** The reduce LLM has to deal with all the noise. Filter at the boundary.
- **Reduce agent with tiny context.** Reduce often sees N intermediate items; if N is large enough, you'll blow the reduce model's context. Either chunk smaller (more map items), use `collapse_threshold`, or pick a bigger reduce model.
- **Mixing agent and callable form for the same phase.** The constructor rejects this — exactly one per phase. If you want both pre-processing and an agent, use the callable form and call the agent inside it.
- **Forgetting that map output is unordered.** `asyncio.gather` returns in input order, so the framework does too — but the *processing* order across items isn't deterministic. If your reduce depends on the items being in input order, fine; if it depends on processing order, that's a bug.

## What this chapter did NOT cover

- The "Lost in the Middle" research in detail — search the paper directly.
- DivergentConvergent for the variety-vs-aggregation trade-off — chapter 17.
- Nesting `Reflection` inside MapReduce reduce — chapter 20.
- The exact `@Durable.step` semantics for replay — chapter 24.

## Where to go next

→ [17-divergent-convergent.md](17-divergent-convergent.md) — the other fan-out pattern, for variety.

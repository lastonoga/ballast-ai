# 17. Divergent-Convergent

**Prerequisites:** [14-patterns-intro.md](14-patterns-intro.md), [16-mapreduce.md](16-mapreduce.md).

## Introduction

LLMs have a known failure mode under brainstorming or open-ended generation tasks: they converge on the first plausible answer and stop exploring. Ask a model to "give me five marketing taglines" and you'll often get five variations of the same idea. Ask for "creative product names" and the names cluster around the first concept the model latched onto. The technical literature calls this the *Artificial Hivemind* — the model's training distribution pulls toward the safe, average, expected answer.

`DivergentConvergent` is the framework's answer: instead of asking once and getting one perspective, run multiple *independent* branches (potentially with different models, different prompts, different seeds), deduplicate the results to keep only meaningfully different ideas, optionally verify them with a scorer, then synthesize a final answer from the surviving candidates. The shape echoes the CREATIVEDC pattern from generative-AI research literature.

This chapter covers when the pattern fits, the constructor's many knobs, how progress events stream to a UI, when *not* to use it (it's expensive), and how to nest it inside `MapReduce`.

## The mental model

```
input
   │
   ├── divergent branches (concurrent, variety) ──► pool of hypotheses
   │       │
   │       ├── branch A (cheap exploration model, high temp)
   │       ├── branch B (different model, different prompt)
   │       └── branch C (yet another angle)
   │
   ├── dedup (embedding-based) ────────────────────► unique hypotheses only
   │
   ├── verify (optional scorer) ───────────────────► top-k by score
   │
   └── synthesize (premium model, low temp) ───────► final OutT
```

The key insight: *variety in the divergent phase compounds into quality in the convergent phase*. A synthesizer given five genuinely different angles produces something better than a synthesizer given five variations on one angle. The dedup step makes sure you actually get variety, not just appearance of it.

## The simplest case

```python
from ballast.patterns.divergent_convergent import (
    DivergentConvergent,
    DivergentBranch,
)

# Three different angles on the same prompt
branches = (
    DivergentBranch(label="conservative", agent=conservative_agent, samples=2),
    DivergentBranch(label="experimental", agent=experimental_agent, samples=2),
    DivergentBranch(label="contrarian", agent=contrarian_agent, samples=1),
)

dc = DivergentConvergent(
    branches=branches,
    synthesizer=premium_agent,
    hypotheses=lambda env: env.hypotheses,   # extract list[HypothesisT] from branch output
    min_hypotheses=2,
)

result = await dc.run(input)
```

What this does: spawn 5 total branch runs (2+2+1), collect hypotheses, dedupe, pass to the synthesizer. Each branch can use a *different* model, prompt, and sampling temperature — that's where the variety comes from.

## Branches: where variety comes from

`DivergentBranch` is the unit of "one angle." Three knobs:

- **`label`** — used in progress events and logs.
- **`agent`** — the agent (or callable Pattern) that runs the angle.
- **`samples`** — how many times to run this branch.

Multiple branches with `samples=1` give you angular variety (different prompts, models). One branch with `samples=5` gives you stochastic variety (same prompt, different samples). The interesting variants combine: 3 branches with `samples=2` = 6 hypotheses spanning 3 angles each with some stochastic spread.

Provider variety is a particularly strong source of diversity:

```python
branches = (
    DivergentBranch(label="openai", agent=Agent(model="openai:gpt-4o"), samples=2),
    DivergentBranch(label="anthropic", agent=Agent(model="anthropic:claude-3-5-sonnet"), samples=2),
    DivergentBranch(label="local", agent=Agent(model="ollama:llama3"), samples=1),
)
```

Cross-provider runs escape provider-specific training-data blind spots.

## Dedup: ensuring real variety

The dedup step uses embeddings to drop near-duplicates:

```python
from ballast.patterns.divergent_convergent import EmbeddingDeduper

dc = DivergentConvergent(
    branches=branches,
    synthesizer=premium_agent,
    hypotheses=lambda env: env.items,
    deduper=EmbeddingDeduper(embedder=my_embedder, threshold=0.9),
)
```

Any pair of hypotheses with cosine similarity > `threshold` collapses to one. `0.9` is reasonable for natural-language hypotheses; tune up for stricter dedup, down for looser.

If you skip the deduper, all hypotheses pass through — sometimes you want that (final ranking is what matters, not uniqueness).

## Verifier: optional quality filter

```python
class FactualityScorer:
    async def score(self, task, hypothesis) -> float:
        # ... LLM-judge call returning 0.0-1.0
        return score

dc = DivergentConvergent(
    branches=branches,
    synthesizer=premium_agent,
    hypotheses=lambda env: env.items,
    verifier=FactualityScorer(),
    top_k=3,   # only the top 3 scored hypotheses go to the synthesizer
)
```

The verifier scores each surviving hypothesis; `top_k` keeps the highest-scoring ones. This is how you avoid the synthesizer wasting tokens on low-quality candidates.

`top_k` requires `verifier` to be set (you can't pick "top 3" without a score).

## Synthesizer: the convergent phase

The synthesizer is the final agent that takes the surviving hypotheses and produces the output. It's almost always a premium model (`gpt-4o`, `claude-3-5-sonnet`) at low temperature — you want it to *synthesize*, not to generate more variety.

By default, the synthesizer gets a prompt formatted as the original input plus the bulleted list of hypotheses. Override via `format_synth_prompt`:

```python
def format_prompt(input: str, hypotheses: list[Idea]) -> str:
    return f"""Original request: {input}

Candidate ideas:
{chr(10).join(f"- {h.title}: {h.description}" for h in hypotheses)}

Synthesize the strongest single recommendation, combining the best of these.
"""

dc = DivergentConvergent(..., format_synth_prompt=format_prompt)
```

## Streaming progress to a UI

`DivergentConvergent` emits typed events to the `divergent_convergent_progress` signal:

- `BranchEnqueued(label, sample_idx)` — a branch run is starting
- `BranchCompleted(label, sample_idx, pool_size)` — a branch run finished, pool now N
- `BranchFailed(label, sample_idx, error_type)` — a branch run errored
- `DedupCompleted(input_count, output_count)` — dedup done; N → M
- `VerifyCompleted(scored_count, top_k_applied)` — verification done
- `ConvergeStarted(candidate_count)` — entering synthesis
- `ConvergeCompleted()` — done

For per-instance callbacks (rather than the global signal):

```python
async def on_event(event):
    await send_to_user(f"{event.__class__.__name__}: {event}")

dc = DivergentConvergent(..., on_progress=on_event)
```

The callback runs at every event boundary. Exceptions in the callback are logged and swallowed (the run continues).

## When NOT to use this pattern

Three signals that DC is the wrong tool:

- **You have a single correct answer.** Variety doesn't help when there's one right answer; you're spending tokens to vote among redundant attempts.
- **Latency is critical.** DC is slow — the divergent phase plus synthesis is many LLM calls. Don't use in latency-sensitive paths.
- **Cost is critical.** A typical DC run is 5-10x the cost of a single call. Use it where output quality justifies the spend.

Use cases that fit: open-ended generation (brainstorming, naming, content ideation), research synthesis where multiple perspectives matter, design exploration. Use cases that don't fit: classification, extraction, summarization, anything with a verifiable ground truth.

## Embedding `DivergentConvergent` inside `MapReduce`

The two patterns compose:

```python
dc = DivergentConvergent(
    branches=brainstorm_branches,
    synthesizer=synthesizer,
    hypotheses=lambda env: env.ideas,
)

mr = MapReduce(
    map_step=lambda chunk: dc.run(chunk),   # one DC per chunk
    reduce_step=aggregate,
    map_concurrency=4,
)
```

For each chunk, run a full DC; aggregate the synthesized outputs. The `map_concurrency` should be lower than usual (DC is heavier than a single agent call), but the pattern is fine. Replay-safety nests cleanly.

## Common mistakes

- **Same model in all branches.** No variety. Cross-provider or cross-prompt is where the win comes from.
- **No deduper.** You'll get 6 hypotheses that are 6 phrasings of one idea. The synthesizer sees no variety, output is no better than a single call.
- **Verifier without `top_k`.** You compute scores but use all of them. Save the synthesizer tokens — set `top_k=3` (or whatever your actual budget allows).
- **Using DC for low-creativity tasks.** Spending 5x the cost on a classification task is wasteful. Match the pattern to the task.
- **Forgetting `min_hypotheses`.** If dedup collapses everything to 1 hypothesis and you require at least 2, the run raises. Set this to a reasonable floor (2-3) for "fail fast if all branches converge."

## What this chapter did NOT cover

- The dedup embedder choice — provider-specific; covered in `ballast.embedders`.
- `PlanAndExecute` — chapter 18.
- The CoALA unit pattern — chapter 19.
- Frontend rendering of `DivergentEvent` — chapter 22 + the notes-app demo's `BrainstormPanel`.

## Where to go next

→ [18-plan-and-execute.md](18-plan-and-execute.md) — the planning family.

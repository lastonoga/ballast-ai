# How to use Divergent-Convergent for brainstorming

**Problem:** Your agent generates "safe", homogeneous outputs — the same predictable answer regardless of input variation. You need to break out of the local optimum and explore the solution space ("Artificial Hivemind" effect, per the article). Then synthesize the best ideas into a final answer.

**Solution:** `DivergentConvergent` pattern (CREATIVEDC methodology). Phase 1 spins up N parallel "divergent" agents that explore freely; Phase 2 dedups, ranks, and synthesizes via a "convergent" agent.

## Minimum

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from ballast import DivergentConvergent


class Hypothesis(BaseModel):
    idea: str
    rationale: str


divergent = Agent(
    model="openai:gpt-4o-mini",
    system_prompt=(
        "Generate ONE novel hypothesis on the user's topic. "
        "Aim for breadth and originality over safety."
    ),
    output_type=Hypothesis,
)

convergent = Agent(
    model="openai:gpt-4o",
    system_prompt=(
        "Given multiple hypotheses, pick the 3 most promising and synthesize "
        "a coherent final answer addressing the user's original question."
    ),
)


pattern = DivergentConvergent(
    divergent_agent=divergent,
    convergent_agent=convergent,
    branch_count=8,                # 8 parallel divergent calls
    dedup_threshold=0.92,          # cosine-similarity-based dedup
    embedder=my_embedder,          # required for dedup
)

result = await pattern.run("How might LLMs evolve in 2027?")
```

Eight divergent agents run in parallel, dedup'd by semantic similarity, then synthesized. The whole pattern is `@Durable.workflow`-wrapped — DBOS replays correctly.

## Stream progress to UI

Pattern emits typed progress events. Subscribe via `on_progress` callback:

```python
from ballast.patterns.divergent_convergent import (
    BranchEnqueued, BranchCompleted, DedupCompleted, ConvergeCompleted,
)


async def emit_to_thread(event) -> None:
    # event is typed (BranchEnqueued | BranchCompleted | ...)
    await thread_event_publisher.send("brainstorm_progress", {
        "type": type(event).__name__,
        "data": event.model_dump() if hasattr(event, "model_dump") else event.__dict__,
    })


pattern = DivergentConvergent(
    divergent_agent=divergent,
    convergent_agent=convergent,
    branch_count=8,
    dedup_threshold=0.92,
    embedder=my_embedder,
    on_progress=emit_to_thread,
)
```

Frontend (notes-app/assistant-ui) auto-renders these as rows with spinners / checkmarks. Apps that don't need the callback get the events via signal (auto-subscribed routing handler emits to thread by default).

## Add a verifier between dedup and converge

If you need to validate hypotheses before they reach the convergent agent (e.g. constraint check, citation requirement), pass a `verifier`:

```python
from ballast import Verifier


class MyVerifier(Verifier[Hypothesis]):
    async def verify(self, hypothesis: Hypothesis) -> bool:
        return len(hypothesis.idea) > 20 and "speculation" not in hypothesis.idea.lower()


pattern = DivergentConvergent(
    divergent_agent=divergent,
    convergent_agent=convergent,
    verifier=MyVerifier(),
    branch_count=8,
    dedup_threshold=0.92,
    embedder=my_embedder,
)
```

Verifier runs after dedup, before convergence. Invalid hypotheses are dropped + reported via `BranchFailed` event.

## Different models per phase

Common pattern: cheap model for exploration (you want many tries), expensive model for synthesis (you want one quality answer).

```python
divergent = Agent(model="openai:gpt-4o-mini", ...)        # cheap
convergent = Agent(model="anthropic:claude-3-5-sonnet", ...) # premium

# Or via different providers entirely:
divergent = Agent(model="openrouter:qwen-2.5-72b", ...)   # diverse generation
convergent = Agent(model="openai:gpt-4o", ...)             # tight synthesis
```

The article's pattern: use 3 different model providers in divergent phase (`gpt-4o`, `claude-3-5-sonnet`, `qwen-2.5-72b` together) to escape architectural blind spots. You'd wire that by giving `divergent_agent` a multi-provider dispatcher (not built-in; small custom Agent).

## Use as a sub-pattern inside MapReduce

Sometimes you want divergent-convergent PER chunk in a MapReduce:

```python
from ballast import MapReduce, DivergentConvergent

per_chunk_dc = DivergentConvergent(
    divergent_agent=extractor_variant,
    convergent_agent=picker,
    branch_count=4,
    dedup_threshold=0.9,
    embedder=embedder,
)

async def map_chunk(chunk: str) -> Scored[Fact]:
    fact = await per_chunk_dc.run(chunk)    # 4-way explore per chunk
    return Scored[Fact](value=fact, ...)

mr = MapReduce(map_step=map_chunk, reduce_agent=synthesizer, map_concurrency=8)
```

For each chunk: 4-way divergent extract + 1-way converge to pick the best fact. Then MapReduce over chunks.

## Caveats

- **Embedder required.** Dedup is cosine-similarity-based. Pass an `Embedder` Protocol implementation (apps wire their own — `openai`, `cohere`, local model, etc.).
- **`branch_count × per-branch cost`** — 8 branches with gpt-4o = 8× the cost of a single call. Use cheap models for divergent phase.
- **Don't use for tasks with one right answer.** Pattern shines when variety matters (brainstorming, hypothesis generation, content rewriting). For deterministic queries (factual lookup), use a single LLM call.
- **Convergent agent sees raw hypotheses.** If outputs are large, the synthesis prompt gets long — consider summarizing each hypothesis before convergence (custom `Synthesizer` wrapper).

## Real example

`examples/notes-app/backend/src/notes_app/workflows/brainstorm_flow.py` uses this pattern for the "brainstorm" button in the UI. Worth reading end-to-end to see how `on_progress` ties to the assistant-ui progress rows.

## Related

- [use-reflection-for-quality.md](use-reflection-for-quality.md) — when you want iterative refinement, not parallel exploration
- [use-mapreduce-for-rag.md](use-mapreduce-for-rag.md) — when you want sharded extraction, not divergent exploration
- Reference: `reference/patterns/divergent-convergent.md`
- Explanation: [article-pain-points.md](../explanation/article-pain-points.md) #4

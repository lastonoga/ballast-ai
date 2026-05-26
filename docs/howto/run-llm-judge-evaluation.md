# How to run LLM-as-Judge evaluation

**Problem:** Your agent is in production. You want to continuously evaluate output quality — not via human review (too slow / costly) but via an LLM judge. The judge's verdicts should land in logfire as span attributes, persist for later analysis, and (optionally) gate the response if quality is too low.

**Solution:** `LLMJudge` (the judge) + `JudgeAfterRun` (the capability that runs the judge per agent run). Configure once; verdicts auto-emit + persist.

## Minimum

```python
from ballast import (
    Agent,
    LLMJudge, JudgeAfterRun,
)


judge = LLMJudge(
    rubric=(
        "Output must:\n"
        "1. Cite at least one source URL.\n"
        "2. Avoid speculation.\n"
        "3. Stay under 500 words."
    ),
    judge_model="openai:gpt-4o-mini",      # cheap model is fine for grading
    threshold=0.7,                          # 0..1 score
)

agent = Agent(
    model="openai:gpt-4o",
    capabilities=[JudgeAfterRun(judge=judge, sync=False)],   # async; doesn't block
)

result = await agent.run("Summarize the latest ML papers")
# Verdict arrives async, logged + persisted; agent response returned normally.
```

The judge fires AFTER the agent's response is ready. With `sync=False`, the user gets the response immediately; the judge runs in the background and writes its verdict to logfire/storage.

## Sync vs async judging

```python
JudgeAfterRun(judge=judge, sync=False)   # default — non-blocking
JudgeAfterRun(judge=judge, sync=True)    # blocking — raises JudgeFailed if below threshold
```

**Use `sync=True`** when:
- Output is HIGH-stakes (financial, medical, legal). Don't ship a sub-threshold response.
- You're testing — want test failures on bad outputs.

**Use `sync=False`** when:
- Latency matters more than catching every bad output.
- You want logging + dashboards, not gating.

## Per-thread persistence

```python
def thread_id_from(ctx) -> str | None:
    return getattr(ctx.deps, "thread_id", None) if ctx.deps else None


cap = JudgeAfterRun(
    judge=judge,
    thread_id_from=thread_id_from,
    on_verdict=lambda v, ctx: log_verdict(v, ctx),
)
```

When `thread_id_from(ctx)` returns a thread ID, the framework persists the verdict as a `judge_verdict` thread event. UI can show "judge approved/flagged this turn" inline.

## Custom verdict handlers

```python
async def alert_on_low_score(verdict, ctx):
    if verdict.score < 0.5:
        await slack.alert(
            f"LOW QUALITY AGENT OUTPUT (score={verdict.score:.2f})\n"
            f"thread={ctx.deps.thread_id}\n"
            f"reason={verdict.reason}"
        )


cap = JudgeAfterRun(judge=judge, on_verdict=alert_on_low_score)
```

`on_verdict` runs after the judge returns. Use for alerting, metric counters, custom storage.

## Pairwise comparison (A/B model evaluation)

```python
from ballast import PairwiseVerdict


judge = LLMJudge.pairwise(
    rubric="Which response is more helpful + accurate?",
    judge_model="anthropic:claude-3-5-sonnet",
)

a = Agent(model="openai:gpt-4o").run(query)
b = Agent(model="anthropic:claude-3-5-sonnet").run(query)
results_a, results_b = await asyncio.gather(a, b)

verdict: PairwiseVerdict = await judge.compare(
    output_a=results_a.output,
    output_b=results_b.output,
    context=query,
)
print(verdict.winner, verdict.reason)
```

Useful for model selection / prompt iteration.

## Custom rubrics per agent

Different agents = different quality bars:

```python
notes_judge = LLMJudge(
    rubric="Notes must be properly formatted markdown with at most 200 words.",
    judge_model="openai:gpt-4o-mini",
    threshold=0.6,
)

publish_judge = LLMJudge(
    rubric=(
        "Posts must be at least 300 words, cite sources, avoid speculative claims, "
        "have an engaging title, end with a clear call to action."
    ),
    judge_model="anthropic:claude-3-5-sonnet",
    threshold=0.85,        # higher bar
)

notes_agent = Agent(model=..., capabilities=[JudgeAfterRun(judge=notes_judge)])
publish_agent = Agent(model=..., capabilities=[JudgeAfterRun(judge=publish_judge, sync=True)])
```

## Fail-open vs fail-closed

```python
JudgeAfterRun(judge=judge, fail_open=True)   # default; judge unavailable → keep going
JudgeAfterRun(judge=judge, fail_open=False)  # judge unavailable → raise JudgeUnavailable
```

`fail_open=True` is the safe production default — a flaky judge model shouldn't block user replies. Set `fail_open=False` only when judge quality IS critical (regulated industries).

## Span attributes

The judge writes its verdict to the agent.run() logfire span as attributes:
- `judge.score: float`
- `judge.passed: bool`
- `judge.reason: str`
- `judge.judge_model: str`

Build logfire dashboards on these (e.g. "average judge score by hour", "% passed by agent name").

## Combine with GoalDriftDetector

```python
agent = Agent(
    model=...,
    capabilities=[
        JudgeAfterRun(judge=output_quality_judge),       # judges OUTPUT quality
        GoalDriftDetector(DriftEngine(                    # judges process drift
            ...,
            judge=make_default_judge(),
        )),
    ],
)
```

Two judges, different concerns. Output judge catches "did the response satisfy the rubric"; drift detector catches "did the agent stay on-task during execution".

## Build offline eval set

For pre-deploy regression testing, capture production traces + replay against the new agent:

```python
from ballast import Dataset, EvalCase


dataset = Dataset.from_traces(
    logfire_query="agent.run AND service.name='my-app' AND time > now() - 7d",
    sample=100,
)

for case in dataset.cases:
    new_agent_result = await new_agent.run(case.input)
    verdict = await judge.grade(new_agent_result.output)
    print(f"{case.id}: score={verdict.score}")
```

See [build-eval-dataset-from-traces.md](build-eval-dataset-from-traces.md) for details.

## Caveats

- **Judge costs add up.** Every agent run × judge call. Use cheap models (`gpt-4o-mini`, `haiku`) and `sync=False` to avoid bottlenecks.
- **Rubrics drift.** Periodically (monthly) review your rubrics — they may be too strict / lax as your product evolves.
- **Don't judge with the same model as the agent.** Self-judging models are too lenient. Use a different provider or at least a different model size.
- **Numeric scores are problematic for general grading.** The article warns about mean-reversion on 1-10 scales. Consider `LLMJudge(...rubric="Output: 'fully', 'partially', 'contradicted'", threshold='fully')` for nuanced cases.

## Related

- [add-goal-drift-detector.md](add-goal-drift-detector.md) — judges process, not output
- [add-logfire-tracing.md](add-logfire-tracing.md) — where verdicts land
- [build-eval-dataset-from-traces.md](build-eval-dataset-from-traces.md) — offline eval
- Reference: `reference/capabilities/llm-judge.md`
- Reference: `reference/evals/scorers.md`
- Explanation: [article-pain-points.md](../explanation/article-pain-points.md) #18

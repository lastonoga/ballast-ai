# 23. Evals

**Prerequisites:** [22-observability.md](22-observability.md).

## Introduction

You've added observability — you can see what the agent did. That answers "did the agent run?" It doesn't answer "did the agent run *well*?" A 200-token completion that's hallucinated and a 200-token completion that's factually correct look identical in trace metrics. Quality is invisible to classical monitoring.

The framework's evals subsystem fills that gap with two complementary mechanisms:

- **Online grading** — `LLMJudge` + `JudgeAfterRun` grade every (or every sampled) production response with a rubric, and the verdicts get persisted so you can dashboard them.
- **Offline grading** — a `Dataset` of `EvalCase`s is run against an agent and produces an `EvalReport` with per-case scores. Datasets can be hand-authored or captured from production traces via the `dataset-from-traces` CLI.

Both lean on the same `Scorer` Protocol so you can mix custom scorers (regex-based, embedding-similarity, business-rule) with LLM-judge scorers.

This chapter walks through `LLMJudge`, the `Scorer` Protocol, the `Dataset` / `EvalCase` / `EvalReport` model, the `dataset-from-traces` CLI, and how to gate CI on evaluation regressions.

## The mental model

```
Production traffic ─► trace store ─► (optional) dataset-from-traces ─► Dataset
                                                                          │
                                                                          ▼
                                                                  EvalRunner
                                                                          │
                                                                          ▼
                          Scorers (LLMJudge / regex / schema / custom) ───►
                                                                          │
                                                                          ▼
                                                                    EvalReport
                                                                  (passed / failed)
```

Two important properties:

- **Scorers compose.** You can run multiple scorers per case (one for schema, one for tone, one for grounding) and the report aggregates per-scorer means.
- **Datasets are versioned content, not data dumps.** Authoring a dataset means choosing cases that *represent* the failure modes you care about — not blindly capturing N production traces.

## `LLMJudge` — rubric-based grading

```python
from ballast.evals import LLMJudge
from pydantic import BaseModel

class QualityVerdict(BaseModel):
    passed: bool
    issues: list[str]
    rationale: str

judge = LLMJudge(
    model="openai:gpt-4o",
    rubric="""
    The summary should be:
    - factually accurate to the source document
    - under 100 words
    - well-structured
    
    Return passed=True only if all three hold.
    """,
    output_type=QualityVerdict,
)

verdict = await judge.grade(output=summary_text, context={"source": document})
```

Pass a `rubric` (the criteria); the judge model evaluates the output against it and returns the typed verdict. You can pass `context` (the source document, the original prompt, anything the judge needs to grade against).

The default judge model is set via `Ballast(...).with_judge_defaults(model="...")` — useful when you have many judges and want one place to swap models.

## `JudgeAfterRun` — wire the judge to every run

```python
from ballast.evals import JudgeAfterRun

agent = Agent(
    model="openai:gpt-4o",
    output_type=ResearchSummary,
    capabilities=[JudgeAfterRun(judge=judge)],
)

result = await agent.run(query)
# Judge has already graded result.output; verdict is in result metadata
# (and persisted as a thread event if you wired the persistence sink)
```

Every run gets graded asynchronously. The grading doesn't block the user-facing response — you see the result first, the verdict lands shortly after. Verdicts emit thread events so you can dashboard them.

For high-volume production, sample:

```python
agent = Agent(
    ...,
    capabilities=[JudgeAfterRun(judge=judge, sample_rate=0.1)],   # 10%
)
```

Grading every call is fine in dev; sample in prod to keep judge costs reasonable.

## `Dataset` + `EvalCase` + `EvalReport`

The offline grading model:

```python
from ballast.evals import Dataset, EvalCase

dataset = Dataset(
    name="research_summarization_v1",
    cases=[
        EvalCase(
            name="known_topic",
            inputs={"topic": "ML deployment at scale"},
            expected="should mention model serving, monitoring, A/B testing",
        ),
        EvalCase(
            name="ambiguous_query",
            inputs={"topic": "Python"},
            expected="should ask for clarification or default to language",
        ),
        # ... more cases
    ],
)

report = await dataset.evaluate(
    runner=my_agent_runner,
    evaluators=[
        SchemaAdherenceScorer(),
        LLMJudgeScorer(judge=quality_judge, threshold=0.8),
    ],
)

print(f"Passed: {report.passed}, scores: {report.scorer_means}")
```

The runner takes each `EvalCase.inputs`, runs the agent, returns an `EvalRunOutput` (output, retries, error). Each scorer scores each output; the report aggregates.

## The `Scorer` Protocol

```python
@runtime_checkable
class Scorer(Protocol):
    threshold: float
    name: str

    async def score(self, run: EvalRunOutput) -> float: ...
```

Three things on a scorer: a threshold (the bar a score must clear to "pass"), a name (for the report), and a `score` method that returns 0.0-1.0.

The framework ships `SchemaAdherenceScorer` (returns 1.0 if output validates with no retries / errors, else 0.0). Apps write their own:

```python
class FactualGroundingScorer:
    name = "factual_grounding"
    threshold = 0.8

    def __init__(self, *, ground_truth_repo):
        self._repo = ground_truth_repo

    async def score(self, run: EvalRunOutput) -> float:
        if run.error or run.output is None:
            return 0.0
        # Check that each claim in run.output appears in ground truth
        claims = extract_claims(run.output)
        grounded = sum(1 for c in claims if await self._repo.contains(c))
        return grounded / max(len(claims), 1)
```

Stateless if you want, stateful if you need to. The protocol is permissive.

## LLM judge as a scorer

The most common scorer is "ask an LLM judge":

```python
class LLMJudgeScorer:
    name = "llm_quality"
    threshold = 0.8

    def __init__(self, *, judge: LLMJudge):
        self._judge = judge

    async def score(self, run: EvalRunOutput) -> float:
        if run.error or run.output is None:
            return 0.0
        verdict = await self._judge.grade(output=run.output)
        return verdict.score   # assume verdict has a 0-1 score
```

Wrap your `LLMJudge` in a scorer; plug into datasets. Now your offline eval includes LLM-graded quality.

## `dataset-from-traces` CLI

Production traces are gold for eval datasets — they're real user inputs that have actually hit the agent. The CLI turns traces into eval cases:

```bash
ballast dataset-from-traces \
    --pattern "research_summarize" \
    --since "2026-01-01" \
    --out datasets/research_v1.yaml \
    --source "myapp.factories:make_inputs_from_trace"
```

`--source` points to a Python factory function that converts trace data into `EvalCase` inputs. The CLI calls this factory for each matching trace; outputs a YAML dataset.

The captured dataset is now a versioned artifact you can run regression tests against.

**Important**: sanitize PII before persisting captured traces as eval cases. Real user input might contain personal data; your `make_inputs_from_trace` factory should strip / redact.

## CI gates on regression

The pattern: every PR runs the dataset; CI fails if the report's `passed` field is False or scorer means drop below threshold.

```python
# tests/evals/test_research_regression.py
import pytest
from ballast.evals import Dataset

@pytest.mark.asyncio
@pytest.mark.eval
async def test_research_dataset_passes():
    dataset = Dataset.load("datasets/research_v1.yaml")
    report = await dataset.evaluate(runner=runner, evaluators=[...])
    assert report.passed, f"Eval regression: {report.scorer_means}"
```

Mark as `eval` (or `integration`) so the suite only runs on dedicated CI jobs (not on every push — they're expensive). Run the eval CI on PRs that touch the agent or the prompts.

## Pairwise comparison for model selection

For "is gpt-4o better than claude-3-5-sonnet for this task?" you want pairwise. The framework doesn't ship a dedicated pairwise scorer (the `Scorer` Protocol is open enough that you write it yourself):

```python
class PairwiseScorer:
    name = "pairwise_vs_baseline"
    threshold = 0.5

    def __init__(self, *, baseline_outputs: dict[str, Any], judge: LLMJudge):
        self._baseline = baseline_outputs
        self._judge = judge

    async def score(self, run: EvalRunOutput) -> float:
        baseline = self._baseline[run.case_name]
        verdict = await self._judge.grade(
            output={"a": run.output, "b": baseline},
            context="Which response is better, a or b?",
        )
        return 1.0 if verdict.winner == "a" else 0.0
```

Run twice — once for each model — and the report tells you which won the pairwise.

## Sanitizing PII from captured traces

Two layers:

- **At span-time** (chapter 22's PII guidance): don't put raw user content into span attributes in the first place.
- **At dataset-capture time**: your `make_inputs_from_trace` factory should redact anything sensitive before persisting to the YAML dataset.

The framework doesn't auto-redact captured traces. That's your responsibility.

## Common mistakes

- **Treating "passed" as a number.** `report.passed` is boolean. If you want to chart trend, use `report.scorer_means` (numeric).
- **No baseline.** A scorer mean of 0.7 — is that good? Capture a baseline on the first run; chart deltas from there.
- **Tiny datasets.** 5 cases gives you almost no statistical signal. Aim for 50+ for meaningful regression detection.
- **All cases are happy paths.** Include adversarial / edge cases — ambiguous queries, off-topic input, malformed inputs. That's where regressions actually show up.
- **Running eval on every push.** It's expensive (LLM calls per case). Mark the suite and run on PR + nightly, not on every commit.
- **Grading by LLM with the same model under test.** Different family for the judge. Otherwise you measure self-agreement, which is meaningless.

## What this chapter did NOT cover

- Continuous eval-as-monitoring (dashboarding `JudgeAfterRun` verdicts over time) — chapter 22.
- The DBOS workflow inspector for trace-style debugging — chapter 24.
- Writing dataset YAML by hand vs from traces — both work; mix as needed.
- Production-grade sampling strategies — covered in logfire's docs.

## Where to go next

→ [24-durability.md](24-durability.md) — the DBOS layer in depth.

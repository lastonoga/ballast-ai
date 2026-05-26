# How to build an eval dataset from production traces

**Problem:** You want to test agent changes against real user inputs before deploying. Production traces hold them — `agent.run("...")` calls with their actual outputs. You need to convert these into a reusable eval set + run new versions against it + compare.

**Solution:** `dataset-from-traces` CLI + `Dataset` / `EvalCase` / `EvalReport` framework primitives. Built-in `SchemaAdherenceScorer` + room for custom `Scorer` Protocol implementations.

## Minimum: capture + replay

### 1. Generate dataset from logfire traces

```bash
uv run python -m ballast.evals.cli dataset-from-traces \
    --logfire-query "agent.run AND service.name='my-app' AND time > now() - 7d" \
    --output datasets/last-week.json \
    --sample 200
```

Output is a JSON file:
```json
{
  "name": "last-week",
  "cases": [
    {"id": "case-001", "input": "Summarize the latest...", "expected_output": {...}},
    {"id": "case-002", "input": "What's the weather in...", "expected_output": "..."}
  ]
}
```

### 2. Run new agent version against it

```python
from ballast.evals import Dataset, EvalRunOutput
from ballast import SchemaAdherenceScorer


dataset = Dataset.from_file("datasets/last-week.json")
new_agent = build_my_new_agent()

scorer = SchemaAdherenceScorer(expected_output_type=MyOutput)

report = await dataset.run(
    runner=lambda case: new_agent.run(case.input),
    scorers=[scorer],
)

print(f"Pass rate: {report.pass_rate:.1%}")
print(f"Mean score: {report.mean_score:.3f}")
for case_id, case_report in report.cases.items():
    if not case_report.passed:
        print(f"FAIL {case_id}: {case_report.reason}")
```

`Dataset.run(...)` parallelizes case execution, captures outputs + scorer verdicts.

## Custom scorers

`Scorer` Protocol:
```python
from ballast.evals import Scorer, EvalCase, ScoreResult


class _CitationCountScorer:
    """Pass if response contains at least N citations."""
    def __init__(self, min_citations: int = 1):
        self._min = min_citations

    async def score(self, case: EvalCase, output) -> ScoreResult:
        citations = count_citations(output)
        return ScoreResult(
            passed=citations >= self._min,
            score=min(1.0, citations / self._min),
            reason=f"found {citations} citations; expected ≥ {self._min}",
        )


report = await dataset.run(
    runner=lambda case: agent.run(case.input),
    scorers=[
        SchemaAdherenceScorer(expected_output_type=MyOutput),
        _CitationCountScorer(min_citations=2),
    ],
)
# Each case gets a list of ScoreResult — one per scorer.
```

Apps wire any number of scorers; report aggregates.

## LLM-as-Judge scorer

```python
from ballast import LLMJudge


judge = LLMJudge(
    rubric="Output is concise, factual, and free of speculation.",
    judge_model="anthropic:claude-3-5-haiku",
    threshold=0.7,
)


class _JudgeScorer:
    def __init__(self, judge): self._judge = judge

    async def score(self, case, output) -> ScoreResult:
        verdict = await self._judge.grade(output)
        return ScoreResult(
            passed=verdict.passed,
            score=verdict.score,
            reason=verdict.reason,
        )


report = await dataset.run(
    runner=lambda case: agent.run(case.input),
    scorers=[_JudgeScorer(judge)],
)
```

## Pairwise comparison: model A vs model B

```python
agent_a = Agent(model="openai:gpt-4o")
agent_b = Agent(model="anthropic:claude-3-5-sonnet")

report_a = await dataset.run(runner=lambda case: agent_a.run(case.input), scorers=[...])
report_b = await dataset.run(runner=lambda case: agent_b.run(case.input), scorers=[...])

print(f"A pass rate: {report_a.pass_rate:.1%}")
print(f"B pass rate: {report_b.pass_rate:.1%}")
print(f"Δ score: {report_b.mean_score - report_a.mean_score:+.3f}")
```

Or use pairwise judge for direct preference scoring (see [run-llm-judge-evaluation.md](run-llm-judge-evaluation.md)).

## Save reports for CI gates

```python
import json
report_path = "reports/eval-{branch}-{commit}.json"
with open(report_path, "w") as f:
    json.dump(report.model_dump(), f, indent=2)
```

In CI:
```bash
# Fail the build if eval pass rate drops
uv run python -c "
import json
report = json.load(open('reports/eval-${BRANCH}.json'))
baseline = json.load(open('reports/eval-main-latest.json'))
assert report['pass_rate'] >= baseline['pass_rate'] - 0.02, 'Regression!'
"
```

## Parallel case execution

```python
report = await dataset.run(
    runner=lambda case: agent.run(case.input),
    scorers=[scorer],
    concurrency=10,        # 10 cases in parallel
)
```

Be mindful of rate limits — 10 concurrent runs × 5 model calls each = 50 concurrent OpenAI requests.

## Filter / slice datasets

```python
# Run only on cases tagged "publishing"
publishing_cases = [c for c in dataset.cases if "publishing" in c.tags]
report = await dataset.run(
    runner=lambda case: agent.run(case.input),
    scorers=[scorer],
    cases=publishing_cases,
)
```

For larger datasets, you might split per-tag and report separately for clear diagnosis.

## Capture failing cases as regression tests

```python
for case_id, case_report in report.cases.items():
    if not case_report.passed:
        # Save case for unit-test corpus
        regression_dataset.add(dataset.get(case_id))

regression_dataset.save("datasets/regressions.json")
```

Run `regressions.json` in CI on every PR — guarantees you don't break previously-fixed cases.

## Caveats

- **Production traces may contain PII.** Sanitize inputs before saving. The `dataset-from-traces` CLI has `--scrub-pattern` regex options.
- **Outputs are stochastic.** A pass-rate of 95% doesn't mean the agent is "almost perfect" — it means it's stable enough that small changes (different model version, new system prompt) will show up as multi-percentage-point swings.
- **`expected_output` from traces is what production RETURNED, not the ideal answer.** Treat it as a snapshot, not ground truth. For ground-truth eval, you need hand-labeled cases.
- **Eval cost.** Each case × each scorer × each agent run. For 200 cases × 3 scorers × 2 agents = 1,200 LLM calls. Budget accordingly.

## Related

- [run-llm-judge-evaluation.md](run-llm-judge-evaluation.md) — judges in production
- [add-logfire-tracing.md](add-logfire-tracing.md) — trace source
- Reference: `reference/evals/dataset.md`
- Reference: `reference/evals/scorers.md`

# 23. Evals

**Prerequisites:** [22-observability.md](22-observability.md).

**What you'll learn:** how to use `LLMJudge` for online grading of every agent response; how to capture production traces into a `Dataset` and replay against a new agent version; the `Scorer` Protocol for custom grading; pairwise comparison for model selection.

## Sections

1. The case for continuous quality grading
2. `LLMJudge` — rubric-based grading; sync vs async modes
3. `JudgeAfterRun` — the capability that wires the judge to every run
4. Per-thread verdict persistence as a thread event
5. `Dataset` + `EvalCase` + `EvalReport` data model
6. The `dataset-from-traces` CLI: turn production into eval set
7. `Scorer` Protocol: writing your own
8. The built-in `SchemaAdherenceScorer`
9. LLM-judge as a scorer
10. Pairwise comparison for A/B testing
11. CI gates on regression
12. Sanitizing PII from captured traces
13. Where to go next

## Next

[24-durability.md](24-durability.md) — the DBOS layer in depth.

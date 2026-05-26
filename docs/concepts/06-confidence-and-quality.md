# 6. Confidence and quality — `Scored[T]`

**Prerequisites:** [03-structured-output.md](03-structured-output.md).

**What you'll learn:** how to wrap any value with rationale + confidence signal using `Scored[T]`; why named labels avoid mean-reversion that hits numeric scales; how `Scored` composes with patterns like `MapReduce` for confidence-aware filtering.

## Sections

1. The case for explicit confidence: downstream filtering, ranking, escalation
2. `Scored[T, ConfidenceT]` — generic wrapper with `value + rationale + confidence`
3. Default labels: `Literal["low", "medium", "high"]` (not 1-10)
4. Why required `rationale: str` forces chain-of-thought
5. `frozen=True` immutability and what it means in practice
6. Custom `ConfidenceT` shapes (int, binary, app-specific)
7. Helpers: `filter_by_min_confidence`, `rank_by_confidence`, `aggregate_by_confidence`
8. Composition with `Ref[T]`: scan_output recurses naturally into `Scored.value`
9. Composition with `CircuitBreaker.is_success`: treat low-confidence runs as failures
10. Where to go next

## Next

[07-capabilities.md](07-capabilities.md) — cross-cutting agent-run hooks.

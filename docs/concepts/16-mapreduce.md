# 16. MapReduce — for documents bigger than context

**Prerequisites:** [14-patterns-intro.md](14-patterns-intro.md), [06-confidence-and-quality.md](06-confidence-and-quality.md).

**What you'll learn:** why a single large prompt loses information in the middle of long documents; how MapReduce shards into bounded chunks; how the framework's `MapReduce` handles concurrent map + global reduce + hierarchical collapse; how confidence-aware reduce filters noise.

## Sections

1. The "Lost in the Middle" effect — why one big prompt fails
2. The MapReduce mental model: sharded extraction + global synthesis
3. `MapReduce(map_step=..., reduce_step=...)` vs `map_agent=... reduce_agent=...`
4. Concurrency bounded by `map_concurrency` semaphore
5. Combining with `Scored[T]` for confidence-aware reduce
6. Per-call retries (`retries`, `retry_backoff_seconds`) for flaky chunks
7. Hierarchical collapse (`collapse_threshold`) for hundreds-of-chunks documents
8. Wrapping map calls with `CircuitBreaker` for graceful skipping
9. Replay-safety via `@Durable.workflow` + per-step memoisation
10. Where to go next

## Next

[17-divergent-convergent.md](17-divergent-convergent.md) — the other fan-out pattern, for variety.

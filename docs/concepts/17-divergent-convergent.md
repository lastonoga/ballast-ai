# 17. Divergent-Convergent

**Prerequisites:** [14-patterns-intro.md](14-patterns-intro.md), [16-mapreduce.md](16-mapreduce.md).

**What you'll learn:** the "Artificial Hivemind" failure mode where LLMs produce homogeneous safe answers; how CREATIVEDC-style divergent exploration + convergent synthesis breaks out of local optima; how to stream typed progress events to a UI.

## Sections

1. The Hivemind problem: LLMs converge on the first plausible answer
2. The two-phase structure: divergent (variety) + convergent (synthesis)
3. `DivergentConvergent(divergent_agent, convergent_agent, branch_count, dedup_threshold, embedder)`
4. Embedding-based dedup to drop redundant hypotheses
5. Optional `verifier` step between dedup and convergence
6. Streaming typed progress events: `on_progress` callback
7. Different models per phase (cheap-for-explore, premium-for-synthesize)
8. Multi-provider divergence to escape architectural blind spots
9. Embedding `DivergentConvergent` inside `MapReduce`
10. When NOT to use this pattern
11. Where to go next

## Next

[18-plan-and-execute.md](18-plan-and-execute.md) — the planning family.

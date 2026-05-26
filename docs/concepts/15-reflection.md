# 15. Reflection

**Prerequisites:** [14-patterns-intro.md](14-patterns-intro.md), [06-confidence-and-quality.md](06-confidence-and-quality.md).

**What you'll learn:** how Writer-Critic-Refiner loops improve output quality without running forever; the role of iteration caps; how to short-circuit with `accept_if`; how to embed reflection inside other patterns via `as_critique`.

## Sections

1. The case for iterative refinement (and the case against it for simple queries)
2. The three roles and what each does
3. The iteration cap — non-negotiable in production
4. `accept_if` for early termination
5. `TypedLoopGuard` baked in: short-circuit on output convergence
6. Composition with `Scored[T]` for confidence-aware acceptance
7. The `as_critique` adapter for embedding a critique loop inside MapReduce / PlanAndExecute steps
8. Cost trade-off: 3× LLM calls per iteration
9. Where to go next

## Next

[16-mapreduce.md](16-mapreduce.md) — the fan-out family.

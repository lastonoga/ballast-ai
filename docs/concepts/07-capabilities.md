# 7. Capabilities

**Prerequisites:** [01-agents.md](01-agents.md), [04-dependencies-and-state.md](04-dependencies-and-state.md).

**What you'll learn:** what a `BallastCapability` is, what hooks it gives you (`for_run`, `before_model_request`, `after_model_request`, `after_run`, `wrap_run`), how capabilities compose by stacking, and the per-run isolation rule that makes them safe.

## Sections

1. Capabilities are cross-cutting concerns layered onto an Agent
2. The hook lifecycle: when each one fires within a single `agent.run()`
3. `for_run(ctx)` — return a per-run clone so counters don't leak
4. `before_model_request` — peek/modify the prompt before the LLM call
5. `after_model_request` — see the response; the right place for token counting
6. `after_run` — final result is in your hands; persist / grade / log
7. `wrap_run` — the all-encompassing hook (used by `ApprovalCapability`)
8. Stacking: the order in `capabilities=[A, B, C]` is the firing order
9. A tour of the built-in capabilities (next 6 chapters use these)
10. Writing your own capability — pattern + smallest-viable example
11. Where to go next

## Next

[08-running-an-app.md](08-running-an-app.md) — assembling the agent + capabilities into a runnable FastAPI app.

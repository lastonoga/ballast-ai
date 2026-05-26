# Tutorials

Learning-oriented walkthroughs. Each tutorial builds something real from scratch in 30–60 minutes.

> **You're here because:** you're new to Ballast and want to learn by doing. Pick the first one, follow it end-to-end, and you'll have a working agent + UI by lunch.

## Tutorial track

| # | Title | What you'll build | Time |
|---|---|---|---|
| 01 | **Quickstart: a notes-taking agent** | Full backend + frontend with chat-style UI, tool calls, persistence | 30 min |
| 02 | **First custom capability** | A `WordCountGuard` that limits agent response length, stacked on top of `BudgetGuard` | 20 min |
| 03 | **First CoALA Unit** | A `ResearchSummarize` unit deployed as a tool, then as a workflow | 30 min |
| 04 | **Add HITL approval** | Wire `ApprovalCapability` so `publish_post` requires human approval before running | 25 min |
| 05 | **Add resilience** | Stack `CircuitBreaker` + `BudgetGuard` + `GoalDriftDetector` on the notes agent | 30 min |

## Conventions

- Every tutorial assumes Python 3.11+, `uv` (or `pip`), and OpenRouter/OpenAI API access.
- Code lives at `examples/notes-app/` — every tutorial extends or modifies it.
- Each tutorial ends with a verified test run + a screenshot of the working result.

---

**Don't have the time?** Read [explanation/why-ballast.md](../explanation/why-ballast.md) for the 5-minute pitch.

**Have a specific problem to solve?** Skip tutorials, go to [how-to](../howto/).

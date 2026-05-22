# Notes — dogfood reference app

Iteratively-built reference application that exercises `ballast-ai-engine` one layer at a time. Each iteration adds exactly one new framework concern; at every checkpoint we pause, write what we learned in `RETRO.md`, and (if needed) change the framework before moving on.

## Domain

Simple "AI assistant for notes": users chat with an assistant that can create, edit, delete, and search short text notes. Tenant-scoped from day one.

## Layout

```
examples/notes-app/
  frontend/   # Next.js 14 + shadcn/ui + assistant-ui (PURE assistant-ui — no custom chat code)
  backend/    # FastAPI app built via Engine.fastapi_app() + pydantic-ai over OpenRouter
  RETRO.md    # combined retro per iteration (frontend + backend learnings)
```

## Iterations

1. **UI shell** — `frontend/` only, assistant-ui mock runtime, threads + chat work end-to-end without a backend.
2. **Backend + single agent + streaming** — FastAPI via `Engine.fastapi_app()`, one OpenRouter agent (Qwen, JSON output), AG-UI streaming. Frontend still on mock — wiring happens in iteration 3.
3. **Notes domain + tools** — SQLModel domain, agent gets CRUD tools, frontend points at backend.
4. **HITLGate + UIChannel** — approval before mutations.
5. **Reflection** — writer/critic loop.
6. **Capabilities** — BudgetGuard + GroundedRetry.
7. **ObservabilityProvider** — logfire traces.
8. **Evals from this app's traces** — `dataset-from-traces` against the app's real DBOS state.

Each iteration corresponds to one or two PRs and ends with a RETRO entry. Framework gaps surfaced here become tasks for the next round of changes in `src/ballast/`.

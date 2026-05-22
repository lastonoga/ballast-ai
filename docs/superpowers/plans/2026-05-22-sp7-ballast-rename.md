# SP7: Rename framework to Ballast + Variant 2 API — Plan

**Goal:** Rename framework from `pydantic-ai-stateflow` to `ballast-ai`. Replace `sf.create_app(...)` with `Ballast(settings).use(providers...).fastapi(...)` pattern.

**Confirmed:**
- Package: `pydantic_ai_stateflow` → `ballast`; PyPI: `pydantic-ai-stateflow` → `ballast-ai`
- CLI: `stateflow` → `ballast`
- Env prefix: `STATEFLOW_` → `BALLAST_`
- Error codes: `STATEFLOW_*` → `BALLAST_*`
- Pyproject section: `[tool.stateflow]` → `[tool.ballast]`
- Classes:
  - `StateflowSettings` → `BallastSettings`
  - `StateflowError` → `BallastError`
  - `StateflowErrorMiddleware` → `BallastErrorMiddleware`
  - `StateflowAgent` → `BallastAgent`
  - `StateflowDurableAgent` → `DurableAgent` (just `DurableAgent`, namespaced via `from ballast import`)
  - `StateflowCapability` → `BallastCapability`
- Usage: `import ballast` (no `as bl` alias)
- API: **Variant 2** — `Ballast(settings).use(DBOSProvider(), ThreadsProvider(...), EventsProvider(...))`; `.fastapi(routers=[...])` returns FastAPI app

## Phases

### Phase 1 — Package rename (mechanical)
- `mv src/pydantic_ai_stateflow src/ballast`
- `sed`-replace `pydantic_ai_stateflow` → `ballast` across .py + .toml + .md
- `sed`-replace `pydantic-ai-stateflow` → `ballast-ai`
- StateflowX class renames (order matters — most specific first)
- `STATEFLOW_` → `BALLAST_`, `[tool.stateflow]` → `[tool.ballast]`
- CLI entrypoint rename
- Run framework + notes-app tests
- Commit

### Phase 2 — Variant 2 API (Ballast + Providers)
- New `ballast/app.py`:
  - `Ballast` class (replaces `create_app` function)
  - `Provider` Protocol
- New `ballast/providers/`:
  - `DBOSProvider(dbos: DBOSConfig | None = None)`
  - `ThreadsProvider(thread_repo)`
  - `EventsProvider(event_log, event_stream)`
  - `ObservabilityProvider(config)`
- `Ballast.fastapi(*, cors=, routers=...)` returns FastAPI app
- Migrate notes-app main.py to Ballast + providers
- Tests
- Commit

# SP2: Settings + Structured Errors — Implementation Plan

**Goal:** Centralize env-var config in `StateflowSettings(BaseSettings)`; introduce `StateflowError` hierarchy with `code`/`detail`/`hint`/`context` + HTTP middleware for structured `application/problem+json` responses.

**Spec:** `docs/superpowers/specs/2026-05-22-sp2-settings-errors-design.md`

---

## Task list

- T1: `settings.py` — `StateflowSettings(BaseSettings)` + sub-models + `_SettingsProxy` + tests
- T2: `errors.py` — `StateflowError` base + flat hierarchy (no generic RuntimeError_/ValidationError_) + tests; migrate existing `EngineInvariantViolation` → `ConfigurationInvariantViolation` and other framework errors to subclass `StateflowError`
- T3: `api/error_middleware.py` — `install_error_handlers(app)` + tri-state expose_tracebacks; auto-installed by `create_app` (gated on `settings.api.install_error_middleware`)
- T4: Migrate notes-app env reads (`OPENROUTER_API_KEY`, `DBOS_DATABASE_URL`, `OPENROUTER_MODEL`) → `settings.llm.openrouter.*` / `settings.dbos.database_url`
- T5: Migrate `HTTPException` callsites in `api/` → `StateflowError` subclasses (ThreadNotFound, etc.)
- T6: `format_error()` impl in `errors.py` (rich fallback to plain) + sample colored output snapshot
- T7: Final test sweep + commit

After all tasks pass: framework + notes-app tests green; `/threads/{nonexistent}` returns `application/problem+json` with code `STATEFLOW_PERSISTENCE_THREAD_NOT_FOUND`.

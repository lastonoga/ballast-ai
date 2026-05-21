# SP1: Core DI/registration rewrite — Implementation Plan

**Goal:** Replace Container/Engine/ServiceProvider with hybrid model (FastAPI Depends for HTTP + explicit constructor for flows + decorator registration). Delete factory anti-patterns. Migrate notes-app + tests.

**Architecture:** Decorators handle registration (kebab-name + HTTP autogen metadata); explicit instances passed to `sf.create_app(workflows=[...], agents=[...])`. Tests use `sf.testing.TestEngine` with dependency_overrides.

**Spec:** `docs/superpowers/specs/2026-05-22-sp1-core-rewrite-design.md`

---

## Task list

- T1: Skeleton — `testing/` stubs + `observability/config.py` (ObservabilityConfig)
- T2: `@sf.workflow` + `@sf.stateflow_agent` decorators (new `runtime/workflows.py`, augment `runtime/agents.py`)
- T3: `api/deps.py` rewrite + module-level framework routers (threads, streaming, dbos)
- T4: Workflow HTTP autogen router builder
- T5: `runtime/app.py` with `create_app()` full impl
- T6: TestEngine real impl + MockAgent + MockFlow + pytest_plugin
- T7: Top-level `__init__.py` — add new exports
- T8: Migrate notes-app brainstorm_flow.py (@sf.workflow) + delete brainstorm_router.py
- T9: Migrate notes-app main.py to `sf.create_app`
- T10: Migrate notes-app tests to TestEngine
- T11: Delete old code (container/engine/providers/ServiceProvider) + framework test cleanup
- T12: Final test sweep + verification

After all tasks pass: framework `uv run pytest tests/ -x -q` + notes-app `cd examples/notes-app/backend && uv run pytest -x -q` both green.

---

Each task is dispatched as one implementer subagent with self-contained brief. Subagent self-reviews; controller checks test gates between tasks; failures trigger fix subagents.

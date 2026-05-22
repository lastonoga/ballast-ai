# SP6: Engine config-holder; drop Infra/RunContext — Implementation Plan

**Goal:** Replace `Infra` + `RunContext` (dataclasses passed everywhere) with a process-singleton `Engine` config holder constructed by `sf.create_app(thread_repo=, event_log=, event_stream=, ...)`. Framework code accesses via lazy `sf.get_engine()`. Per-call data (parent_thread_id, workflow_id) goes as explicit kwargs.

**Architecture shift:** apps own their Thread model + ThreadRepository (subclass framework Protocols + reference impls). Framework's only repository concern is providing Protocols + InMemory reference impls. App passes its repos into `sf.create_app(...)`; framework code reads via `get_engine()`.

**Confirmed in brainstorm:**
- Drop `Infra` + `RunContext` entirely
- Engine is frozen dataclass with `thread_repo`, `event_log`, `event_stream`, cached `broadcaster`
- `sf.get_engine()` lazy getter raises ConfigurationError if create_app not called
- `_set_engine(engine)` called only by `create_app` (idempotent if same engine)
- HTTP routes use `Depends(get_engine_dep)` which reads `request.app.state.engine`
- Apps define own Thread model (subclass) + own Repository (subclass InMemory or implement Protocol)
- Per-call data → explicit kwargs in method/primitive signatures (`parent_thread_id`, etc.)

---

## Task list

- **D1: Framework refactor** — `Engine` dataclass + lazy singleton; `Infra`/`RunContext` removed; `create_app(thread_repo=, event_log=, event_stream=, ...)` signature; ABCs drop ctx args, use `get_engine()`; `stream_response`/`cancel_thread_workflows` primitives drop ctx kwarg; `__init__.py` exports updated; framework tests migrated
- **D2: Notes-app refactor** — `repositories/thread.py`, `repositories/events.py`, `streams.py` with module-level singletons; flow/agent methods drop ctx args + use direct imports; main.py passes to create_app; tests migrated

After all dispatches: framework + notes-app green.

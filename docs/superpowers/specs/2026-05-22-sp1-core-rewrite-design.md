# Sub-project 1: Core DI/registration rewrite

**Status:** Approved (design)
**Date:** 2026-05-22
**Scope:** Framework runtime + notes-app + tests

## Problem

The framework grew a Spring-style DI container (`Container` + `ServiceProvider` +
`Engine`) for a workload that doesn't justify it. Every app pays this tax:

- `examples/notes-app/backend/src/notes_app/main.py:75` defines `build_app(*,
  thread_repo=None, notes_agent=None, notes_repo=None, todo_approval_agent=None,
  todo_flow=None, brainstorm_flow=None, event_log=None, event_stream=None,
  manage_dbos_lifecycle=True)` — nine optional kwargs whose only purpose is
  test injection. This is the `build_X(dep=None)` anti-pattern called out in
  user memory (`project_anti_pattern_build_factory_as_di.md`).
- `Engine` (`src/pydantic_ai_stateflow/runtime/engine.py:32`) ceremoniously
  registers a `providers[]` list to bind values onto a `Container`
  (`src/pydantic_ai_stateflow/runtime/container.py:58`) that then nobody
  actually resolves: routes don't call `Depends(get_container)` (declared at
  `src/pydantic_ai_stateflow/api/deps.py:11` but unused by `threads.py`,
  `streaming/router.py`, `dbos_router.py`, `brainstorm_router.py`); workflows
  read deps directly from constructor args. The container exists only so
  `_bind_domain_repos` (`main.py:163`) can later call
  `app.state.container.bind(NoteRepository, notes)` for a single test
  (`tests/test_smoke.py:165`).
- Agent registration is imperative (`register_agent(agent)` —
  `src/pydantic_ai_stateflow/runtime/agents.py:350`), called from app bootstrap
  with implicit ordering rules and zero static checkability.
- Workflows have no registration story at all — apps hand-write a
  `build_brainstorm_router(*, flow, thread_repo)` wrapper
  (`examples/notes-app/backend/src/notes_app/brainstorm_router.py:37`) per
  workflow that needs an HTTP entry point, and remember to include it in the
  Engine's `extra_routers` list.
- Tests pay extreme boilerplate: `_unique_flow`, `_unique_brainstorm`,
  `_FakeNotesAgent` and an 11-line per-test `build_app(...)` invocation
  (`tests/test_smoke.py:32-117, 157-193`). Most of it exists to mint unique
  `config_name` strings so DBOS doesn't reject duplicate registrations.

The Container/Engine ceremony provides no benefit (no override flow apps use,
no FastAPI integration apps use) while taxing every app and every test.

## Goal

After this rewrite, an app's `main.py` looks like:

```python
import pydantic_ai_stateflow as sf
from notes_app.agent import NotesAgent  # @sf.stateflow_agent — side-effect import
from notes_app.brainstorm_flow import BrainstormFlow  # @sf.workflow
from notes_app.todo_flow import TodoApprovalFlow  # @sf.workflow

app = sf.create_app(
    workflows=[TodoApprovalFlow, BrainstormFlow],
    cors=sf.CORSConfig.permissive_dev(),
    dbos=sf.DBOSConfig(name="notes-app", system_database_url=DSN),
)
```

That's all the wiring an app does. Tests:

```python
def test_thread_round_trip(client):
    r = client.post("/threads", json={})
    assert r.status_code == 201
```

…where `client` is a pytest fixture provided by `sf.testing` that gives a
configured `TestClient` over a fresh DBOS executor with InMemory repos and a
TestModel-backed `NotesAgent`. Overrides are explicit and at the test layer,
not at the wiring layer.

The framework loses ~500 lines of Container/Engine/Provider scaffolding;
apps lose `build_app(dep=None)` and all its callers.

## Non-goals

- **Settings module** (env-var-driven config, secrets, `STATEFLOW_*` knobs) —
  deferred to SP2.
- **Error hierarchy unification** (`EngineInvariantViolation` etc. rename /
  consolidation) — deferred to SP2.
- **CLI / migrations / project scaffolding** (`stateflow init`, `stateflow
  workflow add`, alembic generation) — deferred to SP3.
- **Per-tenant DI scoping** — explicitly out. Tenant context lives in
  `Thread.metadata`, not the framework.
- **Workflow versioning / blue-green rollouts** — out.
- **Removing pre-existing `app.state.{notes_repo, todo_flow, ...}` reads in
  tests** that don't go through `app.dependency_overrides` — the rewrite
  removes the need to write them; existing tests rewrite to use TestEngine
  overrides.
- **Persistence layer change** — Postgres vs InMemory choice stays where it
  is (factory-built repos, no DI).

## Design

### A. DI model: hybrid (FastAPI Depends + explicit constructor)

**HTTP routes** use FastAPI's native `Depends` for any per-request dep. Repos
and singletons are resolved by `Depends(get_thread_repo)` style helpers that
read from `request.app.state`. There is no framework Container interposed.

```python
# src/pydantic_ai_stateflow/api/deps.py (rewritten)
from fastapi import Request
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository

def get_thread_repo(request: Request) -> ThreadRepository:
    return request.app.state.thread_repo

def get_event_log(request: Request) -> EventLogRepository:
    return request.app.state.event_log

def get_event_stream(request: Request) -> EventStream:
    return request.app.state.event_stream
```

Framework routers become plain `APIRouter`s (no `build_X(thread_repo=...)`
factory):

```python
# src/pydantic_ai_stateflow/api/threads.py (rewritten)
router = APIRouter()

@router.get("/threads/{thread_id}")
async def get_thread(
    thread_id: UUID,
    repo: ThreadRepository = Depends(get_thread_repo),
) -> dict[str, Any]:
    ...
```

Apps mount the module-level router; tests override deps with
`app.dependency_overrides[get_thread_repo] = lambda: my_test_repo`.

**Workflows / Flows / Agents** get deps via explicit constructor at instance
construction time. No DI container, no service-locator pattern. A workflow's
deps are arguments to `__init__`; tests construct instances directly with
test doubles.

```python
# Construction is explicit and statically checked:
flow = TodoApprovalFlow(
    notes_repo=InMemoryNoteRepository(),
    thread_repo=InMemoryThreadRepository(),
    event_log=InMemoryEventLogRepository(),
    event_stream=InProcessEventStream(),
)
```

**Cross-cutting concerns** (observability, the event-stream layer, the thread
event broadcaster) are direct calls in `sf.create_app(...)` — not Providers
registered onto a container:

```python
def create_app(*, workflows, dbos, cors=None, observability=None, ...):
    observability = observability or ObservabilityConfig.default()
    observability.install()  # configures logfire + instrument_pydantic_ai
    ...
```

There is no `bind` / `get` / `aget` / `has`. There is no `ServiceProvider`
Protocol. If something is cross-cutting, it gets a config dataclass and a
constructor argument on `create_app`.

### B. Decorator-based registration

#### B.1 `@sf.stateflow_agent` on the agent class

```python
# src/pydantic_ai_stateflow/runtime/agents.py (rewritten section)
def stateflow_agent(cls: type[StateflowAgent]) -> type[StateflowAgent]:
    """Register the agent class in the process-wide registry.

    Name is auto-derived from the class name as kebab-case:
    ``NotesAgent`` → ``"notes-agent"``, ``TodoApprovalAgent`` →
    ``"todo-approval-agent"``. No suffix stripping. No ``name=`` kwarg.

    Side effect: ``_class_registry[cls.kebab_name()] = cls``.
    Instances are NOT created here — the framework calls
    ``cls()`` lazily on first `get_agent(name)` resolution OR explicitly
    via `create_app(agents=[NotesAgent(...)])` when the agent needs
    constructor args.
    """
    name = _kebab_case(cls.__name__)
    if name in _class_registry and _class_registry[name] is not cls:
        raise ValueError(f"Duplicate @stateflow_agent name {name!r}")
    cls.name = name  # set the ClassVar so existing code paths keep working
    _class_registry[name] = cls
    return cls
```

Usage:

```python
@sf.stateflow_agent
class NotesAgent(sf.StateflowDurableAgent):
    metadata_model = NotesMetadata

    def build_agent(self) -> Agent[NoteToolDeps, str]:
        ...

    async def build_deps(self, *, thread, message) -> NoteToolDeps:
        ...
```

The agent class no longer needs to set `name: ClassVar[str]`. It can still
do so explicitly to override the auto-derived name (escape hatch for
backwards compat with serialized `Thread.agent` strings); the decorator
respects an explicit non-default `name` ClassVar.

Instance registration: `create_app(agents=[NotesAgent(...)])` accepts **only
already-constructed instances** — no auto-instantiation. The decorator
mapped the class name to the kebab-name; `type(instance).__name__` is used
to look up the kebab-name + register the instance in `app.state.agents[name]`.

**`register_agent(instance)` is deleted.** All call sites move to
`@stateflow_agent` + passing the instance to `create_app(agents=[...])`.

#### B.2 `@sf.workflow` on the workflow class

```python
# src/pydantic_ai_stateflow/runtime/workflows.py (new)
def workflow(
    cls: type | None = None,
    /,
    *,
    name: str | None = None,
    input: type[BaseModel],
    output: type[BaseModel],
) -> type | Callable[[type], type]:
    """Register a workflow class for HTTP auto-mount + DBOS wrapping.

    Effects on the decorated class ``Cls``:

    1. Apply ``@Durable.dbos_class()`` so ``Cls.__init__`` calls super
       with a stable ``config_name`` and DBOSConfiguredInstance metadata
       is set.
    2. Validate ``Cls.run`` is an ``async def run(self, input: InputType)
       -> OutputType``. If signature doesn't match, raise at import time
       with a precise diagnostic.
    3. Validate ``Cls.run`` is wrapped in ``@Durable.workflow()``.
    4. Record (name, Cls, input, output) in the process-wide workflow
       registry. ``name`` defaults to kebab-case of ``Cls.__name__``
       with trailing ``Flow`` / ``Workflow`` retained verbatim
       (no suffix stripping).

    Apps can still add other methods freely (e.g.
    ``TodoApprovalFlow.open``, ``on_decision``). Only ``run`` is the
    HTTP entry point.
    """
```

Usage:

```python
class BrainstormTask(BaseModel):
    topic: str
    parent_thread_id: UUID

class BrainstormOutcome(BaseModel):
    helper_thread_id: UUID
    proposed_title: str
    proposed_body: str

@sf.workflow(input=BrainstormTask, output=BrainstormOutcome)
class BrainstormFlow(DBOSConfiguredInstance):
    def __init__(self, *, todo_flow, divergent, broadcaster,
                 config_name="brainstorm-flow"):
        super().__init__(config_name=config_name)
        ...

    @sf.Durable.workflow()
    async def run(self, task: BrainstormTask) -> BrainstormOutcome:
        ...
```

The decorator returns the same class so multiple decorators (`@sf.workflow` +
existing `@Durable.dbos_class` if app prefers explicit) compose. The
decorator is idempotent: if `@Durable.dbos_class()` is already applied,
applying it a second time is a no-op (the existing `Durable.dbos_class()`
implementation checks for prior application; if not, we add that check).

`name=` is honored as override. `input=` / `output=` are mandatory — no
implicit inference from `run`'s signature, because users will get cryptic
import-time failures from `get_type_hints` quirks if `from __future__ import
annotations` is in play.

### C. Workflow HTTP auto-generation

For every `@sf.workflow`-decorated class, the framework auto-mounts:

```
POST /workflows/<name>
    Body: <InputType JSON>
    Response: { "workflow_id": str, "started_at": str (ISO) }
```

Default semantics is **fire-and-forget**: start the DBOS workflow with a
deterministic id and return the handle id. Apps wanting blocking semantics
opt in via:

```python
@sf.workflow(input=..., output=..., blocking=True)
class FooFlow:
    ...
```

When `blocking=True`, the endpoint awaits the workflow and returns the
output model directly with HTTP 200.

**Workflow-id determinism.** Default id is
`f"{name}:{sha256(input.model_dump_json())[:16]}"` so identical bodies
collapse to the same in-flight workflow (matches the manual
`brainstorm_router` logic). Apps override via a class-method:

```python
@sf.workflow(input=BrainstormTask, output=BrainstormOutcome)
class BrainstormFlow:
    @staticmethod
    def workflow_id(task: BrainstormTask) -> str:
        return f"brainstorm:{task.parent_thread_id}:{abs(hash(task.topic))}"
```

**Instance resolution.** `create_app(workflows=[brainstorm_instance, ...])`
takes **already-constructed instances only** — no auto-instantiation, no
factory callbacks, no DI magic. The decorator stored `(name, input_type,
output_type)` as ClassVars on the class; `create_app` looks them up via
`type(instance)` and uses the kebab-name as the registry key.

Resolved instances are stored in `app.state.workflows[name] = instance`.

Apps that need **dynamic workflow construction** (per-tenant, per-request)
do NOT register such workflows with `@sf.workflow` for HTTP autogen.
They just construct + invoke at runtime:

```python
async def per_tenant_brainstorm(tenant_id: str, task: BrainstormTask) -> BrainstormOutcome:
    flow = BrainstormFlow(
        todo_flow=get_tenant_todo_flow(tenant_id),
        broadcaster=get_tenant_broadcaster(tenant_id),
        config_name=f"brainstorm-{tenant_id}",
    )
    handle = await sf.Durable.start_workflow(flow.run, task)
    return await handle.get_result()
```

The decorator is for HTTP-exposed singleton workflows; dynamic workflows
just use the class as a normal Python class.
The auto-generated endpoint resolves via:

```python
def get_workflow_instance(name: str):
    def _resolver(request: Request):
        return request.app.state.workflows[name]
    return _resolver

# inside the auto-generated route:
async def _start(
    body: InputModel,
    instance = Depends(get_workflow_instance(name)),
):
    ...
```

This keeps tests symmetrical: `app.dependency_overrides[get_workflow_instance(name)]
= lambda: mock_flow` overrides the resolution.

**Mount mechanism.** `create_app` walks the workflow registry and calls
`app.include_router(_build_workflow_router(name, cls, instance))` per
registration. There is no separate `mount_workflows(app, *flows)` call —
`create_app` handles it.

### D. TestEngine

`pydantic_ai_stateflow.testing` package, new — single import surface for
test wiring.

```python
# src/pydantic_ai_stateflow/testing/__init__.py

class TestEngine:
    """Pre-wired Stateflow app for tests.

    Defaults:
    - InMemoryThreadRepository / InMemoryNoteRepository / InMemoryEventLogRepository
    - InProcessEventStream
    - Per-instance unique ``config_name`` suffix to avoid DBOS instance
      registry collisions across tests in the same process.
    - DBOS launched on ``__enter__``, destroyed on ``__exit__``.
    - SQLite DBOS system DB in a tempdir, per ``TestEngine`` instance.
    - All registered ``@stateflow_agent``s and ``@sf.workflow``s
      auto-instantiated with the in-memory repos.
    """

    @classmethod
    def default(cls) -> TestEngine: ...

    def override(self, target: type, replacement: Any) -> TestEngine:
        """Override a class registration with a replacement.

        - ``target`` is a Protocol / ABC / concrete class.
        - ``replacement`` is an instance to use everywhere ``target``
          is resolved.

        Mechanism:
        - For ``ThreadRepository`` / ``EventLogRepository`` / ``EventStream``:
          replaces the value stored in ``app.state`` AND sets
          ``app.dependency_overrides[get_<x>] = lambda: replacement``.
        - For ``StateflowAgent`` subclasses: replaces the instance in
          the agent registry under the class's kebab name.
        - For ``@sf.workflow`` classes: replaces the instance in
          ``app.state.workflows[name]`` AND sets the FastAPI
          dependency override.

        Idempotent and chainable.
        """

    def test_client(self) -> TestClient:
        """Returns a context-manager FastAPI TestClient.

        Lifecycle:
        - ``__enter__``: ensures all overrides applied, launches DBOS,
          starts FastAPI lifespan.
        - ``__exit__``: stops lifespan, destroys DBOS (
          ``destroy_registry=False`` to keep ``@Durable`` decorations
          across the test session).
        """
```

Usage replaces the entire `_unique_flow` / `_unique_brainstorm` /
`_FakeNotesAgent` / `build_app(...)` apparatus:

```python
def test_threads_crud_and_streaming_fake():
    engine = TestEngine.default()
    engine.override(NotesAgent, MockAgent.with_output("Hello, world!"))

    with engine.test_client() as client:
        r = client.post("/threads", json={})
        assert r.status_code == 201
        ...
```

**`MockAgent` / `MockFlow` helpers** ship in `sf.testing`:

```python
class MockAgent(StateflowAgent):
    """A StateflowAgent backed by pydantic-ai's TestModel.

    Constructors:
    - ``MockAgent.with_output("text")`` — TestModel that always returns
      ``text`` as a single string output.
    - ``MockAgent.with_outputs(["a", "b", ...])`` — TestModel cycling
      through outputs.
    - ``MockAgent.with_tool_calls([...])`` — TestModel scripted to make
      tool calls (advanced).
    """

class MockFlow:
    """A workflow stub for ``engine.override(SomeFlow, MockFlow.returning(...))``.

    - ``MockFlow.returning(OutputModel(...))`` — ``run`` returns that
      pydantic model.
    - ``MockFlow.raising(SomeError("..."))`` — ``run`` raises.
    """
```

**Pytest plugin.** `sf.testing` exports a pytest plugin (`pytest_plugins =
["pydantic_ai_stateflow.testing.pytest_plugin"]`) providing:

```python
@pytest.fixture
def engine() -> Iterator[TestEngine]:
    e = TestEngine.default()
    yield e

@pytest.fixture
def client(engine) -> Iterator[TestClient]:
    with engine.test_client() as c:
        yield c
```

Tests opt in by adding `pytest_plugins = ["pydantic_ai_stateflow.testing"]`
to their `conftest.py`. The plugin is opt-in (not auto-loaded) so apps that
want different defaults (e.g. Postgres repos for integration tests) can
build their own fixture layer.

**DBOS lifecycle isolation.** Each `TestEngine` instance:

1. Generates a unique `dbos_name = f"stateflow-test-{uuid4().hex[:8]}"`.
2. On `__enter__`: `Durable.init(DBOSConfig(name=dbos_name,
   system_database_url=f"sqlite:///{tempdir}/dbos.sqlite"))` then
   `Durable.launch()`.
3. On `__exit__`: `Durable.destroy(destroy_registry=False)` and deletes the
   tempdir.

For `DBOSConfiguredInstance`-derived classes that demand unique
`config_name` per construction (DBOS rejects duplicates), `TestEngine`
auto-suffixes:

```python
# Inside TestEngine.override or instance auto-construction:
cls(config_name=f"{base_name}-{uuid4().hex[:8]}", ...)
```

Apps don't construct workflow instances themselves in tests — `TestEngine`
does it via the workflow registry, and per-instance suffixing happens
once per `TestEngine`. This kills `_unique_flow` and `_unique_brainstorm`
boilerplate.

### E. Single import surface

`import pydantic_ai_stateflow as sf` exposes the following commonly-used
symbols. Apps target 5-7 imports max for 95% of code.

| `sf.*` symbol            | Source module                              | Purpose                                |
|--------------------------|--------------------------------------------|----------------------------------------|
| `sf.create_app`          | `runtime/app.py` (new)                     | Entry point — builds FastAPI app       |
| `sf.stateflow_agent`     | `runtime/agents.py`                        | Decorator: register agent class        |
| `sf.workflow`            | `runtime/workflows.py` (new)               | Decorator: register workflow class     |
| `sf.Agent`               | alias for `StateflowAgent`                 | Base class for non-durable agents      |
| `sf.DurableAgent`        | alias for `StateflowDurableAgent`          | Base class for durable agents          |
| `sf.Durable`             | `durable.py`                               | DBOS facade (`@Durable.workflow` etc.) |
| `sf.CORSConfig`          | `api/cors.py`                              | CORS knobs                             |
| `sf.DBOSConfig`          | re-export of `dbos.DBOSConfig`             | DBOS init config                       |
| `sf.testing.TestEngine`  | `testing/__init__.py`                      | Pre-wired test app                     |
| `sf.testing.MockAgent`   | `testing/mocks.py`                         | TestModel-backed agent                 |
| `sf.testing.MockFlow`    | `testing/mocks.py`                         | Stub workflow                          |
| `sf.events.BranchCompleted` etc. | re-export of pattern event taxonomy | Live progress events                   |

Patterns stay where they are (`sf.DivergentConvergent`, `sf.MapReduce`,
`sf.Reflection`, `sf.MutationPipeline`, `sf.HITLGate`) — they're already
top-level exports.

Deprecated / removed from `sf.*`:

- `sf.Container`, `sf.DefaultContainer`, `sf.Engine`,
  `sf.EngineInvariantViolation`, `sf.ServiceProvider`, `sf.CoreProvider`,
  `sf.PersistenceProvider`, `sf.ObservabilityProvider` (the *provider*
  class — observability config survives under a new name),
  `sf.register_agent`, `sf.clear_agent_registry`, `sf.get_container`,
  `sf.get_engine`, `sf.build_threads_router`, `sf.build_streaming_router`,
  `sf.build_dbos_router`, `sf.build_health_router`, `sf.build_hitl_router`,
  `sf.build_a2a_router`.

The framework routers stop being functions and become module-level
`APIRouter` instances mounted by `create_app`. Apps that wanted to mount
e.g. only the threads router get `sf.routers.threads` exposed.

### F. App entrypoint: `sf.create_app`

```python
# src/pydantic_ai_stateflow/runtime/app.py (new)

def create_app(
    *,
    # Construction targets — INSTANCES ONLY (no classes, no factories).
    workflows: Sequence[object] = (),
    agents: Sequence[StateflowAgent] = (),

    # Cross-cutting infra (sane defaults; apps override)
    thread_repo: ThreadRepository | None = None,
    event_log: EventLogRepository | None = None,
    event_stream: EventStream | None = None,

    # DBOS
    dbos: DBOSConfig | None = None,
    manage_dbos_lifecycle: bool = True,

    # Observability
    observability: ObservabilityConfig | None = None,

    # HTTP
    cors: CORSConfig | None = None,
    extra_routers: Sequence[APIRouter] = (),
    health_checks: dict[str, Callable[[], Awaitable[bool]]] | None = None,
    on_startup: Sequence[LifespanHook] = (),
    on_shutdown: Sequence[LifespanHook] = (),
) -> FastAPI:
    """Construct the FastAPI app for a Stateflow service.

    Order of operations on import:
    1. Observability configures (logfire + instrument_*) — fails fast
       if instrumentation is misconfigured.
    2. Repos resolve (defaults to InMemory* when not supplied).
    3. ``app.state`` populated with repos for ``Depends(get_*)`` resolution.
    4. For each instance in ``workflows=``: read ``(name, input_type,
       output_type)`` from ClassVars set by ``@sf.workflow`` decorator
       on ``type(instance)``; store at ``app.state.workflows[name]``.
       Raises if instance's class lacks the decorator metadata.
    5. For each instance in ``agents=``: read kebab-name from
       ClassVar set by ``@sf.stateflow_agent`` decorator on
       ``type(instance)``; store at ``app.state.agents[name]``.
    6. Built-in routers mounted: health, threads, streaming, dbos.
       Workflow routers mounted (one per registration).
       ``extra_routers`` appended last.
    7. Lifespan registered: launches DBOS on startup (when
       ``manage_dbos_lifecycle=True``) before any other on_startup hook
       runs, so workflow code paths can use DBOS primitives during
       startup hooks.
    """
```

App example (full notes-app `main.py` post-rewrite):

```python
"""FastAPI entry point for the notes-app backend."""
from __future__ import annotations
import os, tempfile
from pathlib import Path
import pydantic_ai_stateflow as sf
from dotenv import load_dotenv

# Side-effect imports register the @stateflow_agent / @sf.workflow classes.
from notes_app.agent import NotesAgent
from notes_app.todo_approval_agent import NotesTodoApprovalAgent
from notes_app.todo_flow import TodoApprovalFlow
from notes_app.brainstorm_flow import BrainstormFlow, build_brainstorm_flow
from notes_app.notes import InMemoryNoteRepository
from notes_app.notes.routes import notes_router  # module-level APIRouter

load_dotenv()


def _dbos_db_url() -> str:
    override = os.environ.get("DBOS_DATABASE_URL")
    if override:
        return override
    return f"sqlite:///{Path(tempfile.gettempdir()) / 'notes-app.dbos.sqlite'}"


# App-level singletons. Explicit construction in main.py — no DI magic.
notes_repo = InMemoryNoteRepository()
thread_repo = sf.InMemoryThreadRepository()
event_log = sf.InMemoryEventLogRepository()
event_stream = sf.InProcessEventStream()

todo_flow = TodoApprovalFlow(
    notes_repo=notes_repo, thread_repo=thread_repo,
    event_log=event_log, event_stream=event_stream,
)
broadcaster = sf.ThreadEventBroadcaster(
    thread_repo=thread_repo, event_log=event_log, event_stream=event_stream,
)
brainstorm = build_brainstorm_flow(todo_flow=todo_flow, broadcaster=broadcaster)

notes_agent = NotesAgent(
    notes_repo=notes_repo, thread_repo=thread_repo,
    event_log=event_log, event_stream=event_stream, todo_flow=todo_flow,
    config_name="notes-app-notes-agent",
)
approval_agent = NotesTodoApprovalAgent(notes_repo=notes_repo)

app = sf.create_app(
    workflows=[todo_flow, brainstorm],
    agents=[notes_agent, approval_agent],
    thread_repo=thread_repo, event_log=event_log, event_stream=event_stream,
    cors=sf.CORSConfig.permissive_dev(),
    dbos=sf.DBOSConfig(name="notes-app", system_database_url=_dbos_db_url()),
    extra_routers=[notes_router],
    observability=sf.ObservabilityConfig(
        service_name="app", environment="dev",
        instrument_fastapi=False,
    ),
)
app.state.notes_repo = notes_repo  # for the notes router's Depends
```

Construction is fully explicit — no class→callable factory map, no zero-arg
auto-instantiation, no DI graph resolution. Apps wire their singletons
top-of-file the same way a FastAPI app constructs `db_pool = create_pool(...)`
before passing it to dependency factories.

For **dynamic per-request workflows** (multi-tenant, A/B testing), apps
skip the registration path entirely — see §C "Instance resolution" for
the runtime-spawn pattern. Decorators are for HTTP-exposed singletons.

### G. What gets deleted

Files:

- `src/pydantic_ai_stateflow/runtime/container.py` — entire file.
- `src/pydantic_ai_stateflow/runtime/engine.py` — entire file.
- `src/pydantic_ai_stateflow/runtime/provider.py` — entire file.
- `src/pydantic_ai_stateflow/runtime/event_stream_provider.py` — entire file.
- `src/pydantic_ai_stateflow/providers/` package (CoreProvider,
  PersistenceProvider) — entire package.

Symbols (with their files):

- `register_agent` / `clear_agent_registry` (`runtime/agents.py`) — replaced
  by `@stateflow_agent` decorator + test-only `_reset_registries()` helper
  used by `TestEngine`.
- `get_container` / `get_engine` (`api/deps.py`) — replaced by
  `get_thread_repo` / `get_event_log` / `get_event_stream` /
  `get_workflow_instance(name)` / `get_agent_instance(name)`.
- `build_threads_router(*, thread_repo, prefix)` → module-level
  `threads_router = APIRouter()` using `Depends`.
- `build_streaming_router(*, thread_repo, event_log, event_stream)` →
  module-level `streaming_router` using `Depends`.
- `build_dbos_router()` → module-level `dbos_router` (no deps needed).
- `build_health_router(checks=)` → `make_health_router(checks)` factory
  retained because checks are per-app; this is the one router-builder
  that stays a function because its body genuinely depends on the
  per-app check map.
- `ObservabilityProvider` (class) → `ObservabilityConfig` dataclass +
  `ObservabilityConfig.install()` method. Same configuration knobs,
  no Container coupling.

`Engine` / `EngineInvariantViolation` symbols removed from `sf.*` exports.
The "ObservabilityProvider must be first" invariant moves into
`ObservabilityConfig.install()` (idempotent; raises if called twice with
different configs).

`app.state.container` and `app.state.engine` cease to exist. `app.state`
holds:

- `app.state.thread_repo: ThreadRepository`
- `app.state.event_log: EventLogRepository`
- `app.state.event_stream: EventStream`
- `app.state.workflows: dict[str, object]`
- `app.state.agents: dict[str, StateflowAgent]`

…and any app-set attributes (e.g. `app.state.notes_repo`).

## Migration

Sequential, one PR per step so each step is independently reviewable.

### Framework

1. **Add new modules side-by-side with the old ones.**
   - Create `src/pydantic_ai_stateflow/runtime/app.py` with `create_app`.
   - Create `src/pydantic_ai_stateflow/runtime/workflows.py` with
     `workflow` decorator + workflow registry.
   - Add `stateflow_agent` decorator to `runtime/agents.py` alongside the
     existing `register_agent`.
   - Add `src/pydantic_ai_stateflow/testing/` package with `TestEngine`,
     `MockAgent`, `MockFlow`, and `pytest_plugin.py`.
   - Add module-level `threads_router`, `streaming_router`, `dbos_router`
     in their respective modules, using `Depends`-based dep resolution.
   - Add `ObservabilityConfig` dataclass in `observability/config.py`.
2. **Wire the new exports** in `pydantic_ai_stateflow/__init__.py`. Keep
   old exports temporarily for the migration window.
3. **Migrate notes-app** (next section).
4. **Delete old code** in a single sweep PR: `container.py`, `engine.py`,
   `provider.py`, `event_stream_provider.py`, `providers/`,
   `build_X_router` functions, `register_agent` /
   `clear_agent_registry`, `get_container`, `get_engine`,
   `ObservabilityProvider` (class). Remove from `__init__.py` exports.
5. Framework tests:
   - Update `tests/runtime/test_engine.py`, `test_container.py`,
     `test_providers.py` — most delete entirely. Replace with
     `tests/runtime/test_app.py` covering `create_app`.
   - Update `tests/api/test_threads_router.py` etc. — drop
     `build_threads_router(thread_repo=...)` calls in favour of
     `TestClient` + dep overrides.
   - Add `tests/testing/test_test_engine.py` covering TestEngine
     lifecycle, override semantics, DBOS isolation.

### notes-app

1. Rewrite `examples/notes-app/backend/src/notes_app/main.py` per the
   example in §F. Delete `build_app`.
2. Decorate `NotesAgent`, `NotesTodoApprovalAgent` with
   `@sf.stateflow_agent`. Drop their `name = "..."` ClassVar (or keep as
   explicit overrides).
3. Decorate `TodoApprovalFlow` with
   `@sf.workflow(input=TodoApprovalInput, output=TodoApprovalOutcome)`.
   Decorate `BrainstormFlow` likewise.
4. Delete `examples/notes-app/backend/src/notes_app/brainstorm_router.py`
   entirely — the `POST /workflows/brainstorm-todo` endpoint is now
   auto-generated. (Workflow-id determinism moves to
   `BrainstormFlow.workflow_id(task)`.)
5. Rewrite `tests/test_smoke.py`:
   - Delete `_unique_flow`, `_unique_brainstorm`, `_FakeNotesAgent`.
   - Use `engine` / `client` fixtures from `sf.testing.pytest_plugin`.
   - Each test: `engine.override(NotesAgent, MockAgent.with_output(...))`
     then `client.post(...)`.
6. Delete `tests/conftest.py` `dbos_runtime` / `fresh_dbos_executor`
   fixtures — `TestEngine` owns DBOS lifecycle.

### BC-break points

- All apps using `build_X_router(...)` functions break — call sites need
  to import the module-level routers or stop wrapping them entirely
  (`create_app` handles mounting).
- All apps using `register_agent(instance)` break — must add
  `@stateflow_agent` decorator and pass instance to `create_app(agents=...)`.
- All apps using `Engine(providers=[...]).fastapi_app(...)` break —
  call `sf.create_app(...)` instead.
- `sf.Container`, `sf.Engine`, `sf.ObservabilityProvider`,
  `sf.get_container`, `sf.get_engine`, `sf.register_agent` all removed
  from public API.

Notes-app is the only in-repo consumer; the BC blast radius is bounded.

## Testing

### Framework

1. **`tests/runtime/test_app.py` (new)**
   - `create_app()` with no workflows / agents — health endpoint
     responds 200; DBOS not launched (no `dbos=` arg).
   - `create_app(workflows=[ZeroArgFlow])` — `POST /workflows/zero-arg-flow`
     responds; instance auto-constructed.
   - `create_app(workflows=[NeedsArgsFlow])` — raises at `create_app` time
     with diagnostic naming the missing args.
   - `create_app(observability=ObservabilityConfig(...))` — logfire
     configured once; calling `create_app` twice with different observability
     configs raises.
2. **`tests/runtime/test_workflows.py` (new)**
   - `@sf.workflow(input=, output=)` on a class missing `run` — raises
     at decoration with precise message.
   - `@sf.workflow` on a class whose `run` is not `@Durable.workflow()`-
     decorated — raises at decoration.
   - Kebab-case naming verified for canonical class names
     (`TodoApprovalFlow` → `todo-approval-flow`, `MyXMLFlow` → `my-xml-flow`).
   - Duplicate registration raises.
3. **`tests/runtime/test_agents.py` (rewritten)**
   - `@stateflow_agent` sets `cls.name` to kebab case.
   - `@stateflow_agent` on duplicate name raises.
   - `@stateflow_agent` respects explicit `name = "..."` ClassVar
     when it differs from kebab-case.
4. **`tests/testing/test_test_engine.py` (new)**
   - `TestEngine.default()` → `engine.test_client()` round-trips a thread
     CRUD without DBOS errors.
   - Two `TestEngine.default()` instances in the same process don't
     collide on DBOS config_name or schema.
   - `engine.override(NotesAgent, MockAgent.with_output("x"))` — agent
     responds with `"x"` on streaming endpoint.
   - `engine.override(ThreadRepository, custom_repo)` — `GET /threads`
     hits `custom_repo`.
   - `engine.override(BrainstormFlow, MockFlow.returning(...))` —
     `POST /workflows/brainstorm-flow` returns the mocked output.
   - DBOS destroyed after `test_client` exits, even on exception inside
     the `with` block.

### notes-app

1. Existing `test_smoke.py` test cases (`test_note_repository_is_bound_in_container`
   becomes `test_notes_repo_resolves_via_depends`, the other two stay
   semantically equivalent).
2. New: `test_brainstorm_workflow_auto_mounted` — `POST
   /workflows/brainstorm-flow` returns a 200 + `workflow_id`, without
   the deleted `build_brainstorm_router`.
3. New: `test_todo_workflow_auto_mounted` — `POST /workflows/todo-approval-flow`.

### TestEngine self-tests

Aside from the new TestEngine test module, the framework's existing
`tests/patterns/*` suite continues to use its own `conftest.py` DBOS
fixture — patterns are tested in isolation from `create_app`. This
isolates the rewrite blast radius: pattern tests don't need to change.

## Open questions

1. **Module-level routers vs `make_X_router()` factories.** The spec
   commits to module-level `APIRouter` instances (e.g.
   `threads_router = APIRouter()`) with `Depends(get_thread_repo)` for
   dep resolution. The risk: if a future need arises for "two threads
   routers with different prefixes mounted on the same app" the
   module-level instance would need to be cloned. Probability low (no
   current callers); revisit if it materializes. **RESOLVED:** stay
   module-level.

2. **Should `sf.create_app` accept `repos: dict[type, object]` for
   bulk override of the three default repos?** Currently it's three
   separate kwargs (`thread_repo`, `event_log`, `event_stream`).
   Three is small enough to not need a bag — recommend status quo. If
   a fourth (e.g. a `MetricsRepository`) lands in SP2 the bag pattern
   becomes worth considering.

3. **Are tests using `pytest_plugins = ["pydantic_ai_stateflow.testing"]`
   robust against test files that intentionally don't want the fixture
   loaded?** pytest treats `pytest_plugins` per-conftest; modules outside
   that conftest's scope don't see the fixture. This is the standard
   pattern (matches `pytest-asyncio`, `pytest-anyio`). No action
   needed; calling out for awareness.

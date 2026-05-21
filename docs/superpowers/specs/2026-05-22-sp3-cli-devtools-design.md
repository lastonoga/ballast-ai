# Sub-project 3: Devtools (CLI + Migrations)

**Status:** Approved (design)
**Date:** 2026-05-22
**Scope:** New `pydantic_ai_stateflow.cli` package + Alembic env.py wiring
to SP2 settings + `stateflow` console script.

## Problem

The notes-app demo (the only current consumer of the framework) starts
the server via an ad-hoc `main()` that calls `uvicorn.run("notes_app.main:app", …)`
directly. Framework users have no first-class tooling and must wire up:

- their own `uvicorn` invocation per project (`uv run uvicorn ...`),
- their own Alembic invocation against the framework's bundled
  `alembic.ini` (which currently has a placeholder `sqlalchemy.url`
  baked in — see `src/pydantic_ai_stateflow/alembic.ini` line 5),
- their own ad-hoc curl/HTTPie commands against the DBOS introspection
  router (`/dbos/threads/.../workflows`) to inspect workflow state,
- their own SSE consumer (browser devtools, or `curl -N`) to tail
  `/threads/{id}/events` while debugging an agent run.

There is no single binary, no convention for "where the app lives", and
no way to introspect DBOS workflows without booting the HTTP server and
opening a browser. Alembic operates on a framework-internal `alembic.ini`
whose DB URL is a hard-coded placeholder, so every consumer has to
override it via `-x sqlalchemy.url=…` or env-vars in env.py.

Reference projects (Django `manage.py`, FastAPI `fastapi dev`,
DBOS `dbos` CLI) all converge on a single, project-aware console script.
The framework should match that bar.

## Goal

Ship a single `stateflow` binary covering the day-1 dev loop:

- `stateflow dev` — auto-detect the app, run uvicorn with reload.
- `stateflow migrate [revision -m "<msg>"]` — wrap Alembic upgrade/revision
  against the framework's bundled migration tree, with the DB URL
  resolved from SP2 settings.
- `stateflow workflows ls` — list DBOS workflows registered in the app
  with status, suitable for the dev console.
- `stateflow events tail <thread-id>` — tail the SSE event stream
  against a running dev server.

Convention-over-configuration: zero CLI args needed in the common case
once `STATEFLOW_APP` (env var) or `[tool.stateflow] app = "..."`
(pyproject.toml) is set.

## Non-goals

- **DI rewrite** — lives in SP1; CLI consumes whatever SP1 produces.
- **Settings module** — lives in SP2; CLI *uses* `settings.dbos.database_url`
  but does not define it.
- **Auto-installed observability defaults** — observability stays opt-in
  per the existing `ObservabilityProvider` wiring in `examples/notes-app`.
- **Production process management** — `stateflow dev` is a dev-loop tool
  (uvicorn `--reload`). Production deploys keep using `uvicorn`/`gunicorn`
  directly against the app object.
- **Multi-app projects** — one `stateflow` invocation maps to one app
  instance. Monorepos with multiple stateflow apps configure
  `STATEFLOW_APP` per shell or pass `--app`.
- **Interactive Alembic flags** — no `migrate downgrade` or `migrate stamp`
  in v1; users drop down to `alembic` directly for those (we document
  the escape hatch).

## Design

### A. CLI architecture

Package: `src/pydantic_ai_stateflow/cli/` (new).

```
src/pydantic_ai_stateflow/cli/
    __init__.py
    main.py            # typer app + subcommand registration
    app_detect.py      # STATEFLOW_APP + pyproject.toml resolution
    commands/
        __init__.py
        dev.py         # `stateflow dev`
        migrate.py     # `stateflow migrate`, `stateflow migrate revision`
        workflows.py   # `stateflow workflows ls`
        events.py      # `stateflow events tail`
```

`main.py` exposes a single `cli = typer.Typer(name="stateflow", ...)`
with subcommands mounted as typer sub-apps:

```python
import typer
from pydantic_ai_stateflow.cli.commands import dev, migrate, workflows, events

cli = typer.Typer(
    name="stateflow",
    help="Devtools for pydantic-ai-stateflow apps.",
    no_args_is_help=True,
)
cli.command(name="dev")(dev.dev)
cli.add_typer(migrate.app, name="migrate")
cli.add_typer(workflows.app, name="workflows")
cli.add_typer(events.app, name="events")
```

**Subcommand framework: `typer`.** It is already a base dependency
(`pyproject.toml` line 17: `typer>=0.25.1`), so no new dep. Typer's
type-hint-driven argument parsing matches the framework's pydantic-first
ergonomics; `--help` and shell completion come free.

**Pretty output: `rich`.** Pulled in transitively by `typer` itself (typer
depends on rich for `rich_help_panel` etc), so no new direct dep is
needed. Used for the `workflows ls` table and color-coded `events tail`.

### B. App auto-detection

Convention (in resolution order):

1. **`--app`** CLI flag (overrides everything; mainly for tests and
   monorepo escape hatch). Form: `module.path:variable_name`.
2. **`STATEFLOW_APP` env var.** Form: `module.path:variable_name`,
   e.g. `notes_app.main:app`.
3. **`[tool.stateflow] app = "..."`** in `./pyproject.toml` (walk up
   from CWD to first `pyproject.toml`).
4. **Error** with explicit hint — no scanning, no globbing.

```python
# app_detect.py
@dataclass(frozen=True)
class AppRef:
    """Parsed ``module.path:variable_name`` reference."""
    module: str
    attr: str

    @property
    def import_string(self) -> str:
        return f"{self.module}:{self.attr}"


def resolve_app_ref(explicit: str | None = None) -> AppRef:
    """Find the app reference; raise typer.BadParameter with a hint."""
    raw = (
        explicit
        or os.environ.get("STATEFLOW_APP")
        or _read_pyproject_app()  # walks up, returns str | None
    )
    if not raw:
        raise typer.BadParameter(
            "Could not locate the stateflow app. Set STATEFLOW_APP="
            "'module.path:variable_name' or add\n"
            "    [tool.stateflow]\n"
            "    app = \"module.path:variable_name\"\n"
            "to pyproject.toml, or pass --app explicitly.",
        )
    module, _, attr = raw.partition(":")
    if not module or not attr:
        raise typer.BadParameter(
            f"App reference {raw!r} must be 'module.path:variable_name'.",
        )
    return AppRef(module=module, attr=attr)


def import_app(ref: AppRef) -> FastAPI:
    """Import the module and return its ``attr`` attribute."""
    mod = importlib.import_module(ref.module)
    return getattr(mod, ref.attr)
```

**Why no scanning fallback?** Per the agreed decision: *explicit beats
magic*. Scanning a repo for `FastAPI()` patterns is brittle (false
positives in tests/, ambiguous in monorepos, breaks when the app is
constructed in a `build_app()` factory and assigned in a non-top-level
scope). One env var or one `pyproject.toml` line is cheap; an
incorrect auto-pick is expensive to debug.

The notes-app already has the right shape at module top-level
(`app: FastAPI = build_app()` — `notes_app/main.py` line 213), so the
docs example becomes:

```toml
# examples/notes-app/backend/pyproject.toml
[tool.stateflow]
app = "notes_app.main:app"
```

### C. `stateflow dev` command

Wraps `uvicorn` with `--reload` enabled. The CLI does not import the app
itself; it passes the import string to uvicorn so uvicorn's reloader
process owns reimporting.

```python
# commands/dev.py
def dev(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(True, "--reload/--no-reload"),
    app_ref: str | None = typer.Option(None, "--app"),
    log_level: str = typer.Option("info", "--log-level"),
) -> None:
    """Run the stateflow app under uvicorn with reload.

    Example:

        stateflow dev
        stateflow dev --host 0.0.0.0 --port 8001
        stateflow dev --app notes_app.main:app --no-reload
    """
    ref = resolve_app_ref(app_ref)
    import uvicorn  # local import — keeps `--help` snappy
    uvicorn.run(
        ref.import_string,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
```

**Why uvicorn `--reload` and not `watchfiles` directly?** Parity. Users
who outgrow `stateflow dev` should be able to copy the equivalent
`uvicorn` invocation 1:1; routing reload through uvicorn means
`stateflow dev` ≡ `uvicorn $STATEFLOW_APP --reload`. Custom watcher
config would diverge.

`notes_app.main.main()` (which today calls `uvicorn.run(...)` itself)
becomes redundant once `stateflow dev` exists; the notes-app docs
section in this spec's Migration note covers swapping it.

### D. `stateflow migrate` command

Sub-typer with two commands:

```
stateflow migrate                              # = alembic upgrade head
stateflow migrate revision -m "add foo col"    # = alembic revision --autogenerate -m "..."
```

`alembic.ini` resolution:

1. `--alembic-ini` flag (escape hatch).
2. `[tool.stateflow] alembic_ini = "..."` in pyproject.toml.
3. Default to the framework's bundled `alembic.ini`
   (`importlib.resources.files("pydantic_ai_stateflow") / "alembic.ini"`),
   which already points at the bundled `alembic/` script_location with
   the framework's migration history.

The third default is the **expected** path for stateflow apps — the
framework owns the migration tree for its own tables (thread, hitl,
outbox, tenant); apps that need their own migrations point
`alembic_ini` at their own file.

**DB URL wiring (env.py + SP2 settings).** Today,
`src/pydantic_ai_stateflow/alembic.ini` hard-codes
`sqlalchemy.url = postgresql+asyncpg://localhost/placeholder` and
`src/pydantic_ai_stateflow/alembic/env.py` reads it via
`config.get_main_option("sqlalchemy.url")`. SP2 introduces
`settings.dbos.database_url`; SP3 wires it into env.py at module import:

```python
# src/pydantic_ai_stateflow/alembic/env.py (post-SP3 patch)
from pydantic_ai_stateflow.settings import get_settings  # SP2

config = context.config

# SP2 override: settings.dbos.database_url wins over alembic.ini's
# placeholder when settings are loadable in the current process.
# When env.py is run from a context without settings (e.g. CI smoke
# `alembic check` with no env vars), the alembic.ini value remains
# the fallback so the file stays standalone-usable.
try:
    _settings = get_settings()
    config.set_main_option(
        "sqlalchemy.url",
        _settings.dbos.database_url,
    )
except Exception:  # SP2-defined config error type once available
    pass
```

`stateflow migrate` shells out to alembic in-process (no subprocess —
import `alembic.config.main` and call it with assembled argv). Errors
propagate as typer exits with non-zero. Example invocations:

```
stateflow migrate                                # upgrade to head
stateflow migrate revision -m "add tenant col"   # autogenerate revision
stateflow migrate --alembic-ini ./custom.ini     # override resolution
```

**Async/sync note.** The current `env.py` runs `asyncio.run(run_migrations_online())`
with `async_engine_from_config`. The DB URL coming from
`settings.dbos.database_url` is the sync `postgresql://` form per
`build_dbos_config()` in `runtime/dbos_setup.py`. Alembic itself does
not require asyncpg; the existing env.py uses async only because the
hard-coded URL is asyncpg-flavored. Post-SP3, env.py should branch on
URL scheme: `postgresql+asyncpg://` → async path; `postgresql://` /
`sqlite://` → sync path. This avoids forcing asyncpg into every
alembic run.

### E. `stateflow workflows ls` command

```python
# commands/workflows.py
@app.command(name="ls")
def ls(
    app_ref: str | None = typer.Option(None, "--app"),
    status: str | None = typer.Option(
        None, "--status",
        help="Filter by status (PENDING|RUNNING|SUCCESS|ERROR|CANCELLED).",
    ),
    limit: int = typer.Option(50, "--limit"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """List DBOS workflows registered in the active app.

    Example:

        stateflow workflows ls
        stateflow workflows ls --status RUNNING
        stateflow workflows ls --json | jq '.[] | .workflow_id'
    """
```

**Flow:**

1. Resolve `STATEFLOW_APP` and `import_module(ref.module)` to trigger
   workflow registration side effects (the imports of `Durable.workflow`-
   decorated callables run at module load).
2. `Durable.init(...)` + `Durable.launch()` must already have been
   called for `Durable.list_workflows(...)` to query the DBOS database.
   The notes-app does this inside the FastAPI lifespan
   (`_launch_dbos` in `notes_app/main.py` line 171). For the CLI we
   replicate that bootstrap directly:

   ```python
   from dbos import DBOSConfig
   from pydantic_ai_stateflow.durable import Durable
   from pydantic_ai_stateflow.settings import get_settings  # SP2

   settings = get_settings()
   Durable.init(DBOSConfig(
       name=settings.dbos.app_name,
       system_database_url=settings.dbos.database_url,
   ))
   Durable.launch()
   try:
       wfs = await Durable.list_workflows(limit=limit, sort_desc=True)
   finally:
       Durable.destroy(destroy_registry=False)
   ```

   `Durable.init` + `Durable.launch` here are **read-only**: we don't
   start a server or accept enqueues, we just give `list_workflows` a
   live DB handle. `destroy_registry=False` keeps the same semantics
   the notes-app uses on app shutdown.

3. **Output: rich table** (default) or **JSON** (`--json`).

Rich table columns: `workflow_id | name | status | started_at | queue`.

```
$ stateflow workflows ls
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳────────────────────┳────────┓
┃ workflow_id                  ┃ name                     ┃ status  ┃ started_at         ┃ queue  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇────────────────────╇────────┩
│ agent-run:abc…:run-1         │ NotesAgent.run           │ SUCCESS │ 2026-05-22 10:14   │ -      │
│ brainstorm:abc…:flow         │ BrainstormFlow.run       │ RUNNING │ 2026-05-22 10:18   │ -      │
│ agent-run:abc…:run-2-approve │ NotesTodoApprovalAgent.. │ PENDING │ 2026-05-22 10:18   │ -      │
└─────────────────────────────┴──────────────────────────┴─────────┴────────────────────┴────────┘
```

`--json` emits the same dicts shape that `_wf_to_dict` in
`api/dbos_router.py` already produces, so machine consumers (CI, scripts)
get a stable schema.

### F. `stateflow events tail` command

HTTP SSE client against a running dev server.

```python
@app.command(name="tail")
def tail(
    thread_id: UUID = typer.Argument(...),
    host: str = typer.Option("localhost", "--host"),
    port: int = typer.Option(8000, "--port"),
    scheme: str = typer.Option("http", "--scheme"),
) -> None:
    """Tail SSE events for a thread from a running stateflow server.

    Example:

        stateflow events tail 9b1a-…
        stateflow events tail 9b1a-… --port 8001
    """
```

**Flow:**

1. Connect with `httpx.AsyncClient` to
   `{scheme}://{host}:{port}/threads/{thread_id}/events`.
2. Use `httpx-sse` (small wrapper) to iterate events. **New optional
   dependency**, but tiny and well-maintained; add to base since the
   tail command is part of the base CLI. Alternative: hand-roll
   line-buffered parsing of the SSE wire format (`id:`, `data:`,
   `\n\n` boundaries). The wire shape is fixed and small (see
   `api/streaming/router.py` line 716-721), so hand-rolling is viable
   if avoiding the dep is preferred. **Default: hand-roll** to keep
   deps minimal — the format is 4 lines of parsing.
3. Pretty-print each event:
   - `kind` color-coded (rich): `text-delta` → dim cyan,
     `message-added` → green, `thread-created` → blue, `error` → red.
   - `seq` aligned, monospace.
   - `payload` formatted via `rich.json.JSON` (one-line for short
     payloads, multi-line for long ones).

```
$ stateflow events tail 9b1a-2c5d-...
[12:01:33] seq=1   thread-created   {"thread_id":"9b1a-..."}
[12:01:34] seq=2   text-delta       {"delta":"Hello"}
[12:01:34] seq=3   text-delta       {"delta":", world"}
[12:01:35] seq=4   message-added    {"role":"assistant","content":"Hello, world"}
```

**Reconnection.** On disconnect (server restart, network blip), the
command reconnects with the last seen `seq` as the `Last-Event-ID`
header — exactly the resume protocol the router already supports
(`api/streaming/router.py` line 712-714). Backoff: 0.5s, 1s, 2s, 4s,
cap 5s; reset on successful reconnect. Ctrl-C exits cleanly.

### G. Console script registration

Add to `pyproject.toml`:

```toml
[project.scripts]
stateflow = "pydantic_ai_stateflow.cli.main:cli"
```

Single entry point. After `pip install pydantic-ai-stateflow`, the
`stateflow` binary is on `$PATH`. In dev (`uv run`), `uv run stateflow ...`
works without an extra install step.

## Migration

One PR per command is reasonable; here's the unified diff plan.

1. **`pyproject.toml`**:
   - Add `[project.scripts] stateflow = "pydantic_ai_stateflow.cli.main:cli"`.
   - No new deps (typer + rich already present transitively).

2. **`src/pydantic_ai_stateflow/cli/`**: new package per layout in §A.

3. **`src/pydantic_ai_stateflow/alembic/env.py`**: insert the SP2
   settings wiring (§D) — guarded `try/except` so the file stays
   standalone-runnable when SP2 isn't available. Branch on URL scheme
   for sync vs async engine.

4. **`examples/notes-app/backend/pyproject.toml`**: add
   `[tool.stateflow] app = "notes_app.main:app"`.

5. **`examples/notes-app/backend/src/notes_app/main.py`**: leave
   `main()` for backward compatibility for one cycle, but the README
   switches to `stateflow dev` as the documented start command.

6. **README + docs**: add a "CLI" section. Document `STATEFLOW_APP` /
   `[tool.stateflow]` convention, the four subcommands, and the
   alembic.ini resolution order.

No BC-breaks. The framework still works without the CLI (apps that
prefer their own `main()` keep it). The alembic.ini env.py change is
backward-compatible because the new settings override path is
`try/except`-guarded and falls back to the ini value.

## Testing

**Unit (typer.testing.CliRunner):**

- `stateflow --help` exits 0, lists all four subcommands.
- `stateflow dev --help` shows host/port/reload options.
- `stateflow workflows ls --app missing.module:app` errors with the
  hinted message from `resolve_app_ref` (verifies error UX).
- `resolve_app_ref` table-tests: env var, pyproject, --app override,
  missing-all error.
- `import_app` raises `AttributeError` cleanly when the attr is missing
  (typer wraps and prints a useful message).

**Integration (subprocess, in CI):**

- `stateflow dev --no-reload --port 18000` started in a background
  thread → poll `GET /health` until 200 → kill → assert clean exit.
- `stateflow migrate` against the framework's bundled `alembic.ini`
  pointed at a `sqlite:///tmp/test.db` URL via settings → assert tables
  created.
- `stateflow events tail <fake-uuid> --port 18000` against a running
  test server → assert it prints initial `: connected` and exits on
  Ctrl-C (SIGINT).

**Smoke (manual, post-merge):**

- `cd examples/notes-app/backend && uv run stateflow dev` → browse
  the frontend, run a brainstorm, observe events in browser.
- In a separate shell: `uv run stateflow workflows ls` → see the
  brainstorm workflow appear.
- `uv run stateflow events tail <thread-id>` → see SSE events stream
  in the terminal during a chat turn.

The notes-app CI smoke (`tests/test_smoke.py` per the existing
divergent-pattern spec) is unaffected — it doesn't invoke the CLI.

## Open questions

1. **`stateflow workflows ls` filtering.** v1 supports `--status` and
   `--limit`; do we want `--thread <uuid>` (which would just translate
   to `workflow_id_prefix=["agent-run:{uuid}:"]`)? Likely yes for
   parity with the HTTP endpoint, but not blocking — can land in v1.1.

2. **`stateflow workflows show <id>`.** Detail view + steps + children
   would mirror the three HTTP endpoints (`/workflows/{id}`,
   `/workflows/{id}/steps`, `/workflows/{id}/children`). Out of scope
   for v1 but the obvious next command; the rich table groundwork from
   `ls` carries over.

3. **`stateflow events tail` against remote prod.** Today the command
   assumes localhost. If apps want to point it at a deployed stateflow
   instance, they can `--host prod.example.com --port 443 --scheme https`
   — but the SSE endpoint may require auth headers we don't pass.
   v1 is local-dev only; remote tailing waits for an auth story.

4. **Process model for `stateflow workflows ls`.** Each invocation
   does `Durable.init` + `Durable.launch` + `Durable.destroy`. On
   SQLite this is cheap; on Postgres it pays a connection cost per
   call. If users start scripting `watch stateflow workflows ls`,
   a persistent daemon would be better — but that's a v2 concern.

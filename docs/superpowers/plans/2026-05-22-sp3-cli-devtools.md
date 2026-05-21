# SP3: CLI + Devtools — Implementation Plan

**Goal:** Ship `stateflow` console script with `dev`/`migrate`/`workflows ls`/`events tail` subcommands. Wire Alembic `env.py` to `settings.dbos.database_url`.

**Spec:** `docs/superpowers/specs/2026-05-22-sp3-cli-devtools-design.md`

---

## Task list

- T1: `cli/` package skeleton — `main.py` (typer cli) + `app_detect.py` (STATEFLOW_APP / [tool.stateflow]) + console_script registration in `pyproject.toml`
- T2: `commands/dev.py` — uvicorn-reload wrapper
- T3: `commands/migrate.py` + `alembic/env.py` SP2-settings wiring (sync/async URL branching)
- T4: `commands/workflows.py` — `workflows ls` with rich table + `--json`
- T5: `commands/events.py` — `events tail` SSE client with reconnect
- T6: notes-app `[tool.stateflow]` config + verification tests

After all tasks pass: `stateflow --help`, `stateflow workflows ls --json`, `stateflow dev` all work; tests green.

"""FastAPI entry point — Ballast + providers.

App-specific singletons (repos, flows, agents) live in their own
modules and are imported here directly — no constructor DI for app
state. The framework's :class:`ballast.Ballast` accepts a
:class:`BallastSettings` instance plus a sequence of providers via
``.use(...)``; the terminal ``.fastapi(...)`` returns the FastAPI app
and installs the process-wide :class:`Engine` singleton (read via
``ballast.get_ballast()``).

Persistence wiring (sqlite by default):

- ``NOTES_APP_DATABASE_URL`` (default ``sqlite+aiosqlite:///./notes-app.sqlite``)
  → notes + threads + messages + event_log persist to a local sqlite file.
- ``NOTES_APP_DATABASE_URL=""`` or ``":memory:"`` → InMemory repos.
- Under pytest (``PYTEST_CURRENT_TEST`` set) → InMemory repos
  unconditionally so test imports of this module don't touch the local
  sqlite file. Tests that need SQL persistence build their own
  sessionmaker.
- DBOS workflow state has its own sqlite file (see ``_dbos_db_url``).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import ballast
from dbos import DBOSConfig
from dotenv import load_dotenv
from fastapi import FastAPI

from ballast.persistence import (
    InMemoryEventLogRepository,
    InMemoryThreadRepository,
    SqlEventLogRepository,
    SqlThreadRepository,
)
from ballast.persistence.approval_card import InMemoryApprovalCardRepository
from ballast.observability.config import ObservabilityConfig
from ballast.settings import get_settings
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from notes_app.agents.notes import notes_agent
from notes_app.agents.todo_approval import approval_agent
from notes_app.repositories import events as _events_module
from notes_app.repositories import note as _note_module
from notes_app.repositories import thread as _thread_module
from notes_app.repositories.note import (
    InMemoryNoteRepository,
    SqlNoteRepository,
)
from notes_app.routes.notes import build_notes_router
from notes_app.agents import agents
from notes_app.routes.streaming import router as streaming_router
from notes_app.routes.workflows import router as workflows_router
from notes_app.settings import get_notes_settings
from notes_app.streams import event_stream
from notes_app.workflows.brainstorm import brainstorm
from notes_app.workflows.todo_approval import todo_flow  # noqa: F401 — DBOS classes self-register on import; needed for propose_todo


load_dotenv()


def _dbos_db_url() -> str:
    url = get_settings().dbos.database_url
    if url:
        return url
    return f"sqlite:///{Path(tempfile.gettempdir()) / 'notes-app.dbos.sqlite'}"


def _should_use_sql() -> bool:
    """Branch on ``NOTES_APP_DATABASE_URL`` and pytest detection.

    - Under pytest (``pytest`` in ``sys.modules``) → always InMemory
      (test imports must NOT touch the local sqlite file; tests that
      need SQL build their own sessionmaker).
    - ``""`` or ``":memory:"`` → InMemory.
    - Anything else → SQL via the configured URL.
    """
    import sys  # noqa: PLC0415

    if "pytest" in sys.modules:
        return False
    url = get_notes_settings().database_url.strip()
    if url == "" or url == ":memory:":
        return False
    return True


# ── Repo wiring ─────────────────────────────────────────────────────────
# Build SQL or InMemory repos based on the URL/test detection, then
# REPLACE the module-level singletons in the three repository modules so
# every existing caller (which imports those singletons by name) sees
# the wired-up instance.
if _should_use_sql():
    from sqlalchemy.ext.asyncio import (  # noqa: PLC0415
        async_sessionmaker,
        create_async_engine,
    )

    _engine = create_async_engine(get_notes_settings().database_url)
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)

    notes_repo = SqlNoteRepository(_sessionmaker)
    thread_repo = SqlThreadRepository(_sessionmaker)
    event_log = SqlEventLogRepository(_sessionmaker)

    # Rebind singletons so the lazy-import sites in agents/workflows pick
    # up the SQL impl too.
    _note_module.notes_repo = notes_repo
    _thread_module.thread_repo = thread_repo  # type: ignore[assignment]
    _events_module.event_log = event_log  # type: ignore[assignment]
else:
    # InMemory path — for tests and ``NOTES_APP_DATABASE_URL=""``/``:memory:``.
    # Use the singletons that the repos modules already constructed at
    # import time so existing tests + monkeypatch sites keep working.
    notes_repo = _note_module.notes_repo  # type: ignore[assignment]
    thread_repo = _thread_module.thread_repo  # type: ignore[assignment]
    event_log = _events_module.event_log  # type: ignore[assignment]
    # Sanity: ensure the InMemory types are what we expect.
    assert isinstance(notes_repo, InMemoryNoteRepository)
    assert isinstance(event_log, InMemoryEventLogRepository)
    assert isinstance(thread_repo, InMemoryThreadRepository)


settings = get_settings()

app: FastAPI = (
    ballast.Ballast(settings)
    .with_observability(
        ObservabilityConfig(
            service_name="app",
            environment="dev",
            instrument_pydantic_ai=True,
            instrument_httpx=True,
            instrument_fastapi=False,
        ),
    )
    .with_dbos(
        DBOSConfig(name="notes-app", system_database_url=_dbos_db_url()),
    )
    .with_thread_repo(thread_repo)
    .with_events(event_log, event_stream)
    # Claude Haiku for the judge: pydantic-evals' judge agent
    # internally uses ``output_type=GradingOutput`` (a BaseModel) which
    # pydantic-ai wraps in ``ToolOutput`` with ``tool_choice="required"``.
    # Qwen 3.6 endpoints on OpenRouter reject that value with a 404
    # (same compat bug NotesAgent dodges via
    # ``output_type=[str, DeferredToolRequests]``). Haiku supports
    # tool_choice cleanly + is cheap enough for fire-and-forget use.
    # ``temperature=0`` keeps verdicts stable across re-runs.
    .with_judge_defaults(
        "openrouter:anthropic/claude-haiku-4.5",
        model_settings=OpenRouterModelSettings(
            temperature=0.0,
            openrouter_usage={"include": True},
        ),
    )
    .with_approval_repo(InMemoryApprovalCardRepository())
    .fastapi(
        cors="dev",
        routers=[
            build_notes_router(thread_repo),
            workflows_router,
            streaming_router,
        ],
    )
)

# Tests + frontend introspection.
app.state.notes_repo = notes_repo
app.state.notes_agent = notes_agent
app.state.todo_approval_agent = approval_agent
app.state.brainstorm = brainstorm


def main() -> None:  # pragma: no cover
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("notes_app.main:app", host=host, port=port, reload=True)


__all__ = ["agents", "app", "event_log", "main", "notes_repo", "thread_repo"]

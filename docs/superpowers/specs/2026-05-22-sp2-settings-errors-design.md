# Sub-project 2: Configuration + Structured Errors

**Status:** Approved (design)
**Date:** 2026-05-22
**Scope:** Framework runtime + notes-app migration

## Problem

The framework currently has two parallel problems that interact badly:

**1. Configuration is scattered.** Every place that needs an env var reaches
for `os.environ.get(...)` directly. Audit (`src/` + notes-app `src/`):

| File | Var |
| --- | --- |
| `src/ballast/logging.py:104,125` | `BALLAST_LOG_LEVEL` |
| `src/ballast/observability/provider.py:103` | `LOGFIRE_TOKEN` (read for diagnostic logging only — logfire SDK reads it again internally) |
| `examples/notes-app/.../main.py:69` | `DBOS_DATABASE_URL` |
| `examples/notes-app/.../main.py:219,220` | `HOST`, `PORT` |
| `examples/notes-app/.../brainstorm_agents.py:59` | `OPENROUTER_API_KEY` |
| `examples/notes-app/.../agent.py:188,191` | `OPENROUTER_MODEL`, `OPENROUTER_API_KEY` |
| `examples/notes-app/.../todo_approval_agent.py:145,148` | `OPENROUTER_MODEL`, `OPENROUTER_API_KEY` |

Consequences: no single place to discover what the framework needs; no
typed defaults; no validation at startup; tests have to monkey-patch
`os.environ` in N places; secrets read three times per request when
multiple agents share an API key.

**2. Errors are unstructured.** The framework has four custom exception
classes (`PatternError` subclasses + `EngineInvariantViolation`) but no
common base, no stable codes, no machine-readable context. FastAPI
routers throw raw `HTTPException(status_code=…, detail="…")` with
free-form string detail (18 callsites across `api/`). The frontend can't
discriminate "thread not found" from "thread expired" from "thread
deleted" without string matching.

When a domain error escapes a workflow into the HTTP layer it currently
either:
- Surfaces as a `500 Internal Server Error` with a stack trace in logs
  and an opaque body, or
- Is caught ad-hoc by the router author and converted to `HTTPException`
  with a hand-rolled status/detail.

Neither is acceptable for an SDK that wants to ship a UI client.

## Goal

Two pieces, designed together because errors will reference settings
(e.g. `BALLAST_LLM_OPENROUTER_API_KEY missing` as a `hint`):

1. **`StateflowSettings(BaseSettings)`** — a single pydantic-settings
   class (with grouped sub-models) is the only place env vars get
   read. Lazy singleton accessed as
   `from ballast import settings`.
2. **`StateflowError` base class hierarchy** — every framework-raised
   error inherits a structured shape: `code`, `detail`, `hint`,
   `context`, plus a class-level `status_code` for HTTP mapping.
   An ASGI middleware renders any unhandled `StateflowError` as
   `application/problem+json`. A pretty-printer formats the same
   shape for stderr / logs.

## Non-goals

- **DI / decorator rewrite** — SP1 owns how the app is constructed and
  how `create_app()` mounts middleware. This spec only specifies the
  middleware *interface* SP1 must wire in.
- **CLI commands** — `stateflow check`, `stateflow doctor`, etc. live
  in SP3. The settings class is built to support them; no commands ship
  here.
- **Observability defaults flipped on** — user explicitly required
  observability stay opt-in via `ObservabilityConfig` (the SP1 replacement
  for `ObservabilityProvider`). `StateflowSettings` holds the *defaults*
  for that config, but importing settings never configures logfire.
- **Generalizing settings to runtime-mutable config** — settings are
  read-once at process start, validated, then immutable. Runtime
  feature flags are out of scope.
- **Translating every existing `HTTPException` callsite** — see
  "Migration of existing errors", we migrate gradually with a
  documented pattern.

## Design

### A. `StateflowSettings`

**Location:** `src/ballast/settings.py` (new module).
Re-exported from the package root as `settings`.

**Env prefix:** `BALLAST_`. Nested models use `__` as the env
delimiter (pydantic-settings standard), e.g.
`BALLAST_DBOS__DATABASE_URL`.

**Why pydantic-settings:** already a transitive dep (logfire pulls it);
ships `.env` parsing, secret-string types, `SettingsConfigDict` for
prefix + delimiter, validation aligned with the rest of the
pydantic-based framework. Standard library for this job; no custom
config loader.

```python
# src/ballast/settings.py
from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBOSSettings(BaseModel):
    """Durable-execution runtime config.

    ``database_url`` may be ``None`` so apps that don't use DBOS still
    load settings without setting a useless var. The DBOS provider
    raises a structured error when it's needed but missing."""
    database_url: str | None = None
    app_name: str = "ballast-ai"


class ObservabilitySettings(BaseModel):
    """Defaults for ``ObservabilityConfig`` (SP1). Importing settings does
    NOT configure logfire — apps construct ``ObservabilityConfig(...)``
    and call ``.install()`` explicitly."""
    logfire_token: SecretStr | None = None
    service_name: str = "ballast-ai"
    environment: str = "dev"
    instrument_pydantic_ai: bool = True
    instrument_httpx: bool = True
    instrument_fastapi: bool = True


class OpenRouterSettings(BaseModel):
    api_key: SecretStr | None = None
    default_model: str | None = None


class LLMSettings(BaseModel):
    """Per-provider sub-models. New providers slot in here."""
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)


class APISettings(BaseModel):
    """HTTP-layer toggles consumed by middleware."""
    # When True (default), StateflowErrorMiddleware is installed by
    # SP1's create_app(). Apps that want to handle StateflowError
    # themselves set this False.
    install_error_middleware: bool = True
    # Whether stack traces are included in problem+json bodies.
    # ``None`` (default) → auto: on iff ``observability.environment == "dev"``.
    # Explicit ``True`` / ``False`` overrides the auto-detect. Safer default
    # for prod (off); convenient for dev (on).
    expose_tracebacks: bool | None = None


class LoggingSettings(BaseModel):
    """Framework logger config. Mirrors the legacy ``BALLAST_LOG_LEVEL``
    env var so existing deployments don't break."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None


class StateflowSettings(BaseSettings):
    """Single source of truth for framework env-driven config.

    Usage:
        from ballast import settings
        url = settings.dbos.database_url

    Env vars use ``BALLAST_`` prefix + ``__`` for nesting:
        BALLAST_DBOS__DATABASE_URL=postgresql://...
        BALLAST_LLM__OPENROUTER__API_KEY=sk-or-...
        BALLAST_OBSERVABILITY__LOGFIRE_TOKEN=...
    """
    model_config = SettingsConfigDict(
        env_prefix="BALLAST_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Apps can subclass + override; lazy import is in-package only.
        case_sensitive=False,
    )

    dbos: DBOSSettings = Field(default_factory=DBOSSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


@lru_cache(maxsize=1)
def _get_settings() -> StateflowSettings:
    return StateflowSettings()


def get_settings() -> StateflowSettings:
    """Return the process-wide cached settings instance.

    First call instantiates and caches; subsequent calls return the
    cached object. Tests reset via ``reset_settings()``.
    """
    return _get_settings()


def reset_settings() -> None:
    """Clear the cache. ONLY for tests — never call in production."""
    _get_settings.cache_clear()


# Public attribute: lazy proxy so ``settings.dbos.database_url`` works
# without an explicit ``get_settings()`` call at every read site.
class _SettingsProxy:
    def __getattr__(self, item: str):
        return getattr(get_settings(), item)


settings: StateflowSettings = _SettingsProxy()  # type: ignore[assignment]
```

**`__init__.py` change:** add `settings`, `StateflowSettings`,
`get_settings`, `reset_settings` to the package exports.

**Test-time override:** `reset_settings()` clears the lru_cache; tests
that need a specific config use a pytest fixture:

```python
@pytest.fixture
def with_settings(monkeypatch):
    def _apply(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        from ballast.settings import reset_settings
        reset_settings()
    return _apply
```

**Read-once semantics:** the singleton is intentionally read-once. Mutating
env vars mid-process and expecting `settings.foo` to refresh is unsupported.
Tests that need different env per case use the fixture above.

### B. `StateflowError` base + subclasses

**Location:** `src/ballast/errors.py` (new top-level module).
Re-exported from the package root.

**Why top-level (not under `runtime/` or `patterns/`):** errors are
cross-cutting. Every layer (grounded, runtime, patterns, capabilities,
api) raises them. Keeping the base in a top-level module avoids the
import-cycle risk we'd hit if it lived inside `runtime/`.

```python
# src/ballast/errors.py
from __future__ import annotations

from typing import Any, ClassVar


class StateflowError(Exception):
    """Base for every framework-raised error.

    Subclasses override class-level attributes; instance args populate
    ``detail`` / ``hint`` / ``context``.

    Attributes:
      code: stable identifier, ``BALLAST_<DOMAIN>_<SPECIFIC>`` format.
        UPPER_SNAKE; never includes free-form text. Frontends + CLI
        switch on this string.
      status_code: HTTP status used by the error middleware when this
        class escapes a route handler. Class-level default; subclasses
        override.
      detail: human-readable one-liner. Required.
      hint: actionable suggestion. Optional.
      context: machine-readable structured info (workflow id, field
        name, retry count, etc.). Optional; default empty dict.
    """

    code: ClassVar[str] = "BALLAST_UNKNOWN"
    status_code: ClassVar[int] = 500

    def __init__(
        self,
        detail: str,
        *,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.detail = detail
        self.hint = hint
        self.context: dict[str, Any] = dict(context or {})
        super().__init__(detail)

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable representation. Used by middleware and logs."""
        return {
            "code": self.code,
            "detail": self.detail,
            "hint": self.hint,
            "context": self.context,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, detail={self.detail!r})"
```

**Hierarchy** (flat: no generic `RuntimeError` / `ValidationError`
intermediates that would shadow stdlib/pydantic names; concrete
subclasses inherit directly from `StateflowError` or from a domain
parent that adds no behaviour beyond grouping):

```
StateflowError
├── ConfigurationError                 (status 500)  BALLAST_CONFIG_*
│   ├── SettingsValidationError                       BALLAST_CONFIG_SETTINGS_INVALID
│   ├── MissingDependencyError                        BALLAST_CONFIG_DEPENDENCY_MISSING
│   └── ConfigurationInvariantViolation               BALLAST_CONFIG_INVARIANT
│           (replaces the old ``EngineInvariantViolation`` — Engine
│            is deleted by SP1; the "invariant check at bootstrap"
│            concept moves to ``ObservabilityConfig.install`` and
│            other one-time config calls.)
├── PersistenceError                   (status 500)  BALLAST_PERSISTENCE_*
│   ├── ThreadNotFound                 (status 404)  BALLAST_PERSISTENCE_THREAD_NOT_FOUND
│   └── ThreadMetadataInvalid          (status 422)  BALLAST_PERSISTENCE_THREAD_METADATA_INVALID
├── AuthError                          (status 401)  BALLAST_AUTH_*
│   └── AuthorizationDenied            (status 403)  BALLAST_AUTH_FORBIDDEN
└── PatternError                       (status 500)  BALLAST_PATTERN_*
    ├── ReflectionExhausted                           BALLAST_PATTERN_REFLECTION_EXHAUSTED
    ├── MutationRejected                              BALLAST_PATTERN_MUTATION_REJECTED
    ├── HITLTimedOut                   (status 504)  BALLAST_PATTERN_HITL_TIMED_OUT
    ├── HITLDenied                     (status 403)  BALLAST_PATTERN_HITL_DENIED
    └── InsufficientDivergence                        BALLAST_PATTERN_INSUFFICIENT_DIVERGENCE
```

**Deleted from earlier draft:** `ContainerBindingMissing` (Container
itself deleted by SP1 — no bindings, no error); generic `RuntimeError_`
and `ValidationError_` parents (no behaviour beyond grouping; concrete
classes carry the semantic info directly via `code` + `status_code`).

**Subclass pattern (concrete example):**

```python
class InsufficientDivergence(PatternError):
    code = "BALLAST_PATTERN_INSUFFICIENT_DIVERGENCE"

    def __init__(
        self,
        *,
        produced: int,
        required: int,
        branch_outcomes: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            f"DivergentConvergent produced {produced} hypotheses; "
            f"min_hypotheses={required}",
            hint=(
                "Lower ``min_hypotheses``, raise ``best_of_n``, or add "
                "branches with higher temperature."
            ),
            context={
                "produced": produced,
                "required": required,
                "branch_outcomes": dict(branch_outcomes or {}),
            },
        )
        # Backwards-compat attributes that existing code reads.
        self.produced = produced
        self.required = required
        self.branch_outcomes = dict(branch_outcomes or {})
```

### C. HTTP middleware

**Location:** `src/ballast/api/error_middleware.py` (new).

**Why a Starlette middleware (not a FastAPI exception handler):**
exception handlers don't catch errors raised in background tasks /
streaming generators after the response has started; middleware can
log + tag the span even when it can't rewrite the response. We also
register a FastAPI `exception_handler(StateflowError)` so synchronous
route exceptions get the structured JSON path; middleware is the
fallback + observability hook.

```python
# src/ballast/api/error_middleware.py
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ballast.errors import StateflowError
from ballast.logging import get_logger

_logger = get_logger(__name__)

PROBLEM_JSON = "application/problem+json"


def _render(err: StateflowError, *, include_trace: bool) -> dict[str, Any]:
    body = {"error": err.to_dict()}
    if include_trace:
        import traceback
        body["error"]["traceback"] = traceback.format_exc()
    return body


async def stateflow_error_handler(request: Request, exc: StateflowError) -> JSONResponse:
    from ballast.settings import get_settings

    settings = get_settings()
    # Tri-state resolution: explicit override > auto from env=="dev" > off.
    expose = settings.api.expose_tracebacks
    if expose is None:
        expose = settings.observability.environment == "dev"
    _emit_log(exc)
    _emit_span_event(exc)
    return JSONResponse(
        content=_render(exc, include_trace=expose),
        status_code=exc.status_code,
        media_type=PROBLEM_JSON,
    )


class StateflowErrorMiddleware(BaseHTTPMiddleware):
    """Catches StateflowError that escapes streaming/background paths."""

    async def dispatch(self, request, call_next):
        try:
            return await call_next(request)
        except StateflowError as exc:
            return await stateflow_error_handler(request, exc)


def install_error_handlers(app: FastAPI) -> None:
    """Mount both the handler and the middleware. Idempotent.

    Called by SP1's ``create_app()`` after router registration. Guarded
    by ``settings.api.install_error_middleware``."""
    app.add_exception_handler(StateflowError, stateflow_error_handler)
    app.add_middleware(StateflowErrorMiddleware)
```

**Response shape (example for `ThreadNotFound`):**

```http
HTTP/1.1 404 Not Found
Content-Type: application/problem+json

{
  "error": {
    "code": "BALLAST_PERSISTENCE_THREAD_NOT_FOUND",
    "detail": "thread 7f3b... not found",
    "hint": "Confirm the thread id matches an active thread; check that it wasn't soft-deleted.",
    "context": {"thread_id": "7f3b8c00-..."}
  }
}
```

When `settings.api.expose_tracebacks=True` (dev only), the `error`
object additionally carries `"traceback": "..."`.

**Logfire integration** (`_emit_span_event`): when `has_logfire()`, the
handler attaches a span event on the current span:

```python
def _emit_span_event(exc: StateflowError) -> None:
    if not has_logfire():
        return
    import logfire
    with logfire.span("stateflow_error", _level="error") as span:
        span.set_attributes({
            "stateflow.error.code": exc.code,
            "stateflow.error.detail": exc.detail,
            "stateflow.error.hint": exc.hint or "",
            **{f"stateflow.error.context.{k}": _stringify(v)
               for k, v in exc.context.items()},
        })
        span.record_exception(exc)
```

Errors raised outside an HTTP request (e.g. background workflows) do
NOT auto-emit. Authors who want a span event in workflow code call
`logfire.span(...)` explicitly — automatic emission only fires when the
middleware/handler runs, because that's the only choke point we control.

### D. CLI / log formatting

**Location:** add `format_error(exc, *, color: bool | None = None) -> str`
to `src/ballast/errors.py`. Used by the framework's own
log path and by SP3 CLI commands.

**Strategy:** `rich` is in the lockfile (transitive via logfire). Soft
import, fallback to plain text. Color auto-detected via
`sys.stderr.isatty()` unless explicitly overridden.

```python
def format_error(exc: StateflowError, *, color: bool | None = None) -> str:
    """Pretty multi-line representation for stderr/logs.

    Returns ANSI-colored text when ``color=True`` (or auto-detected
    from a tty stderr). Plain text otherwise.
    """
    import sys
    use_color = color if color is not None else sys.stderr.isatty()
    try:
        if not use_color:
            raise ImportError
        from rich.console import Console
        from rich.text import Text
        # Render via rich to a string buffer; see implementation note.
        ...
    except ImportError:
        return _format_plain(exc)


def _format_plain(exc: StateflowError) -> str:
    lines = [
        f"[{exc.code}] {exc.detail}",
    ]
    if exc.hint:
        lines.append(f"  hint: {exc.hint}")
    if exc.context:
        lines.append("  context:")
        for k, v in exc.context.items():
            lines.append(f"    {k}: {v!r}")
    return "\n".join(lines)
```

**Sample colored output** (rich):

```
✗ BALLAST_PATTERN_INSUFFICIENT_DIVERGENCE
  DivergentConvergent produced 1 hypotheses; min_hypotheses=2

  hint  Lower min_hypotheses, raise best_of_n, or add branches with higher temperature.

  context
    produced         1
    required         2
    branch_outcomes  {'practical': 'ok', 'creative': 'failed'}
```

- Code is rendered red+bold.
- `hint` label is cyan; body is default.
- `context` key column is dim; values use `repr`.
- Final newline; intended to be written via `print(format_error(exc),
  file=sys.stderr)`.

### E. Migration of existing errors

| Old class | Module | New base / status | New code |
| --- | --- | --- | --- |
| `EngineInvariantViolation` | `runtime/engine.py` (deleted by SP1) | `ConfigurationInvariantViolation` (500) | `BALLAST_CONFIG_INVARIANT` |
| `PatternError` | `patterns/errors.py` | `StateflowError` (500) | `BALLAST_PATTERN` (used only via subclasses) |
| `ReflectionExhausted` | `patterns/errors.py` | `PatternError` (500) | `BALLAST_PATTERN_REFLECTION_EXHAUSTED` |
| `MutationRejected` | `patterns/errors.py` | `PatternError` (500) | `BALLAST_PATTERN_MUTATION_REJECTED` |
| `HITLTimedOut` | `patterns/errors.py` | `PatternError` (504) | `BALLAST_PATTERN_HITL_TIMED_OUT` |
| `HITLDenied` | `patterns/errors.py` | `PatternError` (403) | `BALLAST_PATTERN_HITL_DENIED` |
| `InsufficientDivergence` | `patterns/errors.py` | `PatternError` (500) | `BALLAST_PATTERN_INSUFFICIENT_DIVERGENCE` |
| `GroundedError` / `GroundedBuildError` / `GroundedHydrationError` | `grounded/...` | `StateflowError` (500) | `BALLAST_GROUNDED_*` |
| `BudgetExhausted` | `capabilities/...` | `StateflowError` (429) | `BALLAST_CAPABILITY_BUDGET_EXHAUSTED` |
| `SemanticLoopDetected` | `capabilities/helpers/...` | `StateflowError` (500) | `BALLAST_CAPABILITY_SEMANTIC_LOOP` |

**Strategy:** clean break (per project policy — single-repo caller). Each
subclass keeps the legacy positional/keyword args it already documents,
just routes them into `super().__init__(detail=..., hint=..., context=...)`
and re-stores them as instance attributes for back-compat reads.

**HTTPException callsites** (18 in `api/`): migrate gradually with this
pattern. Routers stop building `HTTPException` for domain conditions
and instead raise the matching `StateflowError`:

```python
# Before:
if thread is None:
    raise HTTPException(status_code=404, detail="thread not found")

# After:
if thread is None:
    raise ThreadNotFound(
        f"thread {thread_id} not found",
        context={"thread_id": str(thread_id)},
    )
```

Authentic transport-level errors (malformed JSON, missing required
query param) stay as `HTTPException` / FastAPI's built-in validation —
those aren't `StateflowError`s.

### F. Audit of env var reads → settings field mapping

| File:line | Env var | Settings field | Migration |
| --- | --- | --- | --- |
| `logging.py:104,125` | `BALLAST_LOG_LEVEL` | `settings.logging.level` | Keep direct `os.environ` read in `logging.py` — it's read at module import before settings can safely be instantiated. Settings field is the *new* canonical name; we keep reading the legacy `BALLAST_LOG_LEVEL` for back-compat (it equals `BALLAST_LOGGING__LEVEL` via the prefix anyway only if we add an alias). Concretely: add `validation_alias=AliasChoices("BALLAST_LOGGING__LEVEL", "BALLAST_LOG_LEVEL")` on `LoggingSettings.level`. |
| `observability/provider.py:103` | `LOGFIRE_TOKEN` | `settings.observability.logfire_token` | Read via `settings.observability.logfire_token.get_secret_value()` when present, fall back to checking `os.environ["LOGFIRE_TOKEN"]` (logfire SDK itself reads that var, so we keep diagnostic parity). |
| `examples/.../main.py:69` | `DBOS_DATABASE_URL` | `settings.dbos.database_url` | Replace `_default_dbos_database_url()` with `settings.dbos.database_url or _sqlite_fallback()`. Keep `DBOS_DATABASE_URL` as an alias (`AliasChoices`) to avoid breaking running deployments. |
| `examples/.../main.py:219,220` | `HOST`, `PORT` | App-level, NOT framework | Stay as plain env reads — these are uvicorn args, not framework config. Out of scope. |
| `examples/.../brainstorm_agents.py:59` | `OPENROUTER_API_KEY` | `settings.llm.openrouter.api_key` | Delete `_resolve_api_key`; the agent's `api_key` constructor arg falls back to `get_settings().llm.openrouter.api_key`. Raises `MissingDependencyError(code=BALLAST_CONFIG_DEPENDENCY_MISSING, hint="set BALLAST_LLM__OPENROUTER__API_KEY or pass api_key=…")` when absent. Add the same `AliasChoices` alias for the legacy `OPENROUTER_API_KEY`. |
| `examples/.../agent.py:188,191` | `OPENROUTER_MODEL`, `OPENROUTER_API_KEY` | `settings.llm.openrouter.default_model`, `.api_key` | Same pattern. |
| `examples/.../todo_approval_agent.py:145,148` | same | same | same |

**Aliasing strategy.** For each legacy var, the corresponding field
declares `validation_alias=AliasChoices("BALLAST_FOO__BAR", "LEGACY_NAME")`
so both names work during transition. Once notes-app + the test suite
read exclusively via `settings`, the legacy aliases can be removed in
a follow-up.

## Migration

Five PR-sized steps, in order:

1. **Add `errors.py` + `settings.py`**, no callsite migration. Tests
   for both modules (see "Testing"). Export from package root. No
   behaviour change yet.
2. **Migrate framework errors.** Update `patterns/errors.py`,
   `runtime/engine.py`, `grounded/...`, `capabilities/...` to inherit
   `StateflowError`. Keep legacy instance-attribute reads. Run full
   test suite — fix any test that asserted exact exception messages.
3. **Add error middleware** (`api/error_middleware.py`). SP1 owns
   wiring it into `create_app()`; until SP1 lands, the notes-app
   `build_app()` calls `install_error_handlers(app)` directly after
   `engine.fastapi_app(...)` returns. Document this as the
   pre-SP1 integration point.
4. **Migrate notes-app env reads to settings.** Delete
   `_resolve_api_key`, `_default_dbos_database_url`; replace with
   `settings.llm.openrouter.api_key.get_secret_value()` /
   `settings.dbos.database_url`. Convert agent ctor fallbacks.
5. **Migrate `HTTPException` callsites.** 18 sites in `api/`; map each
   to a `StateflowError` subclass (most → `ThreadNotFound`,
   `ContainerBindingMissing`, or new specific subclasses).

Each step is independently mergeable + reversible.

## Testing

**Settings (`tests/test_settings.py`):**
- Default values when no env set.
- Env var via prefix + nested delimiter (`BALLAST_DBOS__DATABASE_URL`).
- Legacy alias resolves (`DBOS_DATABASE_URL`).
- `SecretStr` doesn't appear in `repr` of the settings object.
- `reset_settings()` actually drops the cache.
- `.env` file in test cwd is loaded.

**Errors (`tests/test_errors.py`):**
- `to_dict()` shape matches the documented contract.
- Subclass with custom `code` + `status_code` resolves correctly.
- `format_error` plain output is stable (snapshot).
- `format_error` with `color=False` never emits ANSI codes.
- Hint + context fields propagate through subclass `__init__`.

**Middleware (`tests/api/test_error_middleware.py`):**
- Route raising `StateflowError` → 4xx/5xx + `application/problem+json` body.
- Subclass `status_code` honored.
- Streaming response that raises mid-stream → middleware logs but does
  NOT corrupt the in-flight stream (response status already sent;
  expected behaviour — we only catch pre-response errors).
- Background task that raises → handler runs, span event emitted (when
  logfire stub present).
- `settings.api.expose_tracebacks=True` → response body has `traceback`
  key; `False` → no `traceback` key; `None` (default) → auto on iff
  `observability.environment=="dev"`.
- `settings.api.install_error_middleware=False` → `install_error_handlers`
  short-circuits.

**Legacy back-compat (`tests/test_errors_legacy.py`):**
- Existing tests in `tests/patterns/test_divergent_convergent.py` that
  catch `InsufficientDivergence` continue to pass.
- `e.produced`, `e.required`, `e.branch_outcomes` instance attributes
  still readable on the new subclass.

**Observability:**
- `_emit_span_event` is a no-op when `has_logfire()` is False (asserted
  by importing and calling with logfire monkey-patched out).

## Open questions

1. **`format_error` ANSI rendering choice.** Glyph set fixed to `✗`
   for errors. Not load-bearing; could be tweaked later.
2. **Subclass naming collisions.** RESOLVED — generic `RuntimeError_` /
   `ValidationError_` parents dropped from hierarchy. Concrete subclasses
   inherit directly from `StateflowError` or from domain parents
   (`PersistenceError`, `PatternError` etc.) that don't shadow stdlib
   names.
3. **`expose_tracebacks` auto-flip.** RESOLVED — tri-state default
   (`None` → auto from `environment == "dev"`; explicit `True`/`False`
   overrides). See APISettings.expose_tracebacks docstring.

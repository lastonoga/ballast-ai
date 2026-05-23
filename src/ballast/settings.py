"""Single source of truth for framework env-driven config.

See ``docs/superpowers/specs/2026-05-22-sp2-settings-errors-design.md``
§A for the design rationale. Importing this module never has side
effects beyond importing pydantic-settings; the settings singleton is
only constructed on first ``get_settings()`` (or first proxy attribute
access).

Scope: framework-owned config ONLY (DBOS, observability, logging, API
middleware). App-specific config (LLM provider keys, model choices,
business-domain toggles) lives in the app's own settings module — see
``examples/notes-app/backend/src/notes_app/settings.py`` for the
canonical pattern.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBOSSettings(BaseModel):
    """Durable-execution runtime config.

    ``database_url`` may be ``None`` so apps that don't use DBOS still
    load settings without setting a useless var. The DBOS provider
    raises a structured error when it's needed but missing.
    """

    database_url: str | None = None
    app_name: str = "pydantic-ai-stateflow"


class ObservabilitySettings(BaseModel):
    """Defaults for ``ObservabilityConfig`` (SP1). Importing settings does
    NOT configure logfire — apps construct ``ObservabilityConfig(...)``
    and call ``.install()`` explicitly.
    """

    logfire_token: SecretStr | None = None
    service_name: str = "pydantic-ai-stateflow"
    environment: str = "dev"
    instrument_pydantic_ai: bool = True
    instrument_httpx: bool = True
    instrument_fastapi: bool = True


class APISettings(BaseModel):
    """HTTP-layer toggles consumed by middleware."""

    # When True (default), BallastErrorMiddleware is installed by
    # SP1's create_app(). Apps that want to handle BallastError
    # themselves set this False.
    install_error_middleware: bool = True
    # Whether stack traces are included in problem+json bodies.
    # ``None`` (default) → auto: on iff ``observability.environment == "dev"``.
    # Explicit ``True`` / ``False`` overrides the auto-detect. Safer default
    # for prod (off); convenient for dev (on).
    expose_tracebacks: bool | None = None


class LoggingSettings(BaseModel):
    """Framework logger config. Mirrors the legacy ``BALLAST_LOG_LEVEL``
    env var so existing deployments don't break.
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None


class BallastSettings(BaseSettings):
    """Single source of truth for framework env-driven config.

    Usage::

        from ballast import settings
        url = settings.dbos.database_url

    Env vars use ``BALLAST_`` prefix + ``__`` for nesting::

        BALLAST_DBOS__DATABASE_URL=postgresql://...
        BALLAST_OBSERVABILITY__LOGFIRE_TOKEN=...
    """

    model_config = SettingsConfigDict(
        env_prefix="BALLAST_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    dbos: DBOSSettings = Field(default_factory=DBOSSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Auto-run ``alembic upgrade head`` during FastAPI startup. Opt-in
    # (default False) so framework + app tests never trigger migrations.
    # Env: ``BALLAST_AUTO_MIGRATE=true``.
    auto_migrate: bool = False

    # Legacy env-var aliases (vars without the ``BALLAST_`` prefix that
    # predate this settings module). Mapped here rather than via
    # ``AliasChoices`` because pydantic-settings does not honour
    # validation_alias on fields of nested ``BaseModel``s — only on
    # fields of the top-level ``BaseSettings``. Tuple of (env_var, nested
    # path) entries; the validator promotes them into the nested struct
    # iff the canonical ``BALLAST_*`` form isn't also set.
    _LEGACY_ALIASES: ClassVar[tuple[tuple[str, tuple[str, ...]], ...]] = (
        ("DBOS_DATABASE_URL", ("dbos", "database_url")),
        ("BALLAST_LOG_LEVEL", ("logging", "level")),
    )

    @classmethod
    def _read_dotenv_legacy(cls) -> dict[str, str]:
        """Read just the legacy alias names from .env (if present).

        We rely on pydantic-settings to pick up the canonical
        ``BALLAST_*`` keys from .env automatically; this only fills
        gaps for the unprefixed legacy names that won't otherwise be
        seen by the nested-model fields.
        """
        env_file = cls.model_config.get("env_file", ".env")
        if not env_file:
            return {}
        try:
            from dotenv import dotenv_values
        except ImportError:
            return {}
        try:
            return {
                k: v for k, v in dotenv_values(env_file).items()
                if v is not None and any(k == name for name, _ in cls._LEGACY_ALIASES)
            }
        except OSError:
            return {}

    @model_validator(mode="before")
    @classmethod
    def _apply_legacy_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        dotenv_legacy = cls._read_dotenv_legacy()
        for env_var, path in cls._LEGACY_ALIASES:
            value = os.environ.get(env_var) or dotenv_legacy.get(env_var)
            if value is None:
                continue
            # Walk into the dict; don't overwrite a value the canonical
            # source already provided.
            cursor = data
            for key in path[:-1]:
                sub = cursor.get(key)
                if not isinstance(sub, dict):
                    sub = {}
                    cursor[key] = sub
                cursor = sub
            cursor.setdefault(path[-1], value)
        return data


@lru_cache(maxsize=1)
def _get_settings() -> BallastSettings:
    return BallastSettings()


def get_settings() -> BallastSettings:
    """Return the process-wide cached settings instance.

    First call instantiates and caches; subsequent calls return the
    cached object. Tests reset via ``reset_settings()``.
    """
    return _get_settings()


def reset_settings() -> None:
    """Clear the cache. ONLY for tests — never call in production."""
    _get_settings.cache_clear()


class _SettingsProxy:
    """Lazy proxy so ``settings.dbos.database_url`` works without an
    explicit ``get_settings()`` call at every read site."""

    def __getattr__(self, item: str) -> Any:
        return getattr(get_settings(), item)

    def __repr__(self) -> str:
        return repr(get_settings())


settings: BallastSettings = _SettingsProxy()  # type: ignore[assignment]


__all__ = [
    "APISettings",
    "DBOSSettings",
    "LoggingSettings",
    "ObservabilitySettings",
    "BallastSettings",
    "get_settings",
    "reset_settings",
    "settings",
]

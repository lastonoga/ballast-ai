"""Tests for ``ballast.settings``.

Cover defaults, env-var resolution via prefix+delimiter, legacy aliases,
SecretStr repr safety, cache reset, and .env file loading.
"""
from __future__ import annotations

import pytest

from ballast.settings import (
    BallastSettings,
    get_settings,
    reset_settings,
)


# ---------- Defaults ----------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip env vars that could leak in from the developer's shell."""
    for var in [
        "BALLAST_DBOS__DATABASE_URL",
        "BALLAST_LOGGING__LEVEL",
        "BALLAST_LOG_LEVEL",
        "BALLAST_OBSERVABILITY__LOGFIRE_TOKEN",
        "DBOS_DATABASE_URL",
    ]:
        monkeypatch.delenv(var, raising=False)
    reset_settings()
    yield
    reset_settings()


def test_defaults_when_env_unset() -> None:
    s = BallastSettings()
    assert s.dbos.database_url is None
    assert s.dbos.app_name == "pydantic-ai-stateflow"
    assert s.observability.environment == "dev"
    assert s.observability.service_name == "pydantic-ai-stateflow"
    assert s.observability.instrument_pydantic_ai is True
    assert s.api.install_error_middleware is True
    assert s.api.expose_tracebacks is None
    assert s.logging.level is None


# ---------- Nested env-var resolution ----------


def test_env_var_resolves_via_nested_delimiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BALLAST_DBOS__DATABASE_URL", "postgresql://example/db")
    s = BallastSettings()
    assert s.dbos.database_url == "postgresql://example/db"


# ---------- Legacy alias support ----------


def test_legacy_dbos_alias_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DBOS_DATABASE_URL", "postgresql://legacy/db")
    s = BallastSettings()
    assert s.dbos.database_url == "postgresql://legacy/db"


def test_legacy_log_level_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BALLAST_LOG_LEVEL", "DEBUG")
    s = BallastSettings()
    assert s.logging.level == "DEBUG"


# ---------- SecretStr safety ----------


def test_secret_str_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "BALLAST_OBSERVABILITY__LOGFIRE_TOKEN",
        "lf-super-secret-12345",
    )
    s = BallastSettings()
    assert "lf-super-secret-12345" not in repr(s)
    assert "lf-super-secret-12345" not in str(s)


# ---------- Cache lifecycle ----------


def test_reset_settings_drops_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    monkeypatch.setenv("BALLAST_DBOS__DATABASE_URL", "postgresql://new/db")
    # Without reset, still cached
    assert get_settings() is s1
    reset_settings()
    s3 = get_settings()
    assert s3 is not s1
    assert s3.dbos.database_url == "postgresql://new/db"


# ---------- .env file loading ----------


def test_env_file_loaded_from_cwd(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("BALLAST_DBOS__DATABASE_URL=postgresql://envfile/db\n")
    monkeypatch.chdir(tmp_path)
    reset_settings()
    s = BallastSettings()
    assert s.dbos.database_url == "postgresql://envfile/db"

"""Tests for ``pydantic_ai_stateflow.settings``.

Cover defaults, env-var resolution via prefix+delimiter, legacy aliases,
SecretStr repr safety, cache reset, and .env file loading.
"""
from __future__ import annotations

import pytest

from pydantic_ai_stateflow.settings import (
    StateflowSettings,
    get_settings,
    reset_settings,
)


# ---------- Defaults ----------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip env vars that could leak in from the developer's shell."""
    for var in [
        "STATEFLOW_DBOS__DATABASE_URL",
        "STATEFLOW_LLM__OPENROUTER__API_KEY",
        "STATEFLOW_LLM__OPENROUTER__DEFAULT_MODEL",
        "STATEFLOW_LOGGING__LEVEL",
        "STATEFLOW_LOG_LEVEL",
        "STATEFLOW_OBSERVABILITY__LOGFIRE_TOKEN",
        "DBOS_DATABASE_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
    ]:
        monkeypatch.delenv(var, raising=False)
    reset_settings()
    yield
    reset_settings()


def test_defaults_when_env_unset() -> None:
    s = StateflowSettings()
    assert s.dbos.database_url is None
    assert s.dbos.app_name == "pydantic-ai-stateflow"
    assert s.observability.environment == "dev"
    assert s.observability.service_name == "pydantic-ai-stateflow"
    assert s.observability.instrument_pydantic_ai is True
    assert s.llm.openrouter.api_key is None
    assert s.llm.openrouter.default_model is None
    assert s.api.install_error_middleware is True
    assert s.api.expose_tracebacks is None
    assert s.logging.level is None


# ---------- Nested env-var resolution ----------


def test_env_var_resolves_via_nested_delimiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STATEFLOW_DBOS__DATABASE_URL", "postgresql://example/db")
    s = StateflowSettings()
    assert s.dbos.database_url == "postgresql://example/db"


# ---------- Legacy alias support ----------


def test_legacy_dbos_alias_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DBOS_DATABASE_URL", "postgresql://legacy/db")
    s = StateflowSettings()
    assert s.dbos.database_url == "postgresql://legacy/db"


def test_legacy_openrouter_aliases_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-legacy")
    monkeypatch.setenv("OPENROUTER_MODEL", "qwen/qwen3-coder")
    s = StateflowSettings()
    assert s.llm.openrouter.api_key is not None
    assert s.llm.openrouter.api_key.get_secret_value() == "sk-or-legacy"
    assert s.llm.openrouter.default_model == "qwen/qwen3-coder"


def test_legacy_log_level_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STATEFLOW_LOG_LEVEL", "DEBUG")
    s = StateflowSettings()
    assert s.logging.level == "DEBUG"


# ---------- SecretStr safety ----------


def test_secret_str_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-super-secret-12345")
    s = StateflowSettings()
    assert "sk-or-super-secret-12345" not in repr(s)
    assert "sk-or-super-secret-12345" not in str(s)


# ---------- Cache lifecycle ----------


def test_reset_settings_drops_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    monkeypatch.setenv("STATEFLOW_DBOS__DATABASE_URL", "postgresql://new/db")
    # Without reset, still cached
    assert get_settings() is s1
    reset_settings()
    s3 = get_settings()
    assert s3 is not s1
    assert s3.dbos.database_url == "postgresql://new/db"


# ---------- .env file loading ----------


def test_env_file_loaded_from_cwd(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("STATEFLOW_DBOS__DATABASE_URL=postgresql://envfile/db\n")
    monkeypatch.chdir(tmp_path)
    reset_settings()
    s = StateflowSettings()
    assert s.dbos.database_url == "postgresql://envfile/db"

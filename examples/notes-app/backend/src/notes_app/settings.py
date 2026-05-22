"""Notes-app settings — pydantic-settings module owned by THIS app.

The framework (``ballast.settings``) only knows about
framework-owned config (DBOS, observability, logging, API middleware).
App-specific config (LLM provider keys, model choices) lives here.

Env vars:
- ``NOTES_APP_OPENROUTER_API_KEY`` (legacy ``OPENROUTER_API_KEY``)
- ``NOTES_APP_OPENROUTER_DEFAULT_MODEL`` (legacy ``OPENROUTER_MODEL``)
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class NotesAppSettings(BaseSettings):
    """Notes-app config. Read once at startup via ``get_notes_settings()``."""

    openrouter_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "NOTES_APP_OPENROUTER_API_KEY",
            "OPENROUTER_API_KEY",
        ),
    )
    openrouter_default_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "NOTES_APP_OPENROUTER_DEFAULT_MODEL",
            "OPENROUTER_MODEL",
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="NOTES_APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_notes_settings() -> NotesAppSettings:
    return NotesAppSettings()


__all__ = ["NotesAppSettings", "get_notes_settings"]

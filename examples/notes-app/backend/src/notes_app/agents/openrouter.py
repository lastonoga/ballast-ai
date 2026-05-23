"""Single-source-of-truth OpenRouter model + settings construction.

All four notes-app agents (``NotesAgent``, ``NotesTodoApprovalAgent``,
``BrainstormDivergentAgent``, ``BrainstormSynthesizerAgent``) speak to
OpenRouter through the same dance: resolve the API key from settings,
fail loud if missing, build an ``OpenRouterModel`` with the right
``ModelProfile`` (qwen3.6 override etc.), and configure standard knobs
(no reasoning, usage tracking on).

This module collapses that into two helpers:

  - :func:`build_openrouter_model` — model construction
  - :func:`default_model_settings` — knob defaults parameterised by
    temperature

Callers that need to override individual knobs can build their own
``OpenRouterModelSettings`` directly; this helper is just the common
case.
"""

from __future__ import annotations

from pydantic_ai.models.openrouter import (
    OpenRouterModel,
    OpenRouterModelSettings,
)
from pydantic_ai.providers.openrouter import OpenRouterProvider

from ballast.errors import MissingDependencyError

from notes_app.agents.openrouter_profile import profile_for
from notes_app.settings import get_notes_settings

_DEFAULT_MODEL = "qwen/qwen3.6-plus"


def build_openrouter_model(model_name: str | None = None) -> OpenRouterModel:
    """Construct an ``OpenRouterModel`` ready for an ``Agent``.

    Resolves the model id in priority order:

      1. ``model_name`` arg, if supplied
      2. ``settings.openrouter_default_model``, if set
      3. :data:`_DEFAULT_MODEL` fallback

    Resolves the API key from ``settings.openrouter_api_key``; raises
    :class:`MissingDependencyError` if unset (callers see a clear,
    actionable error instead of a downstream 401).

    Applies the qwen3.6 ``ModelProfile`` override via
    :func:`profile_for` when applicable; falls back to pydantic-ai's
    built-in registry for everything else.
    """
    settings = get_notes_settings()
    resolved_model = (
        model_name
        or settings.openrouter_default_model
        or _DEFAULT_MODEL
    )
    api_key = (
        settings.openrouter_api_key.get_secret_value()
        if settings.openrouter_api_key else None
    )
    if not api_key:
        raise MissingDependencyError(
            "OpenRouter API key required to build a notes-app agent",
            hint=(
                "Set NOTES_APP_OPENROUTER_API_KEY (or legacy "
                "OPENROUTER_API_KEY) env var"
            ),
            context={"setting": "notes_app.openrouter_api_key"},
        )
    return OpenRouterModel(
        resolved_model,
        provider=OpenRouterProvider(api_key=api_key),
        profile=profile_for(resolved_model),
    )


def default_model_settings(
    *, temperature: float,
) -> OpenRouterModelSettings:
    """Standard notes-app OpenRouter knobs.

    Same shape across every agent: drop reasoning (cost + latency on
    every turn), include usage in the response so cost dashboards
    have data to chart. Temperature is the only thing that varies per
    agent (notes 0.7, approval 0.3, divergent 0.9, synth 0.2).
    """
    return OpenRouterModelSettings(
        temperature=temperature,
        openrouter_reasoning={"effort": "none"},
        openrouter_usage={"include": True},
    )


__all__ = ["build_openrouter_model", "default_model_settings"]

"""OpenRouter-backed pydantic-ai Agent for the notes app.

Iteration 3 wires CRUD tools into the agent — the agent takes a
``NoteToolDeps`` (a repo + a tenant id) and can ``create_note``,
``list_notes``, ``search_notes``, ``edit_note``, and ``delete_note`` on
the user's behalf. Tool calls surface as Vercel AI SDK v6 ``tool-*``
chunks thanks to ``pydantic_ai.ui.vercel_ai.VercelAIAdapter``, which the
framework's ``build_streaming_router`` delegates to for the wire encoding.
``delete_note`` is declared ``requires_approval=True`` so it surfaces as
an ``approval-requested`` part the frontend can render as an approve/
cancel card.

Output shape decision (iter 3 round 2):
  We use ``output_type=str`` — NOT a structured ``BaseModel`` envelope.
  Reasons:

  - ``ToolOutput`` (pydantic-ai's default for ``BaseModel``) forces
    ``tool_choice="required"`` to drive the synthetic ``final_result``
    tool. OpenRouter's Qwen 3.6 endpoints reject that value.
  - ``NativeOutput`` accepts ``response_format: json_schema`` but the
    model then returns the JSON directly WITHOUT calling real tools.
  - ``PromptedOutput`` works but pollutes the streamed text with JSON.

  ``output_type=str`` sidesteps all three.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import (
    OpenRouterModel,
    OpenRouterModelSettings,
    OpenRouterReasoning,
)
from pydantic_ai.providers.openrouter import OpenRouterProvider

if TYPE_CHECKING:
    from uuid import UUID

    from notes_app.notes.repository import NoteRepository
    from notes_app.notes.tools import NoteToolDeps

DEFAULT_MODEL = "qwen/qwen3.6-plus"
DEFAULT_TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "You are the assistant inside a personal notes app. "
    "You have tools to create, list, search, edit, and delete notes on "
    "the user's behalf. "
    "When the user asks you to create / find / change / remove a note, "
    "USE THE TOOLS to actually do it — do not just describe what you "
    "would do. After running the tools, briefly confirm what happened "
    "(e.g. 'Saved your note titled \"X\"'). "
    "If the user is chatting and not asking for a note action, just "
    "reply conversationally."
)


def build_agent(
    *,
    model_name: str | None = None,
    api_key: str | None = None,
) -> Agent[NoteToolDeps, str]:
    """Build the OpenRouter-backed agent and register the note tools."""
    from notes_app.notes.tools import NoteToolDeps as _Deps
    from notes_app.notes.tools import register_note_tools

    resolved_model = model_name or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY env var is required to build the agent",
        )

    model = OpenRouterModel(
        resolved_model,
        provider=OpenRouterProvider(api_key=resolved_key),
    )
    from pydantic_ai_stateflow.grounded import register_grounded_tools

    agent: Agent[_Deps, str] = Agent(
        model=model,
        output_type=str,
        deps_type=_Deps,
        system_prompt=SYSTEM_PROMPT,
    )
    register_note_tools(agent)
    # Install per-run ``prepare`` hooks on tools whose params are
    # ``Annotated[Ref[T], Selector(...)]``. Replaces the iter-3
    # hand-rolled ``_prepare_note_id_closed_set``.
    register_grounded_tools(agent)
    return agent


def build_notes_deps_factory(
    note_repo: NoteRepository,
) -> Callable[..., NoteToolDeps]:
    """Return a ``deps_factory`` for ``build_streaming_router``.

    Mints a fresh ``NoteToolDeps`` per HTTP request, bound to the
    requesting tenant. The factory signature matches the framework
    contract: ``(thread_id, tenant_id, message)`` keyword args.
    """
    from notes_app.notes.tools import NoteToolDeps

    def factory(*, tenant_id: UUID, **_kwargs: object) -> NoteToolDeps:
        return NoteToolDeps(repo=note_repo, tenant_id=tenant_id)

    return factory


def _parse_bool_env(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_logger = logging.getLogger("notes_app.agent")


def build_model_settings() -> OpenRouterModelSettings | None:
    """Read ``OPENROUTER_*`` env vars into ``OpenRouterModelSettings``.

    Recognized env vars:

    - ``OPENROUTER_TEMPERATURE`` (float, default 0.7)
    - ``OPENROUTER_REASONING_EFFORT`` (minimal|low|medium|high|xhigh) — optional
    - ``OPENROUTER_REASONING_MAX_TOKENS`` (int) — optional, mutually exclusive with effort
    - ``OPENROUTER_REASONING_ENABLED`` (bool) — optional

    OpenRouter rejects requests carrying BOTH ``reasoning.effort`` and
    ``reasoning.max_tokens`` (400 "Only one of … can be specified").
    This function enforces the constraint by preferring ``effort`` and
    dropping ``max_tokens`` with a logged warning when both are set —
    matches OpenRouter's docs (effort is the higher-level knob).

    Returns ``None`` only if NO env var was set; otherwise always
    populates ``temperature`` (defaulting to ``DEFAULT_TEMPERATURE``)
    plus any reasoning fields that were configured. Logs the final
    settings at INFO so misconfiguration is easy to spot at boot.
    """
    settings: OpenRouterModelSettings = OpenRouterModelSettings()
    populated = False

    temp_raw = os.environ.get("OPENROUTER_TEMPERATURE")
    if temp_raw is not None:
        try:
            settings["temperature"] = float(temp_raw)
            populated = True
        except ValueError:
            _logger.warning(
                "OPENROUTER_TEMPERATURE=%r is not a valid float — ignored",
                temp_raw,
            )
    else:
        # Always plumb the documented default so behavior is reproducible.
        settings["temperature"] = DEFAULT_TEMPERATURE
        populated = True

    reasoning: OpenRouterReasoning = {}
    effort = os.environ.get("OPENROUTER_REASONING_EFFORT")
    max_tokens_raw = os.environ.get("OPENROUTER_REASONING_MAX_TOKENS")

    if effort and max_tokens_raw:
        _logger.warning(
            "OpenRouter rejects requests carrying both reasoning.effort "
            "and reasoning.max_tokens — keeping effort=%r, dropping "
            "max_tokens=%r. Unset one in .env to silence this warning.",
            effort, max_tokens_raw,
        )
        max_tokens_raw = None  # effort wins

    if effort:
        # OpenRouterReasoning.effort is a Literal — TypedDict allows any
        # string assignment but the OpenRouter API will reject unknowns.
        reasoning["effort"] = effort  # type: ignore[typeddict-item]
    elif max_tokens_raw is not None:
        try:
            reasoning["max_tokens"] = int(max_tokens_raw)
        except ValueError:
            _logger.warning(
                "OPENROUTER_REASONING_MAX_TOKENS=%r is not a valid int — ignored",
                max_tokens_raw,
            )

    enabled = _parse_bool_env("OPENROUTER_REASONING_ENABLED")
    if enabled is not None:
        reasoning["enabled"] = enabled

    if reasoning:
        settings["openrouter_reasoning"] = reasoning
        populated = True

    if populated:
        _logger.info("OpenRouter model_settings: %s", dict(settings))
    return settings if populated else None

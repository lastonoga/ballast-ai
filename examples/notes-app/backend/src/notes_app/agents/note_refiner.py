"""Reflection loop that polishes ``(title, body)`` before persistence.

When ``create_note`` is called, the agent has produced *some* draft
title/body but they're often vague ("note", "meeting", one-liner
bodies). Reflection runs a critic over the draft; if it fails, a
refiner agent rewrites with the critic's feedback in hand. Bounded
by ``max_iter=2`` so a tool call never spirals.

On exhaustion the caller persists ``ReflectionExhausted.last_draft``
anyway — losing the user's note to a picky critic is worse than
saving an imperfect-but-improved version.

Module-level instances because ``Reflection`` (a
``DBOSConfiguredInstance``) must register BEFORE ``DBOS.launch()``.
"""
from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import (
    OpenRouterModel,
    OpenRouterModelSettings,
)
from pydantic_ai.providers.openrouter import OpenRouterProvider

from ballast import (
    Critique,
    LLMJudge,
    Reflection,
)

from notes_app.settings import get_notes_settings


class ProposedNote(BaseModel):
    """Draft + refined shape Reflection passes around the loop."""

    __hitl_kind__ = "note.create"

    title: str
    body: str


_REFINER_SYSTEM_PROMPT = (
    "You refine note proposals based on critic feedback. "
    "Given the current title + body and a list of issues + "
    "suggestions, return an IMPROVED title and body that addresses "
    "every issue. Preserve the user's original intent — refine, "
    "don't rewrite from scratch. "
    "Title must be specific (NOT vague like 'note', 'untitled', or "
    "just a topic word like 'meeting'). "
    "Body must contain actionable detail or specific information "
    "beyond a one-line restate of the title."
)


def _build_refiner_agent() -> "Agent[None, ProposedNote] | None":
    """Construct the pydantic-ai Agent used as the Reflection writer.

    Claude Haiku — same model the LLMJudge uses (see ``main.py``);
    Qwen 3.6 endpoints on OpenRouter reject ``tool_choice="required"``
    which ``output_type=ProposedNote`` (BaseModel) needs.

    Returns ``None`` if no OpenRouter API key is configured — apps
    (and tests) without a key skip the refinement loop entirely
    (see ``note_refiner`` below). The refiner is a quality
    enhancement, not a correctness requirement.
    """
    settings = get_notes_settings()
    api_key = (
        settings.openrouter_api_key.get_secret_value()
        if settings.openrouter_api_key else None
    )
    if not api_key:
        return None
    return Agent(
        model=OpenRouterModel(
            "anthropic/claude-haiku-4.5",
            provider=OpenRouterProvider(api_key=api_key),
        ),
        output_type=ProposedNote,
        system_prompt=_REFINER_SYSTEM_PROMPT,
        model_settings=OpenRouterModelSettings(
            temperature=0.3,         # a bit of creativity for the rewrite
            openrouter_usage={"include": True},
        ),
    )


_refiner_agent: "Agent[None, ProposedNote] | None" = _build_refiner_agent()


async def _writer(
    task: ProposedNote, history: list[Critique],
) -> ProposedNote:
    """Reflection writer: first iteration returns the original draft;
    subsequent iterations call the refiner with the last critique.
    """
    if not history:
        return task
    assert _refiner_agent is not None  # guarded by note_refiner construction
    last = history[-1]
    issues = "\n".join(f"- {issue}" for issue in last.issues) or "(none)"
    suggestions = (
        "\n".join(f"- {sugg}" for sugg in last.suggestions) or "(none)"
    )
    prompt = (
        f"Refine this note based on the critic's feedback.\n\n"
        f"Current draft:\n"
        f"Title: {task.title}\n"
        f"Body: {task.body}\n\n"
        f"Issues to fix:\n{issues}\n\n"
        f"Suggestions:\n{suggestions}"
    )
    result = await _refiner_agent.run(prompt)
    return result.output


_note_quality_judge = LLMJudge(
    rubric=(
        "Note quality. Pass ONLY if BOTH:\n"
        "1. Title is specific — captures the topic AND a hint of "
        "context (NOT vague like 'note', 'untitled', or a bare topic "
        "word).\n"
        "2. Body contains actionable detail or specific information "
        "beyond what's in the title — NOT a one-line restate."
    ),
    threshold=0.7,
)


# Without a refiner agent (no API key) the critic LLM is also
# unconfigured — wrapping in Reflection adds latency without value
# and forces tests through a DBOS workflow. Expose ``None`` so the
# caller skips the loop and saves the draft directly.
note_refiner: Reflection[ProposedNote, ProposedNote] | None = (
    Reflection(
        writer=_writer,
        critic=_note_quality_judge,
        max_iter=2,              # tools shouldn't loop forever
        config_name="note-refiner",
    )
    if _refiner_agent is not None
    else None
)


__all__ = ["ProposedNote", "note_refiner"]

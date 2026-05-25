"""Built-in ``GoalSource`` implementations.

Apps choose where the original objective comes from:

- ``FirstUserMessage`` — first user prompt in trace (long-running sessions).
- ``LastUserMessage`` — most recent user prompt (per-turn).
- ``WorkflowInput`` — ``ctx.workflow_input`` (workflow surface).
- ``ExplicitGoal(text)`` — statically pinned at wire-up time.
"""
from __future__ import annotations

from pydantic_ai.messages import ModelRequest, UserPromptPart

from ballast.drift._protocols import DriftContext


def _extract_user_prompt(msg) -> str | None:
    if not isinstance(msg, ModelRequest):
        return None
    for part in msg.parts:
        if isinstance(part, UserPromptPart):
            content = part.content
            if isinstance(content, str):
                return content
            # Multimodal content: stringify the structure
            return str(content)
    return None


class FirstUserMessage:
    """First user message in the trace."""

    async def goal(self, ctx: DriftContext) -> str:
        for msg in ctx.messages:
            text = _extract_user_prompt(msg)
            if text is not None:
                return text
        return ""


class LastUserMessage:
    """Most recent user message in the trace."""

    async def goal(self, ctx: DriftContext) -> str:
        for msg in reversed(ctx.messages):
            text = _extract_user_prompt(msg)
            if text is not None:
                return text
        return ""


class WorkflowInput:
    """``ctx.workflow_input`` stringified.

    For workflow surface where no message trace exists. Plain strings pass
    through; non-strings are ``str(...)``-ified.
    """

    async def goal(self, ctx: DriftContext) -> str:
        wf_input = ctx.workflow_input
        if wf_input is None:
            return ""
        if isinstance(wf_input, str):
            return wf_input
        return str(wf_input)


class ExplicitGoal:
    """Goal string pinned at construction time."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def goal(self, ctx: DriftContext) -> str:
        return self._text


__all__ = ["ExplicitGoal", "FirstUserMessage", "LastUserMessage", "WorkflowInput"]

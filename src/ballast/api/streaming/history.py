"""Helpers to bridge server-persisted ``Message`` rows ↔ pydantic-ai
``ModelMessage`` lists.

The framework is **server-stateful**: the source of truth for a thread's
history is the backend's ``ThreadRepository``, not the AG-UI body sent by
the client. Before invoking the agent we hydrate the persisted rows into
the ``message_history=`` kwarg of pydantic-ai's run methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from ballast.persistence.thread.domain import Message


def extract_text(parts: list[Any]) -> str:
    """Concatenate all text parts in order. Non-text parts are skipped.

    Accepts both validated message-part dataclasses (anything with a
    ``type == "text"`` and a ``text: str`` attribute) and raw dicts
    (so it works on persisted repo rows ``[{"type": "text", "text": "..."}]``).
    """
    chunks: list[str] = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            text = p.get("text")
            if isinstance(text, str):
                chunks.append(text)
        else:
            t = getattr(p, "type", None)
            if t == "text":
                text_attr = getattr(p, "text", None)
                if isinstance(text_attr, str):
                    chunks.append(text_attr)
    return "".join(chunks)


def messages_to_model_history(
    rows: list[Message],
    *,
    drop_prompt: str | None = None,
) -> list[ModelMessage]:
    """Convert persisted ``Message`` rows → pydantic-ai ``ModelMessage`` list.

    Conversion rules (verified against pydantic-ai 1.97.0):

    - ``role == "user"`` → ``ModelRequest(parts=[UserPromptPart(content)])``
    - ``role == "assistant"`` → ``ModelResponse(parts=[TextPart(content)])``
      (single text part; tool-call replay is out of scope —
      the model re-derives any tool history from the resulting reply text)
    - empty text → row dropped (no point seeding empty turns)

    ``drop_prompt`` deduplicates the just-persisted current user turn.
    The router persists the user message BEFORE invoking the agent, so
    repo history at runner time includes that turn. The agent's incoming
    AG-UI body ALSO carries that user turn — so we strip the trailing
    user row if its text matches the prompt verbatim. Robust to mid-
    history user turns (we only inspect the LAST row).

    Timestamps are preserved from ``Message.created_at`` so observability
    traces show real wall-clock ordering rather than synthetic "now".
    """
    from pydantic_ai.messages import (  # noqa: PLC0415
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    pruned = list(rows)
    if drop_prompt is not None and pruned:
        last = pruned[-1]
        if last.role == "user" and extract_text(last.parts) == drop_prompt:
            pruned = pruned[:-1]

    out: list[ModelMessage] = []
    for row in pruned:
        text = extract_text(row.parts)
        if not text:
            continue
        if row.role == "user":
            out.append(
                ModelRequest(
                    parts=[UserPromptPart(content=text, timestamp=row.created_at)],
                    timestamp=row.created_at,
                ),
            )
        elif row.role == "assistant":
            out.append(
                ModelResponse(
                    parts=[TextPart(content=text)],
                    timestamp=row.created_at,
                ),
            )
        # silently skip other roles (system/tool)
    return out


__all__ = ["extract_text", "messages_to_model_history"]

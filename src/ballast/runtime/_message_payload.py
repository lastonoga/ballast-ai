"""Strict wire contract for ``message-added`` event-log payloads.

The framework's SSE consumer hands ``message-added`` payloads to
assistant-ui as ``UIMessage``-shaped dicts. assistant-ui keeps them in
state and echoes them back in the next POST body — pydantic-ai's
``UIMessage`` validator is ``extra='forbid'`` and rejects any field
that wasn't expected, so payload-schema drift here turns into a
production 500 on the next user message.

History repeats: a ``state`` field on data-* parts blew this up once
(commit ``6cb3326``); a ``transient`` field on the broadcaster payload
blew it up again (commit ``90abe2c``). Adding a pydantic model with
``extra='forbid'`` as the construction contract means any future leak
fails at the write site (in tests, on the first emit) instead of in
production after a thread accumulates state.

Every site that writes ``event_log.append(kind="message-added", ...)``
builds its payload via :func:`build_message_added_payload`.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class MessageAddedPayload(BaseModel):
    """Field-shape contract for ``message-added`` event-log payloads.

    Field presence is enforced strictly (``extra='forbid'``); the
    contents of ``parts`` are intentionally untyped because the union
    of valid UI parts (text / reasoning / data-* / tool-* / file …)
    is owned by pydantic-ai and evolves there. The strict round-trip
    check against pydantic-ai's full ``UIMessagePart`` union lives in
    the body parser on the way back in — this contract just guards
    the obvious "extra top-level field" class of bugs.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    role: Literal["system", "user", "assistant", "tool"]
    parts: list[dict[str, Any]]


def build_message_added_payload(
    *,
    message_id: str,
    role: str,
    parts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Construct + validate a ``message-added`` payload.

    Returns the JSON-serialisable dict ready to hand to
    ``event_log.append(kind="message-added", payload=...)``. Raises
    ``pydantic.ValidationError`` if any field is missing / malformed.
    """
    return MessageAddedPayload(
        id=message_id,
        role=role,  # type: ignore[arg-type]  # validated by the model
        parts=parts,
    ).model_dump(mode="json")


__all__ = ["MessageAddedPayload", "build_message_added_payload"]

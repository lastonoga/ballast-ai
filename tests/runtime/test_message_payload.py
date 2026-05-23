"""``MessageAddedPayload`` is the strict wire contract for the
``message-added`` event-log payload — guards against extra fields
leaking from backend writes through SSE into the frontend body echo
where pydantic-ai's ``UIMessage`` (``extra='forbid'``) would reject
them with a 500.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from ballast.runtime._message_payload import (
    MessageAddedPayload,
    build_message_added_payload,
)


def test_canonical_payload_round_trips() -> None:
    payload = build_message_added_payload(
        message_id="m1",
        role="assistant",
        parts=[{"type": "text", "text": "hi", "state": "done"}],
    )
    assert payload == {
        "id": "m1",
        "role": "assistant",
        "parts": [{"type": "text", "text": "hi", "state": "done"}],
    }


def test_extra_top_level_field_rejected() -> None:
    """Any future attempt to add e.g. ``transient`` /
    ``conversation_id`` / ``metadata2`` to the payload must fail at
    construction so the bug surfaces in dev, not in production."""
    with pytest.raises(ValidationError, match="extra"):
        MessageAddedPayload(
            id="m1",
            role="assistant",
            parts=[{"type": "text", "text": "hi", "state": "done"}],
            transient=False,  # type: ignore[call-arg]  # the point of the test
        )


def test_unknown_role_rejected() -> None:
    """Roles outside the canonical set are caught early — pydantic-ai's
    ``UIMessage`` doesn't accept role ``observer`` either, so this
    matches the upstream contract."""
    with pytest.raises(ValidationError):
        MessageAddedPayload(
            id="m1",
            role="observer",  # type: ignore[arg-type]  # the point of the test
            parts=[],
        )


def test_missing_field_rejected() -> None:
    with pytest.raises(ValidationError):
        MessageAddedPayload(role="assistant", parts=[])  # type: ignore[call-arg]

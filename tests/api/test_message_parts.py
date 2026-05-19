"""Tests for the typed `MessagePart` union + `extract_text` helper (F3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pydantic_ai_stateflow.api.streaming import MessagePart, extract_text
from pydantic_ai_stateflow.api.streaming.router import (
    _FilePart,
    _PostMessageBody,
    _TextPart,
    _ToolResultPart,
)


def test_text_part_discriminated() -> None:
    body = _PostMessageBody.model_validate(
        {"role": "user", "parts": [{"type": "text", "text": "hi"}]},
    )
    assert len(body.parts) == 1
    assert isinstance(body.parts[0], _TextPart)
    assert body.parts[0].text == "hi"


def test_tool_result_part_discriminated() -> None:
    body = _PostMessageBody.model_validate(
        {
            "role": "user",
            "parts": [
                {
                    "type": "tool-result",
                    "tool_call_id": "tc_1",
                    "result": {"ok": True},
                },
            ],
        },
    )
    part = body.parts[0]
    assert isinstance(part, _ToolResultPart)
    assert part.tool_call_id == "tc_1"
    assert part.result == {"ok": True}


def test_file_part_discriminated() -> None:
    body = _PostMessageBody.model_validate(
        {
            "role": "user",
            "parts": [
                {
                    "type": "file",
                    "data": "aGVsbG8=",
                    "mime_type": "text/plain",
                },
            ],
        },
    )
    part = body.parts[0]
    assert isinstance(part, _FilePart)
    assert part.mime_type == "text/plain"
    assert part.filename is None


def test_unknown_part_type_rejected() -> None:
    with pytest.raises(ValidationError):
        _PostMessageBody.model_validate(
            {"role": "user", "parts": [{"type": "image", "url": "x"}]},
        )


def test_extract_text_concatenates_in_order() -> None:
    body = _PostMessageBody.model_validate(
        {
            "role": "user",
            "parts": [
                {"type": "text", "text": "Hello, "},
                {"type": "text", "text": "world!"},
            ],
        },
    )
    assert extract_text(body.parts) == "Hello, world!"


def test_extract_text_skips_non_text() -> None:
    body = _PostMessageBody.model_validate(
        {
            "role": "user",
            "parts": [
                {"type": "text", "text": "describe: "},
                {
                    "type": "file",
                    "data": "aGVsbG8=",
                    "mime_type": "image/png",
                },
                {"type": "text", "text": "this image"},
                {
                    "type": "tool-result",
                    "tool_call_id": "tc_1",
                    "result": 42,
                },
            ],
        },
    )
    assert extract_text(body.parts) == "describe: this image"


def test_extract_text_accepts_raw_dicts() -> None:
    """Repo rows are stored as `list[dict]`; helper should still work."""
    raw = [
        {"type": "text", "text": "a"},
        {"type": "tool-result", "tool_call_id": "x", "result": 1},
        {"type": "text", "text": "b"},
    ]
    assert extract_text(raw) == "ab"


def test_message_part_re_exported_from_api_streaming() -> None:
    from pydantic_ai_stateflow.api import streaming as api_streaming

    assert api_streaming.MessagePart is MessagePart


def test_message_part_re_exported_from_api() -> None:
    from pydantic_ai_stateflow import api

    assert api.MessagePart is MessagePart


def test_message_part_re_exported_from_top_level() -> None:
    import pydantic_ai_stateflow as paisf

    assert paisf.MessagePart is MessagePart

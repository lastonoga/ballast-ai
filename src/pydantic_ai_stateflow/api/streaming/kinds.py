"""Canonical streaming event kinds.

The wire values match the AG-UI protocol's `EventType` enum exactly
(SCREAMING_SNAKE_CASE), as defined in the official Python SDK:
https://github.com/ag-ui-protocol/ag-ui/blob/main/sdks/python/ag_ui/core/events.py

Spec home: https://docs.ag-ui.com/concepts/events

Encoder support matrix
----------------------

``AGUIEncoder`` — emits all kinds 1:1 (the protocol it implements).

``VercelEncoder`` — translates a subset to the Vercel AI SDK record format
(https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol). Mapping:

    TEXT_MESSAGE_CONTENT   -> 0:"<delta>"
    TOOL_CALL_START        -> b:{toolCallId, toolName}
    TOOL_CALL_ARGS         -> c:{toolCallId, argsTextDelta}
    TOOL_CALL_END          -> 9:{toolCallId, args}
    RUN_FINISHED           -> d:{finishReason:"stop"}
    RUN_ERROR              -> 3:"<message>"
    RUN_STARTED            -> (dropped — Vercel has no equivalent)
    TEXT_MESSAGE_START/END -> (dropped — Vercel infers from text deltas)
"""

from __future__ import annotations

from enum import StrEnum


class StreamEventKind(StrEnum):
    """AG-UI canonical event kinds. Wire value == ``.value``."""

    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic_ai_stateflow.api.streaming.kinds import StreamEventKind

if TYPE_CHECKING:
    from pydantic_ai_stateflow.api.streaming.router import StreamEvent


class VercelEncoder:
    """Encode canonical AG-UI ``StreamEvent``s as Vercel AI SDK records.

    The Vercel ``ai`` stream protocol (https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol)
    uses ``<tag>:<json>\\n`` lines. We translate the AG-UI canonical kinds
    (see :class:`StreamEventKind`) to the Vercel record set:

    =====================  ====================================================
    AG-UI kind             Vercel record
    =====================  ====================================================
    TEXT_MESSAGE_CONTENT   ``0:"<delta>"``
    TOOL_CALL_START        ``b:{"toolCallId":...,"toolName":...}``
    TOOL_CALL_ARGS         ``c:{"toolCallId":...,"argsTextDelta":"<delta>"}``
    TOOL_CALL_END          ``9:{"toolCallId":...,"args":{}}``
    RUN_FINISHED           ``d:{"finishReason":"stop"}``
    RUN_ERROR              ``3:"<message>"``
    RUN_STARTED            (dropped — no Vercel analog)
    TEXT_MESSAGE_START     (dropped — Vercel infers from text deltas)
    TEXT_MESSAGE_END       (dropped — Vercel infers from text deltas)
    =====================  ====================================================

    Dropped events return ``b""``; the router skips empty frames.
    Unknown kinds also drop (return ``b""``) for forward-compatibility.
    """

    media_type = "text/plain; charset=utf-8"

    def encode(self, event: StreamEvent) -> bytes:
        kind = event.kind
        data = event.data

        if kind == StreamEventKind.TEXT_MESSAGE_CONTENT.value:
            return f"0:{json.dumps(data.get('delta', ''))}\n".encode()

        if kind == StreamEventKind.TOOL_CALL_START.value:
            body = {
                "toolCallId": data.get("toolCallId", ""),
                "toolName": data.get("toolCallName", ""),
            }
            return f"b:{json.dumps(body, separators=(',', ':'))}\n".encode()

        if kind == StreamEventKind.TOOL_CALL_ARGS.value:
            body = {
                "toolCallId": data.get("toolCallId", ""),
                "argsTextDelta": data.get("delta", ""),
            }
            return f"c:{json.dumps(body, separators=(',', ':'))}\n".encode()

        if kind == StreamEventKind.TOOL_CALL_END.value:
            body = {"toolCallId": data.get("toolCallId", ""), "args": {}}
            return f"9:{json.dumps(body, separators=(',', ':'))}\n".encode()

        if kind == StreamEventKind.RUN_FINISHED.value:
            return b'd:{"finishReason":"stop"}\n'

        if kind == StreamEventKind.RUN_ERROR.value:
            return f"3:{json.dumps(data.get('message', ''))}\n".encode()

        return b""

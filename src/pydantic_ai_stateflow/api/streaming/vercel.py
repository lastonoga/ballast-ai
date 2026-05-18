from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai_stateflow.api.streaming.router import StreamEvent

_PREFIX = {
    "text_delta": "0",
    "tool_call": "9",
    "tool_call_delta": "c",
    "tool_result": "a",
    "error": "3",
    "done": "d",
}


class VercelEncoder:
    """Encode `StreamEvent`s in the Vercel AI SDK record format.

    One JSON-encoded record per line, prefixed with a single-char tag per
    the @ai-sdk/ui stream protocol. Unknown kinds fall through to the
    generic data prefix (`2`).
    """

    media_type = "text/plain; charset=utf-8"

    def encode(self, event: StreamEvent) -> bytes:
        prefix = _PREFIX.get(event.kind, "2")
        if event.kind == "text_delta":
            value = event.data.get("text", "")
            return f"{prefix}:{json.dumps(value)}\n".encode()
        body = json.dumps(event.data, separators=(",", ":"))
        return f"{prefix}:{body}\n".encode()

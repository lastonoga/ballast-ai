from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai_stateflow.api.streaming.router import StreamEvent


class AGUIEncoder:
    """Encode framework `StreamEvent`s as AG-UI SSE frames.

    Spec 1.13: AG-UI = UI streaming protocol. Encoder is a CONTENT FORMATTER
    — does not own the transport. Frames look like:

        event: text_delta
        data: {"text": "hi"}
        <blank line>

    JSON-encoding the data payload guarantees no raw newlines inside `data:`
    lines (SSE requires single-line data values).
    """

    media_type = "text/event-stream"

    def encode(self, event: StreamEvent) -> bytes:
        payload = json.dumps(event.data, separators=(",", ":"))
        return f"event: {event.kind}\ndata: {payload}\n\n".encode()

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic_ai_stateflow.api.streaming.kinds import StreamEventKind

if TYPE_CHECKING:
    from pydantic_ai_stateflow.api.streaming.router import StreamEvent


class AGUIEncoder:
    """Encode framework ``StreamEvent``s as AG-UI SSE frames.

    Wire format follows the AG-UI protocol — event type tokens are
    SCREAMING_SNAKE_CASE and payload field names are camelCase, matching
    the canonical Python SDK at
    https://github.com/ag-ui-protocol/ag-ui/blob/main/sdks/python/ag_ui/core/events.py
    (spec: https://docs.ag-ui.com/concepts/events).

    Each event becomes::

        event: TEXT_MESSAGE_CONTENT
        data: {"messageId":"...","delta":"hi"}
        <blank line>

    JSON-encoding the data payload guarantees no raw newlines inside
    ``data:`` lines (SSE requires single-line data values). Unknown kinds
    raise ``ValueError`` — callers must use :class:`StreamEventKind`.
    """

    media_type = "text/event-stream"

    def encode(self, event: StreamEvent) -> bytes:
        if event.kind not in StreamEventKind._value2member_map_:
            raise ValueError(f"unknown StreamEvent kind: {event.kind!r}")
        payload = json.dumps(event.data, separators=(",", ":"))
        return f"event: {event.kind}\ndata: {payload}\n\n".encode()

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

    Each event becomes a single ``data:`` line carrying a JSON object
    with the event ``type`` discriminator inlined::

        data: {"type":"TEXT_MESSAGE_CONTENT","messageId":"...","delta":"hi"}
        <blank line>

    Importantly the ``type`` is inside the JSON, NOT in a separate
    ``event:`` SSE header. ``@ag-ui/client``'s ``parseSSEStream`` ignores
    ``event:`` lines entirely — it only joins ``data:`` lines, parses as
    JSON, and validates against a Zod discriminated union keyed on
    ``type``. Emitting ``event: <kind>`` produces ZodError on the client
    (`invalid_union_discriminator: path=["type"]`).

    JSON-encoding guarantees no raw newlines inside the ``data:`` line
    (SSE requires single-line data values). Unknown kinds raise
    ``ValueError`` — callers must use :class:`StreamEventKind`.
    """

    media_type = "text/event-stream"

    def encode(self, event: StreamEvent) -> bytes:
        if event.kind not in StreamEventKind._value2member_map_:
            raise ValueError(f"unknown StreamEvent kind: {event.kind!r}")
        payload = {"type": event.kind, **event.data}
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()

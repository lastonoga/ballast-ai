from __future__ import annotations

import re
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import CapabilityOrdering
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext

from pydantic_ai_stateflow.capabilities.base import StateflowCapability


class PIIGuard(StateflowCapability):
    """Innermost capability: regex-redacts PII from text responses.

    Applied AFTER all other after_model_request hooks (innermost in the
    wrap chain), so other capabilities see the raw text and downstream
    output validation / persistence sees the redacted form.

    For richer detection (NER), subclass and override ``redact(text)`` —
    the regex layer is just a sensible default.
    """

    name = "pii_guard"

    def __init__(
        self,
        *,
        patterns: list[re.Pattern[str]],
        replacement: str = "[REDACTED]",
    ) -> None:
        self.patterns = patterns
        self.replacement = replacement

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    def redact(self, text: str) -> str:
        for pat in self.patterns:
            text = pat.sub(self.replacement, text)
        return text

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        for part in response.parts:
            if isinstance(part, TextPart):
                part.content = self.redact(part.content)
        return response

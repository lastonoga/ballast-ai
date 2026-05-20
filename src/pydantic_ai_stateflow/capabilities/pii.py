"""PIIGuard — pluggable PII redaction for model responses.

The capability is structured around two strategy objects:

- ``PIIDetector`` — returns ``list[PIISpan]`` marking offsets in the
  text that should be scrubbed. Default impl ``RegexDetector`` is a
  pattern-match list; apps wire in their own detector (DB-grounded,
  LLM-based, NER, …) by satisfying the Protocol.

- ``redactor`` — a callable ``(text, spans) -> str`` that produces the
  scrubbed output. Default impl ``constant_redactor("[REDACTED]")``
  replaces every span with a single placeholder; category-aware
  variants are a few lines.

Splitting detection from redaction lets apps add semantic / DB-grounded
checks (e.g. "is this email actually one of OUR users — i.e. a
cross-account leak — or made-up data?") without rewriting the whole
guard. The detector gets the run's ``RunContext`` so it can reach
``ctx.deps`` for repos, current actor, etc.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic_ai import RunContext
from pydantic_ai.capabilities import CapabilityOrdering
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext

from pydantic_ai_stateflow.capabilities.base import StateflowCapability


@dataclass(frozen=True)
class PIISpan:
    """A region of text that should be redacted.

    ``category`` is a free-form tag a redactor MAY use to pick a
    placeholder ("email" → "[EMAIL]" vs the default "[REDACTED]").
    ``detail`` carries detector-side context (which user/entity matched,
    confidence score, etc.) for downstream logging / audit.
    """

    start: int
    end: int
    category: str = "pii"
    detail: str = ""


@runtime_checkable
class PIIDetector(Protocol):
    """Strategy: scan ``text`` and return spans to redact.

    Async because real-world detectors call into a DB / repo / LLM.
    ``ctx`` exposes the run's deps so detectors can authorize, look
    up the current actor, or query an embedding index.
    """

    async def detect(
        self, text: str, *, ctx: RunContext[Any],
    ) -> list[PIISpan]: ...


Redactor = Callable[[str, list[PIISpan]], str]
"""``(text, spans) -> redacted_text``. Spans MUST be applied
right-to-left so earlier offsets stay valid."""


def constant_redactor(replacement: str = "[REDACTED]") -> Redactor:
    """Replace every span with the same placeholder, regardless of category."""

    def _redact(text: str, spans: list[PIISpan]) -> str:
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            text = text[: span.start] + replacement + text[span.end :]
        return text

    return _redact


def categorized_redactor(
    *,
    placeholders: dict[str, str],
    fallback: str = "[REDACTED]",
) -> Redactor:
    """Replace each span with the placeholder mapped from its ``category``.

    Unknown categories fall back to ``fallback``.
    """

    def _redact(text: str, spans: list[PIISpan]) -> str:
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            placeholder = placeholders.get(span.category, fallback)
            text = text[: span.start] + placeholder + text[span.end :]
        return text

    return _redact


class RegexDetector:
    """Default ``PIIDetector``: list of compiled regex patterns.

    Stateless across runs — the same instance is safe to share. Apps
    that want pattern provenance (which regex matched) can pass a list
    of ``(category, pattern)`` tuples via ``patterns_by_category``.
    """

    def __init__(
        self,
        *,
        patterns: list[re.Pattern[str]] | None = None,
        patterns_by_category: dict[str, list[re.Pattern[str]]] | None = None,
    ) -> None:
        if patterns is None and patterns_by_category is None:
            raise ValueError(
                "RegexDetector: pass either `patterns` or `patterns_by_category`",
            )
        self._flat: list[tuple[str, re.Pattern[str]]] = []
        for pat in patterns or []:
            self._flat.append(("pii", pat))
        for cat, pats in (patterns_by_category or {}).items():
            for pat in pats:
                self._flat.append((cat, pat))

    async def detect(
        self, text: str, *, ctx: RunContext[Any],
    ) -> list[PIISpan]:
        del ctx  # regex is context-free
        spans: list[PIISpan] = []
        for category, pat in self._flat:
            for m in pat.finditer(text):
                spans.append(
                    PIISpan(start=m.start(), end=m.end(), category=category),
                )
        return spans


class PIIGuard(StateflowCapability):
    """Innermost capability: redact PII from model text responses.

    Applied AFTER every other ``after_model_request`` hook, so peer
    capabilities see the raw text and downstream output validation /
    persistence / streaming sees the redacted form.

    Pass a ``detector`` (default ``RegexDetector`` requires patterns)
    and an optional ``redactor`` (default ``constant_redactor()``).
    Apps that need DB-grounded or LLM-based detection wire in a custom
    ``PIIDetector`` impl.
    """

    name = "pii_guard"

    def __init__(
        self,
        *,
        detector: PIIDetector,
        redactor: Redactor | None = None,
    ) -> None:
        self.detector = detector
        self.redactor: Redactor = redactor or constant_redactor()

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        for part in response.parts:
            if isinstance(part, TextPart):
                spans = await self.detector.detect(part.content, ctx=ctx)
                if spans:
                    part.content = self.redactor(part.content, spans)
        return response

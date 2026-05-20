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

Streaming
=========

The non-streaming :meth:`PIIGuard.after_model_request` runs AFTER the
model finishes the whole response — too late to scrub a live SSE
stream. :meth:`PIIGuard.wrap_run_event_stream` re-implements the same
detector/redactor pipeline for streaming, with a per-part lookbehind
buffer so partial matches (e.g. an email split across two delta chunks)
get redacted at flush time rather than leaking the head of the match.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from pydantic_ai import RunContext
from pydantic_ai.capabilities import CapabilityOrdering
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.models import ModelRequestContext

from pydantic_ai_stateflow.capabilities.base import StateflowCapability
from pydantic_ai_stateflow.observability.spans import _get_logfire_span
from pydantic_ai_stateflow.observability.trace_names import TraceName


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


# Lookbehind window for streaming redaction. Must exceed the longest
# realistic PII span (emails ~64 chars per local part, international
# phone numbers ~15 digits with separators). 128 chars gives plenty of
# headroom for both without delaying output noticeably for the UI.
_STREAM_LOOKBEHIND = 128


@dataclass
class _StreamPartState:
    """Per-part-index streaming state.

    Tracks the accumulated raw text we've seen for one ``TextPart`` and
    how many chars of the redacted version we've already emitted, so
    each ``PartDeltaEvent`` we yield carries only the new redacted
    suffix.
    """

    raw: str = ""
    """Full raw text accumulated from upstream deltas so far."""

    emitted: str = ""
    """Redacted text we've already pushed downstream for this part."""

    detected_spans: int = 0
    """Span count from the most recent flush — for telemetry only."""

    categories: set[str] = None  # type: ignore[assignment]
    """Categories we've matched in this part so far — for telemetry only."""

    def __post_init__(self) -> None:
        if self.categories is None:
            self.categories = set()


class PIIGuard(StateflowCapability):
    """Innermost capability: redact PII from model text responses.

    Applied AFTER every other ``after_model_request`` hook, so peer
    capabilities see the raw text and downstream output validation /
    persistence / streaming sees the redacted form.

    Pass a ``detector`` (default ``RegexDetector`` requires patterns)
    and an optional ``redactor`` (default ``constant_redactor()``).
    Apps that need DB-grounded or LLM-based detection wire in a custom
    ``PIIDetector`` impl.

    Streaming
    ---------

    :meth:`wrap_run_event_stream` mirrors :meth:`after_model_request`
    for the SSE event stream the API layer hands to the client.
    Buffered lookbehind redaction prevents leaking the head of a PII
    span that gets split across delta chunks (the classic
    ``"alice"`` + ``"@example.com"`` failure mode that triggered this
    capability's streaming branch).
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
        detected_spans = 0
        redacted_chars = 0
        categories: set[str] = set()

        span_fn = _get_logfire_span()
        span_ctx = (
            span_fn(TraceName.CAPABILITY_PII_GUARD.value)
            if span_fn is not None
            else None
        )
        if span_ctx is not None:
            span_ctx.__enter__()
        try:
            for part in response.parts:
                if isinstance(part, TextPart):
                    spans = await self.detector.detect(part.content, ctx=ctx)
                    if spans:
                        original_len = len(part.content)
                        part.content = self.redactor(part.content, spans)
                        redacted_chars += original_len - len(part.content)
                        detected_spans += len(spans)
                        for s in spans:
                            categories.add(s.category)
            return response
        finally:
            if span_ctx is not None:
                # Attach final attrs before closing the span so logfire
                # users can filter on "did anything actually match?".
                try:  # pragma: no branch
                    setter = getattr(span_ctx, "set_attributes", None)
                    if callable(setter):
                        setter({
                            "detected_spans": detected_spans,
                            "redacted_chars": redacted_chars,
                            "categories": ",".join(sorted(categories)),
                        })
                except Exception:  # pragma: no cover  (defensive)
                    pass
                span_ctx.__exit__(None, None, None)

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[Any],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        """Stream-mode redaction with per-part lookbehind buffering.

        Two reasons this exists instead of letting
        :meth:`after_model_request` handle it:

        1. The model's response is delivered to the client chunk-by-chunk
           via SSE; ``after_model_request`` fires AFTER the whole stream
           has been emitted, so the raw text is already in the user's
           browser by the time we'd otherwise scrub.
        2. A regex like ``\\S+@\\S+`` matches the FULL email only after
           the ``@`` arrives. We buffer the last ``_STREAM_LOOKBEHIND``
           characters of raw text per part, run the detector on the
           "settled" prefix (raw minus tail), and emit just the NEW
           redacted suffix as a delta. The tail flushes on
           ``PartEndEvent``.
        """
        states: dict[int, _StreamPartState] = {}

        async def _flush(
            state: _StreamPartState,
            *,
            final: bool,
        ) -> str:
            """Run detector+redactor on the settled prefix; return new emit suffix."""
            if final:
                settled = state.raw
            else:
                # Keep the last LOOKBEHIND chars buffered so partial
                # matches at the tail get a chance to complete.
                if len(state.raw) <= _STREAM_LOOKBEHIND:
                    return ""
                settled = state.raw[: -_STREAM_LOOKBEHIND]
            spans = await self.detector.detect(settled, ctx=ctx)
            redacted = self.redactor(settled, spans) if spans else settled
            state.detected_spans = len(spans)
            for s in spans:
                state.categories.add(s.category)
            if not redacted.startswith(state.emitted):
                # Redaction shifted earlier emitted bytes (rare —
                # shouldn't happen given we only redact INSIDE the
                # settled prefix, but be defensive). Emit nothing
                # further; the final part_end carries the canonical
                # full-redacted text and the renderer will resync from
                # there if it cares to.
                return ""
            new_suffix = redacted[len(state.emitted) :]
            state.emitted = redacted
            return new_suffix

        span_fn = _get_logfire_span()
        span_ctx = (
            span_fn(TraceName.CAPABILITY_PII_GUARD_STREAM.value)
            if span_fn is not None
            else None
        )
        if span_ctx is not None:
            span_ctx.__enter__()
        try:
            async for event in stream:
                if isinstance(event, PartStartEvent) and isinstance(
                    event.part, TextPart,
                ):
                    initial = event.part.content
                    state = _StreamPartState(raw=initial)
                    states[event.index] = state
                    # Swallow any initial content so the renderer doesn't
                    # show raw upfront. We'll either flush it via deltas
                    # (settled-prefix redaction) or via the PartEnd event.
                    yield replace(
                        event,
                        part=replace(event.part, content=""),
                    )
                    # If the initial content already exceeds our lookbehind,
                    # flush whatever's now in the settled prefix.
                    new_suffix = await _flush(state, final=False)
                    if new_suffix:
                        yield PartDeltaEvent(
                            index=event.index,
                            delta=TextPartDelta(content_delta=new_suffix),
                        )
                    continue

                if isinstance(event, PartDeltaEvent) and isinstance(
                    event.delta, TextPartDelta,
                ):
                    state = states.get(event.index)
                    if state is None:
                        # No matching PartStart for a TextPart — pass
                        # through and hope downstream tolerates it.
                        yield event
                        continue
                    state.raw += event.delta.content_delta
                    new_suffix = await _flush(state, final=False)
                    if new_suffix:
                        yield replace(
                            event,
                            delta=replace(
                                event.delta, content_delta=new_suffix,
                            ),
                        )
                    # If nothing new to emit yet (buffered), drop the
                    # event — the bytes are still in state.raw and will
                    # flush on the next delta or part-end.
                    continue

                if isinstance(event, PartEndEvent) and isinstance(
                    event.part, TextPart,
                ):
                    state = states.get(event.index)
                    if state is None:
                        yield event
                        continue
                    # Trust event.part.content over our accumulated raw
                    # — upstream may have applied final corrections.
                    state.raw = event.part.content
                    new_suffix = await _flush(state, final=True)
                    if new_suffix:
                        yield PartDeltaEvent(
                            index=event.index,
                            delta=TextPartDelta(content_delta=new_suffix),
                        )
                    yield replace(
                        event,
                        part=replace(event.part, content=state.emitted),
                    )
                    continue

                # Non-text events pass through unchanged.
                yield event
        finally:
            if span_ctx is not None:
                try:  # pragma: no branch
                    total_detected = sum(s.detected_spans for s in states.values())
                    all_cats: set[str] = set()
                    for s in states.values():
                        all_cats |= s.categories
                    setter = getattr(span_ctx, "set_attributes", None)
                    if callable(setter):
                        setter({
                            "parts_processed": len(states),
                            "total_detected_spans": total_detected,
                            "categories": ",".join(sorted(all_cats)),
                        })
                except Exception:  # pragma: no cover  (defensive)
                    pass
                span_ctx.__exit__(None, None, None)

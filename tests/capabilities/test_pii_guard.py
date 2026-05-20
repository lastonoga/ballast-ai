"""``PIIGuard`` — detector + redactor strategy, custom detector integration."""

import re
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_stateflow.capabilities import (
    PIIGuard,
    PIISpan,
    RegexDetector,
    categorized_redactor,
    constant_redactor,
)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d{10,15}")


def make_fn_model_returning(text: str) -> FunctionModel:
    """FunctionModel returning ``text`` for both ``run`` and streaming.

    Streaming mode is auto-enabled by pydantic-ai when ``PIIGuard``
    overrides ``wrap_run_event_stream``, so even non-streaming tests
    must provide a ``stream_function``.
    """
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    async def stream_fn(
        messages: list[ModelMessage], info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield text

    return FunctionModel(fn, stream_function=stream_fn)


@pytest.mark.asyncio
async def test_pii_guard_redacts_email_via_regex_detector() -> None:
    agent = Agent(
        model=make_fn_model_returning("Contact me at alice@example.com soon."),
        capabilities=[
            PIIGuard(detector=RegexDetector(patterns=[EMAIL_RE])),
        ],
    )
    result = await agent.run("ignored")
    text = str(result.output)
    assert "alice@example.com" not in text
    assert "[REDACTED]" in text


@pytest.mark.asyncio
async def test_pii_guard_redacts_phone_with_constant_redactor_custom_placeholder() -> None:
    agent = Agent(
        model=make_fn_model_returning("Call +1234567890 now."),
        capabilities=[
            PIIGuard(
                detector=RegexDetector(patterns=[PHONE_RE]),
                redactor=constant_redactor("[PHONE]"),
            ),
        ],
    )
    result = await agent.run("ignored")
    text = str(result.output)
    assert "+1234567890" not in text
    assert "[PHONE]" in text


@pytest.mark.asyncio
async def test_pii_guard_passes_through_clean_text() -> None:
    agent = Agent(
        model=make_fn_model_returning("Nothing to see here."),
        capabilities=[
            PIIGuard(detector=RegexDetector(patterns=[EMAIL_RE])),
        ],
    )
    result = await agent.run("ignored")
    assert "Nothing to see here" in str(result.output)


@pytest.mark.asyncio
async def test_categorized_redactor_uses_per_category_placeholders() -> None:
    """``RegexDetector(patterns_by_category=...)`` tags spans; the
    ``categorized_redactor`` swaps in per-category placeholders."""
    agent = Agent(
        model=make_fn_model_returning(
            "Email alice@example.com or call +1234567890.",
        ),
        capabilities=[
            PIIGuard(
                detector=RegexDetector(
                    patterns_by_category={
                        "email": [EMAIL_RE],
                        "phone": [PHONE_RE],
                    },
                ),
                redactor=categorized_redactor(
                    placeholders={"email": "[EMAIL]", "phone": "[PHONE]"},
                ),
            ),
        ],
    )
    result = await agent.run("ignored")
    text = str(result.output)
    assert "alice@example.com" not in text
    assert "+1234567890" not in text
    assert "[EMAIL]" in text
    assert "[PHONE]" in text


@pytest.mark.asyncio
async def test_custom_detector_can_be_async_and_consult_ctx() -> None:
    """Apps can supply a ``PIIDetector`` that hits a DB / agent. We mock
    one here that uses ``ctx.deps`` to decide which substrings are leaks.
    """

    class _OwnedEmailDetector:
        """Marks any email that's in ``ctx.deps['known_emails']`` as leaks."""

        async def detect(self, text, *, ctx):
            spans: list[PIISpan] = []
            known = set(ctx.deps["known_emails"])
            for m in EMAIL_RE.finditer(text):
                if m.group() in known:
                    spans.append(
                        PIISpan(
                            start=m.start(),
                            end=m.end(),
                            category="cross-account",
                            detail=m.group(),
                        ),
                    )
            return spans

    agent: Agent[dict[str, list[str]], str] = Agent(
        model=make_fn_model_returning(
            "Ping bob@otherorg.com and ignore mailinglist@example.com.",
        ),
        deps_type=dict,  # type: ignore[arg-type]
        capabilities=[PIIGuard(detector=_OwnedEmailDetector())],
    )
    result = await agent.run(
        "ignored",
        deps={"known_emails": ["bob@otherorg.com"]},
    )
    text = str(result.output)
    # Owned email got redacted.
    assert "bob@otherorg.com" not in text
    # Unknown email passed through.
    assert "mailinglist@example.com" in text


# ── wrap_run_event_stream tests ───────────────────────────────────────────────
#
# These exercise the streaming path DIRECTLY (synthetic event stream → guard →
# collected output) without booting a full Agent run. The non-streaming path
# above already covers the agent-integration shape.


def _make_run_context() -> RunContext[Any]:
    """Build a minimal ``RunContext`` for direct hook invocation.

    ``RunContext`` is kw-only with three required fields (``deps``,
    ``model``, ``usage``); everything else has defaults. ``RegexDetector``
    doesn't touch ``ctx`` (custom detectors might, but our streaming
    tests use regex), so the stub model + zero-usage suffice.
    """
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.usage import RunUsage

    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


async def _collect(stream) -> list[Any]:
    """Drain an async iterable into a list."""
    out: list[Any] = []
    async for ev in stream:
        out.append(ev)
    return out


async def _gen(events: list[Any]):
    """Wrap a static list as an async generator."""
    for ev in events:
        yield ev


@pytest.mark.asyncio
async def test_wrap_run_event_stream_redacts_email_split_across_deltas() -> None:
    """The user-facing bug: ``alice@example.com`` arrives across TWO
    delta chunks. Without lookbehind buffering the prefix would leak to
    the SSE consumer before the ``@`` arrived and made the regex match.
    """
    guard = PIIGuard(detector=RegexDetector(patterns=[EMAIL_RE]))
    events = [
        PartStartEvent(index=0, part=TextPart(content="")),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="Hi alice")),
        PartDeltaEvent(
            index=0, delta=TextPartDelta(content_delta="@example.com here."),
        ),
        PartEndEvent(
            index=0, part=TextPart(content="Hi alice@example.com here."),
        ),
    ]
    out = await _collect(
        guard.wrap_run_event_stream(_make_run_context(), stream=_gen(events)),
    )

    # Concatenate everything text-shaped that left the guard.
    pieces: list[str] = []
    for ev in out:
        if isinstance(ev, PartStartEvent) and isinstance(ev.part, TextPart):
            pieces.append(ev.part.content)
        elif isinstance(ev, PartDeltaEvent) and isinstance(
            ev.delta, TextPartDelta,
        ):
            pieces.append(ev.delta.content_delta)
        elif isinstance(ev, PartEndEvent) and isinstance(ev.part, TextPart):
            # PartEnd carries the canonical full-redacted text; don't
            # double-count it after the final delta. The assertion below
            # checks it independently.
            pass
    streamed_text = "".join(pieces)
    assert "alice@example.com" not in streamed_text, streamed_text
    assert "[REDACTED]" in streamed_text, streamed_text

    # PartEndEvent also carries the full redacted form (renderer resync point).
    end_events = [
        ev for ev in out
        if isinstance(ev, PartEndEvent) and isinstance(ev.part, TextPart)
    ]
    assert len(end_events) == 1
    assert "alice@example.com" not in end_events[0].part.content
    assert "[REDACTED]" in end_events[0].part.content


@pytest.mark.asyncio
async def test_wrap_run_event_stream_isolates_state_across_parts() -> None:
    """Multiple text parts (e.g. text → tool call → text) must each get
    their own buffer state — leak between parts would either double-emit
    or swallow content."""
    guard = PIIGuard(detector=RegexDetector(patterns=[EMAIL_RE]))
    events = [
        PartStartEvent(index=0, part=TextPart(content="")),
        PartDeltaEvent(
            index=0, delta=TextPartDelta(content_delta="Ping a@b.co please."),
        ),
        PartEndEvent(
            index=0, part=TextPart(content="Ping a@b.co please."),
        ),
        # Different part index — fresh buffer, no carry-over of "Ping".
        PartStartEvent(
            index=1, part=TextPart(content=""), previous_part_kind="text",
        ),
        PartDeltaEvent(
            index=1, delta=TextPartDelta(content_delta="Also c@d.co thanks."),
        ),
        PartEndEvent(
            index=1, part=TextPart(content="Also c@d.co thanks."),
        ),
    ]
    out = await _collect(
        guard.wrap_run_event_stream(_make_run_context(), stream=_gen(events)),
    )
    ends = [
        ev for ev in out
        if isinstance(ev, PartEndEvent) and isinstance(ev.part, TextPart)
    ]
    assert len(ends) == 2
    assert ends[0].part.content == "Ping [REDACTED] please."
    assert ends[1].part.content == "Also [REDACTED] thanks."


@pytest.mark.asyncio
async def test_wrap_run_event_stream_passes_non_text_events() -> None:
    """ToolCall delta events have nothing to redact and must pass through."""
    guard = PIIGuard(detector=RegexDetector(patterns=[EMAIL_RE]))
    tool_call_event = PartStartEvent(
        index=0,
        part=ToolCallPart(tool_name="create_note", args={"title": "x"}),
    )
    tool_delta_event = PartDeltaEvent(
        index=0,
        delta=ToolCallPartDelta(args_delta={"body": "y"}),
    )
    events = [tool_call_event, tool_delta_event]
    out = await _collect(
        guard.wrap_run_event_stream(_make_run_context(), stream=_gen(events)),
    )
    # Both events round-trip identically (object identity not required —
    # equality of relevant fields is what matters).
    assert len(out) == 2
    assert isinstance(out[0], PartStartEvent)
    assert isinstance(out[0].part, ToolCallPart)
    assert out[0].part.tool_name == "create_note"
    assert isinstance(out[1], PartDeltaEvent)
    assert isinstance(out[1].delta, ToolCallPartDelta)
    assert out[1].delta.args_delta == {"body": "y"}

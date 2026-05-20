"""``PIIGuard`` — detector + redactor strategy, custom detector integration."""

import re

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
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
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])
    return FunctionModel(fn)


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

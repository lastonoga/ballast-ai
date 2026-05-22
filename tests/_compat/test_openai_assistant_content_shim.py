"""Verify the OpenAI assistant ``content: null`` → ``""`` shim.

The shim normalizes pydantic-ai's OpenAI-spec-compliant
``{role: 'assistant', content: null, tool_calls: [...]}`` shape to
``{..., content: '', tool_calls: [...]}`` so strict upstreams (notably
Alibaba's Qwen endpoint) accept it.

This isn't testing pydantic-ai's actual chat completion call — it's
testing that our import-time patch correctly rewires the nested
``_MapModelResponseContext._into_message_param`` hook.
"""

from __future__ import annotations

import pytest

# Import the framework so the shim is installed.
import ballast  # noqa: F401


def test_shim_normalizes_null_content_when_tool_calls_present() -> None:
    from pydantic_ai.models.openai import OpenAIChatModel

    mapper_cls = OpenAIChatModel._MapModelResponseContext  # noqa: SLF001

    class _StubMapper:
        """Minimum shape ``_into_message_param`` reads off ``self``."""

        texts: list[str] = []
        thinkings: dict[str, list[str]] = {}
        tool_calls: list[dict] = [
            {"id": "call_1", "type": "function",
             "function": {"name": "delete_note", "arguments": "{}"}},
        ]

    result = mapper_cls._into_message_param(_StubMapper())  # noqa: SLF001
    assert result is not None
    assert result["content"] == "", result
    assert result["tool_calls"] == _StubMapper.tool_calls


def test_shim_normalizes_null_content_when_only_thinking_present() -> None:
    """Reasoning-only assistant turns (ThinkingPart but no text/tool_calls)
    also surface as ``content:None`` from pydantic-ai. Alibaba rejects
    that just like the tool-call-only case — shim must normalize to "".
    """
    from pydantic_ai.models.openai import OpenAIChatModel

    mapper_cls = OpenAIChatModel._MapModelResponseContext  # noqa: SLF001

    class _StubMapper:
        texts: list[str] = []
        thinkings: dict[str, list[str]] = {"reasoning": ["I should…"]}
        tool_calls: list[dict] = []

    result = mapper_cls._into_message_param(_StubMapper())  # noqa: SLF001
    assert result is not None
    assert result["content"] == "", result
    assert "tool_calls" not in result
    # Reasoning text preserved on whatever field name the mapper used.
    assert result.get("reasoning") == "I should…"


def test_shim_leaves_text_only_messages_alone() -> None:
    from pydantic_ai.models.openai import OpenAIChatModel

    mapper_cls = OpenAIChatModel._MapModelResponseContext  # noqa: SLF001

    class _StubMapper:
        texts: list[str] = ["hello"]
        thinkings: dict[str, list[str]] = {}
        tool_calls: list[dict] = []

    result = mapper_cls._into_message_param(_StubMapper())  # noqa: SLF001
    assert result is not None
    assert result["content"] == "hello"
    assert "tool_calls" not in result


def test_shim_leaves_empty_responses_dropped() -> None:
    """Empty response (no text, no thinking, no tool calls) still drops."""
    from pydantic_ai.models.openai import OpenAIChatModel

    mapper_cls = OpenAIChatModel._MapModelResponseContext  # noqa: SLF001

    class _StubMapper:
        texts: list[str] = []
        thinkings: dict[str, list[str]] = {}
        tool_calls: list[dict] = []

    assert mapper_cls._into_message_param(_StubMapper()) is None  # noqa: SLF001


def test_shim_is_idempotent() -> None:
    """Calling install_assistant_content_shim twice is safe."""
    from ballast._compat import install_assistant_content_shim

    install_assistant_content_shim()
    install_assistant_content_shim()


@pytest.mark.skipif(
    "OpenAIChatModel" not in dir(__import__(
        "pydantic_ai.models.openai", fromlist=["OpenAIChatModel"],
    )),
    reason="pydantic-ai installed without openai extras",
)
def test_shim_attribute_attached() -> None:
    """The mapper class actually carries the patched method."""
    from pydantic_ai.models.openai import OpenAIChatModel

    mapper_cls = OpenAIChatModel._MapModelResponseContext  # noqa: SLF001
    # functools.wraps preserves __wrapped__ on the patched callable —
    # we use it as a marker that the shim is in place.
    method = mapper_cls._into_message_param  # noqa: SLF001
    assert hasattr(method, "__wrapped__"), (
        "patched _into_message_param should expose __wrapped__ via wraps"
    )

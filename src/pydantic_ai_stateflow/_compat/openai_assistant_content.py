"""Normalize OpenAI assistant ``content: null`` → ``""`` for tool-call turns.

## Why

The OpenAI Chat Completions spec allows
``{role: "assistant", content: null, tool_calls: [...]}`` — when the
assistant's turn IS the tool call and carries no surrounding text.
pydantic-ai's ``OpenAIChatModel._MapModelResponseContext.\
_into_message_param`` faithfully emits ``content=None`` in that case
(see ``models/openai.py:1311``).

OpenAI, Anthropic-via-OpenRouter, DeepInfra, and most upstreams accept
this. Alibaba's Qwen endpoint validates strictly and rejects the
request with::

    <400> InternalError.Algo.InvalidParameter:
        The content field is a required field.

surfacing as a generic provider 400 from OpenRouter. The same shape
trips a handful of other strict endpoints.

## What the shim does

Wraps ``_into_message_param`` to replace ``content=None`` with
``content=""`` **only** when ``tool_calls`` is also set. Tool-call-
only messages keep their semantic meaning (assistant making a tool
call with no surrounding text); strict-mode upstreams accept ``""``.
All other cases (text content, no tool calls, empty response) pass
through unchanged.

## Why a monkey-patch and not a subclass

``_MapModelResponseContext`` is a nested ``@dataclass`` on
``OpenAIChatModel``, instantiated implicitly by the model when building
requests. Subclassing would require apps to swap their ``Model`` class —
defeats the "framework solves it for everyone" goal. A targeted patch
is the minimum surface needed.

## When to remove

When pydantic-ai upstream either:

1. Adds a model-profile flag like
   ``openai_compat_allow_null_assistant_content: bool`` and routes
   through it, or
2. Defaults to ``content=""`` for tool-call-only assistant turns
   (matches Alibaba/strict endpoints, still valid OpenAI spec).

Either way, delete this file and the import in ``_compat/__init__.py``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


_INSTALLED = False


def install_assistant_content_shim() -> None:
    """Patch OpenAIChatModel's assistant-message mapper. Idempotent.

    Imports pydantic-ai lazily so this module stays import-time-safe
    even if ``pydantic_ai.models.openai`` isn't available (e.g.,
    tests of unrelated subsystems run in slim envs).
    """
    global _INSTALLED  # noqa: PLW0603
    if _INSTALLED:
        return

    try:
        from pydantic_ai.models.openai import OpenAIChatModel  # noqa: PLC0415
    except ImportError:
        # pydantic-ai installed slim without openai extras — nothing to patch.
        _INSTALLED = True
        return

    mapper_cls = OpenAIChatModel._MapModelResponseContext  # noqa: SLF001
    original: Callable[..., Any] = mapper_cls._into_message_param  # noqa: SLF001

    from pydantic_ai_stateflow.logging import get_logger  # noqa: PLC0415

    log = get_logger("pydantic_ai_stateflow._compat.openai_assistant_content")
    log.debug(
        "install_assistant_content_shim() running — mapper_cls=%r original=%r",
        mapper_cls, original,
    )

    @functools.wraps(original)
    def patched(self: Any) -> Any:
        result = original(self)
        # Normalize ``content: None`` → ``""`` on any assistant turn the
        # mapper actually emits. pydantic-ai drops truly-empty responses
        # by returning ``None`` upstream (``openai.py:_into_message_param``
        # check ``if not self.texts and not self.thinkings and not
        # self.tool_calls``), so a non-None result with ``content:None``
        # always carries either tool_calls OR reasoning content — both
        # legitimate OpenAI-spec shapes, both rejected by Alibaba's Qwen
        # endpoint with "content field is a required field". An empty
        # string keeps the semantics ("no surrounding text") and passes
        # strict validators.
        if result is not None and result.get("content") is None:
            tool_calls = result.get("tool_calls") or ()
            has_reasoning = any(
                k in result for k in ("reasoning", "reasoning_details")
            )
            result["content"] = ""
            log.debug(
                "_into_message_param normalized content:None → '' "
                "(tool_calls=%d, tool_call_ids=%s, has_reasoning=%s)",
                len(tool_calls),
                [tc.get("id") for tc in tool_calls] if tool_calls else [],
                has_reasoning,
            )
        log.debug("_into_message_param result=%r", result)
        return result

    mapper_cls._into_message_param = patched  # noqa: SLF001

    # ALSO wrap the model's ``_map_messages`` to log the final outgoing
    # message list. Captures user/tool/assistant turns alike — so if a
    # 400 traces back to a non-assistant content issue we'll see it.
    original_map_messages = OpenAIChatModel._map_messages  # noqa: SLF001

    @functools.wraps(original_map_messages)
    async def patched_map_messages(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await original_map_messages(self, *args, **kwargs)
        if log.isEnabledFor(10):  # DEBUG
            try:
                import json  # noqa: PLC0415

                log.debug(
                    "_map_messages OUTGOING request body:\n%s",
                    json.dumps(list(result), indent=2, default=str),
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "_map_messages OUTGOING (unprintable): %r (err: %s)",
                    result, exc,
                )
        return result

    OpenAIChatModel._map_messages = patched_map_messages  # noqa: SLF001
    _INSTALLED = True


__all__ = ["install_assistant_content_shim"]

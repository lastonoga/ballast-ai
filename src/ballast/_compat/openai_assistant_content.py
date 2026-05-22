"""Pluggable shim that normalizes outgoing OpenAI assistant messages.

## Problem

pydantic-ai's ``OpenAIChatModel._MapModelResponseContext.\
_into_message_param`` faithfully emits ``content=None`` for assistant
turns that carry only ``tool_calls`` or only reasoning (per OpenAI spec).
Most upstreams accept that. Alibaba's Qwen endpoint validates strictly
and rejects with::

    <400> InternalError.Algo.InvalidParameter:
        The content field is a required field.

Tomorrow a different upstream will reject a different field. Hardcoding
"replace ``None`` with ``""``" doesn't scale.

## Design

A ``AssistantMessageNormalizer`` is a strategy that mutates an outgoing
assistant ``ChatCompletionAssistantMessageParam`` dict. The framework
ships ``NullContentNormalizer`` as the default (covers the Alibaba/Qwen
case). Apps register additional normalizers via
``register_assistant_message_normalizer(...)`` or
``configure_assistant_message_normalizers([...])``.

``install_assistant_content_shim()`` monkey-patches
``_into_message_param`` once: every emitted assistant message is fed
through the registry in registration order. Each normalizer mutates
the dict in place (or returns ``None`` to skip).

## Why monkey-patch and not a Model subclass

``_MapModelResponseContext`` is a nested ``@dataclass`` on
``OpenAIChatModel`` (and its subclass ``OpenRouterModel``) instantiated
implicitly by the model when building requests. Subclassing would
require apps to swap the ``Model`` class — defeats the point. A
targeted patch hits every adapter uniformly.

## When to remove

When pydantic-ai upstream adds a model-profile flag that triggers
``content=""`` for strict-mode endpoints, the default normalizer can
be dropped. The plugin surface stays useful for future provider
quirks.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ballast.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence


_log = get_logger(__name__)


@runtime_checkable
class AssistantMessageNormalizer(Protocol):
    """Strategy that mutates an outgoing assistant message dict.

    Implementations may mutate ``message`` in place AND/OR return the
    (possibly new) message. Returning ``None`` is treated as "no
    change". Implementations should be defensive — never raise — so
    one buggy normalizer can't break request handling.
    """

    name: str
    """Human-readable id for DEBUG logs and idempotency keys."""

    def normalize(self, message: dict[str, Any]) -> dict[str, Any] | None: ...


class NullContentNormalizer:
    """Replace ``content: None`` with ``""`` on assistant turns that
    have some other content (tool_calls or reasoning).

    pydantic-ai already drops truly-empty responses upstream (no texts,
    thinkings, or tool_calls → returns ``None`` from
    ``_into_message_param``), so any non-None dict that has
    ``content: None`` definitionally carries either ``tool_calls`` or
    a reasoning field — both valid OpenAI-spec shapes that strict
    upstreams (e.g. Alibaba's Qwen) still reject. An empty string
    preserves the semantics ("no surrounding text").
    """

    name = "null_content"

    def normalize(self, message: dict[str, Any]) -> dict[str, Any] | None:
        if message.get("content") is not None:
            return None
        message["content"] = ""
        return message


# Module-level registry. Mutated by ``register_assistant_message_normalizer``
# / ``configure_assistant_message_normalizers``; read by the monkey-patch.
_normalizers: list[AssistantMessageNormalizer] = []
_patched = False


def register_assistant_message_normalizer(
    normalizer: AssistantMessageNormalizer,
) -> None:
    """Append a normalizer. Idempotent by ``name``."""
    if any(n.name == normalizer.name for n in _normalizers):
        return
    _normalizers.append(normalizer)
    _log.debug(
        "registered assistant-message normalizer: %s", normalizer.name,
    )


def clear_assistant_message_normalizers() -> None:
    """Reset the normalizer registry. Test helper."""
    _normalizers.clear()


def install_assistant_content_shim() -> None:
    """Patch ``OpenAIChatModel._MapModelResponseContext._into_message_param``.

    Idempotent. ImportError-safe — slim test envs without openai extras
    skip silently.

    The patched method:
      1. Runs the original ``_into_message_param``.
      2. If it returns ``None`` (truly empty response), returns ``None``.
      3. Walks the registered normalizers in registration order. Each
         may mutate the dict in place; non-``None`` returns replace
         the running message dict.
      4. Returns the final dict.

    Exceptions inside a normalizer are logged at DEBUG and swallowed —
    a buggy normalizer never breaks request flow.
    """
    global _patched  # noqa: PLW0603
    if _patched:
        return

    try:
        from pydantic_ai.models.openai import OpenAIChatModel  # noqa: PLC0415
    except ImportError:
        _patched = True
        return

    mapper_cls = OpenAIChatModel._MapModelResponseContext  # noqa: SLF001
    original = mapper_cls._into_message_param  # noqa: SLF001

    @functools.wraps(original)
    def patched(self: Any) -> Any:
        result = original(self)
        if result is None:
            return None
        for normalizer in _normalizers:
            try:
                new_result = normalizer.normalize(result)
            except Exception as exc:  # noqa: BLE001
                _log.debug(
                    "normalizer %s raised: %s — skipping",
                    normalizer.name, exc,
                )
                continue
            if new_result is not None:
                result = new_result
        if _log.isEnabledFor(10):  # DEBUG
            _log.debug("_into_message_param normalized result=%r", result)
        return result

    mapper_cls._into_message_param = patched  # noqa: SLF001
    _patched = True
    _log.debug(
        "OpenAI assistant-content shim installed (mapper_cls=%r)",
        mapper_cls,
    )

    # ALSO wrap the model's ``_map_messages`` to log the final outgoing
    # message list when DEBUG logging is on. Captures user/tool/
    # assistant turns alike — useful when chasing strict-upstream 400s
    # that don't involve assistant content.
    original_map_messages = OpenAIChatModel._map_messages  # noqa: SLF001

    @functools.wraps(original_map_messages)
    async def patched_map_messages(
        self: Any, *args: Any, **kwargs: Any,
    ) -> Any:
        result = await original_map_messages(self, *args, **kwargs)
        if _log.isEnabledFor(10):
            try:
                import json  # noqa: PLC0415

                _log.debug(
                    "_map_messages OUTGOING request body:\n%s",
                    json.dumps(list(result), indent=2, default=str),
                )
            except Exception as exc:  # noqa: BLE001
                _log.debug(
                    "_map_messages OUTGOING (unprintable): %r (err: %s)",
                    result, exc,
                )
        return result

    OpenAIChatModel._map_messages = patched_map_messages  # noqa: SLF001


def configure_assistant_message_normalizers(
    normalizers: Sequence[AssistantMessageNormalizer] | None = None,
) -> None:
    """Convenience: install the patch and register normalizers.

    When ``normalizers`` is ``None``, the default
    (``NullContentNormalizer()``) is registered — preserves
    backwards-compatible behaviour for apps that import the framework
    without configuring anything explicitly.

    Pass an explicit sequence to override the default — including
    passing ``[]`` to disable the framework-default normalizer
    entirely (useful for tests that want to observe raw upstream
    behaviour).
    """
    install_assistant_content_shim()
    resolved: Sequence[AssistantMessageNormalizer] = (
        normalizers if normalizers is not None
        else (NullContentNormalizer(),)
    )
    for n in resolved:
        register_assistant_message_normalizer(n)


__all__ = [
    "AssistantMessageNormalizer",
    "NullContentNormalizer",
    "clear_assistant_message_normalizers",
    "configure_assistant_message_normalizers",
    "install_assistant_content_shim",
    "register_assistant_message_normalizer",
]

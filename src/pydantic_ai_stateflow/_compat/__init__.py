"""Framework-level upstream compatibility shims.

Each shim documents the upstream issue it works around and the
conditions for its removal. Shims now follow a **pluggable** pattern:
the framework installs a single monkey-patch and exposes a registry
of strategy objects, so apps can extend the behaviour for new
upstream quirks without copying the patch.

Default strategies are registered at import-time so behaviour stays
unchanged for callers that just ``import pydantic_ai_stateflow``.
Apps that want to override a default reach for the relevant
``configure_*`` helper exported from
``pydantic_ai_stateflow.observability`` / this module.
"""

from pydantic_ai_stateflow._compat.openai_assistant_content import (
    AssistantMessageNormalizer,
    NullContentNormalizer,
    clear_assistant_message_normalizers,
    configure_assistant_message_normalizers,
    install_assistant_content_shim,
    register_assistant_message_normalizer,
)

# Install the OpenAI assistant-content patch with the default normalizer
# (NullContentNormalizer) so existing apps see no behaviour change.
# Override at app startup via configure_assistant_message_normalizers(
# [...own normalizers...]).
configure_assistant_message_normalizers()


__all__ = [
    "AssistantMessageNormalizer",
    "NullContentNormalizer",
    "clear_assistant_message_normalizers",
    "configure_assistant_message_normalizers",
    "install_assistant_content_shim",
    "register_assistant_message_normalizer",
]

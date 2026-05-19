"""Framework-level upstream compatibility shims.

These are import-time patches applied once when ``pydantic_ai_stateflow``
is imported. Each shim documents the upstream issue it works around and
the conditions for its removal.

Keep this module tiny: only changes that genuinely have no app-level
escape hatch belong here. Anything an app can opt into directly (e.g.
``OpenRouterModelSettings.openrouter_provider``) should not be here.
"""

from pydantic_ai_stateflow._compat.openai_assistant_content import (
    install_assistant_content_shim,
)

install_assistant_content_shim()


__all__ = ["install_assistant_content_shim"]

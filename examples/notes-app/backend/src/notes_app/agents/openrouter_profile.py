"""ModelProfile overrides for OpenRouter model ids that pydantic-ai's
built-in registry doesn't recognise correctly yet.

## Why this exists

Pydantic-ai's default ``output_type=BaseModel`` path uses ``ToolOutput``,
which sets ``tool_choice="required"`` on the request. OpenRouter routes
``qwen/qwen3.6-*`` to providers that support native JSON-schema output
(``response_format={"type": "json_schema", ...}``) but **do NOT** support
``tool_choice="required"`` — so every Qwen3.6 call that returns a
``BaseModel`` 404s.

Pydantic-ai 1.87's static qwen profile (``profiles/qwen.py``) matches
only ``qwen3.5`` (regex ``qwen-?3[\\.\\-]5``); 3.6 falls through to a
default profile where ``supports_json_schema_output=False``. That flag
gates ``NativeOutput`` in ``models/__init__.py`` (``raise UserError
("Native structured output is not supported by this model.")``).

This module patches the profile for ``qwen/qwen3.6*`` ids: same flags
as pydantic-ai's built-in qwen3.5 profile (so the patch becomes a
no-op the moment pydantic-ai widens the regex). With the patch in
place, agents can use ``output_type=Schema`` (the default) and rely
on provider-side JSON-schema validation.

Reference: pydantic-ai output docs —
<https://pydantic.dev/docs/ai/core-concepts/output/>.

## Why not PromptedOutput?

We used to wrap outputs in ``PromptedOutput(Schema)`` to dodge the
tool-choice rejection. That works but is "least reliable" per
pydantic-ai's own docs — JSON-in-text without provider-side
schema enforcement. Native mode (this patch) gives provider-side
``response_format`` validation, which is strictly better for
structured outputs.
"""

from __future__ import annotations

from pydantic_ai.profiles import InlineDefsJsonSchemaTransformer, ModelProfile

# Same flags as pydantic-ai's built-in qwen3.5 profile, just applied to
# 3.6 too. When upstream widens its regex this can be deleted.
_QWEN_3_6_NATIVE_PROFILE = ModelProfile(
    json_schema_transformer=InlineDefsJsonSchemaTransformer,
    ignore_streamed_leading_whitespace=True,
    supports_json_schema_output=True,
    supports_json_object_output=True,
    # Force native JSON-schema mode for ``output_type=BaseModel`` agents
    # — without this, pydantic-ai's default ('tool') is picked, which
    # OpenRouter's Qwen3.6 endpoints reject (no ``tool_choice='required'``
    # support).
    default_structured_output_mode="native",
    # Alibaba's Qwen3.6 endpoints reject any ``response_format`` request
    # unless the word "json" appears in the prompt. With this flag set,
    # pydantic-ai inlines the prompted-output template (which contains
    # "JSON object") into the instructions even in native mode —
    # satisfies the provider's check.
    native_output_requires_schema_in_instructions=True,
)


def profile_for(model_id: str) -> ModelProfile | None:
    """Profile override for ``model_id``, or ``None`` to use upstream's.

    Pass the result to ``OpenRouterModel(..., profile=...)``. ``None``
    means "let pydantic-ai's built-in registry resolve it" — the
    constructor accepts ``profile=None`` as the default.
    """
    if model_id.startswith("qwen/qwen3.6") or model_id.startswith("qwen/qwen-3.6"):
        return _QWEN_3_6_NATIVE_PROFILE
    return None


__all__ = ["profile_for"]

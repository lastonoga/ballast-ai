"""OpenRouter-backed pydantic-ai Agent for the notes app.

The agent answers in plain text wrapped in a `ChatReply` JSON object
(iteration 2 has no domain tools). `build_agent()` is exposed so tests can
swap the model.

The framework's `make_runner` adapter (iteration 2.1) handles the
`run_stream → stream_output → diff → emit canonical AG-UI events` loop —
no per-app runner glue is needed anymore.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

DEFAULT_MODEL = "qwen/qwen3.6-plus"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = (
    "You are a helpful assistant inside a notes app. "
    "Answer concisely and conversationally. "
    "Always wrap your reply in the ChatReply JSON object."
)


class ChatReply(BaseModel):
    """Structured envelope for the assistant's reply.

    Iteration 2 keeps this trivial; iteration 3 may grow it (e.g. tool-call
    intents, citations). Pydantic-ai enforces this shape via `output_type`
    which on OpenAI-compatible backends routes through tool-calling
    (Qwen-on-OpenRouter advertises the tool-calling capability).
    """

    reply: str = Field(..., description="Plain-text reply to show to the user.")


def build_agent(
    *,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str = OPENROUTER_BASE_URL,
) -> Agent[None, ChatReply]:
    """Build the OpenRouter-backed agent.

    Resolves `model_name` from `OPENROUTER_MODEL` env (default
    `qwen/qwen3.6-plus`) and `api_key` from `OPENROUTER_API_KEY`.
    """
    resolved_model = model_name or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY env var is required to build the agent"
        )

    provider = OpenAIProvider(base_url=base_url, api_key=resolved_key)
    model = OpenAIModel(resolved_model, provider=provider)
    return Agent(
        model=model,
        output_type=ChatReply,
        system_prompt=SYSTEM_PROMPT,
    )

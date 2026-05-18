"""OpenRouter-backed pydantic-ai Agent + framework `StreamEvent` adapter.

The agent answers in plain text wrapped in a `ChatReply` JSON object (iteration
2 has no domain tools). We expose `build_agent()` so tests can swap the model.

The `make_agent_runner(agent)` factory returns an `AgentRunner` compatible
with `build_streaming_router(agent_runner=...)`. It pulls the latest user
message out of the framework's `_PostMessageBody` (`parts: list[dict]`) and
translates pydantic-ai's per-delta str chunks into framework `StreamEvent`s.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_stateflow.api.streaming import StreamEvent

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


def _extract_user_text(parts: list[dict[str, Any]]) -> str:
    """Pull plain text out of the framework's `parts` payload.

    The framework's `_PostMessageBody` accepts `parts: list[dict]` without
    prescribing a schema. We accept either AG-UI-ish `{"type": "text",
    "text": "..."}` or a flat `{"text": "..."}` for ergonomics.
    """
    chunks: list[str] = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        text = p.get("text")
        if isinstance(text, str):
            chunks.append(text)
            continue
        # OpenAI-style content array fragments
        content = p.get("content")
        if isinstance(content, str):
            chunks.append(content)
    return "\n".join(chunks).strip()


AgentRunner = Callable[..., AsyncIterator[StreamEvent]]


def make_agent_runner(
    agent: Agent[None, ChatReply],
) -> AgentRunner:
    """Return an `agent_runner` compatible with `build_streaming_router`.

    Emits the following `StreamEvent.kind` values (iteration 3 must keep
    these in sync with the assistant-ui frontend mapping):

      - `text_delta` — `{"text": <incremental chunk of reply>}`
      - `done`       — `{"reply": <final reply string>}`
      - `error`      — `{"message": <stringified exception>}`
    """

    async def _runner(
        *,
        thread_id: UUID,
        message: Any,  # framework passes its internal _PostMessageBody
        tenant_id: UUID,
    ) -> AsyncIterator[StreamEvent]:
        del thread_id, tenant_id  # iteration 2: no per-thread history yet
        parts = getattr(message, "parts", []) or []
        user_text = _extract_user_text(parts)
        if not user_text:
            yield StreamEvent(
                kind="error",
                data={"message": "no text content in message.parts"},
            )
            return

        last_emitted = ""
        try:
            async with agent.run_stream(user_text) as result:
                # `stream_output(debounce_by=None)` yields partially-validated
                # ChatReply objects as JSON tokens arrive. We diff against the
                # previously-emitted reply to compute a true delta.
                async for partial in result.stream_output(debounce_by=0.05):
                    current = partial.reply or ""
                    if not current:
                        continue
                    if current.startswith(last_emitted):
                        delta = current[len(last_emitted):]
                        last_emitted = current
                    else:
                        # Model rewrote the prefix (rare with partial validation);
                        # emit the full new value as the delta and resync.
                        delta = current
                        last_emitted = current
                    if delta:
                        yield StreamEvent(kind="text_delta", data={"text": delta})
                final = await result.get_output()
                yield StreamEvent(kind="done", data={"reply": final.reply})
        except Exception as exc:  # noqa: BLE001 — surface to client as SSE error
            yield StreamEvent(kind="error", data={"message": str(exc)})

    return _runner

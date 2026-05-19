"""Server-stateful Vercel-AI streaming endpoint.

The framework owns the thread + message history (multi-tenant, persisted
in ``ThreadRepository``). The wire encoding, body parsing, event taxonomy,
tool-call/text streaming, and tool-approval round-trip are delegated to
``pydantic_ai.ui.vercel_ai.VercelAIAdapter`` — we don't reimplement any of
that.

Vercel AI SDK v6 is targeted (``sdk_version=6``) so that
``@agent.tool(requires_approval=True)`` produces ``approval-requested`` UI
parts on the wire and incoming approval responses are extracted by
``VercelAIAdapter.deferred_tool_results``.

Endpoint contract::

    POST {prefix}/threads/{thread_id}/messages
        Accept: text/event-stream
        Body  : Vercel AI ``RequestData`` JSON (parsed by VercelAIAdapter)
        404   : thread not found (no lazy-create)
        200   : streaming Vercel AI events
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from pydantic_ai_stateflow.api.deps import get_tenant_id
from pydantic_ai_stateflow.api.streaming.history import (
    extract_text,
    messages_to_model_history,
)
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.agent import AgentRunResult
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.settings import ModelSettings
    from starlette.responses import Response


DepsT = TypeVar("DepsT")
OutT = TypeVar("OutT")

DepsFactory = Callable[..., Any] | Callable[..., Awaitable[Any]]
"""Callable that mints per-request agent deps.

Receives keyword arguments ``thread_id: UUID``, ``tenant_id: UUID``, and
``message: ModelMessage`` (the just-arrived user turn). May be sync or
async. Anything that isn't a callable is passed through unchanged.
"""

_TenantDep = Depends(get_tenant_id)


async def _resolve_deps(
    deps_factory: Any,
    *,
    thread_id: UUID,
    tenant_id: UUID,
    message: ModelMessage | None,
) -> Any:
    """Invoke ``deps_factory`` per-request, or pass it through if not callable."""
    if deps_factory is None:
        return None
    if callable(deps_factory):
        result = deps_factory(
            thread_id=thread_id, tenant_id=tenant_id, message=message,
        )
        if inspect.isawaitable(result):
            return await result
        return result
    return deps_factory


def _last_user_text(messages: list[ModelMessage]) -> str:
    """Pull the trailing user-prompt text from the parsed UI messages.

    Walks the list in reverse looking for a ``ModelRequest`` carrying a
    ``UserPromptPart``. Concatenates string parts (matches our
    ``extract_text`` convention). Returns ``""`` if no user turn found.
    """
    from pydantic_ai.messages import (  # noqa: PLC0415
        ModelRequest,
        UserPromptPart,
    )

    for msg in reversed(messages):
        if not isinstance(msg, ModelRequest):
            continue
        text_chunks: list[str] = []
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                content = part.content
                if isinstance(content, str):
                    text_chunks.append(content)
                else:
                    # Multimodal: keep string items, drop binaries.
                    for item in content:
                        if isinstance(item, str):
                            text_chunks.append(item)
        if text_chunks:
            return "".join(text_chunks)
    return ""


_DEFAULT_HISTORY_LIMIT = 200


def build_streaming_router(
    *,
    thread_repo: ThreadRepository,
    agent: Agent[Any, Any],
    deps_factory: DepsFactory | None = None,
    model_settings: ModelSettings | None = None,
    prefix: str = "",
    history_limit: int = _DEFAULT_HISTORY_LIMIT,
) -> APIRouter:
    """Mount ``POST {prefix}/threads/{id}/messages`` as a Vercel-AI stream.

    Server-stateful contract: the thread MUST already exist (404 otherwise
    — no lazy-create). The just-arrived user turn is persisted via
    ``thread_repo.add_message`` BEFORE the model runs, so a client crash
    mid-stream still leaves the thread consistent. After the agent
    completes the assistant reply is persisted via an ``on_complete``
    callback wired into ``VercelAIAdapter.run_stream``.

    Tool-approval responses (Vercel AI SDK v6 ``approval-responded`` parts)
    are extracted by ``VercelAIAdapter.deferred_tool_results`` and threaded
    into ``run_stream`` so ``@agent.tool(requires_approval=True)`` tools
    resume after the user clicks Approve/Cancel.

    ``message_history`` is reconstructed from ``thread_repo.history(...)``
    (excluding the just-persisted current user turn — pydantic-ai re-derives
    that one from the incoming body messages).

    Args:
      thread_repo: source of truth for thread + message persistence.
      agent: pydantic-ai ``Agent`` to run on each request.
      deps_factory: callable invoked per request with
        ``thread_id``, ``tenant_id``, ``message`` kwargs to mint fresh
        deps for the agent. If ``None`` and the agent declares
        ``deps_type``, that's the caller's problem (pydantic-ai will
        raise). Non-callable values pass through unchanged.
      model_settings: forwarded to the agent on every run. Use this to
        thread ``temperature``, OpenRouter reasoning config, etc.
      prefix: optional router prefix.
      history_limit: cap on the number of rows hydrated from the repo
        per request (default 200).
    """
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter  # noqa: PLC0415

    router = APIRouter(prefix=prefix)

    @router.post("/threads/{thread_id}/messages")
    async def post_message(
        request: Request,
        thread_id: UUID,
        tenant_id: UUID = _TenantDep,
    ) -> Response:
        thread = await thread_repo.load(thread_id, tenant_id=tenant_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")

        adapter = await VercelAIAdapter.from_request(
            request, agent=agent, sdk_version=6,
        )

        last_message = adapter.messages[-1] if adapter.messages else None
        prompt_text = _last_user_text(adapter.messages)

        if prompt_text:
            await thread_repo.add_message(
                thread_id,
                role="user",
                parts=[{"type": "text", "text": prompt_text}],
                tenant_id=tenant_id,
            )

        rows = await thread_repo.history(
            thread_id, tenant_id=tenant_id, limit=history_limit,
        )
        history = messages_to_model_history(rows, drop_prompt=prompt_text)

        resolved_deps = await _resolve_deps(
            deps_factory,
            thread_id=thread_id,
            tenant_id=tenant_id,
            message=last_message,
        )

        async def on_complete(result: AgentRunResult[Any]) -> None:
            """Persist the assistant reply on successful completion.

            Skips persistence when ``result.output`` is a
            ``DeferredToolRequests`` — that's a paused run waiting on
            human approval, not a final assistant turn. The next request
            (with approval responses attached) will produce the real
            text reply and we'll persist that.
            """
            from pydantic_ai import DeferredToolRequests  # noqa: PLC0415

            output = result.output
            if isinstance(output, DeferredToolRequests):
                return
            text = output if isinstance(output, str) else str(output)
            if not text:
                return
            await thread_repo.add_message(
                thread_id,
                role="assistant",
                parts=[{"type": "text", "text": text}],
                tenant_id=tenant_id,
            )

        # We already built the adapter (to peek at `messages` for the
        # user-message-persist step). Drive the stream off it directly
        # rather than re-entering ``dispatch_request`` (which would call
        # ``from_request`` a second time and re-parse the same body).
        # Tool-approval responses are auto-extracted from the incoming
        # body by ``adapter.deferred_tool_results`` (called internally
        # by ``run_stream`` when ``deferred_tool_results`` is omitted)
        # and threaded back into the agent run so paused
        # ``requires_approval=True`` tools resume.
        return adapter.streaming_response(
            adapter.run_stream(
                message_history=history,
                deps=resolved_deps,
                model_settings=model_settings,
                on_complete=on_complete,
            ),
        )

    return router


__all__ = [
    "DepsFactory",
    "build_streaming_router",
    "extract_text",
    "messages_to_model_history",
]

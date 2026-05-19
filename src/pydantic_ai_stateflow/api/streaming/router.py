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


def _find_last_role_id(
    history: list[Any], role: str,
) -> UUID | None:
    """Return the id of the last ``Message`` with ``role`` in history, or None.

    Used by the streaming router to thread the new turn's ``parent_id``
    through the message tree:

      - submit-message → parent_of_new_user = last "assistant"
      - regenerate-message → parent_of_new_assistant = last "user"
    """
    for msg in reversed(history):
        if getattr(msg, "role", None) == role:
            return getattr(msg, "id", None)
    return None


def _trim_adapter_messages_to_last_user_turn(adapter: Any) -> None:
    """Replace ``adapter.messages`` with just the trailing user request.

    ``UIAdapter.run_stream*`` appends ``self.messages`` to the
    caller-supplied ``message_history`` (``_adapter.py:511-512``):

    .. code-block:: python

        frontend_messages = self.sanitize_messages(self.messages, ...)
        message_history = [*(message_history or []), *frontend_messages]

    For a server-stateful contract that's the wrong default — the body's
    client-side history would duplicate everything we already loaded
    from the repo, and any stale assistant tool-call parts from an
    aborted prior run would smuggle dangling ``tool_call_id``s into the
    next LLM request, causing the upstream to 400.

    We trim the adapter's cached messages list to only the LAST
    ``ModelRequest`` (the new user turn). ``messages`` is a
    ``cached_property``, which on CPython is stored in ``__dict__`` —
    setting that key pre-empts the property computation and is honored
    by every downstream access (including ``sanitize_messages`` and the
    adapter's own deferred-tool-results extractor, which we still need
    to fire on the FULL body separately if it's relevant).

    NB: ``deferred_tool_results`` is ALSO a cached_property on the
    adapter and reads from the original body — not from ``messages`` —
    so trimming ``messages`` doesn't break HITL approval round-trips.
    Just confirmed by reading ``VercelAIAdapter.deferred_tool_results``
    which iterates ``self.run_input.messages`` directly.
    """
    from pydantic_ai.messages import ModelRequest  # noqa: PLC0415

    msgs = adapter.messages
    last_request: ModelRequest | None = None
    for msg in reversed(msgs):
        if isinstance(msg, ModelRequest):
            last_request = msg
            break
    adapter.__dict__["messages"] = [last_request] if last_request else []


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
        trigger = getattr(adapter.run_input, "trigger", "submit-message")
        is_regenerate = trigger == "regenerate-message"
        # ``deferred_tool_results`` is non-None when the body carries
        # approval responses for a paused ``requires_approval=True`` tool
        # call — VercelAIAdapter extracts them from
        # ``run_input.messages`` and returns a ``DeferredToolResults``
        # which pydantic-ai then matches against in-flight tool calls.
        deferred_results = adapter.deferred_tool_results

        # Server-stateful contract: the repo is the source of truth for
        # conversation history. We trim ``adapter.messages`` down to the
        # latest user turn so the upstream ``UIAdapter.run_stream``
        # doesn't ALSO append the body's full client-side history on top
        # of our ``message_history`` — which would duplicate the current
        # turn AND smuggle stale assistant tool-call parts from a
        # previously-aborted run back into the prompt.
        #
        # EXCEPTION: when ``deferred_tool_results`` is present, the
        # assistant turn that issued the now-being-approved/denied tool
        # call MUST remain in ``adapter.messages`` so pydantic-ai can
        # match the result to the original call. Trimming it would
        # leave the deferred results orphaned — surfacing as
        # "Tool call results were provided, but the message history
        # does not contain any unprocessed tool calls."
        if deferred_results is None:
            _trim_adapter_messages_to_last_user_turn(adapter)

        # Determine parent_id for the new turn so the message tree stays
        # connected. Two flows:
        #
        #   submit-message: parent = last assistant in active branch
        #     (or None for the very first turn). Persist a new user msg
        #     under that parent; on_complete persists the assistant under
        #     the freshly-created user msg.
        #
        #   regenerate-message: don't persist a new user msg — frontend
        #     is asking for a re-run of the existing last user turn.
        #     New assistant becomes a sibling of the prior one (same
        #     parent_id = last user msg's id).
        active_branch_before = await thread_repo.history(
            thread_id, tenant_id=tenant_id, limit=history_limit,
        )

        if is_regenerate:
            # Regenerate semantics: the assistant being regenerated is
            # REPLACED with a sibling whose parent is the same user turn
            # the old assistant was replying to. We skip past any trailing
            # assistant in the active branch to find that user turn.
            last_user_id = _find_last_role_id(active_branch_before, "user")
            new_assistant_parent_id = last_user_id
        else:
            # Submit-message semantics: the new user msg's parent is the
            # LAST message in the active branch (any role). If the prior
            # run was aborted mid-stream, the active branch may end in a
            # user msg with no assistant reply — that's still a valid
            # parent, the new user msg simply continues the conversation.
            new_user_parent_id = (
                active_branch_before[-1].id if active_branch_before else None
            )
            new_assistant_parent_id = None  # set after we persist user

            if prompt_text:
                user_msg = await thread_repo.add_message(
                    thread_id,
                    role="user",
                    parts=[{"type": "text", "text": prompt_text}],
                    tenant_id=tenant_id,
                    parent_id=new_user_parent_id,
                )
                new_assistant_parent_id = user_msg.id

        rows = await thread_repo.history(
            thread_id, tenant_id=tenant_id, limit=history_limit,
        )
        # We already persisted the current user turn AND the trimmed
        # ``adapter.messages`` will carry it back into the prompt via the
        # adapter's frontend_messages merge, so drop it from our
        # repo-driven history to avoid duplication. For regenerate the
        # last user IS the prompt — drop it for the same reason.
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
                parent_id=new_assistant_parent_id,
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

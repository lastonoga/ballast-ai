"""``stream_response`` primitive — apps wire their own streaming routes.

Replaces the framework-owned ``POST /threads/{id}/messages`` route.
Apps write their own FastAPI handlers that resolve the right agent
instance (their concern — opaque ``Thread.agent`` string lookup) and
delegate to ``stream_response(...)`` for the heavy lifting:

  - body-vs-DB sync (edit / regenerate handled implicitly)
  - durable vs inline streaming dispatch (durable path requires
    ``StateflowDurableAgent``)
  - Vercel-AI wire encoding via ``VercelAIAdapter``
  - approval-resume detection + routing
  - assistant-turn persistence (non-durable path)

The companion ``cancel_thread_workflows`` primitive cancels every
active workflow for a thread (durable path only).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic_ai_stateflow.api.streaming.router import (
    _DEFAULT_HISTORY_LIMIT,
    _durable_post_message,
    _parse_body_messages,
    _sync_db_with_body,
    _trim_adapter_messages_to_last_user_prompt,
    extract_text,
    messages_to_model_history,
)
from pydantic_ai_stateflow.api.streaming.wire_encoder import VercelAIWireEncoder
from pydantic_ai_stateflow.errors import (
    CancelNotSupported,
    EmptyMessageBody,
    ThreadNotFound,
)
from pydantic_ai_stateflow.runtime.durable_agent import StateflowDurableAgent
from pydantic_ai_stateflow.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request
    from starlette.responses import Response

    from pydantic_ai_stateflow.runtime.agents import StateflowAgent


_log = get_logger(__name__)


async def stream_response(
    *,
    request: "Request",
    thread_id: UUID,
    agent: "StateflowAgent",
    history_limit: int = _DEFAULT_HISTORY_LIMIT,
) -> "Response":
    """Body-vs-DB sync + agent run + Vercel-AI streaming response.

    The framework's streaming primitive. Apps that own their own
    streaming route resolve the agent for a thread (their concern —
    opaque ``Thread.agent`` string lookup) and delegate to this.

    Dispatches durable vs inline based on ``isinstance(agent,
    StateflowDurableAgent)`` — durable path enqueues a DBOS workflow
    and tails the event log; inline path runs the pydantic-ai Agent
    and streams via ``VercelAIAdapter``.
    """
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter  # noqa: PLC0415

    from pydantic_ai_stateflow.runtime.engine import get_engine  # noqa: PLC0415
    engine = get_engine()
    thread_repo = engine.thread_repo
    event_log = engine.event_log
    event_stream = engine.event_stream

    thread = await thread_repo.load(thread_id)
    if thread is None:
        raise ThreadNotFound(thread_id=str(thread_id))

    if isinstance(agent, StateflowDurableAgent):
        return await _durable_post_message(
            request=request,
            thread_id=thread_id,
            stateflow_agent=agent,
            thread_repo=thread_repo,
            event_log=event_log,
            event_stream=event_stream,
            encoder=VercelAIWireEncoder(),
            history_limit=history_limit,
        )

    # ── Non-durable path ─────────────────────────────────────────────
    body_messages = await _parse_body_messages(request)
    rows = await _sync_db_with_body(
        thread_id=thread_id,
        body_messages=body_messages,
        thread_repo=thread_repo,
        history_limit=history_limit,
    )

    pydantic_agent = agent.agent
    model_settings = agent.model_settings()

    adapter = await VercelAIAdapter.from_request(
        request, agent=pydantic_agent, sdk_version=6,
    )

    last_message = adapter.messages[-1] if adapter.messages else None
    prompt_text = (
        extract_text(rows[-1].parts)
        if rows and rows[-1].role == "user"
        else ""
    )
    deferred_results = adapter.deferred_tool_results

    if deferred_results is None:
        _trim_adapter_messages_to_last_user_prompt(adapter)
    else:
        _log.info(
            "stream_response: deferred_tool_results present "
            "(approvals=%d, calls=%d) — skipping trim",
            len(deferred_results.approvals or {}),
            len(deferred_results.calls or {}),
        )

    history = messages_to_model_history(rows, drop_prompt=prompt_text)

    resolved_deps = await agent.build_deps(
        thread=thread,
        message=last_message,
    )

    async def on_complete(result: Any) -> None:
        from pydantic_ai import DeferredToolRequests  # noqa: PLC0415
        from pydantic_ai.ui.vercel_ai import (  # noqa: PLC0415
            VercelAIAdapter as _VercelAIAdapter,
        )

        output = result.output
        if isinstance(output, DeferredToolRequests):
            _log.info(
                "stream_response on_complete: agent paused with "
                "DeferredToolRequests (thread=%s) — skipping persist",
                thread_id,
            )
            return

        all_msgs = result.all_messages()
        ui_msgs = _VercelAIAdapter.dump_messages(
            all_msgs, sdk_version=6,
        )
        last_user_idx = -1
        for i, m in enumerate(ui_msgs):
            if m.role == "user":
                last_user_idx = i
        asst_parts: list[dict[str, Any]] = []
        for m in ui_msgs[last_user_idx + 1:]:
            if m.role != "assistant":
                continue
            for p in m.parts:
                asst_parts.append(
                    p.model_dump(
                        mode="json",
                        by_alias=True,
                        exclude_none=True,
                    ),
                )

        if not asst_parts:
            return
        await thread_repo.add_message(
            thread_id,
            role="assistant",
            parts=asst_parts,
        )

    if not rows or rows[-1].role != "user":
        raise EmptyMessageBody(
            "Cannot start run: thread has no user message to respond to.",
            hint="POST a user message first.",
        )

    return adapter.streaming_response(
        adapter.run_stream(
            message_history=history,
            deps=resolved_deps,
            model_settings=model_settings,
            on_complete=on_complete,
        ),
    )


async def cancel_thread_workflows(
    *,
    thread_id: UUID,
    agent: "StateflowAgent",
) -> int:
    """Cancel every active workflow for ``thread_id``.

    Only meaningful for ``StateflowDurableAgent`` instances —
    non-durable agents don't have cancellable workflows. Raises
    ``CancelNotSupported`` for non-durable agents.

    Returns the count of workflows that were cancelled.
    """
    if not isinstance(agent, StateflowDurableAgent):
        raise CancelNotSupported(
            "cancel is only meaningful for StateflowDurableAgent "
            "threads; non-durable agents don't have cancellable workflows",
            context={"thread_id": str(thread_id)},
        )
    return await agent.cancel_thread_runs(thread_id)


__all__ = ["cancel_thread_workflows", "stream_response"]

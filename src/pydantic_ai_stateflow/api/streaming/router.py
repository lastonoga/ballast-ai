"""Server-stateful Vercel-AI streaming endpoint.

Single endpoint: ``POST /threads/{thread_id}/messages``. Persists the
new user message AND triggers the agent run AND streams the response
in one shot.

Persistence model is a **flat linear message list** — no parent_id,
no tree, no branches. The UI runtime (``@assistant-ui/react-ai-sdk``
over Vercel ``useChat``) renders a flat array, so we match that shape
end-to-end. Edits and regenerates are handled by **body-vs-DB sync**:

  1. Parse the Vercel-AI body's ``messages`` array
  2. Find the longest common id-prefix with the DB's history
  3. Delete the DB tail past that prefix (rows the client dropped)
  4. Append the body tail past that prefix (rows the client added)
  5. Run the agent
  6. Persist the assistant turn

This subsumes both edit (truncate user-msg-and-after, append edited
user) and regenerate (truncate assistant-and-after, append nothing,
let agent re-emit assistant) without special-case branches.

Vercel AI SDK v6 is targeted (``sdk_version=6``) so that
``@agent.tool(requires_approval=True)`` produces ``approval-requested``
UI parts on the wire and incoming approval responses are extracted by
``VercelAIAdapter.deferred_tool_results``.

Endpoint contract::

    POST {prefix}/threads/{thread_id}/messages
        Accept: text/event-stream
        Body  : Vercel AI ``RequestData`` JSON (parsed by VercelAIAdapter)
        404   : thread not found (no lazy-create)
        200   : streaming Vercel AI events
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

from pydantic_ai_stateflow.errors import ThreadNotFound
from pydantic_ai_stateflow.api.streaming.history import (
    extract_text,
    messages_to_model_history,
)
from pydantic_ai_stateflow.api.streaming.wire_encoder import (
    VercelAIWireEncoder,
    WireEncoder,
)
from pydantic_ai_stateflow.logging import get_logger
from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.observability.trace_names import TraceName
from pydantic_ai_stateflow.persistence.events.repository import (
    EventLogRepository,
)
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository
from pydantic_ai_stateflow.runtime.agents import StateflowAgent
from pydantic_ai_stateflow.runtime.durable_agent import StateflowDurableAgent
from pydantic_ai_stateflow.runtime.event_stream import EventStream


def _resolve_agent_from_app(request: Request, name: str) -> StateflowAgent:
    """Resolve a ``StateflowAgent`` instance by name from ``app.state.agents``.

    Replaces the legacy process-global ``get_agent(name)`` registry —
    ``sf.create_app()`` populates ``app.state.agents`` from the
    ``agents=`` kwarg, and routes now look it up per-request.
    """
    agents = getattr(request.app.state, "agents", None)
    if not agents or name not in agents:
        known = sorted(agents) if agents else []
        raise HTTPException(
            status_code=404,
            detail=(
                f"No agent registered under name {name!r}. "
                f"Known agents: {known}. "
                f"Did you pass it to sf.create_app(agents=[...])?"
            ),
        )
    return agents[name]

_log = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.agent import AgentRunResult
    from starlette.responses import Response

    from pydantic_ai_stateflow.persistence.thread.domain import Message


DepsT = TypeVar("DepsT")
OutT = TypeVar("OutT")

DepsFactory = Callable[..., Any] | Callable[..., Awaitable[Any]]
"""Retained for backwards compatibility of the public type name.

The streaming router itself no longer takes a ``deps_factory`` — apps
register a ``StateflowAgent`` whose ``build_deps`` method serves the
same role.
"""

_DEFAULT_HISTORY_LIMIT = 200


def _pending_tool_call_ids(history: list[Any]) -> set[str]:
    """Tool-call ids in ``history`` that were issued but never returned.

    Walks ModelResponse / ModelRequest parts: collects every
    ``ToolCallPart.tool_call_id`` from assistant turns; subtracts
    every ``ToolReturnPart.tool_call_id`` from subsequent request
    turns. The remainder is the set of currently-pending deferred
    tool calls that need an approval decision.
    """
    from pydantic_ai.messages import (  # noqa: PLC0415
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
    )

    called: set[str] = set()
    returned: set[str] = set()
    for msg in history:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    called.add(part.tool_call_id)
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    returned.add(part.tool_call_id)
    return called - returned


def _trim_adapter_messages_to_last_user_prompt(adapter: Any) -> None:
    """Replace ``adapter.messages`` with a single ModelRequest holding
    just the LAST ``UserPromptPart``.

    ``UIAdapter.run_stream*`` appends ``self.messages`` to the
    caller-supplied ``message_history``. The repo is already our
    source of truth (we just synced the full body into it), so we
    only need the adapter to contribute the **prompt** — the very
    last user UserPromptPart from the body.

    Why not "the last ModelRequest": pydantic-ai's body parser
    **coalesces** consecutive user messages with no assistant between
    them into a single ModelRequest with multiple UserPromptParts.
    Keeping the whole request re-injects prior user turns that are
    already in our repo history → duplicates in the prompt.

    ``messages`` is a ``cached_property`` stored in ``__dict__`` on
    CPython — assigning the key pre-empts the property computation.
    ``deferred_tool_results`` reads from the original body, NOT from
    ``messages``, so trimming doesn't break HITL approval round-trips.
    """
    from pydantic_ai.messages import (  # noqa: PLC0415
        ModelRequest,
        UserPromptPart,
    )

    msgs = adapter.messages
    last_request: ModelRequest | None = None
    for msg in reversed(msgs):
        if isinstance(msg, ModelRequest):
            last_request = msg
            break
    if last_request is None:
        adapter.__dict__["messages"] = []
        return

    last_user_part: UserPromptPart | None = None
    for part in reversed(last_request.parts):
        if isinstance(part, UserPromptPart):
            last_user_part = part
            break
    if last_user_part is None:
        adapter.__dict__["messages"] = [last_request]
        return

    adapter.__dict__["messages"] = [
        ModelRequest(parts=[last_user_part], timestamp=last_request.timestamp),
    ]


async def _parse_body_messages(
    request: Request,
) -> list[dict[str, Any]]:
    """Return the raw Vercel-AI body's ``messages`` array."""
    body = await request.json()
    messages = body.get("messages") or []
    return [m for m in messages if isinstance(m, dict)]


def _normalize_part_for_persist(part: dict[str, Any]) -> dict[str, Any]:
    """Make sure text parts persist with ``state: "done"``.

    assistant-ui's ``MessagePartText`` discriminator on the live useChat
    stream looks at ``state`` to pick the right rendering branch. The
    in-flight body sends text parts without ``state`` (useChat fills it
    in client-side after the stream finishes), but by the time the body
    hits us the user has CONFIRMED the message — it's semantically
    done. Without this normalization the persisted user rows reload as
    invisible (the discriminator picks the wrong branch and
    ``MessagePartText`` throws "can only be used inside text or
    reasoning message parts").
    """
    if part.get("type") == "text" and "state" not in part:
        return {**part, "state": "done"}
    return part


async def _sync_db_with_body(
    *,
    thread_id: UUID,
    body_messages: list[dict[str, Any]],
    thread_repo: ThreadRepository,
    history_limit: int,
) -> list[Message]:
    """Reconcile DB rows with the body's ``messages`` array.

    Finds the longest common id-prefix between the body and the
    persisted history, deletes the DB tail past that prefix, then
    appends the body tail. After this the DB matches body[0..N] where
    N is the body's length minus any assistant rows that the body
    intentionally omitted (for regenerate the body ends in the user
    msg whose response is being regenerated — the assistant has been
    truncated client-side and we drop it from the DB too).

    Returns the freshly-reconciled history.
    """
    db_msgs = await thread_repo.history(thread_id, limit=history_limit)

    # Longest common prefix by id.
    common = 0
    while (
        common < len(body_messages)
        and common < len(db_msgs)
        and body_messages[common].get("id") == db_msgs[common].id
    ):
        common += 1

    # Drop DB rows past the common prefix — the client truncated them.
    to_drop = [m.id for m in db_msgs[common:]]
    if to_drop:
        await thread_repo.delete_messages(thread_id, ids=to_drop)
        _log.info(
            "sync: thread=%s dropped %d trailing msgs (edit/regenerate)",
            thread_id, len(to_drop),
        )

    # Append body rows past the common prefix.
    for entry in body_messages[common:]:
        raw_id = entry.get("id")
        if not isinstance(raw_id, str):
            continue
        role = entry.get("role")
        if role not in {"user", "assistant", "system", "tool"}:
            continue
        parts = entry.get("parts") or []
        if not isinstance(parts, list):
            parts = []
        await thread_repo.add_message(
            thread_id,
            id=raw_id,
            role=role,
            parts=[
                _normalize_part_for_persist(p)
                for p in parts if isinstance(p, dict)
            ],
        )

    return await thread_repo.history(thread_id, limit=history_limit)


def _build_sse_response(
    *,
    encoder: WireEncoder,
    thread_id: UUID,
    event_log: EventLogRepository,
    event_stream: EventStream,
    last_event_id: int,
) -> Response:
    """Build the SSE StreamingResponse that tails the event log.

    Pulled out so both ``enqueue_run`` and ``enqueue_approval_resume``
    can use the same generator without duplicating the polling loop.
    """
    async def _gen() -> AsyncIterator[bytes]:
        import asyncio  # noqa: PLC0415

        for chunk in encoder.initial_events(thread_id=thread_id):
            yield chunk

        last_seq = last_event_id
        poll_interval_s = 0.05
        idle_iterations = 0
        max_idle_iterations = int(30.0 / poll_interval_s)

        while True:
            events = await event_log.read_since(thread_id, after_seq=last_seq)
            if events:
                idle_iterations = 0
                for ev in events:
                    for chunk in encoder.encode_event(ev):
                        yield chunk
                    last_seq = ev.seq
                    if ev.kind in {"done", "cancelled"}:
                        for chunk in encoder.finalize():
                            yield chunk
                        return
            else:
                idle_iterations += 1
                if idle_iterations >= max_idle_iterations:
                    _log.warning(
                        "Durable stream idle for ~30s on thread %s "
                        "(last_seq=%d) — closing",
                        thread_id, last_seq,
                    )
                    for chunk in encoder.finalize():
                        yield chunk
                    return
            await asyncio.sleep(poll_interval_s)

    _ = event_stream  # reserved for future live-signal optimization
    return StreamingResponse(_gen(), media_type=encoder.content_type())


def _parse_last_event_id(request: Request) -> int:
    """Read the SSE-standard ``Last-Event-ID`` header (or 0)."""
    raw = request.headers.get("Last-Event-ID") or request.headers.get(
        "last-event-id",
    )
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        _log.warning("Ignoring malformed Last-Event-ID header: %r", raw)
        return 0


async def _durable_post_message(
    *,
    request: Request,
    thread_id: UUID,
    stateflow_agent: StateflowDurableAgent,
    thread_repo: ThreadRepository,
    event_log: EventLogRepository,
    event_stream: EventStream,
    encoder: WireEncoder,
    history_limit: int,
) -> Response:
    """Durable path: sync DB with body, enqueue workflow, tail event log.

    Two dispatch modes inside one endpoint:

    - **New turn** (normal user prompt): body's last message is user;
      sync DB, enqueue ``enqueue_run`` for a fresh agent turn.
    - **Approval resume**: body carries a paused-tool assistant message
      with ``approval-responded`` parts (assistant-ui's auto-resend
      after the user clicks Approve/Reject); extract the approvals,
      enqueue ``enqueue_approval_resume`` to execute / deny the tool
      with the human decision threaded through.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter  # noqa: PLC0415
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter  # noqa: PLC0415

    # Peek at the body for approval-responses. ``VercelAIAdapter.from_request``
    # parses the body once + caches; ``deferred_tool_results`` is a
    # cached_property that scans for ``approval-responded`` parts and
    # builds a ``DeferredToolResults`` if any are present. None → no
    # approval in the body → normal new-turn flow.
    adapter = await VercelAIAdapter.from_request(
        request, agent=stateflow_agent.agent, sdk_version=6,
    )
    deferred = adapter.deferred_tool_results

    if deferred is not None and deferred.approvals:
        # Approval resume path. Use **body's parsed messages** as
        # message_history (NOT the repo). Reasons:
        #
        # 1. Our ``_persist_assistant_turn`` skips persisting when the
        #    agent paused with ``DeferredToolRequests`` — the assistant
        #    row with the pending ToolCallPart is NOT in the repo.
        # 2. assistant-ui's body carries the full conversation state
        #    INCLUDING the assistant turn with the deferred tool call
        #    (in ``approval-responded`` state after the user clicked).
        #    VercelAIAdapter parses it into proper ``ToolCallPart``s.
        # 3. pydantic-ai matches ``deferred_tool_results[tool_call_id]``
        #    against currently-pending ``ToolCallPart`` entries; body's
        #    history has them, repo doesn't.
        #
        # Without this, the agent raises "Tool call results were
        # provided, but the message history does not contain any
        # unprocessed tool calls."
        history = list(adapter.messages)
        history_dump = ModelMessagesTypeAdapter.dump_python(
            history, mode="json",
        )

        # Filter approvals to currently-pending tool calls in history.
        # Body may carry ``approval-responded`` parts for tool calls
        # that already executed in PRIOR turns (their ``ToolReturnPart``
        # is in history). pydantic-ai requires the
        # ``deferred_tool_results`` keyset to EXACTLY match pending —
        # extras raise "Tool call results need to be provided for all
        # deferred tool calls".
        pending = _pending_tool_call_ids(history)
        approvals_dump: dict[str, bool | dict[str, Any]] = {}
        for tcid, decision in deferred.approvals.items():
            if tcid not in pending:
                continue
            if isinstance(decision, bool):
                approvals_dump[tcid] = decision
            else:
                # ToolDenied(message=...)
                approvals_dump[tcid] = {
                    "message": getattr(decision, "message", "denied"),
                }

        if not approvals_dump:
            # Defensive: no actually-pending approval matched — likely
            # a stale resend after the tool already executed. Fall back
            # to the new-turn path (or no-op SSE) instead of triggering
            # the pydantic-ai "no unprocessed tool calls" error.
            _log.warning(
                "Approval-resume body had %d approvals but none match "
                "a pending tool call in history; ignoring resume",
                len(deferred.approvals),
            )
            return _build_sse_response(
                encoder=encoder,
                thread_id=thread_id,
                event_log=event_log,
                event_stream=event_stream,
                last_event_id=await event_log.latest_seq(thread_id),
            )

        last_event_id = _parse_last_event_id(request)
        if last_event_id == 0:
            last_event_id = await event_log.latest_seq(thread_id)

        await stateflow_agent.enqueue_approval_resume(
            thread_id=thread_id,
            history_dump=history_dump,
            approvals=approvals_dump,
        )
        return _build_sse_response(
            encoder=encoder,
            thread_id=thread_id,
            event_log=event_log,
            event_stream=event_stream,
            last_event_id=last_event_id,
        )

    # New turn path.
    body_messages = await _parse_body_messages(request)
    rows = await _sync_db_with_body(
        thread_id=thread_id,
        body_messages=body_messages,
        thread_repo=thread_repo,
        history_limit=history_limit,
    )

    # Extract prompt from the now-persisted history. Last row IS the
    # new user message (we just synced); if not user (somehow empty
    # body) bail.
    if not rows or rows[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="Cannot start run: thread has no user message to respond to.",
        )
    user_msg = rows[-1]
    prompt_text = extract_text(user_msg.parts)

    history = messages_to_model_history(rows, drop_prompt=prompt_text)
    history_dump = ModelMessagesTypeAdapter.dump_python(history, mode="json")

    # ── Last-Event-ID cutoff ────────────────────────────────────────────────
    # Snapshot ``latest_seq`` BEFORE enqueueing so the SSE consumer
    # doesn't replay every historical event for this thread when
    # ``Last-Event-ID`` is absent.
    last_event_id = _parse_last_event_id(request)
    if last_event_id == 0:
        last_event_id = await event_log.latest_seq(thread_id)

    try:
        await stateflow_agent.enqueue_run(
            thread_id=thread_id,
            user_message_id=user_msg.id,
            prompt=prompt_text,
            history_dump=history_dump,
        )
    except Exception as exc:  # pragma: no cover — DBOS errors wholesale
        _log.info(
            "enqueue_run returned %s for user_msg=%s — "
            "assuming attach-to-existing", type(exc).__name__, user_msg.id,
        )

    return _build_sse_response(
        encoder=encoder,
        thread_id=thread_id,
        event_log=event_log,
        event_stream=event_stream,
        last_event_id=last_event_id,
    )

# ── Module-level router ──────────────────────────────────────────────
#
# Resolves ``thread_repo`` / ``event_log`` / ``event_stream`` via
# ``Depends`` from ``app.state``. ``sf.create_app()`` mounts this
# router. Encoder is fixed to ``VercelAIWireEncoder`` and
# ``history_limit`` to the default (200).

from fastapi import Depends as _Depends  # noqa: E402

from pydantic_ai_stateflow.api.deps import (  # noqa: E402
    get_event_log as _get_event_log,
    get_event_stream as _get_event_stream,
    get_thread_repo as _get_thread_repo,
)

streaming_router = APIRouter()


@streaming_router.post("/threads/{thread_id}/messages")
@traced(
    TraceName.STREAMING_POST_MESSAGE,
    attrs=lambda _request, thread_id, **__: {
        "thread_id": str(thread_id),
    },
)
async def _post_message(
    request: Request,
    thread_id: UUID,
    thread_repo: ThreadRepository = _Depends(_get_thread_repo),
    event_log: EventLogRepository = _Depends(_get_event_log),
    event_stream: EventStream = _Depends(_get_event_stream),
) -> Response:
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter  # noqa: PLC0415

    _log.info("POST /threads/%s/messages received", thread_id)
    thread = await thread_repo.load(thread_id)
    if thread is None:
        _log.warning(
            "POST /threads/%s/messages → 404 (thread not found)",
            thread_id,
        )
        raise ThreadNotFound(thread_id=str(thread_id))

    stateflow_agent = _resolve_agent_from_app(request, thread.agent)

    if isinstance(stateflow_agent, StateflowDurableAgent):
        return await _durable_post_message(
            request=request,
            thread_id=thread_id,
            stateflow_agent=stateflow_agent,
            thread_repo=thread_repo,
            event_log=event_log,
            event_stream=event_stream,
            encoder=VercelAIWireEncoder(),
            history_limit=_DEFAULT_HISTORY_LIMIT,
        )

    # ── Non-durable path ────────────────────────────────────────
    body_messages = await _parse_body_messages(request)
    rows = await _sync_db_with_body(
        thread_id=thread_id,
        body_messages=body_messages,
        thread_repo=thread_repo,
        history_limit=_DEFAULT_HISTORY_LIMIT,
    )

    agent = stateflow_agent.agent
    model_settings = stateflow_agent.model_settings()

    adapter = await VercelAIAdapter.from_request(
        request, agent=agent, sdk_version=6,
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
            "post_message: deferred_tool_results present "
            "(approvals=%d, calls=%d) — skipping trim",
            len(deferred_results.approvals or {}),
            len(deferred_results.calls or {}),
        )

    history = messages_to_model_history(rows, drop_prompt=prompt_text)

    resolved_deps = await stateflow_agent.build_deps(
        thread=thread,
        message=last_message,
    )

    async def on_complete(result: AgentRunResult[Any]) -> None:
        from pydantic_ai import DeferredToolRequests  # noqa: PLC0415
        from pydantic_ai.ui.vercel_ai import (  # noqa: PLC0415
            VercelAIAdapter as _VercelAIAdapter,
        )

        output = result.output
        if isinstance(output, DeferredToolRequests):
            _log.info(
                "on_complete: agent paused with DeferredToolRequests "
                "(thread=%s) — skipping assistant persist",
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
            _log.warning(
                "on_complete: dump_messages returned no assistant "
                "parts (thread=%s output=%r) — nothing persisted",
                thread_id, output,
            )
            return

        persisted = await thread_repo.add_message(
            thread_id,
            role="assistant",
            parts=asst_parts,
        )
        _log.info(
            "on_complete: persisted assistant msg id=%s parts=%d "
            "(kinds=%s)",
            persisted.id, len(asst_parts),
            [p.get("type") for p in asst_parts],
        )

    return adapter.streaming_response(
        adapter.run_stream(
            message_history=history,
            deps=resolved_deps,
            model_settings=model_settings,
            on_complete=on_complete,
        ),
    )


@streaming_router.get("/threads/{thread_id}/events")
async def _thread_events(
    request: Request,
    thread_id: UUID,
    thread_repo: ThreadRepository = _Depends(_get_thread_repo),
    event_log: EventLogRepository = _Depends(_get_event_log),
) -> Response:
    """Long-lived SSE that tails the thread's event log."""
    thread = await thread_repo.load(thread_id)
    if thread is None:
        raise ThreadNotFound(thread_id=str(thread_id))

    last_event_id = _parse_last_event_id(request)
    if last_event_id == 0:
        last_event_id = await event_log.latest_seq(thread_id)

    async def _gen() -> AsyncIterator[bytes]:
        import asyncio  # noqa: PLC0415
        import json  # noqa: PLC0415

        yield b": connected\n\n"

        last_seq = last_event_id
        poll_interval_s = 0.25

        while True:
            if await request.is_disconnected():
                return
            events = await event_log.read_since(
                thread_id, after_seq=last_seq,
            )
            for ev in events:
                payload = json.dumps({
                    "kind": ev.kind,
                    "seq": ev.seq,
                    "payload": ev.payload,
                })
                yield (
                    f"id: {ev.seq}\ndata: {payload}\n\n".encode()
                )
                last_seq = ev.seq
            await asyncio.sleep(poll_interval_s)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@streaming_router.post("/threads/{thread_id}/cancel", status_code=200)
@traced(
    TraceName.STREAMING_POST_MESSAGE,
    attrs=lambda _request, thread_id, **__: {
        "thread_id": str(thread_id),
        "op": "cancel",
    },
)
async def _cancel_thread(
    request: Request,
    thread_id: UUID,
    thread_repo: ThreadRepository = _Depends(_get_thread_repo),
) -> dict[str, Any]:
    """Cancel every active workflow for ``thread_id``."""
    thread = await thread_repo.load(thread_id)
    if thread is None:
        raise ThreadNotFound(thread_id=str(thread_id))

    stateflow_agent = _resolve_agent_from_app(request, thread.agent)
    if not isinstance(stateflow_agent, StateflowDurableAgent):
        raise HTTPException(
            status_code=400,
            detail=(
                "cancel is only meaningful for StateflowDurableAgent "
                "threads; non-durable agents don't have cancellable "
                "workflows"
            ),
        )
    cancelled = await stateflow_agent.cancel_thread_runs(thread_id)
    return {"cancelled": cancelled}


__all__ = [
    "DepsFactory",
    "extract_text",
    "messages_to_model_history",
    "streaming_router",
]

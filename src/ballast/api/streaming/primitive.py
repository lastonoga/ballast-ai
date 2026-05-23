"""``stream_response`` primitive — apps wire their own streaming routes.

Replaces the framework-owned ``POST /threads/{id}/messages`` route.
Apps write their own FastAPI handlers that resolve the right agent
instance (their concern — opaque ``Thread.agent`` string lookup) and
delegate to ``stream_response(...)`` for the heavy lifting:

  - body-vs-DB sync (edit / regenerate as truncate-then-append)
  - durable vs inline streaming dispatch (durable path requires
    ``DurableAgent``)
  - Vercel-AI wire encoding via ``VercelAIAdapter``
  - approval-resume detection + routing
  - assistant-turn persistence (non-durable path)

The companion ``cancel_thread_workflows`` primitive cancels every
active workflow for a thread (durable path only).

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

Vercel AI SDK v6 is targeted (``sdk_version=6``) so that
``@agent.tool(requires_approval=True)`` produces ``approval-requested``
UI parts on the wire and incoming approval responses are extracted by
``VercelAIAdapter.deferred_tool_results``.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from fastapi import Request
from starlette.responses import StreamingResponse

from ballast.api.streaming.history import (
    extract_text,
    messages_to_model_history,
)
from ballast.api.streaming.wire_encoder import (
    VercelAIWireEncoder,
    WireEncoder,
)
from ballast.errors import (
    CancelNotSupported,
    EmptyMessageBody,
    ThreadNotFound,
)
from ballast.logging import get_logger
from ballast.persistence.events.repository import EventLogRepository
from ballast.persistence.thread.repository import ThreadRepository
from ballast.runtime.durable_agent import DurableAgent
from ballast.runtime.event_stream import EventStream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.responses import Response

    from ballast.persistence.thread.domain import Message
    from ballast.runtime.agents import BallastAgent


_log = get_logger(__name__)

DepsT = TypeVar("DepsT")
OutT = TypeVar("OutT")

DepsFactory = Callable[..., Any] | Callable[..., Awaitable[Any]]
"""Retained for backwards compatibility of the public type name.

Apps register a ``BallastAgent`` whose ``build_deps`` method serves
the same role.
"""

_DEFAULT_HISTORY_LIMIT = 200


# ── Approval-resume helpers ────────────────────────────────────────────


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


# ── Body / DB sync ─────────────────────────────────────────────────────


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
) -> "list[Message]":
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


# ── SSE response ───────────────────────────────────────────────────────


_IDLE_TIMEOUT_S = 30.0


def _build_sse_response(
    *,
    encoder: WireEncoder,
    thread_id: UUID,
    event_log: EventLogRepository,
    event_stream: EventStream,
    last_event_id: int,
) -> "Response":
    """Build the SSE StreamingResponse that tails the event log.

    Subscribes to the ``event_stream`` notification channel so the
    generator wakes on new events instead of busy-polling. Each
    notification triggers a ``read_since(cursor)`` from the durable
    log — the stream is best-effort (may lose / duplicate / reorder
    notifications), the log is the source of truth.

    Terminates on:
      - ``done`` / ``cancelled`` event emitted by the workflow
      - ~30s with no new events (defensive close; the SSE consumer
        is expected to reconnect with ``Last-Event-ID``)
    """
    from ballast.runtime.event_stream import thread_channel  # noqa: PLC0415

    async def _gen() -> "AsyncIterator[bytes]":
        import asyncio  # noqa: PLC0415

        try:
            for chunk in encoder.initial_events(thread_id=thread_id):
                yield chunk

            cursor = last_event_id

            # Replay anything between ``last_event_id`` and now so a
            # subscriber that hands us a non-zero cursor doesn't miss
            # events emitted before ``subscribe()`` was wired up.
            for ev in await event_log.read_since(thread_id, after_seq=cursor):
                for chunk in encoder.encode_event(ev):
                    yield chunk
                cursor = ev.seq
                if ev.kind in {"done", "cancelled"}:
                    for chunk in encoder.finalize():
                        yield chunk
                    return

            # Subscribe + tail live. The reader task pumps wake-ups
            # into a local queue so the main loop can ``wait_for`` with
            # a defensive idle timeout without cancelling the
            # underlying subscribe generator (which would close it for
            # good).
            async with event_stream.subscribe(
                thread_channel(thread_id),
            ) as notifications:
                local: asyncio.Queue[None] = asyncio.Queue()

                async def reader() -> None:
                    try:
                        async for _ in notifications:
                            await local.put(None)
                    except asyncio.CancelledError:
                        pass

                reader_task = asyncio.create_task(reader())
                try:
                    while True:
                        try:
                            await asyncio.wait_for(
                                local.get(), timeout=_IDLE_TIMEOUT_S,
                            )
                        except asyncio.TimeoutError:
                            _log.warning(
                                "Durable stream idle for %.0fs on thread "
                                "%s (cursor=%d) — closing",
                                _IDLE_TIMEOUT_S, thread_id, cursor,
                            )
                            for chunk in encoder.finalize():
                                yield chunk
                            return

                        fresh = await event_log.read_since(
                            thread_id, after_seq=cursor,
                        )
                        for ev in fresh:
                            for chunk in encoder.encode_event(ev):
                                yield chunk
                            cursor = ev.seq
                            if ev.kind in {"done", "cancelled"}:
                                for chunk in encoder.finalize():
                                    yield chunk
                                return
                finally:
                    reader_task.cancel()
                    import contextlib  # noqa: PLC0415
                    with contextlib.suppress(asyncio.CancelledError):
                        await reader_task
        except Exception:
            # BaseHTTPMiddleware can't see exceptions raised inside a
            # StreamingResponse body — log here so failures don't vanish
            # into an empty SSE stream.
            _log.exception(
                "Durable SSE stream failed on thread %s", thread_id,
            )
            raise

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


# ── Durable dispatch ───────────────────────────────────────────────────


async def _durable_post_message(
    *,
    request: Request,
    thread_id: UUID,
    stateflow_agent: DurableAgent,
    thread_repo: ThreadRepository,
    event_log: EventLogRepository,
    event_stream: EventStream,
    encoder: WireEncoder,
    history_limit: int,
) -> "Response":
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

    adapter = await VercelAIAdapter.from_request(
        request, agent=stateflow_agent.agent, sdk_version=6,
    )
    deferred = adapter.deferred_tool_results

    if deferred is not None and deferred.approvals:
        history = list(adapter.messages)
        history_dump = ModelMessagesTypeAdapter.dump_python(
            history, mode="json",
        )

        pending = _pending_tool_call_ids(history)
        approvals_dump: dict[str, bool | dict[str, Any]] = {}
        for tcid, decision in deferred.approvals.items():
            if tcid not in pending:
                continue
            if isinstance(decision, bool):
                approvals_dump[tcid] = decision
            else:
                approvals_dump[tcid] = {
                    "message": getattr(decision, "message", "denied"),
                }

        if not approvals_dump:
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

    if not rows or rows[-1].role != "user":
        raise EmptyMessageBody(
            "Cannot start run: thread has no user message to respond to.",
            hint="POST /threads/{id}/messages with a user message first.",
        )
    user_msg = rows[-1]
    prompt_text = extract_text(user_msg.parts)

    history = messages_to_model_history(rows, drop_prompt=prompt_text)
    history_dump = ModelMessagesTypeAdapter.dump_python(history, mode="json")

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
    except Exception:
        # DBOS raises when a workflow_id already exists (idempotent
        # re-POST → attach to the running workflow). That's expected.
        # Everything else gets logged with traceback so real bugs (DB
        # schema, serialization, etc.) don't vanish behind this catch.
        _log.warning(
            "enqueue_run raised for user_msg=%s — assuming "
            "attach-to-existing; traceback below",
            user_msg.id, exc_info=True,
        )

    return _build_sse_response(
        encoder=encoder,
        thread_id=thread_id,
        event_log=event_log,
        event_stream=event_stream,
        last_event_id=last_event_id,
    )


# ── Public primitives ──────────────────────────────────────────────────


async def stream_response(
    *,
    request: "Request",
    thread_id: UUID,
    agent: "BallastAgent",
    history_limit: int = _DEFAULT_HISTORY_LIMIT,
) -> "Response":
    """Body-vs-DB sync + agent run + Vercel-AI streaming response.

    The framework's streaming primitive. Apps that own their own
    streaming route resolve the agent for a thread (their concern —
    opaque ``Thread.agent`` string lookup) and delegate to this.

    Dispatches durable vs inline based on ``isinstance(agent,
    DurableAgent)`` — durable path enqueues a DBOS workflow
    and tails the event log; inline path runs the pydantic-ai Agent
    and streams via ``VercelAIAdapter``.
    """
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter  # noqa: PLC0415

    from ballast.runtime.engine import get_ballast  # noqa: PLC0415
    engine = get_ballast()
    thread_repo = engine.thread_repo
    event_log = engine.event_log
    event_stream = engine.event_stream

    thread = await thread_repo.load(thread_id)
    if thread is None:
        raise ThreadNotFound(thread_id=str(thread_id))

    if isinstance(agent, DurableAgent):
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
        # ``silent=True`` — the assistant turn already streamed live
        # through the Vercel-AI SSE channel; an additional
        # ``message-added`` event log row from the default signal
        # handler would be redundant for any consumer that watched the
        # stream. The persistence write here is only for replay.
        await thread_repo.add_message(
            thread_id,
            role="assistant",
            parts=asst_parts,
            silent=True,
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
    agent: "BallastAgent",
) -> int:
    """Cancel every active workflow for ``thread_id``.

    Only meaningful for ``DurableAgent`` instances —
    non-durable agents don't have cancellable workflows. Raises
    ``CancelNotSupported`` for non-durable agents.

    Returns the count of workflows that were cancelled.
    """
    if not isinstance(agent, DurableAgent):
        raise CancelNotSupported(
            "cancel is only meaningful for DurableAgent "
            "threads; non-durable agents don't have cancellable workflows",
            context={"thread_id": str(thread_id)},
        )
    return await agent.cancel_thread_runs(thread_id)


__all__ = [
    "DepsFactory",
    "cancel_thread_workflows",
    "extract_text",
    "messages_to_model_history",
    "stream_response",
]

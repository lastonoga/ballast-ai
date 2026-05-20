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
from pydantic_ai_stateflow.runtime.agents import get_agent
from pydantic_ai_stateflow.runtime.durable_agent import StateflowDurableAgent
from pydantic_ai_stateflow.runtime.event_stream import EventStream

_log = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.agent import AgentRunResult
    from pydantic_ai.messages import ModelMessage
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
            parts=[p for p in parts if isinstance(p, dict)],
        )

    return await thread_repo.history(thread_id, limit=history_limit)


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
    """Durable path: sync DB with body, enqueue workflow, tail event log."""
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

    from pydantic_ai.messages import ModelMessagesTypeAdapter  # noqa: PLC0415

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

    _ = event_stream  # reserved for future live-signal wiring
    return StreamingResponse(_gen(), media_type=encoder.content_type())


EncoderFactory = Callable[[], WireEncoder]
"""Producer of a per-request encoder instance.

Defaults to ``VercelAIWireEncoder``. Apps swap in their own factory
to support AG-UI / A2A / custom wire formats — the factory pattern
gives each SSE response a fresh encoder instance so per-stream state
(e.g. text-delta accumulation) doesn't leak across requests.
"""


def build_streaming_router(
    *,
    thread_repo: ThreadRepository,
    event_log: EventLogRepository | None = None,
    event_stream: EventStream | None = None,
    encoder_factory: EncoderFactory | None = None,
    prefix: str = "",
    history_limit: int = _DEFAULT_HISTORY_LIMIT,
) -> APIRouter:
    """Mount ``POST {prefix}/threads/{id}/messages`` as a Vercel-AI stream.

    Single-endpoint contract: receives the full Vercel-AI ``messages``
    array, syncs it against the DB (truncate-then-append on the divergent
    suffix), runs the agent, streams events, persists the assistant
    turn at the end.

    The thread MUST already exist (404 otherwise — no lazy-create).
    Its ``agent`` field is the registry key for a ``StateflowAgent``
    instance the app registered at startup.

    Args:
      thread_repo: source of truth for thread + message persistence.
      prefix: optional router prefix.
      history_limit: cap on the number of rows hydrated from the repo
        per request (default 200).
    """
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter  # noqa: PLC0415

    router = APIRouter(prefix=prefix)
    _encoder_factory: EncoderFactory = encoder_factory or VercelAIWireEncoder

    @router.post("/threads/{thread_id}/messages")
    @traced(
        TraceName.STREAMING_POST_MESSAGE,
        attrs=lambda _request, thread_id, **__: {
            "thread_id": str(thread_id),
        },
    )
    async def post_message(
        request: Request,
        thread_id: UUID,
    ) -> Response:
        _log.info("POST /threads/%s/messages received", thread_id)
        thread = await thread_repo.load(thread_id)
        if thread is None:
            _log.warning(
                "POST /threads/%s/messages → 404 (thread not found)",
                thread_id,
            )
            raise HTTPException(status_code=404, detail="thread not found")

        stateflow_agent = get_agent(thread.agent)

        if isinstance(stateflow_agent, StateflowDurableAgent):
            if event_log is None or event_stream is None:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Durable agent requires event_log + event_stream "
                        "wired into build_streaming_router(...)"
                    ),
                )
            return await _durable_post_message(
                request=request,
                thread_id=thread_id,
                stateflow_agent=stateflow_agent,
                thread_repo=thread_repo,
                event_log=event_log,
                event_stream=event_stream,
                encoder=_encoder_factory(),
                history_limit=history_limit,
            )

        # ── Non-durable path ────────────────────────────────────────
        body_messages = await _parse_body_messages(request)
        rows = await _sync_db_with_body(
            thread_id=thread_id,
            body_messages=body_messages,
            thread_repo=thread_repo,
            history_limit=history_limit,
        )

        agent = stateflow_agent.agent
        model_settings = stateflow_agent.model_settings()

        adapter = await VercelAIAdapter.from_request(
            request, agent=agent, sdk_version=6,
        )

        last_message = adapter.messages[-1] if adapter.messages else None
        # Source of truth post-sync is the repo. Use the trailing user
        # row's text as the prompt — ``adapter.messages`` may coalesce
        # consecutive user UIMessages into one ModelRequest with multiple
        # UserPromptParts, so reading a "prompt" off it would either
        # concatenate them (wrong) or pick an arbitrary one.
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

        # The adapter's trimmed last user turn will be appended to
        # ``message_history`` by ``UIAdapter.run_stream``, so drop the
        # last user row from our repo-built history to avoid duplication.
        history = messages_to_model_history(rows, drop_prompt=prompt_text)

        resolved_deps = await stateflow_agent.build_deps(
            thread=thread,
            message=last_message,
        )

        async def on_complete(result: AgentRunResult[Any]) -> None:
            """Persist the assistant reply on successful completion.

            Skips when the run paused with ``DeferredToolRequests`` —
            that's a HITL pause, not a final turn. The next request
            (with approval responses) emits the real assistant.
            """
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

    @router.post("/threads/{thread_id}/cancel", status_code=200)
    @traced(
        TraceName.STREAMING_POST_MESSAGE,
        attrs=lambda _request, thread_id, **__: {
            "thread_id": str(thread_id),
            "op": "cancel",
        },
    )
    async def cancel_thread(
        request: Request,
        thread_id: UUID,
    ) -> dict[str, Any]:
        """Cancel every active workflow for ``thread_id``.

        Idempotent: already-finished workflows are no-ops. Emits a
        synthetic ``cancelled`` event so attached SSE consumers close
        their stream cleanly.
        """
        del request
        thread = await thread_repo.load(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")

        stateflow_agent = get_agent(thread.agent)
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

    return router


__all__ = [
    "DepsFactory",
    "build_streaming_router",
    "extract_text",
    "messages_to_model_history",
]

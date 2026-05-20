"""Server-stateful Vercel-AI streaming endpoint.

The streaming router is a **pure run-trigger**: it does NOT persist
incoming user messages. The frontend's canonical
``ThreadHistoryAdapter.append`` flow (``POST /threads/{id}/messages``)
is the sole persistence path for user input — assistant-ui tracks the
message tree client-side and POSTs each new node with the right
``parent_id`` BEFORE calling ``/runs``. The streaming router then:

  1. Loads the active branch from ``thread_repo`` (source of truth).
  2. Identifies the new user msg by its client id (matched against
     the body's last user UIMessage) — if it isn't in the repo yet
     (direct-curl / legacy client), auto-persist it.
  3. Triggers the agent run; assistant turn is persisted at the end
     of the run (durable: via ``_persist_assistant_turn`` step;
     non-durable: via the ``on_complete`` callback).

Vercel AI SDK v6 is targeted (``sdk_version=6``) so that
``@agent.tool(requires_approval=True)`` produces ``approval-requested`` UI
parts on the wire and incoming approval responses are extracted by
``VercelAIAdapter.deferred_tool_results``.

Endpoint contract::

    POST {prefix}/threads/{thread_id}/runs
        Accept: text/event-stream
        Body  : Vercel AI ``RequestData`` JSON (parsed by VercelAIAdapter)
        404   : thread not found (no lazy-create)
        200   : streaming Vercel AI events
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID, uuid4

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


DepsT = TypeVar("DepsT")
OutT = TypeVar("OutT")

DepsFactory = Callable[..., Any] | Callable[..., Awaitable[Any]]
"""Retained for backwards compatibility of the public type name.

The streaming router itself no longer takes a ``deps_factory`` — apps
register a ``StateflowAgent`` whose ``build_deps`` method serves the
same role.
"""


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
) -> str | None:
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


async def _last_user_msg_from_body(
    request: Request,
) -> tuple[str | None, str]:
    """Return ``(client_id, text)`` for the last user UIMessage in the body.

    ``client_id`` is the frontend-supplied message id (free-form
    string — assistant-ui ships short random ids like
    ``"MbPSd9jddGfC6UAV"``). ``text`` is the concatenated text of
    every text part on that user UIMessage.

    Walks the RAW body's ``messages`` array — NOT
    ``adapter.messages`` — because the adapter coalesces consecutive
    user messages (with no assistant between) into a single
    ``ModelRequest`` with multiple ``UserPromptPart`` entries. That
    coalescing is fine for prompting the model with the full unbroken
    user input, but it would corrupt our "this is the NEW user turn"
    extraction: an orphaned user message from a cancelled prior turn
    would get concatenated with the new text and we'd persist the
    join (``"а тыты"`` instead of just ``"ты"``).

    The raw body's ``messages`` array preserves one entry per
    UIMessage. The last role="user" entry is, by frontend convention
    (Vercel ``useChat.sendMessage``), the user's just-typed turn.
    Returns ``(None, "")`` when no user message exists in the body.
    """
    body = await request.json()
    messages = body.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        raw_id = msg.get("id")
        client_id = raw_id if isinstance(raw_id, str) else None
        chunks: list[str] = []
        for part in msg.get("parts") or []:
            if part.get("type") == "text":
                txt = part.get("text")
                if isinstance(txt, str):
                    chunks.append(txt)
        return client_id, "".join(chunks)
    return None, ""


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


async def _resolve_user_msg_id(
    *,
    thread_id: UUID,
    thread_repo: ThreadRepository,
    history_before: list[Any],
    client_id: str | None,
    prompt_text: str,
) -> str:
    """Find or persist the user message that anchors the new run.

    Canonical path (assistant-ui): the frontend already POSTed the new
    user UIMessage to ``/threads/{id}/messages`` before triggering the
    run. We look it up by ``client_id`` — found → return its id.

    Fallback (direct-curl / legacy clients): no row matches ``client_id``,
    so we auto-persist a new user message under the last active-branch
    msg as parent. Empty ``prompt_text`` falls back to the latest user
    in the active branch (resume-after-approval case where the body
    carries an empty user prompt).
    """
    if client_id is not None:
        all_msgs = await thread_repo.all_messages(thread_id)
        existing = next((m for m in all_msgs if m.id == client_id), None)
        if existing is not None:
            return existing.id

    if not prompt_text:
        # Approval-resume / replay: no fresh user turn — anchor on the
        # latest user in the active branch so the assistant becomes its
        # next sibling-or-child.
        last_user_id = _find_last_role_id(history_before, "user")
        if last_user_id is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Cannot start run: body has no user prompt and no "
                    "prior user message exists in the thread."
                ),
            )
        return last_user_id

    new_user_parent_id = history_before[-1].id if history_before else None
    user_msg = await thread_repo.add_message_with_id(
        thread_id,
        id=client_id or str(uuid4()),
        role="user",
        # ``state: "done"`` matches Vercel UIMessage's TextUIPart for
        # a fully-rendered user turn. On restore via ``chat.setMessages``,
        # parts without ``state`` slip through useChat's discriminator
        # into a different branch and break ``MessagePartText`` rendering.
        parts=[{"type": "text", "text": prompt_text, "state": "done"}],
        parent_id=new_user_parent_id,
    )
    _log.info(
        "submit-message: auto-persisted user msg id=%s parent=%s "
        "(no matching client_id in repo — direct-curl fallback)",
        user_msg.id, new_user_parent_id,
    )
    return user_msg.id


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
    """Pure run-trigger — fire the agent workflow, then tail the event log.

    Reads the thread's active branch from ``thread_repo`` (source of
    truth — the frontend's canonical
    ``ThreadHistoryAdapter.append`` flow has already persisted the new
    user msg via ``POST /threads/{id}/messages``). Starts
    ``stateflow_agent.run`` with a deterministic workflow id keyed off
    the user message id — a request retry reuses the same workflow
    instead of spawning a duplicate. Streams events from the durable
    log (replay from Last-Event-ID then live tail), terminating on
    ``done`` / ``cancelled``.

    SSE chunks come from the ``encoder`` (default VercelAI v6); swap
    via ``encoder_factory`` for AG-UI / A2A / custom wire formats.
    """
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter  # noqa: PLC0415

    # Parse the Vercel-AI body purely to read ``trigger`` and the new
    # user message id — NOT to persist a message.
    adapter = await VercelAIAdapter.from_request(
        request, agent=stateflow_agent.agent, sdk_version=6,
    )

    trigger = getattr(adapter.run_input, "trigger", "submit-message")
    is_regenerate = trigger == "regenerate-message"

    history_before = await thread_repo.history(thread_id, limit=history_limit)

    if is_regenerate:
        # Find the last user msg in the active branch — that's the
        # parent of the new (regenerated) assistant turn.
        last_user_id = _find_last_role_id(history_before, "user")
        if last_user_id is None:
            raise HTTPException(
                status_code=400,
                detail="regenerate-message with no prior user message",
            )
        assistant_parent_id: str = last_user_id
        user_msg_id: str = last_user_id
        prompt_text = ""  # don't re-derive — repo's last user IS the prompt
        _log.info(
            "regenerate-message: new assistant will sibling under user_id=%s",
            last_user_id,
        )
    else:
        client_id, prompt_text = await _last_user_msg_from_body(request)
        user_msg_id = await _resolve_user_msg_id(
            thread_id=thread_id,
            thread_repo=thread_repo,
            history_before=history_before,
            client_id=client_id,
            prompt_text=prompt_text,
        )
        assistant_parent_id = user_msg_id

    # Build history dump for the workflow. ``ModelMessage`` is a
    # TypeAdapter-backed discriminated union (not a BaseModel subclass),
    # so per-instance ``model_dump`` doesn't exist — round-trip the whole
    # list via ``ModelMessagesTypeAdapter`` instead.
    from pydantic_ai.messages import ModelMessagesTypeAdapter  # noqa: PLC0415

    rows = await thread_repo.history(thread_id, limit=history_limit)
    # Determine the prompt text for the agent run from the repo's
    # current active branch (the user msg we just resolved is the
    # latest). For regenerate, the last user in the branch IS the
    # prompt; for submit-message, it's the row we just verified /
    # persisted.
    if not prompt_text:
        prompt_text = (
            extract_text(rows[-1].parts)
            if rows and rows[-1].role == "user"
            else ""
        )
    history = messages_to_model_history(rows, drop_prompt=prompt_text)
    history_dump = ModelMessagesTypeAdapter.dump_python(history, mode="json")

    # ── Last-Event-ID cutoff ────────────────────────────────────────────────
    #
    # If the client sent ``Last-Event-ID`` (SSE reconnect), honor it —
    # they want to catch up on events they missed since that seq.
    #
    # Otherwise (fresh POST) we MUST NOT replay every historical event
    # for this thread — the event_log holds the full event history
    # across every prior workflow run, so without a cutoff the SSE
    # consumer would replay the previous turn's "text-delta" + "done"
    # events first (closing the stream on stale content) before ever
    # seeing the new workflow's output.
    #
    # Snapshot ``latest_seq`` BEFORE enqueueing so we don't race the
    # workflow into emitting events between our snapshot and our
    # enqueue call.
    last_event_id = _parse_last_event_id(request)
    if last_event_id == 0:
        last_event_id = await event_log.latest_seq(thread_id)

    # Enqueue into the per-thread serialization queue (concurrency=1
    # per thread). Workflow id is deterministic per (thread_id,
    # user_msg.id) so a request retry attaches to the existing run
    # instead of spawning a duplicate. Concurrent messages for the
    # same thread queue up and run serially.
    try:
        await stateflow_agent.enqueue_run(
            thread_id=thread_id,
            user_message_id=user_msg_id,
            prompt=prompt_text,
            history_dump=history_dump,
            # Parent of the assistant turn we're about to generate:
            # - submit-message → the user message we just resolved
            # - regenerate-message → the EXISTING last user message
            #   (so the new assistant becomes a sibling of the prior)
            assistant_parent_id=assistant_parent_id,
        )
    except Exception as exc:  # pragma: no cover — DBOS errors caught wholesale
        # Enqueue raises if the workflow id collides with one in a
        # terminal state. For our deterministic key case the typical
        # "collision" outcome is "already enqueued/running" — which
        # is what we want; the SSE stream will tail the existing
        # workflow's events. Log and continue.
        _log.info(
            "enqueue_run returned %s for user_msg=%s — "
            "assuming attach-to-existing", type(exc).__name__, user_msg_id,
        )

    async def _gen() -> AsyncIterator[bytes]:
        import asyncio  # noqa: PLC0415

        for chunk in encoder.initial_events(thread_id=thread_id):
            yield chunk

        last_seq = last_event_id

        # Poll the durable log + tail the live signal channel. Polling
        # is the safety net: ``EventStream`` notifications are
        # best-effort and may not reach a different event loop / thread
        # (DBOS workflow runs in its own asyncio task). The log read is
        # the source of truth — we always re-read from it after each
        # wake-up, never trust the notification payload alone.
        poll_interval_s = 0.05
        idle_iterations = 0
        # Cap idle wait at ~30s of consecutive empty polls — beyond
        # that the workflow is almost certainly stuck / cancelled and
        # we emit a stream-level error instead of hanging the client.
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

    # ``event_stream`` is reserved for future live-signal optimization
    # (avoid polling entirely on the same-loop case). Kept in the
    # signature so apps can wire it now — current implementation just
    # polls the log.
    _ = event_stream

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
    """Mount ``POST {prefix}/threads/{id}/runs`` as a Vercel-AI stream.

    Pure run-trigger contract: the user message has already been
    persisted via the canonical ``POST /threads/{id}/messages`` flow
    (assistant-ui's ``ThreadHistoryAdapter.append``). The thread MUST
    already exist (404 otherwise — no lazy-create). Its ``agent`` field
    is the registry key for a ``StateflowAgent`` instance the app
    registered at startup; the framework resolves that instance via
    ``pydantic_ai_stateflow.runtime.agents.get_agent`` and uses its
    ``agent`` (pydantic-ai ``Agent``), ``build_deps(...)``, and
    ``model_settings()`` to drive the run.

    For direct-curl / legacy clients that bypass the canonical append
    flow, the router auto-persists the body's last user UIMessage if
    no row with its client id exists in the repo — keeps the endpoint
    usable from raw HTTP.

    After the agent completes the assistant reply is persisted via an
    ``on_complete`` callback (non-durable) or a ``@DBOS.step``
    (durable).

    Tool-approval responses (Vercel AI SDK v6 ``approval-responded``
    parts) are extracted by ``VercelAIAdapter.deferred_tool_results``
    and threaded into ``run_stream`` so
    ``@agent.tool(requires_approval=True)`` tools resume after the user
    clicks Approve/Cancel.

    Args:
      thread_repo: source of truth for thread + message persistence.
      prefix: optional router prefix.
      history_limit: cap on the number of rows hydrated from the repo
        per request (default 200).
    """
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter  # noqa: PLC0415

    router = APIRouter(prefix=prefix)
    _encoder_factory: EncoderFactory = encoder_factory or VercelAIWireEncoder

    @router.post("/threads/{thread_id}/runs")
    @traced(
        TraceName.STREAMING_POST_MESSAGE,
        attrs=lambda _request, thread_id, **__: {
            "thread_id": str(thread_id),
        },
    )
    async def post_run(
        request: Request,
        thread_id: UUID,
    ) -> Response:
        _log.info("POST /threads/%s/runs received", thread_id)
        thread = await thread_repo.load(thread_id)
        if thread is None:
            _log.warning(
                "POST /threads/%s/runs → 404 (thread not found)",
                thread_id,
            )
            raise HTTPException(status_code=404, detail="thread not found")

        stateflow_agent = get_agent(thread.agent)

        # Durable path: fire the run as a @DBOS.workflow, then stream
        # events from the durable log + signal channel back to the
        # client. Survives caller cancellation / process restart /
        # reconnect via Last-Event-ID.
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

        agent = stateflow_agent.agent
        model_settings = stateflow_agent.model_settings()

        adapter = await VercelAIAdapter.from_request(
            request, agent=agent, sdk_version=6,
        )

        last_message = adapter.messages[-1] if adapter.messages else None
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
        else:
            _log.info(
                "post_run: deferred_tool_results present "
                "(approvals=%d, calls=%d) — skipping trim",
                len(deferred_results.approvals or {}),
                len(deferred_results.calls or {}),
            )

        history_before = await thread_repo.history(
            thread_id, limit=history_limit,
        )

        if is_regenerate:
            # Regenerate semantics: the assistant being regenerated is
            # REPLACED with a sibling whose parent is the same user
            # turn the old assistant was replying to. The new assistant
            # becomes a sibling of the prior one.
            last_user_id = _find_last_role_id(history_before, "user")
            if last_user_id is None:
                raise HTTPException(
                    status_code=400,
                    detail="regenerate-message with no prior user message",
                )
            new_assistant_parent_id: str | None = last_user_id
            prompt_text = ""
        else:
            client_id, prompt_text = await _last_user_msg_from_body(request)
            user_msg_id = await _resolve_user_msg_id(
                thread_id=thread_id,
                thread_repo=thread_repo,
                history_before=history_before,
                client_id=client_id,
                prompt_text=prompt_text,
            )
            new_assistant_parent_id = user_msg_id

        rows = await thread_repo.history(thread_id, limit=history_limit)
        # Pick the prompt text from the resolved user msg in the repo —
        # for regenerate, that's the last user in the branch; for
        # submit-message, it's the row we just verified/persisted.
        if not prompt_text:
            prompt_text = (
                extract_text(rows[-1].parts)
                if rows and rows[-1].role == "user"
                else ""
            )
        # The adapter will append its trimmed last user turn to
        # ``message_history`` again; drop it from our repo-driven
        # history to avoid duplication.
        history = messages_to_model_history(rows, drop_prompt=prompt_text)

        resolved_deps = await stateflow_agent.build_deps(
            thread=thread,
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

            # Persist the FULL assistant turn — text + reasoning +
            # tool-call/tool-result parts — using
            # ``VercelAIAdapter.dump_messages`` to convert pydantic-ai's
            # ``ModelResponse`` back into the same Vercel UIMessage parts
            # the frontend rendered live. That way page reload restores
            # reasoning chains, tool-call cards, and approval outcomes
            # exactly as they appeared during the streaming run.
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
                    # ``exclude_none=True`` strips explicit nulls
                    # (``providerMetadata``, ``preliminary``, ``approval``
                    # …) so the persisted JSON matches the wire shape
                    # useChat already parses on the live stream.
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
                parent_id=new_assistant_parent_id,
            )
            _log.info(
                "on_complete: persisted assistant msg id=%s parent=%s "
                "parts=%d (kinds=%s)",
                persisted.id, new_assistant_parent_id, len(asst_parts),
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

        Per Q1 from the design discussion: cancels the WHOLE queue for
        the thread (current + queued messages), not just the in-flight
        one. Per Q5: idempotent — already-finished workflows are
        no-ops.

        The SSE consumer attached to this thread sees a
        ``kind="cancelled"`` event land in the event log and closes
        the stream. The WireEncoder decides how that maps to the wire
        (default ``VercelAIWireEncoder`` emits an ``error`` + ``finish``
        pair since Vercel AI SDK v6 has no native abort event).
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

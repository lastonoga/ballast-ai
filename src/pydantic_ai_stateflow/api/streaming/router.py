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
from pydantic_ai_stateflow.runtime.event_stream import (
    EventStream,
    thread_channel,
)

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


async def _last_user_text_from_body(request: Request) -> str:
    """Return the text of the LAST user UIMessage in the request body.

    Walks the raw Vercel-AI body — NOT ``adapter.messages`` — because
    the adapter coalesces consecutive user messages (with no assistant
    between) into a single ``ModelRequest`` with multiple
    ``UserPromptPart`` entries. That coalescing is fine for prompting
    the model with the full unbroken user input, but it corrupts our
    "this is the NEW user turn we just received" extraction: an
    orphaned user message from a cancelled prior turn would get
    concatenated with the new text and we'd persist the join (``"а
    тыты"`` instead of just ``"ты"``).

    The raw body's ``messages`` array preserves one entry per
    UIMessage. The last role="user" entry is, by frontend convention
    (Vercel ``useChat.sendMessage``), the user's just-typed turn.
    """
    body = await request.json()
    messages = body.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        chunks: list[str] = []
        for part in msg.get("parts") or []:
            if part.get("type") == "text":
                txt = part.get("text")
                if isinstance(txt, str):
                    chunks.append(txt)
        if chunks:
            return "".join(chunks)
    return ""


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
    """Durable streaming branch — starts the workflow then tails the event log.

    Flow:
      1. Parse the request body to pull out the new user prompt
         (mirrors the non-durable path's body shape so frontends use
         the same client SDK).
      2. Persist the user message to ``thread_repo`` BEFORE the
         workflow starts. Same idempotency contract as the existing
         path — a request retry with the same body produces the same
         persisted user row.
      3. Build a JSON-friendly ``history_dump`` from the active branch
         (excluding the just-persisted prompt — pydantic-ai re-derives
         it from the workflow argument).
      4. Start ``stateflow_agent.run`` as a DBOS workflow with a
         deterministic ``workflow_id`` keyed off the user message id —
         a refresh / retry from the client reuses the same workflow
         instead of spawning a duplicate.
      5. Stream events from the durable log to the client: replay
         everything ``> Last-Event-ID``, then subscribe to live
         notifications and keep emitting until a ``done`` event lands.

    SSE chunks come from the ``encoder`` (default VercelAI v6); swap
    via ``encoder_factory`` for AG-UI / A2A / custom wire formats.
    """
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter  # noqa: PLC0415

    # The Vercel-AI body parser is the easiest way to extract the
    # incoming user prompt + message id without re-implementing the
    # client-side message shape. We don't drive ``adapter.run_stream``
    # in this path — the workflow does that internally.
    adapter = await VercelAIAdapter.from_request(
        request, agent=stateflow_agent.agent, sdk_version=6,
    )
    # IMPORTANT: pull the new prompt from the RAW body's last user
    # UIMessage — NOT from ``adapter.messages``. ``VercelAIAdapter``
    # coalesces consecutive user messages (no assistant between them)
    # into a single ``ModelRequest`` with multiple ``UserPromptPart``
    # entries. That happens whenever a previous turn was cancelled
    # mid-flight: the orphaned user message stays in the frontend's
    # local state, so the next send carries BOTH the orphan and the
    # new text. ``_last_user_text`` would then return the joined text
    # ("a tyты" instead of just "ты"), corrupting the new prompt.
    prompt_text = await _last_user_text_from_body(request)
    if not prompt_text:
        raise HTTPException(
            status_code=400,
            detail="Durable agent requires a non-empty user prompt",
        )

    # Persist user message (same shape as non-durable path).
    history_before = await thread_repo.history(thread_id, limit=history_limit)
    new_user_parent_id = history_before[-1].id if history_before else None
    user_msg = await thread_repo.add_message(
        thread_id,
        role="user",
        parts=[{"type": "text", "text": prompt_text, "state": "done"}],
        parent_id=new_user_parent_id,
    )

    # Build history dump. ``ModelMessage`` is a TypeAdapter-backed
    # discriminated union (not a BaseModel subclass), so per-instance
    # ``model_dump`` doesn't exist — round-trip the whole list via
    # ``ModelMessagesTypeAdapter`` instead.
    from pydantic_ai.messages import ModelMessagesTypeAdapter  # noqa: PLC0415

    rows = await thread_repo.history(thread_id, limit=history_limit)
    history = messages_to_model_history(rows, drop_prompt=prompt_text)
    history_dump = ModelMessagesTypeAdapter.dump_python(history, mode="json")

    # Enqueue into the per-thread serialization queue (concurrency=1
    # per thread). Workflow id is deterministic per (thread_id,
    # user_msg.id) so a request retry attaches to the existing run
    # instead of spawning a duplicate. Concurrent messages for the
    # same thread queue up and run serially.
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

    try:
        await stateflow_agent.enqueue_run(
            thread_id=thread_id,
            user_message_id=user_msg.id,
            prompt=prompt_text,
            history_dump=history_dump,
            # Parent of the assistant turn we're about to generate IS
            # the user message we just persisted — keeps the message
            # tree consistent with the non-durable path.
            assistant_parent_id=user_msg.id,
        )
    except Exception as exc:  # pragma: no cover — DBOS errors caught wholesale
        # Enqueue raises if the workflow id collides with one in a
        # terminal state. For our deterministic key case the typical
        # "collision" outcome is "already enqueued/running" — which
        # is what we want; the SSE stream will tail the existing
        # workflow's events. Log and continue.
        _log.info(
            "enqueue_run returned %s for user_msg=%s — "
            "assuming attach-to-existing", type(exc).__name__, user_msg.id,
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
    """Mount ``POST {prefix}/threads/{id}/messages`` as a Vercel-AI stream.

    Server-stateful contract: the thread MUST already exist (404 otherwise
    — no lazy-create). Its ``agent`` field is the registry key for a
    ``StateflowAgent`` instance the app registered at startup; the
    framework resolves that instance via
    ``pydantic_ai_stateflow.runtime.agents.get_agent`` and uses its
    ``agent`` (pydantic-ai ``Agent``), ``build_deps(...)``, and
    ``model_settings()`` to drive the run. The streaming endpoint itself
    is agent-agnostic.

    The just-arrived user turn is persisted via ``thread_repo.add_message``
    BEFORE the model runs, so a client crash mid-stream still leaves the
    thread consistent. After the agent completes the assistant reply is
    persisted via an ``on_complete`` callback wired into
    ``VercelAIAdapter.run_stream``.

    Tool-approval responses (Vercel AI SDK v6 ``approval-responded``
    parts) are extracted by ``VercelAIAdapter.deferred_tool_results``
    and threaded into ``run_stream`` so
    ``@agent.tool(requires_approval=True)`` tools resume after the user
    clicks Approve/Cancel.

    ``message_history`` is reconstructed from ``thread_repo.history(...)``
    (excluding the just-persisted current user turn — pydantic-ai
    re-derives that one from the incoming body messages).

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
        prompt_text = _last_user_text(adapter.messages)
        trigger = getattr(adapter.run_input, "trigger", "submit-message")
        is_regenerate = trigger == "regenerate-message"
        _log.debug(
            "post_message parsed: trigger=%s prompt_text=%r "
            "adapter_messages=%d last_role=%s",
            trigger, prompt_text, len(adapter.messages),
            getattr(last_message, "__class__", type(None)).__name__,
        )
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
            _log.debug("post_message: trimming adapter.messages to last user turn")
            _trim_adapter_messages_to_last_user_turn(adapter)
        else:
            _log.info(
                "post_message: deferred_tool_results present "
                "(approvals=%d, calls=%d) — skipping trim",
                len(deferred_results.approvals or {}),
                len(deferred_results.calls or {}),
            )

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
            thread_id, limit=history_limit,
        )

        _log.debug(
            "active_branch_before=%d msgs (last_role=%s)",
            len(active_branch_before),
            active_branch_before[-1].role if active_branch_before else None,
        )

        if is_regenerate:
            # Regenerate semantics: the assistant being regenerated is
            # REPLACED with a sibling whose parent is the same user turn
            # the old assistant was replying to. We skip past any trailing
            # assistant in the active branch to find that user turn.
            last_user_id = _find_last_role_id(active_branch_before, "user")
            new_assistant_parent_id = last_user_id
            _log.info(
                "regenerate-message: new assistant will sibling under "
                "user_id=%s",
                last_user_id,
            )
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
                    # ``state: "done"`` matches Vercel UIMessage's TextUIPart
                    # for a fully-rendered user turn. On restore via
                    # ``chat.setMessages``, parts without ``state`` slip
                    # through useChat's discriminator into a different
                    # branch and break ``MessagePartText`` rendering.
                    parts=[{
                        "type": "text", "text": prompt_text, "state": "done",
                    }],
                    parent_id=new_user_parent_id,
                )
                new_assistant_parent_id = user_msg.id
                _log.info(
                    "submit-message: persisted user msg id=%s parent=%s",
                    user_msg.id, new_user_parent_id,
                )
            else:
                _log.debug(
                    "submit-message: no prompt_text (no new user turn to "
                    "persist) — likely auto-resend after approval",
                )

        rows = await thread_repo.history(thread_id, limit=history_limit)
        # We already persisted the current user turn AND the trimmed
        # ``adapter.messages`` will carry it back into the prompt via the
        # adapter's frontend_messages merge, so drop it from our
        # repo-driven history to avoid duplication. For regenerate the
        # last user IS the prompt — drop it for the same reason.
        history = messages_to_model_history(rows, drop_prompt=prompt_text)
        _log.debug(
            "rebuilt message_history from repo: %d rows → %d ModelMessages",
            len(rows), len(history),
        )

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

            # Real upstream cost (OpenRouter / others) shows up on the
            # pydantic-ai-emitted ``operation.cost`` span attr via the
            # ``ModelResponse.cost`` fallback patch installed by
            # ``ObservabilityProvider`` — no per-route mirroring needed
            # here. See ``observability/cost.py`` for the extractor
            # strategy contract.

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
            #
            # We collect every assistant UIMessage emitted AFTER the
            # latest user prompt and merge their parts into one repo row
            # (assistant-ui groups them visually anyway). Multiple
            # ModelResponses arise when the agent loops through
            # tool_call → tool_return → text in one run.
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
                    # ``UIMessagePart`` is a discriminated-union of
                    # pydantic models. ``exclude_none=True`` strips
                    # explicit nulls (``providerMetadata``,
                    # ``preliminary``, ``approval`` …) so the persisted
                    # JSON matches the wire shape useChat already
                    # parses on the live stream. Keeping the nulls
                    # breaks the rendering side: assistant-ui's
                    # ``MessagePartText`` throws "can only be used
                    # inside text or reasoning message parts" because
                    # explicit-null fields shift the part through a
                    # different discriminator branch in
                    # ``useAISDKRuntime``'s mapping.
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

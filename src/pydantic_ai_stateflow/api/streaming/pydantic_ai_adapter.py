"""Adapter wrapping a pydantic-ai ``Agent`` as an :class:`AgentRunner`.

Eliminates the per-app boilerplate of running ``agent.iter`` → observing
per-node events → diff-against-last-emitted-prefix → emit AG-UI events.

Driven through pydantic-ai's graph iterator (``agent.iter``) so we can
observe BOTH text deltas AND tool-call events from the same run. Verified
against ``pydantic-ai`` 1.97.0 — the event taxonomy we hook into:

- ``PartStartEvent`` with ``part: TextPart`` — first text token in a part.
- ``PartDeltaEvent`` with ``delta: TextPartDelta`` — incremental text.
- ``PartStartEvent`` with ``part: ToolCallPart`` — tool call begins (its
  initial ``args`` snapshot is the first ``tool_call_args`` payload).
- ``PartDeltaEvent`` with ``delta: ToolCallPartDelta`` — tool-call args
  streaming in (a JSON fragment or a partial dict).
- ``FinalResultEvent`` — marks the *synthetic* output tool (the
  ``final_result`` tool pydantic-ai injects when ``output_type`` is a
  ``BaseModel``). We suppress tool-call SSE events for that call by index,
  since it's an implementation detail of structured output — not a real
  tool the frontend should render.
- ``FunctionToolCallEvent`` / ``FunctionToolResultEvent`` (on
  ``CallToolsNode``) — we emit ``tool_call_end`` after each function tool
  call completes (the begin/args came from the previous model-request
  node).
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID, uuid4

from pydantic_ai_stateflow.api.streaming.router import (
    AgentRunner,
    StreamEvent,
    _PostMessageBody,
    extract_text,
)
from pydantic_ai_stateflow.persistence.thread.repository import (
    ThreadRepository,
)

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelMessage

    from pydantic_ai_stateflow.persistence.thread.domain import Message


OutT = TypeVar("OutT")
DepsT = TypeVar("DepsT")


def _make_text_extractor(
    text_field: str | Callable[[Any], str],
) -> Callable[[Any], str]:
    if callable(text_field):
        return text_field

    attr = text_field

    def _get(out: Any) -> str:
        value = getattr(out, attr, None)
        return value or ""

    return _get


async def _resolve_deps(
    deps: Any,
    *,
    thread_id: UUID,
    run_id: UUID,
    message: _PostMessageBody,
    tenant_id: UUID,
) -> Any:
    """Resolve ``deps`` per-request.

    Resolution rules:

    - ``None`` → ``None`` (pydantic-ai treats this as "no deps").
    - callable → invoked with the runner kwargs; awaited if coroutine.
    - anything else (incl. plain class instances *without* ``__call__``,
      dataclass instances, ``BaseModel`` instances) → returned as-is.

    The "is it a factory?" check is the plain ``callable()`` builtin —
    pragmatic, matches what almost every Python user expects. The gotcha:
    if you pass a class *type* (e.g. ``deps=MyDeps``) we'll *call* it with
    runner kwargs. That's almost certainly what you want (``MyDeps``
    becomes its own factory), but pass an instance if you want the
    type-as-value semantics.
    """
    if deps is None:
        return None
    if callable(deps):
        result = deps(
            thread_id=thread_id,
            run_id=run_id,
            message=message,
            tenant_id=tenant_id,
        )
        if inspect.isawaitable(result):
            return await result
        return result
    return deps


def _tool_args_delta_to_str(delta: Any) -> str:
    """Render a ``ToolCallPartDelta.args_delta`` as a string fragment.

    pydantic-ai surfaces ``args_delta`` as either a ``str`` (already a
    JSON fragment from the wire) or a ``dict`` (some providers expose the
    parsed snapshot). Strings pass through; dicts get re-serialized as
    JSON so the wire payload stays parseable on the frontend.
    """
    if delta is None:
        return ""
    if isinstance(delta, str):
        return delta
    try:
        return json.dumps(delta, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(delta)


def _tool_args_to_str(args: Any) -> str:
    """Render a ``ToolCallPart.args`` snapshot as a string fragment.

    Same rationale as :func:`_tool_args_delta_to_str` — pydantic-ai's
    ``ToolCallPart.args`` is typed ``str | dict | None``.
    """
    return _tool_args_delta_to_str(args)


def _messages_to_model_history(
    rows: list[Message],
    *,
    drop_prompt: str | None = None,
) -> list[ModelMessage]:
    """Convert persisted ``Message`` rows → pydantic-ai ``ModelMessage`` list.

    Conversion rules (verified against pydantic-ai 1.97.0):

    - ``role == "user"`` → ``ModelRequest(parts=[UserPromptPart(content)])``
    - ``role == "assistant"`` → ``ModelResponse(parts=[TextPart(content)])``
      (single text part; tool-call replay is out of scope for iter-3 —
      the model re-derives any tool history from the resulting reply text)
    - empty text → row dropped (no point seeding empty turns)

    ``drop_prompt`` deduplicates the just-persisted current user turn.
    The router persists the user message BEFORE calling the runner, so
    repo history at runner time includes that turn. pydantic-ai's
    ``agent.iter(prompt, ...)`` will also synthesize a ``ModelRequest``
    for ``prompt`` — so we strip the trailing user row if its text
    matches the prompt verbatim. Robust to mid-history user turns
    (we only inspect the LAST row).

    Timestamps are preserved from ``Message.created_at`` so observability
    traces show real wall-clock ordering rather than synthetic "now".
    """
    from pydantic_ai.messages import (  # noqa: PLC0415
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    pruned = list(rows)
    if drop_prompt is not None and pruned:
        last = pruned[-1]
        if last.role == "user" and extract_text(last.parts) == drop_prompt:
            pruned = pruned[:-1]

    out: list[ModelMessage] = []
    for row in pruned:
        text = extract_text(row.parts)
        if not text:
            continue
        if row.role == "user":
            out.append(
                ModelRequest(
                    parts=[UserPromptPart(content=text, timestamp=row.created_at)],
                    timestamp=row.created_at,
                ),
            )
        elif row.role == "assistant":
            out.append(
                ModelResponse(
                    parts=[TextPart(content=text)],
                    timestamp=row.created_at,
                ),
            )
        # silently skip other roles (system/tool) — iter-3 scope
    return out


_DEFAULT_HISTORY_LIMIT = 200


def make_runner(  # noqa: C901 — single fused state machine, splitting hurts readability
    agent: Agent[Any, OutT],
    *,
    text_field: str | Callable[[OutT], str] = "reply",
    deps: Any = None,
    thread_repo: ThreadRepository | None = None,
    history_limit: int = _DEFAULT_HISTORY_LIMIT,
) -> AgentRunner:
    """Wrap a pydantic-ai ``Agent`` as an :class:`AgentRunner`.

    Emits the canonical AG-UI sequence::

        RUN_STARTED
          TEXT_MESSAGE_START
          (TEXT_MESSAGE_CONTENT × N  |  TOOL_CALL_START
                                        TOOL_CALL_ARGS × N
                                        TOOL_CALL_END) *
          TEXT_MESSAGE_END
        RUN_FINISHED

    (and ``RUN_ERROR`` on exception). Text and tool-call events may
    interleave depending on how the model orders its parts in a turn;
    consumers should key on ``messageId`` / ``toolCallId`` rather than
    relying on a fixed order.

    Args:
      agent: configured pydantic-ai Agent (typically with a structured
        ``output_type`` BaseModel).
      text_field: how to pull the streaming text out of each progressive
        output snapshot:

        - ``str`` — attribute name on the BaseModel (default ``"reply"``).
        - ``Callable[[OutT], str]`` — applied to each snapshot; must return
          the FULL text so far (the adapter does the diffing).

      deps: dependency injection for ``agent.iter``. Three forms:

        - ``None`` → pass nothing (default).
        - a non-callable value → passed through unchanged on every call.
        - a callable / coroutine function → invoked **per request** with
          keyword args ``thread_id``, ``run_id``, ``message``, ``tenant_id``
          to mint fresh per-request deps (e.g. a ``NoteToolDeps(repo=...,
          tenant_id=...)``).

      thread_repo: optional ``ThreadRepository``. When supplied, the
        runner fetches ``history(thread_id, tenant_id, limit=history_limit)``
        and passes the converted ``list[ModelMessage]`` as
        ``message_history=`` to ``agent.iter``. The just-persisted
        current-turn user row (the router persists BEFORE invoking the
        runner) is filtered out so pydantic-ai doesn't see the prompt
        twice. Omit to keep the legacy stateless behavior.
      history_limit: cap on the number of rows fetched from the repo
        (default 200). Tune this for long-running conversations.

    Tool-call events:
      The adapter observes ``agent.iter``'s per-node events. For each
      tool call the model emits, the adapter forwards a
      ``TOOL_CALL_START`` (carrying the tool name) → one or more
      ``TOOL_CALL_ARGS`` (carrying the JSON-fragment arg deltas) →
      ``TOOL_CALL_END``. The synthetic ``final_result`` output tool that
      pydantic-ai injects when ``output_type`` is a ``BaseModel`` is
      suppressed — it's a structured-output transport, not a tool the
      frontend should render.

    Diffing rules (text path — so the runner emits true deltas, not snapshots):
      - If the new snapshot extends the last emitted text → emit the suffix.
      - If pydantic partial-validation revises the prefix (the new value
        isn't a prefix-extension) → fall back to a full re-emit of the new
        value as the delta. Never emit a negative diff.
    """
    extractor = _make_text_extractor(text_field)

    async def _runner(
        *,
        thread_id: UUID,
        run_id: UUID,
        message: _PostMessageBody,
        tenant_id: UUID,
    ) -> AsyncIterator[StreamEvent]:
        message_id = uuid4()
        prompt = extract_text(message.parts)
        resolved_deps = await _resolve_deps(
            deps,
            thread_id=thread_id,
            run_id=run_id,
            message=message,
            tenant_id=tenant_id,
        )

        yield StreamEvent.run_started(thread_id=thread_id, run_id=run_id)
        yield StreamEvent.text_message_start(message_id=message_id)

        last_emitted = ""
        # Maps part-stream `index` → tool_call_id for tool-call parts seen
        # in the current model-request node. Tracks the in-flight tool
        # call so PartDeltaEvents can be routed to the right tool_call_id.
        tool_index_to_id: dict[int, str] = {}
        # Per-tool-call buffered state populated while observing the
        # model-request stream. Flushed in the CallToolsNode handler so we
        # can SUPPRESS the synthetic output-tool call (pydantic-ai's
        # ``FinalResultEvent`` marks which tool_call_id is the structured
        # output transport; that one must not surface as a tool call to
        # the frontend, since it's an implementation detail of
        # ``output_type=BaseModel``).
        pending_tool_calls: dict[str, dict[str, Any]] = {}
        # tool_call_ids that are the synthetic output tool — suppressed.
        output_tool_call_ids: set[str] = set()
        # Tool calls we've already emitted START for (idempotency guard
        # against the CallToolsNode handler re-emitting).
        emitted_starts: set[str] = set()

        # Imported here so consumers without pydantic-ai installed can
        # still import this module (the framework treats pydantic-ai as a
        # peer dep — make_runner only works WITH it, but importing the
        # module shouldn't crash without it).
        from pydantic_ai import Agent  # noqa: PLC0415
        from pydantic_ai.messages import (  # noqa: PLC0415
            FinalResultEvent,
            FunctionToolCallEvent,
            PartDeltaEvent,
            PartStartEvent,
            TextPart,
            TextPartDelta,
            ToolCallPart,
            ToolCallPartDelta,
        )

        history: list[ModelMessage] = []
        if thread_repo is not None:
            rows = await thread_repo.history(
                thread_id, tenant_id=tenant_id, limit=history_limit,
            )
            history = _messages_to_model_history(rows, drop_prompt=prompt)

        try:
            # Only pass ``message_history=`` when we actually have rows —
            # keeps the call signature minimal so legacy fakes (and any
            # ``Agent``-shaped callable that doesn't accept the kwarg)
            # still work when ``thread_repo`` is omitted.
            agent_run_cm = (
                agent.iter(
                    prompt, deps=resolved_deps, message_history=history,
                )
                if history
                else agent.iter(prompt, deps=resolved_deps)
            )
            async with agent_run_cm as agent_run:
                async for node in agent_run:
                    if Agent.is_model_request_node(node):
                        # Buffer of (tool_call_id, name, args_buffer)
                        # collected while observing this model-request's
                        # stream. We don't know which tool_call_ids are
                        # the synthetic output tool until FinalResultEvent
                        # fires *during* the same node — so we buffer and
                        # flush in the subsequent CallToolsNode where
                        # FunctionToolCallEvent confirms which calls are
                        # real function-tool invocations.
                        tool_index_to_id.clear()
                        pending_tool_calls.clear()
                        async with node.stream(agent_run.ctx) as model_stream:
                            async for event in model_stream:
                                if isinstance(event, PartStartEvent):
                                    part = event.part
                                    if isinstance(part, TextPart):
                                        text = part.content or ""
                                        delta = _diff_text(last_emitted, text)
                                        if delta:
                                            yield StreamEvent.text_message_content(
                                                message_id=message_id,
                                                delta=delta,
                                            )
                                            last_emitted = text
                                    elif isinstance(part, ToolCallPart):
                                        tcid = part.tool_call_id
                                        tool_index_to_id[event.index] = tcid
                                        pending_tool_calls[tcid] = {
                                            "name": part.tool_name,
                                            "args": _tool_args_to_str(part.args),
                                        }
                                elif isinstance(event, PartDeltaEvent):
                                    delta_part = event.delta
                                    if isinstance(delta_part, TextPartDelta):
                                        text_delta = delta_part.content_delta or ""
                                        if text_delta:
                                            yield StreamEvent.text_message_content(
                                                message_id=message_id,
                                                delta=text_delta,
                                            )
                                            last_emitted += text_delta
                                    elif isinstance(delta_part, ToolCallPartDelta):
                                        delta_tcid = tool_index_to_id.get(
                                            event.index,
                                        )
                                        if delta_tcid is None:
                                            continue
                                        args_delta = _tool_args_delta_to_str(
                                            delta_part.args_delta,
                                        )
                                        if args_delta:
                                            pending_tool_calls.setdefault(
                                                delta_tcid, {"name": "", "args": ""},
                                            )
                                            pending_tool_calls[delta_tcid][
                                                "args"
                                            ] += args_delta
                                elif isinstance(event, FinalResultEvent):
                                    # Marks the synthetic structured-output
                                    # tool. Drop it from the buffer so it
                                    # never surfaces as a tool_call_*
                                    # event on the wire.
                                    if event.tool_call_id is not None:
                                        output_tool_call_ids.add(
                                            event.tool_call_id,
                                        )
                                        pending_tool_calls.pop(
                                            event.tool_call_id, None,
                                        )
                    elif Agent.is_call_tools_node(node):
                        async with node.stream(agent_run.ctx) as tool_stream:
                            async for tool_event in tool_stream:
                                if isinstance(tool_event, FunctionToolCallEvent):
                                    tc_tcid = tool_event.part.tool_call_id
                                    if tc_tcid in output_tool_call_ids:
                                        continue
                                    # Flush the buffered START + ARGS now
                                    # that we've confirmed this is a real
                                    # function tool call. If the buffer
                                    # is missing (some providers skip
                                    # PartStartEvent), fall back to the
                                    # event's own part data.
                                    buf = pending_tool_calls.pop(tc_tcid, None)
                                    name = (
                                        buf["name"]
                                        if buf
                                        else tool_event.part.tool_name
                                    )
                                    args_str = (
                                        buf["args"]
                                        if buf
                                        else _tool_args_to_str(tool_event.part.args)
                                    )
                                    if tc_tcid not in emitted_starts:
                                        emitted_starts.add(tc_tcid)
                                        yield StreamEvent.tool_call_start(
                                            tool_call_id=tc_tcid,
                                            tool_call_name=name,
                                            parent_message_id=message_id,
                                        )
                                        if args_str:
                                            yield StreamEvent.tool_call_args(
                                                tool_call_id=tc_tcid,
                                                delta=args_str,
                                            )
                                    yield StreamEvent.tool_call_end(
                                        tool_call_id=tc_tcid,
                                    )

                # After the graph completes, pull the final output and
                # emit any text suffix the part-stream missed (some
                # providers return the structured output without a text
                # part — the `reply` lives only on the parsed object).
                final = agent_run.result.output if agent_run.result else None
                if final is not None:
                    final_text = extractor(final) or ""
                    suffix = _diff_text(last_emitted, final_text)
                    if suffix:
                        yield StreamEvent.text_message_content(
                            message_id=message_id, delta=suffix,
                        )
                        last_emitted = final_text
        except Exception as exc:  # noqa: BLE001 — surface and re-raise
            yield StreamEvent.run_error(message=str(exc))
            raise

        yield StreamEvent.text_message_end(message_id=message_id)
        yield StreamEvent.run_finished(thread_id=thread_id, run_id=run_id)

    return _runner


def _diff_text(last_emitted: str, current: str) -> str:
    """Compute the delta to emit given a snapshot ``current`` and the
    last-emitted prefix. Mirrors the legacy ``stream_output`` diffing
    rules (never emit a negative diff; full re-emit on prefix revision).
    """
    if not current or current == last_emitted:
        return ""
    if current.startswith(last_emitted):
        return current[len(last_emitted):]
    # Partial validation revised the prefix (or shortened it); re-emit
    # the full new value as the delta — clients treat deltas as appends,
    # so a negative diff is never safe.
    return current

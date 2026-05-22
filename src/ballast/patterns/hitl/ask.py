"""``ask_human()`` — await-style durable HITL primitive.

When to reach for this vs :class:`DurableHITLWorkflow`:

* :class:`DurableHITLWorkflow` is **fire-and-forget**: caller invokes
  ``open(...)`` and returns immediately; the post-decision body runs
  later from a separate workflow fiber via ``on_decision``. Right for
  pydantic-ai tools and HTTP handlers that can't block on user
  response time.

* :func:`ask_human` is **await-and-return**: caller awaits the
  primitive inside its OWN durable workflow and gets the verdict back
  inline. The caller writes save / notify / next-step logic
  immediately after the await — no split brain across two halves of
  the codebase. Right for multi-step orchestration flows where HITL is
  one step among several (brainstorm → ask → save, etc.).

Both reach the same internals (helper thread spawn, opening message,
optional parent-thread notify, ``Durable.recv_async`` on the HITL
topic) — the difference is purely who owns the post-decision code path.

## Usage

::

    @Durable.workflow()
    async def my_flow(task: MyTask) -> MyOutcome:
        chosen = await _pick_candidate(task)
        verdict = await ask_human(
            helper_agent=MyHelperAgent,
            context=MyContext(proposal=chosen, parent_thread_id=task.thread_id),
            opening_message="Approve this?",
            notify_parent_thread_id=task.thread_id,
        )
        match verdict:
            case ApprovedResponse():
                return await _save(chosen)
            case ModifiedResponse(modified_proposal=mod):
                return await _save({**chosen, **mod})
            case RejectedResponse() | TimeoutResponse():
                return MyOutcome(saved=False)

Must be called from inside another ``@Durable.workflow``. Calling from
a plain async function works but loses workflow-level durability
guarantees on the caller side.
"""

from __future__ import annotations

import importlib
import uuid
from typing import TYPE_CHECKING, Any
from uuid import UUID

from dbos import DBOS
from pydantic import BaseModel, TypeAdapter

from ballast.durable import Durable
from ballast.logging import get_logger
from ballast.patterns.hitl.response import HITLResponse, TimeoutResponse
from ballast.patterns.hitl.topic import _hitl_topic
from ballast.runtime.event_stream import EventNotification, thread_channel

if TYPE_CHECKING:
    from datetime import timedelta

    from ballast.runtime.agents import BallastAgent

_log = get_logger(__name__)
_RESPONSE_ADAPTER: TypeAdapter[HITLResponse] = TypeAdapter(HITLResponse)

# ≈ 1 year — "wait forever" without violating ``Durable.recv_async``'s
# arithmetic (it rejects ``None`` and adds ``timeout_seconds`` to
# ``time.time()`` for sleep accounting).
_NO_TIMEOUT_SECONDS: float = 365 * 24 * 60 * 60.0

# Stable namespace for deriving deterministic request UUIDs from
# workflow ids. Public-facing helper agents store these as
# ``thread.metadata_["request_id"]`` and the framework parses them
# back as ``UUID`` — using ``uuid5`` keeps the value deterministic
# across workflow replay (a plain ``uuid4`` inside the workflow body
# would diverge on every replay and break ``Durable.recv_async``).
_REQUEST_ID_NAMESPACE = uuid.UUID("a5c1b2e0-1f9d-4f7a-9c3e-9b3a1c5e7f00")


async def ask_human(
    *,
    helper_agent: type[BallastAgent],
    context: BaseModel,
    opening_message: str | None = None,
    timeout: timedelta | None = None,
    notify_parent_thread_id: UUID | None = None,
) -> HITLResponse:
    """Open a helper thread, block until the user responds, return verdict.

    Same durability + recovery semantics as
    :class:`DurableHITLWorkflow` (every side effect is a step; helper
    thread creation, opening message, parent-thread notify are all
    memoised across crashes). Difference is **synchronous from the
    caller's POV**: the verdict comes back as a typed
    :class:`HITLResponse` and the caller acts on it inline.

    Arguments mirror :meth:`DurableHITLWorkflow.open`:

    * ``helper_agent`` — the agent class powering the helper thread.
      Its ``metadata_model`` must validate ``context``.
    * ``context`` — typed context for the helper agent's tools.
      Serialized into the helper thread's ``metadata_``.
    * ``opening_message`` — optional assistant message to seed the
      helper thread so the user sees something on open.
    * ``timeout`` — bound on how long to wait. On timeout returns a
      :class:`TimeoutResponse`. ``None`` ≈ wait forever.
    * ``notify_parent_thread_id`` — when supplied, emits a
      ``thread-created`` event into the parent thread so a UI tailing
      ``/threads/{id}/events`` refreshes its thread list without F5.
    """
    metadata_model = helper_agent.metadata_model
    if metadata_model is None:
        raise ValueError(
            f"{helper_agent.__name__}.metadata_model is None — cannot "
            "use it as a HITL helper agent. Set a metadata_model that "
            "validates the context shape.",
        )
    if not isinstance(context, metadata_model):
        raise TypeError(
            f"context must be an instance of "
            f"{helper_agent.__name__}.metadata_model "
            f"({metadata_model.__name__}), got {type(context).__name__}",
        )

    context_class_fqn = (
        f"{metadata_model.__module__}.{metadata_model.__qualname__}"
    )
    timeout_seconds = (
        timeout.total_seconds() if timeout is not None
        else _NO_TIMEOUT_SECONDS
    )

    payload = await _ask_human_workflow(
        helper_agent_name=helper_agent.name,
        context_dict=context.model_dump(mode="json"),
        context_class_fqn=context_class_fqn,
        opening_message=opening_message,
        timeout_seconds=timeout_seconds,
        notify_parent_thread_id=(
            str(notify_parent_thread_id)
            if notify_parent_thread_id is not None else None
        ),
    )
    return _RESPONSE_ADAPTER.validate_python(payload)


@Durable.workflow()
async def _ask_human_workflow(
    *,
    helper_agent_name: str,
    context_dict: dict[str, Any],
    context_class_fqn: str,
    opening_message: str | None,
    timeout_seconds: float,
    notify_parent_thread_id: str | None,
) -> dict[str, Any]:
    """Durable body of :func:`ask_human`.

    Returns the verdict as a JSON-friendly dict (the public wrapper
    re-validates it into a typed :class:`HITLResponse`). Keeping the
    workflow boundary on primitives matches the pattern in
    :class:`DurableHITLWorkflow` and survives pydantic / pickle
    version drift in DBOS's serialization layer.

    Routing: ``workflow_id`` for the helper agent's outbound
    ``Durable.send_async`` is THIS workflow's own id (the helper sends
    back to us, not a separate decision workflow). ``request_id`` is
    derived deterministically from the workflow id via ``uuid5`` so
    the topic is stable across workflow replay.
    """
    my_workflow_id = DBOS.workflow_id
    request_id = uuid.uuid5(_REQUEST_ID_NAMESPACE, my_workflow_id)

    thread_metadata: dict[str, Any] = dict(context_dict)
    thread_metadata["request_id"] = str(request_id)
    thread_metadata["workflow_id"] = my_workflow_id

    thread_id = await _create_helper_thread(
        agent_name=helper_agent_name,
        metadata=thread_metadata,
    )

    if opening_message:
        await _seed_opening_message(thread_id, opening_message)

    if notify_parent_thread_id is not None:
        await _emit_thread_created(
            parent_thread_id_str=notify_parent_thread_id,
            helper_thread_id=thread_id,
            helper_agent_name=helper_agent_name,
            thread_metadata=thread_metadata,
        )

    topic = _hitl_topic(request_id)
    payload = await Durable.recv_async(topic, timeout_seconds=timeout_seconds)

    if payload is None:
        from datetime import UTC, datetime  # noqa: PLC0415
        response: HITLResponse = TimeoutResponse(
            answered_at=datetime.now(tz=UTC),
        )
    else:
        response = _RESPONSE_ADAPTER.validate_python(payload)

    # Re-hydrate against the typed context class so any subscriber
    # downstream of the workflow caller (e.g. an audit step) can rely
    # on a properly typed instance. We don't NEED the typed context
    # here — we just shape-check it to fail loud on schema drift
    # between caller and helper.
    context_cls = _resolve_class(context_class_fqn)
    _ = context_cls.model_validate(context_dict)

    return response.model_dump(mode="json")


# ── per-step writes (memoised across replay) ─────────────────────────


@Durable.step()
async def _create_helper_thread(
    *, agent_name: str, metadata: dict[str, Any],
) -> UUID:
    from ballast.runtime.engine import get_ballast  # noqa: PLC0415
    thread = await get_ballast().thread_repo.create(
        agent=agent_name, metadata=metadata,
    )
    return thread.id


@Durable.step()
async def _seed_opening_message(
    thread_id: UUID, opening_message: str,
) -> None:
    from ballast.runtime.engine import get_ballast  # noqa: PLC0415
    await get_ballast().thread_repo.add_message(
        thread_id,
        role="assistant",
        parts=[{
            "type": "text",
            "text": opening_message,
            "state": "done",
        }],
    )


@Durable.step()
async def _emit_thread_created(
    *,
    parent_thread_id_str: str,
    helper_thread_id: UUID,
    helper_agent_name: str,
    thread_metadata: dict[str, Any],
) -> None:
    from ballast.runtime.engine import get_ballast  # noqa: PLC0415
    engine = get_ballast()
    parent_id = UUID(parent_thread_id_str)
    _log.info(
        "ask_human notify_parent=%s helper_thread=%s",
        parent_id, helper_thread_id,
    )
    ev = await engine.event_log.append(
        thread_id=parent_id,
        kind="thread-created",
        payload={
            "thread_id": str(helper_thread_id),
            "agent": helper_agent_name,
            "metadata": thread_metadata,
        },
    )
    await engine.event_stream.publish(
        thread_channel(parent_id),
        EventNotification(thread_id=parent_id, seq=ev.seq),
    )


def _resolve_class(fqn: str) -> Any:
    module_path, _, name = fqn.rpartition(".")
    if not module_path:
        raise ValueError(f"FQN must be fully qualified, got {fqn!r}")
    mod = importlib.import_module(module_path)
    return getattr(mod, name)


__all__ = ["ask_human"]

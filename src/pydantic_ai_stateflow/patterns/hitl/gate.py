from __future__ import annotations

import itertools
import uuid as _uuid
from datetime import timedelta
from typing import TYPE_CHECKING, Any, ClassVar, cast
from uuid import UUID, uuid4

from dbos import DBOS, DBOSConfiguredInstance

from pydantic_ai_stateflow.durable import Durable
from pydantic import BaseModel

from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.observability.trace_names import TraceName
from pydantic_ai_stateflow.patterns.errors import HITLDenied, HITLTimedOut
from pydantic_ai_stateflow.patterns.hitl.channel import HITLChannel
from pydantic_ai_stateflow.patterns.hitl.policy import Policy
from pydantic_ai_stateflow.patterns.hitl.prompt import HITLPrompt
from pydantic_ai_stateflow.patterns.hitl.response import (
    HITLResponse,
    TimeoutResponse,
)
from pydantic_ai_stateflow.persistence import HITLRepository

if TYPE_CHECKING:
    from pydantic_ai_stateflow.persistence.thread.repository import (
        ThreadRepository,
    )
    from pydantic_ai_stateflow.runtime.agents import StateflowAgent

_KIND_TO_VERDICT = {
    "approved": "approve",
    "rejected": "reject",
    "modified": "revise",
}

_instance_counter = itertools.count()


@Durable.dbos_class()
class HITLGate(DBOSConfiguredInstance):
    """HITL pause + authz (defense-in-depth on receive).

    Authz is checked twice: at the responder endpoint (so unauthorized
    replies never reach the workflow's recv topic) and again here on
    receive. Denied attempts are persisted to ``hitl_authz_denials``.

    The DBOS workflow id of THIS gate run is stored as
    ``BlockingRequirement.workflow_id`` so the endpoint can route the
    response back via ``Durable.send(destination_id=workflow_id, ...)``.

    Apps that need tenant/workspace scoping carry it inside their
    ``HITLPrompt`` subclass and the ``Policy`` (which receives
    ``resource=prompt``).
    """

    name: ClassVar[str] = "hitl_gate"

    def __init__(
        self,
        *,
        channel: HITLChannel,
        policy: Policy,
        repo: HITLRepository,
        thread_repo: ThreadRepository | None = None,
    ) -> None:
        super().__init__(config_name=f"hitl-gate-{next(_instance_counter)}")
        self.channel = channel
        self.policy = policy
        self.repo = repo
        # Only required by ``ask_helper`` (the agent-thread-based HITL
        # flavour); plain ``run(prompt, …)`` works with any channel and
        # doesn't need thread persistence at the gate level.
        self.thread_repo = thread_repo

    @Durable.workflow()
    @traced(TraceName.PATTERN_HITL_GATE, attrs=lambda self, prompt, *, request_id=None: {
        "pattern": self.name,
        "request_id": str(request_id) if request_id is not None else "<auto>",
    })
    async def run(
        self, prompt: HITLPrompt, *, request_id: UUID | None = None,
    ) -> HITLResponse:
        # Inside a @DBOS.workflow, ``Durable.current_workflow_id()`` is the current
        # workflow's id — what responders need to ``DBOS.send`` back to.
        # DBOS auto-generated nested workflow ids may not be UUIDs (they
        # take the form ``{parent}-{ordinal}``); coerce non-UUID ids to a
        # deterministic UUID so the BlockingRequirement column accepts them.
        raw_id = cast(str, Durable.current_workflow_id())
        try:
            workflow_id = UUID(raw_id)
        except (ValueError, AttributeError, TypeError):
            workflow_id = _uuid.uuid5(_uuid.NAMESPACE_URL, raw_id)

        # ``request_id`` lets the caller pre-allocate the id so it can
        # be embedded in a child thread's metadata BEFORE the gate
        # actually persists the request (needed when the approval UI is
        # a sibling thread that must reference the request by id).
        request = await self.repo.persist_request(
            prompt=prompt.model_dump(mode="json"),
            workflow_id=workflow_id,
            gate_kind=self.name,
            purpose="approval",
            timeout_at=None,
            request_id=request_id,
        )

        response = await self.channel.ask(prompt, request_id=request.id)

        if isinstance(response, TimeoutResponse):
            await self.repo.persist_timeout(request.id)
            raise HITLTimedOut(request_id=request.id)

        verdict = await self.policy.can(
            actor=response.actor_id,
            action="decide",
            resource=prompt,
        )
        if not verdict.is_grant:
            await self.repo.persist_authz_denied(
                request_id=request.id,
                actor_id=response.actor_id or "<anonymous>",
                voter_votes=dict(verdict.votes),
            )
            raise HITLDenied(
                actor_id=response.actor_id or "<anonymous>",
                votes=dict(verdict.votes),
            )

        helper_verdict_payload: dict[str, Any] | None = None
        helper_verdict_context_type: str | None = None
        helper_thread_id: UUID | None = None
        raw_verdict = getattr(response, "helper_verdict", None)
        if raw_verdict is not None:
            helper_verdict_payload = dict(raw_verdict)
            tid_str = helper_verdict_payload.pop("__helper_thread_id__", None)
            fqn = helper_verdict_payload.pop("__context_type_fqn__", None)
            if tid_str is not None:
                helper_thread_id = UUID(tid_str)
            if fqn is not None:
                helper_verdict_context_type = fqn

        await self.repo.persist_response(
            request_id=request.id,
            actor_id=response.actor_id or "<anonymous>",
            verdict=_KIND_TO_VERDICT[response.kind],
            payload=response.model_dump(mode="json"),
            helper_verdict_payload=helper_verdict_payload,
            helper_verdict_context_type=helper_verdict_context_type,
            helper_thread_id=helper_thread_id,
        )
        return response

    async def ask(
        self, prompt: HITLPrompt, *, request_id: UUID | None = None,
    ) -> HITLResponse:
        """Asker Protocol adapter."""
        return await self.run(prompt, request_id=request_id)

    async def ask_helper(
        self,
        *,
        helper_agent: type[StateflowAgent],
        context: BaseModel,
        timeout: timedelta | None = None,
        decision_kinds: set[str] | None = None,
        opening_message: str | None = None,
    ) -> HITLResponse:
        """Open a helper-agent thread bound to ``context`` and block on decision.

        High-level entry point for the case where the HITL surface is
        another ``StateflowAgent`` (a conversational helper in its own
        thread), rather than a generic UI card or webhook.

        Flow:
          1. A new thread is created bound to ``helper_agent.name`` with
             ``context.model_dump()`` (plus a ``request_id`` routing key)
             as its metadata. The helper agent's own ``build_deps``
             validates the dict against ``helper_agent.metadata_model``.
          2. The blocking requirement is persisted with the pre-allocated
             ``request_id`` so the helper agent's tools can route their
             response back via ``Durable.send(_hitl_topic(request_id), …)``.
          3. A thin ``HITLPrompt`` wrapper is built from ``context`` so
             existing ``run`` plumbing (channel.ask → policy.can →
             persist_response) keeps working — the channel's
             ``DBOS.recv_async`` on ``_hitl_topic(request_id)`` does the
             actual waiting.
          4. On approval/rejection/modification the helper agent's
             tools fire ``DBOS.send``; on timeout the gate raises
             ``HITLTimedOut``.

        ``context`` must be an instance of
        ``helper_agent.metadata_model``. ``opening_message`` (optional)
        seeds an assistant message on the new thread so the user sees
        something the moment they open it — otherwise the thread looks
        empty until they type.
        """
        if self.thread_repo is None:
            raise RuntimeError(
                "HITLGate.ask_helper requires a thread_repo — pass one to "
                "HITLGate(...) at construction.",
            )
        metadata_model = helper_agent.metadata_model
        if metadata_model is None:
            raise ValueError(
                f"{helper_agent.__name__}.metadata_model is None — cannot use "
                "it as a HITL helper agent. Set a metadata_model that "
                "validates the context shape.",
            )
        if not isinstance(context, metadata_model):
            raise TypeError(
                f"context must be an instance of "
                f"{helper_agent.__name__}.metadata_model "
                f"({metadata_model.__name__}), got {type(context).__name__}",
            )

        request_id = uuid4()

        thread_metadata: dict[str, Any] = context.model_dump(mode="json")
        thread_metadata["request_id"] = str(request_id)
        thread = await self.thread_repo.create(
            agent=helper_agent.name,
            metadata=thread_metadata,
        )

        if opening_message:
            await self.thread_repo.add_message(
                thread.id,
                role="assistant",
                parts=[{
                    "type": "text",
                    "text": opening_message,
                    "state": "done",
                }],
            )

        # Build a thin HITLPrompt so the existing run() plumbing (policy,
        # persistence, channel.ask → DBOS.recv) keeps working unchanged.
        # ``context`` is serialized into the prompt body so policy
        # voters that introspect ``HITLPrompt`` still see the payload.
        prompt = HITLPrompt(
            title=f"{helper_agent.name} decision",
            context=context.model_dump_json(),
            decision_kinds=decision_kinds or {"approved", "rejected", "modified"},
            timeout=timeout,
        )
        return await self.run(prompt, request_id=request_id)

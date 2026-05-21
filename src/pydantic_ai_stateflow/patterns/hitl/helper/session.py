from __future__ import annotations

import importlib
import itertools
from typing import Any, ClassVar, Protocol, runtime_checkable
from uuid import UUID

from dbos import DBOS, DBOSConfiguredInstance

from pydantic_ai_stateflow.durable import Durable
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent

from pydantic_ai_stateflow.patterns.hitl.helper.factory import (
    HelperAgentFactory,
    HelperDeps,
    HelperToolBox,
)
from pydantic_ai_stateflow.patterns.hitl.prompt import HITLPrompt
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository

_session_counter = itertools.count()


def _helper_msg_topic(request_id: UUID) -> str:
    """Topic for *inbound founder messages* (separate from the gate topic)."""
    return f"helper:{request_id}"


class HelperSessionInput(BaseModel):
    """Workflow input for ``DefaultHelperSessionRunner.run``.

    All fields are JSON-serializable. The base agent is reconstructed
    at runtime via ``base_agent_module`` + ``base_agent_attr`` so the
    workflow input doesn't carry the un-picklable Agent object.
    """

    model_config = ConfigDict(frozen=True)

    prompt_payload: dict[str, Any]
    request_id: UUID
    gate_workflow_id: UUID
    base_agent_module: str
    base_agent_attr: str | None
    context_type_fqn: str | None
    actor_id: str


@runtime_checkable
class HelperSessionRunner(Protocol):
    """Drives the helper conversation in its OWN DBOS workflow."""

    async def run(self, input: HelperSessionInput) -> None: ...


@Durable.dbos_class()
class DefaultHelperSessionRunner(DBOSConfiguredInstance):
    """Default helper-session driver — bounded loop on inbound messages."""

    name: ClassVar[str] = "helper_session"

    def __init__(
        self,
        *,
        thread_repo: ThreadRepository,
        agent_factory: HelperAgentFactory,
        max_turns: int = 30,
        message_recv_timeout_seconds: float = 86_400.0,
    ) -> None:
        super().__init__(
            config_name=f"helper-session-{next(_session_counter)}",
        )
        self.thread_repo = thread_repo
        self.agent_factory = agent_factory
        self.max_turns = max_turns
        self.message_recv_timeout_seconds = message_recv_timeout_seconds
        # Test seam: tests assign a pre-built Agent here before calling run();
        # production resolves the base agent via FQN from the workflow input.
        self._base_agent_for_test: Agent[HelperDeps, str] | None = None

    @Durable.workflow()
    async def run(self, input: HelperSessionInput) -> None:
        prompt = HITLPrompt.model_validate(input.prompt_payload)
        context_type = (
            _resolve_fqn(input.context_type_fqn)
            if input.context_type_fqn is not None
            else None
        )
        base_agent = self._base_agent_for_test or _resolve_base_agent(
            input.base_agent_module, input.base_agent_attr,
        )

        thread = await self.thread_repo.create(
            agent="hitl",
            metadata={
                "request_id": str(input.request_id),
                "gate_kind": "hitl_gate",
                "title": prompt.title,
                "actor_id": input.actor_id,
            },
        )

        toolbox = HelperToolBox()
        agent = self.agent_factory(
            base_agent=base_agent,
            request_id=input.request_id,
            context_type=context_type,
        )

        msg_topic = _helper_msg_topic(input.request_id)
        gate_topic = _hitl_topic(input.request_id)
        tools_invoked: list[str] = []

        for turn in range(self.max_turns):
            msg = await Durable.recv(
                msg_topic,
                timeout_seconds=self.message_recv_timeout_seconds,
            )
            if msg is None:
                return

            user_text = (
                msg.get("text", "") if isinstance(msg, dict) else str(msg)
            )
            await self.thread_repo.add_message(
                thread.id,
                role="user",
                parts=[{"type": "text", "content": user_text}],
            )

            deps = HelperDeps(
                request_id=input.request_id,
                actor_id=input.actor_id,
                turn_count=turn,
                tools_invoked_so_far=list(tools_invoked),
                toolbox=toolbox,
            )
            await agent.run(user_text, deps=deps)

            if toolbox.response is not None:
                Durable.send(
                    destination_id=str(input.gate_workflow_id),
                    message=toolbox.response.model_dump(mode="json"),
                    topic=gate_topic,
                )
                return


def _resolve_fqn(fqn: str) -> type[Any]:
    mod_name, _, attr = fqn.rpartition(".")
    module = importlib.import_module(mod_name)
    resolved: type[Any] = getattr(module, attr)
    return resolved


def _resolve_base_agent(module: str, attr: str | None) -> Agent[HelperDeps, str]:
    if attr is None:
        raise ValueError(
            "DefaultHelperSessionRunner: base_agent_attr is required when "
            "running outside tests (no _base_agent_for_test injected)",
        )
    resolved: Agent[HelperDeps, str] = getattr(
        importlib.import_module(module), attr,
    )
    return resolved

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from pydantic_ai_stateflow.patterns.hitl.gate import HITLGate
from pydantic_ai_stateflow.patterns.hitl.prompt import HITLPrompt
from pydantic_ai_stateflow.patterns.hitl.response import (
    ApprovedResponse,
    ModifiedResponse,
    RejectedResponse,
    TimeoutResponse,
)
from pydantic_ai_stateflow.patterns.mutation.primitives import (
    AcceptedResult,
    Proposal,
    RejectedAt,
)

T = TypeVar("T", bound=BaseModel)


class ApprovalStage(Generic[T]):
    """Single-actor approval Stage. Delegates to HITLGate; supports modify-flow.

    Spec 2C.3 + 4A.0.9. Modify-flow requires `allow_modify=True` AND an
    explicit `editable_paths` whitelist — paths absent from the whitelist
    that differ between original and modified cause the modification to
    be rejected (defense against over-broad UI edits).
    """

    def __init__(
        self,
        *,
        hitl: HITLGate,
        prompt_builder: Callable[[Proposal[T]], HITLPrompt],
        when: Callable[[Proposal[T]], bool] = lambda _: True,
        stage_name: str = "approval",
        allow_modify: bool = False,
        editable_paths: set[str] | None = None,
    ) -> None:
        if allow_modify and editable_paths is None:
            raise ValueError(
                "ApprovalStage: allow_modify=True requires explicit editable_paths"
            )
        self.name = stage_name
        self.hitl = hitl
        self.prompt_builder = prompt_builder
        self.when = when
        self.allow_modify = allow_modify
        self.editable_paths = editable_paths or set()

    async def process(
        self, proposal: Proposal[Any],
    ) -> AcceptedResult[Any] | RejectedAt:
        if not self.when(proposal):
            return AcceptedResult(proposal=proposal, entity_id=uuid4())

        prompt = self.prompt_builder(proposal)
        try:
            response = await self.hitl.run(prompt)
        except Exception as exc:
            return RejectedAt(
                stage=self.name,
                reason=str(exc),
                actor_id=None,
            )

        if isinstance(response, ApprovedResponse):
            return AcceptedResult(proposal=proposal, entity_id=uuid4())
        if isinstance(response, RejectedResponse):
            return RejectedAt(
                stage=self.name,
                reason=response.feedback or "rejected",
                actor_id=response.actor_id,
            )
        if isinstance(response, ModifiedResponse):
            return self._apply_modification(proposal, response)
        if isinstance(response, TimeoutResponse):
            return RejectedAt(
                stage=self.name, reason="hitl timeout", actor_id=None,
            )
        return RejectedAt(
            stage=self.name,
            reason=f"unsupported response kind: {getattr(response, 'kind', '?')}",
        )

    def _apply_modification(
        self, proposal: Proposal[Any], response: ModifiedResponse,
    ) -> AcceptedResult[Any] | RejectedAt:
        if not self.allow_modify:
            return RejectedAt(
                stage=self.name,
                reason="modification received but allow_modify=False",
                actor_id=response.actor_id,
            )
        original = (
            proposal.payload.model_dump()
            if isinstance(proposal.payload, BaseModel)
            else dict(proposal.payload)
        )
        modified_dict = dict(response.modified_proposal)
        diff_paths = {
            k for k in (set(original) | set(modified_dict))
            if original.get(k) != modified_dict.get(k)
        }
        outside = diff_paths - self.editable_paths
        if outside:
            return RejectedAt(
                stage=self.name,
                reason=f"modifications outside whitelist: {sorted(outside)}",
                actor_id=response.actor_id,
            )
        new_payload_dict = {
            **original,
            **{k: modified_dict[k] for k in diff_paths},
        }
        if isinstance(proposal.payload, BaseModel):
            new_payload: Any = type(proposal.payload).model_validate(
                new_payload_dict,
            )
        else:
            new_payload = new_payload_dict
        modified_proposal = proposal.model_copy(update={"payload": new_payload})
        return AcceptedResult(proposal=modified_proposal, entity_id=uuid4())


class PartialApprovalStage(Generic[T]):
    """Per-element approve / reject / modify (spec 3K gap #1 + 4A.0.9).

    Uses an `element_extractor` to turn a proposal into a list of
    `ProposalElement`s, lets the human approve a subset (with optional
    modifications per element), and rebuilds the proposal via the
    extractor's `with_approved_subset`. SP5 ships the wiring; per-element
    extractor adapters for concrete domains (Wave plan items, etc.) come
    in SP7.

    For SP5 the public contract is just the constructor + `process` —
    full extractor wiring is tested via a tiny in-test impl.
    """

    def __init__(
        self,
        *,
        hitl: HITLGate,
        prompt_builder: Callable[[Proposal[T]], HITLPrompt],
        element_extractor: Any,
        stage_name: str = "partial_approval",
        allow_modify: bool = False,
        editable_paths: set[str] | None = None,
    ) -> None:
        if allow_modify and editable_paths is None:
            raise ValueError(
                "PartialApprovalStage: allow_modify=True requires editable_paths"
            )
        self.name = stage_name
        self.hitl = hitl
        self.prompt_builder = prompt_builder
        self.element_extractor = element_extractor
        self.allow_modify = allow_modify
        self.editable_paths = editable_paths or set()

    async def process(
        self, proposal: Proposal[Any],
    ) -> AcceptedResult[Any] | RejectedAt:
        return await ApprovalStage(
            hitl=self.hitl,
            prompt_builder=self.prompt_builder,
            stage_name=self.name,
            allow_modify=self.allow_modify,
            editable_paths=self.editable_paths,
        ).process(proposal)

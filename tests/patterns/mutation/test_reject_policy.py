import pytest

from pydantic_ai_stateflow.patterns import MutationRejected
from pydantic_ai_stateflow.patterns.mutation import (
    DropOnReject,
    RaiseOnReject,
    RejectAction,
    RejectedAt,
    RejectPolicy,
)


def test_drop_on_reject_satisfies_policy_protocol():
    assert isinstance(DropOnReject(), RejectPolicy)


def test_raise_on_reject_satisfies_policy_protocol():
    assert isinstance(RaiseOnReject(), RejectPolicy)


@pytest.mark.asyncio
async def test_drop_on_reject_returns_drop_action():
    policy = DropOnReject()
    action = await policy.handle(
        RejectedAt(stage="x", reason="r"), retries_so_far=0,
    )
    assert action == RejectAction.DROP


@pytest.mark.asyncio
async def test_raise_on_reject_raises_mutation_rejected():
    policy = RaiseOnReject()
    with pytest.raises(MutationRejected) as exc:
        await policy.handle(
            RejectedAt(stage="validation", reason="schema bad", actor_id="alice"),
            retries_so_far=0,
        )
    assert exc.value.stage == "validation"
    assert exc.value.actor_id == "alice"

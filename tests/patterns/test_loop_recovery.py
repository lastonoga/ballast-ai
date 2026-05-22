import pytest

from ballast.capabilities.helpers import Critique, SemanticLoopDetected
from ballast.patterns import AbortOnLoop, LoopRecoveryPolicy


def test_abort_on_loop_satisfies_protocol():
    assert isinstance(AbortOnLoop(), LoopRecoveryPolicy)


@pytest.mark.asyncio
async def test_abort_on_loop_raises_semantic_loop_detected():
    policy = AbortOnLoop()
    with pytest.raises(SemanticLoopDetected):
        await policy.handle(
            ctx=None,
            draft="some draft",
            feedback=[Critique(passed=False, issues=["repeats"])],
        )


def test_loop_recovery_policy_is_runtime_checkable():
    """Custom policies satisfy via structural typing (no base class)."""
    class MyPolicy:
        async def handle(self, ctx, draft, feedback):
            return draft

    assert isinstance(MyPolicy(), LoopRecoveryPolicy)

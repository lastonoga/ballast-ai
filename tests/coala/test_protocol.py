"""``CoALAUnit`` Protocol — structural type for memory-aware units."""
from __future__ import annotations

from ballast.coala import CoALAUnit


def test_runtime_checkable_protocol() -> None:
    class _Stub:
        async def observe(self, input): return input
        async def retrieve(self, observation): return {}
        async def act(self, observation, context): return None
        async def learn(self, observation, context, output): return None

    assert isinstance(_Stub(), CoALAUnit)


def test_protocol_rejects_missing_phase() -> None:
    class _NoLearn:
        async def observe(self, input): return input
        async def retrieve(self, observation): return {}
        async def act(self, observation, context): return None
        # learn missing

    assert not isinstance(_NoLearn(), CoALAUnit)

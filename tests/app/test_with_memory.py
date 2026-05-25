"""``Ballast.with_memory`` — fluent setter that installs EpisodicMemory."""
from __future__ import annotations

from ballast.app import Ballast
from ballast.memory import Scope
from ballast.memory.episodic import EpisodicMemory
from ballast.settings import BallastSettings


class _FakeSource:
    name = "f"
    async def recall(self, **_): return []
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


def test_with_memory_installs_facade() -> None:
    mem = EpisodicMemory(sources=[_FakeSource()])
    app = Ballast(BallastSettings()).with_memory(mem)
    assert app._memory is mem


def test_with_memory_scope_builder_propagates_to_facade() -> None:
    mem = EpisodicMemory(sources=[_FakeSource()])
    builder = lambda: Scope(user_id="from-test")
    app = Ballast(BallastSettings()).with_memory(mem, scope_builder=builder)
    assert mem._default_scope_builder is builder

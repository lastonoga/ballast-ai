"""``Ballast.with_semantic_memory`` + ``with_episodic_memory`` (rename)."""
from __future__ import annotations

import warnings

import pytest

from ballast.app import Ballast
from ballast.memory.episodic import EpisodicMemory
from ballast.memory.semantic import (
    DomainSemanticSource,
    SemanticMemory,
    memory_tool,
)
from ballast.settings import BallastSettings


class _EpisodicSource:
    name = "ep"
    async def recall(self, **_): return []
    async def hydrate(self, episode, *, detail): return episode
    async def remember(self, episode) -> None: return None


class _NotesSemantic(DomainSemanticSource):
    name = "notes"
    @memory_tool
    async def find_by_tag(self, tag: str) -> list[str]: return [tag]


def test_with_semantic_memory_installs_facade() -> None:
    sm = SemanticMemory(sources=[_NotesSemantic()])
    app = Ballast(BallastSettings()).with_semantic_memory(sm)
    assert app._semantic_memory is sm


def test_with_episodic_memory_replaces_with_memory() -> None:
    em = EpisodicMemory(sources=[_EpisodicSource()])
    app = Ballast(BallastSettings()).with_episodic_memory(em)
    assert app._episodic_memory is em


def test_with_memory_alias_still_works_but_warns() -> None:
    """Backward-compat alias — emits DeprecationWarning."""
    em = EpisodicMemory(sources=[_EpisodicSource()])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        app = Ballast(BallastSettings()).with_memory(em)
        assert app._episodic_memory is em
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_with_memory_still_shadows_old_attr_for_phase1_back_compat() -> None:
    """``self._memory`` mirrors ``self._episodic_memory`` so Phase 1
    consumers reading the old attr name continue to work."""
    em = EpisodicMemory(sources=[_EpisodicSource()])
    app = Ballast(BallastSettings()).with_episodic_memory(em)
    assert app._memory is em


def test_both_setters_chain() -> None:
    em = EpisodicMemory(sources=[_EpisodicSource()])
    sm = SemanticMemory(sources=[_NotesSemantic()])
    app = (
        Ballast(BallastSettings())
        .with_episodic_memory(em)
        .with_semantic_memory(sm)
    )
    assert app._episodic_memory is em
    assert app._semantic_memory is sm

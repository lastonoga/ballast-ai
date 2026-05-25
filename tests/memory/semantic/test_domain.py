"""``SemanticSource`` Protocol + ``DomainSemanticSource`` ABC."""
from __future__ import annotations

from ballast.memory.semantic import DomainSemanticSource, SemanticSource


def test_runtime_checkable_protocol() -> None:
    class _Stub:
        name = "stub"
    assert isinstance(_Stub(), SemanticSource)


def test_protocol_rejects_missing_name() -> None:
    class _NoName:
        pass
    assert not isinstance(_NoName(), SemanticSource)


def test_domain_semantic_source_is_subclass_of_protocol() -> None:
    class _MySource(DomainSemanticSource):
        name = "my"
    assert isinstance(_MySource(), SemanticSource)
    assert _MySource().name == "my"


def test_domain_semantic_source_can_be_subclassed_without_methods() -> None:
    """ABC has no abstract methods — subclassing alone is sufficient."""
    class _Empty(DomainSemanticSource):
        name = "empty"
    instance = _Empty()
    assert instance.name == "empty"

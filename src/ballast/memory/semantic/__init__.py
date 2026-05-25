"""Semantic memory — typed facts about the world via repo-wrappers."""
from ballast.memory.semantic._domain import DomainSemanticSource
from ballast.memory.semantic._protocol import SemanticSource

__all__ = ["DomainSemanticSource", "SemanticSource"]

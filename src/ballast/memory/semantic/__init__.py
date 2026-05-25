"""Semantic memory — typed facts about the world via repo-wrappers."""
from ballast.memory.semantic._decorator import memory_tool
from ballast.memory.semantic._domain import DomainSemanticSource
from ballast.memory.semantic._facade import SemanticMemory
from ballast.memory.semantic._protocol import SemanticSource
from ballast.memory.semantic._vector import VectorSemanticSource

__all__ = [
    "DomainSemanticSource",
    "SemanticMemory",
    "SemanticSource",
    "VectorSemanticSource",
    "memory_tool",
]

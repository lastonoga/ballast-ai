"""Recall strategies — pluggable reduction of federated source results."""
from ballast.memory.episodic.strategies._protocol import RecallStrategy
from ballast.memory.episodic.strategies._topk import TopK

__all__ = ["RecallStrategy", "TopK"]

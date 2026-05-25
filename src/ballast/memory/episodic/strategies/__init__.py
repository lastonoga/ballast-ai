"""Recall strategies — pluggable reduction of federated source results."""
from ballast.memory.episodic.strategies._all_relevant import AllRelevant
from ballast.memory.episodic.strategies._protocol import RecallStrategy
from ballast.memory.episodic.strategies._recency import Recency
from ballast.memory.episodic.strategies._topk import TopK

__all__ = ["AllRelevant", "RecallStrategy", "Recency", "TopK"]

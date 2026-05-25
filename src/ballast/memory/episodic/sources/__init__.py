"""Built-in episodic sources."""
from ballast.memory.episodic.sources._thread import ThreadEpisodicSource

try:
    from ballast.memory.episodic.sources._vector import (
        EMBEDDING_DIM,
        EpisodeRow,
        VectorEpisodicSource,
    )
except ImportError:
    EMBEDDING_DIM = None  # type: ignore[assignment]
    EpisodeRow = None  # type: ignore[assignment,misc]
    VectorEpisodicSource = None  # type: ignore[assignment,misc]

__all__ = [
    "EMBEDDING_DIM",
    "EpisodeRow",
    "ThreadEpisodicSource",
    "VectorEpisodicSource",
]

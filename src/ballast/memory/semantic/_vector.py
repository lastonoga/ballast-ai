"""``VectorSemanticSource`` — convenience base for free-text RAG sources."""
from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import SQLModel, select

from ballast.capabilities.helpers.embedder import Embedder
from ballast.memory.semantic._protocol import SemanticSource


class VectorSemanticSource(SemanticSource, ABC):
    """Base class for semantic sources backed by embedded free-text fields.

    Provides typical wiring (``embedder`` + ``sessionmaker``) and a
    helper ``_vector_search`` for the common cosine-distance query.
    Subclasses decide what to expose via ``@memory_tool`` — one search
    method per indexed corpus, or one method total, app's choice.

    The framework provides only the read-side helper. Apps own the
    embedding row schema and write-side indexing (typically a post-save
    hook on the domain repo).
    """

    name: ClassVar[str]

    def __init__(
        self,
        *,
        embedder: Embedder,
        sessionmaker: async_sessionmaker[AsyncSession],
    ) -> None:
        self._embedder = embedder
        self._sessionmaker = sessionmaker

    async def _vector_search(
        self,
        *,
        query: str,
        table: type[SQLModel],
        embedding_column: Any,           # e.g. MyRow.embedding
        k: int,
        where: Any | None = None,        # optional SQLAlchemy WHERE clause
    ) -> list[Any]:
        """Embed ``query`` and cosine-search ``table`` ordered by distance.

        Returns up to ``k`` rows. Subclasses typically project the
        result into a domain type before returning to the caller.
        """
        query_vec = await self._embedder.embed(query)
        async with self._sessionmaker() as session:
            stmt = select(table).order_by(embedding_column.cosine_distance(query_vec))
            if where is not None:
                stmt = stmt.where(where)
            stmt = stmt.limit(k)
            result = await session.execute(stmt)
            return list(result.scalars().all())


__all__ = ["VectorSemanticSource"]

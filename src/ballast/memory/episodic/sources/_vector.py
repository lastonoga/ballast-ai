"""``VectorEpisodicSource`` — pgvector-backed semantic recall."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, JSON, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import Field, SQLModel

from ballast.capabilities.helpers.embedder import Embedder
from ballast.memory._scope import Scope
from ballast.memory.episodic._models import (
    DetailLevel,
    Episode,
    ScoredEpisode,
)

EMBEDDING_DIM = 1536  # matches OpenAI text-embedding-3-small default
_JSON_PORTABLE = JSONB().with_variant(JSON(), "sqlite")


class EpisodeRow(SQLModel, table=True):
    """SQL row for a stored episode."""

    __tablename__ = "episodes"

    id: str = Field(primary_key=True)
    source: str = Field(index=True)
    user_id: str | None = Field(default=None, index=True)
    tenant_id: str | None = Field(default=None, index=True)
    thread_id: str | None = Field(default=None, index=True)
    preview: str
    summary: str | None = None
    full: dict[str, Any] | None = Field(
        default=None, sa_column=Column(_JSON_PORTABLE, nullable=True),
    )
    references_json: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(_JSON_PORTABLE, nullable=False),
    )
    metadata_json: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(_JSON_PORTABLE, nullable=False),
    )
    embedding: list[float] = Field(
        sa_column=Column(Vector(EMBEDDING_DIM), nullable=False),
    )
    occurred_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )


class VectorEpisodicSource:
    """Episodic source backed by Postgres + pgvector."""

    name = "vector"

    def __init__(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        embedder: Embedder,
    ) -> None:
        self._sm = sessionmaker
        self._embedder = embedder

    async def recall(
        self,
        *,
        intent: str,
        scope: Scope,
        k: int,
        detail: DetailLevel,
    ) -> list[ScoredEpisode]:
        query_vec = await self._embedder.embed(intent)
        async with self._sm() as session:
            stmt = (
                select(
                    EpisodeRow,
                    EpisodeRow.embedding.cosine_distance(query_vec).label("dist"),
                )
                .where(EpisodeRow.user_id == getattr(scope, "user_id", None))
            )
            tenant = getattr(scope, "tenant_id", None)
            if tenant is not None:
                stmt = stmt.where(EpisodeRow.tenant_id == tenant)
            stmt = stmt.order_by("dist").limit(k)
            rows = (await session.execute(stmt)).all()
        return [
            ScoredEpisode(
                episode=Episode(
                    id=row.id,
                    source=self.name,
                    occurred_at=row.occurred_at,
                    scope=Scope(
                        user_id=row.user_id,
                        tenant_id=row.tenant_id,
                        thread_id=row.thread_id,
                    ),
                    preview=row.preview,
                    summary=row.summary if detail >= DetailLevel.SUMMARY else None,
                    full=row.full if detail >= DetailLevel.FULL else None,
                    references=[],
                    metadata=row.metadata_json,
                ),
                score=1.0 - float(dist),
            )
            for row, dist in rows
        ]

    async def hydrate(self, episode: Episode, *, detail: DetailLevel) -> Episode:
        async with self._sm() as session:
            row = await session.get(EpisodeRow, episode.id)
            if row is None:
                return episode
            return episode.model_copy(update={
                "summary": row.summary if detail >= DetailLevel.SUMMARY else episode.summary,
                "full": row.full if detail >= DetailLevel.FULL else episode.full,
            })

    async def remember(self, episode: Episode) -> None:
        embed_text = episode.summary or episode.preview
        vec = await self._embedder.embed(embed_text)
        async with self._sm() as session:
            async with session.begin():
                row = EpisodeRow(
                    id=episode.id,
                    source=episode.source,
                    user_id=getattr(episode.scope, "user_id", None),
                    tenant_id=getattr(episode.scope, "tenant_id", None),
                    thread_id=getattr(episode.scope, "thread_id", None),
                    preview=episode.preview,
                    summary=episode.summary,
                    full=episode.full,
                    references_json=[],
                    metadata_json=episode.metadata,
                    embedding=vec,
                    occurred_at=episode.occurred_at,
                )
                session.add(row)


__all__ = ["EMBEDDING_DIM", "EpisodeRow", "VectorEpisodicSource"]

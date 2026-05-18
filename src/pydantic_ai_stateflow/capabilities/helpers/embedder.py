from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Structural type for an async text embedding service.

    Frameworks consume `Embedder` rather than `pydantic_ai.Embedder` directly
    so users can plug in their own (cached, local, batched, etc.) without
    depending on the pydantic-ai concrete class.
    """

    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

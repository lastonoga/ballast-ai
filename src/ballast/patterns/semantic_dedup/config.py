from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SemanticDedupConfig(BaseModel):
    """Serializable knobs for ``SemanticDedup``.

    Behavior (which fields of an item get embedded) is supplied as a
    plain ``Callable[[ItemT], str]`` to the pattern's constructor — it
    can't live in this schema (callables aren't serializable, and
    projecting arbitrary nested fields with a JSON-path mini-language
    would be a worse API than passing a lambda).

    Attributes:
      threshold: cosine-similarity cutoff. Two items with cosine ≥
        ``threshold`` collapse to one. Empirically, ``[0.88, 0.92]``
        keeps near-paraphrases together without eating sibling ideas.
      keep: which member of a near-duplicate cluster survives.

        * ``"first"`` — the earliest occurrence in input order
          (preserves caller intent / priority).
        * ``"longest"`` — the one with the longest projection
          (proxy for "more elaborated"). Useful when items merged from
          multiple agents have varying levels of detail.

    """

    threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    keep: Literal["first", "longest"] = "first"

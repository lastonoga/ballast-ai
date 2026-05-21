"""Shared pydantic schemas for the brainstorm flow.

Lives in its own module so ``brainstorm_agents`` (constructs the
pydantic-ai Agents with ``PromptedOutput(TodoIdeas)`` etc) and
``brainstorm_flow`` (orchestrates them) can both import without
introducing a circular dependency.
"""

from typing import Optional

from pydantic import BaseModel, Field


class TodoIdea(BaseModel):
    """One proposed todo. ``rationale`` is optional — the synthesizer
    fills it to explain why it picked / blended these candidates;
    divergent agents typically omit it."""
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=2000)
    rationale: Optional[str] = None


class TodoIdeas(BaseModel):
    """One divergent agent's batch — 1-5 ideas per call. Cap keeps the
    synthesizer's attention budget reasonable (CreativeDC quantity-
    distinctiveness tradeoff)."""
    ideas: list[TodoIdea] = Field(min_length=1, max_length=5)


__all__ = ["TodoIdea", "TodoIdeas"]

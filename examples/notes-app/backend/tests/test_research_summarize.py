"""``ResearchSummarize`` CoALAUnit — notes-app demo."""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from ballast.coala import CoALAUnit
from notes_app.coala.research_summarize import (
    ResearchObservation, ResearchQuery, ResearchSummarize,
)
from notes_app.repositories.note import InMemoryNoteRepository


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemoryNoteRepository]:
    fresh = InMemoryNoteRepository()
    monkeypatch.setattr("notes_app.repositories.note.notes_repo", fresh)
    yield fresh


def test_satisfies_coala_unit_protocol() -> None:
    assert isinstance(ResearchSummarize(), CoALAUnit)


@pytest.mark.asyncio
async def test_observe_extracts_intent_and_user(
    repo: InMemoryNoteRepository,
) -> None:
    unit = ResearchSummarize()
    obs = await unit.observe(ResearchQuery(user_query="ML in prod"))
    assert isinstance(obs, ResearchObservation)
    assert obs.intent == "ML in prod"


@pytest.mark.asyncio
async def test_retrieve_returns_empty_when_no_matching_notes(
    repo: InMemoryNoteRepository,
) -> None:
    unit = ResearchSummarize()
    obs = await unit.observe(ResearchQuery(user_query="anything"))
    ctx = await unit.retrieve(obs)
    assert ctx.related_notes == []


@pytest.mark.asyncio
async def test_retrieve_returns_search_matches(
    repo: InMemoryNoteRepository,
) -> None:
    await repo.create(title="ml-deployment", body="machine learning in prod")
    await repo.create(title="fashion", body="trends")
    unit = ResearchSummarize()
    obs = await unit.observe(ResearchQuery(user_query="machine learning"))
    ctx = await unit.retrieve(obs)
    titles = {n.title for n in ctx.related_notes}
    assert "ml-deployment" in titles
    assert "fashion" not in titles

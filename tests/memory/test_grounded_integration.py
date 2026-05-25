"""``scan_context`` recognizes ``Ref`` objects inside Episodes / RecallResults
so memory recall feeds grounded output schemas without special-casing."""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel

from ballast.grounded import Ref
from ballast.grounded._scan import scan_context
from ballast.grounded._spec import FieldRole, FieldSpec, OutputSpec
from ballast.memory import Scope
from ballast.memory.episodic import Episode, RecallResult, ScoredEpisode


class _Note(BaseModel):
    id: UUID


def _spec_expecting(target: type) -> OutputSpec:
    """Synthetic OutputSpec naming a single REF-typed field of `target`."""
    return OutputSpec(
        model=BaseModel,
        fields={
            "note": FieldSpec(
                name="note", path="note",
                role=FieldRole.REF, target_type=target,
            ),
        },
    )


def _ep(refs: list[Ref]) -> Episode:
    return Episode(
        id="ep", source="x", occurred_at=datetime.now(UTC),
        scope=Scope(), preview="p", references=refs,
    )


def test_scan_collects_refs_inside_recall_result() -> None:
    u1, u2 = uuid4(), uuid4()
    rr = RecallResult(episodes=[
        ScoredEpisode(episode=_ep([Ref[_Note](u1), Ref[_Note](u2)]), score=0.9),
    ])
    sources = scan_context(rr, _spec_expecting(_Note))
    assert _Note in sources.by_entity_type
    assert set(sources.by_entity_type[_Note]) == {u1, u2}


def test_scan_collects_refs_inside_loose_episode() -> None:
    u = uuid4()
    ep = _ep([Ref[_Note](u)])
    sources = scan_context(ep, _spec_expecting(_Note))
    assert sources.by_entity_type[_Note] == [u]


def test_scan_ignores_refs_of_unrelated_type() -> None:
    """A Ref[Other] shouldn't pollute the _Note collection."""
    class _Other(BaseModel):
        id: UUID
    u = uuid4()
    ep = _ep([Ref[_Other](u)])
    sources = scan_context(ep, _spec_expecting(_Note))
    assert _Note not in sources.by_entity_type   # nothing collected

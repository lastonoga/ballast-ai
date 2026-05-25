"""``ResearchSummarize`` CoALAUnit — demo for the CoALA architecture.

Mixed retrieval (search across notes via notes_repo) + custom act
(LLM-free placeholder summary) + custom learn (logs a learning record).

The unit demonstrates the 4-phase contract in a way that exercises:
  observe — parse user query into typed observation
  retrieve — query the relational notes repository
  act — synthesize a summary from retrieved corpus
  learn — record what was synthesized for future reference
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ballast.auth.context import current_user_id
from ballast.coala import CoALABase

from notes_app.models.note import Note


@dataclass
class ResearchQuery:
    user_query: str


@dataclass
class ResearchObservation:
    intent: str
    user_id: str | None


@dataclass
class ResearchContext:
    related_notes: list[Note] = field(default_factory=list)


@dataclass
class ResearchSummary:
    title: str
    body: str


class ResearchSummarize(CoALABase[
    ResearchQuery, ResearchObservation, ResearchContext, ResearchSummary,
]):
    """Summarize the user's recent research on a topic via notes_repo."""

    async def observe(self, q: ResearchQuery) -> ResearchObservation:
        return ResearchObservation(
            intent=q.user_query,
            user_id=current_user_id(),
        )

    async def retrieve(self, obs: ResearchObservation) -> ResearchContext:
        from notes_app.repositories.note import notes_repo  # noqa: PLC0415
        related = await notes_repo.search(obs.intent)
        return ResearchContext(related_notes=related[:10])

    async def act(
        self, obs: ResearchObservation, ctx: ResearchContext,
    ) -> ResearchSummary:
        if not ctx.related_notes:
            return ResearchSummary(
                title=f"No prior research on {obs.intent!r}",
                body="No matching notes found.",
            )
        bullets = "\n".join(
            f"- {n.title}: {n.body[:80]}" for n in ctx.related_notes
        )
        return ResearchSummary(
            title=f"Research: {obs.intent}",
            body=f"Found {len(ctx.related_notes)} prior notes:\n{bullets}",
        )

    async def learn(
        self,
        obs: ResearchObservation,
        ctx: ResearchContext,
        output: ResearchSummary,
    ) -> None:
        import logging  # noqa: PLC0415
        logging.getLogger("notes_app.coala").info(
            "research_summarize.learn user=%s intent=%s notes=%d title=%r",
            obs.user_id, obs.intent, len(ctx.related_notes), output.title,
        )

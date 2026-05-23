"""Assembles the ``DivergentConvergent`` instance the brainstorm flow uses.

Lives in its own module so the workflow body (``flow.py``) reads
top-to-bottom as orchestration — branch construction + synthesis
prompt rendering details stay here.

Notes-app demo never overrides the divergent knobs (best_of_n,
min_hypotheses, top_k, embedder, …) — they're inlined as
module-level constants instead of carried as ``_build`` kwargs. When
that changes, lift them back into the signature.
"""

from __future__ import annotations

from ballast import DivergentBranch, DivergentConvergent

from notes_app.agents.brainstorm import (
    BrainstormDivergentAgent,
    BrainstormSynthesizerAgent,
)
from notes_app.models.todo import TodoIdea, TodoIdeas
from notes_app.workflows.brainstorm.prompts import (
    CONVERGENT_PROMPT,
    DEFAULT_DIVERGENT_SPECS,
    DEFAULT_SYNTH_MODEL,
    DEFAULT_SYNTH_TEMPERATURE,
)


_BEST_OF_N: int = 1
_MIN_HYPOTHESES: int = 2
_DIVERGENT_CONCURRENCY: int = 3
_CONFIG_NAME: str = "notes-brainstorm-divergent"


def _format_synth_prompt(task: str, candidates: list[TodoIdea]) -> str:
    """Render the candidate pool into a synthesis prompt.

    Lives at the assembly boundary (not on the agent) — part of how
    THIS app wires the pattern, not the synthesizer's own behaviour.
    The pattern receives this as ``format_synth_prompt`` so unwrap
    (envelope → list) and prompt rendering both happen here."""
    lines = [f"Тема: {task}", "", "Кандидаты:"]
    for i, idea in enumerate(candidates, 1):
        lines.append(f"{i}. {idea.title} — {idea.body}")
    return "\n".join(lines)


def _build() -> DivergentConvergent[str, TodoIdeas, TodoIdea, TodoIdea]:
    """One-call assembly of the demo's divergent-convergent runner."""
    branches = tuple(
        DivergentBranch(
            label=spec.label,
            agent=BrainstormDivergentAgent(
                model_name=spec.model,
                system_prompt=spec.system_prompt,
                temperature=spec.temperature,
            ),
        )
        for spec in DEFAULT_DIVERGENT_SPECS
    )
    synthesizer = BrainstormSynthesizerAgent(
        model_name=DEFAULT_SYNTH_MODEL,
        system_prompt=CONVERGENT_PROMPT,
        temperature=DEFAULT_SYNTH_TEMPERATURE,
    )
    return DivergentConvergent[str, TodoIdeas, TodoIdea, TodoIdea](
        branches=branches,
        synthesizer=synthesizer,
        hypotheses=lambda env: env.ideas,
        format_synth_prompt=_format_synth_prompt,
        best_of_n=_BEST_OF_N,
        min_hypotheses=_MIN_HYPOTHESES,
        divergent_concurrency=_DIVERGENT_CONCURRENCY,
        config_name=_CONFIG_NAME,
    )


# Module-level singleton — the workflow body in ``flow.py`` calls
# ``divergent.run(topic)``.
divergent: DivergentConvergent[str, TodoIdeas, TodoIdea, TodoIdea] = _build()


__all__ = ["divergent"]

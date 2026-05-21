"""Notes-app brainstorm flow — thin user of the framework.

End-to-end: user clicks "Brainstorm todo" → POST /workflows/brainstorm-todo
→ ``BrainstormFlow.run(topic, parent_thread_id)`` →
``DivergentConvergent`` fans out to N different LLM agents (each with
its own model, temperature, system prompt), optionally dedups, optionally
verifies, picks one ``TodoIdea`` → opens ``TodoApprovalFlow`` for HITL →
on user approve the note saves and a "Saved your todo" message lands in
the parent thread.

All the heavy lifting lives in the framework:
- ``pydantic_ai_stateflow.patterns.DivergentConvergent`` —
  fan-out + dedup + verify + synthesise (durable, DBOS-queued).
- ``pydantic_ai_stateflow.patterns.SemanticDedup`` — optional dedup.
- ``pydantic_ai_stateflow.patterns.hitl.DurableHITLWorkflow`` —
  ``TodoApprovalFlow`` already subclasses it.

This file does ONLY app-specific gluing: pydantic schemas for ideas,
``DivergentAgent`` / ``Synthesizer`` adapters over pydantic-ai
``Agent`` (so the framework stays pydantic-ai-free), the default
divergent specs we ship with the demo, and a tiny BrainstormFlow class
that calls DivergentConvergent then TodoApprovalFlow.open.

No annotations-future import: pydantic-ai's tool decoration / output
schema introspection needs concrete types at decoration time (same
constraint as ``agent.py`` / ``todo_approval_agent.py``).
"""

import os
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from dbos import DBOS, DBOSConfiguredInstance
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai_stateflow import (
    DivergentBranch,
    DivergentConvergent,
    SemanticDedup,
    SemanticDedupConfig,
)
from pydantic_ai_stateflow.capabilities.helpers.embedder import Embedder

from notes_app.todo_approval_agent import (
    NotesTodoApprovalAgent,
    TodoApprovalContext,
)
from notes_app.todo_flow import TodoApprovalFlow


# ── Domain types ─────────────────────────────────────────────────────────

class TodoIdea(BaseModel):
    """One proposed todo. ``rationale`` is optional and mostly used by
    the synthesizer (it explains which candidate it picked and why)."""
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=2000)
    rationale: Optional[str] = None


class TodoIdeas(BaseModel):
    """One divergent agent's batch. 1-5 ideas per call — enough variety
    without blowing the synthesizer's attention budget per the
    CreativeDC quantity-distinctiveness tradeoff."""
    ideas: list[TodoIdea] = Field(min_length=1, max_length=5)


# ── Pydantic-AI ⇄ framework Protocol adapters ────────────────────────────

@dataclass(frozen=True)
class _PydanticAIDivergentAgent:
    """Adapts a pydantic-ai ``Agent`` to the framework's
    ``DivergentAgent[str, TodoIdea]`` protocol.

    Why an adapter and not a Protocol-satisfying Agent: the framework
    intentionally doesn't import pydantic-ai (it accepts any object
    with ``async def diverge(task) -> list[Hypothesis]``). Apps wrap
    their concrete agent in a one-liner here. Same pattern works for
    StateflowAgent / StateflowDurableAgent — use their ``.agent``
    property to grab the underlying pydantic-ai Agent.
    """
    agent: Agent[None, TodoIdeas]

    async def diverge(self, task: str) -> list[TodoIdea]:
        result = await self.agent.run(task)
        return result.output.ideas


@dataclass(frozen=True)
class _PydanticAISynthesizer:
    """Adapts pydantic-ai ``Agent`` to the framework's
    ``Synthesizer[str, TodoIdea, TodoIdea]`` protocol."""
    agent: Agent[None, TodoIdea]

    async def synthesize(self, *, task: str, candidates: list[TodoIdea]) -> TodoIdea:
        prompt = _format_synth_prompt(task, candidates)
        result = await self.agent.run(prompt)
        return result.output


def _format_synth_prompt(task: str, candidates: list[TodoIdea]) -> str:
    lines = [f"Тема: {task}", "", "Кандидаты:"]
    for i, idea in enumerate(candidates, 1):
        lines.append(f"{i}. {idea.title} — {idea.body}")
    return "\n".join(lines)


# ── Default agent specs for the notes-app demo ───────────────────────────

PRACTICAL_PROMPT = (
    "Ты практик. Тема — это контекст для дел, которые можно сделать "
    "СЕГОДНЯ или на этой неделе. Без абстракций, без 'подумать о…'. "
    "Каждая идея — конкретное действие с измеримым результатом. "
    "Верни 2-3 идеи."
)

CREATIVE_PROMPT = (
    "Ты креатив. Тема — повод для неожиданных, играющих, может быть "
    "слегка дерзких идей. Не бойся странных формулировок. Никаких "
    "очевидных 'купить продукты'. Верни 2-3 идеи."
)

ANALYTICAL_PROMPT = (
    "Ты аналитик. Перед тем как предложить идею, мысленно разбей тему "
    "на под-цели. Каждая идея = шаг в декомпозиции, помеченный "
    "приоритетом в title (например 'P0: …', 'P1: …'). Верни 3-5 идей."
)

CONVERGENT_PROMPT = (
    "Ниже — пул идей todo от нескольких агентов. Тема в первой строке. "
    "Твоя задача — выбрать ОДНУ финальную идею. Можно слегка "
    "отредактировать формулировку или слить две похожие. В rationale "
    "напиши 1-2 предложения: почему именно эта."
)


def _build_openrouter_agent(
    *,
    model_name: str,
    system_prompt: str,
    temperature: float,
    output_type: type,
) -> Agent[None, Any]:
    """Plain pydantic-ai Agent factory over OpenRouter.

    Lives here (not in framework) on purpose — the framework's
    DivergentConvergent / SemanticDedup are provider-agnostic.
    Apps pick their own model SDK and build agents however they like.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY required for brainstorm flow")
    model = OpenRouterModel(
        model_name,
        provider=OpenRouterProvider(api_key=api_key),
        settings=OpenRouterModelSettings(temperature=temperature),
    )
    return Agent(model=model, output_type=output_type, system_prompt=system_prompt)


@dataclass(frozen=True)
class BrainstormAgentSpec:
    """User-facing spec for one branch in the divergent fan-out.

    App passes a tuple of these to ``build_brainstorm_flow`` and the
    factory wraps each in the framework's ``DivergentBranch`` /
    ``DivergentAgent`` plumbing. Apps that want a different provider
    just construct their own ``DivergentBranch`` directly and bypass
    this helper.
    """
    label: str
    model: str
    system_prompt: str
    temperature: float = 0.9


DEFAULT_DIVERGENT_SPECS: tuple[BrainstormAgentSpec, ...] = (
    BrainstormAgentSpec("practical", "qwen/qwen3.6-plus", PRACTICAL_PROMPT),
    BrainstormAgentSpec("creative", "deepseek/deepseek-v3", CREATIVE_PROMPT),
    BrainstormAgentSpec("analyst", "openai/gpt-4o-mini", ANALYTICAL_PROMPT),
)

DEFAULT_SYNTH_MODEL = "anthropic/claude-3.7-sonnet"
DEFAULT_SYNTH_TEMPERATURE = 0.2


# ── BrainstormFlow ───────────────────────────────────────────────────────

@DBOS.dbos_class()
class BrainstormFlow(DBOSConfiguredInstance):
    """Glue between ``DivergentConvergent`` and ``TodoApprovalFlow``.

    Owns one DivergentConvergent instance (constructed at __init__ so
    its ``DBOS.Queue`` registers before ``DBOS.launch()``) and one
    reference to the app's TodoApprovalFlow. ``run`` is a workflow:
    invokes divergent-convergent to pick a single idea, then hands it
    to the HITL flow.

    Stable ``config_name`` so DBOS can rebind in-flight workflows back
    to this instance after a restart.
    """

    def __init__(
        self,
        *,
        todo_flow: TodoApprovalFlow,
        divergent: DivergentConvergent[str, TodoIdea, TodoIdea],
        config_name: str = "notes-brainstorm-flow",
    ) -> None:
        super().__init__(config_name=config_name)
        self._todo_flow = todo_flow
        self._divergent = divergent

    @DBOS.workflow()
    async def run(self, *, topic: str, parent_thread_id: UUID) -> UUID:
        """Run the brainstorm + open HITL. Returns helper thread id."""
        chosen: TodoIdea = await self._divergent.run(topic)
        helper_thread = await self._todo_flow.open(
            helper_agent=NotesTodoApprovalAgent,
            context=TodoApprovalContext(
                proposed_title=chosen.title,
                proposed_body=chosen.body,
                parent_thread_id=parent_thread_id,
            ),
            notify_parent_thread_id=parent_thread_id,
        )
        return helper_thread.id


# ── Factory — assembles the demo wiring ──────────────────────────────────

def build_brainstorm_flow(
    *,
    todo_flow: TodoApprovalFlow,
    divergent_specs: tuple[BrainstormAgentSpec, ...] = DEFAULT_DIVERGENT_SPECS,
    synth_model: str = DEFAULT_SYNTH_MODEL,
    synth_temperature: float = DEFAULT_SYNTH_TEMPERATURE,
    embedder: Embedder | None = None,
    dedup_threshold: float = 0.9,
    best_of_n: int = 1,
    min_hypotheses: int = 2,
    top_k: int | None = None,
    divergent_concurrency: int = 3,
    config_name: str = "notes-brainstorm-flow",
) -> BrainstormFlow:
    """One-call wiring for the notes-app demo.

    All knobs surfaced here are forwarded to the framework. Pass
    ``embedder`` to enable semantic dedup; leave ``None`` to skip the
    dedup stage entirely (the framework treats deduper as optional).

    Apps doing serious work should construct ``DivergentConvergent``
    + ``SemanticDedup`` themselves to plug a custom verifier, a
    different model SDK, mocks for tests, etc — this factory is just
    the default wiring.
    """
    branches = tuple(
        DivergentBranch(
            label=spec.label,
            agent=_PydanticAIDivergentAgent(
                agent=_build_openrouter_agent(
                    model_name=spec.model,
                    system_prompt=spec.system_prompt,
                    temperature=spec.temperature,
                    output_type=TodoIdeas,
                ),
            ),
        )
        for spec in divergent_specs
    )

    synthesizer = _PydanticAISynthesizer(
        agent=_build_openrouter_agent(
            model_name=synth_model,
            system_prompt=CONVERGENT_PROMPT,
            temperature=synth_temperature,
            output_type=TodoIdea,
        ),
    )

    deduper: SemanticDedup[TodoIdea] | None = None
    if embedder is not None:
        deduper = SemanticDedup[TodoIdea](
            embedder=embedder,
            projector=lambda i: f"{i.title}\n{i.body}",
            config=SemanticDedupConfig(threshold=dedup_threshold, keep="longest"),
        )

    divergent = DivergentConvergent[str, TodoIdea, TodoIdea](
        branches=branches,
        synthesizer=synthesizer,
        deduper=deduper,
        best_of_n=best_of_n,
        min_hypotheses=min_hypotheses,
        top_k=top_k,
        divergent_concurrency=divergent_concurrency,
        config_name=f"{config_name}-divergent",
    )

    return BrainstormFlow(
        todo_flow=todo_flow, divergent=divergent, config_name=config_name,
    )


__all__ = [
    "BrainstormAgentSpec",
    "BrainstormFlow",
    "DEFAULT_DIVERGENT_SPECS",
    "TodoIdea",
    "TodoIdeas",
    "build_brainstorm_flow",
]

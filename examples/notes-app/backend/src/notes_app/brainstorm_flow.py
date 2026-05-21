"""Notes-app brainstorm flow — thin user of the framework.

End-to-end: user clicks "Brainstorm todo" → POST /workflows/brainstorm-todo
→ ``BrainstormFlow.run(topic, parent_thread_id)`` →
``DivergentConvergent`` fans out to N different LLM agents (each with
its own model, temperature, system prompt), optionally dedups,
optionally verifies, picks one ``TodoIdea`` → opens
``TodoApprovalFlow`` for HITL → on user approve the note saves and a
"Saved your todo" message lands in the parent thread.

All the heavy lifting lives in the framework:
- ``pydantic_ai_stateflow.patterns.DivergentConvergent`` —
  fan-out + dedup + verify + synthesise (durable, DBOS-queued).
- ``pydantic_ai_stateflow.patterns.SemanticDedup`` — optional dedup.
- ``pydantic_ai_stateflow.patterns.hitl.DurableHITLWorkflow`` —
  ``TodoApprovalFlow`` already subclasses it.

App-specific glue here:
- ``BrainstormDivergentAgent`` / ``BrainstormSynthesizerAgent``
  (``brainstorm_agents.py``) are real ``StateflowAgent`` subclasses
  so we reuse the codebase's existing OpenRouter wiring + model
  settings + capabilities slot.
- Tiny adapters turn each ``StateflowAgent`` into the framework's
  ``DivergentAgent`` / ``Synthesizer`` protocols (which are
  pydantic-ai-agnostic by design).
- ``BrainstormFlow`` workflow chains divergent-convergent into
  ``TodoApprovalFlow.open`` for HITL.

No ``from __future__ import annotations``: pydantic-ai introspects
``get_type_hints()`` at decoration time (same as ``agent.py`` /
``todo_approval_agent.py``).
"""

from dataclasses import dataclass
from uuid import UUID

from dbos import DBOS, DBOSConfiguredInstance
from pydantic_ai_stateflow import (
    DivergentBranch,
    DivergentConvergent,
    SemanticDedup,
    SemanticDedupConfig,
)
from pydantic_ai_stateflow.capabilities.helpers.embedder import Embedder
from pydantic_ai_stateflow.runtime import StateflowAgent

from notes_app.brainstorm_agents import (
    BrainstormDivergentAgent,
    BrainstormSynthesizerAgent,
)
from notes_app.brainstorm_types import TodoIdea, TodoIdeas
from notes_app.todo_approval_agent import (
    NotesTodoApprovalAgent,
    TodoApprovalContext,
)
from notes_app.todo_flow import TodoApprovalFlow


# ── Stateflow ⇄ framework Protocol adapters ──────────────────────────────

@dataclass(frozen=True)
class _StateflowDivergentAgent:
    """Adapts a ``StateflowAgent`` to the framework's
    ``DivergentAgent[str, TodoIdea]`` protocol.

    The framework intentionally doesn't depend on ``StateflowAgent``
    (let alone pydantic-ai) — it accepts any object with
    ``async def diverge(task) -> list[Hypothesis]``. This adapter is
    the seam: it calls into ``stateflow_agent.agent.run(...)`` so
    we automatically pick up ``model_settings()`` and any capabilities
    the StateflowAgent layered on top.
    """
    stateflow_agent: StateflowAgent

    async def diverge(self, task: str) -> list[TodoIdea]:
        result = await self.stateflow_agent.agent.run(
            task, model_settings=self.stateflow_agent.model_settings(),
        )
        ideas: TodoIdeas = result.output
        return ideas.ideas


@dataclass(frozen=True)
class _StateflowSynthesizer:
    """Adapts a ``StateflowAgent`` to the framework's
    ``Synthesizer[str, TodoIdea, TodoIdea]`` protocol."""
    stateflow_agent: StateflowAgent

    async def synthesize(self, *, task: str, candidates: list[TodoIdea]) -> TodoIdea:
        prompt = _format_synth_prompt(task, candidates)
        result = await self.stateflow_agent.agent.run(
            prompt, model_settings=self.stateflow_agent.model_settings(),
        )
        chosen: TodoIdea = result.output
        return chosen


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


@dataclass(frozen=True)
class BrainstormAgentSpec:
    """User-facing spec for one branch in the divergent fan-out.

    App passes a tuple of these to ``build_brainstorm_flow`` and the
    factory wraps each in a ``BrainstormDivergentAgent``. Apps that
    want a different provider construct their own ``DivergentBranch``
    directly with a custom ``StateflowAgent`` and bypass this helper.
    """
    label: str
    model: str
    system_prompt: str
    temperature: float = 0.9


DEFAULT_DIVERGENT_SPECS: tuple[BrainstormAgentSpec, ...] = (
    BrainstormAgentSpec("practical", "qwen/qwen3.6-plus", PRACTICAL_PROMPT),
    BrainstormAgentSpec("creative", "deepseek/deepseek-chat-v3.1", CREATIVE_PROMPT),
    BrainstormAgentSpec("analyst", "openai/gpt-4o-mini", ANALYTICAL_PROMPT),
)

DEFAULT_SYNTH_MODEL = "anthropic/claude-sonnet-4.6"
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

    Builds one ``BrainstormDivergentAgent`` per spec and one
    ``BrainstormSynthesizerAgent``, wraps each in an adapter, hands
    the bundle to ``DivergentConvergent``. Pass ``embedder`` to enable
    semantic dedup; leave ``None`` to skip dedup entirely.

    Apps doing serious work should construct ``DivergentConvergent``
    + ``SemanticDedup`` themselves (custom verifier, mocks for tests,
    different model SDKs) — this factory is just the demo wiring.
    """
    branches = tuple(
        DivergentBranch(
            label=spec.label,
            agent=_StateflowDivergentAgent(
                stateflow_agent=BrainstormDivergentAgent(
                    model_name=spec.model,
                    system_prompt=spec.system_prompt,
                    temperature=spec.temperature,
                ),
            ),
        )
        for spec in divergent_specs
    )

    synthesizer = _StateflowSynthesizer(
        stateflow_agent=BrainstormSynthesizerAgent(
            model_name=synth_model,
            system_prompt=CONVERGENT_PROMPT,
            temperature=synth_temperature,
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

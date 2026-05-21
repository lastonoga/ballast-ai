"""Notes-app brainstorm flow — thin user of the framework.

End-to-end: user clicks "Brainstorm todo" → POST /workflows/brainstorm-todo
→ ``BrainstormFlow.run(BrainstormTask(...))`` →
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
  that ALSO implement the framework's structural ``DivergentAgent`` /
  ``Synthesizer`` protocols directly (``.diverge`` / ``.synthesize``).
  No separate adapter layer — the agent IS the branch.
- ``BrainstormFlow`` workflow chains divergent-convergent into
  ``TodoApprovalFlow.open`` for HITL.

No ``from __future__ import annotations``: pydantic-ai introspects
``get_type_hints()`` at decoration time (same as ``agent.py`` /
``todo_approval_agent.py``).
"""

from dataclasses import dataclass
from typing import Literal, Optional
from uuid import UUID

from dbos import DBOSConfiguredInstance
from pydantic import BaseModel
from pydantic_ai_stateflow import (
    DivergentBranch,
    DivergentConvergent,
    Durable,
    SemanticDedup,
    SemanticDedupConfig,
    ThreadEventBroadcaster,
    ThreadEventType,
)
from pydantic_ai_stateflow.capabilities.helpers.embedder import Embedder
from pydantic_ai_stateflow.patterns.divergent_convergent.events import (
    BranchCompleted,
    BranchEnqueued,
    BranchFailed,
    DivergentEvent,
)

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


# ── Live progress events ─────────────────────────────────────────────────
#
# Brainstorm runs ~25s end-to-end; the user staring at the chat sees
# nothing happening unless we narrate progress. ``BRAINSTORM_PROGRESS``
# is a streaming thread event: one ``message_id`` reused across the
# whole run, so the UI shows ONE animating row that mutates through
# the phases (diverge → converge → hitl) instead of N separate
# messages piling up.


class BrainstormProgress(BaseModel):
    """Snapshot of where ``BrainstormFlow.run`` currently is.

    Frontend renders this as a single line that mutates: an icon
    flips ``running → ok`` per phase, with optional context in
    ``detail`` (e.g. the chosen idea's title once converge finishes).
    """
    step: Literal["diverge", "converge", "hitl"]
    status: Literal["running", "ok", "failed"]
    detail: Optional[str] = None


BRAINSTORM_PROGRESS = ThreadEventType("brainstorm-progress", BrainstormProgress)
"""Wire name on the part is ``data-brainstorm-progress`` — frontend
``makeAssistantDataUI({name: "brainstorm-progress"})`` matches by the
suffix (assistant-ui strips the ``data-`` prefix internally)."""


class BrainstormBranchProgress(BaseModel):
    """Per-branch live status during the divergent fan-out.

    One of these mutates per ``(label, sample_idx)`` pair as the
    branch transitions ``running → ok|failed``. Frontend renders the
    bundle so the user sees individual proposers tick off in parallel
    rather than a single opaque "brainstorming" spinner.
    """
    label: str
    sample_idx: int
    status: Literal["running", "ok", "failed"]
    pool_size: Optional[int] = None
    error_type: Optional[str] = None


BRAINSTORM_BRANCH = ThreadEventType("brainstorm-branch", BrainstormBranchProgress)
"""Wire name ``data-brainstorm-branch``. Each (label, sample_idx) pair
gets a deterministic ``message_id`` so the row reused across the
``running → ok|failed`` updates instead of stacking up."""


def _branch_message_id(parent_thread_id: UUID, label: str, sample_idx: int) -> str:
    """Deterministic message id per branch — stable across workflow
    replay (same parent thread + same branch identity → same id), so
    DBOS retries don't multiply rows in the UI."""
    return f"brainstorm-branch::{parent_thread_id}::{label}::{sample_idx}"


def _make_branch_progress_callback(
    broadcaster: ThreadEventBroadcaster, parent_thread_id: UUID,
):
    """Build an ``on_progress`` callback that maps framework
    ``DivergentEvent``s to per-branch thread events.

    Only branch-level events fan out as ``BRAINSTORM_BRANCH``; the
    coarse-grained ``BRAINSTORM_PROGRESS`` stream still narrates
    diverge/converge/hitl on its own message.
    """
    async def on_progress(event: DivergentEvent) -> None:
        if isinstance(event, BranchEnqueued):
            await BRAINSTORM_BRANCH.emit(
                broadcaster, parent_thread_id,
                BrainstormBranchProgress(
                    label=event.label, sample_idx=event.sample_idx,
                    status="running",
                ),
                message_id=_branch_message_id(
                    parent_thread_id, event.label, event.sample_idx,
                ),
            )
        elif isinstance(event, BranchCompleted):
            await BRAINSTORM_BRANCH.emit(
                broadcaster, parent_thread_id,
                BrainstormBranchProgress(
                    label=event.label, sample_idx=event.sample_idx,
                    status="ok", pool_size=event.pool_size,
                ),
                message_id=_branch_message_id(
                    parent_thread_id, event.label, event.sample_idx,
                ),
            )
        elif isinstance(event, BranchFailed):
            await BRAINSTORM_BRANCH.emit(
                broadcaster, parent_thread_id,
                BrainstormBranchProgress(
                    label=event.label, sample_idx=event.sample_idx,
                    status="failed", error_type=event.error_type,
                ),
                message_id=_branch_message_id(
                    parent_thread_id, event.label, event.sample_idx,
                ),
            )
    return on_progress


# ── BrainstormFlow ───────────────────────────────────────────────────────

class BrainstormTask(BaseModel):
    """Input to ``BrainstormFlow.run`` — one pydantic envelope so the
    workflow's call signature stays stable as the inputs grow (extra
    knobs like ``best_of_n_override``, ``locale`` etc. can be added
    without breaking callers)."""
    topic: str
    parent_thread_id: UUID


class BrainstormOutcome(BaseModel):
    """Output of ``BrainstormFlow.run``.

    The flow is fire-and-forget w.r.t. HITL: ``run`` returns AFTER the
    approval thread is opened but BEFORE the user approves/rejects.
    ``helper_thread_id`` is what the UI needs to scroll the sidebar
    to. ``proposed_title`` / ``proposed_body`` are included so
    observability (and any caller that wants to log what was
    proposed) doesn't need to peek into the helper thread."""
    helper_thread_id: UUID
    proposed_title: str
    proposed_body: str


@Durable.dbos_class()
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
        divergent: DivergentConvergent[str, TodoIdeas, TodoIdea, TodoIdea],
        broadcaster: ThreadEventBroadcaster,
        config_name: str = "notes-brainstorm-flow",
    ) -> None:
        super().__init__(config_name=config_name)
        self._todo_flow = todo_flow
        self._divergent = divergent
        self._broadcaster = broadcaster

    @Durable.workflow()
    async def run(self, task: BrainstormTask) -> BrainstormOutcome:
        """Run the brainstorm + open HITL. Returns the proposed idea
        and the helper thread id (fire-and-forget on the actual
        approval — see ``TodoApprovalFlow.on_decision`` for the
        approve/reject side effects).

        Emits live ``brainstorm-progress`` events into the parent
        thread through a single stream session: same ``message_id``
        across all updates → the UI sees ONE animating row instead
        of N rows.
        """
        parent_thread_id = task.parent_thread_id
        topic = task.topic
        async with BRAINSTORM_PROGRESS.stream(
            self._broadcaster, parent_thread_id,
        ) as progress:
            await progress.update(BrainstormProgress(
                step="diverge", status="running",
                detail=f'Topic: "{topic}"',
            ))
            branch_callback = _make_branch_progress_callback(
                self._broadcaster, parent_thread_id,
            )
            chosen: TodoIdea = await self._divergent.run(
                topic, on_progress=branch_callback,
            )
            await progress.update(BrainstormProgress(
                step="diverge", status="ok",
            ))

            await progress.update(BrainstormProgress(
                step="converge", status="ok",
                detail=f'Chosen: "{chosen.title}"',
            ))

            await progress.update(BrainstormProgress(
                step="hitl", status="running",
                detail="Opening approval thread…",
            ))
            context = TodoApprovalContext(
                proposed_title=chosen.title,
                proposed_body=chosen.body,
                parent_thread_id=parent_thread_id,
            )
            helper_thread = await self._todo_flow.open(
                helper_agent=NotesTodoApprovalAgent,
                context=context,
                opening_message=context.to_opening_message(),
                notify_parent_thread_id=parent_thread_id,
            )
            await progress.update(BrainstormProgress(
                step="hitl", status="ok",
                detail="Approval thread opened in sidebar →",
            ))
        return BrainstormOutcome(
            helper_thread_id=helper_thread.id,
            proposed_title=chosen.title,
            proposed_body=chosen.body,
        )


# ── Factory — assembles the demo wiring ──────────────────────────────────

def _format_synth_prompt(task: str, candidates: list[TodoIdea]) -> str:
    """Render the candidate pool into a synthesis prompt.

    Lives in the factory module (not on the agent) — it's part of how
    THIS app wires the pattern, not part of the synthesizer's own
    behaviour. The pattern receives it as ``format_synth_prompt`` so
    the unwrap (envelope → list) and the prompt-rendering both live
    at the same boundary."""
    lines = [f"Тема: {task}", "", "Кандидаты:"]
    for i, idea in enumerate(candidates, 1):
        lines.append(f"{i}. {idea.title} — {idea.body}")
    return "\n".join(lines)


def build_brainstorm_flow(
    *,
    todo_flow: TodoApprovalFlow,
    broadcaster: ThreadEventBroadcaster,
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
    ``BrainstormSynthesizerAgent`` — these implement the framework's
    ``DivergentAgent`` / ``Synthesizer`` structural protocols directly,
    so they're handed to ``DivergentConvergent`` without an adapter
    layer. Pass ``embedder`` to enable semantic dedup; leave ``None``
    to skip dedup entirely.

    Apps doing serious work should construct ``DivergentConvergent``
    + ``SemanticDedup`` themselves (custom verifier, mocks for tests,
    different model SDKs) — this factory is just the demo wiring.
    """
    branches = tuple(
        DivergentBranch(
            label=spec.label,
            agent=BrainstormDivergentAgent(
                model_name=spec.model,
                system_prompt=spec.system_prompt,
                temperature=spec.temperature,
            ),
        )
        for spec in divergent_specs
    )

    synthesizer = BrainstormSynthesizerAgent(
        model_name=synth_model,
        system_prompt=CONVERGENT_PROMPT,
        temperature=synth_temperature,
    )

    deduper: SemanticDedup[TodoIdea] | None = None
    if embedder is not None:
        deduper = SemanticDedup[TodoIdea](
            embedder=embedder,
            projector=lambda i: f"{i.title}\n{i.body}",
            config=SemanticDedupConfig(threshold=dedup_threshold, keep="longest"),
        )

    divergent = DivergentConvergent[str, TodoIdeas, TodoIdea, TodoIdea](
        branches=branches,
        synthesizer=synthesizer,
        hypotheses=lambda env: env.ideas,
        format_synth_prompt=_format_synth_prompt,
        deduper=deduper,
        best_of_n=best_of_n,
        min_hypotheses=min_hypotheses,
        top_k=top_k,
        divergent_concurrency=divergent_concurrency,
        config_name=f"{config_name}-divergent",
    )

    return BrainstormFlow(
        todo_flow=todo_flow,
        divergent=divergent,
        broadcaster=broadcaster,
        config_name=config_name,
    )


__all__ = [
    "BrainstormAgentSpec",
    "BrainstormFlow",
    "BrainstormOutcome",
    "BrainstormTask",
    "DEFAULT_DIVERGENT_SPECS",
    "TodoIdea",
    "TodoIdeas",
    "build_brainstorm_flow",
]

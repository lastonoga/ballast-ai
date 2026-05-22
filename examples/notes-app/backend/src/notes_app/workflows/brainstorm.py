"""Notes-app brainstorm flow — thin user of the framework.

End-to-end: user clicks "Brainstorm todo" → POST /workflows/brainstorm-flow
→ ``BrainstormFlow.run(BrainstormTask(...))`` →
``DivergentConvergent`` fans out to N different LLM agents (each with
its own model, temperature, system prompt), optionally dedups,
optionally verifies, picks one ``TodoIdea`` → opens
``TodoApprovalFlow`` for HITL → on user approve the note saves and a
"Saved your todo" message lands in the parent thread.

All the heavy lifting lives in the framework:
- ``ballast.patterns.DivergentConvergent`` —
  fan-out + dedup + verify + synthesise (durable, DBOS-queued).
- ``ballast.patterns.SemanticDedup`` — optional dedup.
- ``ballast.patterns.hitl.DurableHITLWorkflow`` —
  ``TodoApprovalFlow`` already subclasses it.

App-specific glue here:
- ``BrainstormDivergentAgent`` / ``BrainstormSynthesizerAgent``
  (``notes_app.agents.brainstorm``) are real ``BallastAgent``
  subclasses that ALSO implement the framework's structural
  ``DivergentAgent`` / ``Synthesizer`` protocols directly
  (``.diverge`` / ``.synthesize``). No separate adapter layer — the
  agent IS the branch.
- ``BrainstormFlow`` workflow chains divergent-convergent into
  ``TodoApprovalFlow.open`` for HITL.

No ``from __future__ import annotations``: pydantic-ai introspects
``get_type_hints()`` at decoration time (same as
``notes_app.agents.notes`` / ``notes_app.agents.todo_approval``).
"""

from dataclasses import dataclass
from uuid import UUID

from dbos import DBOSConfiguredInstance
from ballast import (
    DivergentBranch,
    DivergentConvergent,
    Durable,
    SemanticDedup,
    SemanticDedupConfig,
    ThreadEventBroadcaster,
    ThreadEventType,
    get_engine,
)
from ballast.capabilities.helpers.embedder import Embedder
from ballast.patterns.divergent_convergent.events import (
    BranchCompleted,
    BranchEnqueued,
    BranchFailed,
    DivergentEvent,
)

from notes_app.agents.brainstorm import (
    BrainstormDivergentAgent,
    BrainstormSynthesizerAgent,
)
from notes_app.agents.todo_approval import NotesTodoApprovalAgent
from notes_app.models.brainstorm import BrainstormOutcome, BrainstormTask
from notes_app.models.progress import BrainstormBranchProgress, BrainstormProgress
from notes_app.models.todo import TodoIdea, TodoIdeas
from notes_app.models.todo_approval import TodoApprovalContext


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
    directly with a custom ``BallastAgent`` and bypass this helper.
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


BRAINSTORM_PROGRESS = ThreadEventType("brainstorm-progress", BrainstormProgress)
"""Wire name on the part is ``data-brainstorm-progress`` — frontend
``makeAssistantDataUI({name: "brainstorm-progress"})`` matches by the
suffix (assistant-ui strips the ``data-`` prefix internally)."""


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

    The app mounts ``POST /workflows/brainstorm-flow`` explicitly in
    ``notes_app.routes.workflows`` — the framework no longer
    auto-generates workflow routes.
    """

    @staticmethod
    def workflow_id(task: BrainstormTask) -> str:
        """Deterministic workflow id for the HTTP route.

        Same (parent_thread, topic) → same workflow id so duplicate
        clicks collapse to one in-flight workflow (matches the
        historical ``brainstorm_router.py`` behaviour)."""
        return f"brainstorm:{task.parent_thread_id}:{abs(hash(task.topic))}"

    def __init__(
        self,
        *,
        divergent: DivergentConvergent[str, TodoIdeas, TodoIdea, TodoIdea],
        config_name: str = "notes-brainstorm-flow",
    ) -> None:
        super().__init__(config_name=config_name)
        self._divergent = divergent

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

        Pulls the broadcaster off the framework's process-wide
        ``Engine`` — no per-call ``RunContext`` is needed now that
        ``ballast.create_app`` installs the singleton at startup.
        """
        parent_thread_id = task.parent_thread_id
        topic = task.topic
        broadcaster = get_engine().broadcaster
        async with BRAINSTORM_PROGRESS.stream(
            broadcaster, parent_thread_id,
        ) as progress:
            await progress.update(BrainstormProgress(
                step="diverge", status="running",
                detail=f'Topic: "{topic}"',
            ))
            branch_callback = _make_branch_progress_callback(
                broadcaster, parent_thread_id,
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
            # Direct import of the module-level singleton.
            from notes_app.workflows.todo_approval import todo_flow

            context = TodoApprovalContext(
                proposed_title=chosen.title,
                proposed_body=chosen.body,
                parent_thread_id=parent_thread_id,
            )
            helper_thread = await todo_flow.open(
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
        divergent=divergent,
        config_name=config_name,
    )


# ── Module-level singleton ──────────────────────────────────────────────
# App-specific brainstorm flow. Imported directly by
# ``notes_app.routes.workflows``'s ``POST /workflows/brainstorm-flow``
# route handler.

brainstorm: BrainstormFlow = build_brainstorm_flow()


__all__ = [
    "BRAINSTORM_BRANCH",
    "BRAINSTORM_PROGRESS",
    "BrainstormAgentSpec",
    "BrainstormFlow",
    "DEFAULT_DIVERGENT_SPECS",
    "brainstorm",
    "build_brainstorm_flow",
]

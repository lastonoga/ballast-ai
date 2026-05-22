"""Notes-app brainstorm flow — thin user of the framework.

End-to-end: user clicks "Brainstorm todo" → POST /workflows/brainstorm-flow
→ ``brainstorm(BrainstormTask(...))`` →
``DivergentConvergent`` fans out to N different LLM agents (each with
its own model, temperature, system prompt) → picks one ``TodoIdea`` →
``ask_human()`` opens a helper thread and BLOCKS for the user's
verdict → on approve the note saves (with edits if modified) and a
"Saved your todo" message lands in the parent thread; on reject /
timeout the parent gets a cancellation message.

Architecture: this is a single ``@Durable.workflow`` async function.
No class wrapper, no ``on_decision`` callback split — the flow reads
top-to-bottom like the imperative pipeline it is. HITL is one
``await`` in the middle and the verdict-handling logic sits inline.

Progress is shown to the user as plain chat messages — one per phase
("Brainstorming on …", "Chose …", "Saved …"). Each ``say()`` call
appends an assistant message; the framework's ``message_added``
signal handler emits the log + stream side effects automatically so
the SSE consumer on the parent thread sees the timeline live.

All the heavy lifting lives in the framework:
- ``ballast.patterns.DivergentConvergent`` — fan-out + dedup + verify
  + synthesise (durable, DBOS-queued).
- ``ballast.patterns.SemanticDedup`` — optional dedup.
- ``ballast.ask_human`` — await-style durable HITL primitive.

App-specific glue here:
- ``BrainstormDivergentAgent`` / ``BrainstormSynthesizerAgent``
  (``notes_app.agents.brainstorm``) — real ``BallastAgent``
  subclasses that ALSO implement the framework's structural
  ``DivergentAgent`` / ``Synthesizer`` protocols directly. No
  adapter layer — the agent IS the branch.
- ``NotesTodoApprovalAgent`` (``notes_app.agents.todo_approval``) —
  the helper agent that drives the verdict UI. Used by both this
  flow (via ``ask_human``) and ``NotesAgent.propose_todo`` (via the
  legacy ``DurableHITLWorkflow`` path).

No ``from __future__ import annotations``: pydantic-ai introspects
``get_type_hints()`` at decoration time (same as
``notes_app.agents.notes`` / ``notes_app.agents.todo_approval``).
"""

from dataclasses import dataclass
from uuid import UUID

from ballast import (
    ApprovedResponse,
    DivergentBranch,
    DivergentConvergent,
    Durable,
    ModifiedResponse,
    RejectedResponse,
    SemanticDedup,
    SemanticDedupConfig,
    TimeoutResponse,
    ask_human,
    get_ballast,
)
from ballast.capabilities.helpers.embedder import Embedder

from notes_app.agents.brainstorm import (
    BrainstormDivergentAgent,
    BrainstormSynthesizerAgent,
)
from notes_app.agents.todo_approval import NotesTodoApprovalAgent
from notes_app.models.brainstorm import BrainstormOutcome, BrainstormTask
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
    """User-facing spec for one branch in the divergent fan-out."""
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


# ── Brainstorm workflow ──────────────────────────────────────────────────


def workflow_id(task: BrainstormTask) -> str:
    """Deterministic workflow id for the HTTP route.

    Same ``(parent_thread, topic)`` → same workflow id so duplicate
    clicks collapse to one in-flight workflow."""
    return f"brainstorm:{task.parent_thread_id}:{abs(hash(task.topic))}"


@Durable.workflow()
async def brainstorm(task: BrainstormTask) -> BrainstormOutcome:
    """Diverge → converge → ask user → save (or not).

    One linear durable workflow. Each phase posts a plain assistant
    message into the parent thread so the user sees a step-by-step
    timeline of what the agent is doing. The ``message_added`` signal
    handler (wired by :class:`EventsProvider`) takes care of the
    event-log row + SSE broadcast for each ``say()`` call — no manual
    plumbing here.

    Verdict handling runs INLINE after the ``await ask_human(...)``:

      - ``ApprovedResponse`` → save the proposed idea as a note.
      - ``ModifiedResponse`` → save with the user's edits applied.
      - ``RejectedResponse`` → notify cancellation.
      - ``TimeoutResponse`` → notify timeout.

    Note save + parent-thread notify sit behind DBOS step boundaries
    (the broadcaster default handler is itself a ``@Durable.step``,
    and ``_save_note`` below is too), so a crash mid-flow recovers
    cleanly on restart.
    """
    parent_thread_id = task.parent_thread_id
    topic = task.topic

    await say(parent_thread_id, f'Brainstorming on "{topic}"…')

    chosen: TodoIdea = await _divergent.run(topic)

    await say(
        parent_thread_id,
        f'Chose: "{chosen.title}". Opening approval thread…',
    )

    approval_context = TodoApprovalContext(
        proposed_title=chosen.title,
        proposed_body=chosen.body,
        parent_thread_id=parent_thread_id,
    )
    verdict = await ask_human(
        helper_agent=NotesTodoApprovalAgent,
        context=approval_context,
        opening_message=approval_context.to_opening_message(),
        notify_parent_thread_id=parent_thread_id,
    )

    saved_title: str | None = None
    saved_body: str | None = None

    if isinstance(verdict, ApprovedResponse):
        note = await _save_note(title=chosen.title, body=chosen.body)
        saved_title, saved_body = note.title, note.body
        await say(parent_thread_id, f'Saved your todo titled "{note.title}".')
    elif isinstance(verdict, ModifiedResponse):
        mod = verdict.modified_proposal
        title = str(mod.get("title", chosen.title))
        body = str(mod.get("body", chosen.body))
        note = await _save_note(title=title, body=body)
        saved_title, saved_body = note.title, note.body
        await say(
            parent_thread_id,
            f'Saved your todo titled "{note.title}" (with your edits).',
        )
    elif isinstance(verdict, RejectedResponse):
        reason = (verdict.feedback or "").strip()
        tail = f" ({reason})" if reason else ""
        await say(parent_thread_id, f"Todo creation was cancelled{tail}.")
    else:  # TimeoutResponse
        await say(
            parent_thread_id,
            "Todo approval timed out — nothing was saved.",
        )

    return BrainstormOutcome(
        proposed_title=chosen.title,
        proposed_body=chosen.body,
        saved_title=saved_title,
        saved_body=saved_body,
    )


# ── Inline helpers ───────────────────────────────────────────────────────


async def say(thread_id: UUID, text: str) -> None:
    """Append a plain assistant message to ``thread_id``.

    No step wrapper here — ``ThreadRepository.add_message`` itself
    self-emits the ``message_added`` signal whose default subscriber
    is a ``@Durable.step``. That's where DBOS replay memoisation
    actually matters (the broadcast side effects). Adding another
    step layer here would just nest steps without gaining anything.
    """
    await get_ballast().thread_repo.add_message(
        thread_id,
        role="assistant",
        parts=[{"type": "text", "text": text, "state": "done"}],
    )


@Durable.step()
async def _save_note(*, title: str, body: str):  # noqa: ANN201 — domain type
    """Persist a note via the module-level singleton repo.

    Wrapped as a ``@Durable.step`` so workflow replay sees the same
    note id (DBOS memoises step return values by step name + args)
    instead of double-creating on crash recovery.
    """
    from notes_app.repositories.note import notes_repo  # noqa: PLC0415
    return await notes_repo.create(title=title, body=body)


# ── Synthesis helpers ────────────────────────────────────────────────────


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


def _build_brainstorm_divergent(
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
    config_name: str = "notes-brainstorm-divergent",
) -> DivergentConvergent[str, TodoIdeas, TodoIdea, TodoIdea]:
    """Assemble the demo's ``DivergentConvergent`` instance."""
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

    return DivergentConvergent[str, TodoIdeas, TodoIdea, TodoIdea](
        branches=branches,
        synthesizer=synthesizer,
        hypotheses=lambda env: env.ideas,
        format_synth_prompt=_format_synth_prompt,
        deduper=deduper,
        best_of_n=best_of_n,
        min_hypotheses=min_hypotheses,
        top_k=top_k,
        divergent_concurrency=divergent_concurrency,
        config_name=config_name,
    )


# ── Module-level singleton ──────────────────────────────────────────────

_divergent: DivergentConvergent[str, TodoIdeas, TodoIdea, TodoIdea] = (
    _build_brainstorm_divergent()
)


__all__ = [
    "BrainstormAgentSpec",
    "DEFAULT_DIVERGENT_SPECS",
    "brainstorm",
    "say",
    "workflow_id",
]

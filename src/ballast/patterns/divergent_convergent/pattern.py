from __future__ import annotations

import itertools
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel

from dbos import DBOSConfiguredInstance, Queue

from ballast.durable import Durable
from ballast.observability.spans import traced
from ballast.observability.trace_names import TraceName
from ballast.patterns.divergent_convergent.events import (
    BranchCompleted,
    BranchEnqueued,
    BranchFailed,
    ConvergeCompleted,
    ConvergeStarted,
    DedupCompleted,
    DivergentEvent,
    VerifyCompleted,
    divergent_convergent_progress,
)
from ballast.patterns.divergent_convergent.primitives import (
    DivergentBranch,
    Synthesizer,
    Verifier,
)
from ballast.patterns.errors import InsufficientDivergence

_log = __import__("logging").getLogger(__name__)


InT = TypeVar("InT")
EnvT = TypeVar("EnvT")
HypothesisT = TypeVar("HypothesisT")
OutT = TypeVar("OutT")

# Structural type of any "deduper" the pattern accepts. We don't
# import ``SemanticDedup`` here — the concrete deduper is just anything
# with an async ``run(list) -> list`` method (i.e. satisfies the
# ``Pattern[list[H], list[H]]`` structural contract).
Deduper = Any  # noqa: PYI013 — protocol-as-Any to avoid pinning a class.


@dataclass(frozen=True)
class _ScoredHypothesis(Generic[HypothesisT]):
    """Internal helper — pairs a hypothesis with its verifier score."""
    hypothesis: HypothesisT
    score: float


_instance_counter = itertools.count()


def _to_jsonable(value: Any) -> Any:
    """Best-effort: pydantic models → dict; everything else verbatim."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def _default_format_synth_prompt(task: Any, candidates: list[Any]) -> str:
    """Default synth-prompt renderer: JSON-encoded ``{task, candidates}``.

    Apps that want a domain-specific prompt (e.g. Russian-language
    rendering, ``# Тема: ... Кандидаты: ...``) pass an explicit
    ``format_synth_prompt`` callable to ``DivergentConvergent``."""
    return json.dumps(
        {"task": _to_jsonable(task), "candidates": _to_jsonable(candidates)},
        ensure_ascii=False, indent=2,
    )


@Durable.dbos_class()
class DivergentConvergent(
    DBOSConfiguredInstance, Generic[InT, EnvT, HypothesisT, OutT],
):
    """Fan-out → optional dedup → optional verifier → synthesize.

    Three-phase durable workflow:

    1. **Divergent phase.** For each ``DivergentBranch`` × each of
       ``best_of_n`` samples, enqueue a step that calls
       ``branch.agent.diverge(task)``. Tasks run through a
       per-instance ``DBOS.Queue`` with ``divergent_concurrency`` so
       crashes recover and provider rate-limits aren't violated.

    2. **Reduction.** Pools are flattened. If a ``deduper`` is
       supplied (any ``Pattern[list[H], list[H]]`` —
       ``SemanticDedup`` is the canonical one), it filters near-
       duplicates. The pool size is then compared to
       ``min_hypotheses``; failing the floor raises
       ``InsufficientDivergence`` so callers can decide whether to
       retry with looser settings or escalate.

    3. **Convergence.** If a ``verifier`` is supplied, every survivor
       is scored, sorted descending, and (optionally) sliced to
       ``top_k``. The synthesizer then picks / edits / blends one
       output from whatever pool reaches it.

    Satisfies ``Pattern[InT, OutT]`` structurally.

    -------------------------------------------------------------------
    ANTI-PATTERNS — record-keeping section so future readers
    don't repeat mistakes documented in the literature. Each item
    is backed by a paper / production retrospective.
    -------------------------------------------------------------------

    * **Do NOT mix strong + weak models in ``branches``.** Princeton's
      2025 MoA follow-up showed that introducing a weaker model into
      an ensemble REDUCES quality, because the synthesizer / verifier
      gets pulled toward plausible-but-wrong candidates. Prefer
      3× one strong model with different ``system_prompt`` and
      ``temperature`` over a heterogeneous mix that includes a
      noticeably weaker proposer.

    * **Do NOT scale ``len(branches) * best_of_n`` beyond ~10.**
      CreativeDC (arxiv 2510.26490) documents a quantity-distinctiveness
      tradeoff: as you generate more candidates, the RELATIVE
      originality of any single one drops because the synthesizer's
      attention budget is finite. Sweet spot for creative tasks is
      3-5 branches × 1-2 samples.

    * **Do NOT skip the verifier on critical paths.** BoN-MAV
      ("Best-of-N Multi-Agent Verification", 2025) shows synthesizers
      asked to both score AND pick exhibit first-plausible-candidate
      bias. A dedicated verifier with an explicit rubric corrects this.

    * **Do NOT lower ``min_hypotheses`` to 1 silently.** A pool of one
      means the rest of the pipeline (dedup + verify + synthesize)
      degenerates to "rubber-stamp the only candidate" — at which
      point you don't need this pattern, you need a single agent
      call. Keep ``min_hypotheses >= 2`` for everything except
      benchmarks.

    * **Do NOT route ``best_of_n`` retries to a different model.**
      Self-consistency literature relies on K samples from the SAME
      model under temperature > 0 — mixing models breaks the
      statistical assumption that the majority answer is the most
      likely one.
    -------------------------------------------------------------------
    """

    name: ClassVar[str] = "divergent_convergent"

    def __init__(
        self,
        branches: tuple[DivergentBranch[InT, EnvT], ...],
        synthesizer: Synthesizer[OutT],
        *,
        hypotheses: Callable[[EnvT], list[HypothesisT]],
        format_synth_prompt: Callable[[InT, list[HypothesisT]], str] | None = None,
        deduper: Deduper | None = None,
        verifier: Verifier[HypothesisT] | None = None,
        top_k: int | None = None,
        best_of_n: int = 1,
        min_hypotheses: int = 2,
        per_branch_failure: Literal["strict", "skip"] = "skip",
        divergent_concurrency: int = 4,
        config_name: str | None = None,
        on_progress: Callable[[DivergentEvent], Awaitable[None]] | None = None,
    ) -> None:
        if not branches:
            raise ValueError("DivergentConvergent requires at least one branch")
        if best_of_n < 1:
            raise ValueError("best_of_n must be >= 1")
        if min_hypotheses < 1:
            raise ValueError("min_hypotheses must be >= 1")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1 when set")
        if verifier is None and top_k is not None:
            raise ValueError(
                "top_k requires a verifier — without scores there's no "
                "meaningful ordering to slice by"
            )

        resolved_name = config_name or (
            f"divergent-convergent-{next(_instance_counter)}"
        )
        super().__init__(config_name=resolved_name)

        self._branches: dict[str, DivergentBranch[InT, HypothesisT]] = {}
        for branch in branches:
            if branch.label in self._branches:
                raise ValueError(
                    f"Duplicate branch label {branch.label!r} — labels "
                    "must be unique within one DivergentConvergent instance",
                )
            self._branches[branch.label] = branch

        self._synthesizer = synthesizer
        self._hypotheses = hypotheses
        self._format_synth_prompt = format_synth_prompt or _default_format_synth_prompt
        self._deduper = deduper
        self._verifier = verifier
        self._top_k = top_k
        self._best_of_n = best_of_n
        self._min_hypotheses = min_hypotheses
        self._per_branch_failure = per_branch_failure
        self._on_progress = on_progress

        # Module-level constraint: ``Queue`` must be visible to DBOS
        # before ``DBOS.launch()``. DBOS only emits a quiet warning if
        # a configured instance is registered post-launch and Queue
        # has no built-in check at all — make the failure loud so the
        # invariant is enforced at construction, not at the first
        # silent workflow stall. Tests construct patterns per-fixture
        # AFTER ``DBOS.launch()`` so the check is skipped under pytest
        # (matches the auto-migrate hook's pytest carve-out).
        import sys  # noqa: PLC0415

        if Durable.is_launched() and "pytest" not in sys.modules:
            raise RuntimeError(
                f"DivergentConvergent({resolved_name!r}) constructed "
                "after DBOS.launch(). The pattern's worker queue must "
                "be registered before launch — construct the instance "
                "at module / build_app() scope, alongside the other "
                "queues (AGENT_RUN_QUEUE etc.).",
            )
        self._divergent_queue = Queue(
            name=f"{resolved_name}-divergent",
            concurrency=divergent_concurrency,
        )

    # ── public entrypoint ────────────────────────────────────────────────

    @Durable.workflow()
    @traced(TraceName.PATTERN_DIVERGENT_CONVERGENT, attrs=lambda self, task, **__: {
        "pattern": self.name,
        "branch_count": len(self._branches),
        "best_of_n": self._best_of_n,
    })
    async def run(self, task: InT) -> OutT:
        """Run the divergent → optional dedup → optional verify →
        synthesize pipeline.

        Emits typed progress events on
        :data:`divergent_convergent_progress` at every observable
        boundary. The framework's default routing handler (in
        ``events.py``, connected at module import) catches them and
        — if :data:`progress_thread_var` is set in the active
        context — posts a chat narration via
        :data:`chat_message_requested`. App opts in with::

            with progress_to_thread(thread_id=parent):
                chosen = await _divergent.run(topic)

        Custom subscribers (Slack, metrics, audit) connect their own
        handlers to the typed signal — independent of the chat
        routing.
        """
        async def _publish(event: DivergentEvent) -> None:
            await divergent_convergent_progress.send(sender=self, event=event)
            if self._on_progress is not None:
                try:
                    await self._on_progress(event)
                except Exception:  # noqa: BLE001
                    _log.exception("on_progress callback failed (swallowed)")

        # 1. Divergent fan-out via ``Durable.enqueue`` — auto-injects
        #    the OTel trace carrier so every span emitted inside the
        #    enqueued worker (``_diverge_one`` + pydantic-ai chat
        #    spans) nests under THIS workflow in Logfire instead of
        #    becoming a fresh root trace.
        handles: list[tuple[str, int, Any]] = []
        for label in self._branches:
            for sample_idx in range(self._best_of_n):
                handle = await Durable.enqueue(
                    self._divergent_queue,
                    self._diverge_one, label, sample_idx, task,
                )
                handles.append((label, sample_idx, handle))
                await _publish(BranchEnqueued(
                    label=label, sample_idx=sample_idx,
                ))

        # 2. Collect with per-branch failure policy.
        pools: list[list[HypothesisT]] = []
        outcomes: dict[str, str] = {}
        for label, sample_idx, handle in handles:
            key = f"{label}#{sample_idx}"
            try:
                pool = await handle.get_result()
                pools.append(pool)
                outcomes[key] = f"ok:{len(pool)}"
                await _publish(BranchCompleted(
                    label=label, sample_idx=sample_idx,
                    pool_size=len(pool),
                ))
            except Exception as exc:  # noqa: BLE001 — caller policy decides
                await _publish(BranchFailed(
                    label=label, sample_idx=sample_idx,
                    error_type=type(exc).__name__,
                ))
                if self._per_branch_failure == "strict":
                    raise
                outcomes[key] = f"failed:{type(exc).__name__}"

        merged: list[HypothesisT] = [h for pool in pools for h in pool]

        # 3. Optional dedup.
        if self._deduper is not None and merged:
            input_count = len(merged)
            merged = await self._deduper.run(merged)
            await _publish(DedupCompleted(
                input_count=input_count, output_count=len(merged),
            ))

        # 4. Minimum-cardinality guard.
        if len(merged) < self._min_hypotheses:
            raise InsufficientDivergence(
                produced=len(merged),
                required=self._min_hypotheses,
                branch_outcomes=outcomes,
            )

        # 5. Optional verifier + top-K filtering.
        if self._verifier is not None:
            scored_count = len(merged)
            scored: list[_ScoredHypothesis[HypothesisT]] = []
            for hypothesis in merged:
                score = await self._score_one(task, hypothesis)
                scored.append(_ScoredHypothesis(hypothesis, score))
            scored.sort(key=lambda s: s.score, reverse=True)
            if self._top_k is not None:
                scored = scored[: self._top_k]
            merged = [s.hypothesis for s in scored]
            await _publish(VerifyCompleted(
                scored_count=scored_count, top_k_applied=self._top_k,
            ))

        # 6. Synthesize.
        await _publish(ConvergeStarted(candidate_count=len(merged)))
        result = await self._converge(task, merged)
        await _publish(ConvergeCompleted())
        return result

    # ── steps ────────────────────────────────────────────────────────────

    @Durable.step()
    async def _diverge_one(
        self, label: str, sample_idx: int, task: InT,
    ) -> list[HypothesisT]:
        # ``sample_idx`` is unused in the body — it's there so each
        # best-of-N sample becomes a DISTINCT step invocation (DBOS
        # caches step results by name + args; without it, K samples
        # of the same branch would share the cached first result on
        # workflow replay).
        #
        # OTel context propagation is handled by ``@Durable.step`` —
        # the carrier travels in a magic kwarg from ``Durable.enqueue``
        # and is attached to this fiber before the body runs.
        del sample_idx
        branch = self._branches[label]
        result = await branch.agent.run(task)
        return self._hypotheses(result.output)

    @Durable.step()
    async def _score_one(self, task: InT, hypothesis: HypothesisT) -> float:
        assert self._verifier is not None  # checked at run() level
        return await self._verifier.score(task=task, hypothesis=hypothesis)

    @Durable.step()
    async def _converge(
        self, task: InT, candidates: list[HypothesisT],
    ) -> OutT:
        prompt = self._format_synth_prompt(task, candidates)
        result = await self._synthesizer.run(prompt)
        return result.output

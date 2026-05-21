from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

InT_contra = TypeVar("InT_contra", contravariant=True)
HypothesisT_co = TypeVar("HypothesisT_co", covariant=True)
HypothesisT_contra = TypeVar("HypothesisT_contra", contravariant=True)
OutT_co = TypeVar("OutT_co", covariant=True)


@runtime_checkable
class DivergentAgent(Protocol[InT_contra, HypothesisT_co]):
    """One ``branch`` in the divergent phase.

    Returns a *pool* (``list[Hypothesis]``) per call — typically 2-5
    items. Apps usually wrap a ``pydantic_ai.Agent`` (or one of the
    ``StateflowAgent`` flavours) in a tiny adapter that calls
    ``agent.run(task)`` and unpacks the structured output.

    The framework does NOT import pydantic-ai here on purpose: the
    pattern works with any object that satisfies this protocol, which
    lets apps mix model-backed agents, mocks (for tests), or pure-
    Python heuristics in the same fan-out.
    """

    async def diverge(self, task: InT_contra) -> list[HypothesisT_co]: ...


@runtime_checkable
class Synthesizer(Protocol[InT_contra, HypothesisT_contra, OutT_co]):
    """Convergent reducer over the surviving pool.

    Receives the original ``task`` AND the post-dedup / post-verify
    candidate set, returns the single chosen output. Common shapes:

    * pick-one (returns ``Candidate``),
    * pick-and-edit (returns a modified copy),
    * synthesise-new (returns a blend that wasn't in the pool).

    The pattern is agnostic — that's all decided by the app's
    synthesizer implementation.
    """

    async def synthesize(
        self,
        *,
        task: InT_contra,
        candidates: list[HypothesisT_contra],
    ) -> OutT_co: ...


@runtime_checkable
class Verifier(Protocol[HypothesisT_contra]):
    """Optional scorer applied between dedup and synthesis.

    Returns a float per hypothesis (higher = better). The pattern then
    sorts descending and (if ``top_k`` is set) slices to the top-K
    before handing them to the synthesizer. Common backings:

    * a small reward model / classifier,
    * an LLM judge with a structured rubric,
    * a heuristic over hypothesis fields.

    Per BoN-MAV ("Best-of-N Multi-Agent Verification", 2025), keeping
    the verifier SEPARATE from the synthesizer beats letting one model
    do both — the synthesizer biases toward the first plausible
    candidate it sees, while a dedicated verifier scores them on
    explicit criteria.
    """

    async def score(
        self,
        *,
        task: Any,
        hypothesis: HypothesisT_contra,
    ) -> float: ...


HypothesisT = TypeVar("HypothesisT")
InT = TypeVar("InT")


@dataclass(frozen=True)
class DivergentBranch(Generic[InT, HypothesisT]):
    """One labelled branch in the divergent fan-out.

    ``label`` lands in traces / queue task names so you can tell whose
    pool dominated after convergence. ``agent`` is anything satisfying
    ``DivergentAgent``.
    """
    label: str
    agent: DivergentAgent[InT, HypothesisT]

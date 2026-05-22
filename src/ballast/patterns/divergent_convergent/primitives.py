from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

InT_contra = TypeVar("InT_contra", contravariant=True)
EnvT_co = TypeVar("EnvT_co", covariant=True)
OutT_co = TypeVar("OutT_co", covariant=True)
HypothesisT_contra = TypeVar("HypothesisT_contra", contravariant=True)


@runtime_checkable
class _AgentRunResult(Protocol[OutT_co]):
    """Anything with a typed ``.output`` property.

    Structural mirror of pydantic-ai's ``AgentRunResult`` — declared
    here so the framework doesn't import pydantic-ai at module level.
    """

    @property
    def output(self) -> OutT_co: ...


@runtime_checkable
class DivergentAgent(Protocol[InT_contra, EnvT_co]):
    """One ``branch`` in the divergent phase.

    Returns an *envelope* (``EnvT``) per ``.run(task)`` call. The
    pattern then applies the app-supplied ``hypotheses`` projector to
    extract the ``list[Hypothesis]`` it reduces over.

    The signature deliberately matches pydantic-ai's ``Agent.run`` so
    pydantic-ai agents satisfy it natively — but the framework doesn't
    import pydantic-ai, so apps can substitute mocks or pure-Python
    heuristics.
    """

    async def run(self, task: InT_contra) -> _AgentRunResult[EnvT_co]: ...


@runtime_checkable
class Synthesizer(Protocol[OutT_co]):
    """Convergent reducer over the surviving pool.

    Receives a string prompt rendered by the pattern (which used the
    app-supplied ``format_synth_prompt`` to project ``(task, candidates)``
    into text). ``.output`` IS the final ``OutT`` — no projector
    needed for synthesis since the synthesizer's output type is the
    same as the pattern's result type.
    """

    async def run(self, prompt: str) -> _AgentRunResult[OutT_co]: ...


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
EnvT = TypeVar("EnvT")


@dataclass(frozen=True)
class DivergentBranch(Generic[InT, EnvT]):
    """One labelled branch in the divergent fan-out.

    ``label`` lands in traces / queue task names so you can tell whose
    pool dominated after convergence. ``agent`` is anything satisfying
    ``DivergentAgent[InT, EnvT]`` — typically a ``BallastAgent`` /
    ``pydantic_ai.Agent`` whose output type is ``EnvT``.
    """

    label: str
    agent: DivergentAgent[InT, EnvT]

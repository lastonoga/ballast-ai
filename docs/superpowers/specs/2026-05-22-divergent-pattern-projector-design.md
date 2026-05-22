# DivergentConvergent: own the envelope→hypotheses mapping in the pattern

**Status:** Approved (design)
**Date:** 2026-05-22
**Scope:** `src/ballast/patterns/divergent_convergent/` + notes-app caller

## Problem

`DivergentConvergent` currently requires every branch agent to implement
a `diverge(task) -> list[H]` method, and every synthesizer to implement
`synthesize(task, candidates) -> Out`. These methods are nothing but a
thin wrapper over `agent.run(...)`: they call the underlying pydantic-ai
agent and unpack `result.output` (e.g. `.ideas` for a `TodoIdeas`
envelope).

This creates redundancy and a leaky abstraction:

- Every app-side agent class gains a duplicate `diverge` / `synthesize`
  method that mirrors `agent.run` plus one extra `.ideas` access.
- A `StateflowAgent` subclass becomes coupled to *one* pattern's
  contract instead of being a reusable, neutral object.
- The unwrap (`output.ideas`) is app-level logic about envelope shape;
  the agent class is the wrong owner.

We want envelopes (not bare `list`) for LLM outputs — schema name,
extensibility, tool-output mode compatibility. So the unwrap can't go
away; it has to live somewhere. **The pattern is the right owner**:
the unwrap is part of the "fan-out → reduce" orchestration the pattern
is responsible for.

## Goal

Remove `diverge` / `synthesize` methods from agents. The pattern accepts
agents directly (anything with a structural `.run(...)` matching
pydantic-ai's shape) plus two app-provided callables — one to project
divergent agent output to a hypothesis list, one to render the
synthesis prompt from candidates.

## Non-goals

- HITL-blocking variant of `BrainstormFlow.run` (separate spec).
- Generalizing this design to other patterns (`MapReduce`, `Reflection`,
  `Mutation`). They don't have the agent+envelope problem because they
  operate on data or define internal steps.
- Per-branch envelope variation. All branches in one fan-out are
  assumed to return the same envelope shape (one pattern-wide
  `hypotheses` projector). If a future use case needs per-branch
  projectors, the design extends naturally without breaking callers.

## Design

### Framework: structural agent Protocol

`primitives.py` replaces the method-bearing `DivergentAgent` /
`Synthesizer` protocols with structural shapes that match pydantic-ai's
`Agent.run`:

```python
class _AgentRunResult(Protocol[OutT_co]):
    """Anything with an ``.output`` attribute of the right type.
    Matches pydantic-ai's ``AgentRunResult`` without importing it."""
    @property
    def output(self) -> OutT_co: ...


class DivergentAgent(Protocol[InT_contra, EnvT_co]):
    """A divergent branch's underlying agent.

    Returns an envelope (``EnvT``) per call — the pattern then applies
    the app-provided ``hypotheses`` projector to extract the
    ``list[H]`` it needs for the reduce phase.
    """
    async def run(self, task: InT_contra) -> _AgentRunResult[EnvT_co]: ...


class Synthesizer(Protocol[OutT_co]):
    """The convergent reducer's underlying agent.

    Receives a fully-rendered prompt (string) and returns an envelope
    whose ``.output`` is the final ``OutT`` directly — no projector
    needed for synthesis.
    """
    async def run(self, prompt: str) -> _AgentRunResult[OutT_co]: ...
```

`DivergentBranch` stays `(label, agent)` — no projector per branch.

### Framework: `DivergentConvergent` accepts two callables

```python
class DivergentConvergent(Generic[InT, EnvT, HypT, OutT]):
    def __init__(
        self,
        branches: tuple[DivergentBranch[InT, EnvT], ...],
        synthesizer: Synthesizer[OutT],
        *,
        hypotheses: Callable[[EnvT], list[HypT]],
        format_synth_prompt: Callable[[InT, list[HypT]], str],
        deduper: Deduper | None = None,
        verifier: Verifier[HypT] | None = None,
        top_k: int | None = None,
        best_of_n: int = 1,
        min_hypotheses: int = 2,
        per_branch_failure: Literal["strict", "skip"] = "skip",
        divergent_concurrency: int = 4,
        config_name: str | None = None,
    ) -> None: ...
```

Generic parameters: **4** (was 3) — `EnvT` is new (envelope returned by
divergent agents). The extra type parameter is the cost of letting the
pattern carry the envelope type through to the projector signature; it
makes the call site fully type-checked.

Inside the pattern:

```python
@Durable.step()
async def _diverge_one(self, label, sample_idx, task):
    del sample_idx
    branch = self._branches[label]
    result = await branch.agent.run(task)
    return self._hypotheses(result.output)

@Durable.step()
async def _converge(self, task, candidates):
    prompt = self._format_synth_prompt(task, candidates)
    result = await self._synthesizer.run(prompt)
    return result.output
```

### App: agents lose their adapter methods

`BrainstormDivergentAgent` and `BrainstormSynthesizerAgent` go back to
being **pure `StateflowAgent`** subclasses — no `diverge`, no
`synthesize`, no `_format_synth_prompt` helper inside the synthesizer.

`build_brainstorm_flow` factory wires the agents and the two callables:

```python
branches = tuple(
    DivergentBranch(
        label=spec.label,
        agent=BrainstormDivergentAgent(...),
    )
    for spec in divergent_specs
)

synthesizer = BrainstormSynthesizerAgent(...)

divergent = DivergentConvergent[str, TodoIdeas, TodoIdea, TodoIdea](
    branches=branches,
    synthesizer=synthesizer,
    hypotheses=lambda env: env.ideas,
    format_synth_prompt=_format_synth_prompt,  # module-level fn
    deduper=deduper,
    best_of_n=best_of_n,
    min_hypotheses=min_hypotheses,
    top_k=top_k,
    divergent_concurrency=divergent_concurrency,
    config_name=f"{config_name}-divergent",
)
```

`_format_synth_prompt(task: str, candidates: list[TodoIdea]) -> str`
moves from `brainstorm_agents.py` to module-level in
`brainstorm_flow.py`, next to the factory — it belongs to the app's
wiring, not the agent.

### Determinism / replay

The two new callables run inside `@Durable.step` bodies (`_diverge_one`,
`_converge`). DBOS caches step results, so on workflow replay these
callables don't re-fire — the memoized projection / prompt come back
intact. App-side callables MUST be deterministic and free of side
effects (same input → same output), as required for any step body.
This is documented in the constructor docstring.

The `model_settings()` injection that used to live in the adapter is
gone — pydantic-ai applies the agent's own `model_settings` when
constructed, so `agent.run(task)` already carries the right settings.
For `StateflowAgent`s the framework's `agent` property builds the
underlying `pydantic_ai.Agent` from `build_agent()` + `model_settings()`
unchanged.

## Migration

One commit, BC-break (no transitional layer):

1. Update `primitives.py` Protocols (`DivergentAgent`, `Synthesizer`).
2. Update `pattern.py`: add `hypotheses` + `format_synth_prompt`
   parameters, rewrite `_diverge_one` and `_converge`.
3. Update `tests/patterns/test_divergent_convergent.py` mocks — replace
   `class FakeAgent: async def diverge(...)` with
   `class FakeAgent: async def run(...) -> Result(output=Env(...))`,
   and supply `hypotheses=` + `format_synth_prompt=` in test setup.
4. Update `examples/notes-app/backend/src/notes_app/brainstorm_agents.py`:
   delete `diverge` / `synthesize` / `_format_synth_prompt`.
5. Update `examples/notes-app/backend/src/notes_app/brainstorm_flow.py`:
   move `_format_synth_prompt` next to the factory, pass `hypotheses=`
   and `format_synth_prompt=` to `DivergentConvergent`.
6. Run framework tests + notes-app tests; smoke live brainstorm.

Single caller (notes-app) is in-repo, so BC-break has no external blast
radius.

## Testing

- **Framework unit tests:** existing `test_divergent_convergent.py`
  covers fan-out, dedup, verifier, top-k, failure policy. Migrate
  mocks; add one test asserting that the `hypotheses` projector is
  invoked exactly once per successful branch result.
- **Notes-app smoke:** existing `test_smoke.py` builds a
  `BrainstormFlow` and exercises the FastAPI startup path. Stays green
  because the workflow itself isn't invoked in smoke.
- **Live verification:** run the dev server, click Brainstorm in the
  UI, confirm progress events render and an approval thread spawns.
  This is the only end-to-end behavioural check.

## Open questions

None. Per-branch projector variation is a documented non-goal —
extending later (e.g. `DivergentBranch(label, agent, hypotheses=...)`
overriding the pattern-wide projector) doesn't break the current API.

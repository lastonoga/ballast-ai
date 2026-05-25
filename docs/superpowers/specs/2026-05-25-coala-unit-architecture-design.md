# CoALA Unit Architecture — Single Protocol, Multiple Adapters

**Date:** 2026-05-25
**Status:** Approved (brainstorming) → ready for implementation plan
**Author:** Kir + Claude
**Scope:** Replace shipped Phase 1 (Episodic) + Phase 2 (Semantic) facades
with a unified `CoALAUnit` Protocol + runtime adapters (workflow, tool,
capability). Phase 1+2 code is **deleted**, not deprecated.

## Problem

Phase 1 (`EpisodicMemory` + `EpisodicSource` + 5 strategies + 2 sources +
`RememberTurn` capability) and Phase 2 (`SemanticMemory` + `@memory_tool` +
sources) shipped a **storage-and-strategy-heavy** memory model. Apps had
to either subclass our facades or implement Protocols matching our
prescribed shape.

The architecture failed to deliver SOLID benefits at the contract level:

1. **Three different contracts** for "memory-aware computation" — workflows
   used `EpisodicMemory.episodic_for(...)` (direct), agents used a capability
   (`RememberTurn`), tools used ad-hoc imperative calls. No unifying shape.
2. **Storage prescription** — framework owned `Episode` schema, pgvector
   migration (`0003_episodes`), `EpisodeRow`. Apps that wanted different
   episode shape had to fight the abstraction.
3. **Strategy hierarchy** — 5 `RecallStrategy` impls + `ScoreMerger` impls
   for what is fundamentally one method call: "give me relevant memory."
4. **Apps don't write their memory through us anyway** — notes-app's most
   useful memory access is `notes_repo.search(...)`, which has nothing
   to do with `EpisodicMemory`.

Observed pattern from real usage: apps want to write **whatever Python
they want** inside three lifecycle moments — gather context, do work,
record learnings. The framework's job is to **structure those moments**
and provide **runtime adapters** to execute them in different contexts
(durable workflow / agent tool / agent capability) — not to dictate
storage or query patterns.

## Core insight

CoALA's 4 phases (observe / retrieve / act / learn) ARE the right
abstraction — but they need to be expressed as a **single Protocol**
with **multiple runtime adapters**, not as a separate `EpisodicMemory`
module + `SemanticMemory` module + capability + decorator zoo.

```
                        ┌────────────────────────┐
                        │     CoALAUnit          │
                        │  (single 4-method      │
                        │     Protocol)          │
                        └───────────┬────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
         ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
         │ as_workflow │    │   as_tool    │    │as_capability │
         └──────┬──────┘    └──────┬───────┘    └──────┬───────┘
                ▼                  ▼                   ▼
         @Durable.workflow  pydantic_ai.Tool   BallastCapability
         + per-phase steps  (4 phases run      (observe+retrieve
                            inside one tool    in before_request;
                            call)              learn in after_run)
```

App writes ONE class. Picks an adapter. Framework runs the lifecycle.

## Design

### 1. `CoALAUnit` Protocol

```python
# src/ballast/coala/_protocol.py
from typing import Protocol, TypeVar, runtime_checkable

InT      = TypeVar("InT")
ObsT     = TypeVar("ObsT")
ContextT = TypeVar("ContextT")
OutT     = TypeVar("OutT")


@runtime_checkable
class CoALAUnit(Protocol[InT, ObsT, ContextT, OutT]):
    """Unit of memory-aware computation following CoALA's 4-phase
    decision procedure.

    Same contract regardless of runtime: a workflow, an agent tool, an
    agent capability — any can be wrapped via the corresponding adapter.

    Phase semantics (from Sumers et al., "Cognitive Architectures for
    Language Agents"):
      observe  — parse raw input into structured working-memory state
      retrieve — pull relevant long-term memory based on observation
      act      — reason + ground + execute; produces output
      learn    — persist insights back into long-term memory
    """

    async def observe(self, input: InT) -> ObsT: ...
    async def retrieve(self, observation: ObsT) -> ContextT: ...
    async def act(self, observation: ObsT, context: ContextT) -> OutT: ...
    async def learn(self, observation: ObsT, context: ContextT, output: OutT) -> None: ...
```

### 2. `CoALABase` ABC — ergonomic subclassing with defaults

```python
# src/ballast/coala/_base.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InT      = TypeVar("InT")
ObsT     = TypeVar("ObsT")
ContextT = TypeVar("ContextT")
OutT     = TypeVar("OutT")


class CoALABase(Generic[InT, ObsT, ContextT, OutT], ABC):
    """Minimal-friction base. Apps override only the phases they need.

    ``observe`` defaults to identity (input passes through). Override
    when you need to extract intent / entities / signals before retrieve.

    ``learn`` defaults to no-op. Override to write episodes, facts,
    learned skills, etc. — anything the app wants to persist.

    ``retrieve`` and ``act`` are abstract — every meaningful unit has
    a retrieval step (even if it returns an empty Context) and an act
    step (the actual work)."""

    async def observe(self, input: InT) -> ObsT:
        return input  # type: ignore[return-value]

    @abstractmethod
    async def retrieve(self, observation: ObsT) -> ContextT: ...

    @abstractmethod
    async def act(self, observation: ObsT, context: ContextT) -> OutT: ...

    async def learn(
        self, observation: ObsT, context: ContextT, output: OutT,
    ) -> None:
        return None
```

### 3. Adapter — `as_workflow`

```python
# src/ballast/coala/adapters/workflow.py
from typing import Awaitable, Callable, TypeVar

from ballast.coala._protocol import CoALAUnit
from ballast.durable import Durable

InT  = TypeVar("InT")
OutT = TypeVar("OutT")


def as_workflow(unit: CoALAUnit[InT, ..., ..., OutT]) -> Callable[[InT], Awaitable[OutT]]:
    """Wrap a CoALAUnit as a @Durable.workflow runner.

    Each phase becomes a @Durable.step — memoised on replay, retryable.
    Crash mid-lifecycle: already-completed phases skip; only the
    unfinished tail re-runs.

    Returns a plain async callable. The unit instance is captured via
    closure, NOT serialised by DBOS (callables can't be pickled as
    workflow args; the closure side-steps that).
    """
    @Durable.workflow()
    async def runner(input: InT) -> OutT:
        observation = await _observe_step(unit, input)
        context     = await _retrieve_step(unit, observation)
        output      = await _act_step(unit, observation, context)
        await _learn_step(unit, observation, context, output)
        return output

    runner.__name__ = f"coala_workflow_{type(unit).__name__}"
    runner.__doc__  = (type(unit).__doc__ or "").strip() or None
    return runner


# Each phase wrapped as a step. Steps captured at module level so DBOS
# can register them once; the unit reference passes through self-binding
# of method calls inside the step body.

@Durable.step()
async def _observe_step(unit, input): return await unit.observe(input)

@Durable.step()
async def _retrieve_step(unit, obs):  return await unit.retrieve(obs)

@Durable.step()
async def _act_step(unit, obs, ctx):  return await unit.act(obs, ctx)

@Durable.step()
async def _learn_step(unit, obs, ctx, out): return await unit.learn(obs, ctx, out)
```

**Decision**: per-phase `@Durable.step` wrapping (vs single-step `run`)
is the right granularity. Replay benefit: if `act` succeeded but `learn`
crashed, replay skips `observe`/`retrieve`/`act` and re-runs only `learn`.
Step memoisation handled by DBOS via deterministic args.

### 4. Adapter — `as_tool`

```python
# src/ballast/coala/adapters/tool.py
import inspect
from typing import Any

from pydantic_ai import Tool

from ballast.coala._protocol import CoALAUnit


def as_tool(
    unit: CoALAUnit, *,
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    """Wrap a CoALAUnit as a pydantic-ai Tool.

    From the LLM's POV: one tool call. Internally framework runs all
    four CoALA phases — observe parses LLM-supplied args, retrieve
    fetches memory, act produces output, learn records. Output is
    returned to the agent for next-step reasoning.

    Tool name defaults to ``type(unit).__name__``; description defaults
    to the unit's class docstring. Both overridable via kwargs.
    """
    unit_name = name or type(unit).__name__
    unit_desc = description or (type(unit).__doc__ or "").strip() or None

    # Build a function whose signature matches ``unit.act``'s input
    # shape — pydantic-ai derives the LLM-facing arg schema from it.
    # Strategy: act() takes (observation, context). Tool args become
    # whatever ``observe`` accepts as its input (the InT type).
    observe_sig = inspect.signature(type(unit).observe)
    # Drop ``self`` + reshape: tool function takes the InT input directly.

    async def _runner(**kwargs: Any):
        # Bind kwargs to InT via observe's input signature
        input_param = list(observe_sig.parameters.values())[1]  # after self
        input_value = kwargs.get(input_param.name)
        observation = await unit.observe(input_value)
        context     = await unit.retrieve(observation)
        output      = await unit.act(observation, context)
        await unit.learn(observation, context, output)
        return output

    # Copy signature so pydantic-ai's Tool factory sees the right schema
    _runner.__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter(
                list(observe_sig.parameters.values())[1].name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=list(observe_sig.parameters.values())[1].annotation,
            ),
        ],
    )
    _runner.__name__ = unit_name
    _runner.__doc__  = unit_desc

    return Tool(_runner, name=unit_name, description=unit_desc, takes_ctx=False)
```

**Note**: tool args shape derived from `observe`'s input. If the unit's
`InT` is a `BaseModel`, pydantic-ai exposes nested JSON; if it's a flat
primitive, pydantic-ai exposes a single arg. App controls schema by
choosing `InT`.

### 5. Adapter — `as_capability`

```python
# src/ballast/coala/adapters/capability.py
from typing import Any

from pydantic_ai import RunContext

from ballast.capabilities.base import BallastCapability
from ballast.coala._protocol import CoALAUnit


_OBSERVATION_KEY = "_coala_observation"
_CONTEXT_KEY     = "_coala_context"


def as_capability(
    unit: CoALAUnit, *,
    gate: Any = None,    # Callable[[RunResult], bool | Awaitable[bool]] | None
) -> BallastCapability:
    """Wrap a CoALAUnit as a pydantic-ai capability for an agent.

    Phase → hook mapping:
      observe + retrieve → before_model_request — context computed and
        cached on RunContext for later use; injection strategy is
        app-controlled via override of ``inject_context`` if needed.

      act → the agent's own .iter() loop — NOT framework-mediated.
        The agent reasons + calls tools naturally; CoALA's act phase
        IS the agent run from the framework's POV.

      learn → after_run, gated by optional ``gate`` callback (default:
        always learn). Failures inside ``learn`` are swallowed + logged
        so memory-write bugs never block user-facing replies.
    """
    class _CoALACapability(BallastCapability):
        name = f"coala_{type(unit).__name__}"

        async def before_model_request(
            self, ctx: RunContext[Any], message: Any,
        ) -> Any:
            input_value = _extract_input(ctx, message)
            observation = await unit.observe(input_value)
            context     = await unit.retrieve(observation)
            # Cache on ctx.deps for after_run access
            _stash(ctx, _OBSERVATION_KEY, observation)
            _stash(ctx, _CONTEXT_KEY, context)
            # Inject context into the message — default strategy:
            # prepend as a "memory" system block. Apps override
            # ``inject_context`` on a subclass for custom injection.
            return _default_inject(message, context)

        async def after_run(self, ctx: RunContext[Any], *, result: Any) -> Any:
            try:
                if gate is not None:
                    import asyncio
                    g = gate(result)
                    passed = await g if asyncio.iscoroutine(g) else g
                    if not passed:
                        return result
                observation = _retrieve_stash(ctx, _OBSERVATION_KEY)
                context     = _retrieve_stash(ctx, _CONTEXT_KEY)
                await unit.learn(observation, context, getattr(result, "output", result))
            except Exception:
                import logging
                logging.getLogger("ballast.coala.capability").exception(
                    "CoALA learn() failed (swallowed)",
                )
            return result

    return _CoALACapability()
```

### 6. Composition example — `notes-app`

One CoALA unit demonstrating mixed retrieval (relational + custom logic):

```python
# notes_app/coala/research_summarize.py
from ballast.coala import CoALABase, as_workflow, as_tool, as_capability


@dataclass
class ResearchQuery:
    user_query: str
    target_tag: str | None = None


@dataclass
class ResearchObservation:
    intent: str
    extracted_tag: str | None
    user_id: str | None


@dataclass
class ResearchContext:
    related_notes: list[Note]
    prior_summaries: list[str]


class ResearchSummarize(CoALABase[
    ResearchQuery, ResearchObservation, ResearchContext, Note,
]):
    """Summarize the user's recent research on a topic, save as a note."""

    async def observe(self, q: ResearchQuery) -> ResearchObservation:
        # Custom — parse intent. Apps can use LLM here if needed.
        return ResearchObservation(
            intent=q.user_query,
            extracted_tag=q.target_tag or _extract_tag(q.user_query),
            user_id=current_user_id(),
        )

    async def retrieve(self, obs: ResearchObservation) -> ResearchContext:
        # MIXED: relational + free-text — app controls every byte
        from notes_app.repositories.note import notes_repo
        related = (
            await notes_repo.find_by_tag(obs.extracted_tag)
            if obs.extracted_tag else []
        )
        prior = await _app_episode_store.recent_summaries(
            user_id=obs.user_id, intent=obs.intent, k=3,
        )
        return ResearchContext(related_notes=related, prior_summaries=prior)

    async def act(
        self, obs: ResearchObservation, ctx: ResearchContext,
    ) -> Note:
        body = await _summarize_with_llm(
            corpus=ctx.related_notes, prior=ctx.prior_summaries,
        )
        # Use existing HITL approval flow
        return await create_note_flow(
            ProposedNote(title=f"Research: {obs.intent}", body=body),
        )

    async def learn(
        self, obs: ResearchObservation, ctx: ResearchContext, out: Note,
    ) -> None:
        if out is None:
            return
        await _app_episode_store.append(
            user_id=obs.user_id,
            intent=obs.intent,
            summary=out.body[:300],
            related_note_ids=[n.id for n in ctx.related_notes],
        )
```

App chooses how to use it:

```python
# As a durable workflow (call directly from any code):
research_workflow = as_workflow(ResearchSummarize())
note = await research_workflow(ResearchQuery(user_query="ML in prod"))

# As an agent tool (LLM-callable):
class NotesAgent(DurableAgent):
    def build_agent(self):
        return Agent(
            ...,
            tools=[
                *@NotesAgent.tool decorated,
                as_tool(ResearchSummarize()),    # ← LLM gets "ResearchSummarize" tool
            ],
        )

# As a capability (bolt onto any agent run):
class NotesAgent(DurableAgent):
    def build_agent(self):
        return Agent(
            ...,
            capabilities=[
                *default_capabilities(),
                as_capability(ResearchSummarize()),    # ← context injected per turn
            ],
        )
```

The unit's retrieve/learn methods do whatever the app needs — call
`notes_repo` (RDB), pgvector index, MCP server, Slack API, local file —
framework doesn't care.

## Adapter decision matrix

| Use case | Adapter |
|---|---|
| App calls explicitly from workflow code | `as_workflow` — gets a callable that's durable, recoverable |
| Agent LLM should be able to invoke as named tool | `as_tool` — pydantic-ai Tool with derived schema |
| Inject context into every agent turn / learn after each turn | `as_capability` — fires before/after hooks |
| App needs multiple usage modes for the same unit | Apply multiple adapters to the same instance |

## Phase 1+2 deletion plan

**Delete entirely:**
- `src/ballast/memory/episodic/` — whole subpackage
- `src/ballast/memory/semantic/` — whole subpackage
- `src/ballast/alembic/versions/0003_episodes.py` + the episodes table migration
- `Ballast.with_episodic_memory(...)` / `Ballast.with_semantic_memory(...)` / `Ballast.with_memory(...)` setters
- `self._episodic_memory` / `self._semantic_memory` / `self._memory` attrs from `Ballast.__init__`
- All tests under `tests/memory/`
- Phase 1's `scan_context` recognition of `Episode` / `RecallResult` (revert to Ref-only handling that Phase 1 added — keep the Ref recognition; remove Episode/RecallResult special-casing)
- Notes-app: `notes_app/memory/` subpackage, `RememberTurn` usage in `default_notes_capabilities()`, `_build_episodic_memory()` in `main.py`, episodic recall in `create_note_flow`

**Keep (orthogonal / pre-existing):**
- `Embedder` Protocol (`src/ballast/capabilities/helpers/embedder.py`)
- `current_user_id` / `acting_as` ContextVar (`src/ballast/auth/`)
- `MapReduce` class pattern (`src/ballast/patterns/map_reduce/`)
- `Ref[T]` + `scan_context` recognising `Ref` instances (Phase 1 added the Ref-walking branch — keep, it's orthogonal to memory)
- pgvector dependency in `pyproject.toml` (apps still use it)
- HITL channels (`UICardChannel` / `ThreadChannel` etc. — untouched)
- `Scope` BaseModel — moves from `src/ballast/memory/_scope.py` → `src/ballast/auth/scope.py` (generic scope object, used by repos)

**Migration order**: cleanup first → new CoALA on top → notes-app migration.

## Error handling

- Each phase wrapped in `@Durable.step` (workflow adapter) — DBOS retries on transient errors per its retry config.
- `learn` failures swallowed + logged in `as_capability` (never blocks user reply); raised in `as_workflow` (workflow-level recovery handles).
- `retrieve` returning empty Context is valid — apps explicitly opt into "Context can be empty" by their context type design.
- Bad input to `observe` raises immediately — framework doesn't wrap.

## Testing

- **Per-adapter unit tests**: with a fake `CoALAUnit` that records calls, verify each adapter routes lifecycle correctly (observe → retrieve → act → learn order).
- **Workflow adapter**: verify each phase is a step (memoised); replay-skip on second `run` with same args.
- **Tool adapter**: verify pydantic-ai Tool schema matches unit's `observe` input.
- **Capability adapter**: verify before_model_request injects context; after_run gated by `gate`; failures in learn swallowed.
- **notes-app integration**: one `ResearchSummarize` unit wired in three modes (workflow + tool + capability); end-to-end smoke that all three paths produce the expected save + learn.

## What this design deliberately does NOT do

- **No prescribed storage**. No `Episode` schema, no `EpisodicSource`, no `SemanticSource`. Apps own their tables, vector indexes, API clients.
- **No `gather_context` / `record_learnings` rename** — sticking with CoALA's official terms (`observe` / `retrieve` / `act` / `learn`).
- **No per-stage retry config in CoALABase** — DBOS step retries cover it. App-level retries inside any phase if they want.
- **No auto-discovery / global registry of units** — apps explicitly construct + adapt.
- **No "act IS the agent" magic** for as_capability — framework only fires before/after hooks; the agent's `.iter()` loop is the "act" phase, but framework doesn't try to inspect or wrap it.

## Files touched (Implementation Phase A: cleanup; Phase B: new build)

### Phase A — rip out Phase 1+2

**Delete:**
- `src/ballast/memory/episodic/` (entire)
- `src/ballast/memory/semantic/` (entire)
- `src/ballast/alembic/versions/0003_episodes.py`
- `tests/memory/episodic/` (entire)
- `tests/memory/semantic/` (entire)
- `tests/memory/test_models.py`, `test_scope.py`, `test_facade.py`, etc.
- `examples/notes-app/backend/src/notes_app/memory/` (semantic_sources.py)
- `examples/notes-app/backend/tests/test_notes_semantic.py`

**Modify:**
- `src/ballast/app.py` — drop `with_episodic_memory`, `with_semantic_memory`, `with_memory` setters + corresponding attrs
- `src/ballast/__init__.py` — drop all memory re-exports (`EpisodicMemory`, `SemanticMemory`, etc.); keep `Scope` (moves to auth)
- `src/ballast/memory/__init__.py` → `src/ballast/auth/scope.py` (move `Scope`)
- `src/ballast/grounded/_scan.py` — drop Episode/RecallResult special-cases (keep Ref recognition)
- `examples/notes-app/backend/src/notes_app/agents/notes.py` — drop `RememberTurn` from `default_notes_capabilities`
- `examples/notes-app/backend/src/notes_app/main.py` — drop `_build_episodic_memory()`, `with_episodic_memory`, `with_semantic_memory` calls
- `examples/notes-app/backend/src/notes_app/workflows/create_note.py` — drop episodic recall block
- `tests/persistence/test_semantic_vector.py` — delete (was a Phase 2 leftover)

### Phase B — build CoALAUnit architecture

**Create:**
- `src/ballast/coala/__init__.py` — re-exports
- `src/ballast/coala/_protocol.py` — `CoALAUnit` Protocol + TypeVars
- `src/ballast/coala/_base.py` — `CoALABase` ABC
- `src/ballast/coala/adapters/__init__.py`
- `src/ballast/coala/adapters/workflow.py` — `as_workflow`
- `src/ballast/coala/adapters/tool.py` — `as_tool`
- `src/ballast/coala/adapters/capability.py` — `as_capability`
- `tests/coala/__init__.py` (empty)
- `tests/coala/test_protocol.py`
- `tests/coala/test_base.py`
- `tests/coala/test_workflow_adapter.py`
- `tests/coala/test_tool_adapter.py`
- `tests/coala/test_capability_adapter.py`

**Modify:**
- `src/ballast/__init__.py` — add CoALA re-exports
- `examples/notes-app/backend/src/notes_app/coala/__init__.py` (new)
- `examples/notes-app/backend/src/notes_app/coala/research_summarize.py` (new — demo unit)
- `examples/notes-app/backend/src/notes_app/main.py` — wire the demo unit as both a workflow and an agent tool
- `examples/notes-app/backend/tests/test_research_summarize.py` (new)

## Scope estimate

**Phase A (cleanup):** ~5-7 tasks; mechanical deletion + import-graph updates. ~1-2 days.

**Phase B (new build):** ~10-12 tasks; small files, mostly fresh code. ~2-3 days.

**Total:** ~3-5 days execution time.

## Open follow-ups (out of scope)

- **MCP-backed sources** as CoALAUnit examples (Linear / GitHub / Notion adapters).
- **Phase 4 (learning loop)** — clustering recent learn() outputs → HITL-suggested skill consolidation. Now naturally expressible: cluster output of `learn()` writes; suggest a new CoALAUnit class for the discovered pattern.
- **Sub-agent as CoALAUnit** — `class MyAgent(CoALAUnit)` where `act` invokes a sub-agent's `.run()`. Composability path.
- **Goal drift detection** as a capability that wraps `act` with judge calls — sidecar pattern on top of CoALA.

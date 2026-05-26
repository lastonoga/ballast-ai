# How to test CoALA units

**Problem:** Your CoALA unit has 4 phases (observe / retrieve / act / learn). You want fast unit tests for each phase + a few integration tests for the whole unit. No DBOS workflow, no real LLM, no real database.

**Solution:** Test each phase as a plain async method call. Use in-memory repos for `retrieve`. Use `TestModel` agents inside `act` if your unit calls one. Use the `as_*` adapters only for integration tests.

## Minimum: per-phase tests

```python
import pytest
from ballast.coala import CoALAUnit
from notes_app.coala.research_summarize import (
    ResearchObservation, ResearchQuery, ResearchSummarize,
)
from notes_app.repositories.note import InMemoryNoteRepository


@pytest.fixture
def repo(monkeypatch):
    fresh = InMemoryNoteRepository()
    monkeypatch.setattr("notes_app.repositories.note.notes_repo", fresh)
    return fresh


def test_satisfies_coala_unit_protocol() -> None:
    assert isinstance(ResearchSummarize(), CoALAUnit)


@pytest.mark.asyncio
async def test_observe_extracts_intent(repo) -> None:
    unit = ResearchSummarize()
    obs = await unit.observe(ResearchQuery(user_query="ML in prod"))
    assert isinstance(obs, ResearchObservation)
    assert obs.intent == "ML in prod"


@pytest.mark.asyncio
async def test_retrieve_finds_related_notes(repo) -> None:
    await repo.create(title="ml-deployment", body="machine learning in prod")
    await repo.create(title="fashion", body="trends")

    unit = ResearchSummarize()
    obs = ResearchObservation(intent="machine learning", user_id=None)
    ctx = await unit.retrieve(obs)
    titles = {n.title for n in ctx.related_notes}
    assert "ml-deployment" in titles
    assert "fashion" not in titles


@pytest.mark.asyncio
async def test_retrieve_handles_empty_repo(repo) -> None:
    unit = ResearchSummarize()
    obs = ResearchObservation(intent="anything", user_id=None)
    ctx = await unit.retrieve(obs)
    assert ctx.related_notes == []


@pytest.mark.asyncio
async def test_act_synthesizes_summary() -> None:
    unit = ResearchSummarize()
    obs = ResearchObservation(intent="ML", user_id=None)
    ctx = ResearchContext(related_notes=[Note(title="a", body="b")])
    summary = await unit.act(obs, ctx)
    assert "a" in summary.body


@pytest.mark.asyncio
async def test_learn_logs_without_side_effects(caplog) -> None:
    import logging
    caplog.set_level(logging.INFO, logger="notes_app.coala")
    unit = ResearchSummarize()
    obs = ResearchObservation(intent="ML", user_id=None)
    ctx = ResearchContext()
    summary = ResearchSummary(title="t", body="b")
    await unit.learn(obs, ctx, summary)
    assert "ML" in caplog.text
```

Each phase is an `async def` method — testing is dead simple. No DBOS, no LLM, no adapters.

## Test with TestModel-based agents inside

If your unit's `act` calls an LLM:

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from ballast.coala import CoALABase


class LLMSummarize(CoALABase[Query, Observation, Context, Summary]):
    def __init__(self, model_or_agent=None):
        # Allow test-injection of agent
        self._agent = Agent(
            model=model_or_agent or "openai:gpt-4o-mini",
            output_type=Summary,
        )

    async def retrieve(self, obs): ...
    async def act(self, obs, ctx):
        result = await self._agent.run(prompt_with(ctx))
        return result.output


@pytest.mark.asyncio
async def test_act_with_test_model() -> None:
    test_model = TestModel(custom_output_args={"title": "T", "body": "B"})
    unit = LLMSummarize(model_or_agent=test_model)

    obs = Observation(intent="x")
    ctx = Context(related_items=["a", "b"])
    summary = await unit.act(obs, ctx)
    assert summary.title == "T"
```

**Pattern: pass agent/model via constructor; default to real model for prod, override in tests.**

## Test the full lifecycle (integration-ish)

For a sanity check, run all 4 phases in sequence:

```python
@pytest.mark.asyncio
async def test_full_lifecycle(repo) -> None:
    await repo.create(title="ml-deployment", body="ML in prod")

    unit = ResearchSummarize()
    query = ResearchQuery(user_query="ML")
    obs = await unit.observe(query)
    ctx = await unit.retrieve(obs)
    summary = await unit.act(obs, ctx)
    await unit.learn(obs, ctx, summary)

    assert "ml-deployment" in summary.body
```

No adapters, no DBOS. Sub-100ms test.

## Test through the `as_tool` adapter

```python
from ballast.coala import as_tool
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


@pytest.mark.asyncio
async def test_unit_as_tool(repo) -> None:
    await repo.create(title="ml", body="text")
    unit = ResearchSummarize()
    tool = as_tool(unit)
    assert tool.name == "ResearchSummarize"

    out = await tool.function(q=ResearchQuery(user_query="ml"))
    assert out.title.lower().startswith("research:") or "no prior" in out.title.lower()
```

## Test through the `as_capability` adapter

```python
from ballast.coala import as_capability


@pytest.mark.asyncio
async def test_unit_as_capability_after_run_calls_learn() -> None:
    captured = []

    class _TrackingUnit(CoALABase[str, str, dict, str]):
        async def retrieve(self, obs): return {}
        async def act(self, obs, ctx): return "out"
        async def learn(self, obs, ctx, out):
            captured.append((obs, out))

    cap = as_capability(_TrackingUnit())
    per_run = await cap.for_run(ctx=None)

    class _FakeCtx:
        deps = {}
    class _FakeResult:
        output = "agent result"

    await per_run.after_run(ctx=_FakeCtx(), result=_FakeResult())
    assert captured == [(None, "agent result")] or len(captured) == 1
```

## Test through the `as_workflow` adapter (DBOS-bound)

This requires DBOS fixture — see [test-workflows-with-dbos-fixture.md](test-workflows-with-dbos-fixture.md). Generally avoid for unit tests; reserve for one or two integration smoke tests.

## Caveats

- **CoALABase defaults are part of your test surface.** If you don't override `observe`/`learn`, your tests should still confirm the defaults behave correctly (identity / no-op). The framework has its own tests for these defaults; you don't need to re-test them.
- **`InMemoryNoteRepository` mutability.** Make a fresh instance per test via `pytest.fixture` + `monkeypatch` to avoid cross-test pollution.
- **Don't test the framework's adapter internals.** They're well-tested in `tests/coala/`. Your tests should treat them as black boxes.

## Related

- [build-coala-unit.md](build-coala-unit.md) — designing CoALA units
- [test-agents-with-testmodel.md](test-agents-with-testmodel.md) — TestModel patterns
- [test-workflows-with-dbos-fixture.md](test-workflows-with-dbos-fixture.md) — DBOS bootstrap when needed
- Reference: `reference/coala/coala-unit-protocol.md`

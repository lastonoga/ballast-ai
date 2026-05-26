# 25. Custom extensions

**Prerequisites:** chapters 7, 11, 13, 14, 19, 21, 23 (the Protocol-defining ones).

## Introduction

Every framework eventually meets a use case its built-ins don't cover. The question is what happens then: are you stuck waiting for upstream, monkey-patching, forking? The Ballast answer: every meaningful surface is a Protocol or ABC, so you write your own implementation and slot it in without touching the framework. This chapter is the recipe book for that.

You'll write a custom capability, pattern, step, HITL channel, threshold policy, fallback policy, scorer, drift goal source, embedder, and repository. None of the templates require a framework PR; all of them live entirely in your codebase.

The unifying philosophy: the framework owns *what* (the contract) and *when* (the dispatch order). You own *how* (the implementation).

## The Protocol-first design (recap)

Every extensible surface follows the same shape:

```
ballast.X.Y       # Protocol — what the framework calls
ballast.X.impl_a  # Built-in implementation
ballast.X.impl_b  # Another built-in
your_app.X.impl_z # Your own — same Protocol, same contract
```

Switching from `impl_a` to `your_app.X.impl_z` is a one-line change at the wiring site. The rest of the system doesn't notice — the Protocol is the contract.

## Writing a custom `BallastCapability`

```python
from ballast import BallastCapability
from pydantic_ai.messages import ToolCallPart

class MaxToolCallsPerRun(BallastCapability):
    name = "max_tool_calls"

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self._count = 0

    async def for_run(self, ctx):
        # Per-run isolation — fresh counter per run
        return MaxToolCallsPerRun(max_calls=self.max_calls)

    async def after_model_request(self, ctx, *, request_context, response):
        for part in response.parts:
            if isinstance(part, ToolCallPart):
                self._count += 1
        if self._count > self.max_calls:
            raise RuntimeError(f"too many tool calls ({self._count}); limit {self.max_calls}")
        return response
```

The rules from chapter 7: per-run state on the `for_run` clone, hooks fast, don't raise unless you mean to abort.

## Writing a custom `Pattern`

```python
from typing import ClassVar
from ballast import Durable
from ballast.patterns.dbos import DBOSConfiguredInstance

@Durable.dbos_class()
class TwoStagePipeline(DBOSConfiguredInstance):
    name: ClassVar[str] = "two_stage_pipeline"

    def __init__(self, *, first_agent, second_agent, config_name=None):
        super().__init__(config_name=config_name or "default")
        self._first = first_agent
        self._second = second_agent

    @Durable.workflow()
    async def run(self, input: dict) -> dict:
        intermediate = await self._call_first(input)
        final = await self._call_second(intermediate)
        return final

    @Durable.step()
    async def _call_first(self, input):
        return (await self._first.run(input)).output

    @Durable.step()
    async def _call_second(self, intermediate):
        return (await self._second.run(intermediate)).output
```

The shape: `@Durable.dbos_class` on the class, `DBOSConfiguredInstance` inheritance, `@Durable.workflow` on `run`, `@Durable.step` on per-call methods, `config_name` for instance isolation. That's the durability recipe.

## Writing a custom `Step` for PlanAndExecute

```python
from ballast.patterns.plan_execute import StepContext

class HttpStep:
    async def execute(self, plan_input, dep_outputs, ctx: StepContext):
        url = ctx.params.get("url")
        body = dep_outputs.get(ctx.params["body_from"], {})
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=body)
            return resp.json()

registry = StepRegistry.with_defaults()
registry.register_step("http", HttpStep())
```

Now the planner can emit `PlannedStep(kind="http", params={"url": "...", "body_from": "..."}, ...)`. The executor dispatches your step automatically.

## Writing a custom `HITLChannel`

Extend `DBOSHITLChannel` so you get the durable wait for free:

```python
from ballast.patterns.hitl import DBOSHITLChannel

class TelegramApprovalChannel(DBOSHITLChannel[InT, MyVerdict]):
    def __init__(self, bot, chat_id):
        self._bot = bot
        self._chat_id = chat_id

    async def deliver(self, *, request_id, workflow_id, respond_topic, payload):
        keyboard = build_inline_keyboard(request_id, respond_topic, payload)
        await self._bot.send_message(
            chat_id=self._chat_id,
            text=format_approval_text(payload),
            reply_markup=keyboard,
        )
        # When the user clicks a button, your webhook handler calls:
        # await Durable.send_async(workflow_id, raw_verdict, topic=respond_topic)

    async def decode_verdict(self, raw: Any) -> MyVerdict:
        return MyVerdict(**raw)
```

Two methods to implement. Everything else (`request`, durable suspension) is inherited.

## Writing a custom `ThresholdPolicy`

```python
from datetime import datetime
from collections import deque

class BurstThreshold:
    """Open if N failures in the last `window` AND no successes after them."""

    def __init__(self, max_failures: int = 5, window_seconds: float = 30.0):
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self._events: deque[tuple[datetime, bool]] = deque()  # (when, success)

    def record_success(self, *, when: datetime, scope: str) -> None:
        self._events.append((when, True))
        self._prune(when)

    def record_failure(self, *, when: datetime, scope: str) -> None:
        self._events.append((when, False))
        self._prune(when)

    def should_open(self, *, when: datetime, scope: str) -> bool:
        self._prune(when)
        failures_after_last_success = 0
        for when_e, success in reversed(self._events):
            if success:
                break
            failures_after_last_success += 1
        return failures_after_last_success >= self.max_failures

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.window_seconds)
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
```

Plug into `CircuitBreaker(threshold_factory=lambda: BurstThreshold(...))`.

## Writing a custom `FallbackPolicy`

```python
class RetryWithBackoff:
    def __init__(self, *, retries: int = 3, base_delay: float = 1.0):
        self.retries = retries
        self.base_delay = base_delay

    async def handle(self, stats):
        for i in range(self.retries):
            await asyncio.sleep(self.base_delay * (2 ** i))
            try:
                return await retry_the_underlying_call()
            except Exception:
                continue
        raise CircuitOpenError(stats)
```

Plug into `CircuitBreaker(fallback=RetryWithBackoff(...))`.

Be careful: a retry policy at the *fallback* layer is different from retry inside the call. The breaker has already decided the underlying is failing; retrying it immediately is usually wrong. This example is illustrative; in practice fallbacks usually return degraded values or escalate.

## Writing a custom `Scorer`

```python
class CitationDensityScorer:
    name = "citation_density"
    threshold = 0.5

    async def score(self, run):
        if run.error or run.output is None:
            return 0.0
        text = run.output.summary if hasattr(run.output, "summary") else str(run.output)
        citation_count = len(re.findall(r"\[[0-9]+\]", text))
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        density = citation_count / word_count
        return min(density / 0.05, 1.0)   # 5% citation density = full score
```

Plug into `dataset.evaluate(runner, evaluators=[CitationDensityScorer(), ...])`.

## Writing a custom `GoalSource` / `TraceWindow` / `DriftHandler`

```python
# A goal source that pulls from your app's database
class DatabaseGoalSource:
    def __init__(self, goal_repo):
        self._repo = goal_repo

    async def goal(self, ctx) -> str:
        workflow_id = ctx.workflow_id
        return await self._repo.get_goal_for(workflow_id)


# A window that uses semantic similarity to keep relevant messages
class RelevantOnlyWindow:
    def __init__(self, embedder, k: int = 5):
        self._embedder = embedder
        self._k = k

    async def slice(self, ctx) -> list:
        goal_emb = await self._embedder.embed(ctx.current_goal)
        scored = []
        for msg in ctx.full_trace:
            msg_emb = await self._embedder.embed(format(msg))
            scored.append((cosine_sim(goal_emb, msg_emb), msg))
        scored.sort(reverse=True)
        return [msg for _, msg in scored[:self._k]]


# A handler that POSTs to a webhook
class WebhookDriftHandler:
    def __init__(self, webhook_url, http_client):
        self._url = webhook_url
        self._client = http_client

    async def handle(self, verdict, ctx) -> None:
        await self._client.post(self._url, json={
            "drifted": verdict.drifted,
            "rationale": verdict.rationale,
            "workflow_id": ctx.workflow_id,
        })
```

All three plug into `DriftEngine` constructor.

## Writing a custom `Embedder`

```python
class CohereEmbedder:
    def __init__(self, client, model: str = "embed-v3"):
        self._client = client
        self._model = model

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.embed(texts=[text], model=self._model)
        return resp.embeddings[0]
```

Plug into `SemanticLoopDetector(embedder=CohereEmbedder(...))` or any DC `EmbeddingDeduper`.

## Writing a custom `ThreadRepository` / `ApprovalCardRepository`

For a Mongo-backed thread repository:

```python
class MongoThreadRepository:
    def __init__(self, db):
        self._threads = db.threads
        self._messages = db.messages

    async def create(self, *, agent: str, metadata=None) -> Thread:
        thread_id = uuid4()
        doc = {"_id": str(thread_id), "agent": agent, "metadata": metadata or {}}
        await self._threads.insert_one(doc)
        return Thread(id=thread_id, agent=agent, metadata=metadata or {})

    async def add_message(self, thread_id, *, role, parts, id=None, silent=False) -> Message:
        ...

    # ... rest of the Protocol

app = Ballast(settings).with_thread_repo(MongoThreadRepository(my_mongo_db)).fastapi(...)
```

The framework's routers and capabilities use the Protocol-typed `ThreadRepository`; they don't know or care about the implementation.

## Publishing as a separate package

If your extension is reusable (a custom HITL channel for Slack, a Mongo repo, a specialized scorer), publish it as its own pip package:

```
my_ballast_slack_channel/
├── pyproject.toml
└── my_ballast_slack_channel/
    ├── __init__.py
    └── channel.py
```

Depend on `ballast` in your `pyproject.toml`. Users install with `pip install my_ballast_slack_channel` and import alongside framework primitives. No coordination with framework maintainers needed.

If your extension turns out to be generally useful, propose it as a built-in via PR. But you don't have to — many extensions are perfectly happy as third-party packages.

## End

You've reached the end of the concepts track. The framework has been laid out in full: agents and tools at the bottom, capabilities and patterns and units in the middle, deployment and persistence and observability at the top. Everything you've seen is composable; everything is replaceable.

What to do now: pick an actual problem, build it. The framework was designed by building real applications and noticing what kept being painful — every primitive in here exists because some real app needed it. Use it the same way.

If you got this far in one sitting, you've covered the framework end to end. Time to actually build something — return to [tutorial/](../tutorial/README.md) for the end-to-end project.

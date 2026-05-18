# Iteration 2 retrospective — backend + single agent + streaming

## What worked smoothly

- **`Engine.fastapi_app(extra_routers=[...])` is the right shape.** Wiring a
  real app took ~10 lines: build a repo, build a runner, hand both routers
  to the factory. Lifespan / `app.state.container` / `/healthz` come for
  free.
- **`InMemoryThreadRepository` is a complete drop-in.** No subclassing, no
  override gymnastics — the `ThreadRepository` Protocol means I can later
  swap to Postgres without touching the routers.
- **The `StreamEvent` + `StreamEncoder` split is clean.** `agent_runner`
  speaks the protocol-neutral `StreamEvent`, the framework's `AGUIEncoder`
  turns it into SSE frames. Iteration 3's frontend can pick `ag-ui` or
  `vercel` via `?protocol=` with zero changes here.
- **`Engine(providers=[])` is legal.** Iteration 2 wants no DBOS / no
  Postgres, and the framework didn't fight that — empty provider list +
  `fastapi_app()` boots cleanly.
- **`get_tenant_id` already does the `X-Tenant-Id`-header thing.** No need
  to write tenant middleware just to satisfy the threads router.

## Friction points (framework ergonomics gaps)

1. **No built-in pydantic-ai → `StreamEvent` adapter.** Every consumer of
   `build_streaming_router` is forced to hand-roll the same translation:
   call `agent.run_stream(...)`, iterate `stream_output(...)` (or
   `stream_text(...)`), diff against the last emitted prefix to compute a
   true text delta, emit `text_delta` + `done` + `error`. This is the
   single biggest copy-paste hazard. A shipping
   `pydantic_ai_stateflow.adapters.pydantic_ai.make_runner(agent, *,
   text_field="reply")` would erase ~80 lines of boilerplate per app.
2. **`_PostMessageBody.parts: list[dict]` has no documented schema.** The
   router validates "it's a list of dicts" and stops. Every agent_runner
   re-invents `_extract_user_text(parts)`. Either ship a typed `MessagePart`
   union (text / tool_result / file_ref) or at least a helper
   `framework.api.streaming.extract_text(parts) -> str`.
3. **`AgentRunner` signature is `Callable[..., AsyncIterator[StreamEvent]]`
   with three named kwargs (`thread_id`, `message`, `tenant_id`) only
   discoverable by reading `router.py`.** Should be a Protocol class with
   precise types — mypy currently can't catch a misnamed kwarg in the
   runner.
4. **The router persists the user message but never the assistant reply.**
   After `done`, nothing writes the assistant turn back into the repo, so
   `GET /threads/{id}/messages` will not contain the LLM output. Either the
   runner is contractually responsible for `repo.add_message(role="assistant",
   ...)` after `done` (undocumented), or the router should do it
   automatically from the `done` event payload.
5. **`AGUIEncoder` event names (`text_delta`, `done`) are an undocumented
   contract with the frontend.** There is no `StreamEventKind` enum or
   docstring saying which kinds the AG-UI encoder will pass through, what
   assistant-ui expects, or how to extend. Iteration 3 will discover this
   the hard way.
6. **Engine boot prints nothing.** A booted-with-no-providers app would be
   indistinguishable from a misconfigured one in production logs. A single
   structured log line on boot (`providers=[], invariants=[]`) would help.
7. **`Engine.fastapi_app()` has no `cors=` or `lifespan_hooks=` knob.** Any
   real frontend will need CORS configured before the first browser call.
   Right now I'd have to drop down to mutating the returned `FastAPI`.

## OpenRouter + pydantic-ai notes

The integration is **two-line trivial** in pydantic-ai 1.97.0:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
agent = Agent(
    model=OpenAIModel("qwen/qwen3.6-plus", provider=provider),
    output_type=ChatReply,           # pydantic BaseModel
    system_prompt="You are ...",
)
```

- **Structured output for `output_type=ChatReply` is routed through
  function/tool-calling** on the OpenAI-compatible backend (not
  `response_format: json_schema`). pydantic-ai registers a synthetic
  `final_result` tool whose schema matches `ChatReply`. Qwen models on
  OpenRouter advertise tool-calling, so this works out of the box — no
  `response_format` shim, no JSON-mode fallback needed.
- **Streaming under `output_type=BaseModel` uses partial pydantic validation.**
  `result.stream_output(debounce_by=0.05)` yields progressively-validated
  `ChatReply` instances as JSON tokens arrive. The `reply` field grows
  monotonically in the common case, but pydantic's partial mode is allowed
  to revise (e.g. when a `"` closes a token-truncated string) — our adapter
  defensively handles that by falling back to a full re-emit if the new
  value isn't a prefix-extension.
- **`stream_text(delta=True)` would be simpler but is `str`-output-only.**
  Pinned to `output_type=ChatReply`, we have to use `stream_output` + diff.

## Stream-event contract emitted by `agent_runner`

| kind         | data shape                              | when                       |
| ------------ | --------------------------------------- | -------------------------- |
| `text_delta` | `{"text": "<incremental chunk>"}`       | each partial token group   |
| `done`       | `{"reply": "<final full reply string>"}`| once, on clean completion  |
| `error`      | `{"message": "<stringified exception>"}`| once, on any failure       |

Iteration 3 must verify the assistant-ui frontend's AG-UI subscriber
recognises `text_delta` (it's the AG-UI canonical name) and treats `done`
as a stream terminator. If assistant-ui expects e.g. `text_message_start` /
`text_message_content` / `text_message_end` we'll surface that mismatch
here and either patch the frontend mapping or extend `AGUIEncoder`.

## Framework gaps for iteration 3

- [ ] `pydantic_ai_stateflow.adapters.pydantic_ai.make_runner(agent, ...)`
      that owns the run-stream + diff + emit loop.
- [ ] Typed `MessagePart` union for `_PostMessageBody.parts`, plus
      `extract_text(parts)` helper.
- [ ] `AgentRunner` as a typed `Protocol` (not `Callable[..., ...]`).
- [ ] Auto-persist the assistant reply on `done`, or document the
      runner's contractual responsibility to call `repo.add_message`.
- [ ] `StreamEventKind` constants / enum + table of which kinds each
      encoder (`AGUIEncoder`, `VercelEncoder`) emits, with
      assistant-ui compatibility notes.
- [ ] `Engine.fastapi_app(cors=..., lifespan_hooks=...)`.
- [ ] One structured `INFO` log line on `Engine.boot()`.

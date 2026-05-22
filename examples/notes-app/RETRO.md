# Retrospectives per iteration

Each iteration appends one section here. Format: **what worked**, **what was awkward** (= framework gap), **decided action** (fix now / defer / accept).

---

## Iteration 1 + 2 (parallel: frontend + backend, no wiring yet)

Full per-side notes in `frontend/RETRO.md` and `backend/RETRO.md`. This section consolidates the two and lists the framework changes required before iteration 3 (frontend ↔ backend wiring).

### What worked

- **`Engine.fastapi_app(extra_routers=[...])` was ~10 lines** to stand up a real app: build repo, build runner, hand both routers to the factory. `/healthz`, `app.state.container`, lifespan all free.
- **assistant-ui via the shadcn registry vended 9 components** (`Thread`, `ThreadList`, `Composer`, `EditComposer`, `ActionBar`, `BranchPicker`, `Markdown`, `Reasoning`, `ToolGroup`+`ToolFallback`) covering the entire chat surface. Only ~50 lines of custom client code (a mock adapter + theme toggle).
- **OpenRouter + pydantic-ai integration is two-line trivial** via `OpenAIProvider(base_url="https://openrouter.ai/api/v1", api_key=...)`. `output_type=ChatReply` routes through synthetic `final_result` tool-calling; Qwen on OpenRouter supports it natively, no `response_format` shim needed.

### CRITICAL contract mismatches surfaced

**1. Stream payload semantics (BLOCKER for iter 3).**
- Backend emits **deltas** today: `text_delta {text: "<chunk>"}`, `done {reply: "<full>"}`. The adapter diffs `result.stream_output(...)` to compute true increments.
- assistant-ui's `ChatModelAdapter.run` expects **full snapshots** on every `yield`: `{content: MessagePart[], metadata?, status?}` where each yield is the complete message-so-far.
- **Decision**: switch backend to snapshot-per-yield. Drop the diffing logic; emit the progressive `ChatReply` as it grows. This aligns with how `result.stream_output(debounce_by=...)` natively works anyway.

**2. Wire format selection.** assistant-ui has three viable wire-format adapters:
- **DataStream** (`@assistant-ui/react-data-stream`) — assistant-ui's native protocol, newline-delimited typed events.
- **AG-UI** (`@assistant-ui/react-ag-ui`) — already in our framework spec; `useAgUiThreadRuntime({runtimeUrl})`.
- **Custom `ChatModelAdapter`** — what our mock uses; backend implements its own SSE format and the client adapter consumes it.
- **Decision**: go AG-UI (matches the spec, lowest glue). Verify `@assistant-ui/react-ag-ui` accepts the AG-UI event shape our `AGUIEncoder` produces; if not, the encoder needs to emit `text_message_start` / `text_message_content` / `text_message_end` / `tool_call_start` / `tool_call_args` / `tool_call_end` / `finish` per AG-UI canonical events (this is what our encoder is missing — we only emit `text_delta` / `done` / `error`).

**3. Thread CRUD coverage.** Frontend's `RemoteThreadListAdapter` contract requires: `list`, `rename`, `archive`, `delete`, `initialize`, `generateTitle`. Our `build_threads_router` ships: `POST /threads`, `GET /threads/{id}`, `GET /threads/{id}/messages`. **Missing: list-all, rename, archive (soft delete), delete, generateTitle.** Also tenant header capitalization differs (assistant-ui uses `x-tenant-id` lowercase by convention; FastAPI normalizes anyway, but worth documenting).

**4. Assistant-reply persistence.** Backend persists the user message but not the assistant reply. Either runner contractually does `repo.add_message(role="assistant", ...)` on `done` (currently undocumented), or router auto-persists from the `done` payload. **Decision**: router auto-persists from `done.reply` text — keeps the runner contract narrow.

**5. Abort propagation.** assistant-ui's stop button → `AbortSignal` → HTTP disconnect. Our streaming endpoint doesn't currently react to client disconnect to cancel the upstream `agent.run_stream`. Backend will leak the LLM call after a client stop.

### Framework changes BEFORE iteration 3

Promoted from the per-side RETROs into a concrete fix list:

| # | Change | Source | Effort |
|---|---|---|---|
| F1 | `AGUIEncoder` emits canonical AG-UI events (`text_message_start/content/end`, `tool_call_*`, `finish`) — not bespoke `text_delta`/`done` | both | M |
| F2 | `ballast.adapters.pydantic_ai.make_runner(agent, *, text_field="reply")` — eliminates ~80 LOC of run-stream glue per app | backend | S |
| F3 | Typed `MessagePart` union + `extract_text(parts)` helper on `_PostMessageBody` | backend | S |
| F4 | `AgentRunner` becomes a typed `Protocol` (currently `Callable[..., AsyncIterator[StreamEvent]]`) | backend | XS |
| F5 | Snapshot-mode runner option (yield full content, framework computes deltas if needed) — drops backend diff burden | both | M |
| F6 | `build_threads_router` adds: `GET /threads` (list), `PATCH /threads/{id}` (rename), `POST /threads/{id}/archive`, `DELETE /threads/{id}`, `POST /threads/{id}/title` | frontend | M |
| F7 | Router auto-persists assistant reply on terminal stream event | both | S |
| F8 | `Engine.fastapi_app(cors=..., lifespan_hooks=...)` knobs | backend | XS |
| F9 | `StreamEventKind` enum + per-encoder compatibility table + assistant-ui mapping docs | both | XS |
| F10 | Abort propagation: detect client disconnect in `_gen()`, cancel the upstream async iterator | frontend | M |
| F11 | Tool-call streaming events in `StreamEvent` taxonomy (`tool_call_start`, `tool_call_args_delta`, `tool_call_end`) with stable `toolCallId` | frontend | M |

XS=<30 min, S=<2 h, M=half-day.

### Deferred to later iterations (not blocking iter 3)

- HITL approval card primitive (`<HitlApprovalCard />` + `__hitl_approval` tool-name contract) — iter 4 deliverable.
- Streaming `metadata.timing` for TTFT/tokens-per-sec badges — iter 7 (observability) deliverable.
- `RemoteThreadListAdapter`'s `generateTitle` agent — iter 5 (Reflection) territory.
- Suggestions adapter — defer until UX warrants it.

### Iteration 3 plan (post-fixes)

1. Apply F1–F11 to framework `src/ballast/`, with tests.
2. Switch frontend from `useLocalRuntime(mock)` → `useAgUiThreadRuntime({runtimeUrl: ...})` (or `useRemoteThreadListRuntime` wrapping a remote message runtime — pick the one that ships in `@assistant-ui/react-ag-ui` first).
3. Add CORS to backend; point frontend at `http://localhost:8000` (env var).
4. Add the notes domain (SQLModel `Note`, agent tools `create_note` / `edit_note` / `delete_note` / `search_notes`), one tenant for now.
5. End-to-end smoke: open browser, create thread, ask "make me a note about X", see the tool-call card render, see the note persisted in the backend DB.

---

## Iteration 3 — wiring backend ↔ frontend, notes domain + tools

Two parallel commits: `caad2e4` (backend domain + tools) and `f64091f` (frontend real runtime). End-to-end CRUD smoke against backend confirmed (POST/GET/PATCH/DELETE /threads all 200/204 via curl).

### What worked

- **F1–F11 paid off.** Canonical AG-UI events + auto-persist + thread CRUD endpoints + `make_runner` + `MessagePart` types let both halves land in parallel with no protocol re-negotiation.
- **`@ag-ui/client`'s `HttpAgent` accepted our canonical events as-is** — camelCase field names, event-type names, all wire-compatible. Zero translation layer in the frontend.
- **5 agent tools registered cleanly** (`create_note` / `list_notes` / `search_notes` / `edit_note` / `delete_note`) via `@agent.tool` + `NoteToolDeps(repo, tenant_id)` deps_type.
- **Backend smoke**: `POST /threads` → `GET /threads` → `PATCH /threads/{id}` rename → `DELETE /threads/{id}` 204 all worked end-to-end via curl with a fresh tenant header.

### Friction surfaced (new gap list — F12–F19)

| # | Source | What | Effort |
|---|---|---|---|
| F12 | backend | `make_runner` accepts only static `deps`; multi-tenant apps must hand-roll the diff/emit loop to inject per-request `NoteToolDeps`. Should accept `deps: Any \| Callable[..., Any \| Awaitable[Any]] = None` where the callable receives runner kwargs. | S |
| F13 | backend | We added tool-call `StreamEvent` kinds in F11 but **no producer emits them yet**. `make_runner` is text-only; agent tool calls fire on the backend but the frontend never sees a `tool_call_start/args/end` SSE event, so assistant-ui's tool-call cards never render. Extend `make_runner` (or ship a sibling) to emit canonical tool-call events as pydantic-ai's stream yields tool-call parts. | M |
| F14 | backend | No domain-repo scaffold / docs. `NoteRepository` + `InMemoryNoteRepository` were rebuilt from scratch mirroring `ThreadRepository`. Ship a `ballast.persistence.scaffold` doc or template. | XS |
| F15 | backend | `Engine` has no DI hook for app-defined domain repos. The example uses a module-level singleton; production apps will want `engine.container.bind(NoteRepository, factory)`. Wire repos through the Container. | S |
| F16 | frontend | `HttpAgent` posts to one fixed URL; our backend uses `POST /threads/{id}/messages?protocol=ag-ui`. Subclassed as `NotesAppAgent` overriding `.run(input)` to rebuild `this.url` from `input.threadId`. Ship a thread-agnostic `POST /agent?thread_id=...` runtime URL OR ship a first-class "thread-aware HttpAgent" wrapper from the framework. | S |
| F17 | frontend | `generateTitle` adapter method requires an assistant-stream `ReadableStream`. We return an empty stream (manual rename only). Backend `POST /threads/{id}/title` agent-driven streaming would fix this — deferred from Group C. | M |
| F18 | frontend | `GET /threads` doesn't honor `offset` pagination. With dozens of threads the UI would over-fetch. Add `offset` parameter. | XS |
| F19 | both | Suggestion adapter endpoint (`POST /threads/{id}/suggestions` → `[{prompt: str}]`). Welcome screen suggestions are static today. | S |

### Decisions for next iterations

- **F12 + F13 first** (before iteration 4) — without them, the UI works but feels lobotomised: agent tool-calls succeed on the backend, notes are created in the repo, but the user sees only the final text reply with no tool-call card. The dogfood loses its "look at the tool calls" demo value.
- **F14 + F15** — punt to iteration 4 or 5; not blocking dog-fooding.
- **F16** — punt to iteration 4 (when HITL also touches the streaming endpoint shape).
- **F17 (title streaming)** — wait until iteration 5 (Reflection) — then the title generator can be a Reflection-wrapped agent.
- **F18, F19** — polish; defer.

### Iteration 4 plan (HITL)

Before adding `HITLGate` + approval cards, **fix F12 + F13** so the tool-call demo actually visible. Then:

1. `make_runner` gains `deps` callable + tool-call event emission.
2. App adds `BudgetGuard(max_iterations=10)` so the model can't infinite-loop the tools.
3. Add an `ApprovalStage` wrapping `delete_note` (destructive) via `HITLGate(channel=UIChannel(), policy=AllowAll())`.
4. Frontend ships a `<HitlApprovalCard />` bound to a stable tool name (e.g. `__hitl_approval`).
5. End-to-end: "delete the note about X" → approval card appears → user clicks Approve → note is deleted.

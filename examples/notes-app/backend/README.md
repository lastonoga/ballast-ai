# notes-app backend (iteration 3)

FastAPI backend wired via `Engine.fastapi_app()`, backed by a single
pydantic-ai agent that hits OpenRouter (Qwen, structured JSON output),
with CRUD tools over an in-memory notes domain.

The agent can now:

- **`create_note(title, body) -> Note`** — saves a new note for the
  current tenant; returns the saved `Note` (with `id`) so the model can
  chain follow-ups.
- **`list_notes(limit=20) -> list[Note]`** — newest-first.
- **`search_notes(query, limit=20) -> list[Note]`** — case-insensitive
  substring on title+body, newest-first.
- **`edit_note(note_id, title=None, body=None) -> Note`** — partial
  update; only the provided fields change. Raises if the note doesn't
  belong to the current tenant.
- **`delete_note(note_id) -> str`** — idempotent.

Iteration 4 adds HITLGate to gate the write tools behind explicit user
approval. Iteration 4+ also swaps `InMemoryNoteRepository` for a
Postgres/DBOS-backed one (see `notes/repository.py` TODO).

## Run

```bash
cd examples/notes-app/backend
uv sync --extra dev
cp .env.example .env  # then fill OPENROUTER_API_KEY
uv run alembic upgrade head      # create notes-app.sqlite
uv run uvicorn notes_app.main:app --reload
```

## Persistence

App state (notes, threads, messages, event log) lives in a local SQLite
file (`./notes-app.sqlite`). DBOS workflow state has its own file
(`./notes-app.dbos.sqlite`). Both auto-created on first run.

To set up / upgrade the schema:

    uv run alembic upgrade head

Then start the backend:

    uv run python -m notes_app.main
    # or:
    uv run uvicorn notes_app.main:app --reload --port 8000

Point at a different database with `NOTES_APP_DATABASE_URL`:

    NOTES_APP_DATABASE_URL=postgresql+asyncpg://... uv run alembic upgrade head
    NOTES_APP_DATABASE_URL=postgresql+asyncpg://... uv run uvicorn notes_app.main:app

Set `NOTES_APP_DATABASE_URL=""` (empty) or `NOTES_APP_DATABASE_URL=":memory:"`
to fall back to InMemory repos (process-local, no persistence).

To reset state: `rm notes-app.sqlite` (then re-run alembic upgrade head).

### Tests stay in-memory

`tests/conftest.py` + `main.py` together guarantee that when running
under pytest, the module-level singletons are `InMemoryNoteRepository`,
`InMemoryThreadRepository`, and `InMemoryEventLogRepository` — test
runs never touch the local sqlite file.

## Architecture

```
notes_app/
├── agent.py             # Agent[NoteToolDeps, ChatReply] + build_notes_runner
├── main.py              # FastAPI wiring; module-scope InMemoryNoteRepository
└── notes/
    ├── domain.py        # NoteRow (SQLModel) + Note (immutable projection)
    ├── repository.py    # NoteRepository Protocol + InMemoryNoteRepository
    └── tools.py         # NoteToolDeps + register_note_tools(agent)
```

`NoteToolDeps(repo, tenant_id)` is the per-request dependency the agent
tools see via `ctx.deps`. The runner in `main.py` builds a fresh one per
HTTP request (one tenant per `X-Tenant-Id` header).

## Endpoints

All requests must include `X-Tenant-Id: <uuid>`.

### Create a thread

```bash
TENANT=$(uuidgen)
curl -s -X POST http://127.0.0.1:8000/threads \
  -H "X-Tenant-Id: $TENANT" \
  -H "Content-Type: application/json" \
  -d '{"purpose":"chat","actor_id":"alice"}'
```

Response: full `Thread` JSON. Save its `id`.

### Ask the agent to create a note

```bash
curl -N -X POST http://127.0.0.1:8000/threads/$THREAD_ID/messages \
  -H "X-Tenant-Id: $TENANT" \
  -H "Content-Type: application/json" \
  -d '{
        "role":"user",
        "parts":[{
          "type":"text",
          "text":"Create a note titled \"Grocery list\" with body \"milk, eggs, bread\"."
        }]
      }'
```

The response is an `text/event-stream`; you'll see
`RUN_STARTED → TEXT_MESSAGE_START → TEXT_MESSAGE_CONTENT × N → TEXT_MESSAGE_END → RUN_FINISHED`,
and behind the scenes the in-memory notes store has a new row for `$TENANT`.

Other natural-language prompts you can try:

- "Show me my notes."
- "Search my notes for milk."
- "Update the grocery note to also include yogurt."
- "Delete the grocery note."

## Test

```bash
uv run pytest          # in-memory; skips OpenRouter tests if no key
uv run mypy src
uv run ruff check .
```

## See also

- `RETRO.md` — what worked, friction points, and framework gaps surfaced
  in this iteration.
- `../README.md` — iteration plan for the whole notes-app reference.

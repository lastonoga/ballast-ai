# notes-app backend (iteration 2)

FastAPI backend wired via `Engine.fastapi_app()`, backed by a single
pydantic-ai agent that hits OpenRouter (Qwen, structured JSON output).

No domain logic yet — the agent just chats and returns `{reply: str}`.
Iteration 3 adds the notes domain + tools.

## Run

```bash
cd examples/notes-app/backend
uv sync --extra dev
cp .env.example .env  # then fill OPENROUTER_API_KEY
uv run uvicorn notes_app.main:app --reload
```

The framework's `Engine.fastapi_app()` registers a `/healthz` endpoint, the
threads router, and the streaming router.

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

### Get a thread

```bash
curl -s http://127.0.0.1:8000/threads/$THREAD_ID \
  -H "X-Tenant-Id: $TENANT"
```

### Post a message + stream the reply (AG-UI SSE)

```bash
curl -N -X POST http://127.0.0.1:8000/threads/$THREAD_ID/messages \
  -H "X-Tenant-Id: $TENANT" \
  -H "Content-Type: application/json" \
  -d '{"role":"user","parts":[{"type":"text","text":"hi, what can you do?"}]}'
```

The response is an `text/event-stream` body that emits:

```
event: text_delta
data: {"text":"Hi"}

event: text_delta
data: {"text":"! I can"}

...

event: done
data: {"reply":"Hi! I can help you with..."}
```

## Test

```bash
uv run pytest          # in-memory; skips OpenRouter test if no key
uv run mypy notes_app
uv run ruff check .
```

## See also

- `RETRO.md` — what worked, friction points, and framework gaps surfaced in
  this iteration.
- `../README.md` — iteration plan for the whole notes-app reference.

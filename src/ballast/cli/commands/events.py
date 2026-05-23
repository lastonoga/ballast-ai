"""``ballast events tail <thread-id>`` — SSE tail."""
from __future__ import annotations

import asyncio
import json as _json
from uuid import UUID

import typer
from rich.console import Console
from rich.json import JSON

app = typer.Typer(help="Thread event introspection.", no_args_is_help=True)


@app.command(name="tail")
def tail(
    thread_id: UUID = typer.Argument(...),
    host: str = typer.Option("localhost", "--host"),
    port: int = typer.Option(8000, "--port"),
    scheme: str = typer.Option("http", "--scheme"),
) -> None:
    """Tail SSE events for a thread.

    Example:

        ballast events tail 9b1a-...
        ballast events tail 9b1a-... --port 8001
    """
    asyncio.run(_tail_async(thread_id, host, port, scheme))


async def _tail_async(thread_id: UUID, host: str, port: int, scheme: str) -> None:
    import httpx

    url = f"{scheme}://{host}:{port}/threads/{thread_id}/events"
    console = Console()
    last_event_id: str | None = None
    backoff = 0.5
    while True:
        headers: dict[str, str] = {"Accept": "text/event-stream"}
        if last_event_id is not None:
            headers["Last-Event-ID"] = last_event_id
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    backoff = 0.5  # reset on successful connect
                    event_id: str | None = None
                    data_lines: list[str] = []
                    async for line in response.aiter_lines():
                        if line.startswith("id:"):
                            event_id = line[len("id:"):].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[len("data:"):].strip())
                        elif line == "":
                            # End of event
                            if data_lines:
                                raw = "\n".join(data_lines)
                                _print_event(console, event_id, raw)
                                if event_id is not None:
                                    last_event_id = event_id
                            event_id = None
                            data_lines = []
        except (httpx.HTTPError, OSError) as exc:
            console.print(f"[yellow]disconnected:[/yellow] {exc}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 5.0)
        except KeyboardInterrupt:
            return


def _print_event(console: Console, event_id: str | None, raw: str) -> None:
    try:
        payload = _json.loads(raw)
    except _json.JSONDecodeError:
        console.print(f"seq={event_id or '?'} {raw}")
        return
    kind = str(payload.get("kind", "?"))
    color = {
        "text-delta": "dim cyan",
        "message-added": "green",
        "thread-created": "blue",
        "error": "red",
    }.get(kind, "white")
    console.print(f"seq={event_id or '?'}  [{color}]{kind}[/{color}]")
    if "payload" in payload:
        console.print(JSON(_json.dumps(payload["payload"], default=str)))

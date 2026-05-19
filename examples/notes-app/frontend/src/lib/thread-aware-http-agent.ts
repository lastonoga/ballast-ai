/**
 * Thread-aware variant of AG-UI's `HttpAgent`.
 *
 * Two adaptations on top of the stock `HttpAgent`:
 *
 * 1. **Per-thread URL.** `HttpAgent` posts to a single fixed `url`. Our
 *    backend exposes a per-thread streaming endpoint
 *    (`POST /threads/{threadId}/messages?protocol=ag-ui`), so the URL is
 *    recomputed on every `run()` from `input.threadId`.
 *
 * 2. **Server-stateful body shape.** Stock `HttpAgent.requestInit` JSON-
 *    serializes the full `RunAgentInput` (`{threadId, runId, messages: [...
 *    full client-held history ...], tools, context, ...}`) — the AG-UI
 *    spec assumes a client-stateful server that reruns the agent against
 *    the whole conversation each turn.
 *
 *    Our backend is **server-stateful**: the `Thread` + `Message` rows are
 *    the source of truth, and the backend reconstructs `message_history`
 *    from its own `ThreadRepository` before invoking the pydantic-ai
 *    agent. So we ship ONLY the new user turn in the native
 *    `{role, parts}` shape — everything else (prior turns, tool defs,
 *    context, forwarded props) the backend already has or doesn't need.
 *
 *    We override the protected `requestInit(input)` hook — `@ag-ui/client`
 *    explicitly documents this as the override point: "Returns the fetch
 *    config for the http request. Override this to customize the request."
 *    Everything else (SSE parsing, observable plumbing, abort wiring) is
 *    reused from the base class.
 *
 * Usage:
 *
 *   const agent = new ThreadAwareHttpAgent({
 *     apiUrl: "http://localhost:8000",
 *     headers: { "X-Tenant-Id": tenantId },
 *   });
 *
 * Promote this to a separate npm package when a second app needs it.
 */
import {
  HttpAgent,
  type HttpAgentConfig,
  type RunAgentInput,
} from "@ag-ui/client";

export interface ThreadAwareHttpAgentConfig extends Omit<HttpAgentConfig, "url"> {
  /** Base URL of the backend, e.g. `http://localhost:8000`. */
  apiUrl: string;
  /**
   * Optional override for the per-thread URL builder. Defaults to
   * `${apiUrl}/threads/${threadId}/messages?protocol=ag-ui`.
   */
  buildUrl?: (apiUrl: string, threadId: string) => string;
}

export class ThreadAwareHttpAgent extends HttpAgent {
  private readonly apiUrl: string;
  private readonly buildUrl: (apiUrl: string, threadId: string) => string;

  constructor(config: ThreadAwareHttpAgentConfig) {
    const { apiUrl, buildUrl, ...rest } = config;
    // The `url` passed to super() is a placeholder; `run()` rewrites
    // `this.url` per-call from the incoming `input.threadId`. We keep the
    // placeholder readable so any error logs surface the template form.
    super({ ...rest, url: `${apiUrl}/threads/{threadId}/messages?protocol=ag-ui` });
    this.apiUrl = apiUrl;
    this.buildUrl =
      buildUrl ??
      ((api, id) => `${api}/threads/${id}/messages?protocol=ag-ui`);
  }

  // The base class reads `this.url` synchronously inside its `run()` body
  // to choose the endpoint. Rather than reimplement the rxjs/SSE plumbing,
  // we mutate `this.url` before delegating to `super.run()`.
  override run(input: RunAgentInput) {
    this.url = this.buildUrl(this.apiUrl, input.threadId);
    return super.run(input);
  }

  /**
   * Replace the stock `JSON.stringify(input)` body with our native
   * `{role, parts}` shape carrying ONLY the new user turn. The backend
   * reconstructs `message_history` from its own `ThreadRepository`
   * (server-stateful contract); the rest of `RunAgentInput` (prior
   * `messages`, `tools`, `context`, `state`, `forwardedProps`) is
   * intentionally dropped on the wire.
   *
   * Headers / method / signal are kept identical to the base impl so
   * abort, content-type, and auth-header semantics are preserved.
   */
  protected override requestInit(input: RunAgentInput): RequestInit {
    const lastUserMsg = [...input.messages]
      .reverse()
      .find((m) => m.role === "user");
    const text =
      lastUserMsg && typeof lastUserMsg.content === "string"
        ? lastUserMsg.content
        : "";
    const body = JSON.stringify({
      role: "user",
      parts: [{ type: "text", text }],
    });
    return {
      method: "POST",
      headers: {
        ...this.headers,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body,
      signal: this.abortController.signal,
    };
  }
}

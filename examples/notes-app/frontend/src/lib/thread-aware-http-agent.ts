/**
 * Thread-aware variant of AG-UI's `HttpAgent`.
 *
 * Single adaptation on top of the stock `HttpAgent`:
 *
 * **Per-thread URL.** `HttpAgent` posts to a single fixed `url`. Our
 * backend exposes a per-thread streaming endpoint
 * (`POST /threads/{threadId}/messages`), so the URL is recomputed on
 * every `run()` from `input.threadId`.
 *
 * The wire body is the stock AG-UI `RunAgentInput` JSON — our backend
 * delegates body parsing to `pydantic_ai.ui.ag_ui.AGUIAdapter`, which
 * speaks the canonical AG-UI shape natively. We don't override
 * `requestInit` anymore (server-stateful history is reconstructed from
 * the backend's `ThreadRepository`; the agent simply re-reads it on
 * every request).
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
   * `${apiUrl}/threads/${threadId}/messages`.
   */
  buildUrl?: (apiUrl: string, threadId: string) => string;
}

export class ThreadAwareHttpAgent extends HttpAgent {
  private readonly apiUrl: string;
  private readonly buildUrl: (apiUrl: string, threadId: string) => string;

  constructor(config: ThreadAwareHttpAgentConfig) {
    const { apiUrl, buildUrl, ...rest } = config;
    // The `url` passed to super() is a placeholder; `run()` rewrites
    // `this.url` per-call from the incoming `input.threadId`. Keep the
    // placeholder readable so any error logs surface the template form.
    super({ ...rest, url: `${apiUrl}/threads/{threadId}/messages` });
    this.apiUrl = apiUrl;
    this.buildUrl =
      buildUrl ??
      ((api, id) => `${api}/threads/${id}/messages`);
  }

  // The base class reads `this.url` synchronously inside its `run()` body
  // to choose the endpoint. Rather than reimplement the rxjs/SSE plumbing,
  // we mutate `this.url` before delegating to `super.run()`.
  override run(input: RunAgentInput) {
    this.url = this.buildUrl(this.apiUrl, input.threadId);
    return super.run(input);
  }
}

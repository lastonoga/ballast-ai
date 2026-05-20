/**
 * RemoteThreadListAdapter wired against the notes-app FastAPI backend.
 *
 * Backend endpoints:
 *
 *   GET    /threads?include_archived=false&limit=100&offset=0
 *   POST   /threads               { metadata }   (notes-app endpoint)
 *   GET    /threads/{id}
 *   POST   /threads/{id}/archive
 *   POST   /threads/{id}/unarchive
 *   DELETE /threads/{id}
 *
 * Maps server-side thread rows to the shape assistant-ui's
 * `RemoteThreadListAdapter` expects (`status`, `remoteId`, `title`, ...).
 */
import type {
  RemoteThreadInitializeResponse,
  RemoteThreadListAdapter,
  RemoteThreadListPageOptions,
  RemoteThreadListResponse,
  RemoteThreadMetadata,
} from "@assistant-ui/core";
import { createAssistantStream } from "assistant-stream";

type BackendThread = {
  id: string;
  title?: string | null;
  status?: string;
  archived_at?: string | null;
  agent?: string;
  metadata?: Record<string, unknown>;
};

function toMetadata(t: BackendThread): RemoteThreadThreadMetadataLike {
  const archived =
    t.status === "archived" ||
    (t.archived_at !== null && t.archived_at !== undefined);
  return {
    status: archived ? "archived" : "regular",
    remoteId: t.id,
    title: t.title ?? undefined,
    custom: t.metadata,
  } satisfies RemoteThreadMetadata;
}

// Local alias keeps the satisfies-clause readable.
type RemoteThreadThreadMetadataLike = RemoteThreadMetadata;

export function buildRemoteThreadListAdapter(
  apiUrl: string,
  headers: Record<string, string>,
): RemoteThreadListAdapter {
  const jsonHeaders = { ...headers, "Content-Type": "application/json" };

  async function fetchList(
    includeArchived: boolean,
    opts?: RemoteThreadListPageOptions,
  ): Promise<BackendThread[]> {
    const params = new URLSearchParams();
    params.set("include_archived", String(includeArchived));
    if (opts?.after) params.set("offset", opts.after);
    const r = await fetch(`${apiUrl}/threads?${params}`, { headers });
    if (!r.ok) {
      throw new Error(`GET /threads failed: ${r.status} ${r.statusText}`);
    }
    const body = await r.json();
    // Backend may return a bare array OR { threads: [...] } — accept both.
    return Array.isArray(body) ? body : (body.threads ?? []);
  }

  return {
    async list(opts) {
      // assistant-ui calls list() twice (once for regular, once for archived
      // — the runtime filters internally on `status`). Pull both in parallel
      // and union: cheaper than two adapter round-trips per render.
      const [regular, archived] = await Promise.all([
        fetchList(false, opts),
        fetchList(true, opts).then((all) =>
          all.filter((t) => t.status === "archived" || t.archived_at),
        ),
      ]);
      const byId = new Map<string, BackendThread>();
      for (const t of [...regular, ...archived]) byId.set(t.id, t);
      const response: RemoteThreadListResponse = {
        threads: Array.from(byId.values()).map(toMetadata),
      };
      return response;
    },

    async initialize(): Promise<RemoteThreadInitializeResponse> {
      const r = await fetch(`${apiUrl}/threads`, {
        method: "POST",
        headers: jsonHeaders,
        body: JSON.stringify({}),
      });
      if (!r.ok) {
        throw new Error(`POST /threads failed: ${r.status} ${r.statusText}`);
      }
      const t = (await r.json()) as BackendThread;
      return { remoteId: t.id, externalId: undefined };
    },

    async rename(_remoteId, _newTitle) {
      // Backend dropped PATCH /threads/{id} (no title column anymore).
      // Rename is a client-only UI state for now.
    },

    async archive(remoteId) {
      const r = await fetch(`${apiUrl}/threads/${remoteId}/archive`, {
        method: "POST",
        headers,
      });
      if (!r.ok && r.status !== 404) {
        throw new Error(`archive ${remoteId} failed: ${r.status}`);
      }
    },

    async unarchive(remoteId) {
      const r = await fetch(`${apiUrl}/threads/${remoteId}/unarchive`, {
        method: "POST",
        headers,
      });
      if (!r.ok && r.status !== 404) {
        throw new Error(`unarchive ${remoteId} failed: ${r.status}`);
      }
    },

    async delete(remoteId) {
      const r = await fetch(`${apiUrl}/threads/${remoteId}`, {
        method: "DELETE",
        headers,
      });
      if (!r.ok && r.status !== 404) {
        throw new Error(`DELETE /threads/${remoteId} failed: ${r.status}`);
      }
    },

    async fetch(remoteId): Promise<RemoteThreadMetadata> {
      // assistant-ui's runtime calls fetch() with its internal
      // ``__LOCALID_xxx`` placeholder for threads that haven't been
      // initialized on the backend yet (a fresh draft the user opened
      // but hasn't sent a message in). Hitting the backend with that
      // id 422s on the UUID path validator, AND there's no row to
      // return anyway — short-circuit so the local-only thread stays
      // local until ``initialize()`` flips it to a real remoteId.
      if (remoteId.startsWith("__LOCALID_")) {
        throw new Error(`thread ${remoteId} is local-only; not yet initialized`);
      }
      const r = await fetch(`${apiUrl}/threads/${remoteId}`, { headers });
      if (!r.ok) {
        throw new Error(`GET /threads/${remoteId} failed: ${r.status}`);
      }
      const t = (await r.json()) as BackendThread;
      return toMetadata(t);
    },

    /**
     * Iteration 3: no agent-driven title stream yet.
     *
     * assistant-ui requires this method to return an `AssistantStream`. We
     * return an empty (immediately-closed) stream so the runtime's auto-title
     * step is a no-op; the user can rename via the thread-list more menu (which
     * hits PATCH /threads/{id} via `rename` above).
     *
     * TODO (next round): backend should expose `POST /threads/{id}/title` that
     * streams an `AssistantStream` (one `appendText` call) — see RETRO iter-1
     * framework TODO #7.
     */
    async generateTitle() {
      return createAssistantStream((controller) => {
        controller.close();
      });
    },
  };
}

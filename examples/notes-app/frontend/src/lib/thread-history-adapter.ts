/**
 * ``ThreadHistoryAdapter`` wired against the notes-app FastAPI backend.
 *
 * Persistence flow follows assistant-ui's canonical pattern (see
 * "Build the message endpoints" in the assistant-ui docs):
 *
 *   1. ``load`` — fetch the full thread tree (GET /threads/{id}/messages
 *      returns ALL messages with their ``parent_id``; assistant-ui
 *      reconstructs the tree client-side and renders branch switchers
 *      automatically wherever ``parent_id`` collisions are detected).
 *   2. ``append`` — for every NEW message assistant-ui adds (normal
 *      send, edit, regenerate, branch switch), POST it to the backend
 *      with the ``parent_id`` assistant-ui computed locally. Edits
 *      land as siblings under the original message's parent, so the
 *      backend persists the branch structure 1:1 with the UI state.
 *
 * The streaming router (POST /threads/{id}/messages) reads the body's
 * last user UIMessage id; if it finds a row already persisted (via
 * the ``append`` above) it skips auto-persist and uses the existing
 * message as the assistant's parent.
 */
import { useAui } from "@assistant-ui/react";
import type {
  ThreadHistoryAdapter,
  GenericThreadHistoryAdapter,
  MessageFormatAdapter,
  MessageFormatRepository,
  MessageFormatItem,
} from "@assistant-ui/react";
import { useState } from "react";

// UUID-shape gate for ``append``: the backend POST endpoint runs
// FastAPI body validation with ``id: UUID`` and ``parent_id: UUID |
// None``. assistant-ui-generated client ids (short random strings)
// would 422 the endpoint — skipping the POST for those ids lets the
// streaming router's auto-persist fallback handle them. Once the
// thread round-trips through the backend, all ids become UUIDs.
const _UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
function _isUuid(s: string | undefined | null): boolean {
  return typeof s === "string" && _UUID_RE.test(s);
}


type BackendMessageRow = {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  parent_id: string | null;
  parts: Array<Record<string, unknown>>;
};

class NotesAppThreadHistoryAdapter implements ThreadHistoryAdapter {
  constructor(
    private readonly apiUrl: string,
    private readonly headers: Record<string, string>,
    private readonly aui: ReturnType<typeof useAui>,
  ) {}

  private get _remoteId(): string | undefined {
    return this.aui.threadListItem().getState().remoteId;
  }

  withFormat<TMessage, TStorageFormat extends Record<string, unknown>>(
    _fmt: MessageFormatAdapter<TMessage, TStorageFormat>,
  ): GenericThreadHistoryAdapter<TMessage> {
    const adapter = this;
    return {
      async load(): Promise<MessageFormatRepository<TMessage>> {
        const remoteId = adapter._remoteId;
        if (!remoteId) return { messages: [] };
        const r = await fetch(
          `${adapter.apiUrl}/threads/${remoteId}/messages`,
          { headers: adapter.headers },
        );
        if (!r.ok) {
          throw new Error(
            `GET /threads/${remoteId}/messages failed: ${r.status}`,
          );
        }
        const rows = (await r.json()) as BackendMessageRow[];
        // Backend persists parts in Vercel UIMessage shape already
        // (assistant via VercelAIAdapter.dump_messages; user as
        // {type:"text", text:..., state:"done"}). Hand them through as
        // TMessage — useAISDKRuntime's format adapter will encode/decode
        // through its own internals when needed.
        return {
          messages: rows.map((row) => ({
            parentId: row.parent_id,
            message: {
              id: row.id,
              role: row.role,
              parts: row.parts,
            } as unknown as TMessage,
          })),
        };
      },

      async append(item: MessageFormatItem<TMessage>): Promise<void> {
        const remoteId = adapter._remoteId;
        if (!remoteId) return;
        // ``item.message`` is the assistant-ui ThreadMessage; the
        // format adapter knows its ``id`` and stored shape via the
        // shape we used in ``load`` ({id, role, parts}). Extract
        // them via the public accessor.
        const msg = item.message as unknown as {
          id: string;
          role: "user" | "assistant" | "system" | "tool";
          parts: Array<Record<string, unknown>>;
        };
        // ``id`` may not be a UUID for assistant-ui-generated client
        // ids (assistant-ui ships short random strings). The
        // backend's ``add_message_with_id`` requires UUID-shaped ids
        // (FastAPI body validation). Skip the POST in that case so
        // the streaming router's auto-persist fallback handles it
        // — branches won't render for those rows but normal sends
        // still work. (assistant-ui's stable ids ARE UUIDs after
        // first round-trip through the backend, so the affected
        // window is just the first send before reload.)
        if (!_isUuid(msg.id)) return;
        try {
          await fetch(
            `${adapter.apiUrl}/threads/${remoteId}/messages/append`,
            {
              method: "POST",
              headers: {
                ...adapter.headers,
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                id: msg.id,
                parent_id: item.parentId,
                role: msg.role,
                parts: msg.parts,
              }),
            },
          );
        } catch (err) {
          // Best-effort: if append fails the streaming router's
          // auto-persist path picks it up. Worst case: branches
          // won't appear for this turn on reload.
          // eslint-disable-next-line no-console
          console.warn(
            "[thread-history-adapter] append failed; falling back to "
            + "streaming-router auto-persist",
            err,
          );
        }
      },
    };
  }

  // ThreadHistoryAdapter's non-format methods are required by the type
  // even though useAISDKRuntime only goes through `withFormat`. Stub
  // them so other runtimes that hit the raw API don't break.
  async load() {
    return { headId: null, messages: [] };
  }
  async append() {
    /* no-op */
  }
}

/**
 * Hook that captures ``aui`` once and returns a stable adapter
 * instance — matches ``useAssistantCloudThreadHistoryAdapter``'s
 * shape so it can be dropped into ``RuntimeAdapterProvider``.
 */
export function useNotesAppThreadHistoryAdapter(
  apiUrl: string,
  headers: Record<string, string>,
): ThreadHistoryAdapter {
  const aui = useAui();
  const [adapter] = useState(
    () => new NotesAppThreadHistoryAdapter(apiUrl, headers, aui),
  );
  return adapter;
}

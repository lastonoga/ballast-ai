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
 *   2. ``append`` — for every NEW user message assistant-ui adds (normal
 *      send, edit, regenerate, branch switch), POST it to the backend
 *      with the ``parent_id`` assistant-ui computed locally. Edits land
 *      as siblings under the original message's parent, so the backend
 *      persists the branch structure 1:1 with the UI state. Assistant
 *      messages are persisted server-side at the end of each agent run
 *      (durable: ``StateflowDurableAgent._persist_assistant_turn``;
 *      non-durable: streaming-router ``on_complete``), so we don't
 *      double-append them from the client.
 *
 * The streaming router (POST /threads/{id}/runs) is a pure
 * run-trigger — it reads the active branch from the repo and assumes
 * the new user msg is already persisted (with auto-persist fallback
 * for direct-curl).
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


type BackendMessageRow = {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  parent_id: string | null;
  parts: Array<Record<string, unknown>>;
};

/**
 * Shared signal between the runtime-provider's ``initialize()`` path
 * and ``historyAdapter.load()`` — used to suppress the redundant
 * first ``load()`` that ``useExternalHistory`` fires the instant
 * ``initialize()`` flips ``remoteId`` from undefined → real-UUID
 * mid-session.
 *
 * Without this suppression we get a classic UI duplicate: useChat
 * has already added the user's "привет" to its local state
 * optimistically (the act that triggered ``initialize()`` in the
 * first place is ``sendMessages``), then the brand-new remoteId
 * unblocks ``useExternalHistory``, which calls ``load()`` →
 * backend returns the same row (just persisted via ``append`` or
 * the streaming-router auto-persist fallback) → ``runtime.thread.import``
 * adds it AGAIN on top of the optimistic state, so the message
 * renders twice. By marking the freshly-initialized remoteId here
 * and consuming the marker on first ``load()``, we skip exactly one
 * load — subsequent loads (thread switch, page reload) work as
 * normal because useChat for the active thread is empty then.
 */
export type JustInitializedSink = {
  /** Mark a remoteId as freshly produced by initialize() this session. */
  mark(remoteId: string): void;
  /** Consume + return whether ``load()`` should skip for this id. */
  consume(remoteId: string): boolean;
};

export function createJustInitializedSink(): JustInitializedSink {
  const ids = new Set<string>();
  return {
    mark(remoteId) {
      ids.add(remoteId);
    },
    consume(remoteId) {
      if (!ids.has(remoteId)) return false;
      ids.delete(remoteId);
      return true;
    },
  };
}

class NotesAppThreadHistoryAdapter implements ThreadHistoryAdapter {
  constructor(
    private readonly apiUrl: string,
    private readonly headers: Record<string, string>,
    private readonly aui: ReturnType<typeof useAui>,
    private readonly justInitialized: JustInitializedSink,
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
        // Skip exactly one load() right after initialize() — useChat
        // already holds the user's just-sent message optimistically;
        // re-hydrating from the backend would render it twice. See
        // ``JustInitializedSink`` docstring for the full rationale.
        if (adapter.justInitialized.consume(remoteId)) {
          return { messages: [] };
        }
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
        const msg = item.message as unknown as {
          id: string;
          role: "user" | "assistant" | "system" | "tool";
          parts: Array<Record<string, unknown>>;
        };
        // Assistant turns are persisted server-side at the end of each
        // agent run (durable: ``_persist_assistant_turn``; non-durable:
        // ``on_complete``). Re-POSTing them from the client would
        // double-create rows AND race the server-side persist on
        // parent_id — the streaming run's assistant becomes a child of
        // the resolved user msg; an assistant POST from append would
        // pick a different parent (assistant-ui's locally-computed one).
        // Filter to user messages only.
        if (msg.role !== "user") return;
        try {
          await fetch(
            `${adapter.apiUrl}/threads/${remoteId}/messages`,
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
          // auto-persist fallback picks it up. Worst case: branches
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
  justInitialized: JustInitializedSink,
): ThreadHistoryAdapter {
  const aui = useAui();
  const [adapter] = useState(
    () => new NotesAppThreadHistoryAdapter(apiUrl, headers, aui, justInitialized),
  );
  return adapter;
}

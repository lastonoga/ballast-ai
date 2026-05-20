"use client";

/**
 * `delete_note` approval card.
 *
 * Backend declares `@agent.tool(requires_approval=True)` on `delete_note`,
 * so pydantic-ai pauses the run and the Vercel adapter streams a
 * `tool-approval-request` chunk. assistant-ui's tool-call message-part
 * representation tags the part with `interrupt.payload = part.approval`
 * (see `@assistant-ui/react-ai-sdk/.../convertMessage.js:getToolInterrupt`)
 * and the part's `status.type` becomes `"requires-action"` /
 * `reason: "interrupt"`.
 *
 * Approval reply path: the assistant-ui Vercel runtime does NOT proxy
 * `addToolApprovalResponse` through the per-tool `addResult` callback —
 * it only wires `addResult` to `addToolOutput`, which would be wrong here
 * (would race with the resumed agent tool body and pollute the message
 * history). We therefore reach into `useChatApprovalHelpers()` (published
 * by `<RuntimeProvider />`) and call `addToolApprovalResponse({id,
 * approved})` directly on the underlying `useChat` instance.
 *
 * Auto-resend: the `useChat` instance is configured with
 * `sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses`,
 * so as soon as the user picks Approve or Cancel the conversation is
 * re-POSTed to `/threads/{id}/runs`. The backend pulls the approval
 * out of the incoming `messages` via
 * `VercelAIAdapter.deferred_tool_results`, threads it into the resumed
 * agent run, and we get either a real `delete_note` execution
 * ("deleted {id}") or a `ToolDenied` round-trip that the model
 * acknowledges in plain text.
 */

import { makeAssistantToolUI } from "@assistant-ui/react";
import { Button } from "@/components/ui/button";
import { useChatApprovalHelpers } from "@/components/runtime-provider";

type DeleteNoteArgs = { note_id: string };

/**
 * `interrupt.payload` shape for an AI SDK v6 server-side approval.
 *
 * Source of truth: `pydantic_ai.ui.vercel_ai.request_types.
 * ToolApprovalRequested` (`{type: "approval-requested", id: str}`).
 * The `id` here is the *approval id* the AI SDK wants in
 * `addToolApprovalResponse` — the pydantic adapter sets it to the
 * `tool_call_id` (deterministic), but we never assume that on the
 * frontend; we always pass it back as-is.
 */
type ApprovalInterruptPayload = {
  type: "approval-requested";
  id: string;
};

export const DeleteNoteApproval = makeAssistantToolUI<
  DeleteNoteArgs,
  string
>({
  toolName: "delete_note",
  render: function DeleteNoteApprovalRender({ args, status, interrupt }) {
    const { addToolApprovalResponse } = useChatApprovalHelpers();

    // Only render the actionable card while the tool call is paused
    // pending human approval. Once approved/denied, the resumed run
    // either lands a `tool-output-available` (status `complete`) or
    // `tool-output-denied` chunk that assistant-ui surfaces in its
    // default tool-call ribbon — no card needed.
    if (status.type !== "requires-action" || status.reason !== "interrupt") {
      return null;
    }

    const payload = interrupt?.payload as ApprovalInterruptPayload | undefined;
    const approvalId = payload?.id;
    if (!approvalId) {
      // Shouldn't happen on a server-side approval; bail gracefully so the
      // assistant doesn't get stuck waiting on a non-existent reply.
      console.warn(
        "[delete-note-approval] no approval id on interrupt payload",
        interrupt,
      );
      return null;
    }

    const handle = (approved: boolean) => {
      void addToolApprovalResponse({ id: approvalId, approved });
    };

    return (
      <div className="my-2 rounded-md border border-destructive/40 bg-destructive/5 p-4">
        <p className="text-sm font-medium">
          Delete this note? This cannot be undone.
        </p>
        <p className="mt-1 font-mono text-xs text-muted-foreground">
          id: {args.note_id}
        </p>
        <div className="mt-3 flex gap-2">
          <Button
            size="sm"
            variant="destructive"
            onClick={() => handle(true)}
          >
            Approve
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => handle(false)}
          >
            Cancel
          </Button>
        </div>
      </div>
    );
  },
});

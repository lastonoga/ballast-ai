"use client";

/**
 * Custom assistant-ui data-part renderers for HITL approval chat markers.
 *
 * Backend emits these from the approval card infrastructure via
 * ``UICardChannel``. Each part lands as::
 *
 *   { "type": "data-approval-pending",  "data": { card_id, kind }, "state": "done" }
 *   { "type": "data-approval-resolved", "data": { card_id, approved }, "state": "done" }
 *
 * ``makeAssistantDataUI({name})`` matches the ``data-`` prefix-stripped name
 * and renders the registered component. Mount ``ApprovalEventsUI`` once
 * anywhere inside ``<RuntimeProvider />``.
 */

import { makeAssistantDataUI } from "@assistant-ui/react";

// ── Payload shapes (must mirror the Python models server-side) ──────────

type ApprovalPendingData = {
  card_id: string;
  kind: string;
};

type ApprovalResolvedData = {
  card_id: string;
  approved: boolean;
};

// ── Renderers ───────────────────────────────────────────────────────────

export const ApprovalPendingUI = makeAssistantDataUI<ApprovalPendingData>({
  name: "approval-pending",
  render: (props) => {
    const data = props.data as ApprovalPendingData;
    return (
      <a
        href={`#approval-${data.card_id}`}
        className="text-sm text-blue-600 underline"
      >
        Waiting for your approval →
      </a>
    );
  },
});

export const ApprovalResolvedUI = makeAssistantDataUI<ApprovalResolvedData>({
  name: "approval-resolved",
  render: (props) => {
    const data = props.data as ApprovalResolvedData;
    return (
      <span
        className={`text-sm ${
          data.approved ? "text-emerald-600" : "text-gray-500"
        }`}
      >
        {data.approved ? "Approved ✓" : "Rejected ✗"}
      </span>
    );
  },
});

/**
 * Bundle of all approval chat-marker renderers. Mount once anywhere
 * inside ``<RuntimeProvider />``; each ``makeAssistantDataUI`` call
 * self-registers via a side-effect hook on mount.
 */
export function ApprovalEventsUI() {
  return (
    <>
      <ApprovalPendingUI />
      <ApprovalResolvedUI />
    </>
  );
}

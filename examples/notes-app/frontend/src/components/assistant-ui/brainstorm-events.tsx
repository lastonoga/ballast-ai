"use client";

/**
 * Custom assistant-ui data-part renderers for typed brainstorm-workflow
 * events.
 *
 * Backend emits these from
 * ``notes_app/workflows/brainstorm_events.py`` via the auto-connected
 * ``default_chat_router`` on ``brainstorm_progress``. Each part lands as::
 *
 *   { "type": "data-brainstorm-<event>",
 *     "data": { ...event.model_dump() },
 *     "state": "done" }
 *
 * ``makeAssistantDataUI({name})`` matches the ``data-`` prefix-stripped
 * name and renders the registered component instead of dropping the
 * part silently. Mount each of these once anywhere inside the
 * ``<RuntimeProvider />`` tree (we do it in ``app/page.tsx``).
 *
 * Styling: tailwind + lucide icons; no card framework dep. Visuals
 * mirror the cancel/approve card density from
 * ``delete-note-approval.tsx`` so the brainstorm narration feels
 * native to the chat.
 */

import { makeAssistantDataUI } from "@assistant-ui/react";
import {
  CheckCircle2,
  Clock,
  Sparkles,
  XCircle,
} from "lucide-react";

// ── Payload shapes (must mirror the Pydantic models server-side) ───────

type BrainstormChoseData = {
  type: "brainstorm-chose";
  title: string;
};

type BrainstormSavedData = {
  type: "brainstorm-saved";
  title: string;
  modified: boolean;
};

type BrainstormCancelledData = {
  type: "brainstorm-cancelled";
  reason: string | null;
};

type BrainstormTimedOutData = {
  type: "brainstorm-timed-out";
};

// ── Renderers ──────────────────────────────────────────────────────────

export const BrainstormChoseUI = makeAssistantDataUI<BrainstormChoseData>({
  name: "brainstorm-chose",
  render: (props) => {
    const data = props.data as BrainstormChoseData;
    return (
      <div className="my-2 flex items-start gap-3 rounded-md border border-indigo-200 bg-indigo-50 px-3 py-2.5 dark:border-indigo-900/60 dark:bg-indigo-950/30">
        <Sparkles className="mt-0.5 size-4 shrink-0 text-indigo-600 dark:text-indigo-400" />
        <div className="flex-1 text-sm">
          <div className="font-medium text-indigo-900 dark:text-indigo-200">
            Picked an idea
          </div>
          <div className="mt-0.5 text-indigo-800 dark:text-indigo-300">
            “{data.title}”
          </div>
          <div className="mt-1 text-xs text-indigo-700/80 dark:text-indigo-400/80">
            Opening approval thread…
          </div>
        </div>
      </div>
    );
  },
});

export const BrainstormSavedUI = makeAssistantDataUI<BrainstormSavedData>({
  name: "brainstorm-saved",
  render: (props) => {
    const data = props.data as BrainstormSavedData;
    return (
      <div className="my-2 flex items-start gap-3 rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2.5 dark:border-emerald-900/60 dark:bg-emerald-950/30">
        <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
        <div className="flex-1 text-sm">
          <div className="font-medium text-emerald-900 dark:text-emerald-200">
            Saved
            {data.modified ? (
              <span className="ml-1.5 rounded-full bg-emerald-100 px-1.5 py-0.5 text-[10px] font-normal uppercase tracking-wide text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-300">
                edited
              </span>
            ) : null}
          </div>
          <div className="mt-0.5 text-emerald-800 dark:text-emerald-300">
            “{data.title}”
          </div>
        </div>
      </div>
    );
  },
});

export const BrainstormCancelledUI =
  makeAssistantDataUI<BrainstormCancelledData>({
    name: "brainstorm-cancelled",
    render: (props) => {
      const data = props.data as BrainstormCancelledData;
      return (
        <div className="my-2 flex items-start gap-3 rounded-md border border-rose-200 bg-rose-50 px-3 py-2.5 dark:border-rose-900/60 dark:bg-rose-950/30">
          <XCircle className="mt-0.5 size-4 shrink-0 text-rose-600 dark:text-rose-400" />
          <div className="flex-1 text-sm">
            <div className="font-medium text-rose-900 dark:text-rose-200">
              Cancelled
            </div>
            {data.reason?.trim() ? (
              <div className="mt-0.5 text-rose-800 dark:text-rose-300">
                {data.reason.trim()}
              </div>
            ) : (
              <div className="mt-0.5 text-xs text-rose-700/80 dark:text-rose-400/80">
                No note was saved.
              </div>
            )}
          </div>
        </div>
      );
    },
  });

export const BrainstormTimedOutUI =
  makeAssistantDataUI<BrainstormTimedOutData>({
    name: "brainstorm-timed-out",
    render: () => (
      <div className="my-2 flex items-start gap-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2.5 dark:border-amber-900/60 dark:bg-amber-950/30">
        <Clock className="mt-0.5 size-4 shrink-0 text-amber-600 dark:text-amber-400" />
        <div className="flex-1 text-sm">
          <div className="font-medium text-amber-900 dark:text-amber-200">
            Timed out
          </div>
          <div className="mt-0.5 text-xs text-amber-700/80 dark:text-amber-400/80">
            Approval window expired — nothing was saved.
          </div>
        </div>
      </div>
    ),
  });

/**
 * Bundle of all brainstorm-event renderers. Mount once anywhere
 * inside ``<RuntimeProvider />``; each ``makeAssistantDataUI`` call
 * self-registers via a side-effect hook on mount.
 */
export function BrainstormEventsUI() {
  return (
    <>
      <BrainstormChoseUI />
      <BrainstormSavedUI />
      <BrainstormCancelledUI />
      <BrainstormTimedOutUI />
    </>
  );
}

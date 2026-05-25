"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";

import type { ApprovalCard, Decision } from "../use-approvals";

export function NoteCreateCard({
  card,
  onApprove,
  onReject,
}: {
  card: ApprovalCard;
  onApprove: (decision: Pick<Decision, "modified">) => void;
  onReject: () => void;
}) {
  const initial = card.payload as { title: string; body: string };
  const [title, setTitle] = useState(initial.title);
  const [body, setBody] = useState(initial.body);

  const trimmedTitle = title.trim();
  const isDirty = title !== initial.title || body !== initial.body;
  const canApprove = trimmedTitle.length > 0;

  const submit = () => {
    onApprove({
      modified: isDirty
        ? { title: trimmedTitle, body: body.trim() }
        : null,
    });
  };

  return (
    <div className="rounded border bg-white p-3 my-2 space-y-2">
      <div className="text-xs text-gray-500">Save note?</div>
      <input
        type="text"
        className="w-full rounded border px-2 py-1 text-sm font-semibold"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Title"
      />
      <textarea
        className="w-full rounded border px-2 py-1 text-sm min-h-[80px]"
        value={body}
        onChange={(e) => setBody(e.target.value)}
        placeholder="Body"
      />
      <div className="flex items-center justify-between gap-2 pt-1">
        <span className="text-xs text-gray-400">
          {isDirty ? "with your edits" : ""}
        </span>
        <div className="flex gap-2">
          <Button
            type="button"
            variant="default"
            size="sm"
            disabled={!canApprove}
            onClick={submit}
          >
            Approve
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onReject}
          >
            Reject
          </Button>
        </div>
      </div>
    </div>
  );
}

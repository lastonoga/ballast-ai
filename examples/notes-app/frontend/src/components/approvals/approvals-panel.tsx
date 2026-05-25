"use client";

import { useApprovals } from "./use-approvals";
import { NoteCreateCard } from "./card-renderers/note-create";

const RENDERERS: Record<
  string,
  React.ComponentType<{
    card: import("./use-approvals").ApprovalCard;
    onApprove: () => void;
    onReject: () => void;
  }>
> = {
  "note.create": NoteCreateCard,
};

export function ApprovalsPanel({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { pending, decide } = useApprovals();

  return (
    <aside
      className={`fixed top-0 right-0 h-full w-96 bg-gray-50 border-l shadow-lg transform transition-transform ${
        open ? "translate-x-0" : "translate-x-full"
      }`}
    >
      <header className="p-3 flex items-center justify-between border-b">
        <div className="font-semibold">Approvals</div>
        <button onClick={onClose} className="text-sm">Close</button>
      </header>
      <div className="p-3 overflow-y-auto h-[calc(100%-3rem)]">
        {pending.length === 0 && (
          <div className="text-sm text-gray-500">Nothing pending.</div>
        )}
        {pending.map((card) => {
          const Renderer = RENDERERS[card.kind];
          if (!Renderer) {
            return (
              <div key={card.id} className="text-sm text-red-600">
                No renderer for kind {card.kind}.
              </div>
            );
          }
          return (
            <Renderer
              key={card.id}
              card={card}
              onApprove={() => decide(card.id, { decision: "approve" })}
              onReject={() => decide(card.id, { decision: "reject" })}
            />
          );
        })}
      </div>
    </aside>
  );
}

import { InboxIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export function ApprovalsBadge({
  count,
  onClick,
}: {
  count: number;
  onClick: () => void;
}) {
  const label =
    count === 0
      ? "Approvals — nothing pending"
      : `Approvals — ${count} pending`;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          aria-label={label}
          onClick={onClick}
          className="relative"
        >
          <InboxIcon className="size-4" />
          {count > 0 && (
            <span
              className="absolute -top-0.5 -right-0.5 min-w-4 h-4 px-1 rounded-full
                         bg-red-500 text-white text-[10px] leading-4 text-center
                         font-medium"
            >
              {count}
            </span>
          )}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}

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

export function ApprovalsBadge({
  count,
  onClick,
}: {
  count: number;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="relative px-3 py-1 rounded bg-gray-100"
    >
      Approvals
      {count > 0 && (
        <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full px-1.5">
          {count}
        </span>
      )}
    </button>
  );
}

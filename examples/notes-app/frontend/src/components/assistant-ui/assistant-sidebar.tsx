import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import type { FC, PropsWithChildren } from "react";

import { Thread } from "@/components/assistant-ui/thread";

export const AssistantSidebar: FC<PropsWithChildren> = ({ children }) => {
  return (
    // Sidebar (threads + header) is intentionally narrow (~1/6 of the
    // viewport) so the chat dominates. User can still drag the handle to
    // resize within [12%, 40%]; the chosen size persists per-tab via
    // react-resizable-panels' built-in storage.
    <ResizablePanelGroup orientation="horizontal" autoSaveId="notes-app-sidebar">
      <ResizablePanel defaultSize={17} minSize={12} maxSize={40}>
        {children}
      </ResizablePanel>
      <ResizableHandle />
      <ResizablePanel defaultSize={83} minSize={60}>
        <Thread />
      </ResizablePanel>
    </ResizablePanelGroup>
  );
};

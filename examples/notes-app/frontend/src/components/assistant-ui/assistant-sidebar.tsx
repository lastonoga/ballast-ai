import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import type { FC, PropsWithChildren } from "react";

import { Thread } from "@/components/assistant-ui/thread";

export const AssistantSidebar: FC<PropsWithChildren> = ({ children }) => {
  return (
    // Sidebar starts at 1/4 of the viewport, chat at 3/4. No min/max —
    // user is free to drag the handle anywhere (including 0% to hide
    // the sidebar entirely).
    <ResizablePanelGroup orientation="horizontal">
      <ResizablePanel defaultSize={25}>{children}</ResizablePanel>
      <ResizableHandle />
      <ResizablePanel defaultSize={75}>
        <Thread />
      </ResizablePanel>
    </ResizablePanelGroup>
  );
};

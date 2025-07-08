import { useState, useRef, useEffect } from "react";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { Search } from "lucide-react";

interface SelectableMessageContentProps {
  children: React.ReactNode;
  onExplore: (selectedText: string, selectionInfo?: { startOffset: number; endOffset: number }) => void;
}

export function SelectableMessageContent({ children, onExplore }: SelectableMessageContentProps) {
  const [selectedText, setSelectedText] = useState("");
  const [contextMenuPosition, setContextMenuPosition] = useState({ x: 0, y: 0 });
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleSelectionChange = () => {
      const selection = window.getSelection();
      if (selection && selection.toString().trim()) {
        setSelectedText(selection.toString());
      }
    };

    document.addEventListener("selectionchange", handleSelectionChange);
    return () => {
      document.removeEventListener("selectionchange", handleSelectionChange);
    };
  }, []);

  const handleContextMenu = (e: React.MouseEvent) => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      setSelectedText(selection.toString());
      setContextMenuPosition({ x: e.clientX, y: e.clientY });
    }
  };

  const handleExplore = () => {
    if (selectedText) {
      onExplore(selectedText);
      // Clear selection after triggering explore
      window.getSelection()?.removeAllRanges();
    }
  };

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <div 
          ref={contentRef}
          onContextMenu={handleContextMenu}
          className="selectable-content"
        >
          {children}
        </div>
      </ContextMenuTrigger>
      {selectedText && (
        <ContextMenuContent>
          <ContextMenuItem onClick={handleExplore}>
            <Search className="mr-2 h-4 w-4" />
            Explore Further
          </ContextMenuItem>
        </ContextMenuContent>
      )}
    </ContextMenu>
  );
}
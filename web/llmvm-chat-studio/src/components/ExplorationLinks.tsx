import { Button } from "@/components/ui/button";
import { ExternalLink } from "lucide-react";

interface ExplorationLink {
  selectedText: string;
  explorationThreadId: string;
}

interface ExplorationLinksProps {
  explorations?: ExplorationLink[];
  onOpenExploration: (threadId: string) => void;
}

export function ExplorationLinks({ explorations, onOpenExploration }: ExplorationLinksProps) {
  if (!explorations || explorations.length === 0) return null;

  return (
    <div className="mt-3 space-y-1">
      <div className="text-xs text-gray-500 font-medium">Explorations:</div>
      {explorations.map((exploration, index) => (
        <div key={exploration.explorationThreadId} className="flex items-start gap-2">
          <Button
            variant="link"
            size="sm"
            className="h-auto p-0 text-blue-600 hover:text-blue-800 text-xs font-normal justify-start"
            onClick={() => onOpenExploration(exploration.explorationThreadId)}
          >
            <ExternalLink className="mr-1 h-3 w-3" />
            <span className="truncate max-w-[300px]" title={exploration.selectedText}>
              "{exploration.selectedText}"
            </span>
          </Button>
        </div>
      ))}
    </div>
  );
}
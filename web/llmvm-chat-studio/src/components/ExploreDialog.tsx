import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface ExploreDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selectedText: string;
  onExplore: (prompt: string) => void;
}

export function ExploreDialog({ open, onOpenChange, selectedText, onExplore }: ExploreDialogProps) {
  const [prompt, setPrompt] = useState("");

  const handleExplore = () => {
    if (prompt.trim()) {
      onExplore(prompt);
      setPrompt("");
      onOpenChange(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleExplore();
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Explore Further</DialogTitle>
          <DialogDescription>
            Add context or instructions for exploring the selected text
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4 py-4">
          <div className="bg-gray-50 p-3 rounded-md">
            <p className="text-sm text-gray-600 mb-1">Selected text:</p>
            <p className="text-sm text-gray-900 line-clamp-3">{selectedText}</p>
          </div>
          
          <Textarea
            placeholder="What would you like to explore about this text?"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyPress={handleKeyPress}
            className="min-h-[100px]"
            autoFocus
          />
        </div>
        
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleExplore} disabled={!prompt.trim()}>
            Explore
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
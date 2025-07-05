
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Plus, Trash2 } from "lucide-react";
import { Thread } from "./ChatInterface";

interface ThreadSidebarProps {
  threads: Thread[];
  programs: Thread[];
  activeThreadId: string;
  onThreadSelect: (id: string) => void;
  onThreadDoubleClick?: (id: string) => void;
  onNewThread: () => void;
  onDeleteThread?: (id: string) => void;
}

const ThreadSidebar = ({
  threads,
  programs,
  activeThreadId,
  onThreadSelect,
  onThreadDoubleClick,
  onNewThread,
  onDeleteThread
}: ThreadSidebarProps) => {
  const [clickTimeout, setClickTimeout] = useState<NodeJS.Timeout | null>(null);

  const handleThreadClick = (threadId: string) => {
    if (clickTimeout) {
      // Double click detected
      clearTimeout(clickTimeout);
      setClickTimeout(null);
      onThreadDoubleClick?.(threadId);
    } else {
      // Single click - wait to see if it's a double click
      const timeout = setTimeout(() => {
        onThreadSelect(threadId);
        setClickTimeout(null);
      }, 140);
      setClickTimeout(timeout);
    }
  };

  const formatDate = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return "Just now";
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  return (
    <div className="w-full h-full bg-gray-50 border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">LLMVM</h2>
          <Button
            onClick={onNewThread}
            size="sm"
            className="bg-blue-600 hover:bg-blue-700"
          >
            <Plus size={16} />
          </Button>
        </div>
        <p className="text-sm text-gray-600">
          AI Assistant with Tools & Code Execution
        </p>
      </div>

      {/* Threads and Programs in separate scrollable sections */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Threads Section */}
        <div className="h-1/2 flex flex-col border-b border-gray-200 min-h-0">
          <h3 className="text-xs font-semibold text-gray-900 uppercase px-5 py-2 bg-gray-50 flex-shrink-0">Threads</h3>
          <ScrollArea className="flex-1 min-h-0">
            <div className="p-2 space-y-1">
              {threads.map((thread) => (
                <div
                  key={thread.id}
                  className={`p-3 rounded-lg cursor-pointer transition-all group ${
                    activeThreadId === thread.id
                      ? "bg-blue-100 text-gray-900 border border-blue-200"
                      : "bg-white hover:bg-gray-50 text-gray-700 border border-gray-100"
                  }`}
                  onClick={() => handleThreadClick(thread.id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium truncate text-sm">
                        {thread.title}
                      </h3>
                      <p className={`text-xs mt-1 ${
                        activeThreadId === thread.id ? "text-gray-600" : "text-gray-500"
                      }`}>
                        ID: {thread.llmvmThreadId || thread.id} • {thread.messages.length} messages • {formatDate(thread.lastActivity)}
                      </p>
                      <div className="flex items-center gap-2 mt-2">
                        <span className={`text-xs px-2 py-0.5 rounded ${
                          activeThreadId === thread.id 
                            ? "bg-blue-200 text-blue-800" 
                            : "bg-gray-200 text-gray-700"
                        }`}>
                          {thread.model}
                        </span>
                        <span className={`text-xs px-2 py-0.5 rounded ${
                          thread.mode === "tools" ? "bg-green-100 text-green-800" :
                          thread.mode === "code" ? "bg-purple-100 text-purple-800" : "bg-orange-100 text-orange-800"
                        }`}>
                          {thread.mode}
                        </span>
                      </div>
                    </div>
                    {onDeleteThread && (
                      <Button
                        variant="ghost"
                        size="icon"
                        className="opacity-0 group-hover:opacity-100 transition-opacity ml-2 h-8 w-8"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteThread(thread.id);
                        }}
                      >
                        <Trash2 size={14} className="text-gray-500 hover:text-red-600" />
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </div>

        {/* Programs Section */}
        <div className="h-1/2 flex flex-col min-h-0">
          <h3 className="text-xs font-semibold text-gray-900 uppercase px-5 py-2 bg-gray-50 flex-shrink-0">Programs</h3>
          <ScrollArea className="flex-1 min-h-0">
            <div className="p-2 space-y-1">
              {programs.length > 0 ? (
                programs.map((program) => (
                  <div
                    key={program.id}
                    className={`p-3 rounded-lg cursor-pointer transition-all group ${
                      activeThreadId === program.id
                        ? "bg-purple-100 text-gray-900 border border-purple-200"
                        : "bg-white hover:bg-gray-50 text-gray-700 border border-gray-100"
                    }`}
                    onClick={() => handleThreadClick(program.id)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <h3 className="font-medium truncate text-sm">
                          {program.title}
                        </h3>
                        <p className={`text-xs mt-1 ${
                          activeThreadId === program.id ? "text-gray-600" : "text-gray-500"
                        }`}>
                          ID: {program.llmvmThreadId || program.id} • {formatDate(program.lastActivity)}
                        </p>
                        <div className="flex items-center gap-2 mt-2">
                          <span className={`text-xs px-2 py-0.5 rounded ${
                            activeThreadId === program.id 
                              ? "bg-purple-200 text-purple-800" 
                              : "bg-purple-100 text-purple-700"
                          }`}>
                            Program
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-xs text-gray-500 px-3 py-2 text-center">No programs yet</p>
              )}
            </div>
          </ScrollArea>
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200">
        <p className="text-xs text-gray-500 text-center">
          {threads.length} thread{threads.length !== 1 ? 's' : ''}
          {programs.length > 0 && ` • ${programs.length} program${programs.length !== 1 ? 's' : ''}`}
          {' • '}LLMVM Connected
        </p>
      </div>
    </div>
  );
};

export default ThreadSidebar;

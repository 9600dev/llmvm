import React from 'react';
import { X } from 'lucide-react';
import { Button } from '@/components/ui/button';

export interface Tab {
  id: string;
  threadId: string;
  title: string;
}

interface TabManagerProps {
  tabs: Tab[];
  activeTabId: string;
  onTabSelect: (tabId: string) => void;
  onTabClose: (tabId: string) => void;
}

const TabManager: React.FC<TabManagerProps> = ({ tabs, activeTabId, onTabSelect, onTabClose }) => {
  const handleTabClose = (e: React.MouseEvent, tabId: string) => {
    e.stopPropagation();
    onTabClose(tabId);
  };

  return (
    <div className="flex items-center border-b border-gray-200 bg-gray-50 overflow-x-auto">
      <div className="flex items-center min-w-0">
        {tabs.map((tab) => (
          <div
            key={tab.id}
            className={`
              flex items-center gap-1 px-3 py-2 border-r border-gray-200 cursor-pointer
              min-w-[120px] max-w-[200px] group relative
              ${activeTabId === tab.id 
                ? 'bg-white border-b-white' 
                : 'bg-gray-100 hover:bg-gray-200'
              }
            `}
            onClick={() => onTabSelect(tab.id)}
          >
            <span className="flex-1 truncate text-sm">
              {tab.title}
            </span>
            {tabs.length > 1 && (
              <Button
                variant="ghost"
                size="icon"
                className="h-4 w-4 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={(e) => handleTabClose(e, tab.id)}
              >
                <X className="h-3 w-3" />
              </Button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default TabManager;

import { useRef, useEffect, forwardRef, useImperativeHandle, useCallback } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Copy, User, Bot, CheckCircle, XCircle, Clock, GitBranch } from "lucide-react";
import { Message } from "./ChatInterface";
import { useToast } from "@/hooks/use-toast";
import { ContentRenderer } from "./ContentRenderer";
import { SelectableMessageContent } from "./SelectableMessageContent";
import { ExplorationLinks } from "./ExplorationLinks";

interface MessageDisplayProps {
  messages: Message[];
  onForkMessage?: (messageIndex: number) => void;
  onExploreText?: (selectedText: string, messageIndex: number) => void;
  onOpenExploration?: (threadId: string) => void;
}

export interface MessageDisplayHandle {
  scrollToBottom: () => void;
}

const MessageDisplay = forwardRef<MessageDisplayHandle, MessageDisplayProps>(({ messages, onForkMessage, onExploreText, onOpenExploration }, ref) => {
  const { toast } = useToast();
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const userScrolledRef = useRef(false);
  const lastScrollPositionRef = useRef(0);
  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isStreamingRef = useRef(false);

  // Scroll function with optional debouncing
  const performScroll = useCallback((behavior: ScrollBehavior = 'smooth', immediate: boolean = false) => {
    if (immediate) {
      // Immediate scroll for streaming
      bottomRef.current?.scrollIntoView({ behavior });
    } else {
      // Debounced scroll for other cases
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
      
      scrollTimeoutRef.current = setTimeout(() => {
        bottomRef.current?.scrollIntoView({ behavior });
      }, 50);
    }
  }, []);

  // Expose scrollToBottom method
  useImperativeHandle(ref, () => ({
    scrollToBottom: () => {
      userScrolledRef.current = false;
      performScroll('instant' as ScrollBehavior, true);
    }
  }));

  // Auto-scroll to bottom when new messages arrive or streaming content updates
  useEffect(() => {
    const scrollContainer = scrollAreaRef.current?.querySelector('[data-radix-scroll-area-viewport]');
    if (!scrollContainer) return;

    const checkAndScroll = () => {
      // Check if we're at the bottom (within 100px threshold)
      const isAtBottom = scrollContainer.scrollHeight - scrollContainer.scrollTop - scrollContainer.clientHeight < 100;
      
      // Only auto-scroll if:
      // 1. We're already at the bottom
      // 2. The last message is from the assistant and is being sent (streaming)
      // 3. User hasn't manually scrolled away
      const lastMessage = messages[messages.length - 1];
      const isCurrentlyStreaming = lastMessage?.role === 'assistant' && lastMessage?.status === 'sending';
      
      // Determine if we should auto-scroll
      const shouldAutoScroll = isAtBottom || (isCurrentlyStreaming && !userScrolledRef.current);

      if (shouldAutoScroll) {
        if (isCurrentlyStreaming) {
          // Use instant scrolling immediately during streaming
          performScroll('instant' as ScrollBehavior, true);
        } else {
          // Use smooth scrolling for non-streaming updates
          performScroll('smooth', false);
        }
      }
      
      // Update streaming state
      isStreamingRef.current = isCurrentlyStreaming;
    };

    // Initial check - instant scroll on first render
    if (messages.length > 0) {
      performScroll('instant' as ScrollBehavior, true);
    }

    // Set up MutationObserver for streaming content changes
    let observer: MutationObserver | null = null;
    const lastMessage = messages[messages.length - 1];
    const isStreaming = lastMessage?.role === 'assistant' && lastMessage?.status === 'sending';
    
    if (isStreaming) {
      observer = new MutationObserver(() => {
        checkAndScroll();
      });
      
      // Observe changes in the scroll container
      observer.observe(scrollContainer, {
        childList: true,
        subtree: true,
        characterData: true
      });
    }
    
    // Cleanup on unmount
    return () => {
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
      if (observer) {
        observer.disconnect();
      }
    };
  }, [messages, messages[messages.length - 1]?.content, messages[messages.length - 1]?.llmvmContent, performScroll]);

  // Track user scroll behavior
  useEffect(() => {
    const scrollContainer = scrollAreaRef.current?.querySelector('[data-radix-scroll-area-viewport]');
    if (!scrollContainer) return;

    const handleScroll = () => {
      const currentScrollPosition = scrollContainer.scrollTop;
      const isAtBottom = scrollContainer.scrollHeight - currentScrollPosition - scrollContainer.clientHeight < 100;
      
      // If user scrolled up manually, set the flag
      if (currentScrollPosition < lastScrollPositionRef.current && !isAtBottom) {
        userScrolledRef.current = true;
      } else if (isAtBottom) {
        // Reset flag when user scrolls back to bottom
        userScrolledRef.current = false;
      }
      
      lastScrollPositionRef.current = currentScrollPosition;
    };

    scrollContainer.addEventListener('scroll', handleScroll);
    return () => scrollContainer.removeEventListener('scroll', handleScroll);
  }, []);

  const copyToClipboard = async (message: Message) => {
    let textToCopy = '';
    
    // Extract text based on message structure
    if (message.llmvmContent) {
      // Check if llmvmContent has a getText method (LLMVM SDK objects)
      if (typeof message.llmvmContent.getText === 'function') {
        textToCopy = message.llmvmContent.getText();
      } else if (typeof message.llmvmContent === 'string') {
        textToCopy = message.llmvmContent;
      } else if (Array.isArray(message.llmvmContent)) {
        // Extract text from array of content items
        textToCopy = message.llmvmContent
          .map(item => {
            // Check for getText method first
            if (typeof item?.getText === 'function') return item.getText();
            if (typeof item === 'string') return item;
            if (item?.type === 'text' && item?.text) return item.text;
            if (item?.type === 'text' && typeof item?.content === 'string') return item.content;
            if (item?.type === 'text' && item?.sequence) return item.sequence;
            return '';
          })
          .filter(Boolean)
          .join('');
      } else if (message.llmvmContent?.text) {
        textToCopy = message.llmvmContent.text;
      } else if (message.llmvmContent?.content) {
        textToCopy = message.llmvmContent.content;
      } else if (message.llmvmContent?.sequence) {
        textToCopy = message.llmvmContent.sequence;
      }
    } 
    
    if (!textToCopy && typeof message.content === 'string') {
      textToCopy = message.content;
    } else if (!textToCopy && Array.isArray(message.content)) {
      // Handle array content (similar to llmvmContent)
      textToCopy = message.content
        .map(item => {
          if (typeof item?.getText === 'function') return item.getText();
          if (typeof item === 'string') return item;
          if (item?.type === 'text' && item?.text) return item.text;
          if (item?.type === 'text' && typeof item?.content === 'string') return item.content;
          if (item?.type === 'text' && item?.sequence) return item.sequence;
          return '';
        })
        .filter(Boolean)
        .join('');
    } else if (!textToCopy && message.content && typeof message.content.getText === 'function') {
      // Check if content itself has getText method
      textToCopy = message.content.getText();
    }
    
    if (!textToCopy) {
      toast({
        title: "Copy failed",
        description: "No text content found to copy.",
        variant: "destructive",
      });
      return;
    }
    
    try {
      // Try using the modern clipboard API first
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(textToCopy);
        toast({
          title: "Copied to clipboard",
          description: "Message content has been copied.",
        });
      } else {
        // Fallback for older browsers or non-secure contexts
        const textArea = document.createElement("textarea");
        textArea.value = textToCopy;
        textArea.style.position = "fixed";
        textArea.style.left = "-999999px";
        textArea.style.top = "-999999px";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
          document.execCommand('copy');
          toast({
            title: "Copied to clipboard",
            description: "Message content has been copied.",
          });
        } catch (err) {
          toast({
            title: "Copy failed",
            description: "Unable to copy to clipboard. Please try selecting and copying manually.",
            variant: "destructive",
          });
        } finally {
          textArea.remove();
        }
      }
    } catch (err) {
      console.error('Failed to copy:', err);
      toast({
        title: "Copy failed", 
        description: "Unable to copy to clipboard. Please try selecting and copying manually.",
        variant: "destructive",
      });
    }
  };

  const renderMessage = (message: Message, index: number) => {
    const isUser = message.role === "user";
    
    return (
      <div
        key={message.id}
        className={`flex gap-4 p-6 ${isUser ? "bg-gray-50" : "bg-white"}`}
      >
        {/* Avatar */}
        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser ? "bg-blue-600" : "bg-green-600"
        }`}>
          {isUser ? <User size={16} className="text-white" /> : <Bot size={16} className="text-white" />}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0 overflow-hidden">
          <div className="flex items-center gap-2 mb-2">
            <span className="font-medium text-gray-900">
              {isUser ? "You" : "LLMVM"}
            </span>
            <span className="text-xs text-gray-500">
              {message.timestamp.toLocaleTimeString()}
            </span>
            {message.status && (
              <div className="flex items-center">
                {message.status === "sending" && <Clock size={14} className="text-yellow-600" />}
                {message.status === "success" && <CheckCircle size={14} className="text-green-600" />}
                {message.status === "error" && <XCircle size={14} className="text-red-600" />}
              </div>
            )}
          </div>
          
          <div className="max-w-none overflow-x-auto">
            {onExploreText ? (
              <SelectableMessageContent onExplore={(text) => onExploreText(text, index)}>
                {(() => {
                  if (message.llmvmContent) {
                    return (
                      <ContentRenderer 
                        content={message.llmvmContent} 
                        type={message.type} 
                        isStreaming={message.status === 'sending'}
                        images={message.images}
                      />
                    );
                  } else if (message.type === "code") {
                    return (
                      <pre className="bg-gray-100 p-4 rounded-lg overflow-x-auto border max-w-full">
                        <code className="text-green-700 block whitespace-pre-wrap break-all">{message.content}</code>
                      </pre>
                    );
                  } else {
                    return (
                      <ContentRenderer 
                        content={message.content} 
                        type={message.type} 
                        isStreaming={message.status === 'sending'}
                        images={message.images}
                      />
                    );
                  }
                })()}
              </SelectableMessageContent>
            ) : (
              (() => {
                if (message.llmvmContent) {
                  return (
                    <ContentRenderer 
                      content={message.llmvmContent} 
                      type={message.type} 
                      isStreaming={message.status === 'sending'}
                      images={message.images}
                    />
                  );
                } else if (message.type === "code") {
                  return (
                    <pre className="bg-gray-100 p-4 rounded-lg overflow-x-auto border max-w-full">
                      <code className="text-green-700 block whitespace-pre-wrap break-all">{message.content}</code>
                    </pre>
                  );
                } else {
                  return (
                    <ContentRenderer 
                      content={message.content} 
                      type={message.type} 
                      isStreaming={message.status === 'sending'}
                      images={message.images}
                    />
                  );
                }
              })()
            )}
          </div>

          {/* Exploration links */}
          <ExplorationLinks 
            explorations={message.explorations}
            onOpenExploration={onOpenExploration || (() => {})}
          />

          <div className="flex items-center gap-2 mt-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => copyToClipboard(message)}
              className="text-gray-500 hover:text-gray-900 h-8 px-2"
              title="Copy message"
            >
              <Copy size={14} />
            </Button>
            {onForkMessage && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onForkMessage(index)}
                className="text-gray-500 hover:text-gray-900 h-8 px-2"
                title="Fork conversation from here"
              >
                <GitBranch size={14} />
              </Button>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <ScrollArea ref={scrollAreaRef} className="h-full">
      <div className="min-h-full">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <Bot size={48} className="mx-auto text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-600 mb-2">
                Start a conversation
              </h3>
              <p className="text-gray-500 max-w-md">
                Ask me anything! I can help with code, run tools, and assist with various tasks.
              </p>
            </div>
          </div>
        ) : (
          <div>
            {messages.map((message, index) => renderMessage(message, index))}
            {/* Invisible element at the bottom for scrolling */}
            <div ref={bottomRef} style={{ height: 1 }} />
          </div>
        )}
      </div>
    </ScrollArea>
  );
});

MessageDisplay.displayName = 'MessageDisplay';

export default MessageDisplay;

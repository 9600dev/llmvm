import { useState, useEffect, useCallback, useRef } from "react";
import ThreadSidebar from "./ThreadSidebar";
import MessageDisplay, { MessageDisplayHandle } from "./MessageDisplay";
import MessageInput from "./MessageInput";
import ThreadSettingsDialog, { ThreadSettings } from "./ThreadSettingsDialog";
import TabManager, { Tab } from "./TabManager";
import { Button } from "@/components/ui/button";
import { Menu, X } from "lucide-react";
import { getLLMVMService } from "@/services/llmvm";
import type { Thread as LLMVMThread, Message as LLMVMMessage } from "llmvm-sdk";

export interface Message {
  id: string;
  content: string | any[];
  role: "user" | "assistant";
  timestamp: Date;
  type?: "text" | "code" | "image" | "file" | "browser" | "pdf";
  status?: "sending" | "success" | "error";
  llmvmContent?: any;
  images?: Array<{id: string, data: string}>; // Separate storage for image data
}

export interface Thread {
  id: string;
  title: string;
  messages: Message[];
  lastActivity: Date;
  model: string;
  mode: "tools" | "direct" | "code" | "program";
  settings: ThreadSettings;
  llmvmThreadId?: string;
}

const defaultSettings: ThreadSettings = {
  executor: "anthropic",
  model: "claude-sonnet-4-20250514",
  temperature: 1.0,
  endpoint: "",
  compression: "auto",
  outputTokenLen: 8192,
  thinking: false,
  fullProcessing: true
};

const ChatInterface = () => {
  // Load sidebar state from localStorage or default to true
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    const saved = localStorage.getItem('sidebarOpen');
    return saved !== null ? JSON.parse(saved) : true;
  });
  const [llmvmService] = useState(() => getLLMVMService());
  const [isConnected, setIsConnected] = useState(false);
  const [editingTitle, setEditingTitle] = useState(false);
  const [editedTitle, setEditedTitle] = useState("");
  const [threads, setThreads] = useState<Thread[]>([
    {
      id: "1",
      title: "Welcome to LLMVM",
      messages: [
        {
          id: "1",
          content: "Welcome to LLMVM! I'm your AI assistant with access to tools and code execution. How can I help you today?",
          role: "assistant",
          timestamp: new Date(),
          type: "text",
          status: "success"
        }
      ],
      lastActivity: new Date(),
      model: "claude-sonnet-4-20250514",
      mode: "tools",
      settings: defaultSettings
    }
  ]);
  const [programs, setPrograms] = useState<Thread[]>([]);
  const [tabs, setTabs] = useState<Tab[]>([
    { id: "tab-1", threadId: "1", title: "Welcome to LLMVM" }
  ]);
  const [activeTabId, setActiveTabId] = useState("tab-1");
  const messageDisplayRef = useRef<MessageDisplayHandle>(null);

  const activeTab = tabs.find(t => t.id === activeTabId);
  const activeThreadId = activeTab?.threadId || "1";
  const activeThread = threads.find(t => t.id === activeThreadId) || programs.find(p => p.id === activeThreadId);

  // Scroll to bottom when active tab/thread changes
  useEffect(() => {
    if (activeThreadId) {
      // Use requestAnimationFrame to ensure DOM has updated
      requestAnimationFrame(() => {
        messageDisplayRef.current?.scrollToBottom();
      });
    }
  }, [activeTabId, activeThreadId]);

  // Save sidebar state to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('sidebarOpen', JSON.stringify(sidebarOpen));
  }, [sidebarOpen]);

  // Check LLMVM connection and load threads on mount
  useEffect(() => {
    const checkConnectionAndLoadThreads = async () => {
      try {
        const healthy = await llmvmService.checkHealth();
        setIsConnected(healthy);
        if (!healthy) {
          console.warn('LLMVM service is not reachable at localhost:8011');
        }

        // Load existing threads and programs from LLMVM if connected
        if (healthy) {
          try {
            // Load threads
            const llmvmThreads = await llmvmService.getAllThreads();
            if (llmvmThreads && llmvmThreads.length > 0) {
              // Filter out programs from threads
              const regularThreads = llmvmThreads.filter(t => t.current_mode !== 'program');
              // Sort threads by ID in reverse order (newest first)
              const sortedThreads = [...regularThreads].sort((a, b) => b.id - a.id);

              // Convert LLMVM threads to UI threads
              const uiThreads: Thread[] = sortedThreads.map(llmvmThread => {
                let title = `Thread ${llmvmThread.id}`;

                // Use explicit title if available
                if (llmvmThread.title) {
                  title = llmvmThread.title;
                } else {
                  // Otherwise, get the first user message for the title
                  const firstUserMessage = llmvmThread.messages.find(msg => msg.role === 'user');

                  if (firstUserMessage && firstUserMessage.content) {
                    // Extract text content from the message
                    let textContent = '';
                    if (typeof firstUserMessage.content === 'string') {
                      textContent = firstUserMessage.content;
                    } else if (Array.isArray(firstUserMessage.content)) {
                      // Find first text content in the array
                      const textItem = firstUserMessage.content.find((item: any) =>
                        typeof item === 'string' || item.content_type === 'text' || item.type === 'text'
                      );
                      if (textItem) {
                        textContent = typeof textItem === 'string' ? textItem : (textItem.text || textItem.sequence || '');
                      }
                    }

                    // Take first 30 characters and clean up
                    if (textContent) {
                      title = textContent.substring(0, 30).trim();
                      if (textContent.length > 30) {
                        title += '...';
                      }
                    }
                  }
                }

                return {
                  id: String(llmvmThread.id),
                  title,
                  messages: llmvmThread.messages.map((msg, index) => {
                    // Extract text content from the message
                    let content = '';
                    let llmvmContent = null;
                    
                    if (msg.content) {
                      if (typeof msg.content === 'string') {
                        content = msg.content;
                      } else if (Array.isArray(msg.content)) {
                        // Extract text from content array
                        const textParts = msg.content.map((item: any) => {
                          if (typeof item === 'string') {
                            return item;
                          } else if (item.sequence && (item.content_type === 'text' || item.type === 'text')) {
                            return item.sequence;
                          } else if (item.text) {
                            return item.text;
                          }
                          return '';
                        }).filter(Boolean);
                        
                        content = textParts.join('\n');
                        llmvmContent = msg.content; // Store the full content for proper rendering
                      }
                    }
                    
                    return {
                      id: `${llmvmThread.id}-${index}`,
                      content: content,
                      role: msg.role as "user" | "assistant",
                      timestamp: new Date(),
                      type: "text",
                      status: "success",
                      llmvmContent: llmvmContent // Store LLMVM content if available
                    };
                  }),
                lastActivity: new Date(),
                model: llmvmThread.model,
                mode: llmvmThread.current_mode as "tools" | "direct" | "code" | "program",
                settings: {
                  executor: llmvmThread.executor as "anthropic" | "openai" | "gemini",
                  model: llmvmThread.model,
                  temperature: llmvmThread.temperature,
                  endpoint: "",
                  compression: llmvmThread.compression,
                  outputTokenLen: llmvmThread.output_token_len,
                  thinking: llmvmThread.thinking > 0,
                  fullProcessing: true
                },
                llmvmThreadId: String(llmvmThread.id)
                };
              });

              setThreads(uiThreads);
              // Update the first tab with the newest thread if tabs only has the default
              if (uiThreads.length > 0 && tabs.length === 1 && tabs[0].threadId === "1") {
                setTabs([{ 
                  id: "tab-1", 
                  threadId: uiThreads[0].id, 
                  title: uiThreads[0].title 
                }]);
              }
            }

            // Load programs
            const llmvmPrograms = await llmvmService.getAllPrograms();
            if (llmvmPrograms && llmvmPrograms.length > 0) {
              // Sort programs by ID in reverse order (newest first)
              const sortedPrograms = [...llmvmPrograms].sort((a, b) => b.id - a.id);

              // Convert LLMVM programs to UI threads
              const uiPrograms: Thread[] = sortedPrograms.map(llmvmProgram => {
                const title = llmvmProgram.title || `Program ${llmvmProgram.id}`;

                return {
                  id: String(llmvmProgram.id),
                  title,
                  messages: llmvmProgram.messages.map((msg, index) => {
                    // Extract text content from the message
                    let content = '';
                    let llmvmContent = null;
                    
                    if (msg.content) {
                      if (typeof msg.content === 'string') {
                        content = msg.content;
                      } else if (Array.isArray(msg.content)) {
                        // Extract text from content array
                        const textParts = msg.content.map((item: any) => {
                          if (typeof item === 'string') {
                            return item;
                          } else if (item.sequence && (item.content_type === 'text' || item.type === 'text')) {
                            return item.sequence;
                          } else if (item.text) {
                            return item.text;
                          }
                          return '';
                        }).filter(Boolean);
                        
                        content = textParts.join('\n');
                        llmvmContent = msg.content; // Store the full content for proper rendering
                      }
                    }
                    
                    return {
                      id: `${llmvmProgram.id}-${index}`,
                      content: content,
                      role: msg.role as "user" | "assistant",
                      timestamp: new Date(),
                      type: "text",
                      status: "success",
                      llmvmContent: llmvmContent // Store LLMVM content if available
                    };
                  }),
                  lastActivity: new Date(),
                  model: llmvmProgram.model,
                  mode: "program" as any, // Programs are always in program mode
                  settings: {
                    executor: llmvmProgram.executor as "anthropic" | "openai" | "gemini",
                    model: llmvmProgram.model,
                    temperature: llmvmProgram.temperature,
                    endpoint: "",
                    compression: llmvmProgram.compression,
                    outputTokenLen: llmvmProgram.output_token_len,
                    thinking: llmvmProgram.thinking > 0,
                    fullProcessing: true
                  },
                  llmvmThreadId: String(llmvmProgram.id)
                };
              });

              setPrograms(uiPrograms);
            }
          } catch (error) {
            console.error('Failed to load threads/programs from LLMVM:', error);
          }
        }
      } catch (error) {
        console.error('Failed to connect to LLMVM:', error);
        setIsConnected(false);
      }
    };
    checkConnectionAndLoadThreads();
  }, [llmvmService]);

  const sendMessage = async (content: string, files?: File[]) => {
    if (!activeThread) return;

    // Check for /compile command
    if (content.trim().startsWith('/compile')) {
      // Extract optional compile prompt after /compile
      const compilePrompt = content.trim().substring('/compile'.length).trim();

      // Run compile command
      if (activeThread.llmvmThreadId) {
        await runCompileCommand(activeThread.llmvmThreadId, compilePrompt);
      } else {
        // Create a new thread first if needed
        const llmvmThread = await llmvmService.createNewThread(
          [],
          {
            model: activeThread.settings.model,
            temperature: activeThread.settings.temperature,
            output_token_len: activeThread.settings.outputTokenLen || 0,
            current_mode: activeThread.mode,
            thinking: activeThread.settings.thinking ? 1 : 0,
            executor: activeThread.settings.executor,
            compression: activeThread.settings.compression
          }
        );
        const llmvmThreadId = String(llmvmThread.id);

        // Update thread with LLMVM ID
        setThreads(prev => prev.map(thread =>
          thread.id === activeThreadId
            ? { ...thread, llmvmThreadId }
            : thread
        ));

        await runCompileCommand(llmvmThreadId, compilePrompt);
      }
      return;
    }

    // Process files and prepare content
    let messageContent: any = content;
    let messageType = "text";

    if (files && files.length > 0) {
      // Convert files to base64 for images
      const processedContent: any[] = [];

      // Add text content if present
      if (content.trim()) {
        processedContent.push({
          type: "text",
          content_type: "text",
          sequence: content,
          url: ""
        });
      }

      // Process each file
      for (const file of files) {
        // Convert file to base64
        const base64 = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => {
            const result = reader.result as string;
            // Remove data:xxx;base64, prefix
            const base64Data = result.split(',')[1];
            resolve(base64Data);
          };
          reader.readAsDataURL(file);
        });

        if (file.type.startsWith('image/')) {
          // Handle images with ImageContent
          processedContent.push({
            type: "image",
            content_type: "image",
            sequence: base64,
            image_type: file.type.split('/')[1],
            url: ""
          });
          messageType = "image";
        } else {
          // Handle all other files with FileContent
          processedContent.push({
            type: "FileContent",
            content_type: "file",
            sequence: base64,
            url: file.name // Use filename for display
          });
          messageType = "file";
        }
      }

      messageContent = processedContent.length > 1 ? processedContent : processedContent[0] || content;
    }

    // Create display text for UI
    let displayContent = content;
    if (!content && files && files.length > 0) {
      const imageCount = files.filter(f => f.type.startsWith('image/')).length;
      const fileCount = files.filter(f => !f.type.startsWith('image/')).length;
      const parts = [];
      if (imageCount > 0) parts.push(`${imageCount} image${imageCount > 1 ? 's' : ''}`);
      if (fileCount > 0) parts.push(`${fileCount} file${fileCount > 1 ? 's' : ''}`);
      displayContent = `[${parts.join(' and ')}]`;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      content: displayContent, // Display text in UI
      role: "user",
      timestamp: new Date(),
      type: messageType,
      status: "sending",
      llmvmContent: messageContent // Store the actual content to send
    };

    // Add user message and update thread title if needed
    setThreads(prev => prev.map(thread => {
      if (thread.id === activeThreadId) {
        const updatedThread = {
          ...thread,
          messages: [...thread.messages, userMessage],
          lastActivity: new Date()
        };

        // Update title if this is the first user message or if title is still default
        if (thread.title.startsWith('Thread ') || thread.title === 'New Conversation' || thread.title === 'Welcome to LLMVM') {
          let newTitle = content.substring(0, 30).trim();
          if (content.length > 30) {
            newTitle += '...';
          }
          updatedThread.title = newTitle;
        }

        return updatedThread;
      }
      return thread;
    }));

    // Create assistant message placeholder
    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      content: "",
      role: "assistant",
      timestamp: new Date(),
      type: "text",
      status: "sending"
    };

    setThreads(prev => prev.map(thread =>
      thread.id === activeThreadId
        ? { ...thread, messages: [...thread.messages, assistantMessage], lastActivity: new Date() }
        : thread
    ));

    try {
      // Ensure we have an LLMVM thread
      let llmvmThreadId = activeThread.llmvmThreadId;
      if (!llmvmThreadId) {
        const llmvmThread = await llmvmService.createNewThread(
          [],
          {
            model: activeThread.settings.model,
            temperature: activeThread.settings.temperature,
            output_token_len: activeThread.settings.outputTokenLen || 0,
            current_mode: activeThread.mode,
            thinking: activeThread.settings.thinking ? 1 : 0,
            executor: activeThread.settings.executor,
            compression: activeThread.settings.compression
          }
        );
        llmvmThreadId = String(llmvmThread.id);

        // Update thread with LLMVM ID
        setThreads(prev => prev.map(thread =>
          thread.id === activeThreadId
            ? { ...thread, llmvmThreadId }
            : thread
        ));
      }

      // Send message with streaming
      let streamedContent = "";
      let streamedImages: Array<{id: string, data: string}> = [];
      const response = await llmvmService.sendMessage(
        llmvmThreadId,
        userMessage.llmvmContent || content, // Use processed content if available
        {
          model: activeThread.settings.model,
          temperature: activeThread.settings.temperature,
          maxTokens: activeThread.settings.outputTokenLen || undefined,
          mode: activeThread.mode,
          thinking: activeThread.settings.thinking,
          executor: activeThread.settings.executor,
          compression: activeThread.settings.compression,
          onChunk: (chunk: any) => {
            // Handle different types of chunks
            if (typeof chunk === 'object' && chunk.type === 'image' && chunk.data) {
              // This is an image chunk
              const imageId = `img-${Date.now()}-${streamedImages.length}`;
              streamedImages.push({ id: imageId, data: chunk.data });
              streamedContent += `[IMAGE:${imageId}]\n`;
            } else if (typeof chunk === 'string') {
              streamedContent += chunk;
            } else {
              // For any other object type, try to extract meaningful content
              if (chunk && typeof chunk === 'object') {
                // Check if it has a text property or similar
                const text = chunk.text || chunk.content || chunk.message;
                if (text) {
                  streamedContent += text;
                } else {
                  // Last resort - stringify it but this shouldn't happen
                  console.warn('Unknown chunk type:', chunk);
                  streamedContent += JSON.stringify(chunk);
                }
              } else {
                streamedContent += String(chunk);
              }
            }

            // Update assistant message with streamed content and images
            setThreads(prev => prev.map(thread =>
              thread.id === activeThreadId
                ? {
                    ...thread,
                    messages: thread.messages.map(msg =>
                      msg.id === assistantMessageId
                        ? {
                            ...msg,
                            content: streamedContent,
                            status: "sending",
                            images: streamedImages // Store images separately
                          }
                        : msg.id === userMessage.id
                        ? { ...msg, status: "success" }
                        : msg
                    )
                  }
                : thread
            ));
          }
        }
      );

      // Update status to success but keep the streamed content
      setThreads(prev => prev.map(thread =>
        thread.id === activeThreadId
          ? {
              ...thread,
              messages: thread.messages.map(msg =>
                msg.id === assistantMessageId
                  ? {
                      ...msg,
                      // Keep the streamed content, just update the status
                      content: streamedContent,
                      status: "success",
                      images: streamedImages // Preserve the images
                    }
                  : msg
              )
            }
          : thread
      ));

    } catch (error) {
      console.error('Failed to send message:', error);
      // Update message status to error
      setThreads(prev => prev.map(thread =>
        thread.id === activeThreadId
          ? {
              ...thread,
              messages: thread.messages.map(msg =>
                msg.id === assistantMessageId
                  ? {
                      ...msg,
                      content: "Sorry, I encountered an error processing your message. Please make sure LLMVM is running on localhost:8011.",
                      status: "error"
                    }
                  : msg.id === userMessage.id
                  ? { ...msg, status: "error" }
                  : msg
              )
            }
          : thread
      ));
    }
  };

  const createNewThread = async () => {
    const newThread: Thread = {
      id: Date.now().toString(),
      title: "New Conversation",
      messages: [],
      lastActivity: new Date(),
      model: defaultSettings.model,
      mode: "tools",
      settings: defaultSettings
    };

    // Create LLMVM thread if connected
    if (isConnected) {
      try {
        const llmvmThread = await llmvmService.createNewThread(
          [],
          {
            model: defaultSettings.model,
            temperature: defaultSettings.temperature,
            current_mode: "tools",
            thinking: defaultSettings.thinking ? 1 : 0,
            executor: defaultSettings.executor,
            compression: defaultSettings.compression
          }
        );
        newThread.llmvmThreadId = String(llmvmThread.id);
      } catch (error) {
        console.error('Failed to create LLMVM thread:', error);
      }
    }

    // Add new thread at the beginning (newest first)
    setThreads(prev => [newThread, ...prev]);
    setActiveThreadId(newThread.id);
  };

  const updateThreadSettings = (settings: ThreadSettings) => {
    setThreads(prev => prev.map(thread =>
      thread.id === activeThreadId
        ? { ...thread, settings, model: settings.model }
        : thread
    ));
  };

  const runCompileCommand = async (llmvmThreadId: string, compilePrompt: string) => {
    // Add user message showing the compile command
    const userMessage: Message = {
      id: Date.now().toString(),
      content: `/compile ${compilePrompt}`,
      role: "user",
      timestamp: new Date(),
      type: "text",
      status: "sending"
    };

    setThreads(prev => prev.map(thread =>
      thread.id === activeThreadId
        ? { ...thread, messages: [...thread.messages, userMessage], lastActivity: new Date() }
        : thread
    ));

    // Create assistant message placeholder
    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      content: "Compiling thread into a standalone program...",
      role: "assistant",
      timestamp: new Date(),
      type: "text",
      status: "sending"
    };

    setThreads(prev => prev.map(thread =>
      thread.id === activeThreadId
        ? { ...thread, messages: [...thread.messages, assistantMessage], lastActivity: new Date() }
        : thread
    ));

    try {
      let streamedContent = "";
      const result = await llmvmService.compileThread(
        llmvmThreadId,
        compilePrompt,
        {
          onChunk: (chunk: any) => {
            if (typeof chunk === 'string') {
              streamedContent += chunk;
            } else {
              streamedContent += JSON.stringify(chunk);
            }

            // Update assistant message with streamed content
            setThreads(prev => prev.map(thread =>
              thread.id === activeThreadId
                ? {
                    ...thread,
                    messages: thread.messages.map(msg =>
                      msg.id === assistantMessageId
                        ? {
                            ...msg,
                            content: streamedContent,
                            status: "sending"
                          }
                        : msg.id === userMessage.id
                        ? { ...msg, status: "success" }
                        : msg
                    )
                  }
                : thread
            ));
          }
        }
      );

      // Update status to success
      setThreads(prev => prev.map(thread =>
        thread.id === activeThreadId
          ? {
              ...thread,
              messages: thread.messages.map(msg =>
                msg.id === assistantMessageId
                  ? {
                      ...msg,
                      content: streamedContent || "Thread compiled successfully!",
                      status: "success"
                    }
                  : msg
              )
            }
          : thread
      ));

    } catch (error) {
      console.error('Failed to compile thread:', error);
      // Update message status to error
      setThreads(prev => prev.map(thread =>
        thread.id === activeThreadId
          ? {
              ...thread,
              messages: thread.messages.map(msg =>
                msg.id === assistantMessageId
                  ? {
                      ...msg,
                      content: `Error compiling thread: ${error}`,
                      status: "error"
                    }
                  : msg.id === userMessage.id
                  ? { ...msg, status: "error" }
                  : msg
              )
            }
          : thread
      ));
    }
  };

  const executePython = async (code: string) => {
    if (!activeThread) return;

    // Add user message showing the Python code
    const userMessage: Message = {
      id: Date.now().toString(),
      content: `\`\`\`python\n${code}\n\`\`\``,
      role: "user",
      timestamp: new Date(),
      type: "code",
      status: "sending"
    };

    setThreads(prev => prev.map(thread =>
      thread.id === activeThreadId
        ? { ...thread, messages: [...thread.messages, userMessage], lastActivity: new Date() }
        : thread
    ));

    // Create result message placeholder
    const resultMessageId = (Date.now() + 1).toString();
    const resultMessage: Message = {
      id: resultMessageId,
      content: "",
      role: "assistant",
      timestamp: new Date(),
      type: "code",
      status: "sending"
    };

    setThreads(prev => prev.map(thread =>
      thread.id === activeThreadId
        ? { ...thread, messages: [...thread.messages, resultMessage], lastActivity: new Date() }
        : thread
    ));

    try {
      // Ensure we have an LLMVM thread
      let llmvmThreadId = activeThread.llmvmThreadId;
      if (!llmvmThreadId) {
        const llmvmThread = await llmvmService.createNewThread(
          [],
          {
            model: activeThread.settings.model,
            temperature: activeThread.settings.temperature,
            output_token_len: activeThread.settings.outputTokenLen || 0,
            current_mode: activeThread.mode,
            thinking: activeThread.settings.thinking ? 1 : 0,
            executor: activeThread.settings.executor,
            compression: activeThread.settings.compression
          }
        );
        llmvmThreadId = String(llmvmThread.id);

        // Update thread with LLMVM ID
        setThreads(prev => prev.map(thread =>
          thread.id === activeThreadId
            ? { ...thread, llmvmThreadId }
            : thread
        ));
      }

      // Execute Python code
      const result = await llmvmService.executePython(llmvmThreadId, code);

      // Update with result
      setThreads(prev => prev.map(thread =>
        thread.id === activeThreadId
          ? {
              ...thread,
              messages: thread.messages.map(msg =>
                msg.id === resultMessageId
                  ? {
                      ...msg,
                      content: `\`\`\`\n${JSON.stringify(result, null, 2)}\n\`\`\``,
                      status: "success"
                    }
                  : msg.id === userMessage.id
                  ? { ...msg, status: "success" }
                  : msg
              )
            }
          : thread
      ));

    } catch (error) {
      console.error('Failed to execute Python:', error);
      // Update message status to error
      setThreads(prev => prev.map(thread =>
        thread.id === activeThreadId
          ? {
              ...thread,
              messages: thread.messages.map(msg =>
                msg.id === resultMessageId
                  ? {
                      ...msg,
                      content: `Error executing Python code:\n\`\`\`\n${error}\n\`\`\``,
                      status: "error"
                    }
                  : msg.id === userMessage.id
                  ? { ...msg, status: "error" }
                  : msg
              )
            }
          : thread
      ));
    }
  };

  const handleTitleClick = () => {
    if (activeThread) {
      setEditedTitle(activeThread.title);
      setEditingTitle(true);
    }
  };

  const handleTitleSave = async () => {
    if (!activeThread || !editedTitle.trim()) {
      setEditingTitle(false);
      return;
    }

    const newTitle = editedTitle.trim();

    // Update local state
    setThreads(prev => prev.map(thread =>
      thread.id === activeThreadId
        ? { ...thread, title: newTitle }
        : thread
    ));

    // Update on server if connected
    if (isConnected && activeThread.llmvmThreadId) {
      try {
        await llmvmService.setThreadTitle(activeThread.llmvmThreadId, newTitle);
      } catch (error) {
        console.error('Failed to update thread title on server:', error);
        // Optionally revert the title on error
      }
    }

    setEditingTitle(false);
  };

  const handleTitleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleTitleSave();
    } else if (e.key === 'Escape') {
      setEditingTitle(false);
    }
  };

  // Tab management functions
  const openThreadInNewTab = (threadId: string) => {
    const thread = threads.find(t => t.id === threadId) || programs.find(p => p.id === threadId);
    if (!thread) return;

    // Check if thread is already open in a tab
    const existingTab = tabs.find(t => t.threadId === threadId);
    if (existingTab) {
      setActiveTabId(existingTab.id);
      return;
    }

    // Create new tab
    const newTab: Tab = {
      id: `tab-${Date.now()}`,
      threadId: threadId,
      title: thread.title
    };

    // Find the index of the current active tab
    const activeIndex = tabs.findIndex(t => t.id === activeTabId);
    const newTabs = [...tabs];
    newTabs.splice(activeIndex + 1, 0, newTab); // Insert after current tab
    
    setTabs(newTabs);
    setActiveTabId(newTab.id);
  };

  const openThreadInCurrentTab = (threadId: string) => {
    const thread = threads.find(t => t.id === threadId) || programs.find(p => p.id === threadId);
    if (!thread) return;

    // Check if thread is already open in a tab
    const existingTab = tabs.find(t => t.threadId === threadId);
    if (existingTab) {
      // Switch to existing tab instead of opening in current tab
      setActiveTabId(existingTab.id);
      return;
    }

    // If not already open, update current tab
    if (!activeTab) return;
    setTabs(tabs.map(tab => 
      tab.id === activeTabId 
        ? { ...tab, threadId: threadId, title: thread.title }
        : tab
    ));
  };

  const closeTab = (tabId: string) => {
    const tabIndex = tabs.findIndex(t => t.id === tabId);
    if (tabs.length === 1) return; // Don't close last tab

    const newTabs = tabs.filter(t => t.id !== tabId);
    
    // If closing the active tab, switch to adjacent tab
    if (tabId === activeTabId) {
      const newActiveIndex = tabIndex > 0 ? tabIndex - 1 : 0;
      setActiveTabId(newTabs[newActiveIndex].id);
    }
    
    setTabs(newTabs);
  };

  // Update tab title when thread title changes
  useEffect(() => {
    if (activeThread && activeTab) {
      setTabs(tabs.map(tab => 
        tab.threadId === activeThread.id 
          ? { ...tab, title: activeThread.title }
          : tab
      ));
    }
  }, [activeThread?.title]);

  return (
    <div className="flex h-full w-full bg-white">
      {/* Sidebar */}
      <div
        className={`transition-all duration-300 ease-in-out flex-shrink-0 ${
          sidebarOpen ? 'w-80' : 'w-0'
        }`}
        style={{
          overflow: sidebarOpen ? 'visible' : 'hidden',
          minWidth: sidebarOpen ? '20rem' : '0'
        }}
      >
        {sidebarOpen && (
          <ThreadSidebar
            threads={threads}
            programs={programs}
            activeThreadId={activeThreadId}
            onThreadSelect={(id: string) => {
              openThreadInCurrentTab(id);
            }}
            onThreadDoubleClick={(id: string) => {
              openThreadInNewTab(id);
            }}
            onNewThread={createNewThread}
          />
        )}
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-white">
        {/* Tabs */}
        <TabManager
          tabs={tabs}
          activeTabId={activeTabId}
          onTabSelect={setActiveTabId}
          onTabClose={closeTab}
        />
        
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-white">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-gray-600 hover:text-gray-900"
            >
              {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
            </Button>
            <div>
              {editingTitle ? (
                <input
                  type="text"
                  value={editedTitle}
                  onChange={(e) => setEditedTitle(e.target.value)}
                  onKeyPress={handleTitleKeyPress}
                  onBlur={handleTitleSave}
                  className="text-lg font-semibold text-gray-900 bg-transparent border-b-2 border-blue-500 outline-none"
                  autoFocus
                />
              ) : (
                <h1
                  className="text-lg font-semibold text-gray-900 cursor-pointer hover:text-blue-600"
                  onClick={handleTitleClick}
                  title="Click to edit title"
                >
                  {activeThread?.title || "LLMVM Chat"}
                </h1>
              )}
              <p className="text-sm text-gray-600">
                Model: {activeThread?.settings.model} • Mode: {activeThread?.mode}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {!isConnected && (
              <span className="text-xs text-red-600 bg-red-100 px-2 py-1 rounded">
                Disconnected
              </span>
            )}
            <span className="text-xs text-gray-600 bg-gray-100 px-2 py-1 rounded">
              {activeThread?.settings.executor}
            </span>
            {activeThread && (
              <ThreadSettingsDialog
                settings={activeThread.settings}
                onSettingsChange={updateThreadSettings}
              />
            )}
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-hidden">
          <MessageDisplay ref={messageDisplayRef} messages={activeThread?.messages || []} />
        </div>

        {/* Input */}
        <div className="border-t border-gray-200 bg-white">
          <MessageInput
            onSend={sendMessage}
            settings={activeThread?.settings}
            onSettingsChange={updateThreadSettings}
            onPythonExecute={executePython}
          />
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;

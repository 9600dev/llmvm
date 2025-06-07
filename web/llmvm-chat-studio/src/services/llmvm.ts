import { LLMVMClient, parseStreamChunk, user, assistant, system, Content } from 'llmvm-sdk';
import type {
  SessionThreadModel as LLMVMThread,
  Message as LLMVMMessage,
  Content as LLMVMContent,
  ChatCompletionRequest,
  MessageModel
} from 'llmvm-sdk';

export interface LLMVMConfig {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
}

export class LLMVMService {
  private client: LLMVMClient;
  private currentThreadId: string | null = null;

  constructor(config: LLMVMConfig = {}) {
    this.client = new LLMVMClient({
      baseUrl: config.baseUrl || 'http://localhost:8011',
      apiKey: config.apiKey,
      timeout: config.timeout || 30000,
    });
  }

  async checkHealth(): Promise<boolean> {
    try {
      const response = await this.client.health();
      return response.status === 'ok';
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  async createNewThread(
    messages: LLMVMMessage[] = [],
    options?: Partial<LLMVMThread>
  ): Promise<LLMVMThread> {
    // Convert messages to MessageModel format
    const messageModels: MessageModel[] = messages.map(msg => ({
      role: msg.role,
      content: Array.isArray(msg.message) ? msg.message : [],
      pinned: msg.pinned || 0,
      prompt_cached: msg.prompt_cached || false,
      total_tokens: 0
    }));

    const thread: Partial<LLMVMThread> = {
      id: Date.now(),
      executor: options?.executor || 'openai',
      model: options?.model || 'gpt-4.1',
      compression: options?.compression || 'auto',
      temperature: options?.temperature || 1.0,
      stop_tokens: options?.stop_tokens || [],
      output_token_len: options?.output_token_len || 8192,
      current_mode: options?.current_mode || 'tools',
      thinking: options?.thinking || 0,
      cookies: [],
      messages: messageModels,
      ...options
    };

    const createdThread = await this.client.setThread(thread as LLMVMThread);
    this.currentThreadId = String(createdThread.id);
    return createdThread;
  }

  async getThread(threadId: string): Promise<LLMVMThread | null> {
    try {
      const thread = await this.client.getThread(threadId);
      return thread;
    } catch (error) {
      console.error('Failed to get thread:', error);
      return null;
    }
  }

  async getAllThreads(): Promise<LLMVMThread[]> {
    try {
      const threads = await this.client.getThreads();
      return threads;
    } catch (error) {
      console.error('Failed to get threads:', error);
      return [];
    }
  }

  async getAllPrograms(): Promise<LLMVMThread[]> {
    try {
      const programs = await this.client.getPrograms();
      return programs;
    } catch (error) {
      console.error('Failed to get programs:', error);
      return [];
    }
  }

  async getProgram(idOrName: number | string): Promise<LLMVMThread | null> {
    try {
      const program = await this.client.getProgram(idOrName);
      return program;
    } catch (error) {
      console.error('Failed to get program:', error);
      return null;
    }
  }

  async deleteThread(threadId: string): Promise<boolean> {
    try {
      // LLMVM SDK doesn't have a direct delete method, so we'll clear all threads
      // This is a limitation we'll need to work around
      await this.client.clearThreads();
      return true;
    } catch (error) {
      console.error('Failed to delete thread:', error);
      return false;
    }
  }

  async setThreadTitle(threadId: string, title: string): Promise<void> {
    try {
      await this.client.setThreadTitle(Number(threadId), title);
    } catch (error) {
      console.error('Failed to set thread title:', error);
      throw error;
    }
  }

  async sendMessage(
    threadId: string,
    content: string | LLMVMContent[],
    options: {
      onChunk?: (chunk: any) => void;
      model?: string;
      temperature?: number;
      maxTokens?: number;
      mode?: 'tools' | 'direct' | 'code';
      thinking?: boolean;
      executor?: string;
      compression?: string;
    } = {}
  ): Promise<LLMVMMessage> {
    try {
      // Get the thread first
      const thread = await this.getThread(threadId);
      if (!thread) {
        throw new Error('Thread not found');
      }

      // Add user message in MessageModel format
      const userMessage = user(content);
      const messageModel: MessageModel = {
        role: userMessage.role,
        content: Array.isArray(userMessage.message) ? userMessage.message : [],
        pinned: userMessage.pinned || 0,
        prompt_cached: userMessage.prompt_cached || false,
        total_tokens: 0
      };
      thread.messages.push(messageModel);

      // Update thread settings
      if (options.model) thread.model = options.model;
      if (options.temperature !== undefined) thread.temperature = options.temperature;
      if (options.maxTokens) thread.output_token_len = options.maxTokens;
      if (options.thinking !== undefined) thread.thinking = options.thinking ? 1 : 0;
      if (options.executor) thread.executor = options.executor;
      if (options.compression) thread.compression = options.compression;

      // Save the updated thread
      await this.client.setThread(thread);

      // Determine completion mode
      if (options.mode === 'tools' || thread.current_mode === 'tools') {
        // Use tools completion for streaming
        const response = await this.client.toolsCompletions(
          thread,
          options.onChunk ? (chunk: any) => {
            // Skip thread model chunks
            if (chunk.id !== undefined && chunk.messages !== undefined) {
              return;
            }
            // Handle StreamNode objects
            if (chunk.obj !== undefined && chunk.type !== undefined) {
              // This is a StreamNode
              if (chunk.type === 'bytes') {
                // The obj might be the base64 string directly or wrapped in py/b64
                const imageData = typeof chunk.obj === 'string' ? chunk.obj : chunk.obj?.['py/b64'] || chunk.obj;
                if (imageData) {
                  options.onChunk!({ type: 'image', data: imageData });
                  return;
                }
              }
            }
            const content = parseStreamChunk(chunk);
            if (content) {
              options.onChunk!(content);
            }
          } : undefined
        );

        // The response thread contains the assistant's message
        const lastMessage = response.messages[response.messages.length - 1];
        // Convert MessageModel back to Message format
        return {
          role: lastMessage.role as 'assistant',
          message: lastMessage.content || [],
          pinned: lastMessage.pinned || 0,
          prompt_cached: lastMessage.prompt_cached || false,
          hidden: false
        } as LLMVMMessage;
      } else {
        // Use chat completion
        const request: ChatCompletionRequest = {
          model: thread.model,
          messages: thread.messages.map(msg => ({
            role: msg.role,
            content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
          })),
          temperature: thread.temperature,
          max_tokens: thread.output_token_len,
          stream: !!options.onChunk
        };

        const response = await this.client.chatCompletions(request);

        if (options.onChunk && 'on' in response) {
          // Handle streaming response
          let fullContent = '';
          response.on('data', (chunk: any) => {
            const parsed = parseStreamChunk(chunk);
            if (parsed) {
              fullContent += parsed;
              options.onChunk!(parsed);
            }
          });

          await new Promise((resolve, reject) => {
            response.on('end', resolve);
            response.on('error', reject);
          });

          return assistant(fullContent);
        } else {
          // Non-streaming response
          const choice = response.choices[0];
          return assistant(choice.message.content);
        }
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      throw error;
    }
  }

  async executePython(threadId: string, code: string): Promise<any> {
    try {
      const result = await this.client.executePython(threadId, code);
      return result;
    } catch (error) {
      console.error('Failed to execute Python:', error);
      throw error;
    }
  }

  async compileThread(
    threadId: string,
    compilePrompt?: string,
    options: {
      onChunk?: (chunk: any) => void;
    } = {}
  ): Promise<LLMVMThread> {
    try {
      const result = await this.client.compile(
        Number(threadId),
        compilePrompt,
        options.onChunk ? (chunk: any) => {
          // Skip thread model chunks
          if (chunk.id !== undefined && chunk.messages !== undefined) {
            return;
          }
          const content = parseStreamChunk(chunk);
          if (content) {
            options.onChunk!(content);
          }
        } : undefined
      );
      return result;
    } catch (error) {
      console.error('Failed to compile thread:', error);
      throw error;
    }
  }

  async downloadContent(
    url: string,
    onProgress?: (chunk: any) => void
  ): Promise<any> {
    try {
      const result = await this.client.download(
        { url, type: 'pdf' },
        onProgress
      );
      return result;
    } catch (error) {
      console.error('Failed to download content:', error);
      throw error;
    }
  }

  // Helper method to convert our Message type to LLMVM Message
  convertToLLMVMMessage(message: {
    id: string;
    content: string | any[];
    role: 'user' | 'assistant';
    type?: string;
  }): LLMVMMessage {
    if (message.role === 'user') {
      return user(message.content);
    } else {
      return assistant(message.content);
    }
  }

  // Helper to extract text from LLMVM content
  extractTextFromContent(content: LLMVMContent | LLMVMContent[] | string): string {
    if (typeof content === 'string') {
      return content;
    }

    if (Array.isArray(content)) {
      return content.map(c => this.extractTextFromContent(c)).join('\n');
    }

    if ('text' in content && typeof content.text === 'string') {
      return content.text;
    }

    if ('content_type' in content && content.content_type === 'text' && 'sequence' in content) {
      return content.sequence as string;
    }

    if (content && typeof content === 'object' && 'getText' in content && typeof content.getText === 'function') {
      return content.getText();
    }

    // Handle message array from LLMVM
    if (content && typeof content === 'object' && 'message' in content && Array.isArray(content.message)) {
      return content.message.map((c: any) => this.extractTextFromContent(c)).join('\n');
    }

    return JSON.stringify(content);
  }

  // Convert raw LLMVM message content to proper Content objects
  convertMessageToContent(message: any): LLMVMContent[] {
    if (!message) return [];

    // If it's already a Message object with content array
    if (message.message && Array.isArray(message.message)) {
      return message.message.map((item: any) => {
        // If it already has a render method, it's a Content object
        if (item.render && typeof item.render === 'function') {
          return item;
        }
        // Otherwise, convert it using Content.fromJSON
        try {
          return Content.fromJSON(item);
        } catch (error) {
          console.error('Failed to convert content item:', error, item);
          // Fallback to text content
          return Content.fromJSON({
            type: 'text',
            content_type: 'text',
            sequence: JSON.stringify(item),
            url: ''
          });
        }
      });
    }

    // If it's an array of content items
    if (Array.isArray(message)) {
      return message.map((item: any) => {
        try {
          return Content.fromJSON(item);
        } catch (error) {
          console.error('Failed to convert content item:', error, item);
          return Content.fromJSON({
            type: 'text',
            content_type: 'text',
            sequence: JSON.stringify(item),
            url: ''
          });
        }
      });
    }

    // Single content item
    try {
      return [Content.fromJSON(message)];
    } catch (error) {
      console.error('Failed to convert content:', error, message);
      return [Content.fromJSON({
        type: 'text',
        content_type: 'text',
        sequence: JSON.stringify(message),
        url: ''
      })];
    }
  }
}

// Singleton instance
let llmvmService: LLMVMService | null = null;

export function getLLMVMService(config?: LLMVMConfig): LLMVMService {
  if (!llmvmService) {
    // In development, use proxy path to avoid CORS
    const isDev = import.meta.env.DEV;
    const baseUrl = isDev
      ? '/api' // Use /api prefix which gets proxied to localhost:8011
      : (import.meta.env.VITE_LLMVM_BASE_URL || config?.baseUrl || 'http://localhost:8011');

    llmvmService = new LLMVMService({
      baseUrl,
      apiKey: import.meta.env.VITE_LLMVM_API_KEY || config?.apiKey,
      ...config
    });
  }
  return llmvmService;
}

export function resetLLMVMService(): void {
  llmvmService = null;
}
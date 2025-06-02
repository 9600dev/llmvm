/**
 * LLMVM Client - Main API client for interacting with LLMVM server
 */

import { createParser, ParsedEvent, ReconnectInterval } from 'eventsource-parser';
import {
  LLMVMClientOptions,
  SessionThreadModel,
  MessageModel,
  DownloadItemModel,
  ChatCompletionRequest,
  StreamEvent
} from './types';
import { Message } from './models/Message';
import { decodeJsonpickle } from './utils';

export class LLMVMClient {
  private baseUrl: string;
  private headers: Record<string, string>;
  private timeout: number;

  constructor(options: LLMVMClientOptions = {}) {
    this.baseUrl = options.baseUrl || 'http://localhost:8011';
    this.timeout = options.timeout || 30000;
    this.headers = {
      'Content-Type': 'application/json',
      ...options.headers
    };

    if (options.apiKey) {
      this.headers['Authorization'] = `Bearer ${options.apiKey}`;
    }
  }

  /**
   * Make a fetch request with timeout
   */
  private async fetchWithTimeout(url: string, options: RequestInit): Promise<Response> {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      clearTimeout(id);
      return response;
    } catch (error) {
      clearTimeout(id);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Request timeout after ${this.timeout}ms`);
      }
      throw error;
    }
  }

  /**
   * Health check
   */
  async health(): Promise<{status: string}> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/health`, {
      method: 'GET',
      headers: this.headers
    });

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get a thread by ID
   */
  async getThread(id: number): Promise<SessionThreadModel> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/v1/chat/get_thread?id=${id}`, {
      method: 'GET',
      headers: this.headers
    });

    if (!response.ok) {
      throw new Error(`Failed to get thread: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Set/update a thread
   */
  async setThread(thread: SessionThreadModel): Promise<SessionThreadModel> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/v1/chat/set_thread`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(thread)
    });

    if (!response.ok) {
      throw new Error(`Failed to set thread: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get all threads
   */
  async getThreads(): Promise<SessionThreadModel[]> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/v1/chat/get_threads`, {
      method: 'GET',
      headers: this.headers
    });

    if (!response.ok) {
      throw new Error(`Failed to get threads: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Clear all threads
   */
  async clearThreads(): Promise<void> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/v1/chat/clear_threads`, {
      method: 'GET',
      headers: this.headers
    });

    if (!response.ok) {
      throw new Error(`Failed to clear threads: ${response.statusText}`);
    }
  }

  /**
   * Set thread title
   */
  async setThreadTitle(id: number, title: string): Promise<void> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/v1/chat/set_thread_title`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ id, title })
    });

    if (!response.ok) {
      throw new Error(`Failed to set thread title: ${response.statusText}`);
    }
  }

  /**
   * Set cookies for a thread
   */
  async setCookies(id: number, cookies: Array<{[key: string]: any}>): Promise<SessionThreadModel> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/v1/chat/cookies`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ id, cookies })
    });

    if (!response.ok) {
      throw new Error(`Failed to set cookies: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Execute Python code in a thread
   */
  async executePython(threadId: number, pythonCode: string): Promise<{
    var_name: string;
    var_value: any;
    results: any[];
    error: string;
  }> {
    const response = await this.fetchWithTimeout(
      `${this.baseUrl}/python?thread_id=${threadId}&python_str=${encodeURIComponent(pythonCode)}`,
      {
        method: 'GET',
        headers: this.headers
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to execute Python: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Download content from a URL
   */
  async download(downloadItem: DownloadItemModel, onChunk?: (chunk: any) => void): Promise<SessionThreadModel> {
    const response = await fetch(`${this.baseUrl}/download`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(downloadItem)
    });

    if (!response.ok) {
      throw new Error(`Failed to download: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    if (!reader) {
      throw new Error('No response body');
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            break;
          }
          try {
            const parsed = JSON.parse(data);
            if (onChunk) {
              onChunk(parsed);
            }
          } catch (e) {
            console.error('Failed to parse chunk:', e);
          }
        }
      }
    }

    // The last chunk should be the thread model
    const lastLine = buffer.trim();
    if (lastLine.startsWith('data: ')) {
      const data = lastLine.slice(6);
      if (data !== '[DONE]') {
        return JSON.parse(data);
      }
    }

    throw new Error('No thread data received');
  }

  /**
   * OpenAI-compatible chat completions endpoint
   */
  async chatCompletions(request: ChatCompletionRequest): Promise<any> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      throw new Error(`Chat completion failed: ${response.statusText}`);
    }

    if (request.stream) {
      return this.handleStream(response);
    }

    return response.json();
  }

  /**
   * Tools completions with streaming support
   */
  async toolsCompletions(
    thread: SessionThreadModel,
    onChunk?: (chunk: any) => void
  ): Promise<SessionThreadModel> {
    const response = await fetch(`${this.baseUrl}/v1/tools/completions`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(thread)
    });

    if (!response.ok) {
      throw new Error(`Tools completion failed: ${response.statusText}`);
    }

    return this.handleStreamingResponse(response, onChunk);
  }

  /**
   * Handle streaming response
   */
  private async handleStreamingResponse(
    response: Response,
    onChunk?: (chunk: any) => void
  ): Promise<SessionThreadModel> {
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let lastThreadData: SessionThreadModel | null = null;

    if (!reader) {
      throw new Error('No response body');
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            break;
          }
          try {
            const parsed = decodeJsonpickle(data);

            // Check if this is a thread model (has id and messages)
            if (parsed.id !== undefined && parsed.messages !== undefined) {
              lastThreadData = parsed;
            } else if (onChunk) {
              onChunk(parsed);
            }
          } catch (e) {
            console.error('Failed to parse chunk:', e);
          }
        }
      }
    }

    if (lastThreadData) {
      return lastThreadData;
    }

    throw new Error('No thread data received');
  }

  /**
   * Handle SSE stream for chat completions
   */
  private async *handleStream(response: Response): AsyncGenerator<StreamEvent> {
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    const parser = createParser((event: ParsedEvent | ReconnectInterval) => {
      if (event.type === 'event') {
        if (event.data === '[DONE]') {
          return;
        }
        try {
          const data = JSON.parse(event.data);
          return data;
        } catch (e) {
          console.error('Failed to parse SSE event:', e);
        }
      }
    });

    if (!reader) {
      throw new Error('No response body');
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      parser.feed(chunk);
    }
  }

  /**
   * Create a new thread with initial messages
   */
  async createThread(
    messages: Message[] = [],
    options: Partial<SessionThreadModel> = {}
  ): Promise<SessionThreadModel> {
    const messageModels = messages.map(m => m.toMessageModel());

    const thread: SessionThreadModel = {
      id: 0,
      executor: options.executor || '',
      model: options.model || '',
      compression: options.compression || '',
      temperature: options.temperature || 0.0,
      stop_tokens: options.stop_tokens || [],
      output_token_len: options.output_token_len || 0,
      current_mode: options.current_mode || 'tools',
      thinking: options.thinking || 0,
      cookies: options.cookies || [],
      messages: messageModels
    };

    return this.setThread(thread);
  }

  /**
   * Add messages to an existing thread
   */
  async addMessages(threadId: number, messages: Message[]): Promise<SessionThreadModel> {
    const thread = await this.getThread(threadId);
    const newMessages = messages.map(m => m.toMessageModel());
    thread.messages.push(...newMessages);
    return this.setThread(thread);
  }

  /**
   * Simple completion helper
   */
  async complete(
    messages: Message[],
    options: {
      model?: string;
      temperature?: number;
      maxTokens?: number;
      stream?: boolean;
      onChunk?: (chunk: any) => void;
    } = {}
  ): Promise<Message> {
    // Create a temporary thread
    const thread = await this.createThread(messages, {
      model: options.model,
      temperature: options.temperature,
      output_token_len: options.maxTokens
    });

    // Execute completion
    const result = await this.toolsCompletions(thread, options.onChunk);

    // Extract the last assistant message
    const lastMessage = result.messages[result.messages.length - 1];
    if (lastMessage && lastMessage.role === 'assistant') {
      return Message.fromMessageModel(lastMessage);
    }

    throw new Error('No assistant response received');
  }
}
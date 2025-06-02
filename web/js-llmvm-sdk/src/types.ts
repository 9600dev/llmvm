/**
 * Type definitions for LLMVM SDK
 */

export interface Content {
  type: string;
  sequence: string | Uint8Array | Content[];
  content_type: string;
  url: string;
  original_sequence?: any;
}

export interface TextContent extends Content {
  type: 'TextContent';
  sequence: string;
  content_type: 'text';
}

export interface ImageContent extends Content {
  type: 'ImageContent';
  sequence: string; // base64 encoded
  content_type: 'image';
  image_type?: string;
}

export interface PdfContent extends Content {
  type: 'PdfContent';
  sequence: string; // base64 encoded
  content_type: 'pdf';
}

export interface FileContent extends Content {
  type: 'FileContent';
  sequence: string; // base64 encoded
  content_type: 'file';
}

export interface BrowserContent extends Content {
  type: 'BrowserContent';
  sequence: Content[];
  content_type: 'browser';
}

export interface MarkdownContent extends Content {
  type: 'MarkdownContent';
  sequence: Content[];
  content_type: 'markdown';
}

export interface HTMLContent extends Content {
  type: 'HTMLContent';
  sequence: Content[] | string;
  content_type: 'html';
}

export interface SearchResult extends TextContent {
  title: string;
  snippet: string;
  engine: string;
}

export interface YelpResult extends TextContent {
  title: string;
  link: string;
  neighborhood: string;
  snippet: string;
  reviews: string;
}

export interface HackerNewsResult extends TextContent {
  title: string;
  author: string;
  comment_text: string;
  created_at: string;
}

export interface Message {
  role: 'user' | 'assistant' | 'system' | 'developer';
  message: Content[];
  pinned: number;
  prompt_cached: boolean;
  hidden: boolean;
}

export interface UserMessage extends Message {
  role: 'user';
}

export interface AssistantMessage extends Message {
  role: 'assistant';
  error?: boolean;
  thinking?: Content;
  system_context?: any;
  stop_reason?: string;
  stop_token?: string;
  total_tokens?: number;
  underlying?: any;
}

export interface SystemMessage extends Message {
  role: 'system';
}

export interface MessageModel {
  role: string;
  content: ContentModel[];
  pinned: number;
  prompt_cached: boolean;
  total_tokens: number;
  underlying?: any;
}

export interface ContentModel {
  sequence: any;
  content_type: string;
  original_sequence?: any;
  url: string;
}

export interface SessionThreadModel {
  id: number;
  executor: string;
  model: string;
  compression: string;
  temperature: number;
  stop_tokens: string[];
  output_token_len: number;
  current_mode: string;
  thinking: number;
  cookies: Array<{[key: string]: any}>;
  messages: MessageModel[];
  locals_dict?: {[key: string]: any};
}

export interface DownloadItemModel {
  id: number;
  url: string;
}

export interface ChatCompletionRequest {
  messages: Array<{
    role: string;
    content: string | Array<{type: string; text?: string; image_url?: string}>;
  }>;
  model: string;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  tools?: any[];
}

export interface StreamEvent {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      content?: string;
      tool_calls?: any[];
    };
    finish_reason: string | null;
  }>;
}

export interface LLMVMClientOptions {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
  headers?: Record<string, string>;
}
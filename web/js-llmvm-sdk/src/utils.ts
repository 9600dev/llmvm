/**
 * Utility functions for LLMVM SDK
 */

import { Content as IContent, Message as IMessage } from './types';
import { Content, TextContent } from './models/Content';
import { Message, UserMessage, AssistantMessage, SystemMessage } from './models/Message';

/**
 * Create a text content helper
 */
export function text(content: string, url?: string): TextContent {
  return new TextContent({ sequence: content, url: url || '' });
}

/**
 * Create a user message helper
 */
export function user(content: string | Content | Content[]): UserMessage {
  if (typeof content === 'string') {
    return new UserMessage({ message: [text(content)] });
  } else if (Array.isArray(content)) {
    return new UserMessage({ message: content });
  } else {
    return new UserMessage({ message: [content] });
  }
}

/**
 * Create an assistant message helper
 */
export function assistant(content: string | Content | Content[]): AssistantMessage {
  if (typeof content === 'string') {
    return new AssistantMessage({ message: [text(content)] });
  } else if (Array.isArray(content)) {
    return new AssistantMessage({ message: content });
  } else {
    return new AssistantMessage({ message: [content] });
  }
}

/**
 * Create a system message helper
 */
export function system(content: string): SystemMessage {
  return new SystemMessage({ message: [text(content)] });
}

/**
 * Parse a streaming chunk and extract content
 */
export function parseStreamChunk(chunk: any): string | null {
  // Handle OpenAI-style streaming chunks
  if (chunk.choices && chunk.choices[0] && chunk.choices[0].delta) {
    return chunk.choices[0].delta.content || null;
  }

  // Handle LLMVM TokenNode objects (Python-serialized)
  if (chunk['py/object'] === 'llmvm.common.objects.TokenNode' && 'token' in chunk) {
    return chunk.token || null;
  }

  // Handle plain token objects
  if (chunk.token !== undefined) {
    return chunk.token || null;
  }

  return null;
}

/**
 * Render messages to a container element
 */
export function renderMessages(messages: Message[], container: HTMLElement): void {
  container.innerHTML = '';
  messages.forEach(message => {
    const rendered = message.render();
    container.appendChild(rendered);
  });
}

/**
 * Fully decode a jsonpickle payload to plain JS objects.
 *   – Removes all "py/*" metadata.
 *   – Re-links objects referenced with py/id ↔ py/ref.
 *
 * Accepts either a JSON string or a parsed JS value.
 */
export function decodeJsonpickle(json) {
  const root = typeof json === 'string' ? JSON.parse(json) : json;
  const idMap = new Map();         // id ⇒ final JS object

  function walk(node) {
    if (node === null || typeof node !== 'object') return node;
    if (Array.isArray(node)) return node.map(walk);

    if ('py/b64' in node) {
      return node['py/b64'];
    }

    // Handle reference
    if ('py/ref' in node) return idMap.get(node['py/ref']);

    // Create a fresh object for this node
    const out = {};
    if ('py/id' in node) idMap.set(node['py/id'], out);

    // Copy over real data, recurse where needed
    for (const [k, v] of Object.entries(node)) {
      // if (k.startsWith('py/')) continue;   // drop metadata
      out[k] = walk(v);
    }
    return out;
  }

  return walk(root);
}

/**
 * Add CSS styles for rendering
 */
export function addDefaultStyles(): void {
  if (document.getElementById('llmvm-default-styles')) {
    return;
  }

  const style = document.createElement('style');
  style.id = 'llmvm-default-styles';
  style.textContent = `
    .message {
      margin: 1em 0;
      padding: 1em;
      border-radius: 8px;
      background: #f5f5f5;
    }

    .message.user-message {
      background: #e3f2fd;
    }

    .message.assistant-message {
      background: #f3e5f5;
    }

    .message.system-message {
      background: #fff3e0;
      font-style: italic;
    }

    .message-header {
      font-weight: bold;
      margin-bottom: 0.5em;
      color: #333;
    }

    .message-content {
      color: #333;
      line-height: 1.6;
    }

    .message-content p {
      margin: 0.5em 0;
    }

    .message.error {
      background: #ffebee;
      border: 1px solid #f44336;
    }

    .message.pinned {
      border: 2px solid #2196f3;
    }

    .token-count {
      font-size: 0.8em;
      color: #666;
      font-weight: normal;
    }

    .thinking-content {
      margin: 0.5em 0;
      padding: 0.5em;
      background: rgba(0,0,0,0.05);
      border-radius: 4px;
    }

    .thinking-content summary {
      cursor: pointer;
      user-select: none;
      color: #666;
    }

    .message-meta {
      margin-top: 0.5em;
      font-size: 0.8em;
      color: #666;
    }

    .message-meta span {
      margin-right: 1em;
    }

    /* Content type styles */
    .text-content {
      white-space: pre-wrap;
    }

    .image-content {
      margin: 0.5em 0;
    }

    .image-content img {
      max-width: 100%;
      height: auto;
      border-radius: 4px;
    }

    .image-caption {
      font-size: 0.9em;
      color: #666;
      margin-top: 0.25em;
    }

    .pdf-content embed {
      border: 1px solid #ddd;
      border-radius: 4px;
    }

    .file-content {
      margin: 0.5em 0;
    }

    .file-header {
      font-weight: bold;
      margin-bottom: 0.5em;
      color: #666;
    }

    .file-content pre {
      background: #f5f5f5;
      padding: 1em;
      border-radius: 4px;
      overflow-x: auto;
    }

    .browser-content {
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 1em;
      margin: 0.5em 0;
    }

    .browser-header {
      margin-bottom: 1em;
      padding-bottom: 0.5em;
      border-bottom: 1px solid #eee;
    }

    .search-result {
      margin: 0.5em 0;
      padding: 0.5em;
      border: 1px solid #eee;
      border-radius: 4px;
    }

    .search-result-title {
      font-weight: bold;
      color: #1a73e8;
      text-decoration: none;
      display: block;
      margin-bottom: 0.25em;
    }

    .search-result-title:hover {
      text-decoration: underline;
    }

    .search-result-snippet {
      color: #545454;
      margin: 0.25em 0;
    }

    .search-result-meta {
      font-size: 0.8em;
      color: #70757a;
    }

    .yelp-result, .hn-result {
      margin: 0.5em 0;
      padding: 0.5em;
      border: 1px solid #eee;
      border-radius: 4px;
    }

    .yelp-result-title, .hn-result-title {
      font-weight: bold;
      color: #d32323;
      text-decoration: none;
      display: block;
      margin-bottom: 0.25em;
    }

    .yelp-result-neighborhood {
      color: #666;
      font-size: 0.9em;
      margin-bottom: 0.25em;
    }

    .hn-result-meta {
      color: #828282;
      font-size: 0.8em;
      margin-bottom: 0.5em;
    }

    .hn-result-comment {
      line-height: 1.4;
    }

    /* Loading animation */
    .message.streaming::after {
      content: '...';
      display: inline-block;
      animation: dots 1.5s steps(4, end) infinite;
    }

    @keyframes dots {
      0%, 20% {
        content: '.';
      }
      40% {
        content: '..';
      }
      60% {
        content: '...';
      }
      80%, 100% {
        content: '';
      }
    }
  `;

  document.head.appendChild(style);
}
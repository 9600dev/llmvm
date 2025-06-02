/**
 * LLMVM SDK - JavaScript/TypeScript SDK for LLMVM FastAPI server
 */

// Main client
export { LLMVMClient } from './client';

// Models
export { Content } from './models/Content';
export { 
  TextContent,
  ImageContent,
  PdfContent,
  FileContent,
  BrowserContent,
  MarkdownContent,
  HTMLContent,
  SearchResult,
  YelpResult,
  HackerNewsResult
} from './models/Content';

export {
  Message,
  UserMessage,
  AssistantMessage,
  SystemMessage
} from './models/Message';

// Utilities
export {
  text,
  user,
  assistant,
  system,
  parseStreamChunk,
  renderMessages,
  addDefaultStyles
} from './utils';

// Types
export * from './types';

// Browser-specific initialization
if (typeof window !== 'undefined') {
  // Add default styles when loaded in browser
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      import('./utils').then(({ addDefaultStyles }) => {
        addDefaultStyles();
      });
    });
  } else {
    import('./utils').then(({ addDefaultStyles }) => {
      addDefaultStyles();
    });
  }
}
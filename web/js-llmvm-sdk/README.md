# LLMVM JavaScript/TypeScript SDK

A comprehensive JavaScript/TypeScript SDK for interacting with the LLMVM FastAPI server. This SDK provides type-safe models, content rendering capabilities, and easy-to-use API methods for building web applications that interface with LLMVM.

## Features

- 🚀 **Full TypeScript support** with complete type definitions
- 🎨 **Built-in content rendering** for all LLMVM content types
- 🌊 **Streaming support** for real-time responses
- 🔧 **Easy-to-use API** with both low-level and high-level methods
- 📦 **Multiple build formats** (ESM, CommonJS, UMD)
- 🎯 **Zero dependencies** (except for SSE parsing)

## Installation

```bash
npm install llmvm-sdk
# or
yarn add llmvm-sdk
# or
pnpm add llmvm-sdk
```

## Quick Start

```typescript
import { LLMVMClient, user, assistant, text } from 'llmvm-sdk';

// Initialize the client
const client = new LLMVMClient({
  baseUrl: 'http://localhost:8011',
  timeout: 30000
});

// Create a simple conversation
const messages = [
  user('Hello! Can you help me with JavaScript?'),
];

// Get a completion
const response = await client.complete(messages, {
  model: 'gpt-4',
  temperature: 0.7,
  maxTokens: 1000
});

console.log(response.getText());
```

## Content Types and Rendering

The SDK supports all LLMVM content types with built-in rendering methods:

### Text Content

```typescript
import { TextContent } from 'llmvm-sdk';

const content = new TextContent({
  sequence: 'Hello, world!'
});

// Render to HTML element
const element = content.render();
document.body.appendChild(element);

// Get plain text
const text = content.getText(); // "Hello, world!"
```

### Image Content

```typescript
import { ImageContent } from 'llmvm-sdk';

const content = new ImageContent({
  sequence: 'base64_encoded_image_data',
  image_type: 'image/png',
  url: 'https://example.com/image.png'
});

const element = content.render(); // Creates an <img> element
```

### Browser Content

```typescript
import { BrowserContent, TextContent, ImageContent } from 'llmvm-sdk';

const content = new BrowserContent({
  url: 'https://example.com',
  sequence: [
    new TextContent({ sequence: 'Page title' }),
    new ImageContent({ sequence: 'screenshot_base64' })
  ]
});

const element = content.render(); // Renders all nested content
```

### Markdown Content

```typescript
import { MarkdownContent, TextContent } from 'llmvm-sdk';

const content = new MarkdownContent({
  sequence: [
    new TextContent({ sequence: '# Hello\\n\\nThis is **markdown**!' })
  ]
});

const element = content.render(); // Converts markdown to HTML
```

## API Methods

### Thread Management

```typescript
// Create a new thread
const thread = await client.createThread([
  system('You are a helpful assistant'),
  user('Hello!')
]);

// Get thread by ID
const existingThread = await client.getThread(thread.id);

// Update thread
thread.temperature = 0.8;
await client.setThread(thread);

// Get all threads
const allThreads = await client.getThreads();

// Clear all threads
await client.clearThreads();
```

### Completions

```typescript
// Simple completion
const response = await client.complete([
  user('What is the capital of France?')
], {
  model: 'gpt-4',
  temperature: 0.5
});

// Streaming completion
await client.complete(messages, {
  stream: true,
  onChunk: (chunk) => {
    console.log('Received:', chunk);
  }
});

// Tools completion with full control
const thread = await client.createThread(messages, {
  executor: 'openai',
  model: 'gpt-4',
  current_mode: 'tools',
  temperature: 0.7
});

const result = await client.toolsCompletions(thread, (chunk) => {
  // Handle streaming chunks
  console.log(chunk);
});
```

### OpenAI-Compatible Interface

```typescript
// Use OpenAI-compatible chat completions
const response = await client.chatCompletions({
  model: 'gpt-4',
  messages: [
    { role: 'user', content: 'Hello!' }
  ],
  temperature: 0.7,
  max_tokens: 1000,
  stream: false
});
```

### Python Execution

```typescript
// Execute Python code in a thread context
const result = await client.executePython(thread.id, `
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df.sum()
`);

console.log(result.var_name); // The last variable name
console.log(result.var_value); // The value of that variable
console.log(result.results); // All execution results
```

### Content Download

```typescript
// Download content from a URL
const downloadResult = await client.download(
  { id: thread.id, url: 'https://example.com' },
  (chunk) => {
    // Handle streaming download progress
    console.log('Downloaded chunk:', chunk);
  }
);
```

## Message Helpers

The SDK provides convenient helper functions for creating messages:

```typescript
import { user, assistant, system, text } from 'llmvm-sdk';

// Create messages easily
const messages = [
  system('You are a helpful assistant'),
  user('What is TypeScript?'),
  assistant('TypeScript is a typed superset of JavaScript...'),
  user([
    text('Can you show me an example?'),
    text('Make it simple please.')
  ])
];
```

## Rendering Messages in the Browser

```typescript
import { renderMessages, addDefaultStyles } from 'llmvm-sdk';

// Add default CSS styles (call once)
addDefaultStyles();

// Render messages to a container
const container = document.getElementById('chat-container');
renderMessages(messages, container);

// Or render individual messages
messages.forEach(message => {
  const element = message.render();
  container.appendChild(element);
});
```

## Advanced Usage

### Custom Content Types

You can extend the base Content class to create custom content types:

```typescript
import { Content } from 'llmvm-sdk';

class CustomContent extends Content {
  render(): HTMLElement {
    const div = document.createElement('div');
    div.className = 'custom-content';
    div.textContent = this.getText();
    return div;
  }

  getText(): string {
    return `Custom: ${this.sequence}`;
  }
}
```

### Error Handling

```typescript
try {
  const response = await client.complete(messages);
} catch (error) {
  if (error.message.includes('timeout')) {
    console.error('Request timed out');
  } else {
    console.error('API error:', error);
  }
}
```

### TypeScript Types

The SDK exports all TypeScript interfaces for type-safe development:

```typescript
import type {
  SessionThreadModel,
  MessageModel,
  Content,
  Message,
  LLMVMClientOptions
} from 'llmvm-sdk';

function processThread(thread: SessionThreadModel): void {
  thread.messages.forEach((msg: MessageModel) => {
    console.log(`${msg.role}: ${msg.content[0].sequence}`);
  });
}
```

## Browser Usage

The SDK can be used directly in the browser:

```html
<script src="https://unpkg.com/llmvm-sdk/dist/index.umd.js"></script>
<script>
  const client = new LLMVM.LLMVMClient({
    baseUrl: 'http://localhost:8011'
  });
  
  const messages = [
    LLMVM.user('Hello from the browser!')
  ];
  
  client.complete(messages).then(response => {
    console.log(response.getText());
  });
</script>
```

## Configuration

### Client Options

```typescript
const client = new LLMVMClient({
  // Base URL of the LLMVM server
  baseUrl: 'http://localhost:8011',
  
  // Request timeout in milliseconds
  timeout: 30000,
  
  // Optional API key
  apiKey: 'your-api-key',
  
  // Additional headers
  headers: {
    'X-Custom-Header': 'value'
  }
});
```

### Thread Options

```typescript
const thread = await client.createThread(messages, {
  executor: 'openai',        // or 'anthropic', 'gemini', etc.
  model: 'gpt-4',           // Model name
  compression: 'auto',       // Token compression method
  temperature: 0.7,          // Temperature for randomness
  stop_tokens: ['\\n\\n'],    // Stop generation tokens
  output_token_len: 2048,    // Max output tokens
  current_mode: 'tools',     // Execution mode
  thinking: 0,               // Thinking mode level
  cookies: []                // Browser cookies for web content
});
```

## Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/your-repo/llmvm-sdk.git
cd llmvm-sdk

# Install dependencies
npm install

# Build the SDK
npm run build

# Run tests
npm test

# Watch mode for development
npm run dev
```

### Running Examples

Check the `examples/` directory for complete working examples:

```bash
cd examples
npm install
npm start
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our repository.

## Support

- Documentation: https://docs.llmvm.com
- Issues: https://github.com/your-repo/llmvm-sdk/issues
- Discord: https://discord.gg/llmvm
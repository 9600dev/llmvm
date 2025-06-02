# LLMVM Chat Studio Integration

This chat interface is now integrated with the LLMVM JavaScript SDK to provide a full-featured chat experience with support for:

- **Streaming responses** - Real-time message streaming as the AI generates content
- **Multiple content types** - Images, PDFs, browser content, files, and more
- **Tool execution** - Visual display of tool usage and results
- **Model selection** - Support for Anthropic, OpenAI, and Gemini models
- **Advanced settings** - Temperature, token limits, thinking mode, etc.

## Setup

1. **Start LLMVM Server**
   ```bash
   llmvm --port 8011
   ```

2. **Configure Environment** (optional)
   Edit `.env` file to customize the LLMVM endpoint:
   ```
   VITE_LLMVM_BASE_URL=http://localhost:8011
   VITE_LLMVM_API_KEY=your-api-key-if-needed
   ```

3. **Install Dependencies**
   ```bash
   npm install
   ```

4. **Start Development Server**
   ```bash
   npm run dev
   ```

## Features

### Content Rendering
The chat interface automatically renders different content types returned by LLMVM:

- **Images** - Displayed inline with proper sizing
- **PDFs** - Shown in a scrollable viewer
- **Browser Content** - Web pages with nested content
- **Files** - Code files with syntax highlighting
- **Search Results** - Formatted search results with links

### Streaming
Messages are streamed in real-time as they're generated, providing immediate feedback.

### Thread Management
- Create multiple conversation threads
- Switch between threads
- Each thread maintains its own settings and history
- Threads are synced with LLMVM server

### Model Configuration
- Choose between different AI providers (Anthropic, OpenAI, Gemini)
- Select specific models
- Adjust temperature and token limits
- Enable thinking mode for detailed reasoning

## Architecture

- **ChatInterface.tsx** - Main component managing threads and messages
- **MessageDisplay.tsx** - Renders messages with proper formatting
- **ContentRenderer.tsx** - Handles different LLMVM content types
- **services/llmvm.ts** - LLMVM SDK integration layer

## Troubleshooting

### CORS Issues
The development server includes a proxy configuration to handle CORS. If you still encounter CORS errors:

1. **Option 1: Use the Vite proxy** (default in development)
   - The app automatically proxies requests through Vite in development mode
   - Make sure to restart the dev server after any proxy config changes

2. **Option 2: Configure LLMVM with CORS headers**
   - Start LLMVM with CORS enabled:
   ```bash
   llmvm --port 8011 --cors
   ```

3. **Option 3: Use a browser extension**
   - Install a CORS unblocking extension for development only

### Other Issues

- **"Disconnected" status** - Ensure LLMVM is running on the configured port
- **Message errors** - Check the browser console for detailed error messages
- **Content not rendering** - Verify the LLMVM server is returning proper content types
- **Proxy not working** - Restart the Vite dev server: `npm run dev`
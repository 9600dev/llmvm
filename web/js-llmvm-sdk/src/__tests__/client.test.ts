/**
 * Basic tests for LLMVM Client
 */

import { LLMVMClient } from '../client';
import { user, assistant, system, text } from '../utils';
import { TextContent, ImageContent, BrowserContent } from '../models/Content';
import { UserMessage, AssistantMessage } from '../models/Message';

describe('LLMVMClient', () => {
  let client: LLMVMClient;

  beforeEach(() => {
    client = new LLMVMClient({
      baseUrl: 'http://localhost:8011',
      timeout: 5000
    });
  });

  describe('initialization', () => {
    it('should create client with default options', () => {
      const defaultClient = new LLMVMClient();
      expect(defaultClient).toBeDefined();
    });

    it('should create client with custom options', () => {
      const customClient = new LLMVMClient({
        baseUrl: 'http://custom:8080',
        timeout: 10000,
        apiKey: 'test-key',
        headers: { 'X-Custom': 'value' }
      });
      expect(customClient).toBeDefined();
    });
  });
});

describe('Content Models', () => {
  describe('TextContent', () => {
    it('should create text content', () => {
      const content = new TextContent({
        sequence: 'Hello, world!',
        url: 'https://example.com'
      });
      
      expect(content.getText()).toBe('Hello, world!');
      expect(content.content_type).toBe('text');
      expect(content.url).toBe('https://example.com');
    });

    it('should render text content to HTML', () => {
      const content = new TextContent({ sequence: 'Line 1\\n\\nLine 2' });
      const element = content.render();
      
      expect(element).toBeInstanceOf(HTMLElement);
      expect(element.className).toBe('text-content');
      expect(element.querySelectorAll('p').length).toBe(2);
    });
  });

  describe('ImageContent', () => {
    it('should create image content', () => {
      const content = new ImageContent({
        sequence: 'base64data',
        image_type: 'image/png',
        url: 'https://example.com/image.png'
      });
      
      expect(content.content_type).toBe('image');
      expect(content.image_type).toBe('image/png');
    });
  });

  describe('BrowserContent', () => {
    it('should create browser content with nested content', () => {
      const content = new BrowserContent({
        url: 'https://example.com',
        sequence: [
          new TextContent({ sequence: 'Title' }),
          new TextContent({ sequence: 'Description' })
        ]
      });
      
      expect(content.sequence.length).toBe(2);
      expect(content.getText()).toContain('Title');
      expect(content.getText()).toContain('Description');
    });
  });

  describe('Content.fromJSON', () => {
    it('should create correct content type from JSON', () => {
      const textData = {
        content_type: 'text',
        sequence: 'Hello',
        url: ''
      };
      
      const content = Content.fromJSON(textData);
      expect(content).toBeInstanceOf(TextContent);
      expect(content.getText()).toBe('Hello');
    });
  });
});

describe('Message Models', () => {
  describe('UserMessage', () => {
    it('should create user message from string', () => {
      const message = user('Hello!');
      expect(message.role).toBe('user');
      expect(message.getText()).toBe('Hello!');
    });

    it('should create user message from content', () => {
      const content = new TextContent({ sequence: 'Hello!' });
      const message = user(content);
      expect(message.message.length).toBe(1);
      expect(message.message[0]).toBe(content);
    });

    it('should create user message from content array', () => {
      const contents = [
        new TextContent({ sequence: 'Part 1' }),
        new TextContent({ sequence: 'Part 2' })
      ];
      const message = user(contents);
      expect(message.message.length).toBe(2);
    });
  });

  describe('AssistantMessage', () => {
    it('should create assistant message with metadata', () => {
      const message = new AssistantMessage({
        message: [new TextContent({ sequence: 'Response' })],
        total_tokens: 100,
        stop_reason: 'stop',
        error: false
      });
      
      expect(message.role).toBe('assistant');
      expect(message.total_tokens).toBe(100);
      expect(message.stop_reason).toBe('stop');
      expect(message.error).toBe(false);
    });
  });

  describe('Message rendering', () => {
    it('should render user message', () => {
      const message = user('Test message');
      const element = message.render();
      
      expect(element.className).toContain('user-message');
      expect(element.textContent).toContain('User:');
      expect(element.textContent).toContain('Test message');
    });

    it('should render pinned message', () => {
      const message = user('Pinned message');
      message.pinned = 1;
      const element = message.render();
      
      expect(element.classList.contains('pinned')).toBe(true);
      expect(element.dataset.pinned).toBe('1');
    });

    it('should hide hidden messages', () => {
      const message = user('Hidden message');
      message.hidden = true;
      const element = message.render();
      
      expect(element.style.display).toBe('none');
    });
  });
});

describe('Utility Functions', () => {
  it('should create messages using helper functions', () => {
    const userMsg = user('Question?');
    const assistantMsg = assistant('Answer.');
    const systemMsg = system('You are helpful.');
    
    expect(userMsg).toBeInstanceOf(UserMessage);
    expect(assistantMsg).toBeInstanceOf(AssistantMessage);
    expect(systemMsg.role).toBe('system');
  });

  it('should create text content using helper', () => {
    const content = text('Hello', 'https://example.com');
    expect(content).toBeInstanceOf(TextContent);
    expect(content.sequence).toBe('Hello');
    expect(content.url).toBe('https://example.com');
  });
});
/**
 * Message model classes
 */

import { 
  Message as IMessage,
  UserMessage as IUserMessage,
  AssistantMessage as IAssistantMessage,
  SystemMessage as ISystemMessage,
  Content as IContent,
  MessageModel,
  ContentModel
} from '../types';
import { Content } from './Content';

export abstract class Message implements IMessage {
  role: 'user' | 'assistant' | 'system' | 'developer';
  message: Content[];
  pinned: number;
  prompt_cached: boolean;
  hidden: boolean;

  constructor(data: Partial<IMessage>) {
    this.role = data.role || 'user';
    this.message = data.message?.map(m => m instanceof Content ? m : Content.fromJSON(m)) || [];
    this.pinned = data.pinned || 0;
    this.prompt_cached = data.prompt_cached || false;
    this.hidden = data.hidden || false;
  }

  abstract render(): HTMLElement;
  abstract getText(): string;

  static fromJSON(data: any): Message {
    const role = data.role?.toLowerCase();
    
    switch (role) {
      case 'user':
        return new UserMessage(data);
      case 'assistant':
        return new AssistantMessage(data);
      case 'system':
        return new SystemMessage(data);
      default:
        console.warn(`Unknown message role: ${role}`);
        return new UserMessage(data);
    }
  }

  static fromMessageModel(model: MessageModel): Message {
    const contents = model.content.map((c: ContentModel) => Content.fromJSON(c));
    
    const baseData = {
      message: contents,
      pinned: model.pinned,
      prompt_cached: model.prompt_cached,
    };

    switch (model.role) {
      case 'user':
        return new UserMessage(baseData);
      case 'assistant':
        return new AssistantMessage({
          ...baseData,
          total_tokens: model.total_tokens,
          underlying: model.underlying
        });
      case 'system':
        return new SystemMessage(baseData);
      default:
        return new UserMessage(baseData);
    }
  }

  toMessageModel(): MessageModel {
    return {
      role: this.role,
      content: this.message.map(m => ({
        sequence: m.sequence,
        content_type: m.content_type,
        url: m.url,
        original_sequence: m.original_sequence
      })),
      pinned: this.pinned,
      prompt_cached: this.prompt_cached,
      total_tokens: 0,
      underlying: undefined
    };
  }
}

export class UserMessage extends Message implements IUserMessage {
  role: 'user' = 'user';

  constructor(data: Partial<IUserMessage>) {
    super({
      ...data,
      role: 'user'
    });
  }

  render(): HTMLElement {
    const container = document.createElement('div');
    container.className = 'message user-message';
    
    const header = document.createElement('div');
    header.className = 'message-header';
    header.textContent = 'User:';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    this.message.forEach(item => {
      const rendered = item.render();
      if (rendered instanceof HTMLElement) {
        content.appendChild(rendered);
      } else {
        const div = document.createElement('div');
        div.innerHTML = rendered;
        content.appendChild(div);
      }
    });
    
    container.appendChild(header);
    container.appendChild(content);
    
    if (this.pinned !== 0) {
      container.classList.add('pinned');
      container.dataset.pinned = this.pinned.toString();
    }
    
    if (this.hidden) {
      container.style.display = 'none';
    }
    
    return container;
  }

  getText(): string {
    return this.message.map(m => m.getText()).join('\\n');
  }
}

export class AssistantMessage extends Message implements IAssistantMessage {
  role: 'assistant' = 'assistant';
  error?: boolean;
  thinking?: Content;
  system_context?: any;
  stop_reason?: string;
  stop_token?: string;
  total_tokens?: number;
  underlying?: any;

  constructor(data: Partial<IAssistantMessage>) {
    super({
      ...data,
      role: 'assistant'
    });
    
    this.error = data.error;
    this.thinking = data.thinking ? (data.thinking instanceof Content ? data.thinking : Content.fromJSON(data.thinking)) : undefined;
    this.system_context = data.system_context;
    this.stop_reason = data.stop_reason;
    this.stop_token = data.stop_token;
    this.total_tokens = data.total_tokens;
    this.underlying = data.underlying;
  }

  render(): HTMLElement {
    const container = document.createElement('div');
    container.className = 'message assistant-message';
    
    if (this.error) {
      container.classList.add('error');
    }
    
    const header = document.createElement('div');
    header.className = 'message-header';
    header.textContent = 'Assistant:';
    
    if (this.total_tokens) {
      const tokens = document.createElement('span');
      tokens.className = 'token-count';
      tokens.textContent = ` (${this.total_tokens} tokens)`;
      header.appendChild(tokens);
    }
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    // Render thinking content if present
    if (this.thinking) {
      const thinkingContainer = document.createElement('details');
      thinkingContainer.className = 'thinking-content';
      
      const summary = document.createElement('summary');
      summary.textContent = 'Thinking...';
      thinkingContainer.appendChild(summary);
      
      const thinkingContent = this.thinking.render();
      if (thinkingContent instanceof HTMLElement) {
        thinkingContainer.appendChild(thinkingContent);
      } else {
        const div = document.createElement('div');
        div.innerHTML = thinkingContent;
        thinkingContainer.appendChild(div);
      }
      
      content.appendChild(thinkingContainer);
    }
    
    // Render main message content
    this.message.forEach(item => {
      const rendered = item.render();
      if (rendered instanceof HTMLElement) {
        content.appendChild(rendered);
      } else {
        const div = document.createElement('div');
        div.innerHTML = rendered;
        content.appendChild(div);
      }
    });
    
    // Add metadata if present
    if (this.stop_reason || this.stop_token) {
      const meta = document.createElement('div');
      meta.className = 'message-meta';
      if (this.stop_reason) {
        meta.innerHTML += `<span class="stop-reason">Stop reason: ${this.stop_reason}</span>`;
      }
      if (this.stop_token) {
        meta.innerHTML += `<span class="stop-token">Stop token: ${this.stop_token}</span>`;
      }
      content.appendChild(meta);
    }
    
    container.appendChild(header);
    container.appendChild(content);
    
    if (this.pinned !== 0) {
      container.classList.add('pinned');
      container.dataset.pinned = this.pinned.toString();
    }
    
    if (this.hidden) {
      container.style.display = 'none';
    }
    
    return container;
  }

  getText(): string {
    return this.message.map(m => m.getText()).join('\\n');
  }

  toMessageModel(): MessageModel {
    const base = super.toMessageModel();
    return {
      ...base,
      total_tokens: this.total_tokens || 0,
      underlying: this.underlying
    };
  }
}

export class SystemMessage extends Message implements ISystemMessage {
  role: 'system' = 'system';

  constructor(data: Partial<ISystemMessage>) {
    super({
      ...data,
      role: 'system'
    });
  }

  render(): HTMLElement {
    const container = document.createElement('div');
    container.className = 'message system-message';
    
    const header = document.createElement('div');
    header.className = 'message-header';
    header.textContent = 'System:';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    this.message.forEach(item => {
      const rendered = item.render();
      if (rendered instanceof HTMLElement) {
        content.appendChild(rendered);
      } else {
        const div = document.createElement('div');
        div.innerHTML = rendered;
        content.appendChild(div);
      }
    });
    
    container.appendChild(header);
    container.appendChild(content);
    
    if (this.pinned !== 0) {
      container.classList.add('pinned');
      container.dataset.pinned = this.pinned.toString();
    }
    
    if (this.hidden) {
      container.style.display = 'none';
    }
    
    return container;
  }

  getText(): string {
    return this.message.map(m => m.getText()).join('\\n');
  }
}
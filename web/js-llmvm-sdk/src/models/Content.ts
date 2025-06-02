/**
 * Content model classes with render methods
 */

import { 
  Content as IContent,
  TextContent as ITextContent,
  ImageContent as IImageContent,
  PdfContent as IPdfContent,
  FileContent as IFileContent,
  BrowserContent as IBrowserContent,
  MarkdownContent as IMarkdownContent,
  HTMLContent as IHTMLContent,
  SearchResult as ISearchResult,
  YelpResult as IYelpResult,
  HackerNewsResult as IHackerNewsResult
} from '../types';

export abstract class Content implements IContent {
  type: string;
  sequence: string | Uint8Array | Content[];
  content_type: string;
  url: string;
  original_sequence?: any;

  constructor(data: IContent) {
    this.type = data.type;
    this.sequence = data.sequence;
    this.content_type = data.content_type;
    this.url = data.url;
    this.original_sequence = data.original_sequence;
  }

  abstract render(): HTMLElement | string;
  abstract getText(): string;

  static fromJSON(data: any): Content {
    const contentType = data.content_type || data.type;
    
    // Handle ImageContent with bytes array
    if ((contentType === 'image' || contentType === 'ImageContent') && data.sequence) {
      // If sequence is an array of numbers (bytes), convert to base64
      if (Array.isArray(data.sequence) && typeof data.sequence[0] === 'number') {
        const bytes = new Uint8Array(data.sequence);
        const binary = Array.from(bytes).map(byte => String.fromCharCode(byte)).join('');
        data.sequence = btoa(binary);
      }
    }
    
    switch (contentType) {
      case 'text':
      case 'TextContent':
        return new TextContent(data);
      case 'image':
      case 'ImageContent':
        return new ImageContent(data);
      case 'pdf':
      case 'PdfContent':
        return new PdfContent(data);
      case 'file':
      case 'FileContent':
        return new FileContent(data);
      case 'browser':
      case 'BrowserContent':
        return new BrowserContent(data);
      case 'markdown':
      case 'MarkdownContent':
        return new MarkdownContent(data);
      case 'html':
      case 'HTMLContent':
        return new HTMLContent(data);
      case 'SearchResult':
        return new SearchResult(data);
      case 'YelpResult':
        return new YelpResult(data);
      case 'HackerNewsResult':
        return new HackerNewsResult(data);
      default:
        console.warn(`Unknown content type: ${contentType}`);
        return new TextContent({ ...data, content_type: 'text' });
    }
  }
}

export class TextContent extends Content implements ITextContent {
  type: 'TextContent' = 'TextContent';
  sequence: string;
  content_type: 'text' = 'text';

  constructor(data: Partial<ITextContent>) {
    super({
      type: 'TextContent',
      content_type: 'text',
      url: '',
      ...data,
      sequence: data.sequence || ''
    });
    this.sequence = data.sequence || '';
  }

  render(): HTMLElement {
    const div = document.createElement('div');
    div.className = 'text-content';
    
    // Convert text to paragraphs
    const paragraphs = this.sequence.split('\\n\\n');
    paragraphs.forEach(para => {
      if (para.trim()) {
        const p = document.createElement('p');
        p.textContent = para.trim();
        div.appendChild(p);
      }
    });
    
    return div;
  }

  getText(): string {
    return this.sequence;
  }
}

export class ImageContent extends Content implements IImageContent {
  type: 'ImageContent' = 'ImageContent';
  sequence: string;
  content_type: 'image' = 'image';
  image_type?: string;

  constructor(data: Partial<IImageContent>) {
    super({
      type: 'ImageContent',
      content_type: 'image',
      url: '',
      ...data,
      sequence: data.sequence || ''
    });
    this.sequence = data.sequence || '';
    this.image_type = data.image_type;
  }

  render(): HTMLElement {
    const container = document.createElement('div');
    container.className = 'image-content';
    
    const img = document.createElement('img');
    img.src = `data:${this.image_type || 'image/png'};base64,${this.sequence}`;
    img.alt = this.url || 'Image';
    img.style.maxWidth = '100%';
    img.style.height = 'auto';
    
    if (this.url) {
      const caption = document.createElement('p');
      caption.className = 'image-caption';
      caption.textContent = this.url;
      container.appendChild(caption);
    }
    
    container.appendChild(img);
    return container;
  }

  getText(): string {
    return `[Image: ${this.url || 'embedded image'}]`;
  }
}

export class PdfContent extends Content implements IPdfContent {
  type: 'PdfContent' = 'PdfContent';
  sequence: string;
  content_type: 'pdf' = 'pdf';

  constructor(data: Partial<IPdfContent>) {
    super({
      type: 'PdfContent',
      content_type: 'pdf',
      url: '',
      ...data,
      sequence: data.sequence || ''
    });
    this.sequence = data.sequence || '';
  }

  render(): HTMLElement {
    const container = document.createElement('div');
    container.className = 'pdf-content';
    
    // Create a download link or embed viewer
    if (this.sequence) {
      const blob = new Blob([Uint8Array.from(atob(this.sequence), c => c.charCodeAt(0))], { type: 'application/pdf' });
      const url = URL.createObjectURL(blob);
      
      const embed = document.createElement('embed');
      embed.src = url;
      embed.type = 'application/pdf';
      embed.width = '100%';
      embed.height = '600px';
      
      container.appendChild(embed);
    } else {
      const link = document.createElement('a');
      link.href = this.url;
      link.textContent = `PDF: ${this.url}`;
      link.target = '_blank';
      container.appendChild(link);
    }
    
    return container;
  }

  getText(): string {
    return `[PDF: ${this.url}]`;
  }
}

export class FileContent extends Content implements IFileContent {
  type: 'FileContent' = 'FileContent';
  sequence: string;
  content_type: 'file' = 'file';

  constructor(data: Partial<IFileContent>) {
    super({
      type: 'FileContent',
      content_type: 'file',
      url: '',
      ...data,
      sequence: data.sequence || ''
    });
    this.sequence = data.sequence || '';
  }

  render(): HTMLElement {
    const container = document.createElement('div');
    container.className = 'file-content';
    
    // Try to decode and display as text
    try {
      const text = atob(this.sequence);
      const pre = document.createElement('pre');
      const code = document.createElement('code');
      code.textContent = text;
      pre.appendChild(code);
      
      if (this.url) {
        const header = document.createElement('div');
        header.className = 'file-header';
        header.textContent = `File: ${this.url}`;
        container.appendChild(header);
      }
      
      container.appendChild(pre);
    } catch (e) {
      // If not base64 or can't decode, show as download
      const link = document.createElement('a');
      link.href = this.url;
      link.textContent = `Download: ${this.url}`;
      link.download = this.url.split('/').pop() || 'file';
      container.appendChild(link);
    }
    
    return container;
  }

  getText(): string {
    try {
      return atob(this.sequence);
    } catch {
      return `[File: ${this.url}]`;
    }
  }
}

export class BrowserContent extends Content implements IBrowserContent {
  type: 'BrowserContent' = 'BrowserContent';
  sequence: Content[];
  content_type: 'browser' = 'browser';

  constructor(data: Partial<IBrowserContent>) {
    const sequence = Array.isArray(data.sequence) 
      ? data.sequence.map(item => item instanceof Content ? item : Content.fromJSON(item))
      : [];
    
    super({
      type: 'BrowserContent',
      content_type: 'browser',
      url: '',
      ...data,
      sequence
    });
    this.sequence = sequence;
  }

  render(): HTMLElement {
    const container = document.createElement('div');
    container.className = 'browser-content';
    
    if (this.url) {
      const header = document.createElement('div');
      header.className = 'browser-header';
      header.innerHTML = `<strong>URL:</strong> <a href="${this.url}" target="_blank">${this.url}</a>`;
      container.appendChild(header);
    }
    
    const contentContainer = document.createElement('div');
    contentContainer.className = 'browser-content-items';
    
    this.sequence.forEach(content => {
      const rendered = content.render();
      if (rendered instanceof HTMLElement) {
        contentContainer.appendChild(rendered);
      } else {
        const div = document.createElement('div');
        div.innerHTML = rendered;
        contentContainer.appendChild(div);
      }
    });
    
    container.appendChild(contentContainer);
    return container;
  }

  getText(): string {
    return this.sequence.map(c => c.getText()).join('\\n');
  }
}

export class MarkdownContent extends Content implements IMarkdownContent {
  type: 'MarkdownContent' = 'MarkdownContent';
  sequence: Content[];
  content_type: 'markdown' = 'markdown';

  constructor(data: Partial<IMarkdownContent>) {
    const sequence = Array.isArray(data.sequence) 
      ? data.sequence.map(item => item instanceof Content ? item : Content.fromJSON(item))
      : [];
    
    super({
      type: 'MarkdownContent',
      content_type: 'markdown',
      url: '',
      ...data,
      sequence
    });
    this.sequence = sequence;
  }

  render(): HTMLElement {
    const container = document.createElement('div');
    container.className = 'markdown-content';
    
    // Combine all text content and render as markdown
    const markdownText = this.sequence.map(c => c.getText()).join('\\n');
    
    // Basic markdown to HTML conversion (in production, use a proper markdown parser)
    container.innerHTML = this.convertMarkdownToHTML(markdownText);
    
    return container;
  }

  private convertMarkdownToHTML(markdown: string): string {
    // This is a very basic implementation. In production, use a library like marked.js
    let html = markdown
      // Headers
      .replace(/^### (.*$)/gim, '<h3>$1</h3>')
      .replace(/^## (.*$)/gim, '<h2>$1</h2>')
      .replace(/^# (.*$)/gim, '<h1>$1</h1>')
      // Bold
      .replace(/\*\*(.*)\*\*/g, '<strong>$1</strong>')
      // Italic
      .replace(/\*(.*)\*/g, '<em>$1</em>')
      // Links
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
      // Line breaks
      .replace(/\n/g, '<br>')
      // Code blocks
      .replace(/```([^`]+)```/g, '<pre><code>$1</code></pre>')
      // Inline code
      .replace(/`([^`]+)`/g, '<code>$1</code>');
    
    return html;
  }

  getText(): string {
    return this.sequence.map(c => c.getText()).join('\\n');
  }
}

export class HTMLContent extends Content implements IHTMLContent {
  type: 'HTMLContent' = 'HTMLContent';
  sequence: Content[] | string;
  content_type: 'html' = 'html';

  constructor(data: Partial<IHTMLContent>) {
    const sequence = typeof data.sequence === 'string' 
      ? data.sequence 
      : Array.isArray(data.sequence) 
        ? data.sequence.map(item => item instanceof Content ? item : Content.fromJSON(item))
        : [];
    
    super({
      type: 'HTMLContent',
      content_type: 'html',
      url: '',
      ...data,
      sequence
    });
    this.sequence = sequence;
  }

  render(): HTMLElement {
    const container = document.createElement('div');
    container.className = 'html-content';
    
    if (typeof this.sequence === 'string') {
      // Sanitize HTML before inserting (in production, use a library like DOMPurify)
      container.innerHTML = this.sequence;
    } else {
      this.sequence.forEach(content => {
        const rendered = content.render();
        if (rendered instanceof HTMLElement) {
          container.appendChild(rendered);
        } else {
          const div = document.createElement('div');
          div.innerHTML = rendered;
          container.appendChild(div);
        }
      });
    }
    
    return container;
  }

  getText(): string {
    if (typeof this.sequence === 'string') {
      // Strip HTML tags
      const div = document.createElement('div');
      div.innerHTML = this.sequence;
      return div.textContent || div.innerText || '';
    }
    return this.sequence.map(c => c.getText()).join('\\n');
  }
}

export class SearchResult extends TextContent implements ISearchResult {
  title: string;
  snippet: string;
  engine: string;

  constructor(data: Partial<ISearchResult>) {
    super(data);
    this.title = data.title || '';
    this.snippet = data.snippet || '';
    this.engine = data.engine || '';
  }

  render(): HTMLElement {
    const container = document.createElement('div');
    container.className = 'search-result';
    
    const titleLink = document.createElement('a');
    titleLink.href = this.url;
    titleLink.target = '_blank';
    titleLink.className = 'search-result-title';
    titleLink.textContent = this.title;
    
    const snippet = document.createElement('p');
    snippet.className = 'search-result-snippet';
    snippet.textContent = this.snippet;
    
    const meta = document.createElement('div');
    meta.className = 'search-result-meta';
    meta.textContent = `Source: ${this.engine}`;
    
    container.appendChild(titleLink);
    container.appendChild(snippet);
    container.appendChild(meta);
    
    return container;
  }

  getText(): string {
    return `${this.title}\\n${this.snippet}\\n${this.url}`;
  }
}

export class YelpResult extends TextContent implements IYelpResult {
  title: string;
  link: string;
  neighborhood: string;
  snippet: string;
  reviews: string;

  constructor(data: Partial<IYelpResult>) {
    super(data);
    this.title = data.title || '';
    this.link = data.link || data.url || '';
    this.neighborhood = data.neighborhood || '';
    this.snippet = data.snippet || '';
    this.reviews = data.reviews || '';
  }

  render(): HTMLElement {
    const container = document.createElement('div');
    container.className = 'yelp-result';
    
    const titleLink = document.createElement('a');
    titleLink.href = this.link;
    titleLink.target = '_blank';
    titleLink.className = 'yelp-result-title';
    titleLink.textContent = this.title;
    
    const neighborhood = document.createElement('div');
    neighborhood.className = 'yelp-result-neighborhood';
    neighborhood.textContent = this.neighborhood;
    
    const snippet = document.createElement('p');
    snippet.className = 'yelp-result-snippet';
    snippet.textContent = this.snippet;
    
    if (this.reviews) {
      const reviewsDiv = document.createElement('div');
      reviewsDiv.className = 'yelp-result-reviews';
      reviewsDiv.innerHTML = '<strong>Reviews:</strong>';
      const reviewText = document.createElement('p');
      reviewText.textContent = this.reviews.substring(0, 200) + '...';
      reviewsDiv.appendChild(reviewText);
      container.appendChild(reviewsDiv);
    }
    
    container.appendChild(titleLink);
    container.appendChild(neighborhood);
    container.appendChild(snippet);
    
    return container;
  }

  getText(): string {
    return `${this.title}\\n${this.neighborhood}\\n${this.snippet}\\n${this.reviews}`;
  }
}

export class HackerNewsResult extends TextContent implements IHackerNewsResult {
  title: string;
  author: string;
  comment_text: string;
  created_at: string;

  constructor(data: Partial<IHackerNewsResult>) {
    super(data);
    this.title = data.title || '';
    this.author = data.author || '';
    this.comment_text = data.comment_text || '';
    this.created_at = data.created_at || '';
  }

  render(): HTMLElement {
    const container = document.createElement('div');
    container.className = 'hn-result';
    
    if (this.title) {
      const titleLink = document.createElement('a');
      titleLink.href = this.url;
      titleLink.target = '_blank';
      titleLink.className = 'hn-result-title';
      titleLink.textContent = this.title;
      container.appendChild(titleLink);
    }
    
    const meta = document.createElement('div');
    meta.className = 'hn-result-meta';
    meta.textContent = `by ${this.author} on ${new Date(this.created_at).toLocaleDateString()}`;
    
    const comment = document.createElement('div');
    comment.className = 'hn-result-comment';
    comment.innerHTML = this.comment_text;
    
    container.appendChild(meta);
    container.appendChild(comment);
    
    return container;
  }

  getText(): string {
    return `${this.title}\\nby ${this.author}\\n${this.comment_text}`;
  }
}
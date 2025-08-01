import React, { useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { FileText, Image as ImageIcon, Globe, FileCode, AlertCircle } from "lucide-react";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { HelperContentRenderer } from "./HelperContentRenderer";
import { hasHelperTags } from "@/utils/helperParser";

interface ContentRendererProps {
  content: any;
  type?: string;
  isStreaming?: boolean;
  images?: Array<{id: string, data: string}>;
}

export const ContentRenderer = ({ content, type, isStreaming = false, images }: ContentRendererProps) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // If content has a render method (from LLMVM SDK), use it
    if (content && typeof content.render === 'function' && containerRef.current) {
      try {
        const element = content.render();
        containerRef.current.innerHTML = '';
        if (typeof element === 'string') {
          containerRef.current.innerHTML = element;
        } else {
          containerRef.current.appendChild(element);
        }

        // Add some default styles for LLMVM rendered content
        const existingStyle = containerRef.current.querySelector('style[data-llmvm-styles]');
        if (!existingStyle) {
          const style = document.createElement('style');
          style.setAttribute('data-llmvm-styles', 'true');
          style.textContent = `
            .llmvm-content { max-width: 100%; overflow-wrap: break-word; word-break: break-word; }
            .llmvm-content img { max-width: 100%; height: auto; }
            .llmvm-content pre { background: #f3f4f6; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; max-width: 100%; }
            .llmvm-content code { background: #e5e7eb; padding: 0.125rem 0.25rem; border-radius: 0.25rem; word-break: break-all; }
            .llmvm-browser-content { border: 1px solid #e5e7eb; border-radius: 0.5rem; padding: 1rem; margin: 0.5rem 0; max-width: 100%; }
            .llmvm-pdf-content { border: 1px solid #e5e7eb; border-radius: 0.5rem; padding: 1rem; max-width: 100%; }
            .llmvm-search-result { border: 1px solid #e5e7eb; border-radius: 0.5rem; padding: 1rem; margin: 0.5rem 0; max-width: 100%; }
            .llmvm-content p, .llmvm-content li { overflow-wrap: anywhere; }
          `;
          containerRef.current.appendChild(style);
        }
      } catch (error) {
        console.error('Failed to render LLMVM content:', error);
      }
    }
  }, [content]);

  // If content is a string, check for helper tags first
  if (typeof content === 'string') {
    if (hasHelperTags(content)) {
      return <HelperContentRenderer content={content} isStreaming={isStreaming} images={images} />;
    }
    
    // Replace image placeholders with actual images
    let processedContent = content;
    if (images && images.length > 0) {
      const parts: React.ReactNode[] = [];
      let lastIndex = 0;
      
      // Find all image placeholders
      const imageRegex = /\[IMAGE:([^\]]+)\]/g;
      let match;
      
      while ((match = imageRegex.exec(content)) !== null) {
        // Add text before the image
        if (match.index > lastIndex) {
          parts.push(
            <MarkdownRenderer 
              key={`text-${lastIndex}`} 
              content={content.substring(lastIndex, match.index)} 
              className="max-w-full"
            />
          );
        }
        
        // Find the corresponding image
        const imageId = match[1];
        const imageData = images.find(img => img.id === imageId);
        
        if (imageData) {
          parts.push(
            <Card key={imageId} className="my-4 p-4">
              <div className="flex items-center gap-2 mb-2 text-sm text-gray-600">
                <ImageIcon size={16} />
                <span>Image</span>
              </div>
              <img
                src={`data:image/png;base64,${imageData.data}`}
                alt="Generated content"
                className="max-w-full h-auto rounded"
              />
            </Card>
          );
        }
        
        lastIndex = match.index + match[0].length;
      }
      
      // Add any remaining text
      if (lastIndex < content.length) {
        parts.push(
          <MarkdownRenderer 
            key={`text-${lastIndex}`} 
            content={content.substring(lastIndex)} 
            className="max-w-full"
          />
        );
      }
      
      if (parts.length > 0) {
        return <div className="space-y-2 max-w-full">{parts}</div>;
      }
    }
    
    return <MarkdownRenderer content={content} className="max-w-full" />;
  }

  // If content is an array (multiple content items)
  if (Array.isArray(content)) {
    // First, check if the array contains helper tags when concatenated
    const concatenatedText = content.map(item => {
      if (typeof item === 'string') return item;
      if (item && typeof item === 'object') {
        if (item.text) return item.text;
        if (item.sequence) return item.sequence;
        if (item.content) return item.content;
      }
      return '';
    }).join('');
    
    // If the concatenated content has helper tags, render it as a single block
    if (hasHelperTags(concatenatedText)) {
      return <HelperContentRenderer content={concatenatedText} isStreaming={isStreaming} images={images} />;
    }
    
    // Otherwise, render each item separately
    return (
      <div className="space-y-4 max-w-full">
        {content.map((item, index) => (
          <ContentRenderer key={index} content={item} isStreaming={isStreaming} images={images} />
        ))}
      </div>
    );
  }

  // If content has LLMVM render method BUT it's not TextContent (which we want to render as markdown)
  if (content && typeof content.render === 'function' && content.type !== 'TextContent' && content.content_type !== 'text') {
    return <div ref={containerRef} className="llmvm-content max-w-full" />;
  }

  // Handle specific content types based on structure
  if (content && typeof content === 'object') {
    // Handle TextContent objects or any text-like content
    if (content.type === 'TextContent' || content.type === 'text' || content.content_type === 'text' || content.text || content.sequence) {
      let textContent = '';
      if (typeof content.getText === 'function') {
        textContent = content.getText();
      } else if (typeof content === 'string') {
        textContent = content;
      } else {
        textContent = content.text || content.sequence || content.content || '';
      }
      
      // Check for helper tags
      if (hasHelperTags(textContent)) {
        return <HelperContentRenderer content={textContent} isStreaming={isStreaming} images={images} />;
      }
      return <MarkdownRenderer content={textContent} className="max-w-full" />;
    }

    // Image content
    if (content.type === 'image' || content.type === 'ImageContent' || content.content_type === 'image' || content.image) {
      let imageData = content.image || content.data || content.sequence;
      
      // Handle Uint8Array or array of numbers (bytes)
      if (imageData && (imageData instanceof Uint8Array || (Array.isArray(imageData) && typeof imageData[0] === 'number'))) {
        // Convert bytes to base64
        const bytes = imageData instanceof Uint8Array ? imageData : new Uint8Array(imageData);
        const binary = Array.from(bytes).map(byte => String.fromCharCode(byte)).join('');
        imageData = btoa(binary);
      }
      
      if (imageData && typeof imageData === 'string') {
        return (
          <Card className="p-4">
            <div className="flex items-center gap-2 mb-2 text-sm text-gray-600">
              <ImageIcon size={16} />
              <span>Image</span>
            </div>
            <img
              src={imageData.startsWith('data:') ? imageData : `data:image/png;base64,${imageData}`}
              alt="Generated content"
              className="w-full max-w-full h-auto rounded object-contain"
            />
          </Card>
        );
      }
    }

    // FileContent type (must check before browser content since it also has url)
    if (content.type === 'FileContent' || content.content_type === 'file') {
      return (
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2 text-sm text-gray-600">
            <FileText size={16} />
            <span>File Content: {content.url || 'Unknown file'}</span>
          </div>
          {/* Don't render the base64 content for files */}
        </Card>
      );
    }

    // Browser content
    if (content.type === 'browser' || content.url) {
      return (
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2 text-sm text-gray-600">
            <Globe size={16} />
            <span>Web Content: {content.url || 'Browser'}</span>
          </div>
          {content.text && (
            <div className="text-gray-700 whitespace-pre-wrap break-words overflow-wrap-anywhere">
              {content.text}
            </div>
          )}
          {content.sequence && typeof content.sequence === 'string' && content.type !== 'FileContent' && (
            <div className="mt-4 space-y-2 max-w-full">
              <ContentRenderer content={content.sequence} isStreaming={isStreaming} />
            </div>
          )}
        </Card>
      );
    }

    // PDF content
    if (content.type === 'pdf' || content.pdf) {
      return (
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2 text-sm text-gray-600">
            <FileText size={16} />
            <span>PDF Document</span>
          </div>
          {content.text && (
            <ScrollArea className="h-96 max-w-full">
              <div className="text-gray-700 whitespace-pre-wrap break-words overflow-wrap-anywhere">
                {content.text}
              </div>
            </ScrollArea>
          )}
        </Card>
      );
    }

    // File content
    if (content.type === 'file' || content.path) {
      return (
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2 text-sm text-gray-600">
            <FileCode size={16} />
            <span>File: {content.path || 'Unknown'}</span>
          </div>
          {content.text && (
            <pre className="bg-gray-100 p-4 rounded overflow-x-auto max-w-full">
              <code className="break-all">{content.text}</code>
            </pre>
          )}
        </Card>
      );
    }

    // Search results
    if (content.type === 'search_result' || content.results) {
      return (
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2 text-sm text-gray-600">
            <Globe size={16} />
            <span>Search Results</span>
          </div>
          <div className="space-y-2">
            {(content.results || [content]).map((result: any, index: number) => (
              <div key={index} className="border-l-2 border-blue-500 pl-4">
                <h4 className="font-medium text-blue-600 break-words">{result.title}</h4>
                {result.url && (
                  <a href={result.url} target="_blank" rel="noopener noreferrer"
                     className="text-xs text-gray-500 hover:underline break-all">
                    {result.url}
                  </a>
                )}
                {result.snippet && (
                  <p className="text-sm text-gray-700 mt-1 break-words">{result.snippet}</p>
                )}
              </div>
            ))}
          </div>
        </Card>
      );
    }

    // Code/JSON content
    if (content.code || content.json) {
      return (
        <pre className="bg-gray-100 p-4 rounded overflow-x-auto max-w-full">
          <code className="break-all">{content.code || JSON.stringify(content.json, null, 2)}</code>
        </pre>
      );
    }

    // Generic object - display as JSON
    return (
      <Card className="p-4">
        <div className="flex items-center gap-2 mb-2 text-sm text-gray-600">
          <AlertCircle size={16} />
          <span>Data Object</span>
        </div>
        <pre className="bg-gray-100 p-4 rounded overflow-x-auto text-sm max-w-full">
          <code className="break-all">{JSON.stringify(content, null, 2)}</code>
        </pre>
      </Card>
    );
  }

  // Fallback
  return (
    <div className="text-gray-500 italic">
      Unable to render content
    </div>
  );
};
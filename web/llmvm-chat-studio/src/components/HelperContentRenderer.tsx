import React from 'react';
import { parseHelperTags, ParsedSegment } from '@/utils/helperParser';
import { MarkdownRenderer } from './MarkdownRenderer';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useEffect, useState } from 'react';
import { Card } from "@/components/ui/card";
import { Image as ImageIcon, Copy, Check } from "lucide-react";
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

interface HelperContentRendererProps {
  content: string;
  isStreaming?: boolean;
  images?: Array<{id: string, data: string}>;
}

export const HelperContentRenderer = ({ content, isStreaming = false, images }: HelperContentRendererProps) => {
  const [isDark, setIsDark] = useState(false);
  const [copiedCode, setCopiedCode] = useState<string | null>(null);
  const { toast } = useToast();

  const copyToClipboard = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(code);
    toast({
      title: "Copied to clipboard",
      description: "Code has been copied.",
    });
    setTimeout(() => setCopiedCode(null), 2000);
  };

  useEffect(() => {
    // Check if dark mode is active
    const checkDarkMode = () => {
      setIsDark(document.documentElement.classList.contains('dark'));
    };

    checkDarkMode();

    // Watch for theme changes
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    });

    return () => observer.disconnect();
  }, []);

  const segments = parseHelperTags(content);

  // Function to render content with image placeholders
  const renderContentWithImages = (text: string) => {
    if (!images || images.length === 0) {
      return <MarkdownRenderer content={text} />;
    }

    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    
    // Find all image placeholders
    const imageRegex = /\[IMAGE:([^\]]+)\]/g;
    let match;
    
    while ((match = imageRegex.exec(text)) !== null) {
      // Add text before the image
      if (match.index > lastIndex) {
        parts.push(
          <MarkdownRenderer 
            key={`text-${lastIndex}`} 
            content={text.substring(lastIndex, match.index)} 
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
    if (lastIndex < text.length) {
      parts.push(
        <MarkdownRenderer 
          key={`text-${lastIndex}`} 
          content={text.substring(lastIndex)} 
        />
      );
    }
    
    return parts.length > 0 ? <>{parts}</> : <MarkdownRenderer content={text} />;
  };

  const renderSegment = (segment: ParsedSegment, index: number) => {
    switch (segment.type) {
      case 'helpers_block':
        // Extract content without tags for syntax highlighting
        const codeMatch = segment.content.match(/<helpers>([\s\S]*?)(?:<\/helpers>)?$/);
        const codeContent = codeMatch ? codeMatch[1] : segment.content.replace('<helpers>', '');
        const isCopied = copiedCode === codeContent;

        return (
          <div key={index} className="my-3">
            <div className="text-xs text-gray-500 dark:text-gray-400 mb-1 font-medium">Helper Code:</div>
            <div className="relative overflow-visible group">
              <code className="absolute top-2 left-2 text-xs bg-gray-200 dark:bg-gray-700 px-1 py-0.5 rounded z-10">
                {'<helpers>'}
              </code>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => copyToClipboard(codeContent)}
                className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity z-20 h-8 px-2"
              >
                {isCopied ? <Check size={14} /> : <Copy size={14} />}
              </Button>
              <SyntaxHighlighter
                language="python"
                style={isDark ? oneDark : oneLight}
                customStyle={{
                  margin: 0,
                  borderRadius: '0.5rem',
                  fontSize: '0.9375rem',
                  paddingTop: '1rem',
                  paddingBottom: '2.5rem',
                  paddingRight: '3rem', // Make room for copy button
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  overflowWrap: 'break-word',
                }}
                codeTagProps={{
                  style: {
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }
                }}
              >
                {codeContent}
              </SyntaxHighlighter>
              {segment.isComplete && (
                <code className="absolute bottom-2 left-2 text-xs bg-gray-200 dark:bg-gray-700 px-1 py-0.5 rounded z-10">
                  {'</helpers>'}
                </code>
              )}
            </div>
            {!segment.isComplete && isStreaming && (
              <div className="text-xs text-yellow-600 dark:text-yellow-400 mt-1">
                ⚡ Streaming...
              </div>
            )}
          </div>
        );

      case 'result_block':
        // Extract content without tags for markdown rendering
        const resultMatch = segment.content.match(/<helpers_result>([\s\S]*?)(?:<\/helpers_result>)?$/);
        const resultContent = resultMatch ? resultMatch[1] : segment.content.replace('<helpers_result>', '');

        return (
          <div key={index} className="my-3">
            <div className="text-xs text-gray-500 dark:text-gray-400 mb-1 font-medium">Helper Result:</div>
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3 relative overflow-visible">
              <code className="absolute top-2 left-2 text-xs bg-blue-100 dark:bg-blue-800 px-1 py-0.5 rounded z-10">
                {'<helpers_result>'}
              </code>
              <div className="pt-6 pb-6">
                {renderContentWithImages(resultContent)}
              </div>
              {segment.isComplete && (
                <code className="absolute bottom-2 left-2 text-xs bg-blue-100 dark:bg-blue-800 px-1 py-0.5 rounded z-10">
                  {'</helpers_result>'}
                </code>
              )}
            </div>
            {!segment.isComplete && isStreaming && (
              <div className="text-xs text-yellow-600 dark:text-yellow-400 mt-1">
                ⚡ Streaming...
              </div>
            )}
          </div>
        );

      case 'text':
        // For text segments, render as markdown with image support
        if (segment.content.trim().length > 0) {
          return (
            <div key={index}>
              {renderContentWithImages(segment.content)}
            </div>
          );
        }
        return null;

      default:
        return null;
    }
  };

  return (
    <div className="helper-content">
      {segments.map((segment, index) => renderSegment(segment, index))}
    </div>
  );
};
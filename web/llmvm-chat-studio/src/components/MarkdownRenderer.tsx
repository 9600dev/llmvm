import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { ReactNode, useEffect, useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export const MarkdownRenderer = ({ content, className = "" }: MarkdownRendererProps) => {
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

  // Handle empty content
  if (!content) {
    return <div className="text-gray-400">...</div>;
  }

  return (
    <div className={`prose max-w-none prose-gray dark:prose-invert overflow-hidden ${className}`}>
      <ReactMarkdown 
        remarkPlugins={[remarkGfm]}
        components={{
          // Override code blocks for syntax highlighting
          code({ node, inline, className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '');
            const language = match ? match[1] : '';
            
            // For code blocks (not inline)
            if (!inline && (match || node?.position)) {
              const codeString = String(children).replace(/\n$/, '');
              const isCopied = copiedCode === codeString;
              
              return (
                <div className="relative group my-3">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(codeString)}
                    className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity z-10 h-8 px-2"
                  >
                    {isCopied ? <Check size={14} /> : <Copy size={14} />}
                  </Button>
                  <SyntaxHighlighter
                    language={language || 'text'}
                    style={isDark ? oneDark : oneLight}
                    customStyle={{
                      margin: 0,
                      borderRadius: '0.5rem',
                      fontSize: '0.9375rem',
                      paddingRight: '3rem', // Make room for copy button
                    }}
                    {...props}
                  >
                    {codeString}
                  </SyntaxHighlighter>
                </div>
              );
            }
            
            // For inline code
            return (
              <code className="bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 px-1.5 py-0.5 rounded text-sm font-mono" {...props}>
                {children}
              </code>
            );
          },
          // Handle pre blocks that don't have a language
          pre({ children, ...props }: any) {
            // If the pre block contains a code block with language, it will be handled by the code component
            // This is for pre blocks without language specification
            if (children?.props?.className?.includes('language-')) {
              return <>{children}</>;
            }
            
            return (
              <pre className="bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 overflow-x-auto my-3" {...props}>
                {children}
              </pre>
            );
          },
          // Handle tables to prevent overflow
          table({ children, ...props }: any) {
            return (
              <div className="overflow-x-auto my-4">
                <table className="min-w-full" {...props}>
                  {children}
                </table>
              </div>
            );
          },
          // Handle long links and code
          a({ children, ...props }: any) {
            return (
              <a className="break-words" {...props}>
                {children}
              </a>
            );
          },
          p({ children, ...props }: any) {
            return (
              <p className="break-words" {...props}>
                {children}
              </p>
            );
          }
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};
import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send, Paperclip, Image, Code, Settings, Lasso } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import ThreadSettingsDialog, { ThreadSettings } from "./ThreadSettingsDialog";
import DrawingTool from "./DrawingTool";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface MessageInputProps {
  onSend: (content: string, files?: File[]) => void;
  settings?: ThreadSettings;
  onSettingsChange?: (settings: ThreadSettings) => void;
  onPythonExecute?: (code: string) => void;
}

const MessageInput = ({ onSend, settings, onSettingsChange, onPythonExecute }: MessageInputProps) => {
  const [message, setMessage] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [isPythonMode, setIsPythonMode] = useState(false);
  const [isDrawingMode, setIsDrawingMode] = useState(false);
  const [capturedImages, setCapturedImages] = useState<string[]>([]);
  const [showCaptureDialog, setShowCaptureDialog] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleSend = () => {
    if (!message.trim() && files.length === 0 && capturedImages.length === 0) return;

    // Convert captured images to files
    const imageFiles = capturedImages.map((dataUrl, index) => {
      const arr = dataUrl.split(',');
      const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/png';
      const bstr = atob(arr[1]);
      let n = bstr.length;
      const u8arr = new Uint8Array(n);
      while(n--) {
        u8arr[n] = bstr.charCodeAt(n);
      }
      return new File([u8arr], `captured-${Date.now()}-${index}.png`, { type: mime });
    });

    const allFiles = [...files, ...imageFiles];

    if (isPythonMode && onPythonExecute) {
      onPythonExecute(message);
    } else {
      onSend(message, allFiles);
    }
    setMessage("");
    setFiles([]);
    setCapturedImages([]);
    setShowCaptureDialog(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !showCaptureDialog) {
      e.preventDefault();
      handleSend();
    }
  };

  const handlePaste = async (e: React.ClipboardEvent) => {
    const items = e.clipboardData.items;
    const pastedFiles: File[] = [];

    for (let i = 0; i < items.length; i++) {
      const item = items[i];

      // Handle image paste
      if (item.type.startsWith('image/')) {
        e.preventDefault(); // Prevent default paste behavior for images
        const blob = item.getAsFile();
        if (blob) {
          // Create a file with a proper name
          const file = new File([blob], `pasted-image-${Date.now()}.${item.type.split('/')[1]}`, {
            type: item.type
          });
          pastedFiles.push(file);
        }
      }
      // Let text paste through naturally
    }

    if (pastedFiles.length > 0) {
      setFiles(prev => [...prev, ...pastedFiles]);
      toast({
        title: "Image pasted",
        description: `${pastedFiles.length} image(s) ready to send.`,
      });
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    setFiles(prev => [...prev, ...selectedFiles]);

    if (selectedFiles.length > 0) {
      toast({
        title: "Files added",
        description: `${selectedFiles.length} file(s) ready to upload.`,
      });
    }
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    setFiles(prev => [...prev, ...droppedFiles]);

    if (droppedFiles.length > 0) {
      toast({
        title: "Files dropped",
        description: `${droppedFiles.length} file(s) ready to upload.`,
      });
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrawingCapture = (images: string[]) => {
    setCapturedImages(images);
    setShowCaptureDialog(true);
    toast({
      title: "Images captured",
      description: `${images.length} element(s) captured. Add a message to send with the images.`,
    });
  };

  const handleCancelCapture = () => {
    setCapturedImages([]);
    setShowCaptureDialog(false);
  };

  // Close dialog and send when Enter is pressed in the dialog
  useEffect(() => {
    if (showCaptureDialog) {
      const handleDialogKeyPress = (e: KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          handleSend();
        }
      };

      window.addEventListener("keydown", handleDialogKeyPress);
      return () => window.removeEventListener("keydown", handleDialogKeyPress);
    }
  }, [showCaptureDialog, message, capturedImages]);

  return (
    <>
    <div className="p-4">
      {/* File attachments */}
      {files.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {files.map((file, index) => (
            <div
              key={index}
              className="flex items-center gap-2 bg-gray-200 px-3 py-1 rounded-lg text-sm"
            >
              {file.type.startsWith('image/') ? <Image size={16} /> : <Paperclip size={16} />}
              <span className="text-gray-700 truncate max-w-32">{file.name}</span>
              <button
                onClick={() => removeFile(index)}
                className="text-gray-500 hover:text-red-500 ml-1"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Input area */}
      <div
        className="relative bg-white rounded-lg border border-gray-300 focus-within:border-blue-500 transition-colors"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <Textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          onPaste={handlePaste}
          placeholder={isPythonMode 
            ? "Enter Python code... (Press Enter to execute, Shift+Enter for new line)" 
            : "Message LLMVM... (Press Enter to send, Shift+Enter for new line, Ctrl+V to paste images)"
          }
          className="bg-transparent border-0 resize-none min-h-[100px] max-h-[200px] text-gray-900 placeholder-gray-500 focus:ring-0 pr-24"
          style={{ scrollbarWidth: 'thin' }}
        />

        {/* Action buttons */}
        <div className="absolute bottom-2 right-2 flex items-center gap-1">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={handleFileSelect}
            accept="image/*,.pdf,.txt,.md,.csv,.zip,.js,.py,.html,.css,.json"
          />

          <Button
            variant="ghost"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            className="text-gray-500 hover:text-gray-700 h-8 w-8 p-0"
          >
            <Paperclip size={16} />
          </Button>

          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setIsDrawingMode(!isDrawingMode);
              if (!isDrawingMode) {
                toast({
                  title: "Drawing mode activated",
                  description: "Draw to lasso elements or underline text. Press Enter when done.",
                });
              }
            }}
            className={`h-8 w-8 p-0 ${
              isDrawingMode 
                ? "text-blue-600 hover:text-blue-700 bg-blue-100" 
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            <Lasso size={16} />
          </Button>

          {onPythonExecute && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setIsPythonMode(!isPythonMode);
                toast({
                  title: isPythonMode ? "Chat mode activated" : "Python mode activated",
                  description: isPythonMode ? "Messages will be sent as chat" : "Messages will be executed as Python code",
                });
              }}
              className={`h-8 w-8 p-0 ${
                isPythonMode 
                  ? "text-blue-600 hover:text-blue-700 bg-blue-100" 
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              <Code size={16} />
            </Button>
          )}

          {settings && onSettingsChange && (
            <ThreadSettingsDialog
              settings={settings}
              onSettingsChange={onSettingsChange}
              trigger={
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-gray-500 hover:text-gray-700 h-8 w-8 p-0"
                >
                  <Settings size={16} />
                </Button>
              }
            />
          )}

          <Button
            onClick={handleSend}
            disabled={!message.trim() && files.length === 0 && capturedImages.length === 0}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 h-8 w-8 p-0"
          >
            <Send size={16} />
          </Button>
        </div>
      </div>

      {/* Quick commands hint */}
      <div className="mt-2 text-xs text-gray-500">
        Try commands: <code className="bg-gray-100 px-1 rounded">/search</code>, <code className="bg-gray-100 px-1 rounded">:mode tools</code>, <code className="bg-gray-100 px-1 rounded">:model gpt-4.1</code>
      </div>
    </div>

    <DrawingTool
      isActive={isDrawingMode}
      onCapture={handleDrawingCapture}
      onDeactivate={() => setIsDrawingMode(false)}
    />

    {/* Capture Dialog */}
    <Dialog open={showCaptureDialog} onOpenChange={setShowCaptureDialog}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Add message for captured images</DialogTitle>
        </DialogHeader>
        
        {/* Preview captured images */}
        <div className="grid grid-cols-3 gap-2 mb-4 max-h-48 overflow-y-auto">
          {capturedImages.map((img, index) => (
            <img 
              key={index} 
              src={img} 
              alt={`Captured ${index + 1}`}
              className="w-full h-24 object-contain border rounded"
            />
          ))}
        </div>

        {/* Message input */}
        <Textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Describe what you want to do with these images..."
          className="min-h-[100px] resize-none"
          autoFocus
        />

        {/* Actions */}
        <div className="flex justify-end gap-2 mt-4">
          <Button variant="outline" onClick={handleCancelCapture}>
            Cancel
          </Button>
          <Button onClick={handleSend} disabled={!message.trim()}>
            Send with images
          </Button>
        </div>
      </DialogContent>
    </Dialog>
    </>
  );
};

export default MessageInput;
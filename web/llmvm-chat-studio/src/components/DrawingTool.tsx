import { useEffect, useRef, useState } from "react";
import html2canvas from "html2canvas";
import { useToast } from "@/hooks/use-toast";

interface DrawingToolProps {
  isActive: boolean;
  onCapture: (images: string[]) => void;
  onDeactivate: () => void;
}

interface Point {
  x: number;
  y: number;
}

interface BoundingBox {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

const DrawingTool = ({ isActive, onCapture, onDeactivate }: DrawingToolProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [paths, setPaths] = useState<Point[][]>([]);
  const [currentPath, setCurrentPath] = useState<Point[]>([]);
  const { toast } = useToast();

  const captureElementsRef = useRef<() => void>();
  
  useEffect(() => {
    captureElementsRef.current = captureElements;
  });

  useEffect(() => {
    if (!isActive) {
      setPaths([]);
      setCurrentPath([]);
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;

    // Set canvas size to match window
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    // Clear canvas
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // Add keyboard listener for Enter key
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === "Enter" && isActive) {
        e.preventDefault();
        console.log("Enter pressed, paths:", paths.length);
        captureElementsRef.current?.();
      }
    };

    window.addEventListener("keydown", handleKeyPress);

    return () => {
      window.removeEventListener("keydown", handleKeyPress);
    };
  }, [isActive]);

  useEffect(() => {
    drawPaths();
  }, [paths, currentPath]);

  const drawPaths = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw all paths
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // Draw completed paths with visual feedback
    paths.forEach((path) => {
      if (path.length > 0) {
        ctx.strokeStyle = "#3B82F6";
        ctx.beginPath();
        ctx.moveTo(path[0].x, path[0].y);
        path.forEach((point) => {
          ctx.lineTo(point.x, point.y);
        });
        ctx.stroke();

        // Draw visual feedback for what will be captured
        const bounds = getPathBounds(path);
        if (isClosedPath(path)) {
          // Draw a semi-transparent fill for closed paths
          ctx.fillStyle = "rgba(59, 130, 246, 0.1)";
          ctx.fill();
          
          // Draw bounding box
          ctx.strokeStyle = "rgba(59, 130, 246, 0.5)";
          ctx.lineWidth = 1;
          ctx.setLineDash([5, 5]);
          ctx.strokeRect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);
          ctx.setLineDash([]);
        } else if (isUnderlinePath(path)) {
          // Show capture area for underlines
          ctx.fillStyle = "rgba(59, 130, 246, 0.1)";
          ctx.fillRect(bounds.left, bounds.top - 50, bounds.right - bounds.left, 60);
          
          ctx.strokeStyle = "rgba(59, 130, 246, 0.5)";
          ctx.lineWidth = 1;
          ctx.setLineDash([5, 5]);
          ctx.strokeRect(bounds.left, bounds.top - 50, bounds.right - bounds.left, 60);
          ctx.setLineDash([]);
        }
      }
    });

    // Draw current path
    if (currentPath.length > 0) {
      ctx.strokeStyle = "#3B82F6";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(currentPath[0].x, currentPath[0].y);
      currentPath.forEach((point) => {
        ctx.lineTo(point.x, point.y);
      });
      ctx.stroke();
    }
  };

  const startDrawing = (e: React.MouseEvent) => {
    setIsDrawing(true);
    const point = { x: e.clientX, y: e.clientY };
    setCurrentPath([point]);
  };

  const draw = (e: React.MouseEvent) => {
    if (!isDrawing) return;
    
    const point = { x: e.clientX, y: e.clientY };
    setCurrentPath([...currentPath, point]);
  };

  const stopDrawing = () => {
    if (isDrawing && currentPath.length > 0) {
      setPaths([...paths, currentPath]);
      setCurrentPath([]);
    }
    setIsDrawing(false);
  };

  const getPathBounds = (path: Point[]): BoundingBox => {
    if (path.length === 0) return { left: 0, top: 0, right: 0, bottom: 0 };
    
    let minX = path[0].x;
    let maxX = path[0].x;
    let minY = path[0].y;
    let maxY = path[0].y;

    path.forEach(point => {
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    });

    return { left: minX, top: minY, right: maxX, bottom: maxY };
  };

  const isClosedPath = (path: Point[]): boolean => {
    if (path.length < 3) return false;
    
    const first = path[0];
    const last = path[path.length - 1];
    const distance = Math.sqrt(Math.pow(last.x - first.x, 2) + Math.pow(last.y - first.y, 2));
    
    return distance < 50; // Consider closed if endpoints are within 50 pixels
  };

  const isUnderlinePath = (path: Point[]): boolean => {
    if (path.length < 2) return false;
    
    const bounds = getPathBounds(path);
    const width = bounds.right - bounds.left;
    const height = bounds.bottom - bounds.top;
    
    // Check if it's mostly horizontal (width > 3x height) and not closed
    return width > height * 3 && !isClosedPath(path);
  };

  const captureElements = async () => {
    console.log("captureElements called, paths:", paths);
    const capturedImages: string[] = [];
    
    console.log("Number of paths to process:", paths.length);

    // Create an overlay canvas to draw annotations
    const overlayCanvas = document.createElement('canvas');
    overlayCanvas.style.position = 'fixed';
    overlayCanvas.style.top = '0';
    overlayCanvas.style.left = '0';
    overlayCanvas.style.width = '100%';
    overlayCanvas.style.height = '100%';
    overlayCanvas.style.pointerEvents = 'none';
    overlayCanvas.style.zIndex = '10000';
    overlayCanvas.width = window.innerWidth;
    overlayCanvas.height = window.innerHeight;
    
    const overlayCtx = overlayCanvas.getContext('2d');
    if (!overlayCtx) return;

    // Draw all paths on the overlay
    overlayCtx.strokeStyle = '#3B82F6';
    overlayCtx.lineWidth = 3;
    overlayCtx.lineCap = 'round';
    overlayCtx.lineJoin = 'round';

    // Hide the original canvas
    if (canvasRef.current) {
      canvasRef.current.style.display = 'none';
    }

    // Add overlay to DOM
    document.body.appendChild(overlayCanvas);

    for (const path of paths) {
      try {
        // Draw the path on overlay
        overlayCtx.beginPath();
        if (path.length > 0) {
          overlayCtx.moveTo(path[0].x, path[0].y);
          path.forEach(point => {
            overlayCtx.lineTo(point.x, point.y);
          });
          
          if (isClosedPath(path)) {
            overlayCtx.closePath();
            overlayCtx.stroke();
            overlayCtx.fillStyle = 'rgba(59, 130, 246, 0.1)';
            overlayCtx.fill();
          } else {
            overlayCtx.stroke();
          }
        }

        if (isUnderlinePath(path)) {
          // Handle underline - capture text above the line
          const bounds = getPathBounds(path);
          
          console.log("Capturing underline area:", {
            x: bounds.left,
            y: bounds.top - 50,
            width: bounds.right - bounds.left,
            height: 60
          });

          // Capture including the overlay
          const canvas = await html2canvas(document.body, {
            x: bounds.left,
            y: bounds.top - 50,
            width: bounds.right - bounds.left,
            height: 60,
            useCORS: true,
            allowTaint: true,
            backgroundColor: '#ffffff',
            scale: 2,
            logging: false
          });

          const dataUrl = canvas.toDataURL("image/png");
          console.log("Underline capture successful, data URL length:", dataUrl.length);
          capturedImages.push(dataUrl);
          
        } else if (isClosedPath(path)) {
          // Handle lasso - capture everything inside the bounds
          const bounds = getPathBounds(path);
          
          console.log("Capturing lasso area:", {
            x: bounds.left,
            y: bounds.top,
            width: bounds.right - bounds.left,
            height: bounds.bottom - bounds.top
          });

          // Capture including the overlay
          const canvas = await html2canvas(document.body, {
            x: bounds.left,
            y: bounds.top,
            width: bounds.right - bounds.left,
            height: bounds.bottom - bounds.top,
            useCORS: true,
            allowTaint: true,
            backgroundColor: '#ffffff',
            scale: 2,
            logging: false
          });

          const dataUrl = canvas.toDataURL("image/png");
          console.log("Lasso capture successful, data URL length:", dataUrl.length);
          capturedImages.push(dataUrl);
        } else {
          console.log("Path is neither closed nor underline, skipping");
        }

        // Clear the overlay for next path
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      } catch (error) {
        console.error("Error capturing area:", error);
        toast({
          title: "Capture error",
          description: "Failed to capture the selected area. Please try again.",
          variant: "destructive"
        });
      }
    }

    // Remove overlay and show original canvas
    document.body.removeChild(overlayCanvas);
    if (canvasRef.current) {
      canvasRef.current.style.display = 'block';
    }

    console.log("Total captured images:", capturedImages.length);

    if (capturedImages.length > 0) {
      onCapture(capturedImages);
    } else {
      // If no images captured, show a toast
      toast({
        title: "No areas captured",
        description: "Draw a closed lasso around content or underline text to capture it.",
        variant: "destructive"
      });
    }

    // Clear and deactivate
    setPaths([]);
    setCurrentPath([]);
    onDeactivate();
  };

  if (!isActive) return null;

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 z-50 cursor-crosshair"
      style={{ pointerEvents: "auto" }}
      onMouseDown={startDrawing}
      onMouseMove={draw}
      onMouseUp={stopDrawing}
      onMouseLeave={stopDrawing}
    />
  );
};

export default DrawingTool;
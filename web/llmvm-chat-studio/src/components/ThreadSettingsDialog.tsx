
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Settings } from "lucide-react";

export interface ThreadSettings {
  executor: "anthropic" | "openai" | "gemini" | "llama";
  model: string;
  temperature: number;
  endpoint: string;
  apiKey?: string;
  compression: string;
  outputTokenLen: number;
  thinking: boolean;
  fullProcessing: boolean;
}

interface ThreadSettingsDialogProps {
  settings: ThreadSettings;
  onSettingsChange: (settings: ThreadSettings) => void;
  trigger?: React.ReactNode;
}

const modelOptions = {
  anthropic: [
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-latest"
  ],
  openai: [
    "gpt-4.1",
    "o3",
    "o3-pro",
    "o3-mini",
    "o4-mini"
  ],
  gemini: [
    "gemini-2.5-pro",
    "gemini-2.5-flash"
  ],
  llama: [
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Llama-4-Scout-17B-16E-Instruct-FP8"
  ]
};

const ThreadSettingsDialog = ({ settings, onSettingsChange, trigger }: ThreadSettingsDialogProps) => {
  const [localSettings, setLocalSettings] = useState<ThreadSettings>(settings);
  const [open, setOpen] = useState(false);

  // Update local settings when settings prop changes
  useEffect(() => {
    setLocalSettings(settings);
  }, [settings]);

  const handleSave = () => {
    onSettingsChange(localSettings);
    setOpen(false); // Close the dialog after saving
  };

  const updateSetting = <K extends keyof ThreadSettings>(
    key: K,
    value: ThreadSettings[K]
  ) => {
    const updatedSettings = { ...localSettings, [key]: value };

    // If executor changes, reset model to first available option
    if (key === 'executor') {
      updatedSettings.model = modelOptions[value as keyof typeof modelOptions][0];
      
      // Set endpoint URL for Llama
      if (value === 'llama') {
        updatedSettings.endpoint = 'https://api.llama.com/compat/v1';
      }
    }

    setLocalSettings(updatedSettings);
  };

  const defaultTrigger = (
    <Button variant="ghost" size="sm" className="text-gray-600 hover:text-gray-900">
      <Settings size={16} />
    </Button>
  );

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {trigger || defaultTrigger}
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Thread Settings</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* Executor */}
          <div className="space-y-2">
            <Label htmlFor="executor">Executor</Label>
            <Select
              value={localSettings.executor}
              onValueChange={(value: "anthropic" | "openai" | "gemini" | "llama") => updateSetting('executor', value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select executor" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="anthropic">Anthropic</SelectItem>
                <SelectItem value="openai">OpenAI</SelectItem>
                <SelectItem value="gemini">Gemini</SelectItem>
                <SelectItem value="llama">Llama</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Model */}
          <div className="space-y-2">
            <Label htmlFor="model">Model</Label>
            <Select
              value={localSettings.model}
              onValueChange={(value) => updateSetting('model', value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select model" />
              </SelectTrigger>
              <SelectContent>
                {modelOptions[localSettings.executor].map((model) => (
                  <SelectItem key={model} value={model}>{model}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Temperature */}
          <div className="space-y-2">
            <Label htmlFor="temperature">Temperature</Label>
            <Input
              id="temperature"
              type="number"
              step="0.1"
              min="0"
              max="2"
              value={localSettings.temperature}
              onChange={(e) => updateSetting('temperature', parseFloat(e.target.value) || 1.0)}
            />
          </div>

          {/* Endpoint */}
          <div className="space-y-2">
            <Label htmlFor="endpoint">Endpoint</Label>
            <Input
              id="endpoint"
              type="text"
              value={localSettings.endpoint}
              onChange={(e) => updateSetting('endpoint', e.target.value)}
              placeholder="Enter endpoint URL"
            />
          </div>

          {/* API Key */}
          <div className="space-y-2">
            <Label htmlFor="apiKey">API Key</Label>
            <Input
              id="apiKey"
              type="password"
              value={localSettings.apiKey || ''}
              onChange={(e) => updateSetting('apiKey', e.target.value)}
              placeholder="Enter API key (optional)"
            />
          </div>

          {/* Compression */}
          <div className="space-y-2">
            <Label htmlFor="compression">Compression</Label>
            <Input
              id="compression"
              type="text"
              value={localSettings.compression}
              onChange={(e) => updateSetting('compression', e.target.value)}
              placeholder="Enter compression method"
            />
          </div>

          {/* Output Token Length */}
          <div className="space-y-2">
            <Label htmlFor="outputTokenLen">Output Token Length</Label>
            <Input
              id="outputTokenLen"
              type="number"
              min="1"
              value={localSettings.outputTokenLen}
              onChange={(e) => updateSetting('outputTokenLen', parseInt(e.target.value) || 0)}
            />
          </div>

          {/* Checkboxes */}
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="thinking"
                checked={localSettings.thinking}
                onCheckedChange={(checked) => updateSetting('thinking', !!checked)}
              />
              <Label htmlFor="thinking">Thinking</Label>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="fullProcessing"
                checked={localSettings.fullProcessing}
                onCheckedChange={(checked) => updateSetting('fullProcessing', !!checked)}
              />
              <Label htmlFor="fullProcessing">Full Processing</Label>
            </div>
          </div>

          {/* Save Button */}
          <Button onClick={handleSave} className="w-full">
            Save Settings
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ThreadSettingsDialog;

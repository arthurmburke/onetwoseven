'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  Box,
  Check,
  ChevronRight,
  CircleStop,
  Copy,
  Cpu,
  Download,
  ExternalLink,
  Eye,
  Gauge,
  HardDriveDownload,
  LoaderCircle,
  MemoryStick,
  Play,
  RefreshCw,
  Search,
  Server,
  Settings2,
  Sparkles,
  TerminalSquare,
  Zap,
  X,
} from 'lucide-react';

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';

const API_ORIGIN = process.env.NEXT_PUBLIC_MLX_STUDIO_API || 'http://127.0.0.1:8111';
const DEFAULT_MODEL = 'mlx-community/Qwen3-4B-4bit';

type Backend = 'lm' | 'vlm';
type Phase = 'stopped' | 'downloading' | 'starting' | 'ready' | 'stopping' | 'error';

type PerformanceSettings = {
  draft_model: string;
  num_draft_tokens: number;
  decode_concurrency: number;
  prompt_concurrency: number;
  prefill_step_size: number;
  prompt_cache_size: number;
  prompt_cache_gb: number;
  pipeline: boolean;
  default_max_tokens: number;
  kv_bits: 'none' | '8' | '3.5';
  kv_group_size: number;
  max_kv_size: number;
  quantized_kv_start: number;
  expert_cache_gb: number;
  max_num_seqs: number;
  vision_cache_size: number;
  apc_enabled: boolean;
  apc_num_blocks: number;
  draft_kind: 'auto' | 'dflash' | 'eagle3' | 'mtp';
  draft_block_size: number;
};

const DEFAULT_PERFORMANCE: PerformanceSettings = {
  draft_model: '',
  num_draft_tokens: 3,
  decode_concurrency: 32,
  prompt_concurrency: 8,
  prefill_step_size: 2048,
  prompt_cache_size: 10,
  prompt_cache_gb: 0,
  pipeline: false,
  default_max_tokens: 4096,
  kv_bits: 'none',
  kv_group_size: 64,
  max_kv_size: 0,
  quantized_kv_start: 0,
  expert_cache_gb: 0,
  max_num_seqs: 0,
  vision_cache_size: 20,
  apc_enabled: false,
  apc_num_blocks: 4096,
  draft_kind: 'auto',
  draft_block_size: 0,
};

type NumericPerformanceKey = Exclude<keyof PerformanceSettings, 'draft_model' | 'kv_bits' | 'apc_enabled' | 'draft_kind' | 'pipeline'>;

const PERFORMANCE_BOUNDS: Record<NumericPerformanceKey, [number, number]> = {
  num_draft_tokens: [1, 16],
  decode_concurrency: [1, 128],
  prompt_concurrency: [1, 64],
  prefill_step_size: [128, 32768],
  prompt_cache_size: [0, 100],
  prompt_cache_gb: [0, 1024],
  default_max_tokens: [1, 262144],
  kv_group_size: [16, 256],
  max_kv_size: [0, 1048576],
  quantized_kv_start: [0, 1048576],
  expert_cache_gb: [0, 1024],
  max_num_seqs: [0, 128],
  vision_cache_size: [0, 128],
  apc_num_blocks: [128, 65536],
  draft_block_size: [0, 64],
};

type DownloadStatus = {
  status: 'idle' | 'checking' | 'downloading' | 'complete' | 'cancelled' | 'error';
  downloaded_bytes: number;
  total_bytes: number;
  remaining_bytes: number;
  percent: number;
  files_total: number;
  files_remaining: number;
  speed_bytes_per_second: number;
  eta_seconds: number | null;
  started_at: number | null;
  repository: string | null;
  repository_index: number;
  repositories_total: number;
};

type HistoryPoint = {
  time: number;
  cpu_percent: number;
  rss_bytes: number;
  memory_percent: number;
};

type RuntimeStatus = {
  phase: Phase;
  model_id: string | null;
  backend: Backend | null;
  pid: number | null;
  started_at: number | null;
  uptime_seconds: number;
  last_error: string | null;
  endpoint: string;
  download: DownloadStatus;
  performance: Partial<PerformanceSettings>;
  inference: HistoryPoint;
  system: { memory_total_bytes: number; memory_used_bytes: number; memory_percent: number };
  requests: {
    total: number;
    errors: number;
    active: number;
    input_tokens: number;
    output_tokens: number;
    average_latency_ms: number;
    last_ttft_ms: number;
    average_ttft_ms: number;
    last_decode_tps: number;
    average_decode_tps: number;
    last_prefill_tps: number;
    average_prefill_tps: number;
    cache_hit_tokens: number;
    draft_kind: string | null;
    last_draft_rounds: number;
    last_draft_tokens: number;
    last_draft_accepted_tokens: number;
    draft_rounds: number;
    draft_tokens: number;
    draft_accepted_tokens: number;
    last_speculative_acceptance_rate: number;
    average_speculative_acceptance_rate: number;
  };
  history: HistoryPoint[];
};

type ModelResult = {
  id: string;
  backend: Backend;
  pipeline_tag: string | null;
  downloads: number;
  likes: number;
  last_modified: string | null;
  tags: string[];
  parameters: number | null;
  quantization: string | null;
  cached: boolean;
  url: string;
};

type LogEntry = { time: number; level: string; message: string };
type SystemInfo = {
  chip: string;
  platform: string;
  python: string;
  mlx_lm_installed: boolean;
  mlx_vlm_installed: boolean;
};

type ModelContext = {
  registerTool: (
    tool: {
      name: string;
      title: string;
      description: string;
      inputSchema: Record<string, unknown>;
      annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
      execute: (input: unknown) => unknown | Promise<unknown>;
    },
    options?: { signal?: AbortSignal },
  ) => void | Promise<void>;
};

declare global {
  interface Document {
    modelContext?: ModelContext;
  }
}

const EMPTY_STATUS: RuntimeStatus = {
  phase: 'stopped',
  model_id: null,
  backend: null,
  pid: null,
  started_at: null,
  uptime_seconds: 0,
  last_error: null,
  endpoint: `${API_ORIGIN}/v1`,
  download: {
    status: 'idle',
    downloaded_bytes: 0,
    total_bytes: 0,
    remaining_bytes: 0,
    percent: 0,
    files_total: 0,
    files_remaining: 0,
    speed_bytes_per_second: 0,
    eta_seconds: null,
    started_at: null,
    repository: null,
    repository_index: 0,
    repositories_total: 0,
  },
  performance: {},
  inference: { time: 0, cpu_percent: 0, rss_bytes: 0, memory_percent: 0 },
  system: { memory_total_bytes: 0, memory_used_bytes: 0, memory_percent: 0 },
  requests: {
    total: 0,
    errors: 0,
    active: 0,
    input_tokens: 0,
    output_tokens: 0,
    average_latency_ms: 0,
    last_ttft_ms: 0,
    average_ttft_ms: 0,
    last_decode_tps: 0,
    average_decode_tps: 0,
    last_prefill_tps: 0,
    average_prefill_tps: 0,
    cache_hit_tokens: 0,
    draft_kind: null,
    last_draft_rounds: 0,
    last_draft_tokens: 0,
    last_draft_accepted_tokens: 0,
    draft_rounds: 0,
    draft_tokens: 0,
    draft_accepted_tokens: 0,
    last_speculative_acceptance_rate: 0,
    average_speculative_acceptance_rate: 0,
  },
  history: [],
};

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ORIGIN}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options?.headers || {}) },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as { detail?: string; error?: { message?: string } };
    throw new Error(payload.detail || payload.error?.message || `Request failed (${response.status})`);
  }
  return response.json();
}

function formatBytes(value: number): string {
  if (!value) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const exponent = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** exponent).toFixed(exponent > 2 ? 1 : 0)} ${units[exponent]}`;
}

function formatCompact(value: number): string {
  return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value);
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function formatEta(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return 'Calculating…';
  if (seconds <= 1) return 'Almost done';
  return `${formatDuration(seconds)} remaining`;
}

function phaseLabel(phase: Phase): string {
  return { stopped: 'Offline', downloading: 'Downloading model', starting: 'Loading model', ready: 'Endpoint ready', stopping: 'Stopping', error: 'Runtime error' }[phase];
}

function draftSuggestion(modelId: string): { model: string; backend: Backend; kind: PerformanceSettings['draft_kind']; note: string } | null {
  const model = modelId.toLowerCase();
  if (model.includes('qwen3.8-27b')) {
    return {
      model: 'z-lab/Qwen3.8-27B-DFlash2',
      backend: 'vlm',
      kind: 'dflash',
      note: 'Qwen3.8 needs MLX VLM’s DFlash2 path; conventional MLX LM drafting cannot trim its hybrid cache.',
    };
  }
  if (model.includes('qwen3.5-4b')) {
    return {
      model: 'z-lab/Qwen3.5-4B-DFlash',
      backend: 'vlm',
      kind: 'dflash',
      note: 'Use the matching DFlash checkpoint through MLX VLM.',
    };
  }
  if (model.includes('qwen3-4b')) {
    return {
      model: 'mlx-community/Qwen3-0.6B-4bit',
      backend: 'lm',
      kind: 'auto',
      note: 'A small same-family Qwen3 drafter is a good starting point for single-stream latency.',
    };
  }
  return null;
}

function TelemetryChart({ history }: { history: HistoryPoint[] }) {
  const points = history.slice(-60);
  if (points.length < 2) {
    return (
      <div className="grid h-full place-items-center text-center">
        <div>
          <Activity className="mx-auto size-5 text-zinc-600" />
          <p className="mt-3 text-sm text-zinc-500">Waiting for runtime data</p>
        </div>
      </div>
    );
  }
  const width = 800;
  const height = 210;
  const path = (key: 'cpu_percent' | 'memory_percent') =>
    points
      .map((point, index) => {
        const x = (index / (points.length - 1)) * width;
        const y = height - (Math.min(100, point[key]) / 100) * (height - 18) - 9;
        return `${x},${y}`;
      })
      .join(' ');
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full" role="img" aria-label="CPU and model memory history">
      <defs>
        <linearGradient id="cpuFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#67e8f9" stopOpacity=".18" />
          <stop offset="1" stopColor="#67e8f9" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline points={`0,${height} ${path('cpu_percent')} ${width},${height}`} fill="url(#cpuFill)" stroke="none" />
      <polyline points={path('cpu_percent')} fill="none" stroke="#67e8f9" strokeWidth="2.5" vectorEffect="non-scaling-stroke" />
      <polyline points={path('memory_percent')} fill="none" stroke="#a78bfa" strokeWidth="2" strokeDasharray="5 5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export default function Home() {
  const [status, setStatus] = useState<RuntimeStatus>(EMPTY_STATUS);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL);
  const [selectedBackend, setSelectedBackend] = useState<Backend>('lm');
  const [trustRemoteCode, setTrustRemoteCode] = useState(false);
  const [performance, setPerformance] = useState<PerformanceSettings>(DEFAULT_PERFORMANCE);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [browseBackend, setBrowseBackend] = useState<'all' | Backend>('all');
  const [models, setModels] = useState<ModelResult[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [copied, setCopied] = useState(false);

  const refreshStatus = useCallback(async () => {
    try {
      const next = await apiFetch<RuntimeStatus>('/api/status');
      setStatus(next);
      setConnected(true);
      if (next.model_id && ['downloading', 'starting', 'ready', 'stopping'].includes(next.phase)) {
        setSelectedModel(next.model_id);
        setSelectedBackend(next.backend || 'lm');
      }
    } catch {
      setConnected(false);
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    void apiFetch<SystemInfo>('/api/system').then(setSystem).catch(() => undefined);
    const timer = window.setInterval(refreshStatus, 1500);
    return () => window.clearInterval(timer);
  }, [refreshStatus]);

  useEffect(() => {
    if (!connected) return;
    void apiFetch<{ data: LogEntry[] }>('/api/logs')
      .then((payload) => setLogs(payload.data.slice(-120)))
      .catch(() => undefined);
  }, [connected, status.phase, status.history.length]);

  useEffect(() => {
    if (!libraryOpen) return;
    const requestController = new AbortController();
    const timer = window.setTimeout(async () => {
      setModelsLoading(true);
      try {
        const params = new URLSearchParams({ q: query, backend: browseBackend, limit: '40' });
        const response = await fetch(`${API_ORIGIN}/api/models/search?${params}`, { signal: requestController.signal });
        const payload = await response.json() as { detail?: string; data: ModelResult[] };
        if (!response.ok) throw new Error(payload.detail || 'Model search failed');
        setModels(payload.data);
      } catch (caught) {
        if ((caught as Error).name !== 'AbortError') setError((caught as Error).message);
      } finally {
        setModelsLoading(false);
      }
    }, 250);
    return () => {
      window.clearTimeout(timer);
      requestController.abort();
    };
  }, [libraryOpen, query, browseBackend]);

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    const report = (reason: unknown) => console.warn('WebMCP registration failed', reason);
    const register = (tool: Parameters<ModelContext['registerTool']>[0]) => {
      try {
        void Promise.resolve(context.registerTool(tool, { signal: lifecycle.signal })).catch(report);
      } catch (reason) {
        report(reason);
      }
    };
    register({
      name: 'get_mlx_runtime_status',
      title: 'Get MLX runtime status',
      description: 'Read the active model, endpoint state, and current runtime metrics.',
      inputSchema: { type: 'object', properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: true, untrustedContentHint: false },
      execute: async () => apiFetch<RuntimeStatus>('/api/status'),
    });
    register({
      name: 'start_mlx_model',
      title: 'Start an MLX model',
      description: 'Start a Hugging Face MLX language or vision-language model and update the visible runtime.',
      inputSchema: {
        type: 'object',
        properties: {
          model_id: { type: 'string', description: 'Hugging Face repository ID or local model path.' },
          backend: { type: 'string', enum: ['lm', 'vlm'] },
        },
        required: ['model_id', 'backend'],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute: async (input) => {
        const value = input as { model_id?: unknown; backend?: unknown };
        if (typeof value.model_id !== 'string' || !['lm', 'vlm'].includes(String(value.backend))) {
          throw new Error('model_id and a valid backend are required');
        }
        const next = await apiFetch<RuntimeStatus>('/api/runtime/start', {
          method: 'POST',
          body: JSON.stringify({ model_id: value.model_id, backend: value.backend }),
        });
        setSelectedModel(value.model_id);
        setSelectedBackend(value.backend as Backend);
        setStatus(next);
        return { phase: next.phase, model_id: next.model_id, endpoint: next.endpoint };
      },
    });
    register({
      name: 'stop_mlx_model',
      title: 'Stop the MLX model',
      description: 'Stop the active local inference process and free its memory.',
      inputSchema: { type: 'object', properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute: async () => {
        const next = await apiFetch<RuntimeStatus>('/api/runtime/stop', { method: 'POST' });
        setStatus(next);
        return { phase: next.phase };
      },
    });
    return () => lifecycle.abort();
  }, []);

  const runModel = async () => {
    setBusy(true);
    setError(null);
    try {
      const next = await apiFetch<RuntimeStatus>('/api/runtime/start', {
        method: 'POST',
        body: JSON.stringify({
          model_id: selectedModel,
          backend: selectedBackend,
          trust_remote_code: trustRemoteCode,
          ...performance,
          draft_model: performance.draft_model.trim() || null,
        }),
      });
      setStatus(next);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const stopModel = async () => {
    setBusy(true);
    setError(null);
    try {
      setStatus(await apiFetch<RuntimeStatus>('/api/runtime/stop', { method: 'POST' }));
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const chooseModel = (model: ModelResult | { id: string; backend: Backend }) => {
    setSelectedModel(model.id);
    setSelectedBackend(model.backend);
    setLibraryOpen(false);
    setError(null);
  };

  const setPerformanceNumber = (key: NumericPerformanceKey, value: string, fallback: number) => {
    const parsed = Number.parseInt(value, 10);
    const [minimum, maximum] = PERFORMANCE_BOUNDS[key];
    const bounded = Number.isFinite(parsed) ? Math.min(maximum, Math.max(minimum, parsed)) : fallback;
    setPerformance((current) => ({ ...current, [key]: bounded }));
  };

  const opencodeConfig = useMemo(
    () =>
      JSON.stringify(
        {
          $schema: 'https://opencode.ai/config.json',
          provider: {
            'mlx-studio': {
              npm: '@ai-sdk/openai-compatible',
              name: 'MLX Studio (local)',
              options: { baseURL: 'http://127.0.0.1:8111/v1', apiKey: 'local' },
              models: { [selectedModel]: { name: selectedModel.split('/').at(-1) } },
            },
          },
        },
        null,
        2,
      ),
    [selectedModel],
  );

  const copyConfig = async () => {
    await navigator.clipboard.writeText(opencodeConfig);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  };

  const phaseColor = status.phase === 'ready' ? 'bg-emerald-400' : status.phase === 'error' ? 'bg-red-400' : ['downloading', 'starting'].includes(status.phase) ? 'bg-amber-300' : 'bg-zinc-600';
  const runtimeActive = ['downloading', 'starting', 'ready', 'stopping'].includes(status.phase);
  const packageReady = selectedBackend === 'lm' ? system?.mlx_lm_installed : system?.mlx_vlm_installed;
  const suggestedDraft = draftSuggestion(selectedModel);
  const metrics = [
    { label: 'Model memory', value: formatBytes(status.inference.rss_bytes), hint: `${status.inference.memory_percent.toFixed(1)}% of unified memory`, icon: MemoryStick },
    { label: 'Process CPU', value: `${status.inference.cpu_percent.toFixed(1)}%`, hint: status.pid ? `PID ${status.pid}` : 'No active process', icon: Cpu },
    { label: 'Output tokens', value: formatCompact(status.requests.output_tokens), hint: `${formatCompact(status.requests.input_tokens)} input · ${status.requests.total} requests`, icon: Sparkles },
    { label: 'Decode speed', value: status.requests.last_decode_tps ? `${status.requests.last_decode_tps.toFixed(1)} tok/s` : '—', hint: status.requests.average_decode_tps ? `${status.requests.average_decode_tps.toFixed(1)} tok/s average` : 'Measured after first streamed response', icon: Gauge },
    { label: 'Prefill speed', value: status.requests.last_prefill_tps ? `${status.requests.last_prefill_tps.toFixed(0)} tok/s` : '—', hint: `${formatCompact(status.requests.cache_hit_tokens)} cached · ${status.requests.last_ttft_ms ? `${(status.requests.last_ttft_ms / 1000).toFixed(2)}s TTFT` : 'TTFT pending'}`, icon: Zap },
    {
      label: 'Draft acceptance',
      value: status.requests.last_draft_tokens ? `${status.requests.last_speculative_acceptance_rate.toFixed(1)}%` : '—',
      hint: status.requests.last_draft_tokens
        ? `${status.requests.average_speculative_acceptance_rate.toFixed(1)}% weighted avg · ${formatCompact(status.requests.last_draft_accepted_tokens)}/${formatCompact(status.requests.last_draft_tokens)} last`
        : performance.draft_model ? 'Waiting for a completed drafted response' : 'No draft model active',
      icon: Check,
    },
    { label: 'Avg. latency', value: status.requests.total ? `${(status.requests.average_latency_ms / 1000).toFixed(2)}s` : '—', hint: `${status.requests.active} active · ${status.requests.errors} errors`, icon: Activity },
  ];

  return (
    <main id="overview" className="min-h-screen bg-background text-foreground">
      <div className="studio-grid min-h-screen">
        <aside className="border-r border-white/8 bg-[#090b0f] px-5 py-6">
          <div className="flex items-center gap-3 px-2">
            <div className="grid size-9 place-items-center rounded-xl bg-cyan-400 text-[#061014] shadow-[0_0_28px_rgba(34,211,238,.22)]"><Box className="size-5" strokeWidth={2.4} /></div>
            <div><p className="font-semibold tracking-tight text-white">MLX Studio</p><p className="text-xs text-zinc-500">Local model control</p></div>
          </div>
          <nav className="mt-10 space-y-1" aria-label="Primary">
            <a className="nav-item nav-item-active" href="#overview"><Activity /> Overview</a>
            <button className="nav-item" type="button" onClick={() => setLibraryOpen(true)}><Search /> Model library</button>
            <a className="nav-item" href="#api-access"><TerminalSquare /> API access</a>
          </nav>
          <div className="mt-auto hidden rounded-2xl border border-white/8 bg-white/[.025] p-4 lg:block">
            <div className="mb-3 flex items-center gap-2 text-sm text-zinc-300"><Server className="size-4 text-cyan-300" /> Endpoint</div>
            <code className="block break-all text-xs leading-5 text-zinc-500">http://127.0.0.1:8111/v1</code>
            <div className="mt-4 flex items-center gap-2 text-xs text-zinc-500"><span className={`size-1.5 rounded-full ${connected && status.phase === 'ready' ? 'bg-emerald-400' : 'bg-zinc-600'}`} /> {connected ? phaseLabel(status.phase) : 'Controller offline'}</div>
          </div>
        </aside>

        <section className="min-w-0">
          <header className="flex h-[72px] items-center justify-between border-b border-white/8 px-5 sm:px-8">
            <div><p className="text-sm font-medium text-white">Runtime overview</p><p className="max-w-[55vw] truncate text-xs text-zinc-500">{system?.chip || 'Apple Silicon · MLX'}</p></div>
            <Badge variant="outline" className="border-white/10 bg-white/[.025] text-zinc-400"><span className={`size-1.5 rounded-full ${phaseColor} ${['downloading', 'starting'].includes(status.phase) ? 'animate-pulse' : ''}`} /> {connected ? phaseLabel(status.phase) : 'Controller offline'}</Badge>
          </header>

          <div className="mx-auto max-w-[1440px] p-5 sm:p-8">
            {!connected && <div className="mb-5 flex items-center justify-between gap-4 rounded-xl border border-amber-300/15 bg-amber-300/[.06] px-4 py-3 text-sm text-amber-100"><span>The controller is offline. Run <code className="rounded bg-black/25 px-1.5 py-1 text-xs">./run.sh</code> to connect the GUI.</span><RefreshCw className="size-4 shrink-0" /></div>}
            {error && <div className="mb-5 flex items-start justify-between gap-4 rounded-xl border border-red-400/15 bg-red-400/[.06] px-4 py-3 text-sm text-red-100"><span>{error}</span><button type="button" aria-label="Dismiss error" onClick={() => setError(null)}><X className="size-4" /></button></div>}

            <section className="hero-panel overflow-hidden rounded-[24px] border border-white/10 p-6 sm:p-8">
              <div className="relative z-10 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
                <div className="min-w-0">
                  <p className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[.16em] text-cyan-300"><span className="size-1.5 rounded-full bg-cyan-300" /> Selected model</p>
                  <h1 className="break-words text-2xl font-semibold tracking-[-.035em] text-white sm:text-3xl">{selectedModel}</h1>
                  <div className="mt-4 flex flex-wrap items-center gap-2">
                    <Badge className="bg-white/8 text-zinc-300">{selectedBackend === 'lm' ? 'MLX LM' : 'MLX VLM'}</Badge>
                    {selectedModel.match(/\d+(?:\.\d+)?[-_ ]?bit/i)?.[0] && <Badge variant="outline" className="border-white/10 text-zinc-400">{selectedModel.match(/\d+(?:\.\d+)?[-_ ]?bit/i)?.[0]}</Badge>}
                    <Badge variant="outline" className="border-white/10 text-zinc-400">{selectedBackend === 'vlm' ? 'Vision + text' : 'Text generation'}</Badge>
                    {runtimeActive && <Badge variant="outline" className="border-emerald-400/20 text-emerald-300">{status.phase === 'ready' ? formatDuration(status.uptime_seconds) : phaseLabel(status.phase)}</Badge>}
                  </div>
                </div>
                <div className="flex flex-wrap gap-3">
                  <Button variant="outline" size="lg" className="border-white/12 bg-white/[.04] text-white hover:bg-white/[.08]" onClick={() => setLibraryOpen(true)} disabled={runtimeActive}><Search data-icon="inline-start" /> Browse models</Button>
                  {runtimeActive ? <Button size="lg" variant="destructive" className="px-5" onClick={stopModel} disabled={busy || status.phase === 'stopping'}>{busy || status.phase === 'stopping' ? <LoaderCircle className="animate-spin" /> : <CircleStop />} {status.phase === 'downloading' ? 'Cancel download' : 'Stop server'}</Button> : <Button size="lg" className="bg-cyan-300 px-5 text-[#061014] hover:bg-cyan-200" onClick={runModel} disabled={busy || !connected}>{busy ? <LoaderCircle className="animate-spin" /> : <Play fill="currentColor" />} Start server</Button>}
                </div>
              </div>
              {status.phase === 'downloading' && (
                <div className="relative z-10 mt-6 rounded-2xl border border-cyan-300/15 bg-black/20 p-4 sm:p-5" aria-live="polite">
                  <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
                    <div>
                      <p className="text-sm font-medium text-white">{status.download.status === 'checking' ? 'Checking local cache and model size' : 'Downloading model files'}</p>
                      <p className="mt-1 text-xs text-zinc-500">
                        {status.download.total_bytes
                          ? `${formatBytes(status.download.downloaded_bytes)} of ${formatBytes(status.download.total_bytes)}`
                          : 'Reading repository metadata from Hugging Face…'}
                      </p>
                      {status.download.repository && status.download.repositories_total > 1 && <p className="mt-1 text-xs text-zinc-600">Repository {status.download.repository_index} of {status.download.repositories_total} · {status.download.repository}</p>}
                    </div>
                    <p className="font-mono text-lg font-semibold text-cyan-200">{status.download.total_bytes ? `${status.download.percent.toFixed(1)}%` : '…'}</p>
                  </div>
                  <Progress
                    value={status.download.total_bytes ? status.download.percent : 0}
                    aria-label={status.download.total_bytes ? `Model download ${status.download.percent.toFixed(1)} percent complete` : 'Calculating model download size'}
                    className={`${status.download.status === 'checking' ? 'animate-pulse' : ''} h-2 bg-white/8 [&_[data-slot=progress-indicator]]:bg-cyan-300`}
                  />
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-x-5 gap-y-1 text-xs text-zinc-500">
                    <span>{status.download.speed_bytes_per_second ? `${formatBytes(status.download.speed_bytes_per_second)}/s` : status.download.status === 'checking' ? 'Preparing download' : 'Starting transfer'}</span>
                    <span>{status.download.status === 'checking' ? 'Size pending' : formatEta(status.download.eta_seconds)}</span>
                    {!!status.download.files_remaining && <span>{status.download.files_remaining} files to download</span>}
                  </div>
                </div>
              )}
            </section>

            <Accordion className="mt-5">
              <AccordionItem value="performance" className="rounded-2xl border border-white/8 bg-white/[.025] px-5 sm:px-6">
                <AccordionTrigger className="min-h-20 py-4 hover:no-underline" disabled={runtimeActive}>
                  <div className="flex min-w-0 items-start gap-3 pr-4">
                    <div className="mt-0.5 grid size-9 shrink-0 place-items-center rounded-xl bg-cyan-300/8 text-cyan-300"><Settings2 className="size-4" /></div>
                    <div>
                      <div className="flex flex-wrap items-center gap-2"><span className="text-sm font-medium text-white">Advanced performance</span>{performance.draft_model && <Badge className="bg-cyan-300/10 text-cyan-200"><Zap /> Speculative</Badge>}{selectedBackend === 'vlm' && performance.apc_enabled && <Badge className="bg-violet-400/10 text-violet-300">Prefix cache</Badge>}</div>
                      <p className="mt-1 text-xs font-normal leading-5 text-zinc-500">Tune speculative decoding, batching, prefill, and cache behavior before starting.</p>
                    </div>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="pb-5">
                  <div className="grid gap-4 border-t border-white/7 pt-5 sm:grid-cols-2 xl:grid-cols-4">
                    <div className="sm:col-span-2 xl:col-span-2">
                      <div className="flex items-center justify-between gap-3">
                        <label htmlFor="draft-model" className="text-sm text-zinc-300">Draft model</label>
                        {suggestedDraft && performance.draft_model !== suggestedDraft.model && (
                          <Button
                            type="button"
                            variant="ghost"
                            className="h-7 px-2 text-xs text-cyan-300 hover:bg-cyan-300/8 hover:text-cyan-200"
                            onClick={() => {
                              setSelectedBackend(suggestedDraft.backend);
                              setPerformance((current) => ({ ...current, draft_model: suggestedDraft.model, draft_kind: suggestedDraft.kind }));
                            }}
                          >
                            Use recommended drafter
                          </Button>
                        )}
                      </div>
                      <Input id="draft-model" value={performance.draft_model} onChange={(event) => setPerformance((current) => ({ ...current, draft_model: event.target.value }))} placeholder="Optional compatible HF model ID" className="mt-2 h-10 border-white/10 bg-black/20" />
                      <span className="mt-1.5 block text-xs text-zinc-600">{suggestedDraft?.note || 'Use a compatible drafter for speculative decoding. It is downloaded with the target model.'}</span>
                    </div>
                    <label>
                      <span className="text-sm text-zinc-300">Prefill step</span>
                      <Input type="number" min={128} max={32768} step={128} value={performance.prefill_step_size} onChange={(event) => setPerformanceNumber('prefill_step_size', event.target.value, 2048)} className="mt-2 h-10 border-white/10 bg-black/20" />
                      <span className="mt-1.5 block text-xs text-zinc-600">Larger chunks can speed long prompts but use more memory.</span>
                    </label>
                    <label>
                      <span className="text-sm text-zinc-300">Default max output</span>
                      <Input type="number" min={1} max={262144} step={256} value={performance.default_max_tokens} onChange={(event) => setPerformanceNumber('default_max_tokens', event.target.value, 4096)} className="mt-2 h-10 border-white/10 bg-black/20" />
                      <span className="mt-1.5 block text-xs text-zinc-600">Tokens used only when a request omits max_tokens.</span>
                    </label>
                    {selectedBackend === 'lm' ? (
                      <label>
                        <span className="text-sm text-zinc-300">Draft tokens per step</span>
                        <Input type="number" min={1} max={16} value={performance.num_draft_tokens} onChange={(event) => setPerformanceNumber('num_draft_tokens', event.target.value, 3)} disabled={!performance.draft_model} className="mt-2 h-10 border-white/10 bg-black/20" />
                        <span className="mt-1.5 block text-xs text-zinc-600">Start at 3; test 2–5 for the best acceptance rate.</span>
                      </label>
                    ) : (
                      <>
                        <label>
                          <span className="text-sm text-zinc-300">Draft family</span>
                          <Select value={performance.draft_kind} onValueChange={(value) => setPerformance((current) => ({ ...current, draft_kind: value as PerformanceSettings['draft_kind'] }))} disabled={!performance.draft_model}>
                            <SelectTrigger className="mt-2 h-10 w-full border-white/10 bg-black/20"><SelectValue /></SelectTrigger>
                            <SelectContent><SelectItem value="auto">Auto detect</SelectItem><SelectItem value="dflash">DFlash</SelectItem><SelectItem value="eagle3">EAGLE-3</SelectItem><SelectItem value="mtp">MTP</SelectItem></SelectContent>
                          </Select>
                          <span className="mt-1.5 block text-xs text-zinc-600">Auto detect is safest for supported draft checkpoints.</span>
                        </label>
                        <label>
                          <span className="text-sm text-zinc-300">Max speculative tokens</span>
                          <Input
                            type="number"
                            min={0}
                            max={63}
                            value={performance.draft_block_size ? performance.draft_block_size - 1 : 0}
                            onChange={(event) => {
                              const requested = Math.min(63, Math.max(0, Number.parseInt(event.target.value, 10) || 0));
                              setPerformance((current) => ({ ...current, draft_block_size: requested ? requested + 1 : 0 }));
                            }}
                            disabled={!performance.draft_model}
                            className="mt-2 h-10 border-white/10 bg-black/20"
                          />
                          <span className="mt-1.5 block text-xs text-zinc-600">0 uses the checkpoint default. DFlash may adapt below this ceiling.</span>
                        </label>
                      </>
                    )}

                    {selectedBackend === 'lm' ? (
                      <>
                        <label><span className="text-sm text-zinc-300">Decode concurrency</span><Input type="number" min={1} max={128} value={performance.decode_concurrency} onChange={(event) => setPerformanceNumber('decode_concurrency', event.target.value, 32)} className="mt-2 h-10 border-white/10 bg-black/20" /><span className="mt-1.5 block text-xs text-zinc-600">Maximum parallel decoding requests.</span></label>
                        <label><span className="text-sm text-zinc-300">Prompt concurrency</span><Input type="number" min={1} max={64} value={performance.prompt_concurrency} onChange={(event) => setPerformanceNumber('prompt_concurrency', event.target.value, 8)} className="mt-2 h-10 border-white/10 bg-black/20" /><span className="mt-1.5 block text-xs text-zinc-600">Maximum prompts prefilling together.</span></label>
                        <label><span className="text-sm text-zinc-300">Prompt cache entries</span><Input type="number" min={0} max={100} value={performance.prompt_cache_size} onChange={(event) => setPerformanceNumber('prompt_cache_size', event.target.value, 10)} className="mt-2 h-10 border-white/10 bg-black/20" /><span className="mt-1.5 block text-xs text-zinc-600">Reusable conversation prefixes held in memory.</span></label>
                        <label><span className="text-sm text-zinc-300">Prompt cache cap</span><div className="relative mt-2"><Input type="number" min={0} max={1024} value={performance.prompt_cache_gb} onChange={(event) => setPerformanceNumber('prompt_cache_gb', event.target.value, 0)} className="h-10 border-white/10 bg-black/20 pr-10" /><span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs text-zinc-600">GB</span></div><span className="mt-1.5 block text-xs text-zinc-600">0 leaves the byte cap unset.</span></label>
                        <div className="rounded-xl border border-white/7 bg-black/15 p-4">
                          <div className="flex items-center justify-between gap-4"><div><p className="text-sm text-zinc-300">Pipeline parallelism</p><p className="mt-1 text-xs leading-5 text-zinc-600">For distributed MLX hosts only.</p></div><Switch checked={performance.pipeline} onCheckedChange={(checked) => setPerformance((current) => ({ ...current, pipeline: checked }))} aria-label="Enable pipeline parallelism" /></div>
                        </div>
                      </>
                    ) : (
                      <>
                        <label>
                          <span className="text-sm text-zinc-300">KV cache</span>
                          <Select value={performance.kv_bits} onValueChange={(value) => setPerformance((current) => ({ ...current, kv_bits: value as PerformanceSettings['kv_bits'] }))}>
                            <SelectTrigger className="mt-2 h-10 w-full border-white/10 bg-black/20"><SelectValue /></SelectTrigger>
                            <SelectContent><SelectItem value="none">Full precision</SelectItem><SelectItem value="8">8-bit uniform</SelectItem><SelectItem value="3.5">3.5-bit TurboQuant</SelectItem></SelectContent>
                          </Select>
                          <span className="mt-1.5 block text-xs text-zinc-600">Lower memory enables longer contexts and larger batches.</span>
                        </label>
                        <label><span className="text-sm text-zinc-300">KV group size</span><Input type="number" min={16} max={256} step={16} value={performance.kv_group_size} onChange={(event) => setPerformanceNumber('kv_group_size', event.target.value, 64)} disabled={performance.kv_bits === 'none'} className="mt-2 h-10 border-white/10 bg-black/20" /><span className="mt-1.5 block text-xs text-zinc-600">Uniform quantization group size; 64 is a strong default.</span></label>
                        <label><span className="text-sm text-zinc-300">Quantize KV after</span><Input type="number" min={0} max={1048576} step={128} value={performance.quantized_kv_start} onChange={(event) => setPerformanceNumber('quantized_kv_start', event.target.value, 0)} disabled={performance.kv_bits === 'none'} className="mt-2 h-10 border-white/10 bg-black/20" /><span className="mt-1.5 block text-xs text-zinc-600">Token index; 0 quantizes the full cache.</span></label>
                        <label><span className="text-sm text-zinc-300">Max KV tokens</span><Input type="number" min={0} max={1048576} step={1024} value={performance.max_kv_size} onChange={(event) => setPerformanceNumber('max_kv_size', event.target.value, 0)} className="mt-2 h-10 border-white/10 bg-black/20" /><span className="mt-1.5 block text-xs text-zinc-600">0 uses the model default context cache.</span></label>
                        <label><span className="text-sm text-zinc-300">Expert cache</span><div className="relative mt-2"><Input type="number" min={0} max={1024} value={performance.expert_cache_gb} onChange={(event) => setPerformanceNumber('expert_cache_gb', event.target.value, 0)} className="h-10 border-white/10 bg-black/20 pr-10" /><span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs text-zinc-600">GB</span></div><span className="mt-1.5 block text-xs text-zinc-600">For offloaded MoE checkpoints; 0 uses the server default.</span></label>
                        <label><span className="text-sm text-zinc-300">Max active sequences</span><Input type="number" min={0} max={128} value={performance.max_num_seqs} onChange={(event) => setPerformanceNumber('max_num_seqs', event.target.value, 0)} className="mt-2 h-10 border-white/10 bg-black/20" /><span className="mt-1.5 block text-xs text-zinc-600">0 uses the server default; lower values bound memory.</span></label>
                        <label><span className="text-sm text-zinc-300">Vision cache entries</span><Input type="number" min={0} max={128} value={performance.vision_cache_size} onChange={(event) => setPerformanceNumber('vision_cache_size', event.target.value, 20)} className="mt-2 h-10 border-white/10 bg-black/20" /><span className="mt-1.5 block text-xs text-zinc-600">Reuses encoded images across chat turns.</span></label>
                        <div className="rounded-xl border border-white/7 bg-black/15 p-4 sm:col-span-2 xl:col-span-1">
                          <div className="flex items-center justify-between gap-4"><div><p className="text-sm text-zinc-300">Automatic prefix cache</p><p className="mt-1 text-xs leading-5 text-zinc-600">Reuse shared prompt prefixes.</p></div><Switch checked={performance.apc_enabled} onCheckedChange={(checked) => setPerformance((current) => ({ ...current, apc_enabled: checked }))} aria-label="Enable automatic prefix caching" /></div>
                          {performance.apc_enabled && <label className="mt-3 block"><span className="text-xs text-zinc-500">Cache blocks</span><Input type="number" min={128} max={65536} step={128} value={performance.apc_num_blocks} onChange={(event) => setPerformanceNumber('apc_num_blocks', event.target.value, 4096)} className="mt-1.5 h-9 border-white/10 bg-black/20" /></label>}
                        </div>
                      </>
                    )}
                  </div>
                  {selectedBackend === 'lm' && performance.draft_model && <p className="mt-4 rounded-xl border border-amber-300/10 bg-amber-300/[.04] px-4 py-3 text-xs leading-5 text-amber-100/70">MLX LM disables continuous batching while a draft model is active. This favors one interactive stream; remove the draft model for maximum multi-client throughput.</p>}
                </AccordionContent>
              </AccordionItem>
            </Accordion>

            {connected && packageReady === false && <p className="mt-3 text-xs text-amber-300/80">{selectedBackend === 'lm' ? 'mlx-lm' : 'mlx-vlm'} is not installed in this Python environment. Run <code>./setup.sh</code> before starting.</p>}

            <section className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-7">
              {metrics.map(({ label, value, hint, icon: Icon }) => <article key={label} className="metric-card rounded-2xl border border-white/8 bg-white/[.025] p-5"><div className="flex items-start justify-between"><div><p className="text-sm text-zinc-500">{label}</p><p className="mt-2 text-2xl font-semibold tracking-tight text-white">{value}</p></div><div className="grid size-9 place-items-center rounded-xl bg-white/[.04] text-zinc-400"><Icon className="size-4" /></div></div><p className="mt-4 text-xs text-zinc-600">{hint}</p></article>)}
            </section>

            <section className="mt-5 grid gap-5 xl:grid-cols-[1.25fr_.75fr]">
              <article className="rounded-2xl border border-white/8 bg-white/[.025] p-5 sm:p-6">
                <div className="flex flex-wrap items-center justify-between gap-4"><div><h2 className="font-medium text-white">Live telemetry</h2><p className="mt-1 text-sm text-zinc-500">Process load and unified memory footprint.</p></div><div className="flex gap-4 text-xs"><span className="flex items-center gap-2 text-zinc-500"><i className="size-2 rounded-full bg-cyan-300" /> CPU</span><span className="flex items-center gap-2 text-zinc-500"><i className="size-2 rounded-full bg-violet-400" /> Memory</span></div></div>
                <div className="telemetry-grid mt-6 h-56 overflow-hidden rounded-xl border border-white/5"><TelemetryChart history={status.history} /></div>
                <div className="mt-5 grid gap-4 sm:grid-cols-2">
                  <div><div className="mb-2 flex justify-between text-xs text-zinc-500"><span>System memory</span><span>{formatBytes(status.system.memory_used_bytes)} / {formatBytes(status.system.memory_total_bytes)}</span></div><Progress value={status.system.memory_percent} className="[&_[data-slot=progress-indicator]]:bg-cyan-300" /></div>
                  <div><div className="mb-2 flex justify-between text-xs text-zinc-500"><span>Model memory</span><span>{formatBytes(status.inference.rss_bytes)}</span></div><Progress value={status.inference.memory_percent} className="[&_[data-slot=progress-indicator]]:bg-violet-400" /></div>
                </div>
              </article>

              <article id="api-access" className="scroll-mt-5 rounded-2xl border border-white/8 bg-white/[.025] p-5 sm:p-6">
                <div className="flex items-start justify-between gap-4"><div><h2 className="font-medium text-white">Connect OpenCode</h2><p className="mt-1 text-sm leading-5 text-zinc-500">Save this as <code>opencode.json</code>.</p></div><Button variant="ghost" size="icon-sm" className="text-zinc-500 hover:bg-white/5 hover:text-white" aria-label="Copy configuration" onClick={copyConfig}>{copied ? <Check className="text-emerald-300" /> : <Copy />}</Button></div>
                <pre className="code-window mt-5 max-h-72 overflow-auto rounded-xl border border-white/6 bg-[#080a0e] p-4 text-xs leading-6 text-zinc-400"><code>{opencodeConfig}</code></pre>
                <div className="mt-4 grid gap-2 text-xs text-zinc-500"><div className="flex items-center justify-between rounded-lg border border-white/6 px-3 py-2"><span>Chat completions</span><code>/v1/chat/completions</code></div><div className="flex items-center justify-between rounded-lg border border-white/6 px-3 py-2"><span>Models</span><code>/v1/models</code></div></div>
              </article>
            </section>

            <section className="mt-5 rounded-2xl border border-white/8 bg-[#090b0f]">
              <div className="flex items-center justify-between border-b border-white/8 px-5 py-4"><div className="flex items-center gap-2"><TerminalSquare className="size-4 text-zinc-500" /><h2 className="text-sm font-medium text-zinc-300">Runtime log</h2></div><span className="font-mono text-xs text-zinc-600">{logs.length} lines</span></div>
              <div className="code-window h-48 overflow-auto p-5 font-mono text-xs leading-6" aria-live="polite">{logs.length ? logs.map((entry, index) => <div key={`${entry.time}-${index}`} className={entry.level === 'error' ? 'text-red-300/80' : 'text-zinc-500'}><span className="mr-3 text-zinc-700">{new Date(entry.time * 1000).toLocaleTimeString()}</span>{entry.message}</div>) : <p className="text-zinc-700">Runtime output will appear here.</p>}</div>
            </section>
          </div>
        </section>
      </div>

      <Dialog open={libraryOpen} onOpenChange={setLibraryOpen}>
        <DialogContent className="model-dialog flex max-h-[min(820px,calc(100vh-2rem))] w-[min(920px,calc(100%-2rem))] max-w-none flex-col gap-0 overflow-hidden border border-white/10 bg-[#101319] p-0 text-zinc-100 shadow-2xl">
          <DialogHeader className="border-b border-white/8 p-5 sm:p-6">
            <DialogTitle className="text-lg text-white">Hugging Face model library</DialogTitle>
            <DialogDescription className="text-zinc-500">Browse MLX-ready repositories. Starting a model downloads it to the Hugging Face cache automatically.</DialogDescription>
            <div className="flex flex-col gap-3 pt-3 sm:flex-row">
              <div className="relative flex-1"><Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-zinc-600" /><Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search model, family, or repository…" className="h-10 border-white/10 bg-black/20 pl-9 text-zinc-200 placeholder:text-zinc-600" autoFocus /></div>
              <Tabs value={browseBackend} onValueChange={(value) => setBrowseBackend(value as 'all' | Backend)}><TabsList className="h-10 bg-black/20"><TabsTrigger value="all">All</TabsTrigger><TabsTrigger value="lm">Text</TabsTrigger><TabsTrigger value="vlm">Vision</TabsTrigger></TabsList></Tabs>
            </div>
          </DialogHeader>

          <div className="code-window min-h-0 flex-1 overflow-y-auto p-3 sm:p-4">
            {modelsLoading ? <div className="grid h-64 place-items-center"><div className="text-center text-sm text-zinc-500"><LoaderCircle className="mx-auto mb-3 size-5 animate-spin text-cyan-300" />Searching Hugging Face</div></div> : models.length ? <div className="grid gap-2">{models.map((model) => <button key={model.id} type="button" className="model-row group grid gap-3 rounded-xl border border-white/7 bg-white/[.018] p-4 text-left transition hover:border-cyan-300/20 hover:bg-cyan-300/[.035] sm:grid-cols-[minmax(0,1fr)_auto]" onClick={() => chooseModel(model)}><div className="min-w-0"><div className="flex flex-wrap items-center gap-2"><p className="truncate font-medium text-zinc-200 group-hover:text-white">{model.id}</p>{model.cached && <Badge className="bg-emerald-400/10 text-emerald-300"><HardDriveDownload /> Cached</Badge>}</div><div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-zinc-600"><span>{model.backend === 'vlm' ? <Eye className="mr-1 inline size-3" /> : <Sparkles className="mr-1 inline size-3" />}{model.backend === 'vlm' ? 'Vision language' : 'Text generation'}</span><span><Download className="mr-1 inline size-3" />{formatCompact(model.downloads)}</span>{model.parameters && <span>{formatCompact(model.parameters)} parameters</span>}{model.quantization && <span>{model.quantization}</span>}</div></div><div className="flex items-center gap-2 self-center"><Badge variant="outline" className="border-white/10 text-zinc-500">{model.backend === 'lm' ? 'MLX LM' : 'MLX VLM'}</Badge><ChevronRight className="size-4 text-zinc-700 group-hover:text-cyan-300" /></div></button>)}</div> : <div className="grid h-64 place-items-center text-center"><div><Search className="mx-auto size-5 text-zinc-700" /><p className="mt-3 text-sm text-zinc-400">No compatible MLX models found</p><p className="mt-1 text-xs text-zinc-600">Try a model ID such as mlx-community/Qwen3-4B-4bit.</p>{query.includes('/') && <Button variant="outline" className="mt-4 border-white/10" onClick={() => chooseModel({ id: query.trim(), backend: browseBackend === 'vlm' ? 'vlm' : 'lm' })}>Use exact model ID</Button>}</div></div>}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/8 bg-black/10 px-5 py-4 text-xs text-zinc-600"><label className="flex cursor-pointer items-center gap-2"><input type="checkbox" checked={trustRemoteCode} onChange={(event) => setTrustRemoteCode(event.target.checked)} className="accent-cyan-300" /> Trust repository code when required</label><a href="https://huggingface.co/mlx-community" target="_blank" rel="noreferrer" className="flex items-center gap-1.5 transition hover:text-zinc-300">Open mlx-community <ExternalLink className="size-3" /></a></div>
        </DialogContent>
      </Dialog>
    </main>
  );
}

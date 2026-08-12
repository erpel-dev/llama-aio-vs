/**
 * Load-pane field catalog: labels, help, steps, and enum choices.
 * Navigation is owned by app.ts (↑↓ list · Enter edit · ←→ adjust).
 */
import os from "node:os";
import {
  FLASH_ATTENTION_MODES,
  KV_CACHE_TYPES,
  type FlashAttention,
  type KvCacheType,
} from "@llama-aio/core";

export const CPU_THREAD_MAX = Math.max(1, os.cpus().length || 8);

export const KV_TYPE_HELP: Record<KvCacheType, string> = {
  f16: "Full precision — uses the most VRAM",
  bf16: "Brain-float — similar footprint to f16",
  q8_0: "Halves KV size with little quality loss (default)",
  q4_0: "Smallest KV — can hurt long-context prompt speed",
};

export const FLASH_HELP: Record<FlashAttention, string> = {
  auto: "Let llama.cpp decide (recommended)",
  on: "Force on — often needed for quantized V cache",
  off: "Force off — quantized V will fail to start",
};

export type LoadFieldKind = "preset" | "number" | "enum" | "action";

export type LoadFieldDef =
  | {
      id: string;
      kind: "preset";
      presetId: string;
      label: string;
      help: string;
    }
  | {
      id: string;
      kind: "number";
      label: string;
      help: string;
      step: number;
      /** Which store bucket to patch. */
      store: "load" | "request";
      key: string;
      min: number | "ctxMin";
      max: number | "ctxMax" | "gpuMax" | "cpuMax";
    }
  | {
      id: string;
      kind: "enum";
      label: string;
      help: string;
      store: "load";
      key: string;
      options: Array<{ value: string; name: string }>;
    }
  | {
      id: string;
      kind: "action";
      action: "reload";
      label: string;
      help: string;
    };

export const LOAD_FIELD_DEFS: LoadFieldDef[] = [
  {
    id: "preset:agent",
    kind: "preset",
    presetId: "agent",
    label: "Preset · Coding agent",
    help: "Enter applies: 64K ctx · KV q8_0/q8_0 · 1 slot — good for tools + history.",
  },
  {
    id: "preset:context",
    kind: "preset",
    presetId: "context",
    label: "Preset · Max context",
    help: "Enter applies: largest context that fits VRAM · KV q8_0/q4_0.",
  },
  {
    id: "preset:quality",
    kind: "preset",
    presetId: "quality",
    label: "Preset · Max quality",
    help: "Enter applies: 64K ctx · KV f16/q8_0 — spends VRAM on key precision.",
  },
  {
    id: "contextLength",
    kind: "number",
    label: "Context Length",
    help: "Tokens for prompt + generation (--ctx-size). Larger uses more KV VRAM.",
    step: 512,
    store: "load",
    key: "contextLength",
    min: 512,
    max: "ctxMax",
  },
  {
    id: "gpuOffload",
    kind: "number",
    label: "GPU Offload",
    help: "Max layers in VRAM (-ngl). Model layer count = full offload; 99 usually means all.",
    step: 1,
    store: "load",
    key: "gpuOffload",
    min: 0,
    max: "gpuMax",
  },
  {
    id: "cpuThreads",
    kind: "number",
    label: "CPU Threads",
    help: `CPU thread pool during generation (-t). Max is ${CPU_THREAD_MAX} logical cores.`,
    step: 1,
    store: "load",
    key: "cpuThreads",
    min: 1,
    max: "cpuMax",
  },
  {
    id: "maxConcurrentPredictions",
    kind: "number",
    label: "Concurrent Slots",
    help: "Server slots (-np). Use 1 for chat/agents — values > 1 split context across slots.",
    step: 1,
    store: "load",
    key: "maxConcurrentPredictions",
    min: 1,
    max: 16,
  },
  {
    id: "nCpuMoe",
    kind: "number",
    label: "CPU MoE Layers",
    help: "Keep MoE experts of the first N layers on CPU (--n-cpu-moe). Dense models ignore this.",
    step: 1,
    store: "load",
    key: "nCpuMoe",
    min: 0,
    max: 64,
  },
  {
    id: "cacheTypeK",
    kind: "enum",
    label: "KV Cache Type (K)",
    help: "Key-cache dtype (--cache-type-k). q8_0 is the usual sweet spot.",
    store: "load",
    key: "cacheTypeK",
    options: KV_CACHE_TYPES.map((t) => ({ value: t, name: `${t} — ${KV_TYPE_HELP[t]}` })),
  },
  {
    id: "cacheTypeV",
    kind: "enum",
    label: "KV Cache Type (V)",
    help: "Value-cache dtype (--cache-type-v). Prefer matching K, or q4_0 V with q8_0 K.",
    store: "load",
    key: "cacheTypeV",
    options: KV_CACHE_TYPES.map((t) => ({ value: t, name: `${t} — ${KV_TYPE_HELP[t]}` })),
  },
  {
    id: "flashAttention",
    kind: "enum",
    label: "Flash Attention",
    help: "Flash Attention (-fa). Quantized V cache typically needs Auto or On.",
    store: "load",
    key: "flashAttention",
    options: FLASH_ATTENTION_MODES.map((m) => ({ value: m, name: `${m} — ${FLASH_HELP[m]}` })),
  },
  {
    id: "offloadKvCacheToGpu",
    kind: "enum",
    label: "Offload KV to GPU",
    help: "Keep KV in VRAM (default). Off passes --no-kv-offload (system RAM).",
    store: "load",
    key: "offloadKvCacheToGpu",
    options: [
      { value: "true", name: "On — KV on GPU" },
      { value: "false", name: "Off — KV in system RAM" },
    ],
  },
  {
    id: "evalBatchSize",
    kind: "number",
    label: "Eval Batch Size",
    help: "Logical eval batch (-b). Higher can speed prompt ingest; needs more scratch VRAM.",
    step: 32,
    store: "load",
    key: "evalBatchSize",
    min: 32,
    max: 8192,
  },
  {
    id: "physicalBatchSize",
    kind: "number",
    label: "Physical Batch Size",
    help: "Physical ubatch (-ub). Caps activation scratch; often ≤ eval batch.",
    step: 32,
    store: "load",
    key: "physicalBatchSize",
    min: 32,
    max: 4096,
  },
  {
    id: "speculativeMode",
    kind: "enum",
    label: "Speculative Mode",
    help: "MTP uses next-n layers in the main GGUF. DFlash needs a separate draft GGUF (set draft path in VS Code).",
    store: "load",
    key: "speculativeMode",
    options: [
      { value: "off", name: "Off" },
      { value: "mtp", name: "MTP (draft-mtp)" },
      { value: "dflash", name: "DFlash (draft-dflash)" },
    ],
  },
  {
    id: "maxDraftTokens",
    kind: "number",
    label: "Max Draft Tokens",
    help: "Tokens to draft per step (--spec-draft-n-max). DFlash often uses 8–15.",
    step: 1,
    store: "load",
    key: "maxDraftTokens",
    min: 0,
    max: 64,
  },
  {
    id: "draftGpuOffload",
    kind: "number",
    label: "Draft GPU Offload",
    help: "DFlash draft layers in VRAM (--spec-draft-ngl). 99 ≈ all.",
    step: 1,
    store: "load",
    key: "draftGpuOffload",
    min: 0,
    max: 999,
  },
  {
    id: "temperature",
    kind: "number",
    label: "Temperature",
    help: "Sampling temperature for chat/completions (request body, not a load flag).",
    step: 0.05,
    store: "request",
    key: "temperature",
    min: 0,
    max: 2,
  },
  {
    id: "topP",
    kind: "number",
    label: "Top P",
    help: "Nucleus sampling top-p (request default).",
    step: 0.01,
    store: "request",
    key: "topP",
    min: 0,
    max: 1,
  },
  {
    id: "topK",
    kind: "number",
    label: "Top K",
    help: "Top-k sampling (request default).",
    step: 1,
    store: "request",
    key: "topK",
    min: 0,
    max: 200,
  },
  {
    id: "maxTokens",
    kind: "number",
    label: "Max Tokens",
    help: "Max new tokens per chat response (request default).",
    step: 256,
    store: "request",
    key: "maxTokens",
    min: 256,
    max: 32768,
  },
  {
    id: "action:reload",
    kind: "action",
    action: "reload",
    label: "Reload server",
    help: "Enter (or F12) applies dirty settings and restarts llama-server.",
  },
];

export function roundToStep(value: number, step: number, min: number, max: number): number {
  const clamped = Math.max(min, Math.min(max, value));
  if (step <= 0) {
    return clamped;
  }
  const rounded = Math.round(clamped / step) * step;
  // Avoid float noise on fractional steps (temperature / topP).
  const decimals = step < 1 ? Math.min(6, String(step).split(".")[1]?.length ?? 2) : 0;
  const cleaned = decimals ? Number(rounded.toFixed(decimals)) : rounded;
  return Math.max(min, Math.min(max, cleaned));
}

export function formatFieldValue(value: number | string, step?: number): string {
  if (typeof value === "string") {
    return value;
  }
  if (step !== undefined && step < 1) {
    const decimals = Math.min(6, String(step).split(".")[1]?.length ?? 2);
    return value.toFixed(decimals);
  }
  return String(value);
}

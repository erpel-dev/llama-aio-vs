import * as os from "os";
import type { ModelCapabilities } from "./ggufMetadata";

export type { ModelCapabilities };

/** llama.cpp --cache-type-k / --cache-type-v */
export type KvCacheType = "f16" | "bf16" | "q8_0" | "q4_0";

export const KV_CACHE_TYPES: readonly KvCacheType[] = ["f16", "bf16", "q8_0", "q4_0"];

export function normalizeKvCacheType(value: unknown, fallback: KvCacheType = "q8_0"): KvCacheType {
  return typeof value === "string" && (KV_CACHE_TYPES as readonly string[]).includes(value)
    ? (value as KvCacheType)
    : fallback;
}

/** llama.cpp --reasoning-format */
export type ReasoningFormat = "deepseek-legacy" | "deepseek" | "none" | "auto";

export const REASONING_FORMATS: readonly ReasoningFormat[] = [
  "deepseek-legacy",
  "deepseek",
  "none",
  "auto",
];

export function normalizeReasoningFormat(
  value: unknown,
  fallback: ReasoningFormat = "deepseek-legacy"
): ReasoningFormat {
  return typeof value === "string" && (REASONING_FORMATS as readonly string[]).includes(value)
    ? (value as ReasoningFormat)
    : fallback;
}

/** llama.cpp -fa / --flash-attn */
export type FlashAttention = "auto" | "on" | "off";

export const FLASH_ATTENTION_MODES: readonly FlashAttention[] = ["auto", "on", "off"];

export function normalizeFlashAttention(
  value: unknown,
  fallback: FlashAttention = "auto"
): FlashAttention {
  return typeof value === "string" && (FLASH_ATTENTION_MODES as readonly string[]).includes(value)
    ? (value as FlashAttention)
    : fallback;
}

export interface LlamaLoadSettings {
  /** Context length (--ctx-size) */
  contextLength: number;
  /** GPU layers (-ngl) */
  gpuOffload: number;
  /** CPU threads (-t) */
  cpuThreads: number;
  /** Evaluation batch size (-b) */
  evalBatchSize: number;
  /** Physical batch size (-ub) */
  physicalBatchSize: number;
  /** Max concurrent predictions (-np) */
  maxConcurrentPredictions: number;
  /** MoE: force expert tensors of first N layers onto CPU (--n-cpu-moe) */
  nCpuMoe: number;
  /** Offload KV cache to GPU (default true; false => --no-kv-offload) */
  offloadKvCacheToGpu: boolean;
  /** Key cache dtype (--cache-type-k / -ctk); default q8_0 */
  cacheTypeK: KvCacheType;
  /** Value cache dtype (--cache-type-v / -ctv); default q8_0 */
  cacheTypeV: KvCacheType;
  /** Keep model in memory (--mlock) */
  keepModelInMemory: boolean;
  /** Use mmap (--mmap / --no-mmap) */
  tryMmap: boolean;
  /** Unified KV cache (-kvu / --kv-unified; false => --no-kv-unified) */
  unifiedKvCache: boolean;
  /** Flash Attention (-fa); "auto" leaves the llama.cpp default alone */
  flashAttention: FlashAttention;
  /** Max context checkpoints per slot (-ctxcp / --ctx-checkpoints); llama.cpp default 32 */
  contextCheckpoints: number;
  /** Min chunk size reused from cache via KV shifting (--cache-reuse); 0 = off */
  cacheReuse: number;
  /** How thoughts are returned (--reasoning-format) */
  reasoningFormat: ReasoningFormat;
  /** Thinking token budget (--reasoning-budget); -1 = unrestricted */
  reasoningBudget: number;
  /** RoPE frequency base; null = auto */
  ropeFreqBase: number | null;
  /** RoPE frequency scale; null = auto */
  ropeFreqScale: number | null;
  /** Seed; null = random */
  seed: number | null;
  /** Speculative decoding mode */
  speculativeMode: "off" | "mtp" | "dflash";
  /** Max draft tokens for speculative decoding (--spec-draft-n-max) */
  maxDraftTokens: number;
  /** Min draft tokens for speculative decoding */
  minDraftTokens: number;
  /** Draft probability (--spec-draft-p-min); mainly for MTP */
  draftProbability: number;
  /**
   * Separate draft GGUF for DFlash (`-md` / `--spec-draft-model`).
   * Required when speculativeMode is `dflash`.
   */
  draftModelPath: string;
  /** Draft model GPU layers (`--spec-draft-ngl`); 99 ≈ all */
  draftGpuOffload: number;
}

export interface RequestSettings {
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
}

export interface ExtensionState {
  selectedModelPath: string;
  loadSettings: LlamaLoadSettings;
  requestSettings: RequestSettings;
  /** @deprecated prefer modelCapabilities.maxContextLength */
  modelMaxContext?: number;
  modelCapabilities?: ModelCapabilities;
}

export const DEFAULT_LOAD_SETTINGS: LlamaLoadSettings = {
  // Copilot Chat / agent prompts are large (tools + workspace). Keep one slot
  // so the full contextLength is available (llama.cpp splits ctx across -np).
  contextLength: 65536,
  gpuOffload: 99,
  cpuThreads: Math.max(1, Math.min(8, os.cpus().length || 4)),
  evalBatchSize: 2048,
  physicalBatchSize: 512,
  maxConcurrentPredictions: 1,
  nCpuMoe: 0,
  offloadKvCacheToGpu: true,
  cacheTypeK: "q8_0",
  cacheTypeV: "q8_0",
  keepModelInMemory: false,
  tryMmap: true,
  unifiedKvCache: true,
  flashAttention: "auto",
  contextCheckpoints: 32,
  // Reuse cached prefix chunks after a divergence (helps long agent threads).
  cacheReuse: 256,
  // Keep <think> in content *and* reasoning_content so Copilot Chat sees output.
  reasoningFormat: "deepseek-legacy",
  reasoningBudget: -1,
  ropeFreqBase: null,
  ropeFreqScale: null,
  seed: null,
  speculativeMode: "off",
  maxDraftTokens: 2,
  minDraftTokens: 0,
  draftProbability: 0.75,
  draftModelPath: "",
  draftGpuOffload: 99,
};

/**
 * Fallback sampling for models without a curated mode set (see modelModes.ts).
 * Near-greedy decoding makes reasoning models repeat themselves, so stay off the
 * floor; top_k 20 matches what most current instruct/coder families recommend.
 */
/**
 * A missing value falls back to the default; a present but out-of-range value
 * is clamped. `Number(null)` is 0, so absence is checked before coercion.
 */
function toNumber(value: unknown, fallback: number): number {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function int(value: unknown, min: number, max: number, fallback: number): number {
  return Math.min(Math.max(Math.round(toNumber(value, fallback)), min), max);
}

function float(value: unknown, min: number, max: number, fallback: number): number {
  return Math.min(Math.max(toNumber(value, fallback), min), max);
}

/** Nullable numeric field: anything non-finite collapses to null ("auto"). */
function nullableFloat(value: unknown, min: number, max: number): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? Math.min(Math.max(n, min), max) : null;
}

function toBoolean(value: unknown, fallback: boolean): boolean {
  if (value === true || value === false) {
    return value;
  }
  if (typeof value === "string") {
    const s = value.trim().toLowerCase();
    if (s === "true" || s === "1" || s === "yes") {
      return true;
    }
    if (s === "false" || s === "0" || s === "no") {
      return false;
    }
  }
  if (value === null || value === undefined || value === "") {
    return fallback;
  }
  return Boolean(value);
}

const MAX_TOKENS_HARD_CAP = 8_388_608;

/**
 * Coerce anything (stale persisted state, a cleared number input, a hand-edited
 * globalState blob) into settings that llama-server can actually parse. Without
 * this an emptied field in the webview becomes `Number("")` = 0 and ships as
 * `-ub 0`, and corrupted state ships as `--ctx-size NaN`.
 */
export function normalizeLoadSettings(raw: Partial<LlamaLoadSettings> | undefined): LlamaLoadSettings {
  const d = DEFAULT_LOAD_SETTINGS;
  const s = raw || {};
  const evalBatchSize = int(s.evalBatchSize, 32, 1_048_576, d.evalBatchSize);
  return {
    contextLength: int(s.contextLength, 512, MAX_TOKENS_HARD_CAP, d.contextLength),
    gpuOffload: int(s.gpuOffload, 0, 999, d.gpuOffload),
    cpuThreads: int(s.cpuThreads, 1, 256, d.cpuThreads),
    evalBatchSize,
    // llama-server rejects a physical batch larger than the logical batch.
    physicalBatchSize: Math.min(
      int(s.physicalBatchSize, 32, 1_048_576, d.physicalBatchSize),
      evalBatchSize
    ),
    maxConcurrentPredictions: int(s.maxConcurrentPredictions, 1, 64, d.maxConcurrentPredictions),
    nCpuMoe: int(s.nCpuMoe, 0, 999, d.nCpuMoe),
    offloadKvCacheToGpu: toBoolean(s.offloadKvCacheToGpu, d.offloadKvCacheToGpu),
    cacheTypeK: normalizeKvCacheType(s.cacheTypeK),
    cacheTypeV: normalizeKvCacheType(s.cacheTypeV),
    keepModelInMemory: toBoolean(s.keepModelInMemory, d.keepModelInMemory),
    tryMmap: toBoolean(s.tryMmap, d.tryMmap),
    unifiedKvCache: toBoolean(s.unifiedKvCache, d.unifiedKvCache),
    flashAttention: normalizeFlashAttention(s.flashAttention),
    contextCheckpoints: int(s.contextCheckpoints, 0, 4096, d.contextCheckpoints),
    cacheReuse: int(s.cacheReuse, 0, 1_048_576, d.cacheReuse),
    reasoningFormat: normalizeReasoningFormat(s.reasoningFormat),
    reasoningBudget: int(s.reasoningBudget, -1, MAX_TOKENS_HARD_CAP, d.reasoningBudget),
    ropeFreqBase: nullableFloat(s.ropeFreqBase, 1, 1e9),
    ropeFreqScale: nullableFloat(s.ropeFreqScale, 0.001, 1000),
    seed: s.seed === null || s.seed === undefined ? null : int(s.seed, -1, 2 ** 31 - 1, -1),
    speculativeMode:
      s.speculativeMode === "mtp" || s.speculativeMode === "dflash" ? s.speculativeMode : "off",
    maxDraftTokens: int(s.maxDraftTokens, 0, 64, d.maxDraftTokens),
    minDraftTokens: int(s.minDraftTokens, 0, 64, d.minDraftTokens),
    draftProbability: float(s.draftProbability, 0, 1, d.draftProbability),
    draftModelPath: typeof s.draftModelPath === "string" ? s.draftModelPath.trim() : "",
    draftGpuOffload: int(s.draftGpuOffload, 0, 999, d.draftGpuOffload),
  };
}

export function normalizeRequestSettings(raw: Partial<RequestSettings> | undefined): RequestSettings {
  const d = DEFAULT_REQUEST_SETTINGS;
  const s = raw || {};
  return {
    temperature: float(s.temperature, 0, 2, d.temperature),
    topP: float(s.topP, 0, 1, d.topP),
    topK: int(s.topK, 0, 1000, d.topK),
    maxTokens: int(s.maxTokens, 16, MAX_TOKENS_HARD_CAP, d.maxTokens),
  };
}

export const DEFAULT_REQUEST_SETTINGS: RequestSettings = {
  temperature: 0.5,
  topP: 0.95,
  topK: 20,
  maxTokens: 8192,
};

export interface ServerStatus {
  running: boolean;
  pid?: number;
  port: number;
  host: string;
  modelPath?: string;
  endpoint: string;
  ownedByThisExtension: boolean;
  message: string;
  /** True when running and model/load/launch settings differ from the live server. */
  configDirty?: boolean;
  /** True while start/reload is in progress (model may be loading; HTTP not ready yet). */
  starting?: boolean;
  /** Latest boot progress line for the sidebar (e.g. "Loading model into memory… 12s"). */
  startMessage?: string;
}

export interface HfModelHit {
  id: string;
  downloads: number;
  likes: number;
  pipeline_tag?: string;
  tags?: string[];
}

export interface HfFileHit {
  path: string;
  size: number;
  url: string;
}

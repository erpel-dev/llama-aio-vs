import * as os from "os";
import type { ModelCapabilities } from "./ggufMetadata";
import {
  type GpuSplitMode,
  normalizeGpuSplitMode,
  normalizeTensorSplit,
} from "./gpuSplit";

export type { GpuSplitMode };
export {
  GPU_SPLIT_MODES,
  alignTensorSplitToMainGpu,
  gpuDisplayOrder,
  mainShareForUi,
  mainShareFromSplit,
  normalizeGpuSplitMode,
  normalizeTensorSplit,
  parseTensorSplit,
  tensorSplitForMainShare,
  tensorSplitShares,
  effectiveTensorSplitShares,
  tensorSplitSharesEqual,
  isLegacyGpu0FirstSplit,
} from "./gpuSplit";

export type { ModelCapabilities };

/** llama.cpp --cache-type-k / --cache-type-v */
export type KvCacheType =
  | "f32"
  | "f16"
  | "bf16"
  | "q8_0"
  | "q5_1"
  | "q5_0"
  | "q4_1"
  | "q4_0"
  | "iq4_nl";

/**
 * All types llama.cpp accepts for --cache-type-k / --cache-type-v
 * (common/arg.cpp: kv_cache_types), ordered most → least precise.
 */
export const KV_CACHE_TYPES: readonly KvCacheType[] = [
  "f32",
  "f16",
  "bf16",
  "q8_0",
  "q5_1",
  "q5_0",
  "q4_1",
  "q4_0",
  "iq4_nl",
];

/** True for cache types stored below 16 bits (need the Flash Attention path for V). */
export function isQuantizedKvCacheType(type: KvCacheType): boolean {
  return type !== "f32" && type !== "f16" && type !== "bf16";
}

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

/**
 * llama.cpp `-lzm` / `--lazy-mode`: on-demand reads of large host tensors
 * (PLE / `per_layer_token_embd`). Requires mmap. `auto` = on for tensors > 4 GiB.
 */
export type LazyMode = "auto" | "on" | "off";

export const LAZY_MODES: readonly LazyMode[] = ["auto", "on", "off"];

/** llama.cpp `--lazy-mode auto` threshold. */
export const LAZY_MODE_AUTO_MIN_BYTES = 4 * 1024 ** 3;

export function normalizeLazyMode(value: unknown, fallback: LazyMode = "auto"): LazyMode {
  return typeof value === "string" && (LAZY_MODES as readonly string[]).includes(value)
    ? (value as LazyMode)
    : fallback;
}

/** True when `--lazy-mode` will fault this PLE table from disk instead of keeping it resident. */
export function lazyModeReadsFromDisk(lazyMode: LazyMode, pleBytes: number): boolean {
  if (lazyMode === "on") {
    return true;
  }
  if (lazyMode === "off") {
    return false;
  }
  return Number.isFinite(pleBytes) && pleBytes > LAZY_MODE_AUTO_MIN_BYTES;
}

/**
 * llama.cpp n-gram speculative decoding variants (--spec-type ngram-*).
 * Draft-model-free: drafts from n-grams seen in the prompt itself.
 */
export type NgramSpecVariant = "simple" | "map-k" | "map-k4v" | "mod";

export const NGRAM_SPEC_VARIANTS: readonly NgramSpecVariant[] = [
  "simple",
  "map-k",
  "map-k4v",
  "mod",
];

export function normalizeNgramSpecVariant(
  value: unknown,
  fallback: NgramSpecVariant = "simple"
): NgramSpecVariant {
  return typeof value === "string" && (NGRAM_SPEC_VARIANTS as readonly string[]).includes(value)
    ? (value as NgramSpecVariant)
    : fallback;
}

/**
 * llama.cpp `--spec-type` is a list: n-gram can stack with MTP or DFlash.
 * `ngram-mtp` / `ngram-dflash` emit both implementations.
 */
export type SpeculativeMode = "off" | "mtp" | "dflash" | "ngram" | "ngram-mtp" | "ngram-dflash";

export const SPECULATIVE_MODES: readonly SpeculativeMode[] = [
  "off",
  "mtp",
  "dflash",
  "ngram",
  "ngram-mtp",
  "ngram-dflash",
];

export function normalizeSpeculativeMode(
  value: unknown,
  fallback: SpeculativeMode = "off"
): SpeculativeMode {
  return typeof value === "string" && (SPECULATIVE_MODES as readonly string[]).includes(value)
    ? (value as SpeculativeMode)
    : fallback;
}

export function speculativeUsesNgram(mode: string | undefined): boolean {
  return mode === "ngram" || mode === "ngram-mtp" || mode === "ngram-dflash";
}

export function speculativeUsesMtp(mode: string | undefined): boolean {
  return mode === "mtp" || mode === "ngram-mtp";
}

export function speculativeUsesDflash(mode: string | undefined): boolean {
  return mode === "dflash" || mode === "ngram-dflash";
}

/**
 * MTP drafts a few extra tokens (baked heads ≈ 2, Gemma sidecar ≈ 4).
 * DFlash drafts a diffusion block (llama.cpp examples use 15).
 * Switching modes should not keep the other family's leftover.
 */
export function recommendedMaxDraftTokens(
  mode: SpeculativeMode,
  current: number,
  sidecarMtp = false
): number {
  const mtpDefault = sidecarMtp ? 4 : 2;
  const n = Number.isFinite(current) ? current : 0;
  if (speculativeUsesDflash(mode)) {
    if (n <= 4) return 15;
    return n;
  }
  if (speculativeUsesMtp(mode)) {
    if (n <= 0 || n >= 8) return mtpDefault;
    if (sidecarMtp && n === 2) return 4;
    return n;
  }
  return n;
}

/** llama.cpp ngram-simple/map-* defaults vs ngram-mod (24 / 64). */
export function recommendedNgramSizes(
  variant: NgramSpecVariant,
  n: number,
  m: number
): { ngramSizeN: number; ngramSizeM: number } {
  const simpleDefault = n === 12 && m === 48;
  const modDefault = n === 24 && m === 64;
  if (variant === "mod") {
    return simpleDefault ? { ngramSizeN: 24, ngramSizeM: 64 } : { ngramSizeN: n, ngramSizeM: m };
  }
  return modDefault ? { ngramSizeN: 12, ngramSizeM: 48 } : { ngramSizeN: n, ngramSizeM: m };
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
  /** Dense: force FFN tensors of first N layers onto CPU (--n-cpu-ffn); MoE models ignore this */
  nCpuFfn: number;
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
  /**
   * On-demand reads of large host tensors (`-lzm` / `--lazy-mode`).
   * `auto` is the llama.cpp default (on for tensors > 4 GiB). Requires mmap.
   */
  lazyMode: LazyMode;
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
  speculativeMode: SpeculativeMode;
  /**
   * N-gram speculative decoding variant (--spec-type ngram-simple|map-k|map-k4v|mod).
   * Used when n-gram is on, including stacked `ngram-mtp` / `ngram-dflash`.
   */
  ngramVariant: NgramSpecVariant;
  /**
   * Lookup n-gram length (--spec-ngram-*-size-n; mod variant: --spec-ngram-mod-n-match).
   * llama.cpp defaults: 12 (simple/map-k/map-k4v), 24 (mod match).
   */
  ngramSizeN: number;
  /**
   * Draft m-gram length (--spec-ngram-*-size-m; mod variant: --spec-ngram-mod-n-max).
   * llama.cpp default: 48 (mod max: 64).
   */
  ngramSizeM: number;
  /** Minimum hits before drafting (--spec-ngram-*-min-hits; llama.cpp default 1) */
  ngramMinHits: number;
  /** Max draft tokens (--spec-draft-n-max). MTP ≈ 2 (sidecar ≈ 4); DFlash ≈ 15. */
  maxDraftTokens: number;
  /** Min draft tokens (--spec-draft-n-min). llama.cpp default 0; MTP only. */
  minDraftTokens: number;
  /** Draft probability (--spec-draft-p-min). llama.cpp CLI default is 0; we use 0.75. */
  draftProbability: number;
  /**
   * Separate draft GGUF for DFlash or sidecar MTP (`-md` / `--model-draft`).
   * Required when speculativeMode is `dflash` or `ngram-dflash`. For MTP, set when
   * the next-n heads live in a sibling `mtp-*.gguf` (Gemma 4) rather than the language GGUF.
   */
  draftModelPath: string;
  /** Draft model GPU layers (`--spec-draft-ngl`); 99 ≈ all */
  draftGpuOffload: number;
  /**
   * Vision projector GGUF (`-mm` / `--mmproj`). Empty = text-only.
   * Auto-filled from a sibling `mmproj*.gguf` when a model is selected.
   */
  mmprojPath: string;
  /**
   * GPU-offload the vision projector (llama.cpp `--mmproj-offload`, default on).
   * False passes `--no-mmproj-offload` so CLIP stays in system RAM.
   */
  mmprojOffloadToGpu: boolean;
  /**
   * How to split tensors across GPUs (`--tensor-split`). Empty = omit
   * (llama.cpp splits by VRAM, often 1:1). Stored in **device-index order**
   * (`75,25` = 75% on GPU 0). The UI slider is “% on Main GPU” and is mapped
   * with {@link tensorSplitForMainShare}.
   */
  tensorSplit: string;
  /** How layers/rows are split (`--split-mode`). Default layer. */
  splitMode: GpuSplitMode;
  /**
   * Device that holds the compute graph / scratch (`--main-gpu`).
   * The settings UI also puts this card’s share of weights + KV first.
   */
  mainGpu: number;
}

export interface RequestSettings {
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  /** Min-p sampling (0 = disabled). llama.cpp's built-in default 0.05 is wrong
   *  for most current instruct/coder families, so we default it off. */
  minP: number;
  /** Presence penalty (OpenAI-style; 0 = disabled) */
  presencePenalty: number;
  /** Frequency penalty (OpenAI-style; 0 = disabled) */
  frequencyPenalty: number;
  /** Repetition penalty (llama.cpp style; 1.0 = disabled) */
  repeatPenalty: number;
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
  nCpuFfn: 0,
  offloadKvCacheToGpu: true,
  cacheTypeK: "q8_0",
  cacheTypeV: "q8_0",
  keepModelInMemory: false,
  tryMmap: true,
  lazyMode: "auto",
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
  ngramVariant: "simple",
  ngramSizeN: 12,
  ngramSizeM: 48,
  ngramMinHits: 1,
  // MTP default. DFlash is bumped to 15 on mode switch / recommend.
  maxDraftTokens: 2,
  minDraftTokens: 0,
  // llama.cpp's CLI default is 0.00, which never stops on low-confidence drafts
  // and collapses acceptance. 0.75 is the practical gate (also for DFlash).
  draftProbability: 0.75,
  draftModelPath: "",
  draftGpuOffload: 99,
  mmprojPath: "",
  mmprojOffloadToGpu: true,
  tensorSplit: "",
  splitMode: "layer",
  mainGpu: 0,
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
    nCpuFfn: int(s.nCpuFfn, 0, 999, d.nCpuFfn),
    offloadKvCacheToGpu: toBoolean(s.offloadKvCacheToGpu, d.offloadKvCacheToGpu),
    cacheTypeK: normalizeKvCacheType(s.cacheTypeK),
    cacheTypeV: normalizeKvCacheType(s.cacheTypeV),
    keepModelInMemory: toBoolean(s.keepModelInMemory, d.keepModelInMemory),
    tryMmap: toBoolean(s.tryMmap, d.tryMmap),
    lazyMode: normalizeLazyMode(s.lazyMode),
    unifiedKvCache: toBoolean(s.unifiedKvCache, d.unifiedKvCache),
    flashAttention: normalizeFlashAttention(s.flashAttention),
    contextCheckpoints: int(s.contextCheckpoints, 0, 4096, d.contextCheckpoints),
    cacheReuse: int(s.cacheReuse, 0, 1_048_576, d.cacheReuse),
    reasoningFormat: normalizeReasoningFormat(s.reasoningFormat),
    reasoningBudget: int(s.reasoningBudget, -1, MAX_TOKENS_HARD_CAP, d.reasoningBudget),
    ropeFreqBase: nullableFloat(s.ropeFreqBase, 1, 1e9),
    ropeFreqScale: nullableFloat(s.ropeFreqScale, 0.001, 1000),
    seed: s.seed === null || s.seed === undefined ? null : int(s.seed, -1, 2 ** 31 - 1, -1),
    speculativeMode: normalizeSpeculativeMode(s.speculativeMode),
    ngramVariant: normalizeNgramSpecVariant(s.ngramVariant),
    ngramSizeN: int(s.ngramSizeN, 2, 512, d.ngramSizeN),
    ngramSizeM: int(s.ngramSizeM, 2, 2048, d.ngramSizeM),
    ngramMinHits: int(s.ngramMinHits, 1, 64, d.ngramMinHits),
    maxDraftTokens: int(s.maxDraftTokens, 0, 64, d.maxDraftTokens),
    minDraftTokens: int(s.minDraftTokens, 0, 64, d.minDraftTokens),
    draftProbability: float(s.draftProbability, 0, 1, d.draftProbability),
    draftModelPath: typeof s.draftModelPath === "string" ? s.draftModelPath.trim() : "",
    draftGpuOffload: int(s.draftGpuOffload, 0, 999, d.draftGpuOffload),
    mmprojPath: typeof s.mmprojPath === "string" ? s.mmprojPath.trim() : "",
    mmprojOffloadToGpu: toBoolean(s.mmprojOffloadToGpu, d.mmprojOffloadToGpu),
    tensorSplit: normalizeTensorSplit(s.tensorSplit),
    splitMode: normalizeGpuSplitMode(s.splitMode),
    mainGpu: int(s.mainGpu, 0, 7, d.mainGpu),
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
    minP: float(s.minP, 0, 1, d.minP),
    presencePenalty: float(s.presencePenalty, -2, 2, d.presencePenalty),
    frequencyPenalty: float(s.frequencyPenalty, -2, 2, d.frequencyPenalty),
    repeatPenalty: float(s.repeatPenalty, 0.5, 2, d.repeatPenalty),
  };
}

export const DEFAULT_REQUEST_SETTINGS: RequestSettings = {
  temperature: 0.5,
  topP: 0.95,
  topK: 20,
  maxTokens: 8192,
  // Off by default: llama.cpp's built-in server defaults (min_p 0.05,
  // repeat-last-n 64) are tuned for generic prose, not current instruct/coder
  // families — official guidance for those is min_p 0 and no penalties.
  minP: 0,
  presencePenalty: 0,
  frequencyPenalty: 0,
  repeatPenalty: 1,
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

/**
 * Sidebar/TUI status: a live HTTP server wins over a stuck "starting" flag.
 * Copilot can already be generating while boot UI is still waiting on a toast.
 */
export function effectiveServerUiState(opts: {
  starting?: boolean;
  running?: boolean;
  httpReady?: boolean;
}): { starting: boolean; ready: boolean } {
  const ready = !!(opts.httpReady || opts.running);
  return { ready, starting: !!opts.starting && !ready };
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
  /** Set when this row stands for a split GGUF (`-00001-of-00004`, …). */
  shardCount?: number;
}

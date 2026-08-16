import * as os from "os";
import { formatGpuDeviceLabel, GpuMemoryInfo } from "./gpuInfo";
import { gpuDisplayOrder, parseTensorSplit, effectiveTensorSplitShares } from "./gpuSplit";
import { heuristicMoeExpertShare, ModelCapabilities, readModelCapabilities } from "./ggufMetadata";
import { mmprojFileSize, isMtpDraftFileName, usesSidecarMtp } from "./modelLibrary";
import { KvCacheType, LlamaLoadSettings } from "./types";

export interface MemoryBarSegment {
  key: "weights" | "vision" | "kv" | "overhead" | "draft";
  label: string;
  bytes: number;
}

export interface MemoryBarChart {
  title: string;
  /** Stacked segments (weights / kv / overhead). */
  segments: MemoryBarSegment[];
  totalBytes: number;
  /** Device capacity when known (GPU VRAM or system RAM). */
  capacityBytes?: number;
}

export interface MemoryEstimate {
  fileSizeBytes: number;
  layersTotal: number;
  layersOnGpu: number;
  gpuWeightsBytes: number;
  cpuWeightsBytes: number;
  /** KV at configured context length (full). */
  kvBytes: number;
  /** KV if only a short prompt is in use (~2k tokens). */
  kvBytesWarm: number;
  kvOnGpu: boolean;
  /** MoE expert weight fraction used for --n-cpu-moe accounting. */
  moeExpertShare?: number;
  /** DFlash draft GGUF size when speculativeMode is dflash and a draft is set. */
  draftFileSizeBytes?: number;
  /** Draft weights attributed to GPU (0 when CPU-only / ngl 0). */
  draftGpuWeightsBytes?: number;
  draftCpuWeightsBytes?: number;
  /** Draft KV at full context (always estimated as f16/f16). */
  draftKvBytes?: number;
  /** MTP next-n layer count used for the speculative overhead estimate. */
  mtpLayers?: number;
  /** Approximate MTP head weights (fraction of main GGUF). */
  mtpWeightsBytes?: number;
  /** Extra MTP draft-context KV at full context. */
  mtpKvBytes?: number;
  /** Vision projector GGUF size when `--mmproj` is set. */
  mmprojFileSizeBytes?: number;
  overheadBytes: number;
  gpuOverheadBytes: number;
  cpuOverheadBytes: number;
  /** Est. at full configured context (used for spill warnings + primary bars). */
  totalGpuBytes: number;
  totalCpuBytes: number;
  /** Est. with warm/short context KV (closer to idle / mid-chat). */
  totalGpuBytesWarm: number;
  totalCpuBytesWarm: number;
  gpuTotalBytes?: number;
  gpuUsedBytes?: number;
  gpuName?: string;
  systemRamTotalBytes?: number;
  /** Stacked bars for the sidebar. */
  charts: {
    vram: MemoryBarChart;
    /** Second GPU when two (or more) discrete GPUs are detected. */
    vram2?: MemoryBarChart;
    ram: MemoryBarChart;
  };
  /** Estimated VRAM exceeds detected GPU memory (with headroom). */
  willSpill: boolean;
  /** Soft warnings (partial offload, MoE on CPU, etc.). */
  warnings: string[];
  /** Short human lines for the sidebar. */
  lines: string[];
  /** One-line headline shown above the collapsed details. */
  summary: string;
}

const GiB = 1024 ** 3;
const MiB = 1024 ** 2;

/** Context length used for "warm / mid-chat" KV (not idle-at-load). */
export const WARM_KV_CONTEXT = 2048;

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }
  if (bytes >= GiB) {
    return `${(bytes / GiB).toFixed(bytes >= 10 * GiB ? 1 : 2)} GiB`;
  }
  if (bytes >= MiB) {
    return `${(bytes / MiB).toFixed(0)} MiB`;
  }
  return `${Math.round(bytes / 1024)} KiB`;
}

function gpuLabel(gpu: GpuMemoryInfo, index: number): string {
  return formatGpuDeviceLabel(gpu, index);
}

function buildGpuBarChart(
  index: number,
  gpu: GpuMemoryInfo | undefined,
  weights: number,
  kv: number,
  overhead: number,
  spec: number,
  specLabel: string,
  labeled = false,
  vision = 0
): MemoryBarChart {
  const totalBytes = weights + kv + overhead + spec + vision;
  const title = labeled && gpu
    ? `VRAM · ${gpuLabel(gpu, index)} · est. at full context`
    : "VRAM · est. at full context";
  return {
    title,
    segments: [
      { key: "weights" as const, label: "Weights", bytes: weights },
      { key: "vision" as const, label: "Vision (CLIP)", bytes: vision },
      { key: "draft" as const, label: specLabel, bytes: spec },
      { key: "kv" as const, label: "KV cache (full ctx)", bytes: kv },
      { key: "overhead" as const, label: "Overhead", bytes: overhead },
    ],
    totalBytes,
    capacityBytes: gpu?.totalBytes,
  };
}

/**
 * Compute / graph scratch. The activation buffers scale with the *physical*
 * batch (-ub) and hidden size; -b only adds a smaller scheduling buffer.
 */
export function computeOverheadBytes(
  embeddingLength: number,
  physicalBatchSize: number,
  evalBatchSize: number
): number {
  const safe = (v: number, fallback: number) => (Number.isFinite(v) && v > 0 ? v : fallback);
  const embed = Math.max(2048, safe(embeddingLength, 4096));
  const ubatch = Math.min(Math.max(32, safe(physicalBatchSize, 512)), 8192);
  const batch = Math.min(Math.max(32, safe(evalBatchSize, 2048)), 8192);
  return Math.round(400 * MiB + ubatch * embed * 24 + batch * 8 * 1024);
}

function layersOnGpu(settings: LlamaLoadSettings, blockCount: number): number {
  if (settings.gpuOffload <= 0) {
    return 0;
  }
  if (settings.gpuOffload >= 99) {
    return blockCount;
  }
  return Math.min(settings.gpuOffload, blockCount);
}

/**
 * Approximate bytes per KV element for llama.cpp cache types.
 * Block-quant formats include small scale overhead; these are close enough for UI estimates.
 */
export function kvCacheTypeElemBytes(type: KvCacheType | undefined): number {
  switch (type) {
    case "q4_0":
      return 0.5;
    case "q8_0":
      return 1;
    case "bf16":
    case "f16":
    default:
      return 2;
  }
}

/**
 * Rough KV-cache size for the given K/V cache dtypes (default f16).
 * Handles GQA, per-layer KV heads, and sliding-window attention (e.g. Gemma 4).
 */
export function estimateKvBytes(
  caps: Pick<
    ModelCapabilities,
    | "blockCount"
    | "embeddingLength"
    | "attentionHeadCount"
    | "attentionHeadCountKv"
    | "attentionHeadCountKvPerLayer"
    | "keyLength"
    | "valueLength"
    | "keyLengthSwa"
    | "valueLengthSwa"
    | "slidingWindow"
    | "slidingWindowPattern"
    | "fullAttentionInterval"
    | "recurrentLayers"
  >,
  contextLength: number,
  cacheTypeK: KvCacheType = "q8_0",
  cacheTypeV: KvCacheType = "q8_0"
): number {
  const layers = Math.max(1, caps.blockCount || 1);
  const qHeads = Math.max(1, caps.attentionHeadCount || 8);
  const defaultKvHeads = Math.max(1, caps.attentionHeadCountKv || qHeads);
  const defaultKeyDim = Math.max(
    1,
    caps.keyLength || Math.floor((caps.embeddingLength || qHeads * 128) / qHeads)
  );
  const defaultValDim = Math.max(1, caps.valueLength || defaultKeyDim);
  const swa = caps.slidingWindow && caps.slidingWindow > 0 ? caps.slidingWindow : undefined;
  const pattern = caps.slidingWindowPattern;
  const perLayerKv = caps.attentionHeadCountKvPerLayer;
  const recurrent = caps.recurrentLayers;
  const fullInterval =
    caps.fullAttentionInterval && caps.fullAttentionInterval > 1
      ? caps.fullAttentionInterval
      : undefined;
  const kBytes = kvCacheTypeElemBytes(cacheTypeK);
  const vBytes = kvCacheTypeElemBytes(cacheTypeV);

  let total = 0;
  for (let i = 0; i < layers; i++) {
    // Hybrid (Qwen3.5 etc.): recurrent/linear layers keep a fixed SSM state — no
    // context-scaled KV cache. Only full-attention layers grow with n_ctx.
    const isRecurrent =
      recurrent && recurrent.length === layers
        ? !!recurrent[i]
        : !!(fullInterval && (i + 1) % fullInterval !== 0);
    if (isRecurrent) {
      continue;
    }

    const isSwa = !!(swa && pattern && pattern.length && pattern[i % pattern.length]);
    const nKv = Math.max(1, perLayerKv?.[i] ?? defaultKvHeads);
    // Gemma 4 stores smaller SWA head dims; other SWA models (Muse Glimmer) keep
    // the same key/value width as the global layers.
    const keyDim = isSwa && caps.keyLengthSwa ? Math.max(1, caps.keyLengthSwa) : defaultKeyDim;
    const valDim = isSwa && caps.valueLengthSwa ? Math.max(1, caps.valueLengthSwa) : defaultValDim;
    const tokens = isSwa && swa ? Math.min(contextLength, swa) : contextLength;
    total += (nKv * keyDim * kBytes + nKv * valDim * vBytes) * tokens;
  }
  return total;
}

/** Number of layers that contribute context-scaled KV (excludes hybrid recurrent layers). */
export function countFullAttentionLayers(
  caps: Pick<ModelCapabilities, "blockCount" | "fullAttentionInterval" | "recurrentLayers">
): number {
  const layers = Math.max(1, caps.blockCount || 1);
  const recurrent = caps.recurrentLayers;
  const fullInterval =
    caps.fullAttentionInterval && caps.fullAttentionInterval > 1
      ? caps.fullAttentionInterval
      : undefined;
  let n = 0;
  for (let i = 0; i < layers; i++) {
    const isRecurrent =
      recurrent && recurrent.length === layers
        ? !!recurrent[i]
        : !!(fullInterval && (i + 1) % fullInterval !== 0);
    if (!isRecurrent) {
      n++;
    }
  }
  return n;
}

export function resolveMoeExpertShare(caps: ModelCapabilities): number {
  if (!caps.isMoe) {
    return 0;
  }
  if (caps.moeExpertShare !== undefined && Number.isFinite(caps.moeExpertShare)) {
    return Math.min(0.98, Math.max(0.05, caps.moeExpertShare));
  }
  return heuristicMoeExpertShare(caps.expertCount);
}

/** Load DFlash or sidecar-MTP draft GGUF caps when a draft path is set. */
export function resolveDraftCapabilities(
  settings: LlamaLoadSettings,
  draftCaps?: ModelCapabilities
): ModelCapabilities | undefined {
  const path = (settings.draftModelPath || "").trim();
  const wantDraft =
    settings.speculativeMode === "dflash" ||
    (settings.speculativeMode === "mtp" && isMtpDraftFileName(path));
  if (!wantDraft) {
    return undefined;
  }
  if (draftCaps?.fileSizeBytes && draftCaps.blockCount) {
    return draftCaps;
  }
  if (!path) {
    return undefined;
  }
  try {
    const caps = readModelCapabilities(path);
    return caps.fileSizeBytes && caps.blockCount ? caps : undefined;
  } catch {
    return undefined;
  }
}

interface DraftFootprint {
  fileSizeBytes: number;
  layersTotal: number;
  layersOnGpu: number;
  gpuWeightsBytes: number;
  cpuWeightsBytes: number;
  kvBytes: number;
  kvBytesWarm: number;
  gpuKvBytes: number;
  cpuKvBytes: number;
  gpuKvWarmBytes: number;
  cpuKvWarmBytes: number;
}

/**
 * Draft weights + KV. DFlash forces f16/f16 draft KV; sidecar MTP uses the
 * main cache dtypes. KV sits with the draft weights (GPU when any draft layers
 * are offloaded).
 */
function estimateDraftFootprint(
  draftCaps: ModelCapabilities,
  settings: LlamaLoadSettings,
  contextLength: number,
  warmCtx: number,
  cpuOnly: boolean
): DraftFootprint {
  const fileSize = draftCaps.fileSizeBytes || 0;
  const nLayers = Math.max(1, draftCaps.blockCount || 1);
  const onGpu = cpuOnly
    ? 0
    : settings.draftGpuOffload <= 0
      ? 0
      : settings.draftGpuOffload >= 99
        ? nLayers
        : Math.min(settings.draftGpuOffload, nLayers);
  const gpuWeights = fileSize * (onGpu / nLayers);
  const cpuWeights = Math.max(0, fileSize - gpuWeights);
  const kvK = settings.speculativeMode === "dflash" ? "f16" : settings.cacheTypeK;
  const kvV = settings.speculativeMode === "dflash" ? "f16" : settings.cacheTypeV;
  const kvBytes = estimateKvBytes(draftCaps, contextLength, kvK, kvV);
  const kvBytesWarm = estimateKvBytes(draftCaps, warmCtx, kvK, kvV);
  const kvOnGpu = onGpu > 0;
  return {
    fileSizeBytes: fileSize,
    layersTotal: nLayers,
    layersOnGpu: onGpu,
    gpuWeightsBytes: gpuWeights,
    cpuWeightsBytes: cpuWeights,
    kvBytes,
    kvBytesWarm,
    gpuKvBytes: kvOnGpu ? kvBytes : 0,
    cpuKvBytes: kvOnGpu ? 0 : kvBytes,
    gpuKvWarmBytes: kvOnGpu ? kvBytesWarm : 0,
    cpuKvWarmBytes: kvOnGpu ? 0 : kvBytesWarm,
  };
}

interface MtpFootprint {
  layers: number;
  weightsBytes: number;
  kvBytes: number;
  kvBytesWarm: number;
  gpuWeightsBytes: number;
  cpuWeightsBytes: number;
  gpuKvBytes: number;
  cpuKvBytes: number;
  gpuKvWarmBytes: number;
  cpuKvWarmBytes: number;
}

/**
 * MTP (`draft-mtp`) loads next-n heads from the same GGUF as a small draft model
 * with its own context/KV. Weights are already in the main file size for the
 * primary load; this adds the extra draft-head residency + MTP KV (~single-digit
 * % of total per llama.cpp).
 */
function estimateMtpFootprint(
  caps: ModelCapabilities,
  settings: LlamaLoadSettings,
  contextLength: number,
  warmCtx: number,
  cpuOnly: boolean,
  mainOnGpu: boolean,
  mainKvOnGpu: boolean
): MtpFootprint | undefined {
  if (settings.speculativeMode !== "mtp") {
    return undefined;
  }
  const layers = Math.max(0, Math.floor(caps.nextnPredictLayers || 0));
  if (layers <= 0) {
    return undefined;
  }
  const nLayers = Math.max(1, caps.blockCount || 1);
  const fileSize = caps.fileSizeBytes || 0;
  // Approximate next-n tensor share of the GGUF (draft-mtp re-materializes them).
  const weightsBytes = fileSize * (layers / nLayers);
  // MTP heads are dense attention blocks — estimate KV as `layers` full-attn layers
  // with the main model's head dims and cache dtypes.
  const mtpCaps: Pick<
    ModelCapabilities,
    | "blockCount"
    | "embeddingLength"
    | "attentionHeadCount"
    | "attentionHeadCountKv"
    | "keyLength"
    | "valueLength"
  > = {
    blockCount: layers,
    embeddingLength: caps.embeddingLength,
    attentionHeadCount: caps.attentionHeadCount,
    attentionHeadCountKv: caps.attentionHeadCountKv,
    keyLength: caps.keyLength,
    valueLength: caps.valueLength,
  };
  const kvBytes = estimateKvBytes(
    mtpCaps,
    contextLength,
    settings.cacheTypeK,
    settings.cacheTypeV
  );
  const kvBytesWarm = estimateKvBytes(
    mtpCaps,
    warmCtx,
    settings.cacheTypeK,
    settings.cacheTypeV
  );
  const weightsOnGpu = !cpuOnly && mainOnGpu;
  const kvOnGpu = !cpuOnly && mainKvOnGpu;
  return {
    layers,
    weightsBytes,
    kvBytes,
    kvBytesWarm,
    gpuWeightsBytes: weightsOnGpu ? weightsBytes : 0,
    cpuWeightsBytes: weightsOnGpu ? 0 : weightsBytes,
    gpuKvBytes: kvOnGpu ? kvBytes : 0,
    cpuKvBytes: kvOnGpu ? 0 : kvBytes,
    gpuKvWarmBytes: kvOnGpu ? kvBytesWarm : 0,
    cpuKvWarmBytes: kvOnGpu ? 0 : kvBytesWarm,
  };
}

/**
 * Estimate GPU/CPU footprint for current load settings.
 * Intentionally approximate — good enough for spill warnings and UI hints.
 * @param options.cpuOnly When true (CPU llama.cpp build), everything is attributed to system RAM.
 * @param options.draftCaps Optional pre-read DFlash draft caps (otherwise loaded from draftModelPath).
 * @param options.gpus All detected GPUs (PCI / Vulkan order). When two or more, VRAM is split across bars.
 */
export function estimateMemory(
  caps: ModelCapabilities | undefined,
  settings: LlamaLoadSettings,
  gpu?: GpuMemoryInfo,
  options?: { cpuOnly?: boolean; draftCaps?: ModelCapabilities; gpus?: GpuMemoryInfo[] }
): MemoryEstimate | undefined {
  if (!caps?.fileSizeBytes || !caps.blockCount) {
    return undefined;
  }

  const cpuOnly = !!options?.cpuOnly;
  const fileSize = caps.fileSizeBytes;
  const nLayers = caps.blockCount;
  const onGpu = cpuOnly ? 0 : layersOnGpu(settings, nLayers);
  const frac = onGpu / nLayers;
  const moeExpertShare = resolveMoeExpertShare(caps);

  // MoE experts are most of the file; --n-cpu-moe keeps those of the first N layers on CPU.
  let gpuWeights = fileSize * frac;
  if (!cpuOnly && caps.isMoe && settings.nCpuMoe > 0 && onGpu > 0 && moeExpertShare > 0) {
    const moeCpuLayers = Math.min(settings.nCpuMoe, onGpu);
    gpuWeights = Math.max(0, gpuWeights - fileSize * (moeCpuLayers / nLayers) * moeExpertShare);
  }
  let cpuWeights = Math.max(0, fileSize - gpuWeights);
  const mmprojBytes = mmprojFileSize(settings.mmprojPath);
  const mmprojMissing = !!(settings.mmprojPath || "").trim() && mmprojBytes <= 0;
  // CLIP is a separate model: GPU-offloaded by default, not tensor-split with the LLM.
  // --no-mmproj-offload keeps the projector in system RAM.
  const gpuVisionBytes =
    !cpuOnly && onGpu > 0 && mmprojBytes > 0 && settings.mmprojOffloadToGpu !== false
      ? mmprojBytes
      : 0;
  const cpuVisionBytes = gpuVisionBytes > 0 ? 0 : Math.max(0, mmprojBytes);

  const kvBytes = estimateKvBytes(
    caps,
    settings.contextLength,
    settings.cacheTypeK,
    settings.cacheTypeV
  );
  const warmCtx = Math.min(WARM_KV_CONTEXT, Math.max(512, settings.contextLength));
  const kvBytesWarm = estimateKvBytes(caps, warmCtx, settings.cacheTypeK, settings.cacheTypeV);
  const fullAttnLayers = countFullAttentionLayers(caps);
  const kvOnGpu = !cpuOnly && settings.offloadKvCacheToGpu && onGpu > 0;
  const overheadBytes = computeOverheadBytes(
    caps.embeddingLength || 0,
    settings.physicalBatchSize,
    settings.evalBatchSize
  );
  const gpuOverheadBytes = onGpu > 0 ? overheadBytes : 0;
  const cpuOverheadBytes = onGpu > 0 ? Math.round(overheadBytes * 0.15) : Math.round(overheadBytes * 0.5);
  const gpuKvBytes = kvOnGpu ? kvBytes : 0;
  const cpuKvBytes = kvOnGpu ? 0 : kvBytes;
  const gpuKvWarm = kvOnGpu ? kvBytesWarm : 0;
  const cpuKvWarm = kvOnGpu ? 0 : kvBytesWarm;

  const draftCaps = resolveDraftCapabilities(settings, options?.draftCaps);
  const draft = draftCaps
    ? estimateDraftFootprint(draftCaps, settings, settings.contextLength, warmCtx, cpuOnly)
    : undefined;
  const sidecarMtp = usesSidecarMtp(settings);
  const mtp = sidecarMtp
    ? undefined
    : estimateMtpFootprint(
        caps,
        settings,
        settings.contextLength,
        warmCtx,
        cpuOnly,
        onGpu > 0,
        kvOnGpu
      );

  // Speculative extra (DFlash XOR MTP) — teal bar segment.
  const specGpuBundle =
    (draft ? draft.gpuWeightsBytes + draft.gpuKvBytes : 0) +
    (mtp ? mtp.gpuWeightsBytes + mtp.gpuKvBytes : 0);
  const specCpuBundle =
    (draft ? draft.cpuWeightsBytes + draft.cpuKvBytes : 0) +
    (mtp ? mtp.cpuWeightsBytes + mtp.cpuKvBytes : 0);
  const specGpuWarmBundle =
    (draft ? draft.gpuWeightsBytes + draft.gpuKvWarmBytes : 0) +
    (mtp ? mtp.gpuWeightsBytes + mtp.gpuKvWarmBytes : 0);
  const specCpuWarmBundle =
    (draft ? draft.cpuWeightsBytes + draft.cpuKvWarmBytes : 0) +
    (mtp ? mtp.cpuWeightsBytes + mtp.cpuKvWarmBytes : 0);
  const specLabel = draft
    ? sidecarMtp
      ? "MTP draft (weights + KV)"
      : "DFlash draft (weights + KV)"
    : mtp
      ? `MTP head + KV (${mtp.layers} next-n)`
      : "Speculative";

  const totalGpuBytes = gpuWeights + gpuKvBytes + gpuOverheadBytes + specGpuBundle + gpuVisionBytes;
  const totalCpuBytes = cpuWeights + cpuKvBytes + cpuOverheadBytes + specCpuBundle + cpuVisionBytes;
  const totalGpuBytesWarm = gpuWeights + gpuKvWarm + gpuOverheadBytes + specGpuWarmBundle + gpuVisionBytes;
  const totalCpuBytesWarm = cpuWeights + cpuKvWarm + cpuOverheadBytes + specCpuWarmBundle + cpuVisionBytes;
  const systemRamTotalBytes = os.totalmem();

  const warnings: string[] = [];
  let willSpill = false;

  const shardCount = caps.shardCount ?? 1;
  if (shardCount > 1) {
    const found = caps.shardsFound ?? shardCount;
    if (found < shardCount) {
      warnings.unshift(
        `Split model: only ${found} of ${shardCount} shards were found next to this file. ` +
          `The estimate below covers just those — llama-server needs every shard to load.`
      );
    } else {
      warnings.push(`Split model: ${shardCount} shards totalling ${formatBytes(fileSize)}.`);
    }
  }

  if (cpuOnly) {
    warnings.push(
      "CPU backend: no GPU acceleration — weights, KV cache, and compute use system RAM (GPU Offload is ignored)."
    );
  }

  if (!cpuOnly && onGpu > 0 && onGpu < nLayers) {
    warnings.push(
      `Partial GPU offload: ${nLayers - onGpu}/${nLayers} layers (~${formatBytes(cpuWeights)}) stay in system RAM (slower).`
    );
  }
  if (!cpuOnly && onGpu === 0) {
    warnings.push("GPU offload is 0 — model weights run from system RAM (CPU).");
  }
  if (!cpuOnly && !settings.offloadKvCacheToGpu) {
    warnings.push(
      `KV cache (~${formatBytes(kvBytes)} at full context) is in system RAM, not VRAM.`
    );
  }
  if (!cpuOnly && caps.isMoe && settings.nCpuMoe > 0) {
    warnings.push(
      `CPU MoE layers = ${settings.nCpuMoe}: ~${Math.round(moeExpertShare * 100)}% of weights are experts; those layers’ experts stay in system RAM.`
    );
  }
  if (settings.speculativeMode === "dflash" && !draft) {
    warnings.push(
      "DFlash is on but no draft GGUF is selected — memory bars omit the draft; pick a draft model before starting."
    );
  }
  if (settings.speculativeMode === "mtp" && !mtp && !draft) {
    warnings.push(
      "MTP is on but this GGUF reports no nextn_predict_layers and no sidecar mtp-*.gguf — speculative overhead omitted from the bars."
    );
  }
  if (draft) {
    warnings.push(
      (sidecarMtp ? "MTP sidecar included: " : "DFlash draft included: ") +
        `~${formatBytes(draft.fileSizeBytes)} weights` +
        ` (${draft.layersOnGpu}/${draft.layersTotal} GPU layers)` +
        ` + ~${formatBytes(draft.kvBytes)} draft KV` +
        (sidecarMtp ? "" : " (f16)") +
        " at full context."
    );
  }
  if (mtp) {
    warnings.push(
      `MTP overhead included: ~${formatBytes(mtp.weightsBytes)} next-n head` +
        ` (${mtp.layers} layers) + ~${formatBytes(mtp.kvBytes)} MTP KV at full context.`
    );
  }
  if (mmprojMissing) {
    warnings.push(
      "Vision projector path is set but the file is missing — llama-server --mmproj will fail until you pick a valid mmproj or Clear."
    );
  }

  // Keep ~8% free for the compositor / driver; going above this often spills even if
  // the raw estimate is still slightly under the advertised GPU total.
  const usableFraction = 0.92;
  const gpus: GpuMemoryInfo[] =
    !cpuOnly && options?.gpus?.length
      ? options.gpus.filter((g) => g.totalBytes > 0)
      : !cpuOnly && gpu?.totalBytes
        ? [gpu]
        : [];
  const shares = effectiveTensorSplitShares(
    settings.tensorSplit,
    settings.splitMode,
    settings.mainGpu || 0,
    gpus.length,
    gpus.map((g) => g.totalBytes)
  );
  const mainGpuIndex = gpus.length
    ? Math.min(Math.max(0, settings.mainGpu || 0), gpus.length - 1)
    : 0;
  const perGpuParts = gpus.map((_, i) => {
    const share = shares[i] || 0;
    const isMain = i === mainGpuIndex;
    const weights = gpuWeights * share;
    const kv = gpuKvBytes * share;
    const overhead = isMain ? gpuOverheadBytes : 0;
    const spec = isMain ? specGpuBundle : 0;
    const vision = isMain ? gpuVisionBytes : 0;
    return { weights, kv, overhead, spec, vision, used: weights + kv + overhead + spec + vision };
  });

  if (mmprojBytes > 0) {
    const where =
      gpuVisionBytes > 0
        ? gpus.length
          ? ` on ${gpuLabel(gpus[mainGpuIndex]!, mainGpuIndex)} (CLIP / --mmproj, not tensor-split)`
          : " in VRAM (--mmproj, GPU offload on)"
        : settings.mmprojOffloadToGpu === false
          ? " in system RAM (--no-mmproj-offload)"
          : " in system RAM";
    warnings.push(`Vision projector included: ~${formatBytes(mmprojBytes)}${where}.`);
  }

  if (!cpuOnly && gpus.length) {
    for (let i = 0; i < gpus.length; i++) {
      const used = perGpuParts[i]!.used;
      const cap = gpus[i]!.totalBytes;
      const pct = Math.round((used / cap) * 100);
      const label = gpuLabel(gpus[i]!, i);
      if (used > cap) {
        willSpill = true;
        warnings.unshift(
          `Estimated ${label} at full context ~${formatBytes(used)} is over the full ${formatBytes(cap)} (${pct}%). Expect spill to system RAM (much slower). Lower Context Length, GPU Offload, or use a smaller quant.`
        );
      } else if (used > cap * usableFraction) {
        willSpill = true;
        warnings.unshift(
          `Tight on ${label} at full context: ~${formatBytes(used)} of ${formatBytes(cap)} (${pct}%). Only ~${formatBytes(cap - cap * usableFraction)} is left as safe headroom for the driver — llama.cpp often spills to system RAM at this point. Lower Context Length or GPU Offload.`
        );
      } else if (used > cap * 0.8) {
        warnings.push(
          `Getting full on ${label} at full context: ~${formatBytes(used)} of ${formatBytes(cap)} VRAM (${pct}%). Leave some free for the display driver.`
        );
      }
    }
    if (
      gpus.length >= 2 &&
      settings.splitMode !== "none" &&
      parseTensorSplit(settings.tensorSplit).length < 2
    ) {
      warnings.push(
        "Tensor split is empty — llama.cpp will split by VRAM size (often 1:1). Pick the faster card as Main GPU and raise Weights on main GPU so that card gets more of the model."
      );
    }
  }

  if (cpuOnly && systemRamTotalBytes && totalCpuBytes > systemRamTotalBytes * 0.9) {
    willSpill = true;
    warnings.unshift(
      `Estimated system RAM at full context ~${formatBytes(totalCpuBytes)} is very high vs ${formatBytes(systemRamTotalBytes)}. Lower Context Length or use a smaller model/quant.`
    );
  }

  const lines: string[] = [];
  if (cpuOnly) {
    lines.push("Backend: CPU (x64) — GPU Offload / VRAM not used");
  } else if (gpus.length) {
    for (let i = 0; i < gpus.length; i++) {
      const g = gpus[i]!;
      const free =
        g.usedBytes !== undefined ? Math.max(0, g.totalBytes - g.usedBytes) : undefined;
      lines.push(`${gpuLabel(g, i)} capacity: ${formatBytes(g.totalBytes)}`);
      if (free !== undefined) {
        lines.push(
          `Live ${gpuLabel(g, i)} free now: ~${formatBytes(free)} (current occupancy — not part of the estimate bars)`
        );
      }
    }
    if (gpus.length >= 2) {
      const mainLabel = gpus[mainGpuIndex]
        ? gpuLabel(gpus[mainGpuIndex]!, mainGpuIndex)
        : `GPU ${mainGpuIndex}`;
      if (settings.splitMode === "none") {
        lines.push(`No GPU split — all GPU layers on ${mainLabel} (--split-mode none)`);
      } else {
        const split = parseTensorSplit(settings.tensorSplit);
        const splitLabel = split.length >= 2 ? settings.tensorSplit : "auto (by VRAM)";
        lines.push(
          `Tensor split: ${splitLabel} · split-mode ${settings.splitMode || "layer"} · main ${mainLabel}`
        );
      }
    }
  } else {
    lines.push("GPU VRAM: unknown (could not detect)");
  }
  lines.push(`System RAM capacity: ${formatBytes(systemRamTotalBytes)}`);
  if (cpuOnly) {
    lines.push(`Weights in RAM: ~${formatBytes(cpuWeights)} (${nLayers} layers)`);
    lines.push(
      `KV @ full ${settings.contextLength.toLocaleString()} ctx: ~${formatBytes(kvBytes)} (system RAM)` +
        (fullAttnLayers < nLayers ? ` · ${fullAttnLayers}/${nLayers} full-attn layers` : "")
    );
    if (draft) {
      lines.push(
        `DFlash draft in RAM: ~${formatBytes(draft.cpuWeightsBytes)} weights` +
          ` + ~${formatBytes(draft.kvBytes)} KV (f16)`
      );
    }
    if (mtp) {
      lines.push(
        `MTP in RAM: ~${formatBytes(mtp.weightsBytes)} next-n head` +
          ` + ~${formatBytes(mtp.kvBytes)} MTP KV`
      );
    }
    if (mmprojBytes > 0) {
      lines.push(`Vision projector in RAM: ~${formatBytes(mmprojBytes)}`);
    }
    if (kvBytesWarm < kvBytes) {
      lines.push(
        `KV @ ~${warmCtx.toLocaleString()} ctx (mid-chat): ~${formatBytes(kvBytesWarm)} → total ~${formatBytes(totalCpuBytesWarm)}`
      );
    }
    lines.push(`Est. total system RAM at full context: ~${formatBytes(totalCpuBytes)}`);
  } else {
    lines.push(
      `Weights on GPU: ~${formatBytes(gpuWeights)} (${onGpu}/${nLayers} layers)` +
        (cpuWeights > MiB ? ` · RAM: ~${formatBytes(cpuWeights)}` : "") +
        (caps.isMoe && moeExpertShare > 0
          ? ` · MoE experts ~${Math.round(moeExpertShare * 100)}% of file`
          : "")
    );
    lines.push(
      `KV @ full ${settings.contextLength.toLocaleString()} ctx: ~${formatBytes(kvBytes)}` +
        (kvOnGpu ? " (GPU)" : " (CPU RAM)") +
        (fullAttnLayers < nLayers ? ` · ${fullAttnLayers}/${nLayers} full-attn layers` : "")
    );
    if (draft) {
      lines.push(
        `DFlash draft: ~${formatBytes(draft.gpuWeightsBytes)} GPU / ~${formatBytes(draft.cpuWeightsBytes)} RAM weights` +
          ` (${draft.layersOnGpu}/${draft.layersTotal} layers)` +
          ` · draft KV ~${formatBytes(draft.kvBytes)} f16` +
          (draft.gpuKvBytes > 0 ? " (GPU)" : " (CPU RAM)")
      );
    }
    if (mtp) {
      lines.push(
        `MTP: ~${formatBytes(mtp.gpuWeightsBytes || mtp.cpuWeightsBytes)} next-n head` +
          ` (${mtp.layers} layers)` +
          ` · MTP KV ~${formatBytes(mtp.kvBytes)}` +
          (mtp.gpuKvBytes > 0 ? " (GPU)" : " (CPU RAM)")
      );
    }
    if (mmprojBytes > 0) {
      lines.push(
        `Vision projector: ~${formatBytes(mmprojBytes)}` +
          (gpuVisionBytes > 0
            ? gpus.length
              ? ` (${gpuLabel(gpus[mainGpuIndex]!, mainGpuIndex)}, CLIP / --mmproj)`
              : " (GPU, --mmproj)"
            : settings.mmprojOffloadToGpu === false
              ? " (CPU RAM, --no-mmproj-offload)"
              : " (CPU RAM)")
      );
    }
    if (kvBytesWarm < kvBytes) {
      lines.push(
        `KV @ ~${warmCtx.toLocaleString()} ctx (mid-chat): ~${formatBytes(kvBytesWarm)}` +
          (kvOnGpu ? " (GPU)" : " (CPU RAM)") +
          ` → VRAM ~${formatBytes(totalGpuBytesWarm)}`
      );
    }
    lines.push(
      `Est. total at full context — VRAM: ~${formatBytes(totalGpuBytes)}` +
        (totalCpuBytes > MiB ? ` · system RAM: ~${formatBytes(totalCpuBytes)}` : "")
    );
  }
  lines.push("Bars show estimate at full context. Actual use varies by quant, MoE, and backend.");

  const vramSummary = (() => {
    if (cpuOnly) {
      return (
        `System RAM ~${formatBytes(totalCpuBytes)}` +
        (systemRamTotalBytes ? ` of ${formatBytes(systemRamTotalBytes)}` : "") +
        ` · KV ~${formatBytes(kvBytes)} at full context` +
        (draft ? ` · DFlash +${formatBytes(draft.fileSizeBytes + draft.kvBytes)}` : "") +
        (mtp ? ` · MTP +${formatBytes(mtp.weightsBytes + mtp.kvBytes)}` : "") +
        (mmprojBytes > 0 ? ` · vision +${formatBytes(mmprojBytes)}` : "")
      );
    }
    const specBit =
      specGpuBundle > 0
        ? ` · ${draft ? "DFlash" : "MTP"} +${formatBytes(specGpuBundle)}` +
          (specCpuBundle > MiB ? ` (+${formatBytes(specCpuBundle)} RAM)` : "")
        : "";
    const visionBit = mmprojBytes > 0 ? ` · vision +${formatBytes(mmprojBytes)}` : "";
    const kvBit = ` · KV ~${formatBytes(kvBytes)}${kvOnGpu ? " on GPU" : " in RAM"} · ${onGpu}/${nLayers} layers`;
    if (gpus.length >= 2 && perGpuParts.length >= 2) {
      const order = gpuDisplayOrder(gpus.length, mainGpuIndex).slice(0, 2);
      const parts = order.map((i) => {
        const p = perGpuParts[i]!;
        const cap = gpus[i]!.totalBytes;
        const pct = Math.round((p.used / cap) * 100);
        return `${gpuLabel(gpus[i]!, i)} ~${formatBytes(p.used)} of ${formatBytes(cap)} (${pct}%)`;
      });
      return `VRAM ${parts.join(" · ")}${kvBit}${specBit}${visionBit}`;
    }
    const cap = gpus[0]?.totalBytes ?? gpu?.totalBytes;
    return (
      `VRAM ~${formatBytes(totalGpuBytes)}` +
      (cap ? ` of ${formatBytes(cap)} (${Math.round((totalGpuBytes / cap) * 100)}%)` : "") +
      kvBit +
      specBit +
      visionBit
    );
  })();

  const chartOrder = gpuDisplayOrder(gpus.length, mainGpuIndex);
  const gpuCharts = chartOrder.map((i) => {
    const g = gpus[i]!;
    const p = perGpuParts[i]!;
    return buildGpuBarChart(
      i,
      g,
      p.weights,
      p.kv,
      p.overhead,
      p.spec,
      specLabel,
      gpus.length >= 2,
      p.vision
    );
  });
  const charts = {
    vram:
      gpuCharts[0] ||
      buildGpuBarChart(
        0,
        cpuOnly ? undefined : gpu,
        gpuWeights,
        gpuKvBytes,
        gpuOverheadBytes,
        specGpuBundle,
        specLabel,
        false,
        gpuVisionBytes
      ),
    vram2: gpuCharts[1],
    ram: {
      title: "System RAM · est. at full context",
      segments: [
        { key: "weights" as const, label: "Weights", bytes: cpuWeights },
        { key: "vision" as const, label: "Vision (CLIP)", bytes: cpuVisionBytes },
        {
          key: "draft" as const, label: specLabel, bytes: specCpuBundle,
        },
        { key: "kv" as const, label: "KV cache (full ctx)", bytes: cpuKvBytes },
        { key: "overhead" as const, label: "Overhead", bytes: cpuOverheadBytes },
      ],
      totalBytes: totalCpuBytes,
      capacityBytes: systemRamTotalBytes,
    },
  };

  const primaryGpu = gpus[mainGpuIndex] || gpus[0] || gpu;

  return {
    fileSizeBytes: fileSize,
    layersTotal: nLayers,
    layersOnGpu: onGpu,
    gpuWeightsBytes: gpuWeights,
    cpuWeightsBytes: cpuWeights,
    kvBytes,
    kvBytesWarm,
    kvOnGpu,
    moeExpertShare: caps.isMoe ? moeExpertShare : undefined,
    draftFileSizeBytes: draft?.fileSizeBytes,
    draftGpuWeightsBytes: draft?.gpuWeightsBytes,
    draftCpuWeightsBytes: draft?.cpuWeightsBytes,
    draftKvBytes: draft?.kvBytes,
    mtpLayers: mtp?.layers,
    mtpWeightsBytes: mtp?.weightsBytes,
    mtpKvBytes: mtp?.kvBytes,
    mmprojFileSizeBytes: mmprojBytes > 0 ? mmprojBytes : undefined,
    overheadBytes,
    gpuOverheadBytes,
    cpuOverheadBytes,
    totalGpuBytes,
    totalCpuBytes,
    totalGpuBytesWarm,
    totalCpuBytesWarm,
    gpuTotalBytes: primaryGpu?.totalBytes,
    gpuUsedBytes: primaryGpu?.usedBytes,
    gpuName: primaryGpu?.name,
    systemRamTotalBytes,
    charts,
    willSpill,
    warnings,
    lines,
    summary: vramSummary,
  };
}

/** Compact JSON-safe payload for the webview live calculator (main + optional DFlash draft). */
export function memoryEstimateInputs(
  caps: ModelCapabilities | undefined,
  draftCaps?: ModelCapabilities,
  mmprojFileSizeBytes?: number
): Record<string, unknown> | null {
  if (!caps?.fileSizeBytes) {
    return null;
  }
  const pack = (c: ModelCapabilities) => ({
    fileSizeBytes: c.fileSizeBytes,
    blockCount: c.blockCount,
    embeddingLength: c.embeddingLength || 0,
    attentionHeadCount: c.attentionHeadCount || 0,
    attentionHeadCountKv: c.attentionHeadCountKv || 0,
    attentionHeadCountKvPerLayer: c.attentionHeadCountKvPerLayer || null,
    keyLength: c.keyLength || 0,
    valueLength: c.valueLength || 0,
    keyLengthSwa: c.keyLengthSwa || 0,
    valueLengthSwa: c.valueLengthSwa || 0,
    slidingWindow: c.slidingWindow || 0,
    slidingWindowPattern: c.slidingWindowPattern || null,
    fullAttentionInterval: c.fullAttentionInterval || 0,
    recurrentLayers: c.recurrentLayers || null,
    isMoe: !!c.isMoe,
    moeExpertShare: c.moeExpertShare ?? null,
    expertCount: c.expertCount || 0,
    nextnPredictLayers: c.nextnPredictLayers || 0,
  });
  return {
    ...pack(caps),
    draft: draftCaps?.fileSizeBytes && draftCaps.blockCount ? pack(draftCaps) : null,
    mmprojFileSizeBytes: mmprojFileSizeBytes && mmprojFileSizeBytes > 0 ? mmprojFileSizeBytes : 0,
  };
}

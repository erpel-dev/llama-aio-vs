import * as os from "os";
import { GpuMemoryInfo } from "./gpuInfo";
import { ModelCapabilities } from "./ggufMetadata";
import { LlamaLoadSettings } from "./types";

export interface MemoryBarSegment {
  key: "weights" | "kv" | "overhead";
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
  kvBytes: number;
  kvOnGpu: boolean;
  overheadBytes: number;
  gpuOverheadBytes: number;
  cpuOverheadBytes: number;
  totalGpuBytes: number;
  totalCpuBytes: number;
  gpuTotalBytes?: number;
  gpuUsedBytes?: number;
  gpuName?: string;
  systemRamTotalBytes?: number;
  /** Stacked bars for the sidebar. */
  charts: {
    vram: MemoryBarChart;
    ram: MemoryBarChart;
  };
  /** Estimated VRAM exceeds detected GPU memory (with headroom). */
  willSpill: boolean;
  /** Soft warnings (partial offload, MoE on CPU, etc.). */
  warnings: string[];
  /** Short human lines for the sidebar. */
  lines: string[];
}

const GiB = 1024 ** 3;
const MiB = 1024 ** 2;

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
 * Rough KV-cache size (default f16 K/V in llama.cpp).
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
  contextLength: number
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
  const elemBytes = 2; // f16

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

    const isSwa = !!(swa && pattern && pattern[i]);
    const nKv = Math.max(1, perLayerKv?.[i] ?? defaultKvHeads);
    const keyDim = isSwa
      ? Math.max(1, caps.keyLengthSwa || Math.floor(defaultKeyDim / 2) || defaultKeyDim)
      : defaultKeyDim;
    const valDim = isSwa
      ? Math.max(1, caps.valueLengthSwa || Math.floor(defaultValDim / 2) || defaultValDim)
      : defaultValDim;
    const tokens = isSwa && swa ? Math.min(contextLength, swa) : contextLength;
    total += (nKv * keyDim + nKv * valDim) * tokens * elemBytes;
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

/**
 * Estimate GPU/CPU footprint for current load settings.
 * Intentionally approximate — good enough for spill warnings and UI hints.
 * @param options.cpuOnly When true (CPU llama.cpp build), everything is attributed to system RAM.
 */
export function estimateMemory(
  caps: ModelCapabilities | undefined,
  settings: LlamaLoadSettings,
  gpu?: GpuMemoryInfo,
  options?: { cpuOnly?: boolean }
): MemoryEstimate | undefined {
  if (!caps?.fileSizeBytes || !caps.blockCount) {
    return undefined;
  }

  const cpuOnly = !!options?.cpuOnly;
  const fileSize = caps.fileSizeBytes;
  const nLayers = caps.blockCount;
  const onGpu = cpuOnly ? 0 : layersOnGpu(settings, nLayers);
  const frac = onGpu / nLayers;

  // MoE experts are a large share of the file; --n-cpu-moe keeps those of the first N layers on CPU.
  let gpuWeights = fileSize * frac;
  if (!cpuOnly && caps.isMoe && settings.nCpuMoe > 0 && onGpu > 0) {
    const expertShare = 0.7;
    const moeCpuLayers = Math.min(settings.nCpuMoe, onGpu);
    gpuWeights = Math.max(0, gpuWeights - fileSize * (moeCpuLayers / nLayers) * expertShare);
  }
  const cpuWeights = Math.max(0, fileSize - gpuWeights);

  const kvBytes = estimateKvBytes(caps, settings.contextLength);
  const fullAttnLayers = countFullAttentionLayers(caps);
  const kvOnGpu = !cpuOnly && settings.offloadKvCacheToGpu && onGpu > 0;
  // Compute / graph / batch scratch — grows a bit with batch size.
  const overheadBytes = Math.round(
    400 * MiB + Math.min(settings.evalBatchSize, 4096) * 64 * 1024
  );
  const gpuOverheadBytes = onGpu > 0 ? overheadBytes : 0;
  const cpuOverheadBytes = onGpu > 0 ? Math.round(overheadBytes * 0.15) : Math.round(overheadBytes * 0.5);
  const gpuKvBytes = kvOnGpu ? kvBytes : 0;
  const cpuKvBytes = kvOnGpu ? 0 : kvBytes;

  const totalGpuBytes = gpuWeights + gpuKvBytes + gpuOverheadBytes;
  const totalCpuBytes = cpuWeights + cpuKvBytes + cpuOverheadBytes;
  const systemRamTotalBytes = os.totalmem();

  const warnings: string[] = [];
  let willSpill = false;

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
      `KV cache (~${formatBytes(kvBytes)} at this context) is in system RAM, not VRAM.`
    );
  }
  if (!cpuOnly && caps.isMoe && settings.nCpuMoe > 0) {
    warnings.push(
      `CPU MoE layers = ${settings.nCpuMoe}: expert weights for those layers stay in system RAM.`
    );
  }

  // Keep ~8% free for the compositor / driver; going above this often spills even if
  // the raw estimate is still slightly under the advertised GPU total.
  const usableFraction = 0.92;
  if (!cpuOnly && gpu?.totalBytes) {
    const usableBytes = gpu.totalBytes * usableFraction;
    const pct = Math.round((totalGpuBytes / gpu.totalBytes) * 100);
    if (totalGpuBytes > gpu.totalBytes) {
      willSpill = true;
      warnings.unshift(
        `Estimated VRAM ~${formatBytes(totalGpuBytes)} is over the full ${formatBytes(gpu.totalBytes)} GPU (${pct}%). Expect spill to system RAM (much slower). Lower Context Length, GPU Offload, or use a smaller quant.`
      );
    } else if (totalGpuBytes > usableBytes) {
      willSpill = true;
      warnings.unshift(
        `Tight on VRAM: ~${formatBytes(totalGpuBytes)} of ${formatBytes(gpu.totalBytes)} (${pct}%). Only ~${formatBytes(gpu.totalBytes - usableBytes)} is left as safe headroom for the driver — llama.cpp often spills to system RAM at this point. Lower Context Length or GPU Offload.`
      );
    } else if (totalGpuBytes > gpu.totalBytes * 0.8) {
      warnings.push(
        `Getting full: ~${formatBytes(totalGpuBytes)} of ${formatBytes(gpu.totalBytes)} VRAM (${pct}%). Leave some free for the display driver.`
      );
    }
  }

  if (cpuOnly && systemRamTotalBytes && totalCpuBytes > systemRamTotalBytes * 0.9) {
    willSpill = true;
    warnings.unshift(
      `Estimated system RAM ~${formatBytes(totalCpuBytes)} is very high vs ${formatBytes(systemRamTotalBytes)}. Lower Context Length or use a smaller model/quant.`
    );
  }

  const lines: string[] = [];
  if (cpuOnly) {
    lines.push("Backend: CPU (x64) — GPU Offload / VRAM not used");
  } else if (gpu?.totalBytes) {
    const free =
      gpu.usedBytes !== undefined ? Math.max(0, gpu.totalBytes - gpu.usedBytes) : undefined;
    lines.push(
      `GPU: ${formatBytes(gpu.totalBytes)}` +
        (gpu.name ? ` (${gpu.name})` : "") +
        (free !== undefined ? ` · ~${formatBytes(free)} free now` : "")
    );
  } else {
    lines.push("GPU VRAM: unknown (could not detect)");
  }
  lines.push(`System RAM: ${formatBytes(systemRamTotalBytes)}`);
  if (cpuOnly) {
    lines.push(`Weights in RAM: ~${formatBytes(cpuWeights)} (${nLayers} layers)`);
    lines.push(
      `KV cache @ ${settings.contextLength.toLocaleString()} ctx: ~${formatBytes(kvBytes)} (system RAM)` +
        (fullAttnLayers < nLayers ? ` · ${fullAttnLayers}/${nLayers} full-attn layers` : "")
    );
    lines.push(`Est. total system RAM: ~${formatBytes(totalCpuBytes)}`);
  } else {
    lines.push(
      `Weights on GPU: ~${formatBytes(gpuWeights)} (${onGpu}/${nLayers} layers)` +
        (cpuWeights > MiB ? ` · RAM: ~${formatBytes(cpuWeights)}` : "")
    );
    lines.push(
      `KV cache @ ${settings.contextLength.toLocaleString()} ctx: ~${formatBytes(kvBytes)}` +
        (kvOnGpu ? " (GPU)" : " (CPU RAM)") +
        (fullAttnLayers < nLayers ? ` · ${fullAttnLayers}/${nLayers} full-attn layers` : "")
    );
    lines.push(
      `Est. total VRAM: ~${formatBytes(totalGpuBytes)}` +
        (totalCpuBytes > MiB ? ` · system RAM: ~${formatBytes(totalCpuBytes)}` : "")
    );
  }
  lines.push("Estimate only — actual use varies by quant, MoE, and backend.");

  const charts = {
    vram: {
      title: "VRAM",
      segments: [
        { key: "weights" as const, label: "Weights", bytes: gpuWeights },
        { key: "kv" as const, label: "KV cache", bytes: gpuKvBytes },
        { key: "overhead" as const, label: "Overhead", bytes: gpuOverheadBytes },
      ],
      totalBytes: totalGpuBytes,
      capacityBytes: cpuOnly ? undefined : gpu?.totalBytes,
    },
    ram: {
      title: "System RAM",
      segments: [
        { key: "weights" as const, label: "Weights", bytes: cpuWeights },
        { key: "kv" as const, label: "KV cache", bytes: cpuKvBytes },
        { key: "overhead" as const, label: "Overhead", bytes: cpuOverheadBytes },
      ],
      totalBytes: totalCpuBytes,
      capacityBytes: systemRamTotalBytes,
    },
  };

  return {
    fileSizeBytes: fileSize,
    layersTotal: nLayers,
    layersOnGpu: onGpu,
    gpuWeightsBytes: gpuWeights,
    cpuWeightsBytes: cpuWeights,
    kvBytes,
    kvOnGpu,
    overheadBytes,
    gpuOverheadBytes,
    cpuOverheadBytes,
    totalGpuBytes,
    totalCpuBytes,
    gpuTotalBytes: gpu?.totalBytes,
    gpuUsedBytes: gpu?.usedBytes,
    gpuName: gpu?.name,
    systemRamTotalBytes,
    charts,
    willSpill,
    warnings,
    lines,
  };
}

/** Compact JSON-safe payload for the webview live calculator. */
export function memoryEstimateInputs(caps: ModelCapabilities | undefined): Record<string, unknown> | null {
  if (!caps?.fileSizeBytes) {
    return null;
  }
  return {
    fileSizeBytes: caps.fileSizeBytes,
    blockCount: caps.blockCount,
    embeddingLength: caps.embeddingLength || 0,
    attentionHeadCount: caps.attentionHeadCount || 0,
    attentionHeadCountKv: caps.attentionHeadCountKv || 0,
    attentionHeadCountKvPerLayer: caps.attentionHeadCountKvPerLayer || null,
    keyLength: caps.keyLength || 0,
    valueLength: caps.valueLength || 0,
    keyLengthSwa: caps.keyLengthSwa || 0,
    valueLengthSwa: caps.valueLengthSwa || 0,
    slidingWindow: caps.slidingWindow || 0,
    slidingWindowPattern: caps.slidingWindowPattern || null,
    fullAttentionInterval: caps.fullAttentionInterval || 0,
    recurrentLayers: caps.recurrentLayers || null,
    isMoe: !!caps.isMoe,
  };
}

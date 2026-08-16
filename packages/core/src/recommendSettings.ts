import { detectGpuMemory, detectGpus, GpuMemoryInfo } from "./gpuInfo";
import { tensorSplitForMainShare } from "./gpuSplit";
import { ModelCapabilities } from "./ggufMetadata";
import { estimateMemory, MemoryEstimate } from "./memoryEstimate";
import { isMtpDraftFileName } from "./modelLibrary";
import { LlamaLoadSettings } from "./types";

const GiB = 1024 ** 3;
/** Prefer this context when the model allows it. */
export const PREFERRED_CONTEXT = 65536;
/** Keep at least this much VRAM unused (estimate vs GPU total). */
export const VRAM_HEADROOM_BYTES = 2 * GiB;

/** Main-GPU weight shares to try when fitting a dual-GPU split (device-order). */
const DUAL_GPU_MAIN_SHARES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.45, 0.4, 0.35, 0.3];

export interface RecommendOptions {
  cpuOnly?: boolean;
  gpu?: GpuMemoryInfo;
  /** All detected GPUs. When ≥2, recommend a tensor split before spilling to RAM. */
  gpus?: GpuMemoryInfo[];
  /** Target free VRAM; default 2 GiB (single GPU). Dual GPU uses per-card 92% like the bars. */
  headroomBytes?: number;
  preferredContext?: number;
}

function targetContext(caps: ModelCapabilities, preferred: number): number {
  return Math.min(Math.max(512, preferred), Math.max(512, caps.maxContextLength || preferred));
}

function resolveGpus(options: RecommendOptions): GpuMemoryInfo[] {
  if (options.gpus?.length) {
    return options.gpus.filter((g) => g.totalBytes > 0);
  }
  if (options.gpu?.totalBytes) {
    return [options.gpu];
  }
  return detectGpus().filter((g) => g.totalBytes > 0);
}

function peakOccupancy(est: MemoryEstimate): number {
  const charts = [est.charts.vram, est.charts.vram2].filter(
    (c): c is NonNullable<typeof c> => !!c
  );
  let peak = 0;
  for (const c of charts) {
    if (c.capacityBytes && c.capacityBytes > 0) {
      peak = Math.max(peak, c.totalBytes / c.capacityBytes);
    }
  }
  return peak;
}

function estimateFor(
  caps: ModelCapabilities,
  settings: LlamaLoadSettings,
  gpu: GpuMemoryInfo | undefined,
  cpuOnly: boolean,
  gpus: GpuMemoryInfo[]
): MemoryEstimate | undefined {
  return estimateMemory(caps, settings, gpu, {
    cpuOnly,
    gpus: gpus.length ? gpus : gpu ? [gpu] : undefined,
  });
}

function fitsHeadroom(
  caps: ModelCapabilities,
  settings: LlamaLoadSettings,
  gpu: GpuMemoryInfo | undefined,
  cpuOnly: boolean,
  headroom: number,
  gpus: GpuMemoryInfo[]
): boolean {
  const est = estimateFor(caps, settings, gpu, cpuOnly, gpus);
  if (!est) {
    return false;
  }
  if (gpus.length >= 2) {
    return !est.willSpill;
  }
  if (!gpu?.totalBytes) {
    return false;
  }
  return gpu.totalBytes - est.totalGpuBytes >= headroom;
}

/** Physical batch to aim for when prefill is fully GPU-bound. */
const FAST_PREFILL_UBATCH = 1024;
/** Extra VRAM (beyond the normal headroom) required before raising -ub. */
const UBATCH_EXTRA_HEADROOM = 1 * GiB;

/**
 * Agent prompts are prefill-heavy, so a larger physical batch (-ub) is the main
 * lever on time-to-first-token — but only when the whole model is on the GPU and
 * there is room to spare. Partial offload is CPU-bound anyway.
 */
function tuneBatchSizes(
  settings: LlamaLoadSettings,
  caps: ModelCapabilities,
  gpu: GpuMemoryInfo | undefined,
  gpus: GpuMemoryInfo[],
  headroom: number
): LlamaLoadSettings {
  const fullyOffloaded =
    settings.gpuOffload >= 99 || settings.gpuOffload >= Math.max(1, caps.blockCount || 1);
  if (
    !gpu?.totalBytes ||
    !fullyOffloaded ||
    settings.nCpuMoe > 0 ||
    settings.evalBatchSize < FAST_PREFILL_UBATCH ||
    settings.physicalBatchSize >= FAST_PREFILL_UBATCH
  ) {
    return settings;
  }
  const candidate = { ...settings, physicalBatchSize: FAST_PREFILL_UBATCH };
  return fitsHeadroom(caps, candidate, gpu, false, headroom + UBATCH_EXTRA_HEADROOM, gpus)
    ? candidate
    : settings;
}

/**
 * Speculative mode from GGUF: enable MTP when the file reports next-n layers
 * or a sidecar `mtp-*.gguf` is already attached. Clears invalid MTP; preserves
 * an explicit DFlash setup (draft is separate) unless that draft is an MTP file.
 */
function recommendSpeculative(current: LlamaLoadSettings, caps: ModelCapabilities): Pick<
  LlamaLoadSettings,
  | "speculativeMode"
  | "maxDraftTokens"
  | "minDraftTokens"
  | "draftProbability"
  | "draftModelPath"
  | "draftGpuOffload"
> {
  const sidecarMtp = isMtpDraftFileName(current.draftModelPath);
  const mtpCapable = !!(caps.nextnPredictLayers && caps.nextnPredictLayers > 0) || sidecarMtp;
  const keepDraft = {
    draftModelPath: current.draftModelPath || "",
    draftGpuOffload: current.draftGpuOffload ?? 99,
  };
  // Keep an explicit DFlash setup even when the new GGUF also has MTP heads,
  // unless the attached draft is actually a sidecar MTP GGUF (model switch).
  if (current.speculativeMode === "dflash" && !sidecarMtp) {
    return {
      speculativeMode: "dflash",
      maxDraftTokens: current.maxDraftTokens > 0 ? current.maxDraftTokens : 15,
      minDraftTokens: current.minDraftTokens,
      draftProbability: current.draftProbability,
      ...keepDraft,
    };
  }
  if (mtpCapable) {
    return {
      speculativeMode: "mtp",
      maxDraftTokens: current.maxDraftTokens > 0 ? current.maxDraftTokens : sidecarMtp ? 4 : 2,
      minDraftTokens: current.minDraftTokens,
      draftProbability: current.draftProbability > 0 ? current.draftProbability : 0.75,
      ...keepDraft,
    };
  }
  return {
    speculativeMode: "off",
    maxDraftTokens: current.maxDraftTokens,
    minDraftTokens: current.minDraftTokens,
    draftProbability: current.draftProbability,
    ...keepDraft,
  };
}

/**
 * Full GPU offload with a tensor split that fills both cards. Prefers the most
 * even peak occupancy, then more weight on the Main GPU. Undefined if every
 * split still spills — caller then drops layers / raises --n-cpu-moe.
 */
function fitDualGpuFullOffload(
  base: LlamaLoadSettings,
  caps: ModelCapabilities,
  gpu: GpuMemoryInfo,
  gpus: GpuMemoryInfo[],
  headroom: number
): LlamaLoadSettings | undefined {
  const n = gpus.length;
  const mainGpu = Math.min(Math.max(0, base.mainGpu || 0), n - 1);
  const withAll: LlamaLoadSettings = {
    ...base,
    gpuOffload: 99,
    nCpuMoe: 0,
    splitMode: "layer",
    mainGpu,
  };
  let best: { settings: LlamaLoadSettings; peak: number; share: number } | undefined;
  for (const share of DUAL_GPU_MAIN_SHARES) {
    const candidate = {
      ...withAll,
      tensorSplit: tensorSplitForMainShare(share, mainGpu, n),
    };
    const est = estimateFor(caps, candidate, gpu, false, gpus);
    if (!est || est.willSpill) {
      continue;
    }
    if (!fitsHeadroom(caps, candidate, gpu, false, headroom, gpus)) {
      continue;
    }
    const peak = peakOccupancy(est);
    if (
      !best ||
      peak < best.peak - 0.005 ||
      (Math.abs(peak - best.peak) <= 0.005 && share > best.share)
    ) {
      best = { settings: candidate, peak, share };
    }
  }
  return best?.settings;
}

/**
 * Pick context / GPU offload / CPU MoE / speculative defaults from model + VRAM.
 *
 * - Dual GPU: split across cards first (full -ngl); only then RAM / --n-cpu-moe
 * - MoE: full GPU offload, minimal --n-cpu-moe that leaves headroom
 * - Dense: max layers that leave headroom
 * - CPU backend: context + speculative only (GPU settings left alone; start path forces -ngl 0)
 * - MTP: speculativeMode = mtp when GGUF nextn_predict_layers > 0 or a sidecar MTP draft is set
 */
export function recommendLoadSettings(
  current: LlamaLoadSettings,
  caps: ModelCapabilities,
  options: RecommendOptions = {}
): LlamaLoadSettings {
  const preferred = options.preferredContext ?? PREFERRED_CONTEXT;
  const headroom = options.headroomBytes ?? VRAM_HEADROOM_BYTES;
  const cpuOnly = !!options.cpuOnly;
  const contextLength = targetContext(caps, preferred);
  const speculative = recommendSpeculative(current, caps);

  if (cpuOnly) {
    return {
      ...current,
      contextLength,
      nCpuMoe: caps.isMoe ? current.nCpuMoe : 0,
      ...speculative,
    };
  }

  const gpus = resolveGpus(options);
  const gpu = gpus[0] || options.gpu || detectGpuMemory();
  const nLayers = Math.max(1, caps.blockCount || 1);
  const base: LlamaLoadSettings = {
    ...current,
    contextLength,
    nCpuMoe: 0,
    offloadKvCacheToGpu: current.offloadKvCacheToGpu,
    ...speculative,
    ...(gpus.length >= 2
      ? {
          splitMode: current.splitMode || "layer",
          mainGpu: Math.min(Math.max(0, current.mainGpu || 0), gpus.length - 1),
        }
      : { tensorSplit: "" }),
  };

  // No VRAM info: prefer “all layers” and no CPU MoE (user can tune).
  if (!gpu?.totalBytes) {
    return {
      ...base,
      gpuOffload: 99,
      nCpuMoe: 0,
    };
  }

  const fitted = fitOffload(base, caps, gpu, gpus, nLayers, headroom);
  return tuneBatchSizes(fitted, caps, gpu, gpus, headroom);
}

function fitOffload(
  base: LlamaLoadSettings,
  caps: ModelCapabilities,
  gpu: GpuMemoryInfo,
  gpus: GpuMemoryInfo[],
  nLayers: number,
  headroom: number
): LlamaLoadSettings {
  if (gpus.length >= 2) {
    const splitFit = fitDualGpuFullOffload(base, caps, gpu, gpus, headroom);
    if (splitFit) {
      return splitFit;
    }
    // Both cards still too small for the whole model — keep a VRAM-proportional
    // split and spill the remainder to RAM (layers or MoE experts).
    const mainGpu = base.mainGpu || 0;
    const vramSum = gpus.reduce((a, g) => a + g.totalBytes, 0);
    const mainShare = vramSum > 0 ? (gpus[mainGpu]?.totalBytes ?? 0) / vramSum : 0.5;
    base = {
      ...base,
      tensorSplit: tensorSplitForMainShare(mainShare, mainGpu, gpus.length),
      splitMode: "layer",
    };
  }

  if (caps.isMoe) {
    // Max GPU offload; raise CPU MoE only as needed for headroom.
    const withAllGpu: LlamaLoadSettings = { ...base, gpuOffload: 99, nCpuMoe: 0 };
    let nCpuMoe = nLayers; // worst case if nothing fits
    for (let n = 0; n <= nLayers; n++) {
      const candidate = { ...withAllGpu, nCpuMoe: n };
      if (fitsHeadroom(caps, candidate, gpu, false, headroom, gpus)) {
        nCpuMoe = n;
        break;
      }
    }
    return { ...withAllGpu, nCpuMoe };
  }

  // Dense: as many layers as possible while keeping headroom.
  // Prefer 99 (“all”) when the full model fits.
  for (let n = nLayers; n >= 0; n--) {
    const gpuOffload = n >= nLayers ? 99 : n;
    const candidate = { ...base, gpuOffload, nCpuMoe: 0 };
    if (fitsHeadroom(caps, candidate, gpu, false, headroom, gpus)) {
      return candidate;
    }
  }

  return { ...base, gpuOffload: 0, nCpuMoe: 0 };
}

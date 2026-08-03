import { detectGpuMemory, GpuMemoryInfo } from "./gpuInfo";
import { ModelCapabilities } from "./ggufMetadata";
import { estimateMemory } from "./memoryEstimate";
import { LlamaLoadSettings } from "./types";

const GiB = 1024 ** 3;
/** Prefer this context when the model allows it. */
export const PREFERRED_CONTEXT = 65536;
/** Keep at least this much VRAM unused (estimate vs GPU total). */
export const VRAM_HEADROOM_BYTES = 2 * GiB;

export interface RecommendOptions {
  cpuOnly?: boolean;
  gpu?: GpuMemoryInfo;
  /** Target free VRAM; default 2 GiB. */
  headroomBytes?: number;
  preferredContext?: number;
}

function targetContext(caps: ModelCapabilities, preferred: number): number {
  return Math.min(Math.max(512, preferred), Math.max(512, caps.maxContextLength || preferred));
}

function estimatedFreeVram(
  caps: ModelCapabilities,
  settings: LlamaLoadSettings,
  gpu: GpuMemoryInfo,
  cpuOnly: boolean
): number | undefined {
  const est = estimateMemory(caps, settings, gpu, { cpuOnly });
  if (!est || !gpu.totalBytes) {
    return undefined;
  }
  return gpu.totalBytes - est.totalGpuBytes;
}

function fitsHeadroom(
  caps: ModelCapabilities,
  settings: LlamaLoadSettings,
  gpu: GpuMemoryInfo,
  cpuOnly: boolean,
  headroom: number
): boolean {
  const free = estimatedFreeVram(caps, settings, gpu, cpuOnly);
  return free !== undefined && free >= headroom;
}

/**
 * Speculative mode from GGUF: enable MTP when the file reports next-n layers.
 * Non-MTP models get speculative turned off so a previous MTP selection does not stick.
 */
function recommendSpeculative(current: LlamaLoadSettings, caps: ModelCapabilities): Pick<
  LlamaLoadSettings,
  "speculativeMode" | "maxDraftTokens" | "minDraftTokens" | "draftProbability"
> {
  const mtpCapable = !!(caps.nextnPredictLayers && caps.nextnPredictLayers > 0);
  if (!mtpCapable) {
    return {
      speculativeMode: "off",
      maxDraftTokens: current.maxDraftTokens,
      minDraftTokens: current.minDraftTokens,
      draftProbability: current.draftProbability,
    };
  }
  return {
    speculativeMode: "mtp",
    // Keep existing draft knobs if the user already tuned them; otherwise use defaults.
    maxDraftTokens: current.maxDraftTokens > 0 ? current.maxDraftTokens : 2,
    minDraftTokens: current.minDraftTokens,
    draftProbability: current.draftProbability > 0 ? current.draftProbability : 0.75,
  };
}

/**
 * Pick context / GPU offload / CPU MoE / speculative defaults from model + VRAM.
 *
 * - MoE: full GPU offload, minimal --n-cpu-moe that leaves ≥2 GiB VRAM free
 * - Dense: max layers that leave ≥2 GiB VRAM free
 * - CPU backend: context + speculative only (GPU settings left alone; start path forces -ngl 0)
 * - MTP: speculativeMode = mtp when GGUF nextn_predict_layers > 0
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

  const gpu = options.gpu ?? detectGpuMemory();
  const nLayers = Math.max(1, caps.blockCount || 1);
  const base: LlamaLoadSettings = {
    ...current,
    contextLength,
    nCpuMoe: 0,
    offloadKvCacheToGpu: current.offloadKvCacheToGpu,
    ...speculative,
  };

  // No VRAM info: prefer “all layers” and no CPU MoE (user can tune).
  if (!gpu?.totalBytes) {
    return {
      ...base,
      gpuOffload: 99,
      nCpuMoe: 0,
    };
  }

  if (caps.isMoe) {
    // Max GPU offload; raise CPU MoE only as needed for headroom.
    const withAllGpu: LlamaLoadSettings = { ...base, gpuOffload: 99, nCpuMoe: 0 };
    let nCpuMoe = nLayers; // worst case if nothing fits
    for (let n = 0; n <= nLayers; n++) {
      const candidate = { ...withAllGpu, nCpuMoe: n };
      if (fitsHeadroom(caps, candidate, gpu, false, headroom)) {
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
    if (fitsHeadroom(caps, candidate, gpu, false, headroom)) {
      return candidate;
    }
  }

  return { ...base, gpuOffload: 0, nCpuMoe: 0 };
}

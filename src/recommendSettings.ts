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
 * Pick context / GPU offload / CPU MoE defaults from model + VRAM.
 *
 * - MoE: full GPU offload, minimal --n-cpu-moe that leaves ≥2 GiB VRAM free
 * - Dense: max layers that leave ≥2 GiB VRAM free
 * - CPU backend: context only (GPU settings left alone; start path forces -ngl 0)
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

  if (cpuOnly) {
    return { ...current, contextLength, nCpuMoe: caps.isMoe ? current.nCpuMoe : 0 };
  }

  const gpu = options.gpu ?? detectGpuMemory();
  const nLayers = Math.max(1, caps.blockCount || 1);
  const base: LlamaLoadSettings = {
    ...current,
    contextLength,
    nCpuMoe: 0,
    offloadKvCacheToGpu: current.offloadKvCacheToGpu,
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

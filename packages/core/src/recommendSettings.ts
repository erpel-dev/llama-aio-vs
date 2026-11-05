import { detectGpus, GpuMemoryInfo, isIntegratedGpu } from "./gpuInfo";
import {
  capacityAwareTensorSplit,
  type LayerWeightMassOptions,
  mainShareFromSplit,
  parseTensorSplit,
  retargetTensorSplitMainShare,
  tensorSplitForMainShare,
  tensorSplitForTargetWeightShare,
} from "./gpuSplit";
import { isQwen4expArchitecture, ModelCapabilities } from "./ggufMetadata";
import { estimateMemory, gpuBarCapacityBytes, MemoryEstimate, VRAM_HEADROOM_BYTES } from "./memoryEstimate";
import { isMtpBakedInFile, isMtpSidecarFile } from "./modelLibrary";
import { LlamaLoadSettings, recommendedMaxDraftTokens, speculativeUsesDflash, speculativeUsesNgram } from "./types";

const GiB = 1024 ** 3;
/** Prefer this context when the model allows it. */
export const PREFERRED_CONTEXT = 65536;
/** Floor for the “fit largest context” search (and CUDA-style alignment). */
export const FIT_CONTEXT_MIN = 8192;
export const FIT_CONTEXT_ALIGN = 256;

export interface RecommendOptions {
  cpuOnly?: boolean;
  gpu?: GpuMemoryInfo;
  /** All detected GPUs. When ≥2, recommend a tensor split before spilling to RAM. */
  gpus?: GpuMemoryInfo[];
  /** Target free VRAM vs total; default 2 GiB on every card. */
  headroomBytes?: number;
  preferredContext?: number;
}

function targetContext(caps: ModelCapabilities, preferred: number): number {
  return Math.min(Math.max(512, preferred), Math.max(512, caps.maxContextLength || preferred));
}

function alignContext(n: number): number {
  return Math.max(FIT_CONTEXT_ALIGN, Math.floor(n / FIT_CONTEXT_ALIGN) * FIT_CONTEXT_ALIGN);
}

/**
 * Largest context in [8192, model max] (aligned to 256) whose estimate stays
 * inside the 2 GiB per-card headroom. Falls back to 8192 (or the model max if
 * smaller) when even that spills. With no VRAM info, returns the agent default.
 */
export function fittingContextLength(
  caps: ModelCapabilities,
  settings: LlamaLoadSettings,
  options: RecommendOptions = {}
): number {
  const maxCtx = Math.max(1, caps.maxContextLength || PREFERRED_CONTEXT);
  const high = Math.min(maxCtx, alignContext(maxCtx));
  const lowMin = Math.min(FIT_CONTEXT_MIN, high);
  const cpuOnly = !!options.cpuOnly;
  const gpus = resolveGpus(options);
  const gpu = gpus[0] || options.gpu;
  if (cpuOnly || !gpu?.totalBytes) {
    return Math.min(PREFERRED_CONTEXT, maxCtx);
  }

  const fits = (ctx: number): boolean => {
    const est = estimateFor(caps, { ...settings, contextLength: ctx }, gpu, cpuOnly, gpus);
    return !!est && !est.willSpill;
  };

  if (fits(high)) {
    return high;
  }
  if (!fits(lowMin)) {
    return lowMin;
  }

  let lo = lowMin;
  let hi = high;
  let best = lowMin;
  while (lo <= hi) {
    let mid = alignContext(Math.floor((lo + hi) / 2));
    if (mid < lo) {
      mid = lo;
    }
    if (mid > hi) {
      break;
    }
    if (fits(mid)) {
      best = mid;
      lo = mid + FIT_CONTEXT_ALIGN;
    } else {
      hi = mid - FIT_CONTEXT_ALIGN;
    }
  }
  return Math.min(best, maxCtx);
}

function resolveGpus(options: RecommendOptions): GpuMemoryInfo[] {
  if (options.cpuOnly) return [];
  // An explicit empty result from an async probe must not trigger a sync retry.
  if (options.gpus !== undefined) {
    return options.gpus.filter((g) => g.totalBytes > 0);
  }
  if (options.gpu?.totalBytes) {
    return [options.gpu];
  }
  return detectGpus().filter((g) => g.totalBytes > 0);
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
  // An APU's weights and KV live in GTT, not its small VRAM carve-out.
  const capacity = gpuBarCapacityBytes(gpu);
  if (!capacity) {
    return false;
  }
  return capacity - est.totalGpuBytes >= headroom;
}

/** iGPU carve-outs at or below this rely on shared memory (GTT) for the model. */
const SMALL_CARVE_OUT_BYTES = 2 * GiB;
/** Physical batch to aim for when prefill is fully GPU-bound. */
const FAST_PREFILL_UBATCH = 1024;
/** Extra VRAM (beyond the normal headroom) required before raising -ub. */
const UBATCH_EXTRA_HEADROOM = 1 * GiB;
/**
 * Physical batch when MoE experts / FFN run from system RAM. llama.cpp copies
 * those weights to the GPU once per ubatch for prompt processing, so a big
 * ubatch amortizes the copy. Measured on an 8 GB RX 9060 XT (Ryzen 5700X3D,
 * 5.7–7.7k prompt): flint 24 GB 54 → 230 tok/s and Groot 19B 143 → 560 tok/s
 * going 512 → 4096, generation unchanged (±1%). 8192 gained more prompt speed
 * but its compute buffer pushed Groot over the card.
 */
const CPU_EXPERT_PREFILL_UBATCH = 4096;
/**
 * Extra `--n-cpu-moe` layers worth trading for a bigger ubatch. Groot 19B on
 * 8 GB: -ub 4096 with 3 more CPU layers ran prompts 3.7x faster (143 → 531
 * tok/s) for −6% generation; 6 more layers cost −11%.
 */
const CPU_EXPERT_UBATCH_MAX_EXTRA_LAYERS = 3;

/**
 * Agent prompts are prefill-heavy, so a larger physical batch (-ub) is the main
 * lever on time-to-first-token — but only when the whole model is on the GPU and
 * there is room to spare. Partial offload is CPU-bound anyway.
 */
/**
 * Experts in system RAM: raise -ub (and -b) to {@link CPU_EXPERT_PREFILL_UBATCH}
 * (else half of it) for ~3–4x prompt speed, moving at most
 * {@link CPU_EXPERT_UBATCH_MAX_EXTRA_LAYERS} more expert layers to the CPU
 * to pay for the larger compute buffer. Multi-GPU keeps its fitted split and
 * only takes the bigger batch when it fits as is.
 */
function tuneCpuExpertBatch(
  settings: LlamaLoadSettings,
  caps: ModelCapabilities,
  gpu: GpuMemoryInfo,
  gpus: GpuMemoryInfo[],
  headroom: number
): LlamaLoadSettings {
  const nLayers = Math.max(1, caps.blockCount || 1);
  const maxExtra = caps.isMoe && settings.nCpuMoe > 0 && gpus.length < 2 ? CPU_EXPERT_UBATCH_MAX_EXTRA_LAYERS : 0;
  for (const ubatch of [CPU_EXPERT_PREFILL_UBATCH, CPU_EXPERT_PREFILL_UBATCH / 2]) {
    if (ubatch <= settings.physicalBatchSize) {
      break;
    }
    for (let extra = 0; extra <= maxExtra; extra++) {
      const candidate: LlamaLoadSettings = {
        ...settings,
        physicalBatchSize: ubatch,
        evalBatchSize: Math.max(settings.evalBatchSize, ubatch),
        nCpuMoe: Math.min(nLayers, settings.nCpuMoe + extra),
      };
      if (fitsHeadroom(caps, candidate, gpu, false, headroom, gpus)) {
        return candidate;
      }
    }
  }
  return settings;
}

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
    gpu?.totalBytes &&
    fullyOffloaded &&
    (settings.nCpuMoe > 0 || settings.nCpuFfn > 0) &&
    settings.physicalBatchSize < CPU_EXPERT_PREFILL_UBATCH
  ) {
    return tuneCpuExpertBatch(settings, caps, gpu, gpus, headroom);
  }
  if (
    !gpu?.totalBytes ||
    // Measured on Vega (Renoir/Cezanne): -ub 1024/2048 gives the same prefill
    // as 512 while growing the compute buffer in shared RAM. An APU with a
    // large BIOS carve-out (Strix Halo) is tuned like a discrete card.
    (isIntegratedGpu(gpu) && gpu.totalBytes <= SMALL_CARVE_OUT_BYTES) ||
    !fullyOffloaded ||
    settings.nCpuMoe > 0 ||
    settings.nCpuFfn > 0 ||
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
  const rawDraft = current.draftModelPath || "";
  let draftModelPath = isMtpBakedInFile({ path: rawDraft }) ? "" : rawDraft;
  if (isQwen4expArchitecture(caps.architecture) && isMtpSidecarFile({ path: draftModelPath })) {
    draftModelPath = "";
  }
  const sidecarMtp = isMtpSidecarFile({ path: draftModelPath });
  const mtpCapable = !!(caps.nextnPredictLayers && caps.nextnPredictLayers > 0) || sidecarMtp;
  const keepDraft = {
    draftModelPath,
    draftGpuOffload: current.draftGpuOffload ?? 99,
  };
  // Keep an explicit DFlash setup even when the new GGUF also has MTP heads,
  // unless the attached draft is actually a sidecar MTP GGUF (model switch).
  if (speculativeUsesDflash(current.speculativeMode) && !sidecarMtp) {
    const mode = speculativeUsesNgram(current.speculativeMode) ? "ngram-dflash" : "dflash";
    return {
      speculativeMode: mode,
      maxDraftTokens: recommendedMaxDraftTokens(mode, current.maxDraftTokens),
      minDraftTokens: current.minDraftTokens,
      draftProbability: current.draftProbability > 0 ? current.draftProbability : 0.75,
      ...keepDraft,
    };
  }
  if (mtpCapable) {
    const mode = speculativeUsesNgram(current.speculativeMode) ? "ngram-mtp" : "mtp";
    return {
      speculativeMode: mode,
      maxDraftTokens: recommendedMaxDraftTokens(mode, current.maxDraftTokens, sidecarMtp),
      minDraftTokens: current.minDraftTokens,
      draftProbability: current.draftProbability > 0 ? current.draftProbability : 0.75,
      ...keepDraft,
    };
  }
  if (speculativeUsesNgram(current.speculativeMode)) {
    return {
      speculativeMode: "ngram",
      maxDraftTokens: current.maxDraftTokens,
      minDraftTokens: current.minDraftTokens,
      draftProbability: current.draftProbability,
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
 * Reserve allocations parked on Main, including the projector and a draft
 * explicitly pinned there. An Auto draft follows the main model's split.
 */
function mainParkedBytes(est: MemoryEstimate, settings: LlamaLoadSettings): number {
  const vision = est.charts.vram.segments.find(segment => segment.key === "vision")?.bytes || 0;
  const pinnedDraft = settings.draftDevice
    ? est.charts.vram.segments.find(segment => segment.key === "draft")?.bytes || 0 : 0;
  return est.gpuOverheadBytes + vision + pinnedDraft;
}

function splitableGpuBytes(est: MemoryEstimate): number {
  return est.gpuWeightsBytes + (est.kvOnGpu ? est.kvBytes : 0);
}

function vramTotals(gpus: GpuMemoryInfo[]): number[] {
  return gpus.map((g) => g.totalBytes);
}

function uniqueSplits(splits: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const s of splits) {
    if (!s || seen.has(s)) {
      continue;
    }
    seen.add(s);
    out.push(s);
  }
  return out;
}

function weightMassOptions(caps: ModelCapabilities, settings: LlamaLoadSettings): LayerWeightMassOptions {
  return {
    isMoe: caps.isMoe,
    nCpuMoe: settings.nCpuMoe,
    moeExpertShare: caps.moeExpertShare,
    nCpuFfn: settings.nCpuFfn,
    denseFfnShare: caps.denseFfnShare,
  };
}

function layersOnGpuForSplit(caps: ModelCapabilities, settings: LlamaLoadSettings): number {
  const nLayers = Math.max(1, caps.blockCount || 1);
  return settings.gpuOffload >= 99 ? nLayers : Math.min(nLayers, Math.max(0, settings.gpuOffload));
}

/** Hold a weight-share target while `--n-cpu-moe` / `--n-cpu-ffn` changes layer mass. */
function retargetTensorSplitToWeightShare(
  settings: LlamaLoadSettings,
  caps: ModelCapabilities,
  gpus: GpuMemoryInfo[],
  targetShare: number
): LlamaLoadSettings {
  if (gpus.length < 2 || parseTensorSplit(settings.tensorSplit).length < 2) {
    return settings;
  }
  const next = tensorSplitForTargetWeightShare(
    targetShare,
    settings.mainGpu,
    gpus.length,
    Math.max(1, caps.blockCount || 1),
    layersOnGpuForSplit(caps, settings),
    settings.splitMode,
    weightMassOptions(caps, settings)
  );
  return next ? { ...settings, tensorSplit: next } : settings;
}

/**
 * Full GPU offload with a tensor split that fills Main first (CLIP / MTP / compute
 * already live there), then remaining cards back-to-front. Neighborhood-walks
 * Main’s share if the capacity seed still spills. Undefined if every split
 * still spills — caller then drops layers / raises --n-cpu-moe.
 */
function fitMultiGpuFullOffload(
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
    nCpuFfn: 0,
    splitMode: "layer",
    mainGpu,
  };
  const probe = estimateFor(caps, { ...withAll, tensorSplit: "", splitMode: "layer" }, gpu, false, gpus);
  if (!probe) {
    return undefined;
  }
  const extras = mainParkedBytes(probe, withAll);
  const splitable = splitableGpuBytes(probe);
  const vram = vramTotals(gpus);
  const usableMain = Math.max(0, (vram[mainGpu] || 0) - Math.max(0, headroom) - extras);
  if (splitable <= usableMain && fitsHeadroom(caps, { ...withAll, splitMode: "none", tensorSplit: "" }, gpu, false, headroom, gpus)) {
    return { ...withAll, splitMode: "none", tensorSplit: "" };
  }

  const seed = capacityAwareTensorSplit(vram, mainGpu, extras, splitable, headroom);
  const seedShare = seed ? mainShareFromSplit(seed, mainGpu, n, vram) : 0.5;
  const neighborhood: string[] = [];
  if (seed) {
    neighborhood.push(seed);
    for (const delta of [
      0.02, -0.02, 0.03, -0.03, 0.04, -0.04, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15, 0.2, -0.2, 0.25,
      -0.25, 0.3, -0.3,
    ]) {
      const next = retargetTensorSplitMainShare(seed, mainGpu, seedShare + delta);
      if (next) {
        neighborhood.push(next);
      }
    }
  }
  for (const share of [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]) {
    neighborhood.push(tensorSplitForMainShare(share, mainGpu, n));
  }

  for (const tensorSplit of uniqueSplits(neighborhood)) {
    if (parseTensorSplit(tensorSplit).length < 2) {
      continue;
    }
    const candidate = { ...withAll, tensorSplit, splitMode: "layer" as const };
    if (fitsHeadroom(caps, candidate, gpu, false, headroom, gpus)) {
      const shares = parseTensorSplit(tensorSplit);
      const others = shares.reduce((sum, v, i) => sum + (i === mainGpu ? 0 : v), 0);
      if (others <= 0) {
        return { ...withAll, splitMode: "none", tensorSplit: "" };
      }
      return candidate;
    }
  }
  return undefined;
}

/**
 * Pick context / GPU offload / CPU MoE / speculative defaults from model + VRAM.
 *
 * - Dual/N GPU: full -ngl, fill Main first (capacity-aware), remainder on the other cards; only then RAM / --n-cpu-moe
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
  const gpu = gpus[0] || options.gpu;
  const nLayers = Math.max(1, caps.blockCount || 1);
  const base: LlamaLoadSettings = {
    ...current,
    contextLength,
    nCpuMoe: 0,
    nCpuFfn: 0,
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
    const splitFit = fitMultiGpuFullOffload(base, caps, gpu, gpus, headroom);
    if (splitFit) {
      return splitFit;
    }
    // Cards still too small for the whole model — keep a capacity-aware
    // split and spill the remainder to RAM (layers or MoE experts).
    const mainGpu = Math.min(Math.max(0, base.mainGpu || 0), gpus.length - 1);
    const probe = estimateFor(
      caps,
      { ...base, gpuOffload: 99, nCpuMoe: 0, splitMode: "layer", mainGpu },
      gpu,
      false,
      gpus
    );
    const extras = probe ? mainParkedBytes(probe, base) : 0;
    const splitable = probe ? splitableGpuBytes(probe) : 0;
    const seeded = capacityAwareTensorSplit(vramTotals(gpus), mainGpu, extras, splitable, headroom);
    base = {
      ...base,
      tensorSplit: seeded || tensorSplitForMainShare(
        gpus.reduce((a, g) => a + g.totalBytes, 0) > 0
          ? (gpus[mainGpu]?.totalBytes ?? 0) / gpus.reduce((a, g) => a + g.totalBytes, 0)
          : 0.5,
        mainGpu,
        gpus.length
      ),
      splitMode: "layer",
    };
  }

  const splitWeightTarget =
    gpus.length >= 2 && parseTensorSplit(base.tensorSplit).length >= 2
      ? mainShareFromSplit(base.tensorSplit, base.mainGpu, gpus.length, vramTotals(gpus))
      : undefined;

  if (caps.isMoe) {
    // Max GPU offload; raise CPU MoE only as needed for headroom.
    const withAllGpu: LlamaLoadSettings = { ...base, gpuOffload: 99, nCpuMoe: 0 };
    let best: LlamaLoadSettings = { ...withAllGpu, nCpuMoe: nLayers };
    if (splitWeightTarget != null) {
      best = retargetTensorSplitToWeightShare(best, caps, gpus, splitWeightTarget);
    }
    for (let n = 0; n <= nLayers; n++) {
      let candidate: LlamaLoadSettings = { ...withAllGpu, nCpuMoe: n };
      if (splitWeightTarget != null) {
        candidate = retargetTensorSplitToWeightShare(candidate, caps, gpus, splitWeightTarget);
      }
      if (fitsHeadroom(caps, candidate, gpu, false, headroom, gpus)) {
        best = candidate;
        break;
      }
    }
    return best;
  }

  // Dense: prefer keeping every layer on the GPU and moving dense FFN tensors
  // to system RAM (--n-cpu-ffn) before dropping whole layers — mirrors the MoE
  // path, where experts spill to RAM first.
  const withAllDense: LlamaLoadSettings = { ...base, gpuOffload: 99, nCpuMoe: 0, nCpuFfn: 0 };
  for (let n = 0; n <= nLayers; n++) {
    let candidate: LlamaLoadSettings = { ...withAllDense, nCpuFfn: n };
    if (splitWeightTarget != null) {
      candidate = retargetTensorSplitToWeightShare(candidate, caps, gpus, splitWeightTarget);
    }
    if (fitsHeadroom(caps, candidate, gpu, false, headroom, gpus)) {
      return candidate;
    }
  }

  // Dense: as many layers as possible while keeping headroom.
  // Prefer 99 (“all”) when the full model fits.
  for (let n = nLayers; n >= 0; n--) {
    const gpuOffload = n >= nLayers ? 99 : n;
    const candidate = { ...base, gpuOffload, nCpuMoe: 0, nCpuFfn: 0 };
    if (fitsHeadroom(caps, candidate, gpu, false, headroom, gpus)) {
      return candidate;
    }
  }

  return { ...base, gpuOffload: 0, nCpuMoe: 0, nCpuFfn: 0 };
}

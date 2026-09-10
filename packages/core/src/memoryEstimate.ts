import * as os from "os";
import { formatGpuDeviceLabel, GpuMemoryInfo } from "./gpuInfo";
import {
  assignLayerDevices,
  gpuDisplayOrder,
  parseTensorSplit,
  effectiveTensorSplitShares,
  layerAwareWeightShares,
} from "./gpuSplit";
import {
  heuristicDenseFfnShare,
  heuristicMoeExpertShare,
  isLinearRecurrentHybrid,
  ModelCapabilities,
  readModelCapabilities,
} from "./ggufMetadata";
import { mmprojFileSize, isMtpSidecarFile, usesSidecarMtp } from "./modelLibrary";
import { resolveLoadMode } from "./serverArgs";
import {
  KvCacheType,
  lazyModeReadsFromDisk,
  LlamaLoadSettings,
  speculativeUsesDflash,
  speculativeUsesMtp,
} from "./types";

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
  /**
   * KV at configured context length (full), **including** the per-slot
   * recurrent/SSM state on hybrid layers (see {@link recurrentStateBytes}).
   * `estimateKvBytes(caps, ctx, k, v)` alone is the attention-only part.
   */
  kvBytes: number;
  /** KV if only a short prompt is in use (~2k tokens), same state inclusion. */
  kvBytesWarm: number;
  /**
   * Per-slot recurrent / SSM state bytes folded into {@link kvBytes}
   * (0 for dense and pure-attention models).
   */
  recurrentStateBytes?: number;
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
  /** Main-GPU compute / graph / driver (not tensor-split). */
  gpuOverheadBytes: number;
  /** Per extra GPU that has a layer share (Vulkan/CUDA device heap + sched). */
  peerGpuOverheadBytes: number;
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

/** Keep this much VRAM free on every GPU (compositor / driver / allocator slop). */
export const VRAM_HEADROOM_BYTES = 2 * GiB;
/** Soft “getting full” warning once remaining VRAM drops below this. */
export const VRAM_SOFT_HEADROOM_BYTES = 4 * GiB;

/** True when `used` leaves less than {@link VRAM_HEADROOM_BYTES} free on `cap`. */
export function deviceWouldSpill(used: number, cap: number): boolean {
  return cap > 0 && used > cap - VRAM_HEADROOM_BYTES;
}

/**
 * llama.cpp sizes non-unified KV as n_ctx × n_seq_max. Unified KV is one buffer
 * of n_ctx. `-np` is n_seq_max.
 */
export function kvSlotMultiplier(settings: Pick<LlamaLoadSettings, "maxConcurrentPredictions" | "unifiedKvCache">): number {
  const slots = Math.max(1, Math.round(settings.maxConcurrentPredictions) || 1);
  if (settings.unifiedKvCache || slots <= 1) {
    return 1;
  }
  return slots;
}

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

/** GTT in use is “heavy” once it is real, larger than VRAM use and material. */
const GTT_HEAVY_MIN_BYTES = 1 * GiB;

/**
 * Detect a card that is holding its share in **system RAM** rather than VRAM.
 *
 * RADV silently satisfies large Vulkan allocations out of GTT (GPU-mapped host
 * memory) when the device-local BAR window is too small — a card with
 * Resizable BAR off shows `mem_info_vis_vram_total` well under its real VRAM.
 * The visible symptom is a card with almost no VRAM used, a large GTT figure,
 * and no speed-up, while the estimate (which only knows VRAM) still claims the
 * model is split across both cards.
 *
 * Returns a user-facing warning, or `undefined` when the card looks healthy.
 * Field data (9070 XT + 9060 XT, b10517 Vulkan): the second card reported
 * ~0.1 GiB VRAM against ~14.2 GiB GTT while doing no work.
 */
export function hostFallbackWarning(
  gpu: GpuMemoryInfo | undefined,
  index: number
): string | undefined {
  const cap = gpu?.totalBytes || 0;
  if (!gpu || cap <= 0) {
    return undefined;
  }
  const vramUsed = gpu.usedBytes ?? 0;
  const gttUsed = gpu.gttUsedBytes ?? 0;
  const visTotal = gpu.visVramTotalBytes ?? 0;
  const smallBar = visTotal > 0 && visTotal < cap / 2;
  const gttHeavy =
    gttUsed >= GTT_HEAVY_MIN_BYTES &&
    gttUsed > Math.max(vramUsed * 2, cap * 0.25);
  if (!gttHeavy && !(smallBar && gttUsed >= GTT_HEAVY_MIN_BYTES)) {
    return undefined;
  }
  const label = gpuLabel(gpu, index);
  const barBit = smallBar
    ? ` Its visible-VRAM window is only ~${formatBytes(visTotal)} of ${formatBytes(cap)} ` +
      `(Resizable BAR / Above 4G off)`
    : "";
  return (
    `${label} is using system RAM, not VRAM: ~${formatBytes(gttUsed)} in GTT` +
    (vramUsed > 0 ? ` vs ~${formatBytes(vramUsed)} VRAM` : " with ~0 VRAM") +
    `.${barBit}` +
    ` Layers placed on this card read weights over PCIe, so they run at host-memory speed and the ` +
    `rest of the model only gets one card's bandwidth. Either enable Resizable BAR, drop to one GPU ` +
    `(--split-mode none), or size the split so this card is not asked to hold layers.`
  );
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

/** Backend used to size compute/graph slop (Vulkan heaps are much fatter than CUDA). */
export type ComputeBackend = "vulkan" | "cuda" | "metal" | "cpu" | "unknown";

export interface ComputeOverheadOptions {
  contextLength?: number;
  backend?: ComputeBackend;
  flashAttention?: string;
  /** Share of layers that carry context-scaled attention state (0..1), used
   * only when `discountRecurrentGraph` is set. Linear-recurrent hybrids
   * (Qwen3.5 RCO: fixed SSM state, KV billed only on full-attn layers) run
   * smaller graphs; plain full/SWA interleaves (e.g. Flash-Next) do not.
   * Defaults to 1 (dense) when omitted. */
  fullAttentionFraction?: number;
  /** Opt-in: scale the graph workspace by `fullAttentionFraction`. Only the
   * caller that proved the model is linear-recurrent sets this. */
  discountRecurrentGraph?: boolean;
  /**
   * Vulkan devices that actually hold layers. 1 = `--split-mode none` or a
   * single card (smaller RADV heap). 2+ = a real tensor split (Flash-Next
   * dual-16 GB calibration). Omitted defaults to 1.
   */
  vulkanDeviceCount?: number;
  /**
   * Skip the 2.5 GiB Flash-Next dual-GPU RADV reserve. Defaults to true when
   * `discountRecurrentGraph` is set (Qwen3.5 RCO measured ~2.5 GiB over LACT
   * per card at 60k on two 16 GB cards). Flash-Next keeps the fat heap.
   */
  compactVulkanHeap?: boolean;
}

const BACKEND_OVERHEAD: Record<
  ComputeBackend,
  { driverBytes: number; peerBytes: number; graphElemBytes: number; deviceReservedBytes: number }
> = {
  // Dual-GPU `deviceReservedBytes` is the Flash-Next 64k / two 16 GB RADV
  // residual (weights+KV+graph still ~2.5 GiB under sysfs/LACT per card).
  // Single-device loads use {@link vulkanDeviceReservedBytes} and
  // {@link vulkanSingleDeviceDriverBytes} instead — 768 MiB driver + 1 GiB
  // reserve double-counted RADV heap (Ornith-1.5-9B: 13.6 GiB est vs 12.4 LACT).
  vulkan: {
    driverBytes: 768 * MiB,
    peerBytes: 512 * MiB,
    graphElemBytes: 12,
    deviceReservedBytes: 2.5 * GiB,
  },
  cuda: { driverBytes: 384 * MiB, peerBytes: 256 * MiB, graphElemBytes: 8, deviceReservedBytes: 0 },
  metal: { driverBytes: 384 * MiB, peerBytes: 256 * MiB, graphElemBytes: 8, deviceReservedBytes: 0 },
  cpu: { driverBytes: 0, peerBytes: 0, graphElemBytes: 0, deviceReservedBytes: 0 },
  unknown: {
    driverBytes: 512 * MiB,
    peerBytes: 384 * MiB,
    graphElemBytes: 10,
    deviceReservedBytes: 0,
  },
};

export function inferComputeBackend(
  cpuOnly?: boolean,
  gpus?: Array<{ llamaDeviceId?: string }>
): ComputeBackend {
  if (cpuOnly) {
    return "cpu";
  }
  const ids = (gpus || []).map((g) => (g.llamaDeviceId || "").toLowerCase());
  if (ids.some((id) => id.startsWith("vulkan"))) {
    return "vulkan";
  }
  if (ids.some((id) => /^(cuda|rocm|hip)/.test(id))) {
    return "cuda";
  }
  if (ids.some((id) => id.startsWith("metal"))) {
    return "metal";
  }
  return "unknown";
}

function safePositive(v: number, fallback: number): number {
  return Number.isFinite(v) && v > 0 ? v : fallback;
}

function clampOverheadEmbed(embeddingLength: number): number {
  return Math.max(2048, safePositive(embeddingLength, 4096));
}

function clampOverheadUbatch(physicalBatchSize: number): number {
  return Math.min(Math.max(32, safePositive(physicalBatchSize, 512)), 8192);
}

function clampOverheadBatch(evalBatchSize: number): number {
  return Math.min(Math.max(32, safePositive(evalBatchSize, 2048)), 8192);
}

function clampOverheadContext(contextLength: number | undefined): number {
  return Math.min(Math.max(512, safePositive(contextLength ?? 4096, 4096)), 262144);
}

/** RADV heap slop that is not already billed as graph / activations / driver. */
const VULKAN_SINGLE_DEVICE_RESERVED_BYTES = 256 * MiB;
const VULKAN_MULTI_DEVICE_RESERVED_BYTES = 2.5 * GiB;
/** One-card Vulkan driver tax. Dual-GPU keeps {@link BACKEND_OVERHEAD} 768 MiB. */
const VULKAN_SINGLE_DEVICE_DRIVER_BYTES = 384 * MiB;

/**
 * Per-device Vulkan reserve. One card (or `--split-mode none`): 256 MiB of
 * allocator slop. Two or more cards that hold layers keep the 2.5 GiB
 * Flash-Next dual-GPU calibration unless `compactHeap` is set (Qwen3.5 RCO:
 * that 2.5 GiB/card was the entire 13.9 vs 11.4 GiB LACT gap). CUDA/Metal
 * stay at 0 via {@link BACKEND_OVERHEAD}.
 */
export function vulkanDeviceReservedBytes(
  devicesWithLayers: number,
  compactHeap = false
): number {
  if (!Number.isFinite(devicesWithLayers) || devicesWithLayers <= 0) {
    return 0;
  }
  if (devicesWithLayers <= 1 || compactHeap) {
    return VULKAN_SINGLE_DEVICE_RESERVED_BYTES;
  }
  return VULKAN_MULTI_DEVICE_RESERVED_BYTES;
}

/** Vulkan driver bytes: 384 MiB on one device, 768 MiB when layers are split. */
export function vulkanDriverBytes(devicesWithLayers: number): number {
  if (!Number.isFinite(devicesWithLayers) || devicesWithLayers <= 0) {
    return 0;
  }
  return devicesWithLayers <= 1
    ? VULKAN_SINGLE_DEVICE_DRIVER_BYTES
    : BACKEND_OVERHEAD.vulkan.driverBytes;
}

/**
 * Main-GPU compute / graph / driver. Activations scale with `-ub` and hidden
 * size; the graph workspace also grows with context (Flash Attention keeps
 * that O(n_ctx·n_embd), not O(n_ctx²)). `-b` only adds a small staging buffer.
 */
export function computeOverheadBytes(
  embeddingLength: number,
  physicalBatchSize: number,
  evalBatchSize: number,
  options?: ComputeOverheadOptions
): number {
  const backend = options?.backend || "unknown";
  if (backend === "cpu") {
    return Math.round(256 * MiB);
  }
  const tax = BACKEND_OVERHEAD[backend] || BACKEND_OVERHEAD.unknown;
  const embed = clampOverheadEmbed(embeddingLength);
  const ubatch = clampOverheadUbatch(physicalBatchSize);
  const batch = clampOverheadBatch(evalBatchSize);
  const ctx = clampOverheadContext(options?.contextLength);
  const faOff = (options?.flashAttention || "auto") === "off";
  const graphElem = faOff ? Math.max(tax.graphElemBytes, 36) : tax.graphElemBytes;
  // Several f32 residual / FFN streams (old *24 was ~4× too small).
  const activations = ubatch * embed * 96;
  const batchBuf = batch * 8 * 1024;
  // Linear-recurrent hybrids (Qwen3.5 RCO) keep a fixed SSM state off the
  // full-attn layers, so only the full-attention share bills the graph term.
  // Plain full/SWA interleaves (e.g. Flash-Next qwen4exp) still run dense
  // graphs: GGUF gives them the same interval marker but every layer keeps
  // context-scaled KV, so the discount stays opt-in (clamped to 0.05 so a
  // degenerate value cannot zero the graph entirely).
  const rawFraction = options?.discountRecurrentGraph ? options?.fullAttentionFraction : undefined;
  const fullFraction =
    rawFraction === undefined || !Number.isFinite(rawFraction)
      ? 1
      : Math.min(1, Math.max(0.05, rawFraction));
  const graphRaw = ctx * embed * graphElem * fullFraction;
  // Wide dense models at 64k+ would otherwise invent a 4 GiB graph and
  // push Recommend off full offload on two 16 GB cards. Flash-Next 64k
  // (2560×12) sits under the Vulkan cap; 131k hits it.
  const graphCap = faOff
    ? 6 * GiB
    : backend === "vulkan"
      ? 2.5 * GiB
      : backend === "cuda"
        ? 1.75 * GiB
        : 2 * GiB;
  const graph = Math.min(graphRaw, graphCap);
  const vulkanDevices = options?.vulkanDeviceCount ?? 1;
  const compactHeap =
    options?.compactVulkanHeap !== undefined
      ? options.compactVulkanHeap
      : !!options?.discountRecurrentGraph;
  const reserved =
    backend === "vulkan"
      ? vulkanDeviceReservedBytes(vulkanDevices, compactHeap)
      : tax.deviceReservedBytes;
  const driver = backend === "vulkan" ? vulkanDriverBytes(vulkanDevices) : tax.driverBytes;
  return Math.round(driver + reserved + activations + batchBuf + graph);
}

/**
 * Extra device heap + a smaller sched buffer on every non-main GPU that
 * actually receives layers. llama.cpp builds a ggml backend per device.
 */
export function peerGpuOverheadBytes(
  embeddingLength: number,
  physicalBatchSize: number,
  backend: ComputeBackend = "unknown",
  compactVulkanHeap = false
): number {
  if (backend === "cpu") {
    return 0;
  }
  const tax = BACKEND_OVERHEAD[backend] || BACKEND_OVERHEAD.unknown;
  const embed = clampOverheadEmbed(embeddingLength);
  const ubatch = clampOverheadUbatch(physicalBatchSize);
  const reserved =
    backend === "vulkan"
      ? vulkanDeviceReservedBytes(2, compactVulkanHeap)
      : tax.deviceReservedBytes;
  return Math.round(tax.peerBytes + reserved + ubatch * embed * 32);
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
    case "iq4_nl":
      return 0.5625; // 4-bit super-block, non-linear codebook (18 B / 32 elems)
    case "q4_0":
      return 0.5625; // 18 B / 32 elems (block scale)
    case "q4_1":
      return 0.625; // 20 B / 32 elems (block scale + min)
    case "q5_0":
      return 0.6875; // 22 B / 32 elems
    case "q5_1":
      return 0.75; // 24 B / 32 elems
    case "q8_0":
      return 1;
    case "bf16":
    case "f16":
      return 2;
    case "f32":
      return 4;
    default:
      return 2;
  }
}

type KvCaps = Pick<
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
>;

/**
 * Per-layer KV bytes (0 on recurrent / n_kv=0 layers). Used to put KV on the
 * same GPU that owns the layer under `--split-mode layer`.
 */
export function estimateKvBytesPerLayer(
  caps: KvCaps,
  contextLength: number,
  cacheTypeK: KvCacheType = "q8_0",
  cacheTypeV: KvCacheType = "q8_0"
): number[] {
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
  const recurrentMask = recurrentLayerMask(caps);
  const kBytes = kvCacheTypeElemBytes(cacheTypeK);
  const vBytes = kvCacheTypeElemBytes(cacheTypeV);

  const perLayer: number[] = [];
  for (let i = 0; i < layers; i++) {
    // Hybrid (Qwen3.5 / Ling KDA): recurrent/linear layers keep a fixed SSM
    // state — no context-scaled KV. GGUF often stores n_kv=0 on those layers.
    if (recurrentMask[i]) {
      perLayer.push(0);
      continue;
    }

    const isSwa = !!(swa && pattern && pattern.length && pattern[i % pattern.length]);
    const rawKv = perLayerKv && i < perLayerKv.length ? perLayerKv[i] : defaultKvHeads;
    const nKv = rawKv === undefined || rawKv === null ? defaultKvHeads : rawKv;
    if (nKv <= 0) {
      perLayer.push(0);
      continue;
    }
    // Gemma 4 stores smaller SWA head dims; other SWA models (Muse Glimmer) keep
    // the same key/value width as the global layers.
    const keyDim = isSwa && caps.keyLengthSwa ? Math.max(1, caps.keyLengthSwa) : defaultKeyDim;
    const valDim = isSwa && caps.valueLengthSwa ? Math.max(1, caps.valueLengthSwa) : defaultValDim;
    const tokens = isSwa && swa ? Math.min(contextLength, swa) : contextLength;
    perLayer.push((nKv * keyDim * kBytes + nKv * valDim * vBytes) * tokens);
  }
  return perLayer;
}

/**
 * Per-layer recurrent mask (true = linear / SSM layer with a fixed-size state,
 * no context-scaled KV). Precedence matches `estimateKvBytesPerLayer`: an
 * explicit `recurrent_layers` array, then per-layer KV heads (`n_kv <= 0` marks
 * a recurrent layer, e.g. Ling / bailingmoe3 KDA), then `full_attention_interval`.
 */
export function recurrentLayerMask(
  caps: Pick<
    ModelCapabilities,
    "blockCount" | "fullAttentionInterval" | "recurrentLayers" | "attentionHeadCountKvPerLayer"
  >
): boolean[] {
  const layers = Math.max(1, caps.blockCount || 1);
  const recurrent = caps.recurrentLayers;
  if (recurrent && recurrent.length === layers) {
    return recurrent.map((v) => !!v);
  }
  const perLayerKv = caps.attentionHeadCountKvPerLayer;
  if (perLayerKv && perLayerKv.length === layers) {
    return perLayerKv.map((n) => n <= 0);
  }
  const interval =
    caps.fullAttentionInterval && caps.fullAttentionInterval > 1
      ? caps.fullAttentionInterval
      : undefined;
  return Array.from({ length: layers }, (_, i) => !!(interval && (i + 1) % interval !== 0));
}

type SsmCaps = Pick<
  ModelCapabilities,
  | "blockCount"
  | "embeddingLength"
  | "fullAttentionInterval"
  | "recurrentLayers"
  | "attentionHeadCountKvPerLayer"
  | "ssmStateSize"
  | "ssmInnerSize"
  | "ssmConvKernel"
  | "ssmGroupCount"
>;

/** Guard rails for `ssm.*` values read out of a potentially odd GGUF header. */
const MAX_SSM_STATE_SIZE = 4096;
const MAX_SSM_INNER_SIZE = 1_000_000;

/**
 * Per-layer recurrent (SSM / linear-attention) state bytes **per sequence
 * slot**. Recurrent layers keep no context-scaled KV, but llama.cpp still
 * allocates a fixed f32 state buffer for each of them:
 *
 * - conv state: `(d_conv − 1) × (d_inner + 2·n_group·d_state)`
 * - ssm state:  `d_inner × d_state`
 *
 * State is per sequence, so callers multiply by {@link kvSlotMultiplier}. All
 * zeros when the GGUF exposes no `ssm.*` geometry or the model has no recurrent
 * layers, so dense estimates are unchanged. Deliberately approximate: hybrid
 * layouts differ (Mamba2 vs. gated delta net) but both are the same order.
 */
export function estimateRecurrentStateBytesPerLayer(caps: SsmCaps): number[] {
  const mask = recurrentLayerMask(caps);
  const zeros = () => mask.map(() => 0);
  if (!mask.some(Boolean)) {
    return zeros();
  }
  const rawState = safePositive(caps.ssmStateSize || 0, 0);
  if (rawState <= 0 || rawState > MAX_SSM_STATE_SIZE) {
    return zeros();
  }
  const dState = Math.round(rawState);
  const rawInner = safePositive(caps.ssmInnerSize || 0, safePositive(caps.embeddingLength || 0, 0));
  if (rawInner <= 0 || rawInner > MAX_SSM_INNER_SIZE) {
    return zeros();
  }
  const dInner = Math.round(rawInner);
  const dConv = Math.min(16, Math.max(2, Math.round(safePositive(caps.ssmConvKernel || 0, 4))));
  const nGroup = Math.min(
    Math.round(dInner),
    Math.max(1, Math.round(safePositive(caps.ssmGroupCount || 0, 1)))
  );
  const convChannels = dInner + 2 * nGroup * dState;
  // f32 state buffers (llama.cpp keeps recurrent state in f32).
  const perLayerBytes = (dInner * dState + (dConv - 1) * convChannels) * 4;
  return mask.map((isRecurrent) => (isRecurrent ? perLayerBytes : 0));
}

/**
 * Rough KV-cache size for the given K/V cache dtypes (default `q8_0`, matching
 * {@link DEFAULT_LOAD_SETTINGS}). Handles GQA, per-layer KV heads, and
 * sliding-window attention (e.g. Gemma 4). Recurrent/SSM layers contribute 0
 * here — their fixed state is billed by
 * {@link estimateRecurrentStateBytesPerLayer}.
 */
export function estimateKvBytes(
  caps: KvCaps,
  contextLength: number,
  cacheTypeK: KvCacheType = "q8_0",
  cacheTypeV: KvCacheType = "q8_0"
): number {
  return estimateKvBytesPerLayer(caps, contextLength, cacheTypeK, cacheTypeV).reduce(
    (a, b) => a + b,
    0
  );
}

/** Number of layers that contribute context-scaled KV (excludes hybrid recurrent layers). */
export function countFullAttentionLayers(
  caps: Pick<
    ModelCapabilities,
    "blockCount" | "fullAttentionInterval" | "recurrentLayers" | "attentionHeadCountKvPerLayer"
  >
): number {
  return recurrentLayerMask(caps).filter((isRecurrent) => !isRecurrent).length;
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

/**
 * Fraction of per-layer weight bytes that are dense FFN tensors — what
 * `--n-cpu-ffn` moves to CPU. 0 for MoE models (their FFN is routed experts,
 * handled by `--n-cpu-moe`).
 */
export function resolveDenseFfnShare(caps: ModelCapabilities): number {
  if (caps.isMoe) {
    return 0;
  }
  if (caps.denseFfnShare !== undefined && Number.isFinite(caps.denseFfnShare)) {
    return Math.min(0.95, Math.max(0.05, caps.denseFfnShare));
  }
  return heuristicDenseFfnShare(caps.ffnLength, caps.embeddingLength);
}

/**
 * PLE / engram lookup table share. llama.cpp treats `per_layer_token_embd` as
 * a host-side gather, so these bytes stay in system RAM even at `-ngl 99`.
 */
export function resolvePleShare(caps: ModelCapabilities): number {
  if (caps.pleShare !== undefined && Number.isFinite(caps.pleShare)) {
    return Math.min(0.95, Math.max(0, caps.pleShare));
  }
  return 0;
}

/** Load DFlash or sidecar-MTP draft GGUF caps when a draft path is set. */
export function resolveDraftCapabilities(
  settings: LlamaLoadSettings,
  draftCaps?: ModelCapabilities
): ModelCapabilities | undefined {
  const path = (settings.draftModelPath || "").trim();
  const wantDraft =
    speculativeUsesDflash(settings.speculativeMode) ||
    (speculativeUsesMtp(settings.speculativeMode) && isMtpSidecarFile({ path }));
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
 * KV geometry to use for a draft / speculative GGUF.
 *
 * A sidecar `mtp-*.gguf` carries the **parent model's** metadata — the same
 * `block_count`, `full_attention_interval` and recurrent mask — while holding
 * only the next-n head weights. Sizing its cache from `block_count` billed
 * ~16 layers of full-context KV that llama.cpp never allocates: an MTP head
 * proposes at most `--spec-draft-n-max` tokens, so only the next-n layers keep
 * cache. There is no draft-specific tensor split, but that is unrelated to
 * cache geometry.
 *
 * Measured on the live machine (Qwen3.8-27B + `mtp-Qwen3.8-27B-Q4_0.gguf`,
 * 83200 ctx, f16/f16, `--tensor-split 44,56`): the main card read 18.84 GiB
 * estimated against **12.00 GiB** in sysfs, and 5.08 GiB of the difference was
 * this single term. DFlash drafts are complete models, so they keep their own
 * geometry. The parent's hybrid markers are dropped for the next-n layers so
 * they are billed as plain attention — a length-mismatched mask would zero
 * them out instead.
 */
function draftKvCaps(
  draftCaps: ModelCapabilities,
  settings: LlamaLoadSettings
): ModelCapabilities {
  if (!usesSidecarMtp(settings)) {
    return draftCaps;
  }
  const layers = Math.max(1, Math.floor(draftCaps.nextnPredictLayers || 1));
  if (layers >= Math.max(1, draftCaps.blockCount || 1)) {
    return draftCaps;
  }
  return {
    ...draftCaps,
    blockCount: layers,
    fullAttentionInterval: undefined,
    recurrentLayers: undefined,
    attentionHeadCountKvPerLayer: undefined,
    slidingWindow: undefined,
    slidingWindowPattern: undefined,
  };
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
  const kvK = speculativeUsesDflash(settings.speculativeMode) ? "f16" : settings.cacheTypeK;
  const kvV = speculativeUsesDflash(settings.speculativeMode) ? "f16" : settings.cacheTypeV;
  const kvCaps = draftKvCaps(draftCaps, settings);
  const kvBytes = estimateKvBytes(kvCaps, contextLength, kvK, kvV) * kvSlotMultiplier(settings);
  const kvBytesWarm = estimateKvBytes(kvCaps, warmCtx, kvK, kvV) * kvSlotMultiplier(settings);
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
 * MTP (`draft-mtp`) loads next-n heads from the same GGUF. Those tensors are
 * already in `fileSizeBytes`; llama.cpp `shares_model` does not add extra
 * model weights. Only the extra MTP KV is counted.
 */
function estimateMtpFootprint(
  caps: ModelCapabilities,
  settings: LlamaLoadSettings,
  contextLength: number,
  warmCtx: number,
  cpuOnly: boolean,
  mainKvOnGpu: boolean
): MtpFootprint | undefined {
  if (!speculativeUsesMtp(settings.speculativeMode)) {
    return undefined;
  }
  const layers = Math.max(0, Math.floor(caps.nextnPredictLayers || 0));
  if (layers <= 0) {
    return undefined;
  }
  const nLayers = Math.max(1, caps.blockCount || 1);
  const fileSize = caps.fileSizeBytes || 0;
  // Next-n tensors already live in the main GGUF (llama.cpp shares_model).
  // Counting them again made recommend drop layers / raise CPU-MoE for a cost
  // that is not real. Keep the implied size for UI copy; do not add it to VRAM.
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
  ) * kvSlotMultiplier(settings);
  const kvBytesWarm = estimateKvBytes(
    mtpCaps,
    warmCtx,
    settings.cacheTypeK,
    settings.cacheTypeV
  ) * kvSlotMultiplier(settings);
  const kvOnGpu = !cpuOnly && mainKvOnGpu;
  return {
    layers,
    weightsBytes,
    kvBytes,
    kvBytesWarm,
    gpuWeightsBytes: 0,
    cpuWeightsBytes: 0,
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
  const denseFfnShare = resolveDenseFfnShare(caps);
  const pleShare = resolvePleShare(caps);
  const pleBytes = fileSize * pleShare;
  const loadMode = resolveLoadMode(settings, { caps });
  const pleFromDisk =
    pleBytes > 0 &&
    loadMode === "mmap" &&
    lazyModeReadsFromDisk(settings.lazyMode || "auto", pleBytes);

  // MoE experts are most of the file; --n-cpu-moe keeps those of the first N layers on CPU.
  // PLE n-gram tables are host lookups (llama.cpp LAYER_INPUT) — never on GPU.
  let gpuWeights = fileSize * frac;
  if (!cpuOnly && onGpu > 0 && pleBytes > 0) {
    gpuWeights = Math.max(0, gpuWeights - pleBytes);
  }
  if (!cpuOnly && caps.isMoe && settings.nCpuMoe > 0 && onGpu > 0 && moeExpertShare > 0) {
    const moeCpuLayers = Math.min(settings.nCpuMoe, onGpu);
    gpuWeights = Math.max(0, gpuWeights - fileSize * (moeCpuLayers / nLayers) * moeExpertShare);
  }
  // Dense FFN is the bulk of a dense layer; --n-cpu-ffn keeps those of the first N layers on CPU.
  if (!cpuOnly && !caps.isMoe && settings.nCpuFfn > 0 && onGpu > 0 && denseFfnShare > 0) {
    const ffnCpuLayers = Math.min(settings.nCpuFfn, onGpu);
    gpuWeights = Math.max(0, gpuWeights - fileSize * (ffnCpuLayers / nLayers) * denseFfnShare);
  }
  let cpuWeights = Math.max(0, fileSize - gpuWeights - (pleFromDisk ? pleBytes : 0));
  const mmprojBytes = mmprojFileSize(settings.mmprojPath);
  const mmprojMissing = !!(settings.mmprojPath || "").trim() && mmprojBytes <= 0;
  // CLIP is a separate model: GPU-offloaded by default, not tensor-split with the LLM.
  // --no-mmproj-offload keeps the projector in system RAM.
  const gpuVisionBytes =
    !cpuOnly && onGpu > 0 && mmprojBytes > 0 && settings.mmprojOffloadToGpu !== false
      ? mmprojBytes
      : 0;
  const cpuVisionBytes = gpuVisionBytes > 0 ? 0 : Math.max(0, mmprojBytes);

  const kvBytesRaw = estimateKvBytes(
    caps,
    settings.contextLength,
    settings.cacheTypeK,
    settings.cacheTypeV
  );
  const warmCtx = Math.min(WARM_KV_CONTEXT, Math.max(512, settings.contextLength));
  const kvBytesWarmRaw = estimateKvBytes(caps, warmCtx, settings.cacheTypeK, settings.cacheTypeV);
  const slotMul = kvSlotMultiplier(settings);
  // Recurrent (SSM / linear-attention) layers keep no context-scaled KV, but
  // llama.cpp still allocates a fixed f32 state buffer per layer and per
  // sequence slot. It lives in the same cache buffer as KV, so it is billed
  // here and follows the same GPU/RAM placement and the full/warm split.
  const recurrentStatePerLayer = estimateRecurrentStateBytesPerLayer(caps);
  const recurrentStateBytes = recurrentStatePerLayer.reduce((a, b) => a + b, 0) * slotMul;
  const kvBytes = kvBytesRaw * slotMul + recurrentStateBytes;
  const kvBytesWarm = kvBytesWarmRaw * slotMul + recurrentStateBytes;
  const fullAttnLayers = countFullAttentionLayers(caps);
  const kvOnGpu = !cpuOnly && settings.offloadKvCacheToGpu && onGpu > 0;
  const gpusForBackend: GpuMemoryInfo[] =
    !cpuOnly && options?.gpus?.length
      ? [...options.gpus]
      : !cpuOnly && gpu?.totalBytes
        ? [gpu]
        : [];
  const computeBackend = inferComputeBackend(cpuOnly, gpusForBackend);
  const fullAttentionFraction = fullAttnLayers / nLayers;
  const rcoHeap = isLinearRecurrentHybrid(caps);
  const splitAcrossGpus =
    onGpu > 0 && gpusForBackend.length >= 2 && settings.splitMode !== "none";
  const vulkanDeviceCount = onGpu <= 0 ? 0 : splitAcrossGpus ? gpusForBackend.length : 1;
  const overheadBytes = computeOverheadBytes(
    caps.embeddingLength || 0,
    settings.physicalBatchSize,
    settings.evalBatchSize,
    {
      contextLength: settings.contextLength,
      backend: computeBackend,
      flashAttention: settings.flashAttention,
      fullAttentionFraction,
      discountRecurrentGraph: rcoHeap,
      compactVulkanHeap: rcoHeap,
      vulkanDeviceCount,
    }
  );
  const gpuOverheadBytes = onGpu > 0 ? overheadBytes : 0;
  const peerOverheadEach =
    splitAcrossGpus && computeBackend !== "cpu"
      ? peerGpuOverheadBytes(
          caps.embeddingLength || 0,
          settings.physicalBatchSize,
          computeBackend,
          rcoHeap
        )
      : 0;
  const cpuOverheadBytes = onGpu > 0 ? Math.round(overheadBytes * 0.15) : Math.round(overheadBytes * 0.5);
  const gpus = gpusForBackend;
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
  const layerMassOpts = {
    isMoe: !!caps.isMoe,
    nCpuMoe: settings.nCpuMoe,
    moeExpertShare,
    nCpuFfn: settings.nCpuFfn,
    denseFfnShare,
  };
  const weightShares =
    gpus.length >= 2
      ? layerAwareWeightShares(
          nLayers,
          onGpu,
          shares,
          settings.splitMode,
          mainGpuIndex,
          layerMassOpts
        )
      : shares;
  const layerAssign =
    gpus.length >= 2 && settings.splitMode !== "row" && settings.splitMode !== "tensor"
      ? assignLayerDevices(nLayers, onGpu, shares, settings.splitMode, mainGpuIndex)
      : undefined;
  const kvPerLayerFull = layerAssign
    ? estimateKvBytesPerLayer(
        caps,
        settings.contextLength,
        settings.cacheTypeK,
        settings.cacheTypeV
      ).map((bytes, i) => bytes + (recurrentStatePerLayer[i] || 0))
    : undefined;
  const peerCount = gpus.filter((_, i) => i !== mainGpuIndex && (shares[i] || 0) > 0).length;
  const totalPeerBytes = peerOverheadEach * peerCount;
  const gpuKvBytes = kvOnGpu ? kvBytes : 0;
  const kvOnDevice = (dev: number): number => {
    if (!kvOnGpu) {
      return 0;
    }
    if (!layerAssign || !kvPerLayerFull) {
      return gpuKvBytes * (shares[dev] || 0);
    }
    let sum = 0;
    for (let i = 0; i < layerAssign.length; i++) {
      if (layerAssign[i] === dev) {
        sum += kvPerLayerFull[i] || 0;
      }
    }
    return sum * slotMul;
  };
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
      ? `MTP KV (${mtp.layers} next-n)`
      : "Speculative";

  const totalGpuBytes =
    gpuWeights + gpuKvBytes + gpuOverheadBytes + totalPeerBytes + specGpuBundle + gpuVisionBytes;
  const totalCpuBytes = cpuWeights + cpuKvBytes + cpuOverheadBytes + specCpuBundle + cpuVisionBytes;
  const totalGpuBytesWarm =
    gpuWeights + gpuKvWarm + gpuOverheadBytes + totalPeerBytes + specGpuWarmBundle + gpuVisionBytes;
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
    if (
      gpus.length >= 2 &&
      settings.splitMode !== "none" &&
      settings.splitMode !== "row" &&
      settings.splitMode !== "tensor" &&
      weightShares.some((w, i) => (w || 0) > (shares[i] || 0) + 0.08)
    ) {
      warnings.push(
        "Layer split + CPU MoE: llama.cpp assigns the first layers (cheap after --n-cpu-moe) to earlier GPUs, so later cards hold more expert weight than the tensor-split percentages suggest."
      );
    }
  }
  if (!cpuOnly && !caps.isMoe && settings.nCpuFfn > 0) {
    warnings.push(
      `CPU FFN layers = ${settings.nCpuFfn}: ~${Math.round(denseFfnShare * 100)}% of per-layer weights are dense FFN; those layers’ FFN stays in system RAM.`
    );
  }
  if (!cpuOnly && onGpu > 0 && pleBytes > 0) {
    if (pleFromDisk) {
      warnings.push(
        `PLE n-gram table (~${formatBytes(pleBytes)}, ${Math.round(pleShare * 100)}% of file) is read from disk (--lazy-mode); RAM bars omit the resident table.`
      );
    } else {
      const lazyWantsDisk = lazyModeReadsFromDisk(settings.lazyMode || "auto", pleBytes);
      warnings.push(
        `PLE n-gram table (~${formatBytes(pleBytes)}, ${Math.round(pleShare * 100)}% of file) stays in system RAM.` +
          (lazyWantsDisk && loadMode !== "mmap"
            ? ` Lazy mode needs mmap (current load-mode ${loadMode}).`
            : "")
      );
    }
  }
  if (speculativeUsesDflash(settings.speculativeMode) && !draft) {
    warnings.push(
      "DFlash is on but no draft GGUF is selected — memory bars omit the draft; pick a draft model before starting."
    );
  }
  if (speculativeUsesMtp(settings.speculativeMode) && !mtp && !draft) {
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
      `MTP overhead included: next-n heads are already in the GGUF weights; extra ~${formatBytes(mtp.kvBytes)} MTP KV at full context.`
    );
  }
  if (mmprojMissing) {
    warnings.push(
      "Vision projector path is set but the file is missing — llama-server --mmproj will fail until you pick a valid mmproj or Clear."
    );
  }

  // Keep a fixed 2 GiB free on every card (not 8% of a large GPU).
  const perGpuParts = gpus.map((_, i) => {
    const share = shares[i] || 0;
    const isMain = i === mainGpuIndex;
    const weights = gpuWeights * (weightShares[i] || 0);
    const kv = kvOnDevice(i);
    const overhead = share <= 0 ? 0 : isMain ? gpuOverheadBytes : peerOverheadEach;
    // The draft follows `--tensor-split` like the main model: there is no
    // draft-specific split flag (`--spec-draft-device` only narrows the device
    // list), so parking the whole draft on the main GPU overstated that card
    // by the draft's full size — 1.3 GiB of weights plus KV on the measured
    // 9070/9060 pair, enough to fake a spill on a card with 4 GiB free.
    const spec = specGpuBundle * (weightShares[i] || 0);
    // `--mmproj` is a separate model and is genuinely not tensor-split.
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
      if (!(cap > 0)) {
        continue;
      }
      if (deviceWouldSpill(used, cap)) {
        willSpill = true;
        const pct = Math.round((used / cap) * 100);
        const label = gpuLabel(gpus[i]!, i);
        if (used > cap) {
          warnings.unshift(
            `Estimated ${label} at full context ~${formatBytes(used)} is over the full ${formatBytes(cap)} (${pct}%). Expect spill to system RAM (much slower). Lower Context Length, GPU Offload, or use a smaller quant.`
          );
        } else {
          warnings.unshift(
            `Tight on ${label} at full context: ~${formatBytes(used)} of ${formatBytes(cap)} (${pct}%). Only ~${formatBytes(cap - used)} left — target is ${formatBytes(VRAM_HEADROOM_BYTES)} free. Lower Context Length or GPU Offload.`
          );
        }
      } else if (cap - used < VRAM_SOFT_HEADROOM_BYTES) {
        const pct = Math.round((used / cap) * 100);
        const label = gpuLabel(gpus[i]!, i);
        warnings.push(
          `Getting full on ${label} at full context: ~${formatBytes(used)} of ${formatBytes(cap)} VRAM (${pct}%). Leave some free for the display driver.`
        );
      }
    }
    // Runtime observation, not a prediction — deliberately does not set
    // `willSpill` (that drives the Recommend search, which can only change
    // settings, not the card's BAR window).
    for (let i = 0; i < gpus.length; i++) {
      const hint = hostFallbackWarning(gpus[i], i);
      if (hint) {
        warnings.unshift(hint);
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
      if ((g.gttUsedBytes || 0) > 0) {
        lines.push(
          `Live ${gpuLabel(g, i)} GTT now: ~${formatBytes(g.gttUsedBytes!)} of host RAM mapped to the GPU` +
            ` (VRAM in use ~${formatBytes(g.usedBytes || 0)})`
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
        (recurrentStateBytes > 0 ? " (incl. recurrent state)" : "") +
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
        `MTP: next-n heads already in the GGUF weights (~${formatBytes(mtp.weightsBytes)})` +
          ` · MTP KV ~${formatBytes(mtp.kvBytes)} (RAM)`
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
          : "") +
        (pleShare > 0
          ? pleFromDisk
            ? ` · PLE table ~${Math.round(pleShare * 100)}% on disk`
            : ` · PLE table ~${Math.round(pleShare * 100)}% in RAM`
          : "")
    );
    lines.push(
      `KV @ full ${settings.contextLength.toLocaleString()} ctx: ~${formatBytes(kvBytes)}` +
        (recurrentStateBytes > 0 ? " (incl. recurrent state)" : "") +
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
        `MTP: next-n heads already in the GGUF weights (~${formatBytes(mtp.weightsBytes)})` +
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
  lines.push(
    "Bars show estimate at full context, including Vulkan/CUDA compute and per-GPU heaps. Actual use still varies by quant and driver."
  );

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
    recurrentStateBytes: recurrentStateBytes > 0 ? recurrentStateBytes : undefined,
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
    peerGpuOverheadBytes: peerOverheadEach,
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
    ssmStateSize: c.ssmStateSize || 0,
    ssmInnerSize: c.ssmInnerSize || 0,
    ssmConvKernel: c.ssmConvKernel || 0,
    ssmGroupCount: c.ssmGroupCount || 0,
    isMoe: !!c.isMoe,
    moeExpertShare: c.moeExpertShare ?? null,
    expertCount: c.expertCount || 0,
    ffnLength: c.ffnLength || 0,
    denseFfnShare: c.denseFfnShare ?? null,
    pleShare: c.pleShare ?? null,
    architecture: c.architecture || "",
    nextnPredictLayers: c.nextnPredictLayers || 0,
  });
  return {
    ...pack(caps),
    draft: draftCaps?.fileSizeBytes && draftCaps.blockCount ? pack(draftCaps) : null,
    mmprojFileSizeBytes: mmprojFileSizeBytes && mmprojFileSizeBytes > 0 ? mmprojFileSizeBytes : 0,
  };
}

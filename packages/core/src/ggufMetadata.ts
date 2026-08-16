import * as fs from "fs";
import * as path from "path";
import { LlamaLoadSettings } from "./types";
import { isMtpDraftFileName } from "./modelLibrary";

export interface ModelCapabilities {
  path: string;
  name?: string;
  architecture?: string;
  /** Hard max context from GGUF ({arch}.context_length) */
  maxContextLength: number;
  /** Transformer block/layer count ({arch}.block_count) — caps GPU offload / n-cpu-moe */
  blockCount: number;
  /** Embedding / hidden size ({arch}.embedding_length) */
  embeddingLength?: number;
  /** Attention heads ({arch}.attention.head_count) */
  attentionHeadCount?: number;
  /** KV heads for GQA — scalar, or per-layer (e.g. Gemma 4) */
  attentionHeadCountKv?: number;
  /** Per-layer KV heads when GGUF stores an array */
  attentionHeadCountKvPerLayer?: number[];
  /** Key head dim ({arch}.attention.key_length); prefer over embedding/heads */
  keyLength?: number;
  /** Value head dim */
  valueLength?: number;
  /** Sliding-window attention length (tokens), if present */
  slidingWindow?: number;
  /** Per-layer: true = SWA layer. Expanded to `blockCount` (GGUF may store a scalar period or a short repeating array). */
  slidingWindowPattern?: boolean[];
  /** Key/value dims for SWA layers (Gemma 4) */
  keyLengthSwa?: number;
  valueLengthSwa?: number;
  /**
   * Hybrid models (e.g. Qwen3.5): every Nth layer is full attention; others are
   * recurrent/linear (fixed-size state, no growing KV). Matches llama.cpp.
   */
  fullAttentionInterval?: number;
  /** Per-layer: true = recurrent / linear-attention (no context-scaled KV) */
  recurrentLayers?: boolean[];
  /** Total size on disk (bytes), summed over all shards — proxy for weight memory */
  fileSizeBytes?: number;
  /** Number of shards the split GGUF declares (1 for a single file) */
  shardCount?: number;
  /** Shards actually present next to the selected file */
  shardsFound?: number;
  /**
   * Fraction of tensor bytes that are MoE routed-expert weights (`*_exps`).
   * Used with `--n-cpu-moe` to move expert weight off the GPU estimate.
   * Undefined when not MoE or scan failed (caller should use a heuristic).
   */
  moeExpertShare?: number;
  /** Total MoE experts if present */
  expertCount?: number;
  /** Experts used per token if present */
  expertUsedCount?: number;
  /** True when expert_count > 0 */
  isMoe: boolean;
  /** Default RoPE base from GGUF if present */
  ropeFreqBase?: number;
  /** Next-n / MTP layers if present (speculative-related) */
  nextnPredictLayers?: number;
  fileType?: number | string;
}

const GGUF_MAGIC = 0x46554747; // "GGUF" little-endian

/**
 * Sanity ceilings for counts read out of the file header. Without these a
 * corrupt or truncated download can claim billions of entries and freeze the
 * extension host in a synchronous read loop.
 */
const MAX_KV_ENTRIES = 100_000;
const MAX_TENSORS = 1_000_000;
const MAX_ARRAY_ITEMS = 4_000_000;
const MAX_TENSOR_DIMS = 8;

function checkedCount(value: bigint, limit: number, what: string): number {
  if (value < 0n || value > BigInt(limit)) {
    throw new Error(`Invalid GGUF ${what} count: ${value} (limit ${limit}) — file may be corrupt`);
  }
  return Number(value);
}

type GgufValue = number | boolean | string | GgufValue[];

function readExact(fd: number, size: number, position: number): { buf: Buffer; next: number } {
  const buf = Buffer.alloc(size);
  const n = fs.readSync(fd, buf, 0, size, position);
  if (n !== size) {
    throw new Error(`Unexpected EOF reading GGUF (wanted ${size}, got ${n})`);
  }
  return { buf, next: position + size };
}

function readU32(fd: number, pos: number): { value: number; next: number } {
  const { buf, next } = readExact(fd, 4, pos);
  return { value: buf.readUInt32LE(0), next };
}

function readU64(fd: number, pos: number): { value: bigint; next: number } {
  const { buf, next } = readExact(fd, 8, pos);
  return { value: buf.readBigUInt64LE(0), next };
}

function readString(fd: number, pos: number): { value: string; next: number } {
  const len = readU64(fd, pos);
  const n = Number(len.value);
  if (!Number.isFinite(n) || n < 0 || n > 16 * 1024 * 1024) {
    throw new Error(`Invalid GGUF string length: ${len.value}`);
  }
  const { buf, next } = readExact(fd, n, len.next);
  return { value: buf.toString("utf8"), next };
}

function readValue(fd: number, pos: number, type: number): { value: GgufValue; next: number } {
  switch (type) {
    case 0: {
      // UINT8
      const { buf, next } = readExact(fd, 1, pos);
      return { value: buf.readUInt8(0), next };
    }
    case 1: {
      const { buf, next } = readExact(fd, 1, pos);
      return { value: buf.readInt8(0), next };
    }
    case 2: {
      const { buf, next } = readExact(fd, 2, pos);
      return { value: buf.readUInt16LE(0), next };
    }
    case 3: {
      const { buf, next } = readExact(fd, 2, pos);
      return { value: buf.readInt16LE(0), next };
    }
    case 4: {
      const { buf, next } = readExact(fd, 4, pos);
      return { value: buf.readUInt32LE(0), next };
    }
    case 5: {
      const { buf, next } = readExact(fd, 4, pos);
      return { value: buf.readInt32LE(0), next };
    }
    case 6: {
      const { buf, next } = readExact(fd, 4, pos);
      return { value: buf.readFloatLE(0), next };
    }
    case 7: {
      const { buf, next } = readExact(fd, 1, pos);
      return { value: buf.readUInt8(0) !== 0, next };
    }
    case 8:
      return readString(fd, pos);
    case 9: {
      // ARRAY
      const at = readU32(fd, pos);
      const n = readU64(fd, at.next);
      let p = n.next;
      const arr: GgufValue[] = [];
      const count = checkedCount(n.value, MAX_ARRAY_ITEMS, "array length");
      for (let i = 0; i < count; i++) {
        const v = readValue(fd, p, at.value);
        arr.push(v.value);
        p = v.next;
      }
      return { value: arr, next: p };
    }
    case 10: {
      const { buf, next } = readExact(fd, 8, pos);
      return { value: Number(buf.readBigUInt64LE(0)), next };
    }
    case 11: {
      const { buf, next } = readExact(fd, 8, pos);
      return { value: Number(buf.readBigInt64LE(0)), next };
    }
    case 12: {
      const { buf, next } = readExact(fd, 8, pos);
      return { value: buf.readDoubleLE(0), next };
    }
    default:
      throw new Error(`Unsupported GGUF value type: ${type}`);
  }
}

function asNumber(v: GgufValue | undefined): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) {
    return v;
  }
  if (typeof v === "boolean") {
    return v ? 1 : 0;
  }
  if (typeof v === "string" && v.trim() !== "" && !Number.isNaN(Number(v))) {
    return Number(v);
  }
  return undefined;
}

/**
 * llama.cpp `attention.sliding_window_pattern`: a per-layer bool/0-1 array
 * (possibly shorter than n_layer — tile it) or a scalar period N, matching
 * `set_swa_pattern(N)`: SWA when `il % N < N-1` (N=0 all SWA, N=1 all dense).
 * Muse Glimmer official GGUFs use a 4-entry array; Unsloth uses UINT32 4.
 */
export function resolveSlidingWindowPattern(
  raw: unknown,
  blockCount: number
): boolean[] | undefined {
  const n = Math.max(1, blockCount);
  if (raw === undefined || raw === null || raw === "") {
    return undefined;
  }
  if (typeof raw === "number" && Number.isFinite(raw)) {
    const period = Math.max(0, Math.floor(raw));
    return Array.from({ length: n }, (_, il) => period === 0 || (il % period < period - 1));
  }
  if (typeof raw === "boolean") {
    return Array.from({ length: n }, () => raw);
  }
  if (!Array.isArray(raw) || raw.length === 0) {
    return undefined;
  }
  const flags = raw.map((v) => {
    if (typeof v === "boolean") {
      return v;
    }
    if (typeof v === "number") {
      return !!v;
    }
    return false;
  });
  if (flags.length >= n) {
    return flags.slice(0, n);
  }
  return Array.from({ length: n }, (_, i) => flags[i % flags.length]!);
}

/** Read GGUF key/value metadata (header only — does not scan tensors). */
export function readGgufMetadata(filePath: string): Record<string, GgufValue> {
  const fd = fs.openSync(filePath, "r");
  try {
    const { meta } = readGgufHeader(fd);
    return meta;
  } finally {
    fs.closeSync(fd);
  }
}

interface GgufHeaderScan {
  meta: Record<string, GgufValue>;
  /** Byte offset of tensor data section (aligned). */
  dataStart: number;
  tensors: Array<{ name: string; offset: number }>;
}

function readGgufHeader(fd: number): GgufHeaderScan {
  let pos = 0;
  const magic = readU32(fd, pos);
  if (magic.value !== GGUF_MAGIC) {
    throw new Error("Not a GGUF file");
  }
  pos = magic.next;
  const version = readU32(fd, pos);
  pos = version.next;
  if (version.value < 2 || version.value > 3) {
    // Still attempt parse for forward compatibility within known layout.
  }
  const tensorCount = readU64(fd, pos);
  pos = tensorCount.next;
  const kvCount = readU64(fd, pos);
  pos = kvCount.next;

  const meta: Record<string, GgufValue> = {};
  const nKv = checkedCount(kvCount.value, MAX_KV_ENTRIES, "metadata");
  for (let i = 0; i < nKv; i++) {
    const key = readString(fd, pos);
    pos = key.next;
    const type = readU32(fd, pos);
    pos = type.next;
    const val = readValue(fd, pos, type.value);
    pos = val.next;
    meta[key.value] = val.value;
  }

  const nTensor = checkedCount(tensorCount.value, MAX_TENSORS, "tensor");
  const tensors: Array<{ name: string; offset: number }> = [];
  for (let i = 0; i < nTensor; i++) {
    const name = readString(fd, pos);
    pos = name.next;
    const nDims = readU32(fd, pos);
    pos = nDims.next;
    if (nDims.value > MAX_TENSOR_DIMS) {
      throw new Error(`Invalid GGUF tensor dimensions: ${nDims.value} — file may be corrupt`);
    }
    for (let d = 0; d < nDims.value; d++) {
      const dim = readU64(fd, pos);
      pos = dim.next;
    }
    const type = readU32(fd, pos);
    pos = type.next;
    const offset = readU64(fd, pos);
    pos = offset.next;
    tensors.push({ name: name.value, offset: Number(offset.value) });
  }

  const alignmentRaw = meta["general.alignment"];
  const alignment =
    typeof alignmentRaw === "number" && alignmentRaw > 0 ? Math.floor(alignmentRaw) : 32;
  const dataStart = Math.floor((pos + alignment - 1) / alignment) * alignment;
  return { meta, dataStart, tensors };
}

/**
 * Share of GGUF tensor bytes belonging to routed MoE experts (`*_exps` in the name).
 * Shared-expert tensors (`*shexp*`) are excluded — `--n-cpu-moe` mainly moves `*_exps`.
 */
function computeMoeExpertShare(
  tensors: Array<{ name: string; offset: number }>,
  dataStart: number,
  fileSize: number
): number | undefined {
  if (!tensors.length || fileSize <= dataStart) {
    return undefined;
  }
  const sorted = [...tensors].sort((a, b) => a.offset - b.offset);
  let total = 0;
  let expert = 0;
  for (let i = 0; i < sorted.length; i++) {
    const off = sorted[i].offset;
    const next = i + 1 < sorted.length ? sorted[i + 1].offset : Math.max(off, fileSize - dataStart);
    const size = Math.max(0, next - off);
    total += size;
    if (/_exps(?:\.|$)/i.test(sorted[i].name)) {
      expert += size;
    }
  }
  if (total <= 0 || expert <= 0) {
    return undefined;
  }
  return Math.min(0.98, Math.max(0.05, expert / total));
}

/**
 * Share of GGUF tensor bytes belonging to routed MoE experts (`*_exps` in the name).
 * Shared-expert tensors (`*shexp*`) are excluded — `--n-cpu-moe` mainly moves `*_exps`.
 */
export function measureMoeExpertShare(filePath: string): number | undefined {
  const fd = fs.openSync(filePath, "r");
  try {
    const { dataStart, tensors } = readGgufHeader(fd);
    let fileSize = 0;
    try {
      fileSize = fs.fstatSync(fd).size;
    } catch {
      return undefined;
    }
    return computeMoeExpertShare(tensors, dataStart, fileSize);
  } catch {
    return undefined;
  } finally {
    fs.closeSync(fd);
  }
}

/** Heuristic when tensor scan is unavailable. */
export function heuristicMoeExpertShare(expertCount?: number): number {
  if (!expertCount || expertCount <= 0) {
    return 0.75;
  }
  if (expertCount >= 128) {
    return 0.9;
  }
  if (expertCount >= 64) {
    return 0.85;
  }
  if (expertCount >= 16) {
    return 0.8;
  }
  return 0.75;
}

export function readModelCapabilities(filePath: string): ModelCapabilities {
  const fd = fs.openSync(filePath, "r");
  let meta: Record<string, GgufValue>;
  let dataStart = 0;
  let tensors: Array<{ name: string; offset: number }> = [];
  try {
    const header = readGgufHeader(fd);
    meta = header.meta;
    dataStart = header.dataStart;
    tensors = header.tensors;
  } finally {
    fs.closeSync(fd);
  }
  const arch = typeof meta["general.architecture"] === "string" ? meta["general.architecture"] : undefined;
  const name = typeof meta["general.name"] === "string" ? meta["general.name"] : undefined;

  const pickArchRaw = (...suffixes: string[]): GgufValue | undefined => {
    for (const suffix of suffixes) {
      if (arch) {
        const key = `${arch}.${suffix}`;
        if (key in meta) {
          return meta[key];
        }
      }
      for (const [k, raw] of Object.entries(meta)) {
        if (k.endsWith(`.${suffix}`) || k === suffix) {
          return raw;
        }
      }
    }
    return undefined;
  };

  const pickArchNumber = (...suffixes: string[]): number | undefined => {
    return asNumber(pickArchRaw(...suffixes));
  };

  const pickArchNumberArray = (...suffixes: string[]): number[] | undefined => {
    const raw = pickArchRaw(...suffixes);
    if (!Array.isArray(raw) || raw.length === 0) {
      return undefined;
    }
    const nums = raw.map((v) => asNumber(v)).filter((v): v is number => v !== undefined);
    return nums.length === raw.length ? nums : undefined;
  };

  const pickArchBoolArray = (...suffixes: string[]): boolean[] | undefined => {
    const raw = pickArchRaw(...suffixes);
    if (!Array.isArray(raw) || raw.length === 0) {
      return undefined;
    }
    if (raw.every((v) => typeof v === "boolean")) {
      return raw as boolean[];
    }
    // Some writers store 0/1
    if (raw.every((v) => typeof v === "number" || typeof v === "boolean")) {
      return raw.map((v) => !!v);
    }
    return undefined;
  };

  const maxContextLength = Math.max(512, pickArchNumber("context_length") || 8192);
  const blockCount = Math.max(1, pickArchNumber("block_count") || 32);
  const embeddingLength = pickArchNumber("embedding_length");
  const attentionHeadCount = pickArchNumber("attention.head_count", "head_count");
  const attentionHeadCountKvPerLayer = pickArchNumberArray("attention.head_count_kv", "head_count_kv");
  const attentionHeadCountKv =
    attentionHeadCountKvPerLayer?.[0] !== undefined
      ? // Prefer a representative scalar when constant; else first entry for legacy fields.
        attentionHeadCountKvPerLayer.every((n) => n === attentionHeadCountKvPerLayer[0])
        ? attentionHeadCountKvPerLayer[0]
        : undefined
      : pickArchNumber("attention.head_count_kv", "head_count_kv");
  const keyLength = pickArchNumber("attention.key_length", "key_length");
  const valueLength = pickArchNumber("attention.value_length", "value_length");
  const keyLengthSwa = pickArchNumber("attention.key_length_swa");
  const valueLengthSwa = pickArchNumber("attention.value_length_swa");
  const slidingWindow = pickArchNumber("attention.sliding_window", "sliding_window");
  let slidingWindowPattern = resolveSlidingWindowPattern(
    pickArchRaw("attention.sliding_window_pattern"),
    blockCount
  );
  // llama.cpp muse-glimmer defaults the period to 4 when the key is missing.
  if (
    !slidingWindowPattern &&
    slidingWindow &&
    slidingWindow > 0 &&
    (arch || "").toLowerCase() === "muse-glimmer"
  ) {
    slidingWindowPattern = resolveSlidingWindowPattern(4, blockCount);
  }
  const fullAttentionInterval = pickArchNumber("full_attention_interval");
  let recurrentLayers = pickArchBoolArray(
    "attention.recurrent_layers",
    "attention.is_recurrent",
    "recurrent_layers"
  );
  // Qwen3.5 / similar hybrids: derive recurrent mask from full_attention_interval
  // when GGUF doesn't store an explicit array (llama.cpp does the same).
  if ((!recurrentLayers || recurrentLayers.length !== blockCount) && fullAttentionInterval && fullAttentionInterval > 1) {
    recurrentLayers = Array.from({ length: blockCount }, (_, i) => (i + 1) % fullAttentionInterval !== 0);
  }
  const expertCount = pickArchNumber("expert_count");
  const expertUsedCount = pickArchNumber("expert_used_count");
  const ropeFreqBase = pickArchNumber("rope.freq_base");
  const nextnPredictLayers = pickArchNumber("nextn_predict_layers");
  const fileType = meta["general.file_type"];

  const { bytes: fileSizeBytes, shardCount, shardsFound } = totalModelBytes(filePath);

  const isMoe = (expertCount || 0) > 0;
  let moeExpertShare: number | undefined;
  if (isMoe) {
    moeExpertShare =
      computeMoeExpertShare(tensors, dataStart, fileSizeBytes || 0) ??
      heuristicMoeExpertShare(expertCount);
  }

  return {
    path: filePath,
    name,
    architecture: arch,
    maxContextLength,
    blockCount,
    embeddingLength,
    attentionHeadCount,
    attentionHeadCountKv,
    attentionHeadCountKvPerLayer,
    keyLength,
    valueLength,
    keyLengthSwa,
    valueLengthSwa,
    slidingWindow,
    slidingWindowPattern,
    fullAttentionInterval,
    recurrentLayers,
    fileSizeBytes,
    shardCount,
    shardsFound,
    moeExpertShare,
    expertCount,
    expertUsedCount,
    isMoe,
    ropeFreqBase,
    nextnPredictLayers,
    fileType: typeof fileType === "number" || typeof fileType === "string" ? fileType : undefined,
  };
}

/** `Qwen3-30B-Q4_K_M-00001-of-00003.gguf` → every shard name in the set. */
export function shardFileNames(fileName: string): string[] | undefined {
  const m = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/i.exec(fileName);
  if (!m) {
    return undefined;
  }
  const [, stem, , totalRaw] = m;
  const total = Number(totalRaw);
  if (!Number.isFinite(total) || total < 1 || total > 999) {
    return undefined;
  }
  return Array.from(
    { length: total },
    (_, i) => `${stem}-${String(i + 1).padStart(5, "0")}-of-${totalRaw}.gguf`
  );
}

/**
 * Total on-disk size of a model, summing every shard of a split GGUF.
 * Sizing only the selected shard made a 3-part model look a third of its real
 * weight, which fed straight into the VRAM estimate and the offload advice.
 */
export function totalModelBytes(filePath: string): {
  bytes: number | undefined;
  shardCount: number;
  shardsFound: number;
} {
  const names = shardFileNames(path.basename(filePath));
  if (!names) {
    try {
      return { bytes: fs.statSync(filePath).size, shardCount: 1, shardsFound: 1 };
    } catch {
      return { bytes: undefined, shardCount: 1, shardsFound: 0 };
    }
  }
  const dir = path.dirname(filePath);
  let bytes = 0;
  let shardsFound = 0;
  for (const name of names) {
    try {
      bytes += fs.statSync(path.join(dir, name)).size;
      shardsFound++;
    } catch {
      // Missing shard — reported via shardsFound so callers can warn.
    }
  }
  return { bytes: bytes || undefined, shardCount: names.length, shardsFound };
}

function finite(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) ? (value as number) : fallback;
}

/** True when a GGUF is a DFlash draft model (`general.architecture = dflash`). */
export function isDflashDraftArchitecture(architecture?: string): boolean {
  return (architecture || "").toLowerCase() === "dflash";
}

/**
 * True for Gemma 4 (and similar) sidecar MTP drafters
 * (`general.architecture = gemma4-assistant`).
 */
export function isMtpDraftArchitecture(architecture?: string): boolean {
  const a = (architecture || "").toLowerCase().replace(/_/g, "-");
  return a.endsWith("-assistant");
}

/** Clamp load settings to what the GGUF model actually supports. */
export function clampLoadSettingsToModel(
  settings: LlamaLoadSettings,
  caps: ModelCapabilities
): LlamaLoadSettings {
  const contextLength = Math.min(
    Math.max(512, finite(settings.contextLength, 512)),
    Math.max(512, finite(caps.maxContextLength, 512))
  );
  // -ngl 99/999 means "all"; keep that, otherwise clamp to block count.
  const gpuOffload =
    settings.gpuOffload >= 99 ? settings.gpuOffload : Math.min(Math.max(0, settings.gpuOffload), caps.blockCount);
  const nCpuMoe = caps.isMoe
    ? Math.min(Math.max(0, settings.nCpuMoe), caps.blockCount)
    : 0;

  let speculativeMode = settings.speculativeMode;
  const bakedMtp = !!(caps.nextnPredictLayers && caps.nextnPredictLayers > 0);
  const sidecarMtp = isMtpDraftFileName(settings.draftModelPath);
  // MTP without next-n layers and without a sidecar drafter crashes llama-server
  // ("model doesn't contain MTP layers").
  if (speculativeMode === "mtp" && !bakedMtp && !sidecarMtp) {
    speculativeMode = "off";
  }
  // DFlash without a draft GGUF path can't start speculative — leave mode but
  // serverArgs will omit flags until draftModelPath is set.

  return {
    ...settings,
    contextLength,
    gpuOffload,
    nCpuMoe,
    cpuThreads: Math.min(Math.max(1, settings.cpuThreads), 256),
    maxConcurrentPredictions: Math.min(Math.max(1, settings.maxConcurrentPredictions), 64),
    speculativeMode,
    draftModelPath: typeof settings.draftModelPath === "string" ? settings.draftModelPath.trim() : "",
    draftGpuOffload: Math.min(Math.max(0, settings.draftGpuOffload ?? 99), 999),
    mmprojPath: typeof settings.mmprojPath === "string" ? settings.mmprojPath.trim() : "",
  };
}

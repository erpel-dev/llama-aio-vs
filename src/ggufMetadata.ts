import * as fs from "fs";
import { LlamaLoadSettings } from "./types";

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
  /** Per-layer: true = SWA layer */
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
  /** GGUF file size on disk (bytes) — proxy for weight memory */
  fileSizeBytes?: number;
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
      const count = Number(n.value);
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

/** Read GGUF key/value metadata (header only — does not scan tensors). */
export function readGgufMetadata(filePath: string): Record<string, GgufValue> {
  const fd = fs.openSync(filePath, "r");
  try {
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
    const n = Number(kvCount.value);
    for (let i = 0; i < n; i++) {
      const key = readString(fd, pos);
      pos = key.next;
      const type = readU32(fd, pos);
      pos = type.next;
      const val = readValue(fd, pos, type.value);
      pos = val.next;
      meta[key.value] = val.value;
    }
    return meta;
  } finally {
    fs.closeSync(fd);
  }
}

export function readModelCapabilities(filePath: string): ModelCapabilities {
  const meta = readGgufMetadata(filePath);
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
  const slidingWindowPattern = pickArchBoolArray("attention.sliding_window_pattern");
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

  let fileSizeBytes: number | undefined;
  try {
    fileSizeBytes = fs.statSync(filePath).size;
  } catch {
    // ignore
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
    expertCount,
    expertUsedCount,
    isMoe: (expertCount || 0) > 0,
    ropeFreqBase,
    nextnPredictLayers,
    fileType: typeof fileType === "number" || typeof fileType === "string" ? fileType : undefined,
  };
}

/** Clamp load settings to what the GGUF model actually supports. */
export function clampLoadSettingsToModel(
  settings: LlamaLoadSettings,
  caps: ModelCapabilities
): LlamaLoadSettings {
  const contextLength = Math.min(
    Math.max(512, settings.contextLength),
    caps.maxContextLength
  );
  // -ngl 99/999 means "all"; keep that, otherwise clamp to block count.
  const gpuOffload =
    settings.gpuOffload >= 99 ? settings.gpuOffload : Math.min(Math.max(0, settings.gpuOffload), caps.blockCount);
  const nCpuMoe = caps.isMoe
    ? Math.min(Math.max(0, settings.nCpuMoe), caps.blockCount)
    : 0;

  let speculativeMode = settings.speculativeMode;
  if (speculativeMode === "mtp" && !(caps.nextnPredictLayers && caps.nextnPredictLayers > 0)) {
    // Keep selection but it's best-effort; leave as-is so user can still try if build supports draft.
  }

  return {
    ...settings,
    contextLength,
    gpuOffload,
    nCpuMoe,
    cpuThreads: Math.min(Math.max(1, settings.cpuThreads), 256),
    maxConcurrentPredictions: Math.min(Math.max(1, settings.maxConcurrentPredictions), 64),
    speculativeMode,
  };
}

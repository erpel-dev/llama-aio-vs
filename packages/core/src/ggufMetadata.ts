import * as fs from "fs";
import * as path from "path";
import { isMtpBakedInFile, isMtpSidecarFile } from "./modelLibrary";
import { LlamaLoadSettings, speculativeUsesMtp, speculativeUsesNgram } from "./types";

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
   * Hybrid models (e.g. Qwen3.5 / qwen4exp): every Nth layer is full attention;
   * others are recurrent/linear (fixed-size state, no growing KV). Matches llama.cpp.
   */
  fullAttentionInterval?: number;
  /** Per-layer: true = recurrent / linear-attention (no context-scaled KV) */
  recurrentLayers?: boolean[];
  /**
   * Recurrent (SSM / linear-attention) state geometry from `{arch}.ssm.*`.
   * Recurrent layers carry no context-scaled KV, but llama.cpp still allocates
   * a fixed f32 state buffer per layer **per sequence slot**. Undefined when the
   * GGUF has no `ssm.*` keys (dense or non-SSM hybrid).
   */
  ssmStateSize?: number;
  /** SSM inner (expanded) width — `ssm.inner_size`; falls back to `embedding_length` */
  ssmInnerSize?: number;
  /** SSM depthwise conv kernel width — `ssm.conv_kernel` (llama.cpp default 4) */
  ssmConvKernel?: number;
  /** SSM groups — `ssm.group_count` (llama.cpp default 1) */
  ssmGroupCount?: number;
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
  /** Dense FFN intermediate size ({arch}.feed_forward_length); undefined for MoE models */
  ffnLength?: number;
  /**
   * Fraction of tensor bytes that are dense FFN weights (`ffn_(gate|up|down).weight`).
   * Used with `--n-cpu-ffn` to move dense FFN weight off the GPU estimate.
   * Undefined when the scan failed (caller should use a heuristic).
   */
  denseFfnShare?: number;
  /**
   * Fraction of tensor bytes that are per-layer embedding / PLE n-gram tables
   * (`per_layer_token_embd`). llama.cpp keeps these as host lookups (not GPU
   * compute); Qwen3.8-Flash-Next's table is tens of GiB.
   */
  pleShare?: number;
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

/** Window size for header reads. One syscall covers many small fields. */
const GGUF_READ_CHUNK = 1024 * 1024;

/**
 * Sequential GGUF reader. Small fields are served from a 1 MiB window so a
 * model with hundreds of thousands of tensors is not one `readSync` per field.
 * The returned buffer view is only valid until the next read that refills.
 */
class GgufReader {
  private window: Buffer = Buffer.alloc(0);
  private windowPos = 0;

  constructor(private readonly fd: number) {}

  private fill(position: number, size: number): void {
    if (size <= 0) {
      return;
    }
    if (position >= this.windowPos && position + size <= this.windowPos + this.window.length) {
      return;
    }
    const want = Math.max(size, GGUF_READ_CHUNK);
    const buf = Buffer.alloc(want);
    let got = 0;
    while (got < size) {
      const n = fs.readSync(this.fd, buf, got, want - got, position + got);
      if (n <= 0) {
        break;
      }
      got += n;
    }
    if (got < size) {
      throw new Error(`Unexpected EOF reading GGUF (wanted ${size}, got ${got})`);
    }
    this.window = got === buf.length ? buf : buf.subarray(0, got);
    this.windowPos = position;
  }

  readU8(pos: number): { value: number; next: number } {
    this.fill(pos, 1);
    return { value: this.window.readUInt8(pos - this.windowPos), next: pos + 1 };
  }

  readU32(pos: number): { value: number; next: number } {
    this.fill(pos, 4);
    return { value: this.window.readUInt32LE(pos - this.windowPos), next: pos + 4 };
  }

  readU64(pos: number): { value: bigint; next: number } {
    this.fill(pos, 8);
    return { value: this.window.readBigUInt64LE(pos - this.windowPos), next: pos + 8 };
  }

  readI8(pos: number): { value: number; next: number } {
    this.fill(pos, 1);
    return { value: this.window.readInt8(pos - this.windowPos), next: pos + 1 };
  }

  readU16(pos: number): { value: number; next: number } {
    this.fill(pos, 2);
    return { value: this.window.readUInt16LE(pos - this.windowPos), next: pos + 2 };
  }

  readI16(pos: number): { value: number; next: number } {
    this.fill(pos, 2);
    return { value: this.window.readInt16LE(pos - this.windowPos), next: pos + 2 };
  }

  readI32(pos: number): { value: number; next: number } {
    this.fill(pos, 4);
    return { value: this.window.readInt32LE(pos - this.windowPos), next: pos + 4 };
  }

  readF32(pos: number): { value: number; next: number } {
    this.fill(pos, 4);
    return { value: this.window.readFloatLE(pos - this.windowPos), next: pos + 4 };
  }

  readI64(pos: number): { value: bigint; next: number } {
    this.fill(pos, 8);
    return { value: this.window.readBigInt64LE(pos - this.windowPos), next: pos + 8 };
  }

  readF64(pos: number): { value: number; next: number } {
    this.fill(pos, 8);
    return { value: this.window.readDoubleLE(pos - this.windowPos), next: pos + 8 };
  }

  readBytes(pos: number, size: number): { buf: Buffer; next: number } {
    if (size === 0) {
      return { buf: Buffer.alloc(0), next: pos };
    }
    this.fill(pos, size);
    const start = pos - this.windowPos;
    return { buf: this.window.subarray(start, start + size), next: pos + size };
  }
}

function readString(r: GgufReader, pos: number): { value: string; next: number } {
  const len = r.readU64(pos);
  const n = Number(len.value);
  if (!Number.isFinite(n) || n < 0 || n > 16 * 1024 * 1024) {
    throw new Error(`Invalid GGUF string length: ${len.value}`);
  }
  const { buf, next } = r.readBytes(len.next, n);
  return { value: buf.toString("utf8"), next };
}

function readValue(r: GgufReader, pos: number, type: number): { value: GgufValue; next: number } {
  switch (type) {
    case 0: {
      const v = r.readU8(pos);
      return { value: v.value, next: v.next };
    }
    case 1: {
      const v = r.readI8(pos);
      return { value: v.value, next: v.next };
    }
    case 2: {
      const v = r.readU16(pos);
      return { value: v.value, next: v.next };
    }
    case 3: {
      const v = r.readI16(pos);
      return { value: v.value, next: v.next };
    }
    case 4: {
      const v = r.readU32(pos);
      return { value: v.value, next: v.next };
    }
    case 5: {
      const v = r.readI32(pos);
      return { value: v.value, next: v.next };
    }
    case 6: {
      const v = r.readF32(pos);
      return { value: v.value, next: v.next };
    }
    case 7: {
      const v = r.readU8(pos);
      return { value: v.value !== 0, next: v.next };
    }
    case 8:
      return readString(r, pos);
    case 9: {
      const at = r.readU32(pos);
      const n = r.readU64(at.next);
      let p = n.next;
      const arr: GgufValue[] = [];
      const count = checkedCount(n.value, MAX_ARRAY_ITEMS, "array length");
      for (let i = 0; i < count; i++) {
        const v = readValue(r, p, at.value);
        arr.push(v.value);
        p = v.next;
      }
      return { value: arr, next: p };
    }
    case 10: {
      const v = r.readU64(pos);
      return { value: Number(v.value), next: v.next };
    }
    case 11: {
      const v = r.readI64(pos);
      return { value: Number(v.value), next: v.next };
    }
    case 12: {
      const v = r.readF64(pos);
      return { value: v.value, next: v.next };
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

/**
 * Architectures whose GGUFs usually omit `attention.sliding_window_pattern`
 * because llama.cpp hard-codes the layout in `llama_model::load_hparams`
 * (`set_swa_pattern(N)`: SWA when `il % N < N-1`). Without this table every
 * layer was billed as a full-context layer, overestimating KV 2–5× for
 * Gemma 2/3, gpt-oss and Llama 4.
 *
 * `window` is llama.cpp's fallback `n_swa` when the GGUF has no
 * `attention.sliding_window` key (Llama 4 never writes one).
 */
const ARCH_SWA_DEFAULTS: Record<string, { period: number; window?: number }> = {
  "muse-glimmer": { period: 4 },
  gemma2: { period: 2, window: 4096 },
  gemma3: { period: 6, window: 1024 },
  gemma3n: { period: 5, window: 512 },
  "gemma-embedding": { period: 6, window: 1024 },
  "gpt-oss": { period: 2, window: 128 },
  llama4: { period: 4, window: 8192 },
  cohere2: { period: 4, window: 4096 },
  exaone4: { period: 4 },
};

/** llama.cpp's built-in SWA layout for `arch`, when it has one. */
export function defaultSwaLayout(
  arch: string | undefined
): { period: number; window?: number } | undefined {
  return ARCH_SWA_DEFAULTS[(arch || "").toLowerCase()];
}

/** True when capabilities read before this table existed should be refreshed. */
export function capsMissDefaultSwaPattern(caps: {
  architecture?: string;
  slidingWindowPattern?: boolean[];
  blockCount?: number;
}): boolean {
  const layout = defaultSwaLayout(caps.architecture);
  if (!layout) {
    return false;
  }
  return (
    !caps.slidingWindowPattern ||
    caps.slidingWindowPattern.length < (caps.blockCount || 0)
  );
}

/** Read GGUF key/value metadata (header only — does not scan tensors). */
export function readGgufMetadata(filePath: string): Record<string, GgufValue> {
  return readGgufHeaderFromFile(filePath).meta;
}

interface GgufHeaderScan {
  meta: Record<string, GgufValue>;
  /** Byte offset of tensor data section (aligned). */
  dataStart: number;
  tensors: Array<{ name: string; offset: number }>;
}

function readGgufHeader(r: GgufReader): GgufHeaderScan {
  let pos = 0;
  const magic = r.readU32(pos);
  if (magic.value !== GGUF_MAGIC) {
    throw new Error("Not a GGUF file");
  }
  pos = magic.next;
  const version = r.readU32(pos);
  pos = version.next;
  if (version.value < 2 || version.value > 3) {
    // Still attempt parse for forward compatibility within known layout.
  }
  const tensorCount = r.readU64(pos);
  pos = tensorCount.next;
  const kvCount = r.readU64(pos);
  pos = kvCount.next;

  const meta: Record<string, GgufValue> = {};
  const nKv = checkedCount(kvCount.value, MAX_KV_ENTRIES, "metadata");
  for (let i = 0; i < nKv; i++) {
    const key = readString(r, pos);
    pos = key.next;
    const type = r.readU32(pos);
    pos = type.next;
    const val = readValue(r, pos, type.value);
    pos = val.next;
    meta[key.value] = val.value;
  }

  const nTensor = checkedCount(tensorCount.value, MAX_TENSORS, "tensor");
  const tensors: Array<{ name: string; offset: number }> = [];
  for (let i = 0; i < nTensor; i++) {
    const name = readString(r, pos);
    pos = name.next;
    const nDims = r.readU32(pos);
    pos = nDims.next;
    if (nDims.value > MAX_TENSOR_DIMS) {
      throw new Error(`Invalid GGUF tensor dimensions: ${nDims.value} — file may be corrupt`);
    }
    for (let d = 0; d < nDims.value; d++) {
      const dim = r.readU64(pos);
      pos = dim.next;
    }
    const type = r.readU32(pos);
    pos = type.next;
    const offset = r.readU64(pos);
    pos = offset.next;
    tensors.push({ name: name.value, offset: Number(offset.value) });
  }

  const alignmentRaw = meta["general.alignment"];
  const alignment =
    typeof alignmentRaw === "number" && alignmentRaw > 0 ? Math.floor(alignmentRaw) : 32;
  const dataStart = Math.floor((pos + alignment - 1) / alignment) * alignment;
  return { meta, dataStart, tensors };
}

interface HeaderCacheEntry {
  mtimeMs: number;
  size: number;
  scan: GgufHeaderScan;
}

/** Parsed headers keyed by path + mtime + size. Callers must not mutate `scan`. */
const HEADER_CACHE_MAX = 32;
const headerCache = new Map<string, HeaderCacheEntry>();

function rememberHeader(key: string, entry: HeaderCacheEntry): void {
  headerCache.delete(key);
  headerCache.set(key, entry);
  while (headerCache.size > HEADER_CACHE_MAX) {
    const oldest = headerCache.keys().next().value;
    if (oldest === undefined) {
      break;
    }
    headerCache.delete(oldest);
  }
}

/** Header scan memoised by path, mtime, and size. */
function readGgufHeaderFromFile(filePath: string): GgufHeaderScan {
  const key = path.resolve(filePath);
  const st = fs.statSync(filePath);
  const hit = headerCache.get(key);
  if (hit && hit.mtimeMs === st.mtimeMs && hit.size === st.size) {
    return hit.scan;
  }
  const fd = fs.openSync(filePath, "r");
  try {
    const scan = readGgufHeader(new GgufReader(fd));
    rememberHeader(key, { mtimeMs: st.mtimeMs, size: st.size, scan });
    return scan;
  } finally {
    fs.closeSync(fd);
  }
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
  try {
    const { dataStart, tensors } = readGgufHeaderFromFile(filePath);
    const fileSize = fs.statSync(filePath).size;
    return computeMoeExpertShare(tensors, dataStart, fileSize);
  } catch {
    return undefined;
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

/**
 * llama.cpp `general.architecture` for Qwen3.8-Flash-Next (HF `qwen4_exp`).
 * Underscores/hyphens are stripped so `qwen4_exp` and `qwen4-exp` match.
 */
export function isQwen4expArchitecture(architecture?: string): boolean {
  const a = (architecture || "").toLowerCase().replace(/[_-]/g, "");
  return a === "qwen4exp";
}

/**
 * True for linear-recurrent hybrids whose non-full layers carry a fixed SSM
 * state instead of context-scaled attention.
 *
 * qwen35 / qwen35moe (Qwen3.5 RCO) always qualify. qwen4exp is a full/SWA
 * interleave **unless** the GGUF actually stores SSM geometry — Flash-Next
 * GSQ-RCO reuses the `qwen4exp` tag with 36/48 recurrent layers and
 * `ssm.state_size`. Those must take the compact graph/heap, or dual-GPU
 * VRAM bars keep the 2.5 GiB Flash-Next reserve. A qwen4exp file with no
 * `ssm.*` keys stays dense.
 */
export function isLinearRecurrentHybrid(caps?: {
  architecture?: string;
  ssmStateSize?: number;
}): boolean {
  if (!caps) {
    return false;
  }
  const arch = (caps.architecture || "").toLowerCase().replace(/[._-]/g, "");
  if (arch.startsWith("qwen35")) {
    return true;
  }
  const ssm = caps.ssmStateSize;
  return typeof ssm === "number" && Number.isFinite(ssm) && ssm > 0 && ssm <= 4096;
}

/** llama.cpp `per_layer_token_embd` / `ple_ngram` — not the small `*.ple.*` projections. */
function isPleTensorName(name: string): boolean {
  return /per_layer_token_embd/i.test(name) || /ple_ngram/i.test(name);
}

function tensorByteTotals(
  tensors: Array<{ name: string; offset: number }>,
  dataStart: number,
  fileSize: number,
  match: (name: string) => boolean
): { total: number; matched: number } | undefined {
  if (!tensors.length || fileSize <= dataStart) {
    return undefined;
  }
  const sorted = [...tensors].sort((a, b) => a.offset - b.offset);
  let total = 0;
  let matched = 0;
  for (let i = 0; i < sorted.length; i++) {
    const off = sorted[i].offset;
    const next = i + 1 < sorted.length ? sorted[i + 1].offset : Math.max(off, fileSize - dataStart);
    const size = Math.max(0, next - off);
    total += size;
    if (match(sorted[i].name)) {
      matched += size;
    }
  }
  if (total <= 0) {
    return undefined;
  }
  return { total, matched };
}

/** Header-scan one GGUF and return tensor-byte totals, using that file's size. */
function measureTensorClassInFile(
  filePath: string,
  match: (name: string) => boolean
): { total: number; matched: number } | undefined {
  try {
    const { dataStart, tensors } = readGgufHeaderFromFile(filePath);
    const fileSize = fs.statSync(filePath).size;
    return tensorByteTotals(tensors, dataStart, fileSize, match);
  } catch {
    return undefined;
  }
}

function isMoeExpertTensor(name: string): boolean {
  return /_exps(?:\.|$)/i.test(name);
}

function isDenseFfnTensor(name: string): boolean {
  return /ffn_(?:gate|up|down)\.weight$/i.test(name);
}

/**
 * Share of tensor bytes matching `match`, summed across every shard.
 * Each shard is sized from its own file length — passing the split total into
 * one shard's tensor list made the last tensor absorb the other shards (B-37).
 */
function measureClassShareForModel(
  filePath: string,
  match: (name: string) => boolean,
  selected: { tensors: Array<{ name: string; offset: number }>; dataStart: number } | undefined,
  clamp: { min: number; max: number }
): number | undefined {
  const names = shardFileNames(path.basename(filePath));
  const dir = path.dirname(filePath);
  const selectedName = path.basename(filePath);
  const files = names || [selectedName];
  let total = 0;
  let matched = 0;
  for (const name of files) {
    const p = names ? path.join(dir, name) : filePath;
    const reuse = selected && name === selectedName ? selected : undefined;
    let part: { total: number; matched: number } | undefined;
    if (reuse) {
      let fileSize = 0;
      try {
        fileSize = fs.statSync(p).size;
      } catch {
        continue;
      }
      part = tensorByteTotals(reuse.tensors, reuse.dataStart, fileSize, match);
    } else {
      part = measureTensorClassInFile(p, match);
    }
    if (!part) {
      continue;
    }
    total += part.total;
    matched += part.matched;
  }
  if (total <= 0 || matched <= 0) {
    return undefined;
  }
  return Math.min(clamp.max, Math.max(clamp.min, matched / total));
}

/** When the GGUF scan misses PLE tensors on qwen4exp, assume ~40% of the file. */
export function heuristicPleShare(architecture?: string): number {
  return isQwen4expArchitecture(architecture) ? 0.4 : 0;
}

/** Pin the PLE n-gram table to CPU (`--override-tensor per_layer_token_embd.=CPU`). */
export function shouldPinPleToCpu(
  caps?: Pick<ModelCapabilities, "architecture" | "pleShare">
): boolean {
  if (!caps) {
    return false;
  }
  if (typeof caps.pleShare === "number" && Number.isFinite(caps.pleShare) && caps.pleShare > 0) {
    return true;
  }
  return isQwen4expArchitecture(caps.architecture);
}

/**
 * Heuristic dense FFN tensor share when the scan is unavailable: per layer,
 * FFN holds 3 × embed × ffn_len elements vs ~4 × embed² for attention.
 * Llama-style (ffn = 4×embed) → 75%; SwiGLU (~8/3×embed) → ~67%.
 */
export function heuristicDenseFfnShare(
  ffnLength?: number,
  embeddingLength?: number
): number {
  const ffn = Math.max(0, ffnLength || 0);
  const embed = Math.max(0, embeddingLength || 0);
  if (ffn <= 0 || embed <= 0) {
    return 0.7;
  }
  const ffnElems = 3 * ffn;
  const attnElems = 4 * embed;
  return Math.min(0.95, Math.max(0.05, ffnElems / (ffnElems + attnElems)));
}

interface CapsCacheEntry {
  mtimeMs: number;
  size: number;
  at: number;
  caps: ModelCapabilities;
}

/**
 * Parsed headers keyed by path, validated against mtime/size. The TTL bounds
 * staleness for inputs the stat cannot see (shards appearing next to the file).
 */
const CAPS_CACHE_TTL_MS = 30_000;
const CAPS_CACHE_MAX = 32;
const capsCache = new Map<string, CapsCacheEntry>();

/** Forget parsed GGUF headers (tests, or after files were replaced in place). */
export function invalidateModelCapabilitiesCache(filePath?: string): void {
  if (!filePath) {
    capsCache.clear();
    headerCache.clear();
    return;
  }
  const key = path.resolve(filePath);
  capsCache.delete(key);
  headerCache.delete(key);
}

/**
 * Header-only GGUF parse, memoised by path + mtime + size. The header itself
 * is read in 1 MiB chunks (see {@link GgufReader}).
 */
export function readModelCapabilities(filePath: string): ModelCapabilities {
  const key = path.resolve(filePath);
  const now = Date.now();
  let st: fs.Stats | undefined;
  try {
    st = fs.statSync(filePath);
  } catch {
    // fall through — the uncached read reports the real error
  }
  const hit = st ? capsCache.get(key) : undefined;
  if (hit && st && hit.mtimeMs === st.mtimeMs && hit.size === st.size && now - hit.at < CAPS_CACHE_TTL_MS) {
    return structuredClone(hit.caps);
  }
  const caps = readModelCapabilitiesUncached(filePath);
  if (st) {
    capsCache.delete(key);
    capsCache.set(key, { mtimeMs: st.mtimeMs, size: st.size, at: now, caps: structuredClone(caps) });
    while (capsCache.size > CAPS_CACHE_MAX) {
      const oldest = capsCache.keys().next().value;
      if (oldest === undefined) {
        break;
      }
      capsCache.delete(oldest);
    }
  }
  return caps;
}

function readModelCapabilitiesUncached(filePath: string): ModelCapabilities {
  const header = readGgufHeaderFromFile(filePath);
  const meta = header.meta;
  const dataStart = header.dataStart;
  const tensors = header.tensors;
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
  let slidingWindow = pickArchNumber("attention.sliding_window", "sliding_window");
  let slidingWindowPattern = resolveSlidingWindowPattern(
    pickArchRaw("attention.sliding_window_pattern"),
    blockCount
  );
  // llama.cpp hard-codes the SWA layout for several families (Gemma 2/3,
  // gpt-oss, Llama 4, …) instead of reading it from the GGUF.
  const swaDefault = defaultSwaLayout(arch);
  if (swaDefault) {
    if (!(slidingWindow && slidingWindow > 0) && swaDefault.window) {
      slidingWindow = swaDefault.window;
    }
    if (!slidingWindowPattern && slidingWindow && slidingWindow > 0) {
      slidingWindowPattern = resolveSlidingWindowPattern(swaDefault.period, blockCount);
    }
  }
  const fullAttentionInterval =
    pickArchNumber("full_attention_interval") || pickArchNumber("layer_group_size");
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
  // Ling / bailingmoe3: KDA layers are stored as n_kv=0, MLA as n_kv=1. Treating
  // 0 as "one head" (or as the model's 16 Q-heads) inflates KV by ~10–50×.
  if (
    (!recurrentLayers || recurrentLayers.length !== blockCount) &&
    attentionHeadCountKvPerLayer &&
    attentionHeadCountKvPerLayer.length === blockCount &&
    attentionHeadCountKvPerLayer.some((n) => n <= 0) &&
    attentionHeadCountKvPerLayer.some((n) => n > 0)
  ) {
    recurrentLayers = attentionHeadCountKvPerLayer.map((n) => n <= 0);
  }
  const expertCount = pickArchNumber("expert_count");
  const expertUsedCount = pickArchNumber("expert_used_count");
  const ropeFreqBase = pickArchNumber("rope.freq_base");
  const nextnPredictLayers = pickArchNumber("nextn_predict_layers");
  // Recurrent-state geometry (Mamba-style SSM / gated delta net). Absent on
  // dense and pure-attention hybrids, so the state term stays 0 there.
  const ssmStateSize = pickArchNumber("ssm.state_size");
  const ssmInnerSize = pickArchNumber("ssm.inner_size");
  const ssmConvKernel = pickArchNumber("ssm.conv_kernel");
  const ssmGroupCount = pickArchNumber("ssm.group_count");
  const fileType = meta["general.file_type"];

  const { bytes: fileSizeBytes, shardCount, shardsFound } = totalModelBytes(filePath);

  const isMoe = (expertCount || 0) > 0;
  const selectedTensors = { tensors, dataStart };
  let moeExpertShare: number | undefined;
  if (isMoe) {
    moeExpertShare =
      measureClassShareForModel(filePath, isMoeExpertTensor, selectedTensors, { min: 0.05, max: 0.98 }) ??
      heuristicMoeExpertShare(expertCount);
  }

  // Dense FFN pricing for --n-cpu-ffn. Only meaningful for non-MoE models —
  // MoE routers keep their experts in ffn_*_exps tensors, matched separately.
  const ffnLength = pickArchNumber("feed_forward_length");
  const denseFfnShare = !isMoe && ffnLength
    ? measureClassShareForModel(filePath, isDenseFfnTensor, selectedTensors, { min: 0.05, max: 0.95 }) ??
      heuristicDenseFfnShare(ffnLength, embeddingLength)
    : undefined;

  const pleShare =
    measureClassShareForModel(filePath, isPleTensorName, selectedTensors, { min: 0.01, max: 0.95 }) ??
    (isQwen4expArchitecture(arch) ? heuristicPleShare(arch) : undefined);

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
    ssmStateSize,
    ssmInnerSize,
    ssmConvKernel,
    ssmGroupCount,
    fileSizeBytes,
    shardCount,
    shardsFound,
    moeExpertShare,
    ffnLength,
    denseFfnShare,
    pleShare,
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
  // Dense counterpart: --n-cpu-ffn targets dense FFN tensors, which MoE models
  // only have as (tiny) shared experts — reset it there.
  const nCpuFfn = caps.isMoe
    ? 0
    : Math.min(Math.max(0, settings.nCpuFfn ?? 0), caps.blockCount);

  let speculativeMode = settings.speculativeMode;
  let draftModelPath = typeof settings.draftModelPath === "string" ? settings.draftModelPath.trim() : "";
  if (draftModelPath && isMtpBakedInFile({ path: draftModelPath })) {
    draftModelPath = "";
  }
  const bakedMtp = !!(caps.nextnPredictLayers && caps.nextnPredictLayers > 0);
  // qwen4exp Unsloth `mtp-*.gguf` is not a llama.cpp `--model-draft` (missing
  // output_hc_norm.weight). Treat as no sidecar so Start does not pass it.
  if (isQwen4expArchitecture(caps.architecture) && isMtpSidecarFile({ path: draftModelPath })) {
    draftModelPath = "";
  }
  const sidecarMtp = isMtpSidecarFile({ path: draftModelPath });
  // MTP without next-n layers and without a sidecar drafter crashes llama-server
  // ("model doesn't contain MTP layers").
  if (speculativeUsesMtp(speculativeMode) && !bakedMtp && !sidecarMtp) {
    speculativeMode = speculativeUsesNgram(speculativeMode) ? "ngram" : "off";
  }
  // DFlash without a draft GGUF path can't start speculative — leave mode but
  // serverArgs will omit flags until draftModelPath is set.

  return {
    ...settings,
    contextLength,
    gpuOffload,
    nCpuMoe,
    nCpuFfn,
    cpuThreads: Math.min(Math.max(1, settings.cpuThreads), 256),
    maxConcurrentPredictions: Math.min(Math.max(1, settings.maxConcurrentPredictions), 64),
    speculativeMode,
    draftModelPath,
    draftGpuOffload: Math.min(Math.max(0, settings.draftGpuOffload ?? 99), 999),
    mmprojPath: typeof settings.mmprojPath === "string" ? settings.mmprojPath.trim() : "",
  };
}

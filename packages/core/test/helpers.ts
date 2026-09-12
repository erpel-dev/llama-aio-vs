import { ModelCapabilities } from "../src/ggufMetadata";
import { LlamaLoadSettings, DEFAULT_LOAD_SETTINGS } from "../src/types";

export const GiB = 1024 ** 3;

/** A plain dense model: 48 layers, GQA 40/8, 5120 hidden, 18 GiB of weights. */
export function denseCaps(overrides: Partial<ModelCapabilities> = {}): ModelCapabilities {
  return {
    path: "/models/test-dense.gguf",
    name: "test-dense",
    architecture: "qwen3",
    fileSizeBytes: 18 * GiB,
    blockCount: 48,
    embeddingLength: 5120,
    attentionHeadCount: 40,
    attentionHeadCountKv: 8,
    keyLength: 128,
    valueLength: 128,
    maxContextLength: 262144,
    isMoe: false,
    ...overrides,
  } as ModelCapabilities;
}

export function moeCaps(overrides: Partial<ModelCapabilities> = {}): ModelCapabilities {
  return denseCaps({
    name: "test-moe",
    isMoe: true,
    expertCount: 128,
    expertUsedCount: 8,
    moeExpertShare: 0.9,
    ...overrides,
  });
}

export function loadSettings(overrides: Partial<LlamaLoadSettings> = {}): LlamaLoadSettings {
  return { ...DEFAULT_LOAD_SETTINGS, ...overrides };
}

export type MiniGgufValue =
  | { type: "string"; value: string }
  | { type: "u32"; value: number }
  | { type: "u32[]"; value: number[] }
  | { type: "bool[]"; value: boolean[] };

/**
 * Bytes of a header-only GGUF v3 file (no tensors) carrying `kv`. Enough for
 * `readModelCapabilities` to exercise metadata-derived fields.
 */
export function miniGguf(kv: Record<string, MiniGgufValue>): Buffer {
  const parts: Buffer[] = [];
  const u32 = (n: number) => {
    const b = Buffer.alloc(4);
    b.writeUInt32LE(n, 0);
    return b;
  };
  const u64 = (n: number) => {
    const b = Buffer.alloc(8);
    b.writeBigUInt64LE(BigInt(n), 0);
    return b;
  };
  const str = (s: string) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s, "utf8")]);
  parts.push(u32(0x46554747), u32(3), u64(0), u64(Object.keys(kv).length));
  for (const [key, entry] of Object.entries(kv)) {
    parts.push(str(key));
    switch (entry.type) {
      case "string":
        parts.push(u32(8), str(entry.value));
        break;
      case "u32":
        parts.push(u32(4), u32(entry.value));
        break;
      case "u32[]":
        parts.push(u32(9), u32(4), u64(entry.value.length), ...entry.value.map(u32));
        break;
      case "bool[]":
        parts.push(
          u32(9),
          u32(7),
          u64(entry.value.length),
          ...entry.value.map((v) => Buffer.from([v ? 1 : 0]))
        );
        break;
    }
  }
  // Pad so the file has a non-empty "data" section like a real model.
  parts.push(Buffer.alloc(64));
  return Buffer.concat(parts);
}

/** Value that follows `flag` in an argv array, or undefined when absent. */
export function argValue(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

/** Values that follow every occurrence of `flag` in an argv array. */
export function argValues(args: string[], flag: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < args.length - 1; i++) {
    if (args[i] === flag) {
      out.push(args[i + 1]!);
    }
  }
  return out;
}

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

/** Value that follows `flag` in an argv array, or undefined when absent. */
export function argValue(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

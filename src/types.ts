import * as os from "os";
import type { ModelCapabilities } from "./ggufMetadata";

export type { ModelCapabilities };

export interface LlamaLoadSettings {
  /** Context length (--ctx-size) */
  contextLength: number;
  /** GPU layers (-ngl) */
  gpuOffload: number;
  /** CPU threads (-t) */
  cpuThreads: number;
  /** Evaluation batch size (-b) */
  evalBatchSize: number;
  /** Physical batch size (-ub) */
  physicalBatchSize: number;
  /** Max concurrent predictions (-np) */
  maxConcurrentPredictions: number;
  /** MoE: force expert tensors of first N layers onto CPU (--n-cpu-moe) */
  nCpuMoe: number;
  /** Offload KV cache to GPU (default true; false => --no-kv-offload) */
  offloadKvCacheToGpu: boolean;
  /** Keep model in memory (--mlock) */
  keepModelInMemory: boolean;
  /** Use mmap (--mmap / --no-mmap) */
  tryMmap: boolean;
  /** Unified KV cache (experimental; passed when supported) */
  unifiedKvCache: boolean;
  /** Context checkpoints / cache reuse */
  contextCheckpoints: number;
  /** RoPE frequency base; null = auto */
  ropeFreqBase: number | null;
  /** RoPE frequency scale; null = auto */
  ropeFreqScale: number | null;
  /** Seed; null = random */
  seed: number | null;
  /** Speculative decoding mode */
  speculativeMode: "off" | "mtp";
  /** Max draft tokens for speculative decoding */
  maxDraftTokens: number;
  /** Min draft tokens for speculative decoding */
  minDraftTokens: number;
  /** Draft probability */
  draftProbability: number;
}

export interface RequestSettings {
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
}

export interface ExtensionState {
  selectedModelPath: string;
  loadSettings: LlamaLoadSettings;
  requestSettings: RequestSettings;
  /** @deprecated prefer modelCapabilities.maxContextLength */
  modelMaxContext?: number;
  modelCapabilities?: ModelCapabilities;
}

export const DEFAULT_LOAD_SETTINGS: LlamaLoadSettings = {
  // Copilot Chat / agent prompts are large (tools + workspace). Keep one slot
  // so the full contextLength is available (llama.cpp splits ctx across -np).
  contextLength: 65536,
  gpuOffload: 99,
  cpuThreads: Math.max(1, Math.min(8, os.cpus().length || 4)),
  evalBatchSize: 2048,
  physicalBatchSize: 512,
  maxConcurrentPredictions: 1,
  nCpuMoe: 0,
  offloadKvCacheToGpu: true,
  keepModelInMemory: false,
  tryMmap: true,
  unifiedKvCache: true,
  contextCheckpoints: 32,
  ropeFreqBase: null,
  ropeFreqScale: null,
  seed: null,
  speculativeMode: "off",
  maxDraftTokens: 2,
  minDraftTokens: 0,
  draftProbability: 0.75,
};

export const DEFAULT_REQUEST_SETTINGS: RequestSettings = {
  temperature: 0.2,
  topP: 0.95,
  topK: 40,
  maxTokens: 2048,
};

export interface ServerStatus {
  running: boolean;
  pid?: number;
  port: number;
  host: string;
  modelPath?: string;
  endpoint: string;
  ownedByThisExtension: boolean;
  message: string;
  /** True when running and model/load/launch settings differ from the live server. */
  configDirty?: boolean;
  /** True while start/reload is in progress (model may be loading; HTTP not ready yet). */
  starting?: boolean;
  /** Latest boot progress line for the sidebar (e.g. "Loading model into memory… 12s"). */
  startMessage?: string;
}

export interface HfModelHit {
  id: string;
  downloads: number;
  likes: number;
  pipeline_tag?: string;
  tags?: string[];
}

export interface HfFileHit {
  path: string;
  size: number;
  url: string;
}

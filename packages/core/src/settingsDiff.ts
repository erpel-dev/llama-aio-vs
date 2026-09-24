import { LlamaLoadSettings } from "./types";

export interface SettingChange {
  key: keyof LlamaLoadSettings | "model" | "launchMode";
  label: string;
  from: string;
  to: string;
}

const LABELS: Partial<Record<keyof LlamaLoadSettings, string>> = {
  contextLength: "ctx",
  gpuOffload: "GPU layers",
  cpuThreads: "threads",
  evalBatchSize: "batch",
  physicalBatchSize: "ubatch",
  maxConcurrentPredictions: "slots",
  nCpuMoe: "CPU MoE",
  nCpuFfn: "CPU FFN",
  offloadKvCacheToGpu: "KV on GPU",
  cacheTypeK: "KV K",
  cacheTypeV: "KV V",
  keepModelInMemory: "mlock",
  tryMmap: "mmap",
  lazyMode: "lazy mode",
  unifiedKvCache: "unified KV",
  flashAttention: "flash attn",
  contextCheckpoints: "checkpoints",
  cacheReuse: "cache reuse",
  reasoningFormat: "reasoning format",
  reasoningBudget: "reasoning budget",
  ropeFreqBase: "RoPE base",
  ropeFreqScale: "RoPE scale",
  seed: "seed",
  speculativeMode: "speculative",
  maxDraftTokens: "draft max",
  minDraftTokens: "draft min",
  draftProbability: "draft p",
  draftModelPath: "draft model",
  draftGpuOffload: "draft GPU layers",
  ngramVariant: "n-gram variant",
  ngramSizeN: "n-gram size",
  ngramSizeM: "n-gram draft",
  ngramMinHits: "n-gram hits",
  mmprojPath: "vision projector",
  mmprojOffloadToGpu: "vision on GPU",
  tensorSplit: "tensor split",
  splitMode: "split mode",
  mainGpu: "main GPU",
};

function baseName(p: string): string {
  const s = String(p || "").replace(/\\/g, "/");
  return s.slice(s.lastIndexOf("/") + 1) || "none";
}

function fmtValue(key: keyof LlamaLoadSettings, v: unknown): string {
  if (v === undefined || v === null || v === "") {
    return key === "tensorSplit" ? "auto" : "none";
  }
  if (typeof v === "boolean") {
    return v ? "on" : "off";
  }
  if (key === "contextLength" && typeof v === "number") {
    return v >= 1024 && v % 1024 === 0 ? `${v / 1024}k` : v.toLocaleString("en-US");
  }
  if (key === "gpuOffload" && typeof v === "number" && v >= 99) {
    return "all";
  }
  if ((key === "draftModelPath" || key === "mmprojPath") && typeof v === "string") {
    return baseName(v);
  }
  return String(v);
}

/**
 * Load settings that differ between the running server and the sidebar.
 * Keys outside {@link LABELS} (internal bookkeeping) are ignored.
 */
export function diffLoadSettings(
  launched: Partial<LlamaLoadSettings> | undefined,
  current: Partial<LlamaLoadSettings>
): SettingChange[] {
  if (!launched) {
    return [];
  }
  const out: SettingChange[] = [];
  for (const key of Object.keys(LABELS) as Array<keyof LlamaLoadSettings>) {
    const a = launched[key];
    const b = current[key];
    if (a === undefined && b === undefined) {
      continue;
    }
    if (JSON.stringify(a ?? null) === JSON.stringify(b ?? null)) {
      continue;
    }
    out.push({ key, label: LABELS[key]!, from: fmtValue(key, a), to: fmtValue(key, b) });
  }
  return out;
}

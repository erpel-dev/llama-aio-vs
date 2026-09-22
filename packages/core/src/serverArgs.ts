import { parseTensorSplit } from "./gpuSplit";
import type { GpuMemoryInfo } from "./gpuInfo";
import {
  heuristicPleShare,
  isQwen4expArchitecture,
  shouldPinPleToCpu,
  type ModelCapabilities,
} from "./ggufMetadata";
import { usesSidecarMtp } from "./modelLibrary";
import {
  LlamaLoadSettings,
  lazyModeReadsFromDisk,
  normalizeLoadSettings,
  RequestSettings,
  speculativeUsesDflash,
  speculativeUsesMtp,
  speculativeUsesNgram,
} from "./types";

export type LlamaLoadMode = "mmap" | "mlock" | "none";

/**
 * llama.cpp `--load-mode`. Lazy PLE reads need mmap, so `--n-cpu-moe` /
 * `--n-cpu-ffn` no longer force `none` when `--lazy-mode` will actually
 * fault a large table.
 */
export function resolveLoadMode(
  settings: LlamaLoadSettings,
  options?: { caps?: ModelCapabilities; platform?: string }
): LlamaLoadMode {
  const platform = options?.platform ?? process.platform;
  if (settings.keepModelInMemory && platform !== "win32") {
    return "mlock";
  }
  const mmapRequested = settings.tryMmap || settings.keepModelInMemory;
  if (!mmapRequested) {
    return "none";
  }
  const cpuOverride = settings.nCpuMoe > 0 || (settings.nCpuFfn ?? 0) > 0;
  if (cpuOverride && !keepMmapForLazy(settings, options?.caps)) {
    return "none";
  }
  return "mmap";
}

/** Keep mmap so `--lazy-mode` can fault PLE / engram rows from disk. */
export function keepMmapForLazy(
  settings: LlamaLoadSettings,
  caps?: Pick<ModelCapabilities, "architecture" | "pleShare" | "fileSizeBytes">
): boolean {
  const lazy = settings.lazyMode || "auto";
  if (lazy === "off" || !settings.tryMmap) {
    return false;
  }
  if (lazy === "on") {
    return true;
  }
  if (!caps) {
    return false;
  }
  const file = caps.fileSizeBytes || 0;
  const share =
    typeof caps.pleShare === "number" && Number.isFinite(caps.pleShare) && caps.pleShare > 0
      ? caps.pleShare
      : heuristicPleShare(caps.architecture);
  const pleBytes = file * share;
  if (lazyModeReadsFromDisk("auto", pleBytes)) {
    return true;
  }
  // qwen4exp tables are always huge; file size may be unknown before the scan.
  return isQwen4expArchitecture(caps.architecture) && file <= 0;
}

/**
 * CPU builds ignore -ngl / GPU KV offload / --n-cpu-moe / --n-cpu-ffn. Apply the
 * same zeros for launch args and dirty-fingerprint so the sidebar matches what
 * start ships.
 */
export function normalizeLoadSettingsForCpuBackend(
  settings: LlamaLoadSettings
): LlamaLoadSettings {
  return {
    ...settings,
    gpuOffload: 0,
    offloadKvCacheToGpu: false,
    mmprojOffloadToGpu: false,
    nCpuMoe: 0,
    nCpuFfn: 0,
  };
}

/** Shell-ish display form of an argv list (clipboard / log), not a spawn argv. */
export function formatCommandDisplay(argv: readonly string[]): string {
  return argv.map(quoteCommandArg).join(" ");
}

function quoteCommandArg(value: string): string {
  if (value.length === 0) {
    return '""';
  }
  if (!/[\s"]/.test(value)) {
    return value;
  }
  return `"${value.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

/**
 * Build llama-server CLI args from load settings + model path.
 * Settings are normalized here as well as in the store, so no caller can put a
 * NaN/undefined/out-of-range value on the command line.
 *
 * `options.requestSampling` ships the current request defaults as server-side
 * sampling flags, so raw clients that omit sampling fields inherit them instead
 * of llama.cpp's generic built-ins (top_k 40, min_p 0.05, …). Per-request
 * values still win for our own frontends.
 */
export function buildServerArgs(
  modelPath: string,
  host: string,
  port: number,
  rawSettings: LlamaLoadSettings,
  options?: { gpus?: GpuMemoryInfo[]; requestSampling?: RequestSettings; caps?: ModelCapabilities }
): string[] {
  const settings = normalizeLoadSettings(rawSettings);
  const args: string[] = [
    "-m",
    modelPath,
    "--host",
    host,
    "--port",
    String(port),
    "--ctx-size",
    String(settings.contextLength),
    "-ngl",
    String(settings.gpuOffload),
    "-t",
    String(settings.cpuThreads),
    "-b",
    String(settings.evalBatchSize),
    "-ub",
    String(settings.physicalBatchSize),
    "-np",
    String(settings.maxConcurrentPredictions),
  ];

  // --cache-reuse: min chunk size reused via KV shifting (0 = off).
  const cacheReuse = Math.max(0, settings.cacheReuse ?? 0);
  if (cacheReuse > 0) {
    args.push("--cache-reuse", String(cacheReuse));
  }

  // --ctx-checkpoints: only when it differs from the llama.cpp default, so older
  // builds without the flag keep working out of the box.
  const checkpoints = Math.max(0, settings.contextCheckpoints ?? 32);
  if (checkpoints !== 32) {
    args.push("--ctx-checkpoints", String(checkpoints));
  }

  if (settings.nCpuMoe > 0) {
    args.push("--n-cpu-moe", String(settings.nCpuMoe));
  }

  // Dense counterpart to --n-cpu-moe: first N layers' dense FFN tensors stay
  // in system RAM. MoE models reset this to 0 at normalize time. Only sent
  // when non-zero, so older builds without the flag keep booting.
  if ((settings.nCpuFfn ?? 0) > 0) {
    args.push("--n-cpu-ffn", String(settings.nCpuFfn));
  }

  if (!settings.offloadKvCacheToGpu) {
    args.push("--no-kv-offload");
  }

  // KV cache quantization (default q8_0 — ~½ KV size vs f16, good for long agent ctx).
  args.push("--cache-type-k", settings.cacheTypeK || "q8_0");
  args.push("--cache-type-v", settings.cacheTypeV || "q8_0");

  // Unified KV buffer across sequences (-kvu / -no-kvu).
  args.push(settings.unifiedKvCache ? "--kv-unified" : "--no-kv-unified");

  // Only pass -fa when overriding; "auto" is the llama.cpp default and the
  // value form is not accepted by older builds.
  const flashAttention = settings.flashAttention || "auto";
  if (flashAttention !== "auto") {
    args.push("--flash-attn", flashAttention);
  }

  // deepseek-legacy keeps thoughts in message.content (with <think> tags) *and*
  // reasoning_content, so Copilot Chat providers that only read `content` still
  // see the stream.
  args.push("--reasoning-format", settings.reasoningFormat || "deepseek-legacy");

  // Cap thinking tokens so think models can't spend the whole reply on reasoning.
  const reasoningBudget = settings.reasoningBudget ?? -1;
  if (reasoningBudget >= 0) {
    args.push("--reasoning-budget", String(reasoningBudget));
  }

  // Server-side sampling defaults (see doc comment above). Only shipped when a
  // caller provides request sampling, so older call sites keep working.
  const rs = options?.requestSampling;
  if (rs) {
    args.push("--temp", String(rs.temperature));
    args.push("--top-p", String(rs.topP));
    args.push("--top-k", String(rs.topK));
    args.push("--min-p", String(rs.minP));
    args.push("--presence-penalty", String(rs.presencePenalty));
    args.push("--frequency-penalty", String(rs.frequencyPenalty));
    args.push("--repeat-penalty", String(rs.repeatPenalty));
  }

  // Prefer current --load-mode over deprecated --mmap / --mlock / --no-mmap.
  // mlock is poorly supported on Windows — fall back to mmap when pinning is requested.
  // --n-cpu-moe / --n-cpu-ffn normally force none (mmap + overrides is slower),
  // except when --lazy-mode needs mmap to fault a large PLE table.
  args.push("--load-mode", resolveLoadMode(settings, { caps: options?.caps }));

  // Only pass -lzm when overriding; "auto" is the llama.cpp default and older
  // builds without the flag keep working.
  const lazyMode = settings.lazyMode || "auto";
  if (lazyMode !== "auto") {
    args.push("--lazy-mode", lazyMode);
  }

  if (settings.ropeFreqBase != null) {
    args.push("--rope-freq-base", String(settings.ropeFreqBase));
  }

  if (settings.ropeFreqScale != null) {
    args.push("--rope-freq-scale", String(settings.ropeFreqScale));
  }

  if (settings.seed != null) {
    args.push("--seed", String(settings.seed));
  }

  // Vision projector (CLIP / mmproj). llama.cpp defaults GPU-offload on.
  const mmproj = (settings.mmprojPath || "").trim();
  if (mmproj) {
    args.push("--mmproj", mmproj);
    if (!settings.mmprojOffloadToGpu) {
      args.push("--no-mmproj-offload");
    }
  }

  // Multi-GPU: omit defaults so a single-GPU / older llama.cpp still starts.
  // --tensor-split is GPU0,GPU1,… (Vulkan/CUDA device order).
  // --split-mode none keeps the model on one card: drop the split and, when
  // llama.cpp device ids are known, pass --device so the other GPUs stay free.
  if (settings.gpuOffload > 0) {
    const split = parseTensorSplit(settings.tensorSplit);
    if (settings.splitMode === "none") {
      const gpus = (options?.gpus || []).filter((g) => g.totalBytes > 0);
      const main = gpus.length
        ? Math.min(Math.max(0, settings.mainGpu || 0), gpus.length - 1)
        : Math.max(0, settings.mainGpu || 0);
      const deviceId = (gpus[main]?.llamaDeviceId || "").trim();
      args.push("--split-mode", "none");
      if (deviceId) {
        args.push("--device", deviceId);
      } else if (main !== 0) {
        args.push("--main-gpu", String(main));
      }
    } else if (split.length >= 2) {
      args.push("--tensor-split", settings.tensorSplit);
      args.push("--split-mode", settings.splitMode || "layer");
      args.push("--main-gpu", String(settings.mainGpu || 0));
    } else if (settings.splitMode && settings.splitMode !== "layer") {
      args.push("--split-mode", settings.splitMode);
      if (settings.mainGpu !== 0) {
        args.push("--main-gpu", String(settings.mainGpu));
      }
    } else if (settings.mainGpu !== 0) {
      args.push("--main-gpu", String(settings.mainGpu));
    }
  }

  // Speculative decoding (llama.cpp ≥ ~b10xxx uses --spec-*). `--spec-type` is a
  // list: n-gram can stack with MTP or DFlash (cheap lookup first, neural draft
  // fills misses). llama.cpp then runs impls in a hardcoded priority order.
  if (speculativeUsesNgram(settings.speculativeMode)) {
    appendNgramSpecArgs(args, settings);
  }
  if (speculativeUsesMtp(settings.speculativeMode)) {
    args.push("--spec-type", "draft-mtp");
    args.push("--spec-draft-n-max", String(effectiveMaxDraftTokens(settings)));
    if (settings.minDraftTokens > 0) {
      args.push("--spec-draft-n-min", String(settings.minDraftTokens));
    }
    if (settings.draftProbability > 0) {
      args.push("--spec-draft-p-min", String(settings.draftProbability));
    }
    // Gemma 4 (and similar) load next-n heads from a sibling GGUF, not the
    // language file. Qwen-style baked-in MTP omits --model-draft.
    if (usesSidecarMtp(settings) && settings.draftModelPath) {
      args.push("--model-draft", settings.draftModelPath);
      args.push("--spec-draft-ngl", String(settings.draftGpuOffload));
    }
  } else if (speculativeUsesDflash(settings.speculativeMode) && settings.draftModelPath) {
    // DFlash needs a separate draft GGUF trained for the target model.
    args.push("--spec-type", "draft-dflash");
    args.push("--model-draft", settings.draftModelPath);
    args.push("--spec-draft-n-max", String(effectiveMaxDraftTokens(settings)));
    args.push("--spec-draft-ngl", String(settings.draftGpuOffload));
    if (settings.draftProbability > 0) {
      args.push("--spec-draft-p-min", String(settings.draftProbability));
    }
    // Quantized draft KV collapses acceptance (llama.cpp#25725); keep f16.
    args.push("--cache-type-k-draft", "f16");
    args.push("--cache-type-v-draft", "f16");
  }

  // llama.cpp --fit defaults to on, but it cannot change user-set -ngl,
  // --tensor-split, or --n-cpu-moe (CPU tensor overrides). The probe then
  // logs "failed to fit params … n_gpu_layers already set … abort" and
  // mis-accounts multi-GPU + CPU MoE. We always set --ctx-size / -ngl, so
  // skip it (also required for sidecar MTP / DFlash draft probes).
  args.push("--fit", "off");

  // Qwen3.8-Flash-Next (and Gemma 3n-style PLE): keep the n-gram lookup table on
  // CPU. llama.cpp's LAYER_INPUT default usually does this anyway; the override
  // makes it explicit so -ngl 99 doesn't pull tens of GiB onto VRAM.
  if (shouldPinPleToCpu(options?.caps)) {
    args.push("--override-tensor", "per_layer_token_embd.=CPU");
  }

  return args;
}

/** N-gram `--spec-type` + per-variant size flags. Safe to call in stacked modes. */
function appendNgramSpecArgs(args: string[], settings: LlamaLoadSettings): void {
  const variant = settings.ngramVariant || "simple";
  if (variant === "mod") {
    args.push("--spec-type", "ngram-mod");
    args.push("--spec-ngram-mod-n-match", String(settings.ngramSizeN));
    args.push("--spec-ngram-mod-n-max", String(Math.max(settings.ngramSizeM, settings.ngramSizeN)));
  } else {
    args.push("--spec-type", `ngram-${variant}`);
    args.push(`--spec-ngram-${variant}-size-n`, String(settings.ngramSizeN));
    args.push(`--spec-ngram-${variant}-size-m`, String(Math.max(settings.ngramSizeM, settings.ngramSizeN)));
    args.push(`--spec-ngram-${variant}-min-hits`, String(settings.ngramMinHits));
  }
}

/** Draft length actually passed as `--spec-draft-n-max` (DFlash defaults to 15). */
export function effectiveMaxDraftTokens(settings: LlamaLoadSettings): number {
  if (speculativeUsesDflash(settings.speculativeMode)) {
    return Math.max(1, settings.maxDraftTokens || 15);
  }
  if (speculativeUsesMtp(settings.speculativeMode)) {
    return Math.max(1, settings.maxDraftTokens);
  }
  return settings.maxDraftTokens;
}

/**
 * Stable fingerprint of everything that requires a server restart/reload
 * to take effect (model path + load settings + launch mode).
 */
export function serverConfigFingerprint(
  modelPath: string,
  rawSettings: LlamaLoadSettings,
  launchMode: string = ""
): string {
  // Normalize first so the fingerprint describes the args we would actually run.
  const settings = normalizeLoadSettings(rawSettings);
  const spec = settings.speculativeMode;
  return JSON.stringify({
    modelPath: (modelPath || "").trim(),
    launchMode: launchMode || "",
    contextLength: settings.contextLength,
    gpuOffload: settings.gpuOffload,
    cpuThreads: settings.cpuThreads,
    evalBatchSize: settings.evalBatchSize,
    physicalBatchSize: settings.physicalBatchSize,
    maxConcurrentPredictions: settings.maxConcurrentPredictions,
    nCpuMoe: settings.nCpuMoe,
    nCpuFfn: settings.nCpuFfn ?? 0,
    offloadKvCacheToGpu: !!settings.offloadKvCacheToGpu,
    cacheTypeK: settings.cacheTypeK || "q8_0",
    cacheTypeV: settings.cacheTypeV || "q8_0",
    keepModelInMemory: !!settings.keepModelInMemory,
    tryMmap: !!settings.tryMmap,
    lazyMode: settings.lazyMode || "auto",
    unifiedKvCache: !!settings.unifiedKvCache,
    flashAttention: settings.flashAttention || "auto",
    reasoningFormat: settings.reasoningFormat || "deepseek-legacy",
    reasoningBudget: settings.reasoningBudget ?? -1,
    contextCheckpoints: settings.contextCheckpoints,
    cacheReuse: settings.cacheReuse ?? 0,
    ropeFreqBase: settings.ropeFreqBase,
    ropeFreqScale: settings.ropeFreqScale,
    seed: settings.seed,
    speculativeMode: spec,
    maxDraftTokens: spec === "off" ? 0 : effectiveMaxDraftTokens(settings),
    minDraftTokens: speculativeUsesMtp(spec) ? settings.minDraftTokens : 0,
    draftProbability:
      speculativeUsesMtp(spec) || speculativeUsesDflash(spec) ? settings.draftProbability : 0,
    draftModelPath:
      speculativeUsesDflash(spec) || usesSidecarMtp(settings) ? settings.draftModelPath || "" : "",
    draftGpuOffload:
      speculativeUsesDflash(spec) || usesSidecarMtp(settings) ? settings.draftGpuOffload : 0,
    // N-gram knobs are CLI flags → restart-relevant. Omitted entirely when
    // n-gram speculation is off so old fingerprints stay stable.
    ...(speculativeUsesNgram(spec)
      ? {
          ngramVariant: settings.ngramVariant || "simple",
          ngramSizeN: settings.ngramSizeN,
          ngramSizeM: settings.ngramSizeM,
          ngramMinHits: settings.ngramMinHits,
        }
      : {}),
    mmprojPath: settings.mmprojPath || "",
    mmprojOffloadToGpu: !!settings.mmprojOffloadToGpu,
    tensorSplit: settings.tensorSplit || "",
    splitMode: settings.splitMode || "layer",
    mainGpu: settings.mainGpu || 0,
  });
}

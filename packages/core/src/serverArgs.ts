import { LlamaLoadSettings, normalizeLoadSettings } from "./types";

/**
 * CPU builds ignore -ngl / GPU KV offload / --n-cpu-moe. Apply the same zeros
 * for launch args and dirty-fingerprint so the sidebar matches what start ships.
 */
export function normalizeLoadSettingsForCpuBackend(
  settings: LlamaLoadSettings
): LlamaLoadSettings {
  return {
    ...settings,
    gpuOffload: 0,
    offloadKvCacheToGpu: false,
    nCpuMoe: 0,
  };
}

/**
 * Build llama-server CLI args from load settings + model path.
 * Settings are normalized here as well as in the store, so no caller can put a
 * NaN/undefined/out-of-range value on the command line.
 */
export function buildServerArgs(
  modelPath: string,
  host: string,
  port: number,
  rawSettings: LlamaLoadSettings
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

  // Prefer current --load-mode over deprecated --mmap / --mlock / --no-mmap.
  // mlock is poorly supported on Windows — fall back to mmap when pinning is requested.
  if (settings.keepModelInMemory && process.platform !== "win32") {
    args.push("--load-mode", "mlock");
  } else if (!settings.tryMmap && !settings.keepModelInMemory) {
    args.push("--load-mode", "none");
  } else {
    args.push("--load-mode", "mmap");
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

  // Speculative decoding (llama.cpp ≥ ~b10xxx uses --spec-*).
  if (settings.speculativeMode === "mtp") {
    args.push("--spec-type", "draft-mtp");
    args.push("--spec-draft-n-max", String(effectiveMaxDraftTokens(settings)));
    if (settings.minDraftTokens > 0) {
      args.push("--spec-draft-n-min", String(settings.minDraftTokens));
    }
    if (settings.draftProbability > 0) {
      args.push("--spec-draft-p-min", String(settings.draftProbability));
    }
  } else if (settings.speculativeMode === "dflash" && settings.draftModelPath) {
    // DFlash needs a separate draft GGUF trained for the target model.
    args.push("--spec-type", "draft-dflash");
    args.push("--model-draft", settings.draftModelPath);
    args.push("--spec-draft-n-max", String(effectiveMaxDraftTokens(settings)));
    args.push("--spec-draft-ngl", String(settings.draftGpuOffload));
    // Quantized draft KV collapses acceptance (llama.cpp#25725); keep f16.
    args.push("--cache-type-k-draft", "f16");
    args.push("--cache-type-v-draft", "f16");
    // llama.cpp --fit defaults to on; its draft-memory probe fails with
    // "dflash requires ctx_other to be set" and aborts load. We already set
    // --ctx-size / -ngl ourselves, so turn fit off for DFlash.
    args.push("--fit", "off");
  }

  return args;
}

/** Draft length actually passed as `--spec-draft-n-max` (DFlash defaults to 15). */
export function effectiveMaxDraftTokens(settings: LlamaLoadSettings): number {
  if (settings.speculativeMode === "dflash") {
    return Math.max(1, settings.maxDraftTokens || 15);
  }
  if (settings.speculativeMode === "mtp") {
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
    offloadKvCacheToGpu: !!settings.offloadKvCacheToGpu,
    cacheTypeK: settings.cacheTypeK || "q8_0",
    cacheTypeV: settings.cacheTypeV || "q8_0",
    keepModelInMemory: !!settings.keepModelInMemory,
    tryMmap: !!settings.tryMmap,
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
    minDraftTokens: spec === "mtp" ? settings.minDraftTokens : 0,
    draftProbability: spec === "mtp" ? settings.draftProbability : 0,
    draftModelPath: spec === "dflash" ? settings.draftModelPath || "" : "",
    draftGpuOffload: spec === "dflash" ? settings.draftGpuOffload : 0,
  });
}

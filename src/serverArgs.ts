import { LlamaLoadSettings } from "./types";

/** Build llama-server CLI args from load settings + model path. */
export function buildServerArgs(
  modelPath: string,
  host: string,
  port: number,
  settings: LlamaLoadSettings
): string[] {
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
    "--cache-reuse",
    String(Math.max(0, settings.contextCheckpoints)),
  ];

  if (settings.nCpuMoe > 0) {
    args.push("--n-cpu-moe", String(settings.nCpuMoe));
  }

  if (!settings.offloadKvCacheToGpu) {
    args.push("--no-kv-offload");
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

  // Speculative / MTP (llama.cpp ≥ ~b10xxx renamed --draft → --spec-*).
  if (settings.speculativeMode === "mtp") {
    args.push("--spec-type", "draft-mtp");
    args.push("--spec-draft-n-max", String(Math.max(1, settings.maxDraftTokens)));
    if (settings.minDraftTokens > 0) {
      args.push("--spec-draft-n-min", String(settings.minDraftTokens));
    }
    if (settings.draftProbability > 0) {
      args.push("--spec-draft-p-min", String(settings.draftProbability));
    }
  }

  return args;
}

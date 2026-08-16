/** llama-server OpenAI-compat `timings` object (fields vary by build). */
export interface LlamaTimings {
  /** Prompt tokens served from the KV prefix cache (-1 / absent on older builds). */
  cache_n?: number;
  prompt_n?: number;
  prompt_ms?: number;
  prompt_per_second?: number;
  predicted_n?: number;
  predicted_ms?: number;
  predicted_per_second?: number;
  /** Speculative / MTP draft tokens generated. */
  draft_n?: number;
  /** Speculative / MTP draft tokens accepted. */
  draft_n_accepted?: number;
}

/** Finite tok/s greater than zero — 0/NaN from llama.cpp means "not measured yet". */
export function positiveRate(n: unknown): number | undefined {
  return typeof n === "number" && Number.isFinite(n) && n > 0 ? n : undefined;
}

/**
 * Generation / prompt tok/s from a timings object.
 * Prefer `*_per_second`; fall back to n/ms when the rate field is 0 or absent
 * (some builds log `tg` to the console but leave `predicted_per_second` unset).
 */
export function ratesFromTimings(t?: LlamaTimings): {
  genTokPerSec?: number;
  promptTokPerSec?: number;
} {
  if (!t) {
    return {};
  }
  const gen =
    positiveRate(t.predicted_per_second) ??
    (typeof t.predicted_n === "number" &&
    typeof t.predicted_ms === "number" &&
    t.predicted_n > 0 &&
    t.predicted_ms > 0
      ? t.predicted_n / (t.predicted_ms / 1000)
      : undefined);
  const prompt =
    positiveRate(t.prompt_per_second) ??
    (typeof t.prompt_n === "number" &&
    typeof t.prompt_ms === "number" &&
    t.prompt_n > 0 &&
    t.prompt_ms > 0
      ? t.prompt_n / (t.prompt_ms / 1000)
      : undefined);
  const out: { genTokPerSec?: number; promptTokPerSec?: number } = {};
  const genRate = positiveRate(gen);
  const promptRate = positiveRate(prompt);
  if (genRate !== undefined) {
    out.genTokPerSec = genRate;
  }
  if (promptRate !== undefined) {
    out.promptTokPerSec = promptRate;
  }
  return out;
}

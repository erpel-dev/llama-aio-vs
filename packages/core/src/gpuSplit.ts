/** llama.cpp --split-mode */
export type GpuSplitMode = "layer" | "row" | "none";

export const GPU_SPLIT_MODES: readonly GpuSplitMode[] = ["layer", "row", "none"];

export function normalizeGpuSplitMode(
  value: unknown,
  fallback: GpuSplitMode = "layer"
): GpuSplitMode {
  return typeof value === "string" && (GPU_SPLIT_MODES as readonly string[]).includes(value)
    ? (value as GpuSplitMode)
    : fallback;
}

/**
 * Parse llama.cpp `--tensor-split` ("3,1" / "0.75,0.25") into ≥2 positive numbers.
 * Invalid or a single value → [] (omit the flag; llama.cpp splits by VRAM).
 */
export function parseTensorSplit(raw: unknown): number[] {
  if (typeof raw !== "string") {
    return [];
  }
  const s = raw.trim();
  if (!s) {
    return [];
  }
  const parts = s
    .split(/[,/;:\s]+/)
    .map((p) => Number(p))
    .filter((n) => Number.isFinite(n) && n > 0)
    .slice(0, 8);
  return parts.length >= 2 ? parts : [];
}

/** Canonical "3,1" / "0.75,0.25", or "" if empty/invalid. */
export function normalizeTensorSplit(raw: unknown): string {
  const parts = parseTensorSplit(raw);
  if (parts.length < 2) {
    return "";
  }
  return parts.map((n) => (Number.isInteger(n) ? String(n) : String(n))).join(",");
}

/**
 * Per-GPU fractions for weights + KV. Explicit `--tensor-split` wins;
 * otherwise llama.cpp's default of VRAM-proportional shares.
 *
 * Shares are always in **device-index order** (GPU0, GPU1, …), matching
 * llama.cpp `--tensor-split`. `--main-gpu` does not reorder this string.
 */
export function tensorSplitShares(
  tensorSplit: string | undefined,
  gpuCount: number,
  vramBytes: number[]
): number[] {
  const n = Math.max(1, Math.round(gpuCount) || 1);
  if (n === 1) {
    return [1];
  }
  const parsed = parseTensorSplit(tensorSplit || "");
  if (parsed.length >= 2) {
    const parts = parsed.slice(0, n);
    while (parts.length < n) {
      parts.push(0);
    }
    const sum = parts.reduce((a, b) => a + b, 0) || 1;
    return parts.map((p) => p / sum);
  }
  const totals = vramBytes.slice(0, n);
  while (totals.length < n) {
    totals.push(0);
  }
  const sum = totals.reduce((a, b) => a + b, 0);
  if (sum <= 0) {
    return Array.from({ length: n }, () => 1 / n);
  }
  return totals.map((b) => b / sum);
}

/**
 * Weight/KV shares for the memory bars and fit check.
 * `--split-mode none` parks everything on `--main-gpu` (the other cards stay empty).
 */
export function effectiveTensorSplitShares(
  tensorSplit: string | undefined,
  splitMode: GpuSplitMode | undefined,
  mainGpu: number,
  gpuCount: number,
  vramBytes: number[]
): number[] {
  const n = clampGpuCount(gpuCount);
  if (n === 1) {
    return [1];
  }
  if (splitMode === "none") {
    const main = clampMainGpuIndex(mainGpu, n);
    return Array.from({ length: n }, (_, i) => (i === main ? 1 : 0));
  }
  return tensorSplitShares(tensorSplit, n, vramBytes);
}

export const MAIN_GPU_SHARE_MIN = 0.1;
export const MAIN_GPU_SHARE_MAX = 0.9;
export const DEFAULT_MAIN_GPU_SHARE = 0.75;

function clampGpuCount(gpuCount: number): number {
  return Math.max(1, Math.round(gpuCount) || 1);
}

function clampMainGpuIndex(mainGpu: number, gpuCount: number): number {
  const n = clampGpuCount(gpuCount);
  const m = Math.round(Number(mainGpu) || 0);
  return Math.min(Math.max(0, m), n - 1);
}

function clampMainShare(mainShare: number): number {
  const n = Number(mainShare);
  if (!Number.isFinite(n)) {
    return DEFAULT_MAIN_GPU_SHARE;
  }
  return Math.min(MAIN_GPU_SHARE_MAX, Math.max(MAIN_GPU_SHARE_MIN, n));
}

/** Chart / summary order: main GPU first, then the others in device-index order. */
export function gpuDisplayOrder(gpuCount: number, mainGpu: number): number[] {
  const n = Math.max(0, Math.round(gpuCount) || 0);
  const order = Array.from({ length: n }, (_, i) => i);
  if (n < 2) {
    return order;
  }
  const main = clampMainGpuIndex(mainGpu, n);
  if (main > 0) {
    order.splice(main, 1);
    order.unshift(main);
  }
  return order;
}

/** Fraction of weights+KV currently on `--main-gpu` (device-order split). */
export function mainShareFromSplit(
  tensorSplit: string | undefined,
  mainGpu: number,
  gpuCount: number,
  vramBytes: number[]
): number {
  const n = clampGpuCount(gpuCount);
  const shares = tensorSplitShares(tensorSplit, n, vramBytes);
  return shares[clampMainGpuIndex(mainGpu, n)] ?? 1;
}

/**
 * UI share for the Main GPU: existing `--tensor-split` share, but never less
 * than the largest device share. Picking GPU 1 as Main with a leftover `3,1`
 * (75% on GPU 0) becomes 75% on GPU 1.
 */
export function mainShareForUi(
  tensorSplit: string | undefined,
  mainGpu: number,
  gpuCount: number,
  vramBytes: number[]
): number {
  const n = clampGpuCount(gpuCount);
  const shares = tensorSplitShares(tensorSplit, n, vramBytes);
  const current = shares[clampMainGpuIndex(mainGpu, n)] ?? 1;
  if (n < 2 || parseTensorSplit(tensorSplit || "").length < 2) {
    return current;
  }
  return Math.max(current, ...shares);
}

/**
 * llama.cpp `--tensor-split` in device-index order so `mainGpu` gets
 * `mainShare` of weights+KV. Remainder is split evenly across the others.
 * Empty for a single GPU.
 */
export function tensorSplitForMainShare(
  mainShare: number,
  mainGpu: number,
  gpuCount: number
): string {
  const n = clampGpuCount(gpuCount);
  if (n < 2) {
    return "";
  }
  const share = clampMainShare(mainShare);
  const main = clampMainGpuIndex(mainGpu, n);
  const mainPct = Math.round(share * 100);
  const restPct = 100 - mainPct;
  const others = n - 1;
  const base = Math.floor(restPct / others);
  let rem = restPct - base * others;
  const percents = Array.from({ length: n }, (_, i) => {
    if (i === main) {
      return mainPct;
    }
    const extra = rem > 0 ? 1 : 0;
    if (rem > 0) {
      rem -= 1;
    }
    return base + extra;
  });
  return percents.join(",");
}

/**
 * Rewrite an explicit device-order split so the Main GPU receives the UI
 * share (at least the largest existing slice). Empty / auto is left empty.
 */
export function alignTensorSplitToMainGpu(
  tensorSplit: string | undefined,
  mainGpu: number,
  gpuCount: number,
  vramBytes: number[]
): string {
  const n = clampGpuCount(gpuCount);
  if (n < 2 || parseTensorSplit(tensorSplit || "").length < 2) {
    return "";
  }
  return tensorSplitForMainShare(
    mainShareForUi(tensorSplit, mainGpu, n, vramBytes),
    mainGpu,
    n
  );
}

export function tensorSplitSharesEqual(
  a: string | undefined,
  b: string | undefined,
  gpuCount: number,
  vramBytes: number[]
): boolean {
  const n = clampGpuCount(gpuCount);
  const sa = tensorSplitShares(a, n, vramBytes);
  const sb = tensorSplitShares(b, n, vramBytes);
  if (sa.length !== sb.length) {
    return false;
  }
  return sa.every((v, i) => Math.abs(v - (sb[i] || 0)) < 0.005);
}

/**
 * Old UI presets that always put the larger slice on GPU 0 (`3,1` = 75% GPU0).
 * Combined with Main GPU ≠ 0 this is the dual-GPU bug: the faster card was
 * selected as Main but still received the smaller share.
 */
const LEGACY_GPU0_FIRST = new Set(["3,1", "2,1", "4,1", "3,2"]);

export function isLegacyGpu0FirstSplit(tensorSplit: string | undefined): boolean {
  return LEGACY_GPU0_FIRST.has(normalizeTensorSplit(tensorSplit));
}

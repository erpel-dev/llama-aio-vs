/**
 * Live multi-GPU split math for the sidebar sliders. Mirrors core/src/gpuSplit.ts so the
 * form can preview a split without a host round-trip. Strictly typed.
 */

/** GPU fields the sidebar receives in its state payload. */
export interface SidebarGpu {
  totalBytes?: number;
  usedBytes?: number;
  name?: string;
  shortName?: string;
  index?: number;
  llamaDeviceId?: string;
}

/** CPU-offload knobs that change how much weight a GPU layer carries. */
export interface MassOpts {
  isMoe?: boolean;
  nCpuMoe?: number;
  moeExpertShare?: number;
  nCpuFfn?: number;
  denseFfnShare?: number;
}

/** GGUF dims the share estimates need (subset of the state's memInputs). */
export interface ShareInputs {
  isMoe?: boolean;
  moeExpertShare?: number | null;
  expertCount?: number;
  denseFfnShare?: number | null;
  ffnLength?: number;
  embeddingLength?: number;
}

export type SplitMode = string;

/** Short device name ("RX 9070"); the full name and llama.cpp id live in tooltips. */
export function gpuLabel(gpu: SidebarGpu | undefined, index: number): string {
  if (gpu && gpu.shortName) return String(gpu.shortName);
  const id = (gpu && gpu.llamaDeviceId) ? String(gpu.llamaDeviceId) : ('GPU ' + index);
  return id;
}

export function gpuFullLabel(gpu: SidebarGpu | undefined, index: number): string {
  const name = (gpu && gpu.name) ? String(gpu.name).trim() : '';
  const id = (gpu && gpu.llamaDeviceId) ? String(gpu.llamaDeviceId) : ('GPU ' + index);
  return name ? (id + ' · ' + name) : id;
}

export function parseTensorSplit(raw: unknown): number[] {
  const s = String(raw || '').trim();
  if (!s) return [];
  const parts = s.split(/[,/;:\s]+/).map(Number).filter((n) => isFinite(n) && n >= 0).slice(0, 8);
  return parts.length >= 2 && parts.some((n) => n > 0) ? parts : [];
}

export function clampMainGpu(mainGpu: unknown, n: number): number {
  return Math.min(Math.max(0, Math.round(Number(mainGpu) || 0)), Math.max(0, n - 1));
}

export function tensorSplitShares(raw: unknown, gpus: SidebarGpu[] | undefined): number[] {
  const n = Math.max(1, (gpus && gpus.length) || 1);
  if (n === 1) return [1];
  const parsed = parseTensorSplit(raw);
  if (parsed.length >= 2) {
    const parts = parsed.slice(0, n);
    while (parts.length < n) parts.push(0);
    const sum = parts.reduce((a, b) => a + b, 0) || 1;
    return parts.map((p) => p / sum);
  }
  const totals = (gpus || []).map((g) => g.totalBytes || 0);
  while (totals.length < n) totals.push(0);
  const sum = totals.reduce((a, b) => a + b, 0);
  if (sum <= 0 || totals.some((b) => b <= 0)) return Array.from({ length: n }, () => 1 / n);
  return totals.map((b) => b / sum);
}

export function effectiveTensorSplitShares(
  raw: unknown,
  gpus: SidebarGpu[] | undefined,
  splitMode: SplitMode,
  mainGpu: unknown
): number[] {
  const n = Math.max(1, (gpus && gpus.length) || 1);
  if (n === 1) return [1];
  if (splitMode === 'none') {
    const main = clampMainGpu(mainGpu, n);
    return Array.from({ length: n }, (_, i) => (i === main ? 1 : 0));
  }
  return tensorSplitShares(raw, gpus);
}

// Mirrors assignLayerDevices / layerAwareWeightShares in gpuSplit.ts.
export function layerAwareWeightSharesLive(
  nLayers: number,
  onGpu: number,
  shares: number[],
  splitMode: SplitMode,
  mainGpu: unknown,
  opts: MassOpts | undefined
): number[] {
  const nDev = Math.max(1, (shares && shares.length) || 1);
  const fallback = shares && shares.length === nDev ? shares : Array.from({ length: nDev }, () => 1 / nDev);
  if (splitMode === 'row' || splitMode === 'tensor') return fallback;
  const n = Math.max(0, Math.round(nLayers) || 0);
  const gpuN = Math.min(n, Math.max(0, Math.round(onGpu) || 0));
  const start = n - gpuN;
  const assign = Array.from({ length: n }, () => -1);
  const main = clampMainGpu(mainGpu, nDev);
  if (gpuN <= 0) return fallback;
  if (splitMode === 'none' || nDev === 1) {
    for (let i = start; i < n; i++) assign[i] = main;
  } else {
    for (let k = 0; k < gpuN; k++) {
      const pos = (k + 1) / gpuN;
      let cum = 0;
      let dev = nDev - 1;
      for (let d = 0; d < nDev; d++) {
        cum += shares[d] || 0;
        if (pos <= cum + 1e-9) { dev = d; break; }
      }
      assign[start + k] = dev;
    }
  }
  const mass = Array.from({ length: nDev }, () => 0);
  let tot = 0;
  const moe = !!(opts && opts.isMoe);
  const nCpuMoe = Math.max(0, Math.round((opts && opts.nCpuMoe) || 0));
  const moeShare = Math.min(0.98, Math.max(0, (opts && opts.moeExpertShare) || 0));
  const nCpuFfn = Math.max(0, Math.round((opts && opts.nCpuFfn) || 0));
  const ffnShare = Math.min(0.95, Math.max(0, (opts && opts.denseFfnShare) || 0));
  for (let i = 0; i < assign.length; i++) {
    const d = assign[i];
    if (d < 0 || d >= nDev) continue;
    let m = 1;
    if (moe && i < nCpuMoe && moeShare > 0) m = Math.max(0.02, 1 - moeShare);
    else if (!moe && i < nCpuFfn && ffnShare > 0) m = Math.max(0.05, 1 - ffnShare);
    mass[d] += m;
    tot += m;
  }
  if (tot <= 0) return fallback;
  return mass.map((x) => x / tot);
}

export function needsWeightAwareTensorSplitLive(splitMode: SplitMode, opts: MassOpts | undefined): boolean {
  if (splitMode === 'row' || splitMode === 'tensor' || splitMode === 'none') return false;
  if (opts && opts.isMoe) return (opts.nCpuMoe || 0) > 0 && (opts.moeExpertShare || 0) > 0;
  return !!(opts && (opts.nCpuFfn || 0) > 0 && (opts.denseFfnShare || 0) > 0);
}

export function mainWeightShareFromSplitLive(
  raw: unknown,
  mainGpu: unknown,
  gpus: SidebarGpu[] | undefined,
  splitMode: SplitMode,
  opts: MassOpts | undefined,
  nLayers: number,
  onGpu: number
): number {
  const shares = effectiveTensorSplitShares(raw, gpus, splitMode, mainGpu);
  const n = (gpus && gpus.length) || 1;
  const main = clampMainGpu(mainGpu, n);
  if (n < 2 || !needsWeightAwareTensorSplitLive(splitMode, opts)) return shares[main] || 1;
  const weight = layerAwareWeightSharesLive(nLayers, onGpu, shares, splitMode, mainGpu, opts);
  return weight[main] || shares[main] || 1;
}

export function tensorSplitForTargetWeightShareLive(
  targetShare: number,
  mainGpu: unknown,
  gpuCount: number,
  nLayers: number,
  onGpu: number,
  splitMode: SplitMode,
  opts: MassOpts | undefined,
  gpus: SidebarGpu[] | undefined
): string {
  const n = Math.max(1, Math.round(gpuCount) || 1);
  if (n < 2) return '';
  const target = Math.min(0.9, Math.max(0.1, Number(targetShare) || 0.5));
  if (nLayers < 1 || !needsWeightAwareTensorSplitLive(splitMode, opts)) {
    return tensorSplitForMainShare(target, mainGpu, n);
  }
  const main = clampMainGpu(mainGpu, n);
  const targetPct = Math.round(target * 100);
  let best = tensorSplitForMainShare(target, mainGpu, n);
  let bestErr = Infinity;
  let bestPct = targetPct;
  for (let pct = 10; pct <= 90; pct++) {
    const candidate = tensorSplitForMainShare(pct / 100, mainGpu, n);
    const shares = tensorSplitShares(candidate, gpus);
    const weight = layerAwareWeightSharesLive(nLayers, onGpu, shares, splitMode || 'layer', mainGpu, opts);
    const err = Math.abs((weight[main] || 0) - target);
    if (err < bestErr - 1e-12) {
      best = candidate;
      bestErr = err;
      bestPct = pct;
    } else if (err <= bestErr + 1e-12) {
      const closer = Math.abs(pct - targetPct) < Math.abs(bestPct - targetPct);
      if (closer || (pct < bestPct && Math.abs(pct - targetPct) === Math.abs(bestPct - targetPct))) {
        best = candidate;
        bestErr = err;
        bestPct = pct;
      }
    }
  }
  return best;
}

export function tensorSplitForMainShare(mainShare: unknown, mainGpu: unknown, gpuCount: number): string {
  const n = Math.max(1, Math.round(gpuCount) || 1);
  if (n < 2) return '';
  const share = Math.min(0.9, Math.max(0.1, Number(mainShare) || 0.75));
  const main = clampMainGpu(mainGpu, n);
  const mainPct = Math.round(share * 100);
  const restPct = 100 - mainPct;
  const others = n - 1;
  const base = Math.floor(restPct / others);
  let rem = restPct - base * others;
  const percents: number[] = [];
  for (let i = 0; i < n; i++) {
    if (i === main) {
      percents.push(mainPct);
    } else {
      const extra = rem > 0 ? 1 : 0;
      if (rem > 0) rem -= 1;
      percents.push(base + extra);
    }
  }
  return percents.join(',');
}

export function isLegacyGpu0FirstSplit(raw: unknown): boolean {
  const n = String(raw || '').replace(/\s+/g, '');
  return n === '3,1' || n === '2,1' || n === '4,1' || n === '3,2';
}

export function moeExpertShareOf(inputs: ShareInputs | null | undefined): number {
  if (!inputs || !inputs.isMoe) return 0;
  if (inputs.moeExpertShare != null && isFinite(inputs.moeExpertShare)) {
    return Math.min(0.98, Math.max(0.05, Number(inputs.moeExpertShare)));
  }
  const n = Number(inputs.expertCount) || 0;
  if (n >= 128) return 0.9;
  if (n >= 64) return 0.85;
  if (n >= 16) return 0.8;
  return 0.75;
}

// Mirrors resolveDenseFfnShare in core/memoryEstimate.ts.
export function denseFfnShareOf(inputs: ShareInputs | null | undefined): number {
  if (!inputs || inputs.isMoe) return 0;
  if (inputs.denseFfnShare != null && isFinite(inputs.denseFfnShare)) {
    return Math.min(0.95, Math.max(0.05, Number(inputs.denseFfnShare)));
  }
  const ffn = Number(inputs.ffnLength) || 0;
  const embed = Number(inputs.embeddingLength) || 0;
  if (ffn <= 0 || embed <= 0) return 0.7;
  const ffnElems = 3 * ffn;
  const attnElems = 4 * embed;
  return Math.min(0.95, Math.max(0.05, ffnElems / (ffnElems + attnElems)));
}

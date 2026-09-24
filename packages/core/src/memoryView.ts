import { formatGpuDeviceLabel, GpuMemoryInfo, isIntegratedGpu } from "./gpuInfo";
import { ModelCapabilities } from "./ggufMetadata";
import {
  deviceWouldSpill,
  estimateMemory,
  gpuBarCapacityBytes,
  MemoryBarSegment,
  MemoryEstimate,
  resolveDraftCapabilities,
  VRAM_HEADROOM_BYTES,
} from "./memoryEstimate";
import { fittingContextLength, recommendLoadSettings } from "./recommendSettings";
import { LlamaLoadSettings } from "./types";

const GiB = 1024 ** 3;
const MiB = 1024 ** 2;

/** Sidebar unit style: GiB with one decimal, MiB below 1 GiB. */
function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }
  if (bytes >= GiB) {
    return `${(bytes / GiB).toFixed(1)} GiB`;
  }
  return `${Math.max(1, Math.round(bytes / MiB))} MiB`;
}

/** good = headroom kept · tight = fits but below the headroom target · spill = over capacity. */
export type FitLevel = "good" | "tight" | "spill" | "unknown";

const LEVEL_RANK: Record<FitLevel, number> = { good: 0, unknown: 1, tight: 2, spill: 3 };

export interface DeviceUsageView {
  key: string;
  kind: "gpu" | "igpu" | "ram";
  /** Short name for the sidebar ("RX 9070"). */
  label: string;
  /** Full device label for tooltips ("Vulkan0 · AMD Radeon RX 9070 …"). */
  fullLabel: string;
  gpuIndex?: number;
  usedBytes: number;
  capacityBytes?: number;
  freeBytes?: number;
  /** Current occupancy reported by the driver (not part of the estimate). */
  liveUsedBytes?: number;
  segments: MemoryBarSegment[];
  level: FitLevel;
}

export interface MemoryFix {
  id: "context" | "gpus" | "kv" | "auto";
  label: string;
  /** Outcome, e.g. "2.0 GiB free" or "fits". */
  detail: string;
  patch: Partial<LlamaLoadSettings>;
  level: FitLevel;
}

export interface MemoryView {
  level: FitLevel;
  headline: string;
  targetFreeBytes: number;
  /** Devices that receive part of the model (bars). */
  devices: DeviceUsageView[];
  /** GPUs that get nothing under the current split ("RX 9060 XT idle"). */
  idle: string[];
  /** One dim line under the bars (small system-RAM share). */
  footnote: string;
  /** Soft notes for the Breakdown drawer (device-fit warnings are folded into the headline). */
  notes: string[];
  lines: string[];
  fixes: MemoryFix[];
}

export interface MemoryViewOptions {
  cpuOnly?: boolean;
  draftCaps?: ModelCapabilities;
  /** Compute one-click fixes (a few extra estimates) when the verdict is not good. */
  withFixes?: boolean;
}

/** "AMD Radeon RX 9070/9070 XT/9070 GRE" → "RX 9070"; "NVIDIA GeForce RTX 4070" → "RTX 4070". */
export function shortGpuName(gpu: Pick<GpuMemoryInfo, "name" | "llamaDeviceId"> | undefined, index: number): string {
  let name = (gpu?.name || "").trim();
  if (!name || /^(amdgpu|nvidia|i915|xe)$/i.test(name)) {
    return (gpu?.llamaDeviceId || "").trim() || `GPU ${index}`;
  }
  name = name.replace(/\(.*?\)/g, " ");
  // "RX 9070/9070 XT/9070 GRE" are variants of one chip → "RX 9070". "RX 9050 / 9060 XT"
  // is one device id shared by two different chips → keep both rather than guess.
  const alts = name.split("/").map((s) => s.trim()).filter(Boolean);
  if (alts.length > 1) {
    const num = /(\d{3,5})/.exec(alts[0]!)?.[1];
    const sameChip = !!num && alts.slice(1).every((a) => a.startsWith(num));
    name = sameChip ? alts[0]! : alts.join("/");
  }
  name = name.replace(/\b(AMD|ATI|NVIDIA|Intel\(R\)|Intel|Advanced Micro Devices,? Inc\.?|Corporation)\b/gi, " ");
  name = name.replace(/\b(Radeon|GeForce)\b\s*(?=(RX|RTX|GTX|Pro|VII)\b)/gi, " ");
  name = name.replace(/\s+/g, " ").trim();
  return name || `GPU ${index}`;
}

function levelFor(used: number, cap: number | undefined, integrated: boolean): FitLevel {
  if (!(cap && cap > 0)) {
    return "unknown";
  }
  if (used > cap) {
    return "spill";
  }
  if (!integrated && deviceWouldSpill(used, cap)) {
    return "tight";
  }
  return "good";
}

function worst(levels: FitLevel[]): FitLevel {
  let out: FitLevel = "good";
  for (const l of levels) {
    if (LEVEL_RANK[l] > LEVEL_RANK[out]) {
      out = l;
    }
  }
  return out;
}

/** Device-fit sentences in `warnings` that the headline replaces. */
const DEVICE_FIT_WARNING = /^(Tight on |Getting full |Estimated .* at full context ~.* is over the full )/;

function devicesOf(est: MemoryEstimate, gpus: GpuMemoryInfo[], cpuOnly: boolean): {
  devices: DeviceUsageView[];
  idle: string[];
  ram: DeviceUsageView;
} {
  const ramChart = est.charts.ram;
  const ram: DeviceUsageView = {
    key: "ram",
    kind: "ram",
    label: "System RAM",
    fullLabel: "System RAM",
    usedBytes: ramChart.totalBytes,
    capacityBytes: ramChart.capacityBytes,
    freeBytes: ramChart.capacityBytes ? ramChart.capacityBytes - ramChart.totalBytes : undefined,
    segments: ramChart.segments,
    level: ramChart.capacityBytes
      ? ramChart.totalBytes > ramChart.capacityBytes
        ? "spill"
        : ramChart.totalBytes > ramChart.capacityBytes * 0.9
          ? "tight"
          : "good"
      : "unknown",
  };
  if (cpuOnly) {
    return { devices: [ram], idle: [], ram };
  }
  const devices: DeviceUsageView[] = [];
  const idle: string[] = [];
  const charts = est.charts.gpus?.length
    ? est.charts.gpus
    : [{ ...est.charts.vram, gpuIndex: 0 }];
  for (const chart of charts) {
    const g = gpus[chart.gpuIndex];
    const label = g ? shortGpuName(g, chart.gpuIndex) : "GPU";
    if (!(chart.totalBytes > 0)) {
      if (g) {
        idle.push(label);
      }
      continue;
    }
    const integrated = isIntegratedGpu(g);
    const cap = chart.capacityBytes ?? gpuBarCapacityBytes(g);
    devices.push({
      key: `gpu${chart.gpuIndex}`,
      kind: integrated ? "igpu" : "gpu",
      label,
      fullLabel: g ? formatGpuDeviceLabel(g, chart.gpuIndex) : "GPU",
      gpuIndex: chart.gpuIndex,
      usedBytes: chart.totalBytes,
      capacityBytes: cap,
      freeBytes: cap ? cap - chart.totalBytes : undefined,
      liveUsedBytes: g?.usedBytes,
      segments: chart.segments,
      level: levelFor(chart.totalBytes, cap, integrated),
    });
  }
  // Partial offload / CPU MoE: RAM holds real weight, so it earns a bar.
  if (ram.usedBytes >= 2 * GiB || ram.level !== "good") {
    devices.push(ram);
  }
  return { devices, idle, ram };
}

function headlineFor(level: FitLevel, devices: DeviceUsageView[], cpuOnly: boolean, est: MemoryEstimate): string {
  const gpuDevs = devices.filter((d) => d.kind !== "ram");
  const names = (list: DeviceUsageView[]) => list.map((d) => d.label).join(" + ");
  if (cpuOnly) {
    const ram = devices[0]!;
    if (level === "spill") {
      return `~${formatBytes(ram.usedBytes - (ram.capacityBytes || 0))} more than system RAM. Lower context or pick a smaller quant.`;
    }
    if (level === "tight") {
      return `Uses ~${formatBytes(ram.usedBytes)} of ${formatBytes(ram.capacityBytes || 0)} system RAM — little left for the OS.`;
    }
    return ram.capacityBytes
      ? `Fits in system RAM with ${formatBytes(ram.freeBytes || 0)} to spare (CPU backend).`
      : `~${formatBytes(ram.usedBytes)} of system RAM (CPU backend).`;
  }
  if (!gpuDevs.length) {
    return est.layersOnGpu > 0
      ? `~${formatBytes(est.totalGpuBytes)} VRAM — GPU memory could not be detected.`
      : `Runs from system RAM (GPU offload 0): ~${formatBytes(est.totalCpuBytes)}.`;
  }
  if (level === "unknown") {
    return `~${formatBytes(est.totalGpuBytes)} VRAM — GPU capacity unknown.`;
  }
  const worstDev = [...gpuDevs].sort((a, b) => (a.freeBytes ?? 0) - (b.freeBytes ?? 0))[0]!;
  if (level === "spill") {
    const overGpu = gpuDevs.find((d) => d.level === "spill");
    if (overGpu) {
      return `${formatBytes(overGpu.usedBytes - (overGpu.capacityBytes || 0))} over VRAM on ${overGpu.label}. It will spill to system RAM and generation will slow down a lot.`;
    }
    return "System RAM is over capacity. Lower context or GPU offload.";
  }
  if (level === "tight") {
    return `Fits, but only ${formatBytes(Math.max(0, worstDev.freeBytes ?? 0))} free on ${worstDev.label} (target ${formatBytes(VRAM_HEADROOM_BYTES)}).`;
  }
  if (gpuDevs.length === 1) {
    return `Fits on ${worstDev.label} with ${formatBytes(worstDev.freeBytes ?? 0)} to spare.`;
  }
  return `Fits on ${names(gpuDevs)} · at least ${formatBytes(worstDev.freeBytes ?? 0)} free on each.`;
}

function minGpuFree(view: Pick<MemoryView, "devices">): number | undefined {
  const free = view.devices
    .filter((d) => d.kind !== "ram" && d.freeBytes !== undefined)
    .map((d) => d.freeBytes!);
  return free.length ? Math.min(...free) : undefined;
}

function fmtTokens(n: number): string {
  if (n >= 1024 && n % 1024 === 0) {
    return `${n / 1024}k`;
  }
  return n >= 1000 ? `${Math.round(n / 1000)}k` : String(n);
}

/**
 * Turn a core estimate into the sidebar's fit verdict. The webview renders this
 * as-is; it has no estimator of its own.
 */
export function buildMemoryView(
  caps: ModelCapabilities | undefined,
  settings: LlamaLoadSettings,
  est: MemoryEstimate | undefined,
  gpus: GpuMemoryInfo[],
  options: MemoryViewOptions = {}
): MemoryView | undefined {
  if (!est) {
    return undefined;
  }
  const cpuOnly = !!options.cpuOnly;
  const { devices, idle, ram } = devicesOf(est, gpus, cpuOnly);
  let level = worst(devices.map((d) => d.level));
  // Keep the verdict in step with the start/reload spill confirmation.
  if (est.willSpill && LEVEL_RANK[level] < LEVEL_RANK.tight) {
    level = "tight";
  }
  const footnoteBits: string[] = [];
  if (idle.length) {
    footnoteBits.push(`${idle.join(", ")} idle`);
  }
  if (!cpuOnly && !devices.includes(ram)) {
    footnoteBits.push(`System RAM ${formatBytes(ram.usedBytes)}`);
  }
  const view: MemoryView = {
    level,
    headline: headlineFor(level, devices, cpuOnly, est),
    targetFreeBytes: VRAM_HEADROOM_BYTES,
    devices,
    idle,
    footnote: footnoteBits.join(" · "),
    notes: est.warnings.filter((w) => !DEVICE_FIT_WARNING.test(w)),
    lines: est.lines,
    fixes: [],
  };
  if (options.withFixes && caps && (level === "tight" || level === "spill")) {
    view.fixes = suggestMemoryFixes(caps, settings, gpus, options, view);
  }
  return view;
}

function viewFor(
  caps: ModelCapabilities,
  settings: LlamaLoadSettings,
  gpus: GpuMemoryInfo[],
  options: MemoryViewOptions
): MemoryView | undefined {
  const est = estimateMemory(caps, settings, options.cpuOnly ? undefined : gpus[0], {
    cpuOnly: options.cpuOnly,
    draftCaps: options.draftCaps,
    gpus: options.cpuOnly ? undefined : gpus,
  });
  return buildMemoryView(caps, settings, est, gpus, { ...options, withFixes: false });
}

const AUTO_FIT_KEYS: Array<keyof LlamaLoadSettings> = [
  "contextLength",
  "gpuOffload",
  "nCpuMoe",
  "nCpuFfn",
  "splitMode",
  "tensorSplit",
  "mainGpu",
];

/**
 * Up to three single-step changes that move the verdict toward "good", each
 * checked with the real estimator: shorter context, use idle GPUs, smaller
 * KV types, then the full recommend search.
 */
export function suggestMemoryFixes(
  caps: ModelCapabilities,
  settings: LlamaLoadSettings,
  gpus: GpuMemoryInfo[],
  options: MemoryViewOptions = {},
  current?: MemoryView
): MemoryFix[] {
  const cpuOnly = !!options.cpuOnly;
  const base = current ?? viewFor(caps, settings, gpus, options);
  if (!base) {
    return [];
  }
  const baseFree = minGpuFree(base) ?? -Infinity;
  const candidates: Array<Pick<MemoryFix, "id" | "label" | "patch">> = [];

  const fitCtx = fittingContextLength(caps, settings, { cpuOnly, gpus });
  if (fitCtx < settings.contextLength) {
    candidates.push({ id: "context", label: `Context ${fmtTokens(fitCtx)}`, patch: { contextLength: fitCtx } });
  }
  if (!cpuOnly && gpus.length >= 2 && settings.splitMode === "none" && base.idle.length) {
    candidates.push({
      id: "gpus",
      label: gpus.length === 2 ? "Use both GPUs" : `Use all ${gpus.length} GPUs`,
      patch: { splitMode: "layer", tensorSplit: "" },
    });
  }
  const highPrecision = (t: string) => t === "f32" || t === "f16" || t === "bf16";
  if (highPrecision(settings.cacheTypeK) || highPrecision(settings.cacheTypeV)) {
    candidates.push({ id: "kv", label: "KV q8_0", patch: { cacheTypeK: "q8_0", cacheTypeV: "q8_0" } });
  } else if (!["q4_0", "q4_1", "iq4_nl"].includes(settings.cacheTypeV) && settings.flashAttention !== "off") {
    // Quantized V needs flash attention; K stays put (it is the precision-sensitive half).
    candidates.push({ id: "kv", label: "KV V q4_0", patch: { cacheTypeV: "q4_0" } });
  }
  const rec = recommendLoadSettings(settings, caps, {
    cpuOnly,
    gpus,
    preferredContext: settings.contextLength,
  });
  const autoPatch: Partial<LlamaLoadSettings> = {};
  for (const key of AUTO_FIT_KEYS) {
    if (rec[key] !== settings[key]) {
      (autoPatch as Record<string, unknown>)[key] = rec[key];
    }
  }
  if (Object.keys(autoPatch).length) {
    candidates.push({ id: "auto", label: "Auto-fit", patch: autoPatch });
  }

  const fixes: MemoryFix[] = [];
  for (const c of candidates) {
    const after = viewFor(caps, { ...settings, ...c.patch }, gpus, options);
    if (!after) {
      continue;
    }
    const free = minGpuFree(after);
    const better =
      LEVEL_RANK[after.level] < LEVEL_RANK[base.level] ||
      (after.level === base.level && free !== undefined && free > baseFree + 0.1 * GiB);
    if (!better) {
      continue;
    }
    const detail =
      free === undefined
        ? after.level === "good"
          ? "fits"
          : after.level
        : free < 0
          ? `still ${formatBytes(-free)} over`
          : `${formatBytes(free)} free`;
    fixes.push({ ...c, detail, level: after.level });
  }
  // Fixes that reach a better verdict first; ties keep the least-disruptive order above.
  return fixes
    .map((f, i) => ({ f, i }))
    .sort((a, b) => LEVEL_RANK[a.f.level] - LEVEL_RANK[b.f.level] || a.i - b.i)
    .map(({ f }) => f)
    .slice(0, 3);
}

/** Estimate + verdict in one call: the single memory path for the sidebar and spill checks. */
export function computeMemoryView(
  caps: ModelCapabilities | undefined,
  settings: LlamaLoadSettings,
  gpus: GpuMemoryInfo[],
  options: MemoryViewOptions = {}
): { estimate: MemoryEstimate | undefined; view: MemoryView | undefined } {
  const cpuOnly = !!options.cpuOnly;
  const draftCaps = options.draftCaps ?? resolveDraftCapabilities(settings);
  const estimate = estimateMemory(caps, settings, cpuOnly ? undefined : gpus[0], {
    cpuOnly,
    draftCaps,
    gpus: cpuOnly ? undefined : gpus,
  });
  const view = buildMemoryView(caps, settings, estimate, gpus, { ...options, draftCaps });
  return { estimate, view };
}

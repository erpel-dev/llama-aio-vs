import { formatBytes, type MemoryBarChart } from "@llama-aio/core";
import { theme } from "./theme.js";

export type MemBarTone = "ok" | "warn" | "bad" | "muted";

export const MEM_SEG_FG = {
  weights: theme.memWeights,
  vision: theme.memVision,
  draft: theme.memDraft,
  kv: theme.memKv,
  overhead: theme.memOverhead,
} as const;

export function memToneFg(tone: MemBarTone): string {
  if (tone === "bad") {
    return theme.bad;
  }
  if (tone === "warn") {
    return theme.warn;
  }
  if (tone === "ok") {
    return theme.ok;
  }
  return theme.muted;
}

/** Subtitle + spill tone — same thresholds as the VS Code stacked bars (80% / 92%). */
export function chartSubtitle(chart: MemoryBarChart): { text: string; tone: MemBarTone } {
  const pct = chart.capacityBytes
    ? Math.round((chart.totalBytes / chart.capacityBytes) * 100)
    : undefined;
  const over = pct !== undefined && pct > 92;
  const warn = !over && pct !== undefined && pct > 80;
  return {
    text:
      `~${formatBytes(chart.totalBytes)}` +
      (chart.capacityBytes
        ? ` / ${formatBytes(chart.capacityBytes)}${pct !== undefined ? ` (${pct}%)` : ""}`
        : ""),
    tone: over ? "bad" : warn ? "warn" : "muted",
  };
}

export type MemBarPart = {
  key: "weights" | "vision" | "kv" | "overhead" | "draft" | "free";
  cols: number;
  fg: string;
};

/** Proportional column widths for a stacked bar of `width` cells. */
export function barParts(chart: MemoryBarChart, width: number): MemBarPart[] {
  const w = Math.max(8, Math.floor(width));
  const capacity = chart.capacityBytes || chart.totalBytes || 1;
  const scale = Math.max(capacity, chart.totalBytes || 0) || 1;
  const parts: MemBarPart[] = [];
  let used = 0;
  for (const seg of chart.segments) {
    if (!seg.bytes || seg.bytes <= 0) {
      continue;
    }
    const cols = Math.max(1, Math.round((seg.bytes / scale) * w));
    parts.push({
      key: seg.key,
      cols,
      fg: MEM_SEG_FG[seg.key],
    });
    used += cols;
  }
  while (used > w && parts.length) {
    const last = parts[parts.length - 1]!;
    if (last.cols > 1) {
      last.cols -= 1;
      used -= 1;
    } else {
      break;
    }
  }
  const free = Math.max(0, w - used);
  if (free > 0) {
    parts.push({ key: "free", cols: free, fg: theme.memTrack });
  }
  return parts;
}

export function barCell(cols: number): string {
  return cols > 0 ? "█".repeat(cols) : "";
}

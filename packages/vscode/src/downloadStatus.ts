/**
 * Status-bar text for the active Hugging Face download.
 */
import { formatBytes, type DownloadJobSnapshot } from "@llama-aio/core";

function formatSpeed(bytesPerSec: number): string {
  if (!bytesPerSec || bytesPerSec < 1) {
    return "";
  }
  return `${formatBytes(bytesPerSec)}/s`;
}

function formatEta(seconds?: number): string {
  if (!seconds || seconds < 0 || !Number.isFinite(seconds)) {
    return "";
  }
  if (seconds < 60) {
    return `${seconds}s left`;
  }
  if (seconds < 3600) {
    return `${Math.round(seconds / 60)} min left`;
  }
  return `${(seconds / 3600).toFixed(1)} h left`;
}

export function formatDownloadStatusLine(jobs: DownloadJobSnapshot[]): string {
  const active = jobs.filter((j) => j.state === "running" || j.state === "queued" || j.state === "paused");
  if (!active.length) {
    return "";
  }
  const run = active.find((j) => j.state === "running") || active[0]!;
  const pct = run.total > 0 ? Math.round((run.received / run.total) * 100) : 0;
  const speed = formatSpeed(run.bytesPerSec);
  const left = formatEta(run.etaSeconds);
  return `$(cloud-download) ${run.label} ${pct}%${speed ? " · " + speed : ""}${left ? " · " + left : ""}`;
}

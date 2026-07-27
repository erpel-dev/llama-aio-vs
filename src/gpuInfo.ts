import * as fs from "fs";
import { execFileSync } from "child_process";
import * as path from "path";

export interface GpuMemoryInfo {
  /** Total VRAM in bytes (largest detected discrete GPU). */
  totalBytes: number;
  /** Used VRAM if known. */
  usedBytes?: number;
  name?: string;
  source: string;
}

function readSysfsVram(): GpuMemoryInfo | undefined {
  if (process.platform !== "linux") {
    return undefined;
  }
  const roots = ["/sys/class/drm"];
  let best: GpuMemoryInfo | undefined;
  for (const root of roots) {
    let entries: string[];
    try {
      entries = fs.readdirSync(root);
    } catch {
      continue;
    }
    for (const ent of entries) {
      if (!ent.startsWith("card")) {
        continue;
      }
      const totalPath = path.join(root, ent, "device", "mem_info_vram_total");
      try {
        if (!fs.existsSync(totalPath)) {
          continue;
        }
        const totalBytes = Number(fs.readFileSync(totalPath, "utf8").trim());
        if (!Number.isFinite(totalBytes) || totalBytes < 512 * 1024 * 1024) {
          continue;
        }
        let usedBytes: number | undefined;
        const usedPath = path.join(root, ent, "device", "mem_info_vram_used");
        if (fs.existsSync(usedPath)) {
          const u = Number(fs.readFileSync(usedPath, "utf8").trim());
          if (Number.isFinite(u)) {
            usedBytes = u;
          }
        }
        let name: string | undefined;
        try {
          const uevent = fs.readFileSync(path.join(root, ent, "device", "uevent"), "utf8");
          const m = /DRIVER=(\S+)/.exec(uevent);
          if (m) {
            name = m[1];
          }
        } catch {
          // ignore
        }
        if (!best || totalBytes > best.totalBytes) {
          best = { totalBytes, usedBytes, name, source: `sysfs:${ent}` };
        }
      } catch {
        // ignore card
      }
    }
  }
  return best;
}

function readNvidiaSmi(): GpuMemoryInfo | undefined {
  try {
    const out = execFileSync(
      "nvidia-smi",
      ["--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
      { encoding: "utf8", timeout: 3000, windowsHide: true }
    ).trim();
    if (!out) {
      return undefined;
    }
    // Pick the GPU with most VRAM if multiple.
    let best: GpuMemoryInfo | undefined;
    for (const line of out.split("\n")) {
      const parts = line.split(",").map((s) => s.trim());
      if (parts.length < 2) {
        continue;
      }
      const name = parts[0];
      const totalMiB = Number(parts[1]);
      const usedMiB = parts[2] !== undefined ? Number(parts[2]) : undefined;
      if (!Number.isFinite(totalMiB)) {
        continue;
      }
      const totalBytes = totalMiB * 1024 * 1024;
      const usedBytes =
        usedMiB !== undefined && Number.isFinite(usedMiB) ? usedMiB * 1024 * 1024 : undefined;
      if (!best || totalBytes > best.totalBytes) {
        best = { totalBytes, usedBytes, name, source: "nvidia-smi" };
      }
    }
    return best;
  } catch {
    return undefined;
  }
}

function readRocmSmi(): GpuMemoryInfo | undefined {
  if (process.platform === "win32") {
    return undefined;
  }
  try {
    const out = execFileSync("rocm-smi", ["--showmeminfo", "vram", "--csv"], {
      encoding: "utf8",
      timeout: 4000,
    });
    // Heuristic parse: look for large integers that look like bytes or MiB.
    const nums = [...out.matchAll(/\b(\d{6,})\b/g)].map((m) => Number(m[1]));
    const candidates = nums.filter((n) => n > 512 * 1024 * 1024);
    if (candidates.length) {
      return {
        totalBytes: Math.max(...candidates),
        name: "AMD GPU",
        source: "rocm-smi",
      };
    }
  } catch {
    // ignore
  }
  return undefined;
}

/**
 * Windows fallback for AMD/Intel when nvidia-smi is absent.
 * Note: AdapterRAM is often a uint32 and can under-report >4 GiB cards.
 */
function readWindowsWmiVram(): GpuMemoryInfo | undefined {
  if (process.platform !== "win32") {
    return undefined;
  }
  try {
    const out = execFileSync(
      "powershell.exe",
      [
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json -Compress",
      ],
      { encoding: "utf8", timeout: 6000, windowsHide: true }
    ).trim();
    if (!out) {
      return undefined;
    }
    const parsed = JSON.parse(out) as
      | { Name?: string; AdapterRAM?: number }
      | Array<{ Name?: string; AdapterRAM?: number }>;
    const items = Array.isArray(parsed) ? parsed : [parsed];
    let best: GpuMemoryInfo | undefined;
    for (const item of items) {
      const name = String(item.Name || "");
      if (!name || /Microsoft Basic/i.test(name)) {
        continue;
      }
      const totalBytes = Number(item.AdapterRAM);
      if (!Number.isFinite(totalBytes) || totalBytes < 512 * 1024 * 1024) {
        continue;
      }
      if (!best || totalBytes > best.totalBytes) {
        best = { totalBytes, name, source: "wmi" };
      }
    }
    return best;
  } catch {
    return undefined;
  }
}

/** Detect primary GPU VRAM. Cached briefly because sysfs/nvidia-smi are stable. */
let cached: { at: number; info?: GpuMemoryInfo } | undefined;

export function detectGpuMemory(force = false): GpuMemoryInfo | undefined {
  const now = Date.now();
  if (!force && cached && now - cached.at < 15_000) {
    return cached.info;
  }
  const info =
    readNvidiaSmi() || readSysfsVram() || readRocmSmi() || readWindowsWmiVram();
  cached = { at: now, info };
  return info;
}

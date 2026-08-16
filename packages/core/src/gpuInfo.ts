import * as fs from "fs";
import { execFileSync, spawnSync } from "child_process";
import * as path from "path";

export interface GpuMemoryInfo {
  /** Total VRAM in bytes for this GPU. */
  totalBytes: number;
  /** Used VRAM if known. */
  usedBytes?: number;
  name?: string;
  source: string;
  /** PCI slot (`0000:28:00.0`) when known. */
  pciSlot?: string;
  /**
   * Index in llama.cpp `--device` / `--tensor-split` / `--main-gpu` order.
   * Vulkan often does **not** match PCI slot order.
   */
  index?: number;
  /** llama.cpp `--list-devices` id (`Vulkan0`, `CUDA1`) when known. */
  llamaDeviceId?: string;
}

export interface LlamaListedDevice {
  id: string;
  index: number;
  name: string;
  totalBytes?: number;
  usedBytes?: number;
}

const MIN_VRAM = 512 * 1024 * 1024;

function pciSortKey(slot?: string): string {
  return (slot || "zzzz").toLowerCase();
}

function sortAndIndex(gpus: GpuMemoryInfo[]): GpuMemoryInfo[] {
  const sorted = [...gpus].sort((a, b) => pciSortKey(a.pciSlot).localeCompare(pciSortKey(b.pciSlot)));
  return sorted.map((g, i) => ({ ...g, index: i }));
}

function dedupeByPci(gpus: GpuMemoryInfo[]): GpuMemoryInfo[] {
  const seen = new Set<string>();
  const out: GpuMemoryInfo[] = [];
  for (const g of gpus) {
    const key = g.pciSlot || `${g.source}:${g.name || ""}:${g.totalBytes}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(g);
  }
  return out;
}

function shortLspciName(line: string): string | undefined {
  // "28:00.0 VGA ...: Advanced Micro Devices, Inc. [AMD/ATI] Navi 48 [Radeon RX 9070/9070 XT/9070 GRE] [1002:7550]"
  const bracket = /\[([^[\]]*(?:Radeon|GeForce|RTX|Arc|Instinct)[^[\]]*)\]/i.exec(line);
  if (bracket) {
    return bracket[1].replace(/\s+/g, " ").trim();
  }
  const afterVendor = /:\s(?:Advanced Micro Devices[^:]*:\s)?(.+?)\s*\[[0-9a-f]{4}:[0-9a-f]{4}\]/i.exec(
    line
  );
  if (afterVendor) {
    return afterVendor[1]
      .replace(/^Advanced Micro Devices, Inc\. \[AMD\/ATI\]\s*/i, "")
      .replace(/^NVIDIA Corporation\s*/i, "")
      .replace(/^Intel Corporation\s*/i, "")
      .trim();
  }
  return undefined;
}

function lspciName(pciSlot: string): string | undefined {
  const short = pciSlot.replace(/^0000:/, "");
  try {
    const out = execFileSync("lspci", ["-s", short, "-nn"], {
      encoding: "utf8",
      timeout: 1500,
      windowsHide: true,
    }).trim();
    if (out) {
      return shortLspciName(out);
    }
  } catch {
    // ignore
  }
  return undefined;
}

function sysfsGpuName(deviceDir: string, uevent: string, pciSlot?: string): string | undefined {
  for (const file of ["product_name", "label"]) {
    try {
      const t = fs.readFileSync(path.join(deviceDir, file), "utf8").trim();
      if (t && !/^(amdgpu|nvidia|i915|xe)$/i.test(t)) {
        return t;
      }
    } catch {
      // ignore
    }
  }
  if (pciSlot) {
    const fromPci = lspciName(pciSlot);
    if (fromPci) {
      return fromPci;
    }
  }
  const driver = /DRIVER=(\S+)/.exec(uevent)?.[1];
  return driver || undefined;
}

function readSysfsGpus(): GpuMemoryInfo[] | undefined {
  if (process.platform !== "linux") {
    return undefined;
  }
  const root = "/sys/class/drm";
  let entries: string[];
  try {
    entries = fs.readdirSync(root);
  } catch {
    return undefined;
  }
  const found: GpuMemoryInfo[] = [];
  for (const ent of entries) {
    // Skip render nodes (card0-renderD128) — they share the same device.
    if (!/^card\d+$/.test(ent)) {
      continue;
    }
    const deviceDir = path.join(root, ent, "device");
    const totalPath = path.join(deviceDir, "mem_info_vram_total");
    try {
      if (!fs.existsSync(totalPath)) {
        continue;
      }
      const totalBytes = Number(fs.readFileSync(totalPath, "utf8").trim());
      if (!Number.isFinite(totalBytes) || totalBytes < MIN_VRAM) {
        continue;
      }
      let usedBytes: number | undefined;
      const usedPath = path.join(deviceDir, "mem_info_vram_used");
      if (fs.existsSync(usedPath)) {
        const u = Number(fs.readFileSync(usedPath, "utf8").trim());
        if (Number.isFinite(u)) {
          usedBytes = u;
        }
      }
      let uevent = "";
      try {
        uevent = fs.readFileSync(path.join(deviceDir, "uevent"), "utf8");
      } catch {
        // ignore
      }
      const pciSlot = /PCI_SLOT_NAME=(\S+)/.exec(uevent)?.[1];
      const name = sysfsGpuName(deviceDir, uevent, pciSlot);
      found.push({
        totalBytes,
        usedBytes,
        name,
        source: `sysfs:${ent}`,
        pciSlot,
      });
    } catch {
      // ignore card
    }
  }
  if (!found.length) {
    return undefined;
  }
  return sortAndIndex(dedupeByPci(found));
}

function readNvidiaSmiAll(): GpuMemoryInfo[] | undefined {
  try {
    const out = execFileSync(
      "nvidia-smi",
      [
        "--query-gpu=name,memory.total,memory.used,pci.bus_id",
        "--format=csv,noheader,nounits",
      ],
      { encoding: "utf8", timeout: 3000, windowsHide: true }
    ).trim();
    if (!out) {
      return undefined;
    }
    const found: GpuMemoryInfo[] = [];
    for (const line of out.split("\n")) {
      const parts = line.split(",").map((s) => s.trim());
      if (parts.length < 2) {
        continue;
      }
      const name = parts[0];
      const totalMiB = Number(parts[1]);
      const usedMiB = parts[2] !== undefined ? Number(parts[2]) : undefined;
      const pciSlot = parts[3] || undefined;
      if (!Number.isFinite(totalMiB)) {
        continue;
      }
      const totalBytes = totalMiB * 1024 * 1024;
      if (totalBytes < MIN_VRAM) {
        continue;
      }
      const usedBytes =
        usedMiB !== undefined && Number.isFinite(usedMiB) ? usedMiB * 1024 * 1024 : undefined;
      found.push({ totalBytes, usedBytes, name, source: "nvidia-smi", pciSlot });
    }
    if (!found.length) {
      return undefined;
    }
    return sortAndIndex(found);
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
    const candidates = nums.filter((n) => n > MIN_VRAM);
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
function readWindowsWmiGpus(): GpuMemoryInfo[] | undefined {
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
    const found: GpuMemoryInfo[] = [];
    for (const item of items) {
      const name = String(item.Name || "");
      if (!name || /Microsoft Basic/i.test(name)) {
        continue;
      }
      const totalBytes = Number(item.AdapterRAM);
      if (!Number.isFinite(totalBytes) || totalBytes < MIN_VRAM) {
        continue;
      }
      found.push({ totalBytes, name, source: "wmi" });
    }
    if (!found.length) {
      return undefined;
    }
    return sortAndIndex(found);
  } catch {
    return undefined;
  }
}

/** Detected GPUs, cached briefly because sysfs/nvidia-smi are stable. */
let cached: { at: number; binKey: string; gpus: GpuMemoryInfo[] } | undefined;

/**
 * Parse `llama-server --list-devices` text. Order is llama.cpp `--tensor-split`
 * / `--main-gpu` order (Vulkan0, Vulkan1, …) — not PCI slot order.
 */
export function parseLlamaListDevices(text: string): LlamaListedDevice[] {
  const out: LlamaListedDevice[] = [];
  const re = /^\s*(\S+): (.+?) \((\d+)\s*MiB(?:,\s*(\d+)\s*MiB free)?\)/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    const id = m[1] || "";
    const rawName = (m[2] || "").replace(/\s*\(RADV[^)]*\)/i, "").trim();
    const totalMiB = Number(m[3]);
    const freeMiB = m[4] != null && m[4] !== "" ? Number(m[4]) : undefined;
    const indexMatch = /(\d+)$/.exec(id);
    const index = indexMatch ? Number(indexMatch[1]) : out.length;
    const totalBytes = Number.isFinite(totalMiB) ? totalMiB * 1024 * 1024 : undefined;
    const usedBytes =
      totalBytes != null && freeMiB != null && Number.isFinite(freeMiB)
        ? Math.max(0, totalBytes - freeMiB * 1024 * 1024)
        : undefined;
    out.push({ id, index, name: rawName, totalBytes, usedBytes });
  }
  return out.sort((a, b) => a.index - b.index);
}

/** Distinctive model tokens (`rx9070`, `rtx4090`) used to pair sysfs names with llama.cpp. */
export function gpuModelTokens(name: string): string[] {
  const n = (name || "").toLowerCase();
  const tokens: string[] = [];
  const rx = /rx\s*(\d{3,4})/gi;
  let m: RegExpExecArray | null;
  while ((m = rx.exec(n))) {
    tokens.push(`rx${m[1]}`);
  }
  const nv = /(rtx|gtx)\s*(\d{3,4})/gi;
  while ((m = nv.exec(n))) {
    tokens.push(`${m[1]}${m[2]}`);
  }
  const arc = /arc\s+([a-z]?\d{3,4})/gi;
  while ((m = arc.exec(n))) {
    tokens.push(`arc${m[1]}`);
  }
  return [...new Set(tokens)];
}

/**
 * Reorder sysfs/PCI GPUs to llama.cpp `--list-devices` order and attach
 * `Vulkan0` / `CUDA0` ids. Unmatched cards keep PCI order at the end.
 */
export function orderGpusLikeLlama(
  gpus: GpuMemoryInfo[],
  devices: LlamaListedDevice[]
): GpuMemoryInfo[] {
  if (!devices.length || !gpus.length) {
    return gpus;
  }
  const unused = [...gpus];
  const ordered: GpuMemoryInfo[] = [];
  for (const dev of devices) {
    const tokens = gpuModelTokens(dev.name);
    let idx = tokens.length
      ? unused.findIndex((g) => gpuModelTokens(g.name || "").some((t) => tokens.includes(t)))
      : -1;
    if (idx < 0) {
      idx = unused.length ? 0 : -1;
    }
    if (idx < 0) {
      continue;
    }
    const g = unused.splice(idx, 1)[0];
    if (!g) {
      continue;
    }
    ordered.push({
      ...g,
      name: g.name && !/^(amdgpu|nvidia|i915|xe)$/i.test(g.name.trim()) ? g.name : dev.name,
      totalBytes: g.totalBytes || dev.totalBytes || 0,
      usedBytes: dev.usedBytes ?? g.usedBytes,
      index: ordered.length,
      llamaDeviceId: dev.id,
    });
  }
  for (const g of unused) {
    ordered.push({ ...g, index: ordered.length });
  }
  return ordered;
}

function readLlamaListDevices(binary: string): LlamaListedDevice[] | undefined {
  const bin = (binary || "").trim();
  if (!bin || bin === "llama-server" || !fs.existsSync(bin)) {
    return undefined;
  }
  try {
    const dir = path.dirname(bin);
    const lib = path.join(dir, "..", "lib");
    const ld = [dir, lib, process.env.LD_LIBRARY_PATH].filter(Boolean).join(path.delimiter);
    const result = spawnSync(bin, ["--list-devices"], {
      encoding: "utf8",
      timeout: 8000,
      windowsHide: true,
      cwd: dir,
      env: { ...process.env, LD_LIBRARY_PATH: ld },
    });
    const text = `${result.stdout || ""}\n${result.stderr || ""}`;
    const listed = parseLlamaListDevices(text);
    return listed.length ? listed : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Discrete GPUs. When `llamaServerBinary` is set, order matches llama.cpp
 * `--list-devices` (the order `--tensor-split` / `--main-gpu` use).
 */
export function detectGpus(force = false, llamaServerBinary?: string): GpuMemoryInfo[] {
  const now = Date.now();
  const binKey = (llamaServerBinary || "").trim();
  if (!force && cached && now - cached.at < 15_000 && cached.binKey === binKey) {
    return cached.gpus;
  }
  const nvidia = readNvidiaSmiAll();
  const sysfs = nvidia ? undefined : readSysfsGpus();
  const wmi = nvidia || sysfs ? undefined : readWindowsWmiGpus();
  const rocm = nvidia || sysfs || wmi ? undefined : readRocmSmi();
  let gpus = nvidia || sysfs || wmi || (rocm ? [rocm] : []);
  if (binKey && gpus.length >= 2) {
    const listed = readLlamaListDevices(binKey);
    if (listed?.length) {
      gpus = orderGpusLikeLlama(gpus, listed);
    }
  }
  cached = { at: now, binKey, gpus };
  return gpus;
}

/** `Vulkan0 · Radeon RX 9070 XT` when llama.cpp id is known; else `GPU 0 · …`. */
export function formatGpuDeviceLabel(
  gpu: { name?: string; llamaDeviceId?: string } | undefined,
  index: number
): string {
  const name = (gpu?.name || "").trim();
  const pretty = name && !/^(amdgpu|nvidia|i915|xe)$/i.test(name) ? name : "";
  const id = (gpu?.llamaDeviceId || "").trim() || `GPU ${index}`;
  return pretty ? `${id} · ${pretty}` : id;
}

/** Detect primary (largest) GPU VRAM. Cached with {@link detectGpus}. */
export function detectGpuMemory(force = false): GpuMemoryInfo | undefined {
  const gpus = detectGpus(force);
  if (!gpus.length) {
    return undefined;
  }
  return gpus.reduce((best, g) => (g.totalBytes > best.totalBytes ? g : best));
}

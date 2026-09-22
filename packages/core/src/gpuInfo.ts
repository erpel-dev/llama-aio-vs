import * as fs from "fs";
import { execFileSync, spawnSync } from "child_process";
import * as path from "path";
import { activeInstallLock } from "./installSwap";

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
  /**
   * GTT in use: system RAM currently mapped for this GPU (Linux amdgpu
   * `mem_info_gtt_used`). A card whose GTT dwarfs its VRAM use is holding the
   * model in host memory — RADV does that when the device-local BAR window is
   * too small (`mem_info_vis_vram_total`) or the allocation is host-visible.
   */
  gttUsedBytes?: number;
  /** GTT aperture size (`mem_info_gtt_total`). */
  gttTotalBytes?: number;
  /**
   * Visible (BAR-mapped) VRAM total — `mem_info_vis_vram_total`. A value far
   * below `totalBytes` means Resizable BAR / Above-4G is off, which pushes
   * large Vulkan allocations into GTT.
   */
  visVramTotalBytes?: number;
  /** Used part of the visible VRAM window (`mem_info_vis_vram_used`). */
  visVramUsedBytes?: number;
}

export interface LlamaListedDevice {
  id: string;
  index: number;
  name: string;
  totalBytes?: number;
  usedBytes?: number;
}

const MIN_VRAM = 512 * 1024 * 1024;
/**
 * `Win32_VideoController.AdapterRAM` is a uint32. Cards larger than 4 GiB
 * saturate here (often `0xFFF00000` or `0xFFFFFFFF`), so that reading is not
 * the device size.
 */
export const ADAPTER_RAM_UINT32_CAP = 0x1_0000_0000;
/** Lower edge of the saturation band. A genuine 2–3 GiB card stays below it. */
const ADAPTER_RAM_CAP_FLOOR = 0xe000_0000;
const MAX_PLAUSIBLE_VRAM = 512 * 1024 ** 3;

/** True when a WMI `AdapterRAM` value is the uint32 ceiling, not a measured size. */
export function adapterRamLooksCapped(totalBytes: number): boolean {
  return totalBytes >= ADAPTER_RAM_CAP_FLOOR && totalBytes <= 0xffff_ffff;
}

function isIgnoredDisplayName(name: string): boolean {
  return /microsoft basic|basic display|basic render|remote display|virtual display|indirect display|hyper-v|parsec|iddcx|idd sample|virtual monitor|meta virtual/i.test(
    name
  );
}

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
      // Host-memory + BAR-window counters. Without these the app cannot tell
      // "idle card" apart from "card quietly holding the model in system RAM".
      const readMemCounter = (file: string): number | undefined => {
        try {
          const p = path.join(deviceDir, file);
          if (!fs.existsSync(p)) {
            return undefined;
          }
          const v = Number(fs.readFileSync(p, "utf8").trim());
          return Number.isFinite(v) ? v : undefined;
        } catch {
          return undefined;
        }
      };
      const gttUsedBytes = readMemCounter("mem_info_gtt_used");
      const gttTotalBytes = readMemCounter("mem_info_gtt_total");
      const visVramTotalBytes = readMemCounter("mem_info_vis_vram_total");
      const visVramUsedBytes = readMemCounter("mem_info_vis_vram_used");
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
        gttUsedBytes,
        gttTotalBytes,
        visVramTotalBytes,
        visVramUsedBytes,
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
 * Windows WMI fallback. `AdapterRAM` saturates at 4 GiB; callers replace that
 * with `--list-devices` or `HardwareInformation.qwMemorySize` when they can.
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
    const found = parseWindowsWmiVideoControllers(out);
    return found.length ? sortAndIndex(found) : undefined;
  } catch {
    return undefined;
  }
}

/**
 * `reg query` text. Windows may emit UTF-16LE (with or without a BOM) when
 * the console code page is Unicode; treating that as UTF-8 drops every line.
 */
export function decodeRegistryQueryOutput(buf: Buffer): string {
  if (buf.length >= 2 && buf[0] === 0xff && buf[1] === 0xfe) {
    return buf.subarray(2).toString("utf16le");
  }
  if (buf.length >= 4 && buf[1] === 0 && buf[3] === 0) {
    return buf.toString("utf16le").replace(/^\uFEFF/, "");
  }
  return buf.toString("utf8");
}

function parseRegBinaryQword(hex: string): number | undefined {
  const clean = hex.replace(/[^0-9a-fA-F]/g, "");
  if (!clean || clean.length % 2 !== 0 || clean.length > 16) {
    return undefined;
  }
  const bytes: number[] = [];
  for (let i = 0; i < clean.length; i += 2) {
    bytes.push(Number.parseInt(clean.slice(i, i + 2), 16));
  }
  const fromEndian = (little: boolean): number => {
    let n = 0n;
    for (let i = 0; i < bytes.length; i++) {
      const shift = little ? i : bytes.length - 1 - i;
      n += BigInt(bytes[i]!) << BigInt(8 * shift);
    }
    return Number(n);
  };
  const plausible = (n: number) =>
    Number.isSafeInteger(n) && n >= MIN_VRAM && n <= MAX_PLAUSIBLE_VRAM;
  const le = fromEndian(true);
  if (plausible(le)) {
    return le;
  }
  const be = fromEndian(false);
  return plausible(be) ? be : undefined;
}

/** Parse one `HardwareInformation.qwMemorySize` data cell from `reg query`. */
export function parseQwMemorySizeField(type: string, data: string): number | undefined {
  const kind = type.toUpperCase();
  const raw = data.trim();
  if (kind === "REG_QWORD" || kind === "REG_DWORD") {
    const n = /^0x/i.test(raw) ? Number.parseInt(raw, 16) : Number(raw);
    if (!Number.isSafeInteger(n) || n <= 0) {
      return undefined;
    }
    return n;
  }
  if (kind === "REG_BINARY") {
    return parseRegBinaryQword(raw);
  }
  return undefined;
}

export interface WindowsRegistryGpu {
  name: string;
  totalBytes: number;
}

const DISPLAY_CLASS_GUID = "{4d36e968-e325-11ce-bfc1-08002be10318}";

/**
 * Parse `reg query HKLM\...\Class\{4d36e968-...} /s`.
 * Only `HardwareInformation.qwMemorySize` counts — `MemorySize` is the same
 * uint32 cap as WMI `AdapterRAM`.
 */
export function parseWindowsGpuRegistry(text: string): WindowsRegistryGpu[] {
  const blocks = new Map<string, { name?: string; totalBytes?: number }>();
  let current: string | undefined;
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (/^HKEY_/i.test(trimmed) && trimmed.toLowerCase().includes(DISPLAY_CLASS_GUID.toLowerCase())) {
      const key = /\\(\d{4})$/.exec(trimmed);
      current = key?.[1];
      if (current && !blocks.has(current)) {
        blocks.set(current, {});
      }
      if (!key) {
        current = undefined;
      }
      continue;
    }
    if (!current) {
      continue;
    }
    const field = /^\s*(DriverDesc|HardwareInformation\.qwMemorySize)\s+(REG_\w+)\s+(.+)$/i.exec(line);
    if (!field) {
      continue;
    }
    const rec = blocks.get(current);
    if (!rec) {
      continue;
    }
    if (/^DriverDesc$/i.test(field[1] || "")) {
      rec.name = (field[3] || "").trim();
    } else {
      const bytes = parseQwMemorySizeField(field[2] || "", field[3] || "");
      if (bytes && bytes >= MIN_VRAM && bytes <= MAX_PLAUSIBLE_VRAM) {
        rec.totalBytes = bytes;
      }
    }
  }
  const out: WindowsRegistryGpu[] = [];
  for (const rec of blocks.values()) {
    if (!rec.name || !rec.totalBytes || isIgnoredDisplayName(rec.name)) {
      continue;
    }
    out.push({ name: rec.name, totalBytes: rec.totalBytes });
  }
  return out;
}

function readWindowsRegistryGpus(): WindowsRegistryGpu[] | undefined {
  if (process.platform !== "win32") {
    return undefined;
  }
  try {
    const buf = execFileSync(
      "reg.exe",
      [
        "query",
        `HKLM\\SYSTEM\\CurrentControlSet\\Control\\Class\\${DISPLAY_CLASS_GUID}`,
        "/s",
      ],
      { timeout: 6000, windowsHide: true }
    );
    const found = parseWindowsGpuRegistry(decodeRegistryQueryOutput(buf));
    return found.length ? found : undefined;
  } catch {
    return undefined;
  }
}

/** CIM JSON from `Win32_VideoController`. `AdapterRAM` may arrive as a string. */
export function parseWindowsWmiVideoControllers(json: string): GpuMemoryInfo[] {
  const parsed = JSON.parse(json) as
    | { Name?: string; AdapterRAM?: number | string }
    | Array<{ Name?: string; AdapterRAM?: number | string }>;
  const items = Array.isArray(parsed) ? parsed : [parsed];
  const found: GpuMemoryInfo[] = [];
  for (const item of items) {
    const name = String(item.Name || "");
    if (!name || isIgnoredDisplayName(name)) {
      continue;
    }
    const totalBytes = Number(item.AdapterRAM);
    if (!Number.isFinite(totalBytes) || totalBytes < MIN_VRAM) {
      continue;
    }
    found.push({ totalBytes, name, source: "wmi" });
  }
  return found;
}

/** Detected GPUs, cached briefly because sysfs/nvidia-smi are stable. */
let cached: { at: number; binKey: string; gpus: GpuMemoryInfo[] } | undefined;
/** Last 2+ GPU list with llama.cpp ids — kept across a flaky reload probe. */
let lastGoodGpus: GpuMemoryInfo[] | undefined;

/**
 * Keep a previously good dual-GPU list when a re-probe drops a card or loses
 * `--list-devices` ids. Reloading while llama-server holds Vulkan often lists
 * one device at 0 MiB; treating that as capacity dumps the whole model on GPU 0.
 */
export function preferStableGpuList(
  next: GpuMemoryInfo[],
  previous: GpuMemoryInfo[] | undefined
): GpuMemoryInfo[] {
  if (!previous?.length) {
    return next;
  }
  if (previous.length >= 2 && next.length < previous.length) {
    return previous;
  }
  if (
    previous.length >= 2 &&
    previous.length === next.length &&
    previous.some((g) => g.llamaDeviceId) &&
    !next.some((g) => g.llamaDeviceId)
  ) {
    return previous;
  }
  // A later WMI probe can fall back to the 4 GiB cap after a good reading.
  if (
    previous.length === next.length &&
    next.some((g, i) => adapterRamLooksCapped(g.totalBytes) && (previous[i]?.totalBytes || 0) > ADAPTER_RAM_UINT32_CAP)
  ) {
    return previous;
  }
  return next;
}

export function resetGpuDetectionCache(): void {
  cached = undefined;
  lastGoodGpus = undefined;
}

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
      // Positional fallback is only safe for a card we cannot identify at all
      // (`amdgpu`, a blank lspci name). Stealing a card that clearly carries a
      // *different* model number would attach this `VulkanN` id — and with it
      // `--main-gpu` / `--tensor-split` and the VRAM total — to the wrong
      // physical device, which reads as "the small card is idle".
      const unidentifiable = unused.findIndex((g) => gpuModelTokens(g.name || "").length === 0);
      idx = unidentifiable;
    }
    if (idx < 0) {
      continue;
    }
    const g = unused.splice(idx, 1)[0];
    if (!g) {
      continue;
    }
    const sysfsTotal = g.totalBytes > 0 ? g.totalBytes : 0;
    const listedTotal = dev.totalBytes && dev.totalBytes > 0 ? dev.totalBytes : 0;
    // WMI AdapterRAM saturates at 4 GiB. A larger --list-devices size is real.
    // Linux sysfs totals are left alone (source is not "wmi").
    const upgradeCapped =
      g.source === "wmi" && listedTotal > sysfsTotal && adapterRamLooksCapped(sysfsTotal);
    ordered.push({
      ...g,
      name: g.name && !/^(amdgpu|nvidia|i915|xe)$/i.test(g.name.trim()) ? g.name : dev.name,
      // `||` would take a 0 MiB `--list-devices` reading over a good sysfs total.
      totalBytes: upgradeCapped ? listedTotal : sysfsTotal || listedTotal,
      usedBytes: dev.usedBytes ?? g.usedBytes,
      index: ordered.length,
      llamaDeviceId: dev.id,
      source: upgradeCapped ? "wmi+list-devices" : g.source,
    });
  }
  for (const g of unused) {
    ordered.push({ ...g, index: ordered.length });
  }
  return ordered;
}

/** Pair a WMI name with a registry `DriverDesc` or a `--list-devices` name. */
export function windowsGpuNamesMatch(a: string, b: string): boolean {
  const ta = gpuModelTokens(a);
  const tb = gpuModelTokens(b);
  if (ta.length && tb.length) {
    return ta.some((t) => tb.includes(t));
  }
  const norm = (s: string) =>
    s
      .toLowerCase()
      .replace(/\(r\)|\(tm\)/gi, "")
      .replace(/[^a-z0-9]+/g, " ")
      .trim();
  const na = norm(a);
  const nb = norm(b);
  if (!na || !nb) {
    return false;
  }
  return na === nb || na.includes(nb) || nb.includes(na);
}

/**
 * True when WMI is missing or still sitting on the 4 GiB cap, and
 * `--list-devices` did not already report a larger card.
 */
export function windowsVramNeedsRegistry(
  gpus: GpuMemoryInfo[],
  listed: LlamaListedDevice[] | undefined
): boolean {
  const capped = gpus.length === 0 || gpus.some((g) => adapterRamLooksCapped(g.totalBytes));
  if (!capped) {
    return false;
  }
  if (!listed?.length) {
    return true;
  }
  return !listed.some((d) => (d.totalBytes || 0) > ADAPTER_RAM_UINT32_CAP);
}

/**
 * Windows VRAM: `--list-devices` replaces a capped `AdapterRAM`, then
 * `HardwareInformation.qwMemorySize` if the device list is missing or 0 MiB.
 * An empty WMI list becomes the device list, or the registry entries.
 */
export function resolveWindowsGpuList(
  wmi: GpuMemoryInfo[],
  listed: LlamaListedDevice[] | undefined,
  registry: WindowsRegistryGpu[] | undefined
): GpuMemoryInfo[] {
  let gpus = wmi.filter((g) => !isIgnoredDisplayName(g.name || ""));
  if (listed?.length) {
    if (gpus.length) {
      gpus = orderGpusLikeLlama(gpus, listed);
    } else {
      gpus = [];
      for (const dev of listed) {
        if (!dev.totalBytes || dev.totalBytes < MIN_VRAM) {
          continue;
        }
        gpus.push({
          totalBytes: dev.totalBytes,
          usedBytes: dev.usedBytes,
          name: dev.name,
          source: "llama-list-devices",
          index: gpus.length,
          llamaDeviceId: dev.id,
        });
      }
    }
  }
  const reg = (registry || []).filter(
    (r) => r.name && r.totalBytes >= MIN_VRAM && !isIgnoredDisplayName(r.name)
  );
  if (!reg.length) {
    return gpus;
  }
  if (!gpus.length) {
    return sortAndIndex(
      reg.map((r) => ({ totalBytes: r.totalBytes, name: r.name, source: "registry" }))
    );
  }
  const used = new Set<number>();
  return gpus.map((g) => {
    if (!adapterRamLooksCapped(g.totalBytes)) {
      return g;
    }
    let matchAt = reg.findIndex((r, i) => !used.has(i) && windowsGpuNamesMatch(r.name, g.name || ""));
    if (matchAt < 0 && gpus.length === 1 && reg.length === 1 && !used.has(0)) {
      const only = reg[0];
      const gpuTokens = gpuModelTokens(g.name || "");
      const regTokens = gpuModelTokens(only?.name || "");
      // A single unnamed adapter can take the only registry qword.
      // A named model must not inherit a different card's size.
      if (only && gpuTokens.length === 0 && regTokens.length === 0) {
        matchAt = 0;
      }
    }
    if (matchAt < 0) {
      return g;
    }
    const match = reg[matchAt];
    if (!match || match.totalBytes <= g.totalBytes) {
      return g;
    }
    used.add(matchAt);
    return { ...g, totalBytes: match.totalBytes, source: "registry" };
  });
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
  const windowsVram =
    process.platform === "win32" && !nvidia && !sysfs;
  const windowsNeedsList =
    windowsVram &&
    ((wmi || []).length === 0 ||
      (wmi || []).length >= 2 ||
      (wmi || []).some((g) => adapterRamLooksCapped(g.totalBytes)));
  // `--list-devices` maps the vulkan\bin exe; skip while an install holds it.
  // One Windows GPU still needs it: WMI AdapterRAM cannot report past 4 GiB.
  let listed: LlamaListedDevice[] | undefined;
  if (binKey && !activeInstallLock() && (gpus.length >= 2 || windowsNeedsList)) {
    listed = readLlamaListDevices(binKey);
  }
  if (windowsVram) {
    const registry = windowsVramNeedsRegistry(wmi || [], listed)
      ? readWindowsRegistryGpus()
      : undefined;
    gpus = resolveWindowsGpuList(wmi || [], listed, registry);
  } else if (listed?.length && gpus.length >= 2) {
    gpus = orderGpusLikeLlama(gpus, listed);
  }
  gpus = preferStableGpuList(gpus, force ? undefined : lastGoodGpus);
  if (gpus.length >= 2) {
    lastGoodGpus = gpus;
  } else if (
    gpus.length === 1 &&
    (gpus[0]?.totalBytes || 0) > ADAPTER_RAM_UINT32_CAP &&
    (lastGoodGpus?.length || 0) <= 1
  ) {
    lastGoodGpus = gpus;
  }
  cached = { at: now, binKey, gpus };
  return gpus;
}

/** Dedicated VRAM this small is usually a UMA carve-out, not a discrete card. */
const INTEGRATED_VRAM_MAX_BYTES = 2 * 1024 ** 3;

/**
 * True for APUs / iGPUs (Vega/Rembrandt/Intel UHD, “Onboard IGD”, or a tiny
 * VRAM bar next to a large GTT aperture). Discrete cards — even with ReBAR
 * off and a big GTT figure — stay false.
 */
export function isIntegratedGpu(
  gpu: Pick<GpuMemoryInfo, "name" | "llamaDeviceId" | "totalBytes" | "gttTotalBytes"> | undefined
): boolean {
  if (!gpu || !(gpu.totalBytes > 0)) {
    return false;
  }
  const name = `${gpu.name || ""} ${gpu.llamaDeviceId || ""}`.toLowerCase();
  if (
    /onboard|\bigd\b|integrated|iris|uhd graphics|hd graphics|radeon graphics|vega mobile|cezanne|renoir|lucienne|barcelo|mendocino|rembrandt|raphael|phoenix|hawk.?point|strix|gfx[0-9]+c\b/.test(
      name
    )
  ) {
    return true;
  }
  const gtt = gpu.gttTotalBytes || 0;
  return gpu.totalBytes <= INTEGRATED_VRAM_MAX_BYTES && gtt >= 4 * 1024 ** 3 && gtt >= gpu.totalBytes * 4;
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

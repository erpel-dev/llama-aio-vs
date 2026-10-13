import { execFileSync } from "child_process";
import * as fs from "fs";
import * as path from "path";

/**
 * AMD ROCm library locations under `$ROCM_PATH` or `/opt/rocm`.
 * The Ubuntu llama.cpp archive links hipBLAS and does not ship it.
 * Newer TheRock packages keep libraries in `core-<version>/lib`.
 */
const HIPBLAS_NAMES = ["libhipblas.so", "libhipblas.so.2", "libhipblas.so.3"];

export interface RocmSupport {
  /**
   * Official llama.cpp archive exists for this OS/arch, an AMD GPU is present,
   * and on Linux libhipblas is visible.
   */
  available: boolean;
  reason?: string;
}

/** Injected inputs for {@link detectAmdGpu}. Omit every field for a live probe. */
export interface AmdGpuProbe {
  /** Directory shaped like `/sys/bus/pci/devices`. */
  pciRoot?: string;
  /** `lspci -nn` text. `null` skips that probe. */
  lspciText?: string | null;
  /** `rocminfo` text. `null` skips that probe. */
  rocminfoText?: string | null;
  /** `rocm-smi --showproductname` text. `null` skips that probe. */
  rocmSmiText?: string | null;
  /** Windows `Win32_VideoController` PNP ids. `null` skips WMI. */
  pnpText?: string | null;
  /** PATH used to find `lspci`, `rocminfo`, and `rocm-smi`. */
  env?: NodeJS.ProcessEnv;
}

/**
 * Same window as the CUDA and Vulkan probes. A GPU does not appear or
 * disappear while the extension host is running.
 */
const AMD_GPU_DETECT_TTL_MS = 5 * 60 * 1000;
const LINUX_PCI_ROOT = "/sys/bus/pci/devices";
/**
 * VGA (`0x0300`), 3D (`0x0302`), other display (`0x0380`), and processing
 * accelerators (`0x1200`, Instinct). The AMD HDMI audio function is `0x0403`
 * and shares vendor `1002`, so vendor alone is not enough.
 */
const AMD_GPU_CLASS_PREFIXES = ["0x0300", "0x0302", "0x0380", "0x1200"];

let amdGpuCache: { value: boolean; at: number } | undefined;

/** Drop the cached AMD-GPU verdict. Called with the other hardware probes. */
export function resetAmdGpuDetectionCache(): void {
  amdGpuCache = undefined;
}

export function amdGpuMissingReason(): string {
  return "No AMD GPU detected (sysfs / lspci / rocminfo / rocm-smi)";
}

/** True for PCI vendor 1002 (AMD/ATI), ignoring surrounding whitespace. */
export function pciVendorIsAmd(vendor: string): boolean {
  const v = vendor.trim().toLowerCase();
  return v === "0x1002" || v === "1002";
}

/** True for a display or compute class. False for the HDMI audio function. */
export function pciClassIsAmdGpu(deviceClass: string): boolean {
  const c = deviceClass.trim().toLowerCase();
  return AMD_GPU_CLASS_PREFIXES.some((prefix) => c.startsWith(prefix));
}

/** True when this PCI device is an AMD GPU rather than AMD HDMI audio. */
export function pciIdsLookLikeAmdGpu(vendor: string, deviceClass: string): boolean {
  return pciVendorIsAmd(vendor) && pciClassIsAmdGpu(deviceClass);
}

/**
 * True when a `/sys/bus/pci/devices` tree contains an AMD display or
 * compute device. A missing directory is "no GPU", not an error.
 */
export function sysfsPciHasAmdGpu(pciRoot: string): boolean {
  if (!pciRoot) {
    return false;
  }
  let entries: string[];
  try {
    entries = fs.readdirSync(pciRoot);
  } catch {
    return false;
  }
  for (const ent of entries) {
    const dir = path.join(pciRoot, ent);
    let vendor = "";
    try {
      vendor = fs.readFileSync(path.join(dir, "vendor"), "utf8");
    } catch {
      continue;
    }
    if (!pciVendorIsAmd(vendor)) {
      continue;
    }
    let deviceClass = "";
    try {
      deviceClass = fs.readFileSync(path.join(dir, "class"), "utf8");
    } catch {
      continue;
    }
    if (pciClassIsAmdGpu(deviceClass)) {
      return true;
    }
  }
  return false;
}

/** True when `lspci -nn` lists an AMD display or compute device. */
export function lspciTextHasAmdGpu(text: string): boolean {
  for (const line of text.split("\n")) {
    if (!/\[(0300|0302|0380|1200)\]/.test(line)) {
      continue;
    }
    if (/\[1002:[0-9a-f]{4}\]/i.test(line)) {
      return true;
    }
  }
  return false;
}

/**
 * True when rocminfo printed a GPU agent. A CPU-only agent still exits 0,
 * so the exit code is not the signal.
 */
export function rocminfoTextHasGpu(text: string): boolean {
  return /Device Type:\s+GPU\b/i.test(text);
}

/** True when `rocm-smi --showproductname` listed a card. */
export function rocmSmiTextHasGpu(text: string): boolean {
  if (/no amd gpus/i.test(text)) {
    return false;
  }
  return /GPU\[\d+\]/i.test(text);
}

/** True when a Windows video-controller PNP id is AMD (`VEN_1002`). */
export function pnpTextHasAmdGpu(text: string): boolean {
  return /VEN_1002/i.test(text);
}

function isLiveProbe(probe: AmdGpuProbe | undefined): boolean {
  if (!probe) {
    return true;
  }
  return (
    probe.pciRoot === undefined &&
    probe.lspciText === undefined &&
    probe.rocminfoText === undefined &&
    probe.rocmSmiText === undefined &&
    probe.pnpText === undefined &&
    probe.env === undefined
  );
}

function commandOnPath(command: string, env: NodeJS.ProcessEnv): string | undefined {
  const pathEnv = env.PATH || "";
  const sep = process.platform === "win32" ? ";" : ":";
  const exts =
    process.platform === "win32" ? (env.PATHEXT || ".EXE;.CMD;.BAT").split(";").filter(Boolean) : [""];
  for (const dir of pathEnv.split(sep)) {
    if (!dir) {
      continue;
    }
    for (const ext of exts) {
      const candidateName =
        process.platform === "win32" && !path.extname(command) ? command + ext : command;
      const full = path.join(dir, candidateName);
      try {
        if (fs.existsSync(full) && fs.statSync(full).isFile()) {
          return full;
        }
      } catch {
        // ignore
      }
    }
  }
  return undefined;
}

function commandOutput(file: string, args: string[], timeoutMs: number): string | undefined {
  try {
    return execFileSync(file, args, {
      encoding: "utf8",
      timeout: timeoutMs,
      windowsHide: true,
      stdio: ["ignore", "pipe", "ignore"],
    });
  } catch (err) {
    const stdout = (err as { stdout?: string | Buffer }).stdout;
    if (typeof stdout === "string" && stdout.trim()) {
      return stdout;
    }
    if (Buffer.isBuffer(stdout) && stdout.length) {
      return stdout.toString("utf8");
    }
    return undefined;
  }
}

function readInstalledTool(
  name: string,
  args: string[],
  env: NodeJS.ProcessEnv,
  timeoutMs: number
): string | undefined {
  const bin = commandOnPath(name, env);
  if (!bin) {
    return undefined;
  }
  return commandOutput(bin, args, timeoutMs);
}

function readWindowsVideoPnp(): string | undefined {
  if (process.platform !== "win32") {
    return undefined;
  }
  return commandOutput(
    "powershell.exe",
    [
      "-NoProfile",
      "-NonInteractive",
      "-Command",
      "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty PNPDeviceID",
    ],
    6000
  );
}

function textOrLive(
  injected: string | null | undefined,
  live: boolean,
  read: () => string | undefined
): string {
  if (injected !== undefined) {
    return injected ?? "";
  }
  if (!live) {
    return "";
  }
  return read() ?? "";
}

function detectAmdGpuUncached(probe: AmdGpuProbe, live: boolean): boolean {
  const env = probe.env ?? process.env;
  const pciRoot = probe.pciRoot ?? (live && process.platform === "linux" ? LINUX_PCI_ROOT : "");
  if (pciRoot && sysfsPciHasAmdGpu(pciRoot)) {
    return true;
  }

  const lspci = textOrLive(probe.lspciText, live, () => readInstalledTool("lspci", ["-nn"], env, 1500));
  if (lspci && lspciTextHasAmdGpu(lspci)) {
    return true;
  }

  const rocminfo = textOrLive(probe.rocminfoText, live, () => readInstalledTool("rocminfo", [], env, 4000));
  if (rocminfo && rocminfoTextHasGpu(rocminfo)) {
    return true;
  }

  const smi = textOrLive(probe.rocmSmiText, live, () =>
    readInstalledTool("rocm-smi", ["--showproductname"], env, 4000)
  );
  if (smi && rocmSmiTextHasGpu(smi)) {
    return true;
  }

  const pnp = textOrLive(probe.pnpText, live && process.platform === "win32", readWindowsVideoPnp);
  return !!pnp && pnpTextHasAmdGpu(pnp);
}

/**
 * True when this machine has an AMD GPU. Checks PCI sysfs (vendor `1002` and
 * a display or compute class), then `lspci -nn`, then `rocminfo` / `rocm-smi`.
 * Windows falls through to `Win32_VideoController` PNP ids (`VEN_1002`).
 * The live result is cached. Pass a probe to test without touching the machine.
 */
export function detectAmdGpu(probe?: AmdGpuProbe): boolean {
  if (!isLiveProbe(probe)) {
    return detectAmdGpuUncached(probe ?? {}, false);
  }
  const now = Date.now();
  if (amdGpuCache && now - amdGpuCache.at < AMD_GPU_DETECT_TTL_MS) {
    return amdGpuCache.value;
  }
  const value = detectAmdGpuUncached({}, true);
  amdGpuCache = { value, at: now };
  return value;
}

function dirHasHipblas(dir: string): boolean {
  return HIPBLAS_NAMES.some((name) => fs.existsSync(path.join(dir, name)));
}

/** ROCm roots. `ROCM_PATH` wins over `/opt/rocm`. */
export function rocmRoots(env: NodeJS.ProcessEnv = process.env): string[] {
  const roots: string[] = [];
  const fromEnv = (env.ROCM_PATH || "").trim();
  if (fromEnv) {
    roots.push(fromEnv);
  }
  if (process.platform !== "win32") {
    const def = "/opt/rocm";
    if (!roots.includes(def)) {
      roots.push(def);
    }
  }
  return roots;
}

/** Existing library directories under one ROCm root. Empty when the root is missing. */
export function discoverRocmLibraryDirs(root: string): string[] {
  if (!root || !fs.existsSync(root)) {
    return [];
  }
  const found: string[] = [];
  const push = (dir: string) => {
    if (!dir || found.includes(dir) || !fs.existsSync(dir)) {
      return;
    }
    found.push(dir);
  };
  push(path.join(root, "lib"));
  push(path.join(root, "lib64"));
  let entries: string[] = [];
  try {
    entries = fs.readdirSync(root);
  } catch {
    entries = [];
  }
  for (const name of entries) {
    if (name === "lib" || name === "lib64" || name.startsWith("core-")) {
      push(path.join(root, name, "lib"));
    }
  }
  return found;
}

export function rocmRuntimePresent(env: NodeJS.ProcessEnv = process.env): boolean {
  return rocmRoots(env).some((root) => discoverRocmLibraryDirs(root).some((dir) => dirHasHipblas(dir)));
}

export function rocmRuntimeMissingReason(): string {
  return (
    "The Linux ROCm build does not include the AMD runtime. Install a ROCm version that matches this nightly " +
    "(libhipblas.so under /opt/rocm or ROCM_PATH), add your user to the render and video groups, and log in again."
  );
}

export interface RocmSupportOptions {
  /** When set, skip the live GPU probe. Tests pass this. */
  amdGpu?: boolean;
}

/**
 * Whether this machine can download and run an official ROCm llama-server.
 * An AMD GPU is required. On Linux, a missing hipBLAS runtime is reported
 * only after that GPU is found.
 */
export function rocmSupport(
  env: NodeJS.ProcessEnv = process.env,
  options?: RocmSupportOptions
): RocmSupport {
  if (process.arch !== "x64" || (process.platform !== "win32" && process.platform !== "linux")) {
    return {
      available: false,
      reason: "No ROCm release packages for this OS/arch. Official builds are Windows x64 and Linux x64.",
    };
  }
  const amdGpu = options?.amdGpu ?? detectAmdGpu();
  if (!amdGpu) {
    return { available: false, reason: amdGpuMissingReason() };
  }
  if (process.platform === "linux" && !rocmRuntimePresent(env)) {
    return { available: false, reason: rocmRuntimeMissingReason() };
  }
  return { available: true };
}

/**
 * Env for a ROCm llama-server. Prepends hipBLAS library dirs and sets
 * `ROCM_PATH` when the user has not set it. Does not set
 * `HSA_OVERRIDE_GFX_VERSION`; that override is for cards the build does not list.
 */
export function withRocmRuntimeEnv(base: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...base };
  if (process.platform === "win32") {
    return env;
  }
  const dirs: string[] = [];
  let chosenRoot: string | undefined;
  for (const root of rocmRoots(env)) {
    const hits = discoverRocmLibraryDirs(root).filter((dir) => dirHasHipblas(dir));
    if (!hits.length) {
      continue;
    }
    chosenRoot = chosenRoot || root;
    for (const dir of hits) {
      if (!dirs.includes(dir)) {
        dirs.push(dir);
      }
    }
  }
  if (!dirs.length) {
    return env;
  }
  if (!env.ROCM_PATH && chosenRoot) {
    env.ROCM_PATH = chosenRoot;
  }
  const prefix = dirs.join(path.delimiter);
  env.LD_LIBRARY_PATH = env.LD_LIBRARY_PATH ? `${prefix}${path.delimiter}${env.LD_LIBRARY_PATH}` : prefix;
  return env;
}

export function formatRocmLaunchFailure(binary: string, probe: string): string {
  const lines = [`Cannot execute the ROCm llama-server at ${binary}.`, `Probe: ${probe}`];
  if (process.platform === "linux" || /libhipblas|libamdhip64|librocblas/i.test(probe)) {
    lines.push("", rocmRuntimeMissingReason());
  } else {
    lines.push("", "Install the AMD ROCm or HIP runtime that matches this build, then try again.");
  }
  return lines.join("\n");
}

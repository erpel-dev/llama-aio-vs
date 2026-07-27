import * as fs from "fs";
import * as https from "https";
import * as os from "os";
import * as path from "path";
import { execFile, execFileSync } from "child_process";
import { promisify } from "util";
import * as vscode from "vscode";
import { ensureDirs, getBackendInstallDir, getInstallDir, getLlamaServerBinary, InstallBackendId, withBinaryDirEnv } from "./paths";
import { SettingsStore } from "./settings";

const execFileAsync = promisify(execFile);

const UI_BACKENDS: UiBackend[] = ["vulkan", "cuda", "cpu"];

interface GithubAsset {
  name: string;
  browser_download_url: string;
  size: number;
}

interface GithubRelease {
  tag_name: string;
  assets: GithubAsset[];
}

interface BackendVersionInfo {
  tag?: string;
  asset?: string;
  configured?: string;
  resolved?: string;
}

function httpGetJson<T>(url: string, headers: Record<string, string> = {}): Promise<T> {
  return new Promise((resolve, reject) => {
    const req = https.get(
      url,
      {
        headers: {
          "User-Agent": "llama-aio-vs",
          Accept: "application/vnd.github+json",
          ...headers,
        },
      },
      (res) => {
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          httpGetJson<T>(res.headers.location, headers).then(resolve, reject);
          res.resume();
          return;
        }
        const chunks: Buffer[] = [];
        res.on("data", (c) => chunks.push(c));
        res.on("end", () => {
          const body = Buffer.concat(chunks).toString("utf8");
          if (!res.statusCode || res.statusCode >= 400) {
            reject(new Error(`HTTP ${res.statusCode}: ${body.slice(0, 200)}`));
            return;
          }
          try {
            resolve(JSON.parse(body) as T);
          } catch (e) {
            reject(e);
          }
        });
      }
    );
    req.on("error", reject);
  });
}

function downloadFile(url: string, dest: string, onProgress?: (pct: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const doGet = (u: string) => {
      https
        .get(u, { headers: { "User-Agent": "llama-aio-vs" } }, (res) => {
          if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            doGet(res.headers.location);
            res.resume();
            return;
          }
          if (!res.statusCode || res.statusCode >= 400) {
            reject(new Error(`Download failed: HTTP ${res.statusCode}`));
            res.resume();
            return;
          }
          const total = Number(res.headers["content-length"] || 0);
          let received = 0;
          ensureDirs(path.dirname(dest));
          const out = fs.createWriteStream(dest);
          res.on("data", (chunk: Buffer) => {
            received += chunk.length;
            if (total > 0 && onProgress) {
              onProgress(Math.floor((received / total) * 100));
            }
          });
          res.pipe(out);
          out.on("finish", () => out.close(() => resolve()));
          out.on("error", reject);
        })
        .on("error", reject);
    };
    doGet(url);
  });
}

export type LlamaBackend = "auto" | "vulkan" | "rocm" | "cuda" | "cpu" | "openvino" | "sycl";

/** Backends offered in the sidebar switcher. */
export type UiBackend = "vulkan" | "cuda" | "cpu";

export interface BackendOption {
  id: UiBackend;
  label: string;
  available: boolean;
  reason?: string;
  /** True when this backend already has a local install. */
  installed: boolean;
  /** Release tag from that backend's VERSION file, e.g. b10154 */
  installedTag?: string;
  /** True when this is the currently selected/active backend. */
  active: boolean;
}

export interface InstalledBuildInfo {
  /** Release tag from VERSION file, e.g. b10107 */
  tag?: string;
  /** Full `llama-server --version` first line when binary exists */
  binaryVersion?: string;
  /** Full --version stdout (trimmed) */
  binaryVersionDetail?: string;
  asset?: string;
  /** Configured backend setting */
  configuredBackend: LlamaBackend;
  /** Concrete UI backend currently used for the binary path */
  activeBackend: UiBackend;
  /** Backend inferred from installed asset name */
  resolvedBackend?: UiBackend | string;
}

function isArchive(name: string): boolean {
  return /\.(zip|tar\.gz|tgz)$/i.test(name);
}

export function inferBackendFromAsset(assetName: string | undefined): string | undefined {
  if (!assetName) {
    return undefined;
  }
  const n = assetName.toLowerCase();
  if (n.includes("vulkan")) {
    return "vulkan";
  }
  if (/cuda/.test(n) && !n.startsWith("cudart-")) {
    return "cuda";
  }
  if (n.includes("rocm") || n.includes("hip-radeon")) {
    return "rocm";
  }
  if (n.includes("openvino")) {
    return "openvino";
  }
  if (n.includes("sycl")) {
    return "sycl";
  }
  if (!/(vulkan|rocm|hip|cuda|openvino|sycl)/.test(n)) {
    return "cpu";
  }
  return undefined;
}

function commandExists(bin: string): boolean {
  try {
    if (process.platform === "win32") {
      execFileSync("where.exe", [bin], { stdio: "ignore" });
    } else {
      execFileSync("sh", ["-c", `command -v ${bin}`], { stdio: "ignore" });
    }
    return true;
  } catch {
    return false;
  }
}

/** True when an NVIDIA GPU / CUDA runtime looks present on this machine. */
export function detectCudaHardware(): boolean {
  if (process.env.CUDA_PATH || process.env.CUDA_HOME) {
    return true;
  }
  if (process.platform === "linux") {
    try {
      if (fs.existsSync("/dev/nvidia0")) {
        return true;
      }
    } catch {
      // ignore
    }
  }
  if (commandExists("nvidia-smi")) {
    try {
      execFileSync("nvidia-smi", ["-L"], { stdio: "ignore", timeout: 3000 });
      return true;
    } catch {
      return false;
    }
  }
  return false;
}

/** True when Vulkan looks usable (or we can at least install the Vulkan build). */
export function detectVulkanHardware(): boolean {
  if (process.platform === "darwin") {
    return false;
  }
  if (commandExists("vulkaninfo")) {
    return true;
  }
  if (process.platform === "linux") {
    try {
      if (fs.existsSync("/usr/share/vulkan/icd.d") || fs.existsSync("/etc/vulkan/icd.d")) {
        return true;
      }
    } catch {
      // ignore
    }
    // Soft default: official Vulkan linux builds are the usual GPU path.
    return true;
  }
  if (process.platform === "win32") {
    const sysRoot = process.env.SystemRoot || "C:\\Windows";
    return (
      fs.existsSync(path.join(sysRoot, "System32", "vulkan-1.dll")) ||
      fs.existsSync(path.join(sysRoot, "SysWOW64", "vulkan-1.dll"))
    );
  }
  return false;
}

function scoreAsset(name: string, platform: NodeJS.Platform, arch: string, backend: LlamaBackend): number {
  const n = name.toLowerCase();
  if (!isArchive(n) || n.includes("cudart-") || n.includes("-ui.") || n.includes("xcframework")) {
    return -1;
  }

  // Platform / arch gate
  if (platform === "darwin") {
    if (arch === "arm64" && !/macos.*arm64|darwin.*arm64/.test(n)) {
      return -1;
    }
    if (arch !== "arm64" && !/macos.*x64|darwin.*x64/.test(n)) {
      return -1;
    }
  } else if (platform === "win32") {
    if (!n.includes("win")) {
      return -1;
    }
    if (arch === "arm64" ? !n.includes("arm64") : n.includes("arm64")) {
      return -1;
    }
  } else {
    // linux
    if (!/(ubuntu|linux)/.test(n)) {
      return -1;
    }
    if (arch === "arm64") {
      if (!/(arm64|aarch64)/.test(n)) {
        return -1;
      }
    } else if (!/x64|x86_64/.test(n) || /(arm64|aarch64|s390x)/.test(n)) {
      return -1;
    }
  }

  const wanted =
    backend === "auto"
      ? platform === "win32"
        ? ["cuda", "vulkan", "cpu"]
        : platform === "darwin"
          ? ["cpu"]
          : ["vulkan", "rocm", "cpu"]
      : [backend];

  for (let i = 0; i < wanted.length; i++) {
    const b = wanted[i];
    let match = false;
    if (b === "vulkan") {
      match = n.includes("vulkan");
    } else if (b === "rocm") {
      match = n.includes("rocm") || n.includes("hip-radeon");
    } else if (b === "cuda") {
      match = /cuda-\d+/.test(n) && !n.includes("cudart");
    } else if (b === "openvino") {
      match = n.includes("openvino");
    } else if (b === "sycl") {
      match = n.includes("sycl");
    } else if (b === "cpu") {
      // Plain CPU package: ubuntu-x64 / win-cpu-x64 / macos-*, not a GPU backend build.
      match =
        !/(vulkan|rocm|hip|cuda|openvino|sycl)/.test(n) &&
        (n.includes("-cpu-") ||
          /bin-ubuntu-x64\.|bin-ubuntu-arm64\.|bin-macos-|bin-linux-/.test(n));
    }
    if (match) {
      // Prefer higher in wanted list; slight boost for exact plain names.
      let score = 1000 - i * 100 + (n.includes("vulkan") && b === "vulkan" ? 10 : 0);
      // Prefer CUDA 12.4 runtime packages (widely compatible) over bleeding 13.x.
      if (b === "cuda") {
        if (n.includes("cuda-12.4")) {
          score += 40;
        } else if (n.includes("cuda-12")) {
          score += 25;
        } else if (n.includes("cuda-11")) {
          score += 10;
        }
      }
      return score;
    }
  }
  return -1;
}

export function pickAsset(
  assets: GithubAsset[],
  backend: LlamaBackend = "auto"
): GithubAsset | undefined {
  const platform = process.platform;
  const arch = process.arch;
  let best: GithubAsset | undefined;
  let bestScore = -1;
  for (const asset of assets) {
    const score = scoreAsset(asset.name, platform, arch, backend);
    if (score > bestScore) {
      best = asset;
      bestScore = score;
    }
  }
  return best;
}

/** Matching cudart zip for a Windows CUDA binary asset (required DLLs). */
export function pickCudartAsset(
  assets: GithubAsset[],
  mainAssetName: string
): GithubAsset | undefined {
  const m = /cuda-(\d+(?:\.\d+)?)/i.exec(mainAssetName);
  const ver = m?.[1];
  let best: GithubAsset | undefined;
  let bestScore = -1;
  for (const asset of assets) {
    const n = asset.name.toLowerCase();
    if (!isArchive(n) || !n.includes("cudart") || !n.includes("win")) {
      continue;
    }
    if (archMismatchWin(n)) {
      continue;
    }
    let score = 1;
    if (ver && n.includes(`cuda-${ver}`)) {
      score += 100;
    } else if (ver && n.includes(`cuda-${ver.split(".")[0]}`)) {
      score += 40;
    }
    if (score > bestScore) {
      best = asset;
      bestScore = score;
    }
  }
  return best;
}

function archMismatchWin(name: string): boolean {
  if (process.arch === "arm64") {
    return !name.includes("arm64");
  }
  return name.includes("arm64");
}

function copyFilesIntoBin(srcRoot: string, binDir: string, extensions?: string[]): void {
  const stack = [srcRoot];
  while (stack.length) {
    const dir = stack.pop()!;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) {
        stack.push(full);
        continue;
      }
      if (!e.isFile()) {
        continue;
      }
      if (extensions && !extensions.some((ext) => e.name.toLowerCase().endsWith(ext))) {
        continue;
      }
      const dst = path.join(binDir, e.name);
      try {
        fs.copyFileSync(full, dst);
      } catch {
        // ignore collisions / locks
      }
    }
  }
}

async function extractArchive(archivePath: string, destDir: string): Promise<void> {
  ensureDirs(destDir);
  if (archivePath.endsWith(".zip")) {
    if (process.platform === "win32") {
      await execFileAsync("powershell.exe", [
        "-NoProfile",
        "-Command",
        `Expand-Archive -Path '${archivePath.replace(/'/g, "''")}' -DestinationPath '${destDir.replace(/'/g, "''")}' -Force`,
      ]);
    } else {
      await execFileAsync("unzip", ["-o", archivePath, "-d", destDir]);
    }
  } else {
    await execFileAsync("tar", ["-xzf", archivePath, "-C", destDir]);
  }
}

function findBinaryAfterExtract(root: string): string | undefined {
  const target = process.platform === "win32" ? "llama-server.exe" : "llama-server";
  const stack = [root];
  while (stack.length) {
    const dir = stack.pop()!;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) {
        stack.push(full);
      } else if (e.name === target) {
        return full;
      }
    }
  }
  return undefined;
}

export class LlamaInstaller {
  constructor(private readonly store: SettingsStore) {}

  getBackend(): LlamaBackend {
    const value = (this.store.getConfig().get<string>("backend") || "auto").toLowerCase();
    const allowed: LlamaBackend[] = ["auto", "vulkan", "rocm", "cuda", "cpu", "openvino", "sycl"];
    return (allowed.includes(value as LlamaBackend) ? value : "auto") as LlamaBackend;
  }

  async setBackend(backend: LlamaBackend): Promise<void> {
    await this.store.getConfig().update("backend", backend, vscode.ConfigurationTarget.Global);
  }

  /** Map config / auto to a concrete UI backend folder name. */
  resolveActiveUiBackend(configured = this.getBackend()): UiBackend {
    this.migrateLegacyInstallIfNeeded();
    if (configured === "vulkan" || configured === "cuda" || configured === "cpu") {
      return configured;
    }
    // auto / other: prefer an already-installed backend, else hardware preference.
    const prefer: UiBackend[] =
      process.platform === "darwin"
        ? ["cpu"]
        : process.platform === "win32"
          ? detectCudaHardware()
            ? ["cuda", "vulkan", "cpu"]
            : ["vulkan", "cuda", "cpu"]
          : detectCudaHardware()
            ? ["cuda", "vulkan", "cpu"]
            : ["vulkan", "cpu", "cuda"];
    for (const id of prefer) {
      if (this.hasBackendInstalled(id)) {
        return id;
      }
    }
    for (const id of prefer) {
      if (id === "cuda" && detectCudaHardware()) {
        return "cuda";
      }
      if (id === "vulkan" && detectVulkanHardware()) {
        return "vulkan";
      }
      if (id === "cpu") {
        return "cpu";
      }
    }
    return "cpu";
  }

  getActiveInstallDir(backend?: UiBackend): string {
    const active = backend || this.resolveActiveUiBackend();
    return getBackendInstallDir(this.store.getConfig(), active as InstallBackendId);
  }

  /** Read VERSION metadata for a backend (or active). */
  readBackendVersion(backend?: UiBackend): BackendVersionInfo {
    const dir = this.getActiveInstallDir(backend);
    return readVersionFile(path.join(dir, "VERSION"));
  }

  hasBackendInstalled(backend: UiBackend): boolean {
    this.migrateLegacyInstallIfNeeded();
    const dir = getBackendInstallDir(this.store.getConfig(), backend);
    const binary = getLlamaServerBinary(dir);
    return fs.existsSync(binary);
  }

  /**
   * Move pre-per-backend layout (`llama.cpp/bin` + `VERSION`) into
   * `llama.cpp/<backend>/` once.
   */
  migrateLegacyInstallIfNeeded(): void {
    const root = getInstallDir(this.store.getConfig());
    const legacyBin = path.join(root, "bin");
    const legacyVersion = path.join(root, "VERSION");
    if (!fs.existsSync(legacyBin) && !fs.existsSync(legacyVersion)) {
      return;
    }
    // Already migrated if any backend dir has a binary.
    if (UI_BACKENDS.some((b) => fs.existsSync(getLlamaServerBinary(getBackendInstallDir(this.store.getConfig(), b))))) {
      // Clean leftover root bin/VERSION if backend copies exist.
      try {
        if (fs.existsSync(legacyBin)) {
          fs.rmSync(legacyBin, { recursive: true, force: true });
        }
        if (fs.existsSync(legacyVersion)) {
          fs.unlinkSync(legacyVersion);
        }
      } catch {
        // ignore
      }
      return;
    }

    const info = readVersionFile(legacyVersion);
    let backend: UiBackend = "vulkan";
    const resolved = (info.resolved || inferBackendFromAsset(info.asset) || "").toLowerCase();
    if (resolved === "cpu" || resolved === "cuda" || resolved === "vulkan") {
      backend = resolved;
    } else if (this.getBackend() === "cpu" || this.getBackend() === "cuda" || this.getBackend() === "vulkan") {
      backend = this.getBackend() as UiBackend;
    }

    const dest = getBackendInstallDir(this.store.getConfig(), backend);
    ensureDirs(dest);
    try {
      if (fs.existsSync(legacyBin) && !fs.existsSync(path.join(dest, "bin"))) {
        fs.renameSync(legacyBin, path.join(dest, "bin"));
      } else if (fs.existsSync(legacyBin)) {
        fs.rmSync(legacyBin, { recursive: true, force: true });
      }
      if (fs.existsSync(legacyVersion) && !fs.existsSync(path.join(dest, "VERSION"))) {
        fs.renameSync(legacyVersion, path.join(dest, "VERSION"));
      } else if (fs.existsSync(legacyVersion)) {
        fs.unlinkSync(legacyVersion);
      }
    } catch {
      // Best-effort migration; install can still re-download.
    }
  }

  /** Sidebar options: Vulkan / CUDA (if HW) / CPU — with install status. */
  getUiBackendOptions(): BackendOption[] {
    this.migrateLegacyInstallIfNeeded();
    const archLabel = process.arch === "arm64" ? "arm64" : "x64";
    const cudaHw = detectCudaHardware();
    const vulkanOk = detectVulkanHardware();
    const cudaAssetLikely =
      (process.platform === "win32" || process.platform === "linux") && process.arch !== "arm64";
    const active = this.resolveActiveUiBackend();

    const base: Array<Omit<BackendOption, "installed" | "installedTag" | "active">> = [
      {
        id: "vulkan",
        label: "Vulkan",
        available: vulkanOk,
        reason: vulkanOk ? undefined : "Not available on this platform",
      },
      {
        id: "cuda",
        label: "CUDA",
        available: cudaHw && cudaAssetLikely,
        reason: !cudaAssetLikely
          ? "No CUDA release packages for this OS/arch"
          : cudaHw
            ? undefined
            : "No NVIDIA GPU detected (nvidia-smi / CUDA)",
      },
      {
        id: "cpu",
        label: `CPU (${archLabel})`,
        available: true,
      },
    ];

    return base.map((opt) => {
      const ver = this.readBackendVersion(opt.id);
      const installed = this.hasBackendInstalled(opt.id);
      return {
        ...opt,
        installed,
        installedTag: ver.tag,
        active: opt.id === active,
      };
    });
  }

  getInstalledInfo(): InstalledBuildInfo {
    this.migrateLegacyInstallIfNeeded();
    const configuredBackend = this.getBackend();
    const activeBackend = this.resolveActiveUiBackend(configuredBackend);
    const installDir = this.getActiveInstallDir(activeBackend);
    const version = readVersionFile(path.join(installDir, "VERSION"));

    let binaryVersion: string | undefined;
    let binaryVersionDetail: string | undefined;
    const binary = getLlamaServerBinary(installDir);
    if (fs.existsSync(binary)) {
      try {
        const out = execFileSync(binary, ["--version"], {
          encoding: "utf8",
          timeout: 5000,
          windowsHide: true,
          cwd: path.dirname(binary),
          env: withBinaryDirEnv(binary),
        })
          .toString()
          .trim();
        binaryVersionDetail = out;
        binaryVersion = out.split("\n")[0]?.trim() || out;
      } catch {
        // binary may fail to start without GPU libs
      }
    }

    return {
      tag: version.tag,
      binaryVersion,
      binaryVersionDetail,
      asset: version.asset,
      configuredBackend,
      activeBackend,
      resolvedBackend: version.resolved || inferBackendFromAsset(version.asset) || activeBackend,
    };
  }

  async installOrUpgrade(
    progress?: vscode.Progress<{ message?: string; increment?: number }>,
    backendOverride?: LlamaBackend
  ): Promise<string> {
    this.migrateLegacyInstallIfNeeded();
    const backendSetting = backendOverride || this.getBackend();
    const uiBackend =
      backendSetting === "vulkan" || backendSetting === "cuda" || backendSetting === "cpu"
        ? backendSetting
        : this.resolveActiveUiBackend(backendSetting);

    progress?.report({ message: `Fetching latest llama.cpp release (backend: ${uiBackend})…` });
    const release = await httpGetJson<GithubRelease>(
      "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    );

    const asset = pickAsset(release.assets || [], uiBackend);
    if (!asset) {
      throw new Error(
        `No suitable llama.cpp release asset found for ${process.platform}/${process.arch} backend=${uiBackend}. Tag: ${release.tag_name}`
      );
    }

    const installDir = getBackendInstallDir(this.store.getConfig(), uiBackend);
    ensureDirs(installDir);
    const tmpDir = path.join(os.tmpdir(), "llama-aio-vs");
    ensureDirs(tmpDir);
    const archivePath = path.join(tmpDir, asset.name);

    progress?.report({ message: `Downloading ${asset.name}…` });
    await downloadFile(asset.browser_download_url, archivePath, (pct) => {
      progress?.report({ message: `Downloading ${asset.name}… ${pct}%` });
    });

    progress?.report({ message: "Extracting…" });
    const extractTo = path.join(tmpDir, `extract-${Date.now()}`);
    ensureDirs(extractTo);
    await extractArchive(archivePath, extractTo);

    const found = findBinaryAfterExtract(extractTo);
    if (!found) {
      throw new Error("Extracted archive but could not find llama-server binary.");
    }

    // Replace only this backend's bin dir — leave other backends intact.
    const binDir = path.join(installDir, "bin");
    fs.rmSync(binDir, { recursive: true, force: true });
    ensureDirs(binDir);
    const destBin = path.join(binDir, path.basename(found));
    fs.copyFileSync(found, destBin);
    if (process.platform !== "win32") {
      fs.chmodSync(destBin, 0o755);
    }

    const srcDir = path.dirname(found);
    for (const name of fs.readdirSync(srcDir)) {
      const src = path.join(srcDir, name);
      const dst = path.join(binDir, name);
      if (fs.statSync(src).isFile() && src !== found) {
        try {
          fs.copyFileSync(src, dst);
          if (process.platform !== "win32" && !name.includes(".")) {
            fs.chmodSync(dst, 0o755);
          }
        } catch {
          // ignore
        }
      }
    }
    // Also pull DLLs / shared libs that may live in nested folders of the archive.
    if (process.platform === "win32") {
      copyFilesIntoBin(extractTo, binDir, [".dll"]);
    }

    // Windows CUDA builds need the matching cudart zip (cudart64_*.dll, cublas, …).
    if (uiBackend === "cuda" && process.platform === "win32") {
      const cudart = pickCudartAsset(release.assets || [], asset.name);
      if (cudart) {
        progress?.report({ message: `Downloading CUDA runtime ${cudart.name}…` });
        const cudartArchive = path.join(tmpDir, cudart.name);
        await downloadFile(cudart.browser_download_url, cudartArchive, (pct) => {
          progress?.report({ message: `Downloading ${cudart.name}… ${pct}%` });
        });
        const cudartExtract = path.join(tmpDir, `cudart-extract-${Date.now()}`);
        ensureDirs(cudartExtract);
        await extractArchive(cudartArchive, cudartExtract);
        copyFilesIntoBin(cudartExtract, binDir, [".dll"]);
        try {
          fs.rmSync(cudartExtract, { recursive: true, force: true });
          fs.unlinkSync(cudartArchive);
        } catch {
          // ignore cleanup
        }
      }
    }

    const resolved = inferBackendFromAsset(asset.name) || uiBackend;
    fs.writeFileSync(
      path.join(installDir, "VERSION"),
      `${release.tag_name}\nasset=${asset.name}\nbackend=${uiBackend}\nresolved=${resolved}\n`,
      "utf8"
    );

    progress?.report({ message: `Installed ${release.tag_name} (${asset.name}) → ${uiBackend}` });
    return getLlamaServerBinary(installDir);
  }

  getInstalledVersion(): string | undefined {
    const info = this.getInstalledInfo();
    return info.tag || info.binaryVersion;
  }

  getInstalledAsset(): string | undefined {
    return this.getInstalledInfo().asset;
  }
}

function readVersionFile(versionFile: string): BackendVersionInfo {
  try {
    const lines = fs.readFileSync(versionFile, "utf8").split("\n");
    return {
      tag: lines[0]?.trim() || undefined,
      asset: lines.find((l) => l.startsWith("asset="))?.slice("asset=".length).trim(),
      configured: lines.find((l) => l.startsWith("backend="))?.slice("backend=".length).trim(),
      resolved: lines.find((l) => l.startsWith("resolved="))?.slice("resolved=".length).trim(),
    };
  } catch {
    return {};
  }
}

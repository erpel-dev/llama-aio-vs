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

export interface GithubAsset {
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
            reject(new Error(`Download failed: HTTP ${res.statusCode} for ${u}`));
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

/** True when a URL responds with 2xx (follows redirects). Uses a Range GET to avoid full download. */
/**
 * Probe result for a release asset. `status` distinguishes "GitHub says this
 * asset does not exist" from "we could not reach GitHub" — collapsing both into
 * false made a missing upload and a dead network produce the same error.
 */
interface UrlProbe {
  exists: boolean;
  status?: number;
  error?: string;
}

function probeUrl(url: string): Promise<UrlProbe> {
  return new Promise((resolve) => {
    const doGet = (u: string) => {
      const req = https.get(
        u,
        { headers: { "User-Agent": "llama-aio-vs", Range: "bytes=0-0" } },
        (res) => {
          if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            res.resume();
            doGet(res.headers.location);
            return;
          }
          const status = res.statusCode || 0;
          res.resume();
          resolve({ exists: status >= 200 && status < 400, status });
        }
      );
      req.on("error", (err) => resolve({ exists: false, error: err.message }));
      req.setTimeout(8000, () => {
        req.destroy();
        resolve({ exists: false, error: "timed out after 8s" });
      });
    };
    doGet(url);
  });
}

async function urlExists(url: string): Promise<boolean> {
  return (await probeUrl(url)).exists;
}

export const LLAMA_CPP_RELEASES_URL = "https://github.com/ggml-org/llama.cpp/releases";
export const LLAMA_CPP_DOWNLOAD_BASE =
  "https://github.com/ggml-org/llama.cpp/releases/download";

/** Normalize user input (`b10154`, `10154`, release URL) to a tag like `b10154`. */
export function normalizeReleaseTag(input: string): string {
  let s = (input || "").trim();
  if (!s) {
    throw new Error("Release tag is empty.");
  }
  const fromUrl =
    /\/releases\/tag\/(b?\d+)/i.exec(s) ||
    /#release-(b?\d+)/i.exec(s) ||
    /\/download\/(b\d+)\//i.exec(s);
  if (fromUrl) {
    s = fromUrl[1];
  } else {
    const bare = /\b(b?\d{3,})\b/i.exec(s);
    if (bare) {
      s = bare[1];
    }
  }
  s = s.replace(/^v/i, "");
  if (/^\d+$/.test(s)) {
    s = `b${s}`;
  }
  if (!/^b\d+$/i.test(s)) {
    throw new Error(
      `Invalid release tag "${input}". Use e.g. b10154, or a URL like ${LLAMA_CPP_RELEASES_URL}/tag/b10154`
    );
  }
  return `b${s.slice(1)}`;
}

/** Direct-download asset name candidates for a tag + backend (no GitHub API). */
export function candidateAssetNames(tag: string, backend: UiBackend): string[] {
  const t = normalizeReleaseTag(tag);
  const arch = process.arch === "arm64" ? "arm64" : "x64";
  if (process.platform === "darwin") {
    return [`llama-${t}-bin-macos-${arch}.tar.gz`];
  }
  if (process.platform === "win32") {
    if (backend === "vulkan") {
      return [`llama-${t}-bin-win-vulkan-${arch}.zip`];
    }
    if (backend === "cuda") {
      return [
        `llama-${t}-bin-win-cuda-12.4-${arch}.zip`,
        `llama-${t}-bin-win-cuda-12.5-${arch}.zip`,
        `llama-${t}-bin-win-cuda-13.3-${arch}.zip`,
        `llama-${t}-bin-win-cuda-13.0-${arch}.zip`,
      ];
    }
    return [`llama-${t}-bin-win-cpu-${arch}.zip`];
  }
  // linux
  if (backend === "vulkan") {
    return [`llama-${t}-bin-ubuntu-vulkan-${arch}.tar.gz`];
  }
  if (backend === "cuda") {
    // Official Ubuntu CUDA archives are rare; try common patterns then fail with a clear message.
    return [
      `llama-${t}-bin-ubuntu-cuda-12.4-${arch}.tar.gz`,
      `llama-${t}-bin-ubuntu-${arch}-cuda-12.4.tar.gz`,
    ];
  }
  return [`llama-${t}-bin-ubuntu-${arch}.tar.gz`];
}

export function candidateCudartNames(mainAssetName?: string): string[] {
  const m = mainAssetName ? /cuda-(\d+(?:\.\d+)?)/i.exec(mainAssetName) : undefined;
  const ver = m?.[1];
  const preferred = ver ? [`cudart-llama-bin-win-cuda-${ver}-x64.zip`] : [];
  return [
    ...preferred,
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.3-x64.zip",
    "cudart-llama-bin-win-cuda-12.5-x64.zip",
  ].filter((n, i, arr) => arr.indexOf(n) === i);
}

export function directAssetUrl(tag: string, assetName: string): string {
  return `${LLAMA_CPP_DOWNLOAD_BASE}/${normalizeReleaseTag(tag)}/${assetName}`;
}

/**
 * Resolve the newest llama.cpp release tag without using api.github.com.
 * Follows the HTML redirect: `/releases/latest` → `/releases/tag/b#####`.
 */
export function resolveLatestReleaseTag(): Promise<string> {
  const start = `${LLAMA_CPP_RELEASES_URL}/latest`;
  return new Promise((resolve, reject) => {
    const visit = (url: string, hops: number) => {
      if (hops > 8) {
        reject(new Error("Too many redirects while resolving latest llama.cpp release."));
        return;
      }
      const req = https.get(
        url,
        {
          headers: { "User-Agent": "llama-aio-vs", Accept: "text/html" },
        },
        (res) => {
          const loc = res.headers.location;
          if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && loc) {
            res.resume();
            const next = loc.startsWith("http") ? loc : new URL(loc, url).toString();
            visit(next, hops + 1);
            return;
          }

          // Final URL may already be .../releases/tag/b##### even on 200.
          const finalUrl = url;
          res.resume();
          try {
            resolve(normalizeReleaseTag(finalUrl));
          } catch (e) {
            // Fallback: some responses land on /releases with tag only in body — rare.
            reject(
              e instanceof Error
                ? e
                : new Error(`Could not parse latest release tag from ${finalUrl}`)
            );
          }
        }
      );
      req.on("error", reject);
      req.setTimeout(12_000, () => {
        req.destroy();
        reject(new Error("Timed out resolving latest llama.cpp release tag."));
      });
    };
    visit(start, 0);
  });
}

const LATEST_TAG_CACHE_MS = 45 * 60 * 1000;
let latestTagCache: { tag: string; fetchedAt: number } | undefined;

/** Compare release tags (`b10173`). Positive if `a` is newer than `b`. */
export function compareReleaseTags(a: string, b: string): number {
  const na = Number.parseInt(normalizeReleaseTag(a).replace(/^b/i, ""), 10);
  const nb = Number.parseInt(normalizeReleaseTag(b).replace(/^b/i, ""), 10);
  if (Number.isFinite(na) && Number.isFinite(nb)) {
    return na - nb;
  }
  return normalizeReleaseTag(a).localeCompare(normalizeReleaseTag(b), undefined, {
    numeric: true,
    sensitivity: "base",
  });
}

export function peekCachedLatestReleaseTag(): string | undefined {
  if (!latestTagCache) {
    return undefined;
  }
  if (Date.now() - latestTagCache.fetchedAt > LATEST_TAG_CACHE_MS) {
    return undefined;
  }
  return latestTagCache.tag;
}

export function invalidateLatestReleaseTagCache(): void {
  latestTagCache = undefined;
}

/** Cached latest-tag lookup (redirect only; no GitHub API). */
export async function getLatestReleaseTagCached(force = false): Promise<string> {
  if (!force) {
    const cached = peekCachedLatestReleaseTag();
    if (cached) {
      return cached;
    }
  }
  const tag = await resolveLatestReleaseTag();
  latestTagCache = { tag, fetchedAt: Date.now() };
  return tag;
}

export interface UpdateCheckInfo {
  latestTag?: string;
  installedTag?: string;
  /** True when latest is newer than the active backend install. */
  updateAvailable: boolean;
  /** Network/parse failed and no usable cache. */
  checkFailed: boolean;
  /** Still waiting on first network check. */
  pending: boolean;
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

/**
 * Explain an install failure in terms of what the release actually contains.
 * `listed` is undefined when the API could not be reached at all.
 */
export function describeMissingAsset(
  tag: string,
  backend: UiBackend,
  probes: Array<{ name: string; probe: { status?: number; error?: string } }>,
  listed: GithubAsset[] | undefined
): string {
  const reachedGithub = probes.some((p) => typeof p.probe.status === "number");
  const lines: string[] = [];

  if (!reachedGithub && listed === undefined) {
    lines.push(
      `Could not reach GitHub while looking for ${tag} (${backend}, ${process.platform}/${process.arch}).`,
      `Check your network or proxy, then try again.`
    );
    const firstError = probes.find((p) => p.probe.error)?.probe.error;
    if (firstError) {
      lines.push(`Last error: ${firstError}`);
    }
    return lines.join("\n");
  }

  lines.push(
    `No ${backend} archive for ${process.platform}/${process.arch} in llama.cpp release ${tag}.`
  );

  if (listed && listed.length === 0) {
    lines.push(
      "",
      "GitHub reports this release has no downloadable assets at all — the build is",
      "probably still uploading, or was published incomplete. Pick an earlier tag."
    );
  } else if (listed) {
    const names = listed.map((a) => a.name);
    lines.push("", `The release does contain ${names.length} asset(s):`);
    lines.push(...names.slice(0, 12).map((n) => `• ${n}`));
    if (names.length > 12) {
      lines.push(`…and ${names.length - 12} more.`);
    }
    lines.push("", "None of them match this platform and backend.");
  } else {
    lines.push("", "Tried these names:", ...probes.map((p) => `• ${p.name} → ${describeProbe(p.probe)}`));
  }

  lines.push(
    "",
    `Open ${LLAMA_CPP_RELEASES_URL}/tag/${tag} and use “Install from archive…”, or pick another tag/backend.`
  );
  return lines.join("\n");
}

function describeProbe(probe: { status?: number; error?: string }): string {
  if (probe.error) {
    return probe.error;
  }
  if (probe.status === 404) {
    return "404 (not uploaded)";
  }
  if (probe.status === 403) {
    return "403 (rate limited or blocked)";
  }
  return `HTTP ${probe.status ?? "?"}`;
}

/**
 * Move a fully staged `bin.new-*` tree into place.
 * The old tree is kept until the new one is in position, and restored if the
 * final rename fails (Windows can refuse while a DLL is still mapped).
 */
function swapInstallDir(stageDir: string, binDir: string): void {
  const backupDir = `${binDir}.old-${Date.now()}`;
  const hadPrevious = fs.existsSync(binDir);
  if (hadPrevious) {
    fs.renameSync(binDir, backupDir);
  }
  try {
    fs.renameSync(stageDir, binDir);
  } catch (err) {
    if (hadPrevious) {
      fs.renameSync(backupDir, binDir);
    }
    fs.rmSync(stageDir, { recursive: true, force: true });
    throw err;
  }
  if (hadPrevious) {
    fs.rmSync(backupDir, { recursive: true, force: true });
  }
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

  /**
   * Compare active backend install vs latest release tag (cached ~45 min).
   * Pass force=true to refresh the network check.
   */
  async getUpdateCheck(force = false): Promise<UpdateCheckInfo> {
    const installedTag = this.readBackendVersion().tag;
    const cached = force ? undefined : peekCachedLatestReleaseTag();
    if (cached) {
      return {
        latestTag: cached,
        installedTag,
        updateAvailable: !installedTag || compareReleaseTags(cached, installedTag) > 0,
        checkFailed: false,
        pending: false,
      };
    }

    try {
      const latestTag = await getLatestReleaseTagCached(force);
      return {
        latestTag,
        installedTag,
        updateAvailable: !installedTag || compareReleaseTags(latestTag, installedTag) > 0,
        checkFailed: false,
        pending: false,
      };
    } catch {
      return {
        installedTag,
        updateAvailable: false,
        checkFailed: true,
        pending: false,
      };
    }
  }

  /** Sync snapshot using cache only (no network). */
  peekUpdateCheck(): UpdateCheckInfo {
    const installedTag = this.readBackendVersion().tag;
    const cached = peekCachedLatestReleaseTag();
    if (!cached) {
      return {
        installedTag,
        updateAvailable: false,
        checkFailed: false,
        pending: true,
      };
    }
    return {
      latestTag: cached,
      installedTag,
      updateAvailable: !installedTag || compareReleaseTags(cached, installedTag) > 0,
      checkFailed: false,
      pending: false,
    };
  }

  async installOrUpgrade(
    progress?: vscode.Progress<{ message?: string; increment?: number }>,
    backendOverride?: LlamaBackend,
    options?: { force?: boolean }
  ): Promise<string> {
    progress?.report({ message: "Resolving latest llama.cpp release tag…" });
    let tag: string;
    try {
      tag = await getLatestReleaseTagCached(true);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      throw new Error(
        `${msg}\n\nCould not resolve the latest tag from ${LLAMA_CPP_RELEASES_URL}/latest.\n` +
          `Use “Install release tag…” or “Install from archive…” instead.`
      );
    }
    progress?.report({ message: `Latest release: ${tag}` });

    const backendSetting = backendOverride || this.getBackend();
    const uiBackend =
      backendSetting === "vulkan" || backendSetting === "cuda" || backendSetting === "cpu"
        ? backendSetting
        : this.resolveActiveUiBackend(backendSetting);
    const current = this.readBackendVersion(uiBackend).tag;
    if (!options?.force && current && compareReleaseTags(tag, current) <= 0) {
      progress?.report({ message: `Already on ${current}` });
      return current;
    }

    return this.installByTag(tag, progress, backendOverride);
  }

  /**
   * Install a specific release tag via direct `releases/download` URLs (no GitHub API).
   * Tag examples: `b10154`, or a releases page/tag URL.
   */
  /**
   * Assets GitHub reports for a release, or undefined when the API is
   * unreachable/rate-limited. Only used as a fallback after name probing fails,
   * so routine installs still cost zero API calls.
   */
  private async listReleaseAssets(tag: string): Promise<GithubAsset[] | undefined> {
    try {
      const release = await httpGetJson<GithubRelease>(
        `https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/${encodeURIComponent(tag)}`,
        { Accept: "application/vnd.github+json" }
      );
      return Array.isArray(release?.assets) ? release.assets : [];
    } catch {
      return undefined;
    }
  }

  async installByTag(
    tagInput: string,
    progress?: vscode.Progress<{ message?: string; increment?: number }>,
    backendOverride?: LlamaBackend
  ): Promise<string> {
    this.migrateLegacyInstallIfNeeded();
    const tag = normalizeReleaseTag(tagInput);
    const backendSetting = backendOverride || this.getBackend();
    const uiBackend =
      backendSetting === "vulkan" || backendSetting === "cuda" || backendSetting === "cpu"
        ? backendSetting
        : this.resolveActiveUiBackend(backendSetting);

    const candidates = candidateAssetNames(tag, uiBackend);
    progress?.report({ message: `Looking up ${tag} assets for ${uiBackend}…` });

    let assetName: string | undefined;
    let assetUrl: string | undefined;
    const probes: Array<{ name: string; probe: UrlProbe }> = [];
    for (const name of candidates) {
      const url = directAssetUrl(tag, name);
      progress?.report({ message: `Checking ${name}…` });
      const probe = await probeUrl(url);
      probes.push({ name, probe });
      if (probe.exists) {
        assetName = name;
        assetUrl = url;
        break;
      }
    }

    // Name guessing failed. Ask the API what the release really contains — this
    // is the difference between "no CPU archive found" and "this release only
    // published 3 assets, none of them for linux/x64".
    if (!assetName || !assetUrl) {
      progress?.report({ message: `Asking GitHub what ${tag} actually published…` });
      const listed = await this.listReleaseAssets(tag);
      const picked = listed && pickAsset(listed, uiBackend);
      if (!picked) {
        throw new Error(describeMissingAsset(tag, uiBackend, probes, listed));
      }
      assetName = picked.name;
      assetUrl = picked.browser_download_url;
    }

    const tmpDir = path.join(os.tmpdir(), "llama-aio-vs");
    ensureDirs(tmpDir);
    const archivePath = path.join(tmpDir, assetName);
    progress?.report({ message: `Downloading ${assetName}…` });
    await downloadFile(assetUrl, archivePath, (pct) => {
      progress?.report({ message: `Downloading ${assetName}… ${pct}%` });
    });

    let cudartPath: string | undefined;
    if (uiBackend === "cuda" && process.platform === "win32") {
      for (const cudartName of candidateCudartNames(assetName)) {
        const cudartUrl = directAssetUrl(tag, cudartName);
        if (!(await urlExists(cudartUrl))) {
          continue;
        }
        progress?.report({ message: `Downloading ${cudartName}…` });
        cudartPath = path.join(tmpDir, cudartName);
        await downloadFile(cudartUrl, cudartPath, (pct) => {
          progress?.report({ message: `Downloading ${cudartName}… ${pct}%` });
        });
        break;
      }
    }

    return this.installFromArchive(archivePath, {
      uiBackend,
      tag,
      assetName,
      cudartArchivePath: cudartPath,
      progress,
    });
  }

  /**
   * Install from a local `.zip` / `.tar.gz` (and optional Windows cudart zip).
   * Does not contact the GitHub API.
   */
  async installFromArchive(
    archivePath: string,
    options: {
      uiBackend?: UiBackend;
      tag?: string;
      assetName?: string;
      cudartArchivePath?: string;
      progress?: vscode.Progress<{ message?: string; increment?: number }>;
    } = {}
  ): Promise<string> {
    this.migrateLegacyInstallIfNeeded();
    const progress = options.progress;
    if (!archivePath || !fs.existsSync(archivePath)) {
      throw new Error(`Archive not found: ${archivePath}`);
    }
    const assetName = options.assetName || path.basename(archivePath);
    if (/cudart/i.test(assetName) && !options.cudartArchivePath) {
      throw new Error(
        `"${assetName}" looks like a CUDA runtime pack, not llama-server. ` +
          `Select the main llama-*-bin-* archive first (optionally add cudart as the second file).`
      );
    }
    if (!isArchive(assetName)) {
      throw new Error(`Unsupported archive type: ${assetName} (need .zip, .tar.gz, or .tgz)`);
    }

    const inferred = inferBackendFromAsset(assetName);
    let uiBackend: UiBackend =
      options.uiBackend ||
      (inferred === "vulkan" || inferred === "cuda" || inferred === "cpu"
        ? inferred
        : this.resolveActiveUiBackend());

    // Prefer explicit folder when user already selected a UI backend and asset is ambiguous.
    if (options.uiBackend) {
      uiBackend = options.uiBackend;
    }

    const tag =
      options.tag ||
      (/b\d+/i.exec(assetName)?.[0] ?? /b\d+/i.exec(options.tag || "")?.[0] ?? "local");

    const installDir = getBackendInstallDir(this.store.getConfig(), uiBackend);
    ensureDirs(installDir);
    const tmpDir = path.join(os.tmpdir(), "llama-aio-vs");
    ensureDirs(tmpDir);

    progress?.report({ message: "Extracting…" });
    const extractTo = path.join(tmpDir, `extract-${Date.now()}`);
    ensureDirs(extractTo);
    await extractArchive(archivePath, extractTo);

    const found = findBinaryAfterExtract(extractTo);
    if (!found) {
      throw new Error(
        `Could not find llama-server in ${assetName}. Make sure this is an official llama.cpp binary archive.`
      );
    }

    // Stage the new tree next to the live one and swap at the end, so a failed
    // copy can never leave the user without a working llama-server.
    const binDir = path.join(installDir, "bin");
    const stageDir = path.join(installDir, `bin.new-${process.pid}-${Date.now()}`);
    fs.rmSync(stageDir, { recursive: true, force: true });
    ensureDirs(stageDir);
    const destBin = path.join(stageDir, path.basename(found));
    fs.copyFileSync(found, destBin);
    if (process.platform !== "win32") {
      fs.chmodSync(destBin, 0o755);
    }

    const srcDir = path.dirname(found);
    const copyFailures: string[] = [];
    for (const name of fs.readdirSync(srcDir)) {
      const src = path.join(srcDir, name);
      const dst = path.join(stageDir, name);
      if (fs.statSync(src).isFile() && src !== found) {
        try {
          fs.copyFileSync(src, dst);
          // Ship every llama-* tool executable, not just llama-server.
          if (process.platform !== "win32" && (!name.includes(".") || /^llama-/i.test(name))) {
            fs.chmodSync(dst, 0o755);
          }
        } catch (err) {
          copyFailures.push(`${name}: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    }
    if (process.platform === "win32") {
      copyFilesIntoBin(extractTo, stageDir, [".dll"]);
    }
    if (copyFailures.length) {
      fs.rmSync(stageDir, { recursive: true, force: true });
      throw new Error(
        `Install aborted — could not copy ${copyFailures.length} file(s) from ${assetName}; ` +
          `the previous install is untouched.\n${copyFailures.slice(0, 5).join("\n")}`
      );
    }
    swapInstallDir(stageDir, binDir);

    if (options.cudartArchivePath && fs.existsSync(options.cudartArchivePath)) {
      progress?.report({ message: "Extracting CUDA runtime…" });
      const cudartExtract = path.join(tmpDir, `cudart-extract-${Date.now()}`);
      ensureDirs(cudartExtract);
      await extractArchive(options.cudartArchivePath, cudartExtract);
      copyFilesIntoBin(cudartExtract, binDir, [".dll"]);
      try {
        fs.rmSync(cudartExtract, { recursive: true, force: true });
      } catch {
        // ignore
      }
    } else if (uiBackend === "cuda" && process.platform === "win32") {
      const hasCudart = fs.readdirSync(binDir).some((n) => /cudart64|cublas/i.test(n));
      if (!hasCudart) {
        progress?.report({
          message:
            "Installed CUDA build without cudart DLLs — llama-server may fail until you import cudart-*.zip",
        });
      }
    }

    try {
      fs.rmSync(extractTo, { recursive: true, force: true });
    } catch {
      // ignore
    }

    await this.setBackend(uiBackend);
    const resolved = inferBackendFromAsset(assetName) || uiBackend;
    fs.writeFileSync(
      path.join(installDir, "VERSION"),
      `${tag}\nasset=${assetName}\nbackend=${uiBackend}\nresolved=${resolved}\n`,
      "utf8"
    );
    // Keep update UI in sync when we just installed this tag.
    if (tag && tag !== "local") {
      const cached = peekCachedLatestReleaseTag();
      if (!cached || compareReleaseTags(tag, cached) >= 0) {
        latestTagCache = { tag, fetchedAt: Date.now() };
      }
    }

    progress?.report({ message: `Installed ${tag} (${assetName}) → ${uiBackend}` });
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

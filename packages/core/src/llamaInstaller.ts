import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { Readable, Transform } from "stream";
import { pipeline } from "stream/promises";
import { execFile, execFileSync } from "child_process";
import { promisify } from "util";
import { swapInstallDir, withInstallLock, activeInstallLock } from "./installSwap";
import {
  findSteamRun,
  invalidateLlamaServerProbeCache,
  isNixOS,
  probeLlamaServerRunnable,
} from "./nixCompat";
import {
  ensureDirs,
  findLlamaServerOnPath,
  getBackendInstallDir,
  getInstallDir,
  getLlamaServerBinary,
  getLockPath,
  InstallBackendId,
  whichOnPath,
  withBinaryDirEnv,
} from "./paths";
import { stopLlamaServersUsingDir, isPidAlive, executableIsUnderDir } from "./processIdentity";
import { SettingsStore } from "./settings";
import { resetAmdGpuDetectionCache, rocmSupport, withRocmRuntimeEnv } from "./rocmRuntime";
import { syclSupport, withSyclRuntimeEnv } from "./syclRuntime";

const execFileAsync = promisify(execFile);

/** Backends that download into `~/.llama-aio-vs/llama.cpp/<id>/`. */
const UI_BACKENDS: Array<Exclude<UiBackend, "path">> = ["vulkan", "cuda", "cpu", "sycl", "rocm"];

function installBlockReason(backend: UiBackend): string | undefined {
  const support = backend === "sycl" ? syclSupport() : backend === "rocm" ? rocmSupport() : undefined;
  if (support && !support.available) {
    return support.reason || `${backend} is not available on this system.`;
  }
  return undefined;
}

function isManagedBackend(backend: string): backend is Exclude<UiBackend, "path"> {
  return (
    backend === "vulkan" ||
    backend === "cuda" ||
    backend === "cpu" ||
    backend === "sycl" ||
    backend === "rocm"
  );
}

/** Env used to probe and launch an installed binary. SYCL and ROCm add their runtimes. */
export function envForInstalledBinary(binary: string, backend: UiBackend): NodeJS.ProcessEnv {
  const env = withBinaryDirEnv(binary);
  if (backend === "sycl") {
    return withSyclRuntimeEnv(env);
  }
  if (backend === "rocm") {
    return withRocmRuntimeEnv(env);
  }
  return env;
}

/** User aborted an install. The previous llama.cpp build is left in place. */
export class InstallCancelledError extends Error {
  constructor() {
    super("Install cancelled.");
    this.name = "InstallCancelledError";
  }
}

export type InstallPhase = "resolve" | "download" | "extract" | "replace";

/** Sidebar-facing install update. `percent` is 0–100 within the current file. */
export interface InstallProgress {
  message?: string;
  increment?: number;
  phase?: InstallPhase;
  percent?: number;
  received?: number;
  total?: number;
  bytesPerSec?: number;
  etaSeconds?: number;
  fileIndex?: number;
  fileCount?: number;
  assetName?: string;
  releaseTag?: string;
}

/**
 * Progress sink for long installs. `report` stays compatible with
 * `vscode.Progress` and the TUI. `onInstall` carries bytes, speed, and phase
 * for the sidebar card.
 */
export interface ProgressReporter {
  report(value: { message?: string; increment?: number }): void;
  onInstall?(value: InstallProgress): void;
}

/**
 * Map a download tick onto one notification bar. Two files (Windows CUDA +
 * cudart) each take half of 0–100.
 */
export function installNotificationPercent(update: InstallProgress): number | undefined {
  if (update.phase !== "download" || update.percent == null || !Number.isFinite(update.percent)) {
    return undefined;
  }
  const count = Math.max(1, update.fileCount || 1);
  const index = Math.min(count, Math.max(1, update.fileIndex || 1));
  const span = 100 / count;
  const pct = Math.max(0, Math.min(100, update.percent));
  return Math.round((index - 1) * span + (pct / 100) * span);
}

function emitInstall(progress: ProgressReporter | undefined, update: InstallProgress): void {
  if (!progress) {
    return;
  }
  if (progress.onInstall) {
    progress.onInstall(update);
    return;
  }
  progress.report({ message: update.message });
}

function throwIfCancelled(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new InstallCancelledError();
  }
}

export interface GithubAsset {
  name: string;
  browser_download_url: string;
  /**
   * API asset endpoint. GET with `Accept: application/octet-stream` — github.com
   * `/releases/download` URLs reset HTTP/2 streams (`NGHTTP2_REFUSED_STREAM`).
   */
  url?: string;
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

const GITHUB_UA = "llama-aio-vs";
const GITHUB_API_LATEST = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";
/** Includes prereleases (llama.cpp nightlies are `b####` and marked prerelease). */
const GITHUB_API_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30";
const githubApiTagUrl = (tag: string) =>
  `https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/${encodeURIComponent(tag)}`;
const BUILD_TAG_RE = /^b\d+$/i;

function isRetryableNetworkError(err: unknown): boolean {
  const parts = [err instanceof Error ? `${err.name} ${err.message}` : String(err)];
  if (err instanceof Error && err.cause) {
    parts.push(String(err.cause));
  }
  return /hang up|ECONNRESET|ETIMEDOUT|EAI_AGAIN|ECONNREFUSED|aborted|AbortError|TimeoutError|fetch failed|UND_ERR|socket|NGHTTP2|ERR_HTTP2|REFUSED_STREAM/i.test(
    parts.join(" ")
  );
}

/**
 * Timer abort that can be disarmed after `fetch()` resolves (headers in).
 * `AbortSignal.timeout()` keeps ticking through the body and will kill a long download.
 */
export function createClearableTimeoutSignal(ms: number): { signal: AbortSignal; clear: () => void } {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), ms);
  return {
    signal: ac.signal,
    clear: () => clearTimeout(timer),
  };
}

/**
 * GitHub's HTML site resets Node's `https.get` (HTTP/1.1) with `socket hang up`.
 * Undici `fetch` (HTTP/2) follows redirects and actually completes.
 *
 * `timeoutMode: "headers"` disarms the timer once headers arrive so archive
 * bodies can take longer than the header wait. Default `"total"` still bounds
 * small JSON / HEAD responses.
 */
function anySignal(signals: Array<AbortSignal | undefined>): AbortSignal | undefined {
  const list = signals.filter((signal): signal is AbortSignal => !!signal);
  if (list.length === 0) {
    return undefined;
  }
  if (list.length === 1) {
    return list[0];
  }
  if (typeof AbortSignal.any === "function") {
    return AbortSignal.any(list);
  }
  const ac = new AbortController();
  for (const signal of list) {
    if (signal.aborted) {
      ac.abort(signal.reason);
      return ac.signal;
    }
    signal.addEventListener("abort", () => ac.abort(signal.reason), { once: true });
  }
  return ac.signal;
}

async function githubFetch(
  url: string,
  init: RequestInit = {},
  timeoutMs = 20_000,
  timeoutMode: "total" | "headers" = "total"
): Promise<Response> {
  let lastErr: unknown;
  for (let attempt = 1; attempt <= 3; attempt++) {
    throwIfCancelled(init.signal ?? undefined);
    const timed = timeoutMode === "headers" ? createClearableTimeoutSignal(timeoutMs) : undefined;
    const timeoutSignal = timed?.signal ?? (init.signal ? undefined : AbortSignal.timeout(timeoutMs));
    try {
      const res = await fetch(url, {
        ...init,
        headers: {
          "User-Agent": GITHUB_UA,
          ...(init.headers as Record<string, string> | undefined),
        },
        redirect: init.redirect ?? "follow",
        signal: anySignal([init.signal ?? undefined, timeoutSignal]),
      });
      timed?.clear();
      return res;
    } catch (err) {
      timed?.clear();
      if (init.signal?.aborted) {
        throw new InstallCancelledError();
      }
      lastErr = err;
      if (attempt === 3 || !isRetryableNetworkError(err)) {
        throw err;
      }
      await new Promise((r) => setTimeout(r, 400 * attempt));
    }
  }
  throw lastErr;
}

/**
 * Turn a rate-limited GitHub API response into something a user can act on.
 * Unauthenticated api.github.com allows 60 requests/hour/IP, and the raw body
 * is a JSON blob that reads like a crash rather than a temporary limit.
 */
export function describeGithubHttpError(status: number, res: Response, body: string): string {
  if (status !== 403 && status !== 429) {
    return `HTTP ${status}: ${body.slice(0, 200)}`;
  }
  const remaining = res.headers.get("x-ratelimit-remaining");
  const reset = Number(res.headers.get("x-ratelimit-reset") || 0);
  const limited = status === 429 || remaining === "0" || /rate limit/i.test(body);
  if (!limited) {
    return `HTTP ${status}: ${body.slice(0, 200)}`;
  }
  let when = "";
  if (Number.isFinite(reset) && reset > 0) {
    const mins = Math.max(1, Math.ceil((reset * 1000 - Date.now()) / 60_000));
    when = ` Try again in ~${mins} min.`;
  }
  return (
    `GitHub API rate limit reached (HTTP ${status}).${when} ` +
    `Anonymous requests are capped at 60/hour; wait for the limit to reset, ` +
    `or use “Install from archive…” with a file you download from the releases page.`
  );
}

async function httpGetJson<T>(
  url: string,
  headers: Record<string, string> = {},
  signal?: AbortSignal
): Promise<T> {
  const res = await githubFetch(url, {
    headers: { Accept: "application/vnd.github+json", ...headers },
    signal,
  });
  const body = await res.text();
  if (!res.ok) {
    throw new Error(describeGithubHttpError(res.status, res, body));
  }
  return JSON.parse(body) as T;
}

/** How long to wait for download response headers (redirects included). */
const DOWNLOAD_HEADER_TIMEOUT_MS = 60_000;
const DOWNLOAD_ATTEMPTS = 3;

interface DownloadTick {
  received: number;
  total: number;
  percent?: number;
  bytesPerSec: number;
  etaSeconds?: number;
}

async function downloadFile(
  url: string,
  dest: string,
  onProgress?: (tick: DownloadTick) => void,
  signal?: AbortSignal
): Promise<void> {
  let lastErr: unknown;
  for (let attempt = 1; attempt <= DOWNLOAD_ATTEMPTS; attempt++) {
    throwIfCancelled(signal);
    try {
      await downloadFileOnce(url, dest, onProgress, signal);
      return;
    } catch (err) {
      try {
        fs.rmSync(dest, { force: true });
      } catch {
        // truncated leftover from a dropped connection
      }
      if (signal?.aborted || err instanceof InstallCancelledError) {
        throw new InstallCancelledError();
      }
      lastErr = err;
      if (attempt === DOWNLOAD_ATTEMPTS || !isRetryableNetworkError(err)) {
        throw err;
      }
      await new Promise((r) => setTimeout(r, 400 * attempt));
    }
  }
  throw lastErr;
}

async function downloadFileOnce(
  url: string,
  dest: string,
  onProgress?: (tick: DownloadTick) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await githubFetch(
    url,
    { headers: { Accept: "application/octet-stream" }, signal },
    DOWNLOAD_HEADER_TIMEOUT_MS,
    "headers"
  );
  if (!res.ok) {
    throw new Error(`Download failed: HTTP ${res.status} for ${url}`);
  }
  if (!res.body) {
    throw new Error(`Download failed: empty body for ${url}`);
  }
  const total = Number(res.headers.get("content-length") || 0);
  let received = 0;
  const started = Date.now();
  let lastEmit = 0;
  let lastPct = -1;
  const emit = (force = false) => {
    const elapsed = Math.max(0.001, (Date.now() - started) / 1000);
    const bytesPerSec = received / elapsed;
    const percent = total > 0 ? Math.min(100, Math.floor((received / total) * 100)) : undefined;
    const now = Date.now();
    if (!force && percent === lastPct && now - lastEmit < 200) {
      return;
    }
    lastPct = percent ?? lastPct;
    lastEmit = now;
    onProgress?.({
      received,
      total,
      percent,
      bytesPerSec,
      etaSeconds:
        total > 0 && bytesPerSec > 0 ? Math.max(0, Math.round((total - received) / bytesPerSec)) : undefined,
    });
  };
  ensureDirs(path.dirname(dest));
  const src = Readable.fromWeb(res.body as import("stream/web").ReadableStream<Uint8Array>);
  const counter = new Transform({
    transform(chunk: Buffer | Uint8Array, _enc, cb) {
      received += chunk.length;
      emit();
      cb(null, chunk);
    },
  });
  await pipeline(src, counter, fs.createWriteStream(dest), { signal });
  emit(true);
}

async function downloadReleaseFile(
  progress: ProgressReporter | undefined,
  url: string,
  dest: string,
  assetName: string,
  fileIndex: number,
  fileCount: number,
  releaseTag: string | undefined,
  signal?: AbortSignal
): Promise<void> {
  const label = fileCount > 1 ? `${assetName} (${fileIndex} of ${fileCount})` : assetName;
  const base = {
    phase: "download" as const,
    assetName,
    fileIndex,
    fileCount,
    releaseTag,
  };
  emitInstall(progress, { ...base, message: `Downloading ${label}…` });
  await downloadFile(
    url,
    dest,
    (tick) => {
      const pct = tick.percent != null ? ` ${tick.percent}%` : "";
      emitInstall(progress, {
        ...base,
        message: `Downloading ${label}…${pct}`,
        percent: tick.percent,
        received: tick.received,
        total: tick.total,
        bytesPerSec: tick.bytesPerSec,
        etaSeconds: tick.etaSeconds,
      });
    },
    signal
  );
}

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

async function probeUrl(url: string, signal?: AbortSignal): Promise<UrlProbe> {
  try {
    const res = await githubFetch(url, { method: "HEAD", signal }, 15_000);
    return { exists: res.ok, status: res.status };
  } catch (err) {
    if (signal?.aborted || err instanceof InstallCancelledError) {
      throw err instanceof InstallCancelledError ? err : new InstallCancelledError();
    }
    return { exists: false, error: err instanceof Error ? err.message : String(err) };
  }
}

async function urlExists(url: string, signal?: AbortSignal): Promise<boolean> {
  return (await probeUrl(url, signal)).exists;
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
  if (!BUILD_TAG_RE.test(s)) {
    throw new Error(
      `Invalid release tag "${input}". Use a nightly (b10154), a stable tag (v0.2.0), or a URL like ${LLAMA_CPP_RELEASES_URL}/tag/b10154`
    );
  }
  return `b${s.slice(1)}`;
}

/** Stable llama.cpp tag (`v0.2.0`) from typed input or a releases URL. */
export function parseStableReleaseTag(input: string): string | undefined {
  const s = (input || "").trim();
  if (!s) {
    return undefined;
  }
  const fromUrl = /\/releases\/tag\/(v?\d+\.\d+\.\d+)/i.exec(s);
  const raw = fromUrl?.[1] ?? (/^(v?\d+\.\d+\.\d+)$/i.exec(s)?.[1]);
  if (!raw) {
    return undefined;
  }
  const digits = raw.replace(/^v/i, "");
  return `v${digits}`;
}

export function parseNightlyTagFile(text: string): string | undefined {
  const m = /\b(b\d+)\b/i.exec((text || "").trim());
  return m ? `b${m[1].slice(1)}` : undefined;
}

/** Newest `b####` nightly that actually has binary archives. Skips `vX.Y.Z` stables. */
export function pickNewestBuildTag(
  releases: Array<{ tag_name?: string; assets?: Array<{ name: string }> }>
): string | undefined {
  let bestN = -1;
  let best: string | undefined;
  for (const release of releases) {
    const raw = (release.tag_name || "").trim();
    if (!BUILD_TAG_RE.test(raw)) {
      continue;
    }
    if (release.assets) {
      const hasBinary = release.assets.some((a) => /^llama-b\d+-bin-/i.test(a.name));
      if (!hasBinary) {
        continue;
      }
    }
    const n = Number.parseInt(raw.slice(1), 10);
    if (Number.isFinite(n) && n > bestN) {
      bestN = n;
      best = `b${n}`;
    }
  }
  return best;
}

async function tryReadNightlyPointerTag(stableTag: string): Promise<string | undefined> {
  try {
    const res = await githubFetch(`${LLAMA_CPP_DOWNLOAD_BASE}/${stableTag}/nightly-tag.txt`);
    if (!res.ok) {
      return undefined;
    }
    return parseNightlyTagFile(await res.text());
  } catch {
    return undefined;
  }
}

/** Direct-download asset name candidates for a tag + backend (no GitHub API). */
export function candidateAssetNames(tag: string, backend: UiBackend): string[] {
  const t = normalizeReleaseTag(tag);
  const arch = process.arch === "arm64" ? "arm64" : "x64";
  // Official SYCL nightlies are Windows x64 and Ubuntu x64 (fp16 preferred).
  // ROCm filenames embed the toolkit version (`rocm-10.0`), which moves, so the
  // release asset list is the only reliable source. Direct URL guesses stay empty.
  if (backend === "rocm") {
    return [];
  }
  if (backend === "sycl") {
    if (process.platform === "win32" && arch === "x64") {
      return [`llama-${t}-bin-win-sycl-x64.zip`];
    }
    if (process.platform === "linux" && arch === "x64") {
      return [
        `llama-${t}-bin-ubuntu-sycl-fp16-${arch}.tar.gz`,
        `llama-${t}-bin-ubuntu-sycl-fp32-${arch}.tar.gz`,
      ];
    }
    return [];
  }
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
 * Newest llama.cpp nightly (`b####`). GitHub's `/releases/latest` is now the
 * stable `vX.Y.Z` tag, which has no binary archives — only `nightly-tag.txt`.
 */
export async function resolveLatestReleaseTag(signal?: AbortSignal): Promise<string> {
  throwIfCancelled(signal);
  try {
    const releases = await httpGetJson<GithubRelease[]>(GITHUB_API_RELEASES, {}, signal);
    const tag = pickNewestBuildTag(releases);
    if (tag) {
      return tag;
    }
  } catch (err) {
    throwIfCancelled(signal);
    if (err instanceof InstallCancelledError) {
      throw err;
    }
    // Rate limit or API outage — try the stable pointer, then the HTML list.
  }

  try {
    const latest = await httpGetJson<GithubRelease>(GITHUB_API_LATEST, {}, signal);
    const name = latest?.tag_name?.trim() || "";
    if (BUILD_TAG_RE.test(name)) {
      return normalizeReleaseTag(name);
    }
    if (name) {
      const pointed = await tryReadNightlyPointerTag(name);
      if (pointed) {
        return pointed;
      }
    }
  } catch (err) {
    throwIfCancelled(signal);
    if (err instanceof InstallCancelledError) {
      throw err;
    }
    // keep falling through
  }

  throwIfCancelled(signal);
  const res = await githubFetch(LLAMA_CPP_RELEASES_URL, { signal });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} resolving latest llama.cpp nightly from ${res.url}`);
  }
  const html = await res.text();
  const scraped = [...html.matchAll(/\/releases\/tag\/(b\d+)/gi)].map((m) => ({
    tag_name: m[1],
  }));
  const fromHtml = pickNewestBuildTag(scraped);
  if (fromHtml) {
    return fromHtml;
  }
  throw new Error(
    `Could not find a llama.cpp nightly tag (b####) on ${LLAMA_CPP_RELEASES_URL}`
  );
}

const LATEST_TAG_CACHE_MS = 45 * 60 * 1000;
let latestTagCache: { tag: string; fetchedAt: number } | undefined;

/**
 * Compare release tags (`b10173`). Positive if `a` is newer than `b`.
 * Returns undefined when either tag is not a llama.cpp build tag (`local`, `PATH`, empty).
 */
export function compareReleaseTags(a: string, b: string): number | undefined {
  const naTag = tryNormalizeReleaseTag(a);
  const nbTag = tryNormalizeReleaseTag(b);
  if (!naTag || !nbTag) {
    return undefined;
  }
  const na = Number.parseInt(naTag.replace(/^b/i, ""), 10);
  const nb = Number.parseInt(nbTag.replace(/^b/i, ""), 10);
  if (Number.isFinite(na) && Number.isFinite(nb)) {
    return na - nb;
  }
  return naTag.localeCompare(nbTag, undefined, {
    numeric: true,
    sensitivity: "base",
  });
}

function tryNormalizeReleaseTag(input: string): string | undefined {
  try {
    return normalizeReleaseTag(input);
  } catch {
    return undefined;
  }
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

/** Cached latest-tag lookup. */
export async function getLatestReleaseTagCached(force = false, signal?: AbortSignal): Promise<string> {
  throwIfCancelled(signal);
  if (!force) {
    const cached = peekCachedLatestReleaseTag();
    if (cached) {
      return cached;
    }
  }
  try {
    const tag = await resolveLatestReleaseTag(signal);
    latestTagCache = { tag, fetchedAt: Date.now() };
    return tag;
  } catch (err) {
    if (err instanceof InstallCancelledError) {
      throw err;
    }
    // Keep a previously resolved tag so Upgrade still works if GitHub blips.
    if (latestTagCache?.tag) {
      return latestTagCache.tag;
    }
    throw err;
  }
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

export type LlamaBackend =
  | "auto"
  | "vulkan"
  | "rocm"
  | "cuda"
  | "cpu"
  | "openvino"
  | "sycl"
  | "path";

/** Backends offered in the sidebar / TUI switcher. */
export type UiBackend = "vulkan" | "cuda" | "cpu" | "sycl" | "rocm" | "path";

export interface BackendOption {
  id: UiBackend;
  label: string;
  available: boolean;
  reason?: string;
  /** True when this backend already has a local install (or PATH binary). */
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
  /** False when the binary exists but `llama-server --version` cannot run (common on NixOS). */
  binaryRunnable?: boolean;
  /** Short probe error when binaryRunnable is false. */
  binaryRunError?: string;
  asset?: string;
  /** Configured backend setting */
  configuredBackend: LlamaBackend;
  /** Concrete UI backend currently used for the binary path */
  activeBackend: UiBackend;
  /** Backend inferred from installed asset name */
  resolvedBackend?: UiBackend | string;
  /** Absolute path when using the System (PATH) backend. */
  pathBinary?: string;
  /** True when running on NixOS (helps the UI show FHS / nix-ld guidance). */
  nixOs?: boolean;
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

/** PATH lookup without spawning a shell (`sh -c command -v` cost a fork per call). */
function commandExists(bin: string): boolean {
  return !!whichOnPath(bin);
}

/**
 * Hardware presence does not change while the extension host runs, but these
 * probes used to run `nvidia-smi` (up to 3 s) on every status-bar tick and
 * every sidebar refresh via resolveActiveUiBackend(). Cache them.
 */
export const HARDWARE_DETECT_TTL_MS = 5 * 60 * 1000;
const hardwareCache = new Map<string, { value: boolean; at: number }>();

function cachedDetect(key: string, probe: () => boolean, now = Date.now()): boolean {
  const hit = hardwareCache.get(key);
  if (hit && now - hit.at < HARDWARE_DETECT_TTL_MS) {
    return hit.value;
  }
  const value = probe();
  hardwareCache.set(key, { value, at: now });
  return value;
}

/** Forget cached GPU/Vulkan/ROCm detection (tests, or after a driver install). */
export function resetHardwareDetectionCache(): void {
  hardwareCache.clear();
  resetAmdGpuDetectionCache();
}

/** True when an NVIDIA GPU / CUDA runtime looks present on this machine. */
export function detectCudaHardware(): boolean {
  return cachedDetect("cuda", detectCudaHardwareUncached);
}

function detectCudaHardwareUncached(): boolean {
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
  return cachedDetect("vulkan", detectVulkanHardwareUncached);
}

function detectVulkanHardwareUncached(): boolean {
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
      // fp16 is upstream's recommended Ubuntu SYCL build. Asset order must not flip it.
      if (b === "sycl") {
        if (n.includes("sycl-fp16")) {
          score += 40;
        } else if (n.includes("sycl-fp32")) {
          score += 10;
        }
      }
      // The ROCm version is part of the filename and changes between nightlies.
      // Prefer the highest `rocm-N` over an older `hip-radeon` zip in the same release.
      if (b === "rocm") {
        const ver = /rocm-(\d+)(?:\.(\d+))?/.exec(n);
        if (ver) {
          // Stay under the 100-point gap between wanted backends, so a newer
          // ROCm archive cannot outscore Vulkan when the picker is on auto.
          score += Math.min(80, Number(ver[1]) * 2 + Math.min(9, Number(ver[2] || 0)));
        } else if (n.includes("hip-radeon")) {
          score += 1;
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

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Drop a stale server.lock.json after we killed the process that owned those binaries. */
function clearServerLockIfUsingDir(binDir: string): void {
  const lockFile = getLockPath();
  try {
    if (!fs.existsSync(lockFile)) {
      return;
    }
    const lock = JSON.parse(fs.readFileSync(lockFile, "utf8")) as { pid?: number; binary?: string };
    const dead = !lock.pid || !isPidAlive(lock.pid);
    const ours = executableIsUnderDir(lock.binary, binDir);
    if (dead || ours) {
      fs.unlinkSync(lockFile);
    }
  } catch {
    // ignore
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
  const { command, argv } = buildArchiveExtractCommand(archivePath, destDir);
  await execFileAsync(command, argv);
}

/**
 * Unpack llama.cpp zips/tarballs without PowerShell (`Expand-Archive` is a
 * common Defender false-positive). Windows 10+ ships `tar.exe` (libarchive).
 */
export function buildArchiveExtractCommand(
  archivePath: string,
  destDir: string
): { command: string; argv: string[] } {
  if (archivePath.endsWith(".zip") && process.platform !== "win32") {
    return { command: "unzip", argv: ["-o", archivePath, "-d", destDir] };
  }
  const tar = process.platform === "win32" ? "tar.exe" : "tar";
  if (archivePath.endsWith(".zip")) {
    return { command: tar, argv: ["-xf", archivePath, "-C", destDir] };
  }
  return { command: tar, argv: ["-xzf", archivePath, "-C", destDir] };
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
    const allowed: LlamaBackend[] = [
      "auto",
      "vulkan",
      "rocm",
      "cuda",
      "cpu",
      "openvino",
      "sycl",
      "path",
    ];
    return (allowed.includes(value as LlamaBackend) ? value : "auto") as LlamaBackend;
  }

  async setBackend(backend: LlamaBackend): Promise<void> {
    await this.store.getConfig().update("backend", backend);
  }

  /** Map config / auto to a concrete UI backend folder name. */
  resolveActiveUiBackend(configured = this.getBackend()): UiBackend {
    this.migrateLegacyInstallIfNeeded();
    if (configured === "path" || isManagedBackend(configured)) {
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
    if (active === "path") {
      // PATH backend has no install dir; callers should use findLlamaServerOnPath.
      return getInstallDir(this.store.getConfig());
    }
    return getBackendInstallDir(this.store.getConfig(), active as InstallBackendId);
  }

  /** Read VERSION metadata for a backend (or active). */
  readBackendVersion(backend?: UiBackend): BackendVersionInfo {
    const active = backend || this.resolveActiveUiBackend();
    if (active === "path") {
      return {};
    }
    const dir = this.getActiveInstallDir(active);
    return readVersionFile(path.join(dir, "VERSION"));
  }

  hasBackendInstalled(backend: UiBackend): boolean {
    this.migrateLegacyInstallIfNeeded();
    if (backend === "path") {
      return !!findLlamaServerOnPath();
    }
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
    const configured = this.getBackend();
    if (isManagedBackend(resolved)) {
      backend = resolved;
    } else if (isManagedBackend(configured)) {
      backend = configured;
    }

    const dest = getBackendInstallDir(this.store.getConfig(), backend as InstallBackendId);
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

  /** Sidebar options: Vulkan / ROCm (AMD GPU) / CUDA (NVIDIA) / SYCL / CPU / System (PATH). */
  getUiBackendOptions(): BackendOption[] {
    this.migrateLegacyInstallIfNeeded();
    const archLabel = process.arch === "arm64" ? "arm64" : "x64";
    const cudaHw = detectCudaHardware();
    const vulkanOk = detectVulkanHardware();
    const cudaAssetLikely =
      (process.platform === "win32" || process.platform === "linux") && process.arch !== "arm64";
    const sycl = syclSupport();
    const rocm = rocmSupport();
    const active = this.resolveActiveUiBackend();
    const pathBinary = findLlamaServerOnPath();

    const base: Array<Omit<BackendOption, "installed" | "installedTag" | "active">> = [
      {
        id: "vulkan",
        label: "Vulkan",
        available: vulkanOk,
        reason: vulkanOk ? undefined : "Not available on this platform",
      },
      {
        id: "rocm",
        label: "ROCm",
        available: rocm.available,
        reason: rocm.reason,
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
        id: "sycl",
        label: "SYCL",
        available: sycl.available,
        reason: sycl.reason,
      },
      {
        id: "cpu",
        label: `CPU (${archLabel})`,
        available: true,
      },
      {
        id: "path",
        label: "System (PATH)",
        available: true,
        reason: pathBinary
          ? undefined
          : "llama-server not found on PATH — install a system package first",
      },
    ];

    return base.map((opt) => {
      if (opt.id === "path") {
        return {
          ...opt,
          installed: !!pathBinary,
          installedTag: pathBinary ? path.basename(pathBinary) : undefined,
          active: opt.id === active,
        };
      }
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
    const nixOs = isNixOS();

    if (activeBackend === "path") {
      const pathBinary = findLlamaServerOnPath();
      let binaryVersion: string | undefined;
      let binaryVersionDetail: string | undefined;
      let binaryRunnable: boolean | undefined;
      let binaryRunError: string | undefined;
      if (pathBinary) {
        const probe = probeLlamaServerRunnable(pathBinary, { env: withBinaryDirEnv(pathBinary) });
        if (probe.ok) {
          binaryRunnable = true;
          binaryVersion = probe.detail;
          binaryVersionDetail = `${probe.detail} · ${pathBinary}`;
        } else {
          binaryRunnable = false;
          binaryRunError = probe.detail;
          binaryVersionDetail = pathBinary;
        }
      }
      return {
        binaryVersion,
        binaryVersionDetail,
        binaryRunnable,
        binaryRunError,
        configuredBackend,
        activeBackend,
        resolvedBackend: "path",
        pathBinary,
        nixOs,
      };
    }

    const installDir = this.getActiveInstallDir(activeBackend);
    const version = readVersionFile(path.join(installDir, "VERSION"));

    let binaryVersion: string | undefined;
    let binaryVersionDetail: string | undefined;
    let binaryRunnable: boolean | undefined;
    let binaryRunError: string | undefined;
    const binary = getLlamaServerBinary(installDir);
    // Other windows exec llama-server --version every few seconds; skip that
    // while an install holds the lock so Windows can rename vulkan\bin.
    if (fs.existsSync(binary) && !activeInstallLock()) {
      const env = envForInstalledBinary(binary, activeBackend);
      let probe = probeLlamaServerRunnable(binary, { env });
      let viaSteamRun = false;
      if (!probe.ok) {
        const steamRun = findSteamRun();
        if (steamRun) {
          const wrapped = probeLlamaServerRunnable(binary, { env, wrapper: steamRun });
          if (wrapped.ok) {
            probe = wrapped;
            viaSteamRun = true;
          }
        }
      }
      if (probe.ok) {
        binaryRunnable = true;
        binaryVersion = probe.detail;
        binaryVersionDetail = viaSteamRun ? `${probe.detail} (via steam-run)` : probe.detail;
      } else {
        binaryRunnable = false;
        binaryRunError = probe.detail;
      }
    }

    return {
      tag: version.tag,
      binaryVersion,
      binaryVersionDetail,
      binaryRunnable,
      binaryRunError,
      asset: version.asset,
      configuredBackend,
      activeBackend,
      resolvedBackend: version.resolved || inferBackendFromAsset(version.asset) || activeBackend,
      nixOs,
    };
  }

  /**
   * Compare active backend install vs latest release tag (cached ~45 min).
   * Pass force=true to refresh the network check.
   */
  async getUpdateCheck(force = false): Promise<UpdateCheckInfo> {
    if (this.resolveActiveUiBackend() === "path") {
      return {
        installedTag: findLlamaServerOnPath() ? "PATH" : undefined,
        updateAvailable: false,
        checkFailed: false,
        pending: false,
      };
    }
    const installedTag = this.readBackendVersion().tag;
    const cached = force ? undefined : peekCachedLatestReleaseTag();
    if (cached) {
      return {
        latestTag: cached,
        installedTag,
        updateAvailable: !installedTag || (compareReleaseTags(cached, installedTag) ?? 0) > 0,
        checkFailed: false,
        pending: false,
      };
    }

    try {
      const latestTag = await getLatestReleaseTagCached(force);
      return {
        latestTag,
        installedTag,
        updateAvailable: !installedTag || (compareReleaseTags(latestTag, installedTag) ?? 0) > 0,
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

  /**
   * Sync snapshot using cache only (no network).
   * Never throws: a local or unparsable VERSION tag must not break sidebar refresh.
   */
  peekUpdateCheck(): UpdateCheckInfo {
    try {
      if (this.resolveActiveUiBackend() === "path") {
        return {
          installedTag: findLlamaServerOnPath() ? "PATH" : undefined,
          updateAvailable: false,
          checkFailed: false,
          pending: false,
        };
      }
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
      const cmp = installedTag ? compareReleaseTags(cached, installedTag) : undefined;
      return {
        latestTag: cached,
        installedTag,
        updateAvailable: !installedTag || (cmp !== undefined && cmp > 0),
        checkFailed: false,
        pending: false,
      };
    } catch {
      return {
        updateAvailable: false,
        checkFailed: true,
        pending: false,
      };
    }
  }

  async installOrUpgrade(
    progress?: ProgressReporter,
    backendOverride?: LlamaBackend,
    options?: { force?: boolean; signal?: AbortSignal }
  ): Promise<string> {
    const backendSetting = backendOverride || this.getBackend();
    if (backendSetting === "path" || (!backendOverride && this.resolveActiveUiBackend() === "path")) {
      throw new Error(
        "The System (PATH) backend uses a llama-server already on your PATH. " +
          "Switch to Vulkan, ROCm, CUDA, SYCL, or CPU to download an official llama.cpp build."
      );
    }

    throwIfCancelled(options?.signal);
    emitInstall(progress, {
      phase: "resolve",
      message: "Resolving latest llama.cpp release tag…",
    });
    let tag: string;
    try {
      tag = await getLatestReleaseTagCached(true, options?.signal);
    } catch (e) {
      if (e instanceof InstallCancelledError) {
        throw e;
      }
      const msg = e instanceof Error ? e.message : String(e);
      throw new Error(
        `${msg}\n\nCould not resolve the latest llama.cpp nightly (b-tag) from GitHub.\n` +
          `Use “Install release tag…” or “Install from archive…” instead.`
      );
    }
    emitInstall(progress, {
      phase: "resolve",
      message: `Latest release: ${tag}`,
      releaseTag: tag,
    });

    const uiBackend = isManagedBackend(backendSetting)
      ? backendSetting
      : this.resolveActiveUiBackend(backendSetting);
    const current = this.readBackendVersion(uiBackend).tag;
    const cmp = current ? compareReleaseTags(tag, current) : undefined;
    if (!options?.force && current && cmp !== undefined && cmp <= 0) {
      emitInstall(progress, { phase: "resolve", message: `Already on ${current}`, releaseTag: current });
      return current;
    }

    return this.installByTag(tag, progress, backendOverride, { signal: options?.signal });
  }

  /**
   * Assets GitHub reports for a release, or undefined when the API is
   * unreachable/rate-limited.
   */
  private async listReleaseAssets(tag: string, signal?: AbortSignal): Promise<GithubAsset[] | undefined> {
    try {
      const release = await httpGetJson<GithubRelease>(githubApiTagUrl(tag), {}, signal);
      return Array.isArray(release?.assets) ? release.assets : [];
    } catch (err) {
      if (signal?.aborted || err instanceof InstallCancelledError) {
        throw err instanceof InstallCancelledError ? err : new InstallCancelledError();
      }
      return undefined;
    }
  }

  /** Install a specific release tag (`b10154` or a releases URL). */
  async installByTag(
    tagInput: string,
    progress?: ProgressReporter,
    backendOverride?: LlamaBackend,
    options?: { signal?: AbortSignal }
  ): Promise<string> {
    return withInstallLock(
      () => this.installByTagUnlocked(tagInput, progress, backendOverride, options?.signal),
      { progress }
    );
  }

  private async installByTagUnlocked(
    tagInput: string,
    progress?: ProgressReporter,
    backendOverride?: LlamaBackend,
    signal?: AbortSignal
  ): Promise<string> {
    this.migrateLegacyInstallIfNeeded();
    const stable = parseStableReleaseTag(tagInput);
    let tag: string;
    throwIfCancelled(signal);
    if (stable) {
      emitInstall(progress, {
        phase: "resolve",
        message: `Stable ${stable} — resolving matching nightly…`,
        releaseTag: stable,
      });
      const pointed = await tryReadNightlyPointerTag(stable);
      if (!pointed) {
        throw new Error(
          `Stable tag ${stable} has no binary archives (llama.cpp ships those on nightlies). ` +
            `Could not read nightly-tag.txt. Paste a b-tag such as b10587, or use Install / Upgrade.`
        );
      }
      tag = pointed;
    } else {
      tag = normalizeReleaseTag(tagInput);
    }
    const backendSetting = backendOverride || this.getBackend();
    const uiBackend = isManagedBackend(backendSetting)
      ? backendSetting
      : this.resolveActiveUiBackend(backendSetting);
    if (uiBackend === "sycl" || uiBackend === "rocm") {
      const blocked = installBlockReason(uiBackend);
      if (blocked) {
        throw new Error(blocked);
      }
    }

    emitInstall(progress, {
      phase: "resolve",
      message: `Looking up ${tag} assets for ${uiBackend}…`,
      releaseTag: tag,
    });

    let assetName: string | undefined;
    let assetUrl: string | undefined;
    const probes: Array<{ name: string; probe: UrlProbe }> = [];

    const listed = await this.listReleaseAssets(tag, signal);
    const picked = listed && pickAsset(listed, uiBackend);
    if (picked) {
      assetName = picked.name;
      assetUrl = picked.url || picked.browser_download_url;
    } else {
      const candidates = candidateAssetNames(tag, uiBackend);
      for (const name of candidates) {
        throwIfCancelled(signal);
        const url = directAssetUrl(tag, name);
        emitInstall(progress, {
          phase: "resolve",
          message: `Checking ${name}…`,
          releaseTag: tag,
          assetName: name,
        });
        const probe = await probeUrl(url, signal);
        probes.push({ name, probe });
        if (probe.exists) {
          assetName = name;
          assetUrl = url;
          break;
        }
      }
    }

    if (!assetName || !assetUrl) {
      throw new Error(describeMissingAsset(tag, uiBackend, probes, listed));
    }

    const tmpDir = path.join(os.tmpdir(), "llama-aio-vs");
    ensureDirs(tmpDir);
    const archivePath = path.join(tmpDir, assetName);
    const cudartPlanned =
      uiBackend === "cuda" && process.platform === "win32" && listed
        ? pickCudartAsset(listed, assetName)
        : undefined;
    const fileCount = cudartPlanned ? 2 : 1;
    await downloadReleaseFile(progress, assetUrl, archivePath, assetName, 1, fileCount, tag, signal);

    let cudartPath: string | undefined;
    if (uiBackend === "cuda" && process.platform === "win32") {
      const cudart = cudartPlanned;
      const cudartUrl = cudart?.url || cudart?.browser_download_url;
      const cudartName = cudart?.name;
      if (cudartUrl && cudartName) {
        cudartPath = path.join(tmpDir, cudartName);
        await downloadReleaseFile(progress, cudartUrl, cudartPath, cudartName, 2, 2, tag, signal);
      } else {
        for (const name of candidateCudartNames(assetName)) {
          throwIfCancelled(signal);
          const url = directAssetUrl(tag, name);
          if (!(await urlExists(url, signal))) {
            continue;
          }
          cudartPath = path.join(tmpDir, name);
          await downloadReleaseFile(progress, url, cudartPath, name, 2, 2, tag, signal);
          break;
        }
      }
    }

    return this.installFromArchiveUnlocked(archivePath, {
      uiBackend,
      tag,
      assetName,
      cudartArchivePath: cudartPath,
      progress,
      signal,
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
      progress?: ProgressReporter;
      signal?: AbortSignal;
    } = {}
  ): Promise<string> {
    return withInstallLock(() => this.installFromArchiveUnlocked(archivePath, options), {
      progress: options.progress,
    });
  }

  private async installFromArchiveUnlocked(
    archivePath: string,
    options: {
      uiBackend?: UiBackend;
      tag?: string;
      assetName?: string;
      cudartArchivePath?: string;
      progress?: ProgressReporter;
      signal?: AbortSignal;
    } = {}
  ): Promise<string> {
    this.migrateLegacyInstallIfNeeded();
    const progress = options.progress;
    const signal = options.signal;
    throwIfCancelled(signal);
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
      (inferred && isManagedBackend(inferred) ? inferred : this.resolveActiveUiBackend());

    // Prefer explicit folder when user already selected a UI backend and asset is ambiguous.
    if (options.uiBackend) {
      uiBackend = options.uiBackend;
    }
    if (!isManagedBackend(uiBackend)) {
      throw new Error(
        "Cannot install an archive into the System (PATH) backend. Switch to Vulkan, ROCm, CUDA, SYCL, or CPU first."
      );
    }
    const blocked = installBlockReason(uiBackend);
    if (blocked) {
      throw new Error(blocked);
    }

    const tag =
      options.tag ||
      (/b\d+/i.exec(assetName)?.[0] ?? /b\d+/i.exec(options.tag || "")?.[0] ?? "local");

    const installDir = getBackendInstallDir(this.store.getConfig(), uiBackend);
    ensureDirs(installDir);
    const tmpDir = path.join(os.tmpdir(), "llama-aio-vs");
    ensureDirs(tmpDir);

    emitInstall(progress, { phase: "extract", message: "Extracting…", assetName, releaseTag: tag });
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

    if (signal?.aborted) {
      fs.rmSync(extractTo, { recursive: true, force: true });
      fs.rmSync(stageDir, { recursive: true, force: true });
      throw new InstallCancelledError();
    }
    emitInstall(progress, {
      phase: "replace",
      message: "Stopping llama-server so binaries can be replaced…",
      assetName,
      releaseTag: tag,
    });
    await stopLlamaServersUsingDir(binDir);
    clearServerLockIfUsingDir(binDir);
    if (process.platform === "win32") {
      // Mapped DLLs can linger after taskkill returns.
      await sleep(400);
    }
    try {
      await swapInstallDir(stageDir, binDir, { progress });
    } catch (err) {
      try {
        fs.rmSync(stageDir, { recursive: true, force: true });
      } catch {
        // ignore
      }
      throw err;
    }
    invalidateLlamaServerProbeCache();

    if (options.cudartArchivePath && fs.existsSync(options.cudartArchivePath)) {
      emitInstall(progress, {
        phase: "extract",
        message: "Extracting CUDA runtime…",
        assetName,
        releaseTag: tag,
      });
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
      const cmp = cached ? compareReleaseTags(tag, cached) : undefined;
      if (!cached || (cmp !== undefined && cmp >= 0)) {
        latestTagCache = { tag, fetchedAt: Date.now() };
      }
    }

    emitInstall(progress, {
      phase: "replace",
      message: `Installed ${tag} (${assetName}) → ${uiBackend}`,
      assetName,
      releaseTag: tag,
    });
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

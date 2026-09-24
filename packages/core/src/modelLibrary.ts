import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import type { ConfigAccessor } from "./config";
import { getModelsDir } from "./paths";
import { speculativeUsesDflash, speculativeUsesMtp } from "./types";

export interface LocalModelEntry {
  path: string;
  /** Display source, e.g. "LM Studio", "Llama AIO" */
  source: string;
  sizeBytes: number;
  /** When this row stands for a split GGUF (`-00001-of-00005`, …). */
  shardCount?: number;
}

interface ScanRoot {
  dir: string;
  source: string;
  /** Max directory depth relative to root (files deeper are skipped). */
  maxDepth?: number;
}

const MIN_MODEL_BYTES = 32 * 1024 * 1024; // skip vocab / tiny stubs
const MIN_MMPROJ_BYTES = 1024 * 1024; // CLIP projectors are ~20–900 MB
const MIN_MTP_BYTES = 1024 * 1024; // sidecar MTP drafters are typically hundreds of MB
const SKIP_NAME_RE = /^(ggml-vocab)/i;
const GiB = 1024 * 1024 * 1024;
/** Smaller/larger size ratio at which Foo.gguf and Foo-mtp.gguf are the same model. */
const BAKED_IN_NEAR_COPY_RATIO = 0.85;
/** Extra bytes still allowed for a baked-in MTP head on a multi-GB pair. */
const BAKED_IN_MAX_EXTRA_BYTES = 2 * GiB;
/** Unpaired `*-mtp.gguf` below this is treated as a sidecar, not a language GGUF. */
const UNPAIRED_SIDECAR_MAX_BYTES = 2 * GiB;

/** Path basename that works for OS paths and Hugging Face `dir/file.gguf` listings. */
export function listingBaseName(name: string): string {
  const n = (name || "").replace(/\\/g, "/");
  const i = n.lastIndexOf("/");
  return i >= 0 ? n.slice(i + 1) : n;
}

function listingParentName(name: string): string {
  const n = (name || "").replace(/\\/g, "/").replace(/\/+$/, "");
  const i = n.lastIndexOf("/");
  if (i <= 0) {
    return "";
  }
  return listingBaseName(n.slice(0, i));
}

/** True for llama.cpp vision projector GGUFs (`mmproj-F16.gguf`, …). */
export function isMmprojFileName(name: string): boolean {
  return /mmproj/i.test(listingBaseName(name));
}

/** Importance-matrix dumps (`imatrix-qwen3.8-27b.gguf`) — not a runnable model. */
export function isImatrixFileName(name: string): boolean {
  const base = listingBaseName(name);
  return /^imatrix[-_.]/i.test(base) || /^imatrix\.gguf$/i.test(base);
}

/**
 * Pick the best projector from a list of paths. Prefers F16, then BF16, then
 * higher-bit quants — matching what llama.cpp docs usually recommend.
 */
export function preferMmprojPath(paths: string[]): string | undefined {
  if (!paths.length) {
    return undefined;
  }
  const score = (p: string): number => {
    const n = path.basename(p).toLowerCase();
    if (/\bf16\b/.test(n) && !n.includes("bf16")) {
      return 0;
    }
    if (n.includes("bf16")) {
      return 1;
    }
    if (n.includes("f32")) {
      return 2;
    }
    if (n.includes("q8")) {
      return 3;
    }
    if (n.includes("q6") || n.includes("q5")) {
      return 4;
    }
    if (n.includes("q4")) {
      return 5;
    }
    return 6;
  };
  return [...paths].sort((a, b) => score(a) - score(b) || a.localeCompare(b))[0];
}

/** First usable `mmproj*.gguf` sitting next to a language GGUF. */
export function findSiblingMmproj(modelPath: string): string | undefined {
  const trimmed = (modelPath || "").trim();
  if (!trimmed) {
    return undefined;
  }
  const dir = path.dirname(trimmed);
  let names: string[];
  try {
    names = fs.readdirSync(dir);
  } catch {
    return undefined;
  }
  const found: string[] = [];
  for (const name of names) {
    if (!name.toLowerCase().endsWith(".gguf") || !isMmprojFileName(name)) {
      continue;
    }
    const full = path.join(dir, name);
    try {
      const st = fs.statSync(full);
      if (st.isFile() && st.size >= MIN_MMPROJ_BYTES) {
        found.push(full);
      }
    } catch {
      // skip unreadable
    }
  }
  return preferMmprojPath(found);
}

/**
 * Projector to load with this language GGUF.
 * Switching models drops a projector from another folder and attaches a sibling
 * `mmproj*.gguf` when one exists. Reloading the same model keeps a manual pick.
 */
export function resolveMmprojPath(
  modelPath: string,
  current: string | undefined,
  pathChanged: boolean
): string {
  if (!pathChanged) {
    const existing = (current || "").trim();
    if (existing && fs.existsSync(existing)) {
      return existing;
    }
    if (existing) {
      return findSiblingMmproj(modelPath) || "";
    }
    return "";
  }
  return findSiblingMmproj(modelPath) || "";
}

/** Size of a projector file, or 0 when missing / unreadable. */
export function mmprojFileSize(mmprojPath: string | undefined): number {
  const p = (mmprojPath || "").trim();
  if (!p) {
    return 0;
  }
  try {
    const st = fs.statSync(p);
    return st.isFile() ? st.size : 0;
  } catch {
    return 0;
  }
}

/**
 * Name looks MTP-related: `mtp-*.gguf`, files under `MTP/`, or `*-mtp.gguf`.
 * This is not enough to decide sidecar vs baked-in — use {@link classifyGgufFile}.
 */
export function isMtpDraftFileName(name: string): boolean {
  const base = listingBaseName(name);
  if (!base.toLowerCase().endsWith(".gguf") || isMmprojFileName(base) || isImatrixFileName(base)) {
    return false;
  }
  return /^mtp[-_]/i.test(base) || /-mtp\.gguf$/i.test(base) || listingParentName(name).toLowerCase() === "mtp";
}

/** Sidecar from the name alone — Unsloth `mtp-*.gguf` or anything in an `MTP/` folder. */
export function isConfidentMtpSidecarPath(name: string): boolean {
  const base = listingBaseName(name);
  if (!base.toLowerCase().endsWith(".gguf") || isMmprojFileName(base) || isImatrixFileName(base)) {
    return false;
  }
  return /^mtp[-_]/i.test(base) || listingParentName(name).toLowerCase() === "mtp";
}

function hasMtpSuffix(name: string): boolean {
  return /-mtp\.gguf$/i.test(listingBaseName(name));
}

/** Stem used to pair `Foo.gguf` with `Foo-mtp.gguf`. */
export function mtpPairStem(name: string): string {
  return listingBaseName(name)
    .toLowerCase()
    .replace(/\.gguf$/i, "")
    .replace(/-mtp$/i, "");
}

export function extractQuantToken(name: string): string | undefined {
  const base = listingBaseName(name).toLowerCase();
  const m = base.match(
    /[-_.](iq\d+_[a-z0-9]+|q\d+_k(?:_[a-z]+)?|q\d+_\d|q\d+|bf16|f16|f32)(?:-mtp)?\.gguf$/i
  );
  return m?.[1]?.toLowerCase();
}

export type GgufListedFile = { path: string; size: number };

export type GgufFileRole = "language" | "mmproj" | "imatrix" | "mtp-sidecar" | "mtp-baked";

function isNearCopySize(a: number, b: number): boolean {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  if (lo < MIN_MODEL_BYTES || hi <= 0) {
    return false;
  }
  if (lo / hi >= BAKED_IN_NEAR_COPY_RATIO) {
    return true;
  }
  return lo >= 2 * GiB && hi - lo <= BAKED_IN_MAX_EXTRA_BYTES;
}

function findMtpLanguagePair(
  mtpPath: string,
  siblings: GgufListedFile[]
): GgufListedFile | undefined {
  const stem = mtpPairStem(mtpPath);
  const self = listingBaseName(mtpPath).toLowerCase();
  return siblings.find((s) => {
    const base = listingBaseName(s.path).toLowerCase();
    if (base === self || hasMtpSuffix(s.path) || isMmprojFileName(s.path) || isImatrixFileName(s.path)) {
      return false;
    }
    return mtpPairStem(s.path) === stem;
  });
}

/**
 * Classify a GGUF in a repo or folder listing.
 *
 * ISTA-style `Foo-mtp.gguf` next to a near-copy `Foo.gguf` is a full language
 * model with the MTP head baked in — not a Gemma 4 sidecar.
 */
export function classifyGgufFile(file: GgufListedFile, siblings: GgufListedFile[] = []): GgufFileRole {
  if (isMmprojFileName(file.path)) {
    return "mmproj";
  }
  if (isImatrixFileName(file.path)) {
    return "imatrix";
  }
  if (isConfidentMtpSidecarPath(file.path)) {
    return "mtp-sidecar";
  }
  if (hasMtpSuffix(file.path)) {
    const pair = findMtpLanguagePair(file.path, siblings);
    if (pair) {
      if (isNearCopySize(file.size, pair.size)) {
        return "mtp-baked";
      }
      if (file.size > 0 && file.size < pair.size * 0.5) {
        return "mtp-sidecar";
      }
      // Tiny equal test fixtures fall through here as baked-in.
      return file.size > 0 && file.size + BAKED_IN_MAX_EXTRA_BYTES < pair.size
        ? "mtp-sidecar"
        : "mtp-baked";
    }
    if (file.size > 0 && file.size < UNPAIRED_SIDECAR_MAX_BYTES) {
      return "mtp-sidecar";
    }
    return "language";
  }
  return "language";
}

function tryFileSize(filePath: string): number {
  try {
    const st = fs.statSync(filePath);
    return st.isFile() ? st.size : 0;
  } catch {
    return 0;
  }
}

function listGgufStats(dir: string): GgufListedFile[] {
  let names: string[];
  try {
    names = fs.readdirSync(dir);
  } catch {
    return [];
  }
  const out: GgufListedFile[] = [];
  for (const name of names) {
    if (!name.toLowerCase().endsWith(".gguf")) {
      continue;
    }
    const full = path.join(dir, name);
    try {
      const st = fs.statSync(full);
      if (st.isFile()) {
        out.push({ path: full, size: st.size });
      }
    } catch {
      // skip unreadable
    }
  }
  return out;
}

function siblingsForPath(filePath: string): GgufListedFile[] {
  const trimmed = (filePath || "").trim();
  if (!trimmed) {
    return [];
  }
  const dir = path.dirname(trimmed);
  const here = listGgufStats(dir);
  const mtpDir = listGgufStats(path.join(dir, "MTP"));
  return here.concat(mtpDir);
}

/** True when this file should be passed as `--model-draft` (sidecar), not started as the main GGUF. */
export function isMtpSidecarFile(file: { path: string; size?: number }, siblings?: GgufListedFile[]): boolean {
  const p = (file.path || "").trim();
  if (!p) {
    return false;
  }
  if (isConfidentMtpSidecarPath(p)) {
    return true;
  }
  const size = file.size != null && file.size > 0 ? file.size : tryFileSize(p);
  const sibs = siblings ?? siblingsForPath(p);
  return classifyGgufFile({ path: p, size }, sibs) === "mtp-sidecar";
}

export function isMtpBakedInFile(file: { path: string; size?: number }, siblings?: GgufListedFile[]): boolean {
  const p = (file.path || "").trim();
  if (!p || isConfidentMtpSidecarPath(p)) {
    return false;
  }
  const size = file.size != null && file.size > 0 ? file.size : tryFileSize(p);
  const sibs = siblings ?? siblingsForPath(p);
  return classifyGgufFile({ path: p, size }, sibs) === "mtp-baked";
}

/** Draft path is a full-model duplicate of `mainPath` (ISTA `Foo` + `Foo-mtp`). */
export function isBakedInMtpNearCopy(mainPath: string, draftPath: string): boolean {
  const main = (mainPath || "").trim();
  const draft = (draftPath || "").trim();
  if (!main || !draft) {
    return false;
  }
  try {
    if (fs.existsSync(main) && fs.existsSync(draft) && fs.realpathSync(main) === fs.realpathSync(draft)) {
      return true;
    }
  } catch {
    if (path.resolve(main) === path.resolve(draft)) {
      return true;
    }
  }
  const mainSize = tryFileSize(main);
  const draftSize = tryFileSize(draft);
  const siblings = siblingsForPath(main).concat(siblingsForPath(draft));
  if (mainSize) {
    siblings.push({ path: main, size: mainSize });
  }
  if (draftSize) {
    siblings.push({ path: draft, size: draftSize });
  }
  return classifyGgufFile({ path: draft, size: draftSize }, siblings) === "mtp-baked";
}

/**
 * Pick the sidecar that matches this language GGUF's stem / quant, else the
 * usual Unsloth preference (`mtp-*` at repo root, then Q4).
 */
export function matchMtpDraftToLanguage(languagePath: string, draftPaths: string[]): string | undefined {
  if (!draftPaths.length) {
    return undefined;
  }
  const stem = mtpPairStem(languagePath);
  const exact = draftPaths.filter((p) => mtpPairStem(p) === stem && listingBaseName(p) !== listingBaseName(languagePath));
  if (exact.length) {
    return preferMtpDraftPath(exact);
  }
  const quant = extractQuantToken(languagePath);
  if (quant) {
    const qMatch = draftPaths.filter((p) => extractQuantToken(p) === quant);
    if (qMatch.length) {
      return preferMtpDraftPath(qMatch);
    }
  }
  return preferMtpDraftPath(draftPaths);
}

/** MTP mode that loads a separate GGUF via `--model-draft` (Gemma 4), not baked-in next-n heads. */
export function usesSidecarMtp(settings: {
  speculativeMode?: string;
  draftModelPath?: string;
}): boolean {
  const draft = (settings.draftModelPath || "").trim();
  return speculativeUsesMtp(settings.speculativeMode) && isMtpSidecarFile({ path: draft });
}

/**
 * Pick the best sidecar MTP GGUF. Prefers the repo-root `mtp-*.gguf` llama.cpp
 * `-hf` auto-discovers (Unsloth's recommended smart Q4_0), then Q4, Q8, F16.
 */
export function preferMtpDraftPath(paths: string[]): string | undefined {
  if (!paths.length) {
    return undefined;
  }
  const score = (p: string): number => {
    const base = listingBaseName(p).toLowerCase();
    const inMtpDir = listingParentName(p).toLowerCase() === "mtp";
    const rootPrefixed = /^mtp[-_]/.test(base);
    let quant = 0;
    if (base.includes("q4")) {
      quant = 1;
    } else if (base.includes("q8")) {
      quant = 2;
    } else if (base.includes("q6") || base.includes("q5")) {
      quant = 3;
    } else if (base.includes("bf16") || (/\bf16\b/.test(base) && !base.includes("bf16"))) {
      quant = 4;
    } else if (base.includes("f32")) {
      quant = 5;
    }
    const location = rootPrefixed && !inMtpDir ? 0 : inMtpDir ? 2 : 1;
    return location * 10 + quant;
  };
  return [...paths].sort((a, b) => score(a) - score(b) || a.localeCompare(b))[0];
}

/**
 * First usable sidecar MTP GGUF next to a language model, including an `MTP/`
 * subdirectory (Unsloth's extra Q4_0 / Q8_0 / BF16 drafters). Skips ISTA-style
 * full-model `Foo-mtp.gguf` near-copies.
 */
/**
 * llama.cpp cannot load Unsloth's Flash-Next `mtp-*.gguf` as `--model-draft`
 * (`output_hc_norm.weight` missing). qwen4exp MTP stays off until the main
 * GGUF reports next-n layers.
 */
export function languageRejectsSidecarMtp(modelPath: string): boolean {
  const h = (modelPath || "").toLowerCase();
  return (
    /qwen4exp/.test(h) ||
    /qwen4_exp/.test(h) ||
    /qwen4-exp/.test(h) ||
    /qwen3\.8[-_.]?flash/.test(h) ||
    /qwen-?3\.8[-_.]?flash/.test(h) ||
    /qwen38[-_.]?flash/.test(h) ||
    /flash[-_]?next/.test(h)
  );
}

export function findSiblingMtpDraft(modelPath: string): string | undefined {
  const trimmed = (modelPath || "").trim();
  if (!trimmed || languageRejectsSidecarMtp(trimmed)) {
    return undefined;
  }
  const siblings = siblingsForPath(trimmed);
  const mainSize = tryFileSize(trimmed);
  if (mainSize) {
    siblings.push({ path: trimmed, size: mainSize });
  }
  let mainKey = path.resolve(trimmed);
  try {
    if (fs.existsSync(trimmed)) {
      mainKey = fs.realpathSync(trimmed);
    }
  } catch {
    // keep resolve()
  }
  const candidates = siblings.filter((s) => {
    if (s.size < MIN_MTP_BYTES) {
      return false;
    }
    let key = path.resolve(s.path);
    try {
      if (fs.existsSync(s.path)) {
        key = fs.realpathSync(s.path);
      }
    } catch {
      // keep resolve()
    }
    return key !== mainKey && classifyGgufFile(s, siblings) === "mtp-sidecar";
  });
  return matchMtpDraftToLanguage(
    trimmed,
    candidates.map((c) => c.path)
  );
}

function usableDraftPath(modelPath: string, draftPath: string): boolean {
  if (!draftPath || !fs.existsSync(draftPath)) {
    return false;
  }
  return !isBakedInMtpNearCopy(modelPath, draftPath);
}

/**
 * Sidecar MTP drafter to load with this language GGUF.
 * Switching models attaches a sibling `mtp-*.gguf` when one exists. Reloading
 * the same model keeps a manual pick (including a DFlash draft). A leftover
 * baked-in `*-mtp.gguf` near-copy is cleared so it is not passed as `--model-draft`.
 */
export function resolveMtpDraftPath(
  modelPath: string,
  current: string | undefined,
  pathChanged: boolean,
  speculativeMode?: string
): string {
  const existing = (current || "").trim();
  const sibling = findSiblingMtpDraft(modelPath);

  if (!pathChanged) {
    if (usableDraftPath(modelPath, existing)) {
      return existing;
    }
    if (existing && (isMtpDraftFileName(existing) || isBakedInMtpNearCopy(modelPath, existing))) {
      return sibling || "";
    }
    return existing || "";
  }

  if (sibling) {
    return sibling;
  }
  if (speculativeUsesDflash(speculativeMode) && usableDraftPath(modelPath, existing) && !isMtpSidecarFile({ path: existing })) {
    return existing;
  }
  return "";
}

const SKIP_DIR_NAMES = new Set([
  ".git",
  "node_modules",
  "__pycache__",
  "blobs", // HF content-addressed store; named .gguf live under snapshots/
]);

function expand(p: string): string {
  let out = p.trim();
  // Expand %VAR% (Windows) and $VAR / ${VAR} lightly for custom dirs.
  out = out.replace(/%([^%]+)%/g, (_, name: string) => process.env[name] || process.env[name.toUpperCase()] || "");
  if (out.startsWith("~/") || out.startsWith("~\\") || out === "~") {
    out = path.join(os.homedir(), out.slice(2));
  }
  return out;
}

function readLmStudioDownloadsFolder(): string | undefined {
  const candidates = [
    path.join(os.homedir(), ".lmstudio", "settings.json"),
    ...(process.env.APPDATA
      ? [path.join(process.env.APPDATA, "LM Studio", "settings.json")]
      : []),
  ];
  for (const settingsPath of candidates) {
    try {
      const raw = JSON.parse(fs.readFileSync(settingsPath, "utf8")) as {
        downloadsFolder?: string;
      };
      const folder = (raw.downloadsFolder || "").trim();
      if (folder) {
        return expand(folder);
      }
    } catch {
      // try next
    }
  }
  return undefined;
}

function hfHubCacheDirs(): string[] {
  const dirs: string[] = [];
  const hub = (process.env.HF_HUB_CACHE || "").trim();
  if (hub) {
    dirs.push(expand(hub));
  }
  const hfHome = (process.env.HF_HOME || "").trim();
  if (hfHome) {
    dirs.push(path.join(expand(hfHome), "hub"));
  }
  const xdg = (process.env.XDG_CACHE_HOME || "").trim();
  const cacheRoot = xdg ? expand(xdg) : path.join(os.homedir(), ".cache");
  dirs.push(path.join(cacheRoot, "huggingface", "hub"));
  dirs.push(path.join(os.homedir(), ".cache", "huggingface", "hub"));
  if (process.env.LOCALAPPDATA) {
    dirs.push(path.join(process.env.LOCALAPPDATA, "huggingface", "hub"));
  }
  return dirs;
}

/** Default / well-known GGUF roots used by common local-LLM tools. */
export function discoverModelRoots(config: ConfigAccessor): ScanRoot[] {
  const home = os.homedir();
  const localApp = process.env.LOCALAPPDATA || "";
  const appData = process.env.APPDATA || "";
  const roots: ScanRoot[] = [{ dir: getModelsDir(config), source: "Llama AIO" }];

  const lmCustom = readLmStudioDownloadsFolder();
  if (lmCustom) {
    roots.push({ dir: lmCustom, source: "LM Studio" });
  }
  roots.push(
    { dir: path.join(home, ".lmstudio", "models"), source: "LM Studio" },
    { dir: path.join(home, ".cache", "lm-studio", "models"), source: "LM Studio" }
  );
  if (localApp) {
    roots.push({ dir: path.join(localApp, "LM Studio", "models"), source: "LM Studio" });
  }

  roots.push(
    { dir: path.join(home, ".unsloth", "studio", "cache", "huggingface", "hub"), source: "Unsloth Studio" },
    { dir: path.join(home, ".unsloth", "studio", "models"), source: "Unsloth Studio" },
    { dir: path.join(home, ".local", "share", "unsloth"), source: "Unsloth Studio" },
    // Avoid scanning entire ~/.unsloth (contains llama.cpp vocab stubs).
    { dir: path.join(home, ".unsloth", "models"), source: "Unsloth" }
  );

  for (const dir of hfHubCacheDirs()) {
    roots.push({ dir, source: "Hugging Face cache", maxDepth: 6 });
  }

  // Other common tools (Linux + Windows locations)
  roots.push(
    { dir: path.join(home, ".cache", "gpt4all", "models"), source: "GPT4All" },
    { dir: path.join(home, "Documents", "GPT4All", "Models"), source: "GPT4All" },
    { dir: path.join(home, ".local", "share", "Jan", "data", "models"), source: "Jan" },
    { dir: path.join(home, "jan", "models"), source: "Jan" },
    { dir: path.join(home, "text-generation-webui", "models"), source: "text-generation-webui" },
    { dir: path.join(home, "oobabooga", "text-generation-webui", "models"), source: "text-generation-webui" },
    { dir: path.join(home, ".cache", "llama.cpp"), source: "llama.cpp", maxDepth: 4 },
    { dir: path.join(home, ".ollama", "models", "manifests"), source: "Ollama", maxDepth: 2 } // rarely has .gguf
  );
  if (localApp) {
    roots.push(
      { dir: path.join(localApp, "nomic.ai", "GPT4All"), source: "GPT4All" },
      { dir: path.join(localApp, "jan", "data", "models"), source: "Jan" },
      { dir: path.join(localApp, "llama.cpp"), source: "llama.cpp", maxDepth: 4 }
    );
  }
  if (appData) {
    roots.push(
      { dir: path.join(appData, "jan", "data", "models"), source: "Jan" },
      { dir: path.join(appData, "GPT4All", "Models"), source: "GPT4All" }
    );
  }

  const extra = config.get<string[]>("extraModelDirs") || [];
  for (const d of extra) {
    const trimmed = (d || "").trim();
    if (trimmed) {
      roots.push({ dir: expand(trimmed), source: "Custom" });
    }
  }

  // Deduplicate by resolved path; first source wins.
  const seen = new Set<string>();
  const unique: ScanRoot[] = [];
  for (const r of roots) {
    let key: string;
    try {
      key = fs.existsSync(r.dir) ? fs.realpathSync(r.dir) : path.resolve(r.dir);
    } catch {
      key = path.resolve(r.dir);
    }
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    unique.push(r);
  }
  return unique;
}

function shouldSkipLanguageFile(name: string, size: number, siblings: GgufListedFile[] = []): boolean {
  if (!name.toLowerCase().endsWith(".gguf")) {
    return true;
  }
  if (SKIP_NAME_RE.test(name) || isMmprojFileName(name) || isImatrixFileName(name)) {
    return true;
  }
  if (classifyGgufFile({ path: name, size }, siblings) === "mtp-sidecar") {
    return true;
  }
  if (size < MIN_MODEL_BYTES) {
    return true;
  }
  if (name.endsWith(".partial") || name.endsWith(".incomplete") || name.endsWith(".tmp")) {
    return true;
  }
  return false;
}

function shouldSkipMmprojFile(name: string, size: number, _siblings: GgufListedFile[] = []): boolean {
  if (!name.toLowerCase().endsWith(".gguf") || !isMmprojFileName(name)) {
    return true;
  }
  if (size < MIN_MMPROJ_BYTES) {
    return true;
  }
  if (name.endsWith(".partial") || name.endsWith(".incomplete") || name.endsWith(".tmp")) {
    return true;
  }
  return false;
}

function shouldSkipMtpDraftFile(name: string, size: number, siblings: GgufListedFile[] = []): boolean {
  if (!name.toLowerCase().endsWith(".gguf")) {
    return true;
  }
  if (classifyGgufFile({ path: name, size }, siblings) !== "mtp-sidecar") {
    return true;
  }
  if (size < MIN_MTP_BYTES) {
    return true;
  }
  if (name.endsWith(".partial") || name.endsWith(".incomplete") || name.endsWith(".tmp")) {
    return true;
  }
  return false;
}

function walkGgufs(
  root: ScanRoot,
  out: Map<string, LocalModelEntry>,
  skip: (name: string, size: number, siblings: GgufListedFile[]) => boolean
): void {
  if (!fs.existsSync(root.dir)) {
    return;
  }
  const maxDepth = root.maxDepth ?? 8;
  const stack: Array<{ dir: string; depth: number }> = [{ dir: root.dir, depth: 0 }];

  while (stack.length) {
    const { dir, depth } = stack.pop()!;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    const siblingStats: GgufListedFile[] = [];
    for (const ent of entries) {
      if (!ent.isFile() && !ent.isSymbolicLink()) {
        continue;
      }
      if (!ent.name.toLowerCase().endsWith(".gguf")) {
        continue;
      }
      try {
        const st = fs.statSync(path.join(dir, ent.name));
        if (st.isFile()) {
          siblingStats.push({ path: ent.name, size: st.size });
        }
      } catch {
        // skip unreadable
      }
    }
    for (const ent of entries) {
      if (ent.name.startsWith(".") && ent.name !== ".cache") {
        // skip hidden except we already handle .cache dirs as roots
        if (ent.isDirectory()) {
          continue;
        }
      }
      const full = path.join(dir, ent.name);
      if (ent.isDirectory()) {
        if (SKIP_DIR_NAMES.has(ent.name)) {
          continue;
        }
        if (depth + 1 <= maxDepth) {
          stack.push({ dir: full, depth: depth + 1 });
        }
        continue;
      }
      if (!ent.isFile() && !ent.isSymbolicLink()) {
        continue;
      }
      try {
        const st = fs.statSync(full);
        if (!st.isFile() || skip(ent.name, st.size, siblingStats)) {
          continue;
        }
        // Dedupe by real path (HF snapshots often symlink into blobs/) but keep the
        // user-facing .gguf path for loading / display.
        let dedupeKey = full;
        try {
          dedupeKey = fs.realpathSync(full);
        } catch {
          // keep full
        }
        if (out.has(dedupeKey)) {
          continue;
        }
        out.set(dedupeKey, {
          path: full,
          source: root.source,
          sizeBytes: st.size,
        });
      } catch {
        // ignore unreadable
      }
    }
  }
}

const SHARD_NAME_RE = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/i;

/**
 * Group key for a split GGUF on disk (`Foo-00002-of-00005.gguf` → dirname + stem).
 * Same idea as Hugging Face `splitGgufGroupKey`, but for OS paths.
 */
export function localShardGroupKey(filePath: string): string | undefined {
  const base = listingBaseName(filePath);
  const m = SHARD_NAME_RE.exec(base);
  if (!m) {
    return undefined;
  }
  const total = Number(m[3]);
  if (!Number.isFinite(total) || total < 1 || total > 999) {
    return undefined;
  }
  return path.join(path.dirname(filePath), m[1]!.toLowerCase());
}

/** Drop `-00001-of-00005` so a split set shows as one model name. */
export function displayGgufTitle(filePath: string): string {
  const base = listingBaseName(filePath);
  const m = SHARD_NAME_RE.exec(base);
  const stem = m?.[1] || base.replace(/\.gguf$/i, "");
  return stem;
}

/** GGUF `general.name` values that are content hashes, not titles. */
const HASH_LIKE_MODEL_NAME = /^[0-9a-f]{12,}$/i;

export function isHashLikeModelName(name: string | undefined): boolean {
  return HASH_LIKE_MODEL_NAME.test((name || "").trim());
}

/**
 * Sidebar / toast title. Prefer `general.name`, but a hash-like value
 * ("Dec5630A5621B75…") falls back to the file name.
 */
export function displayModelTitle(name: string | undefined, filePath?: string): string {
  const trimmed = (name || "").trim();
  const fromFile = filePath ? displayGgufTitle(filePath) : "";
  if (!trimmed || isHashLikeModelName(trimmed)) {
    return fromFile || trimmed;
  }
  return trimmed;
}

const QUANT_TOKEN = /^(I?Q\d(?:_[A-Z0-9]+)*|F16|BF16|F32|MXFP4(?:_MOE)?|TQ\d_\d)$/i;
const SIZE_TOKEN = /^\d+(?:\.\d+)?[BM](?:-?A\d+(?:\.\d+)?B)?$/i;

/**
 * "family size · quant" for the sidebar header, e.g.
 * `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` → "Swift 1.5 Qwen3.8 27B · IQ3_XXS".
 * Keeps `general.name` when it already carries a parameter count; otherwise
 * the file name is usually the more informative of the two.
 */
export function friendlyModelTitle(name: string | undefined, filePath?: string): string {
  const fallback = displayModelTitle(name, filePath);
  const stem = filePath ? displayGgufTitle(filePath) : "";
  // Split on "-" only so quant tokens like IQ3_XXS stay whole.
  const parts = stem.split(/-/).filter(Boolean);
  let quant = parts.find((p) => QUANT_TOKEN.test(p)) || "";
  if (!quant) {
    const m = /(?:^|[-_.])(I?Q\d(?:_[A-Z0-9]+)+|MXFP4|BF16|F16)(?=$|[-_.])/i.exec(stem);
    quant = m?.[1] || "";
  }
  const trimmedName = (name || "").trim();
  const nameHasSize = /\b\d+(?:\.\d+)?\s?[BM]\b/i.test(trimmedName);
  if (trimmedName && !isHashLikeModelName(trimmedName) && nameHasSize) {
    return quant && !trimmedName.toLowerCase().includes(quant.toLowerCase())
      ? `${trimmedName} · ${quant.toUpperCase()}`
      : trimmedName;
  }
  const sizeIdx = parts.findIndex((p) => SIZE_TOKEN.test(p));
  if (sizeIdx <= 0) {
    return fallback;
  }
  // "30B-A3B" arrives as two dash parts.
  let size = parts[sizeIdx]!;
  if (/^A\d+(?:\.\d+)?B$/i.test(parts[sizeIdx + 1] || "")) {
    size = `${size}-${parts[sizeIdx + 1]}`;
  }
  const family = parts.slice(0, sizeIdx).join(" ");
  return quant ? `${family} ${size} · ${quant.toUpperCase()}` : `${family} ${size}`;
}

/**
 * One library row per split GGUF. Points at the first shard; `sizeBytes` is the
 * sum of every part we found (B-38).
 */
export function collapseLocalModelEntries(entries: LocalModelEntry[]): LocalModelEntry[] {
  const groups = new Map<string, LocalModelEntry[]>();
  const singles: LocalModelEntry[] = [];
  for (const e of entries) {
    const stem = localShardGroupKey(e.path);
    if (!stem) {
      singles.push(e);
      continue;
    }
    const key = `${e.source}\0${stem}`;
    const group = groups.get(key) || [];
    group.push(e);
    groups.set(key, group);
  }
  const collapsed = [...groups.values()].map((group) => {
    const sorted = [...group].sort((a, b) => listingBaseName(a.path).localeCompare(listingBaseName(b.path)));
    const first = sorted[0]!;
    return {
      ...first,
      sizeBytes: group.reduce((sum, part) => sum + (part.sizeBytes || 0), 0),
      shardCount: group.length,
    };
  });
  return [...singles, ...collapsed];
}

/**
 * Library scans walk ~25 candidate roots (LM Studio, HF cache, Jan, …) and
 * stat every GGUF. The sidebar asks for the list on every refresh, so hold the
 * result briefly; anything that adds or removes a model calls
 * {@link invalidateModelLibraryCache}.
 */
export const MODEL_LIBRARY_CACHE_TTL_MS = 10_000;
type LibraryScanKind = "language" | "mmproj" | "mtp-draft";
const libraryCache = new Map<string, { at: number; entries: LocalModelEntry[] }>();

/** Drop cached library scans (after downloads, imports, deletions, or a manual refresh). */
export function invalidateModelLibraryCache(): void {
  libraryCache.clear();
}

function libraryCacheKey(kind: LibraryScanKind, config: ConfigAccessor): string {
  const extra = (config.get<string[]>("extraModelDirs") || []).map((d) => (d || "").trim());
  return [kind, getModelsDir(config), ...extra].join("\u0000");
}

function cachedScan(
  kind: LibraryScanKind,
  config: ConfigAccessor,
  scan: () => LocalModelEntry[]
): LocalModelEntry[] {
  const key = libraryCacheKey(kind, config);
  const now = Date.now();
  const hit = libraryCache.get(key);
  if (hit && now - hit.at < MODEL_LIBRARY_CACHE_TTL_MS) {
    return hit.entries.map((e) => ({ ...e }));
  }
  const entries = scan();
  libraryCache.set(key, { at: now, entries });
  return entries.map((e) => ({ ...e }));
}

/** Scan Llama AIO library + common third-party download folders for GGUF models. */
export function listLocalModelEntries(config: ConfigAccessor): LocalModelEntry[] {
  return cachedScan("language", config, () => scanLocalModelEntries(config));
}

function scanLocalModelEntries(config: ConfigAccessor): LocalModelEntry[] {
  const map = new Map<string, LocalModelEntry>();
  for (const root of discoverModelRoots(config)) {
    walkGgufs(root, map, shouldSkipLanguageFile);
  }
  return collapseLocalModelEntries([...map.values()]).sort((a, b) => {
    if (a.source !== b.source) {
      // Llama AIO first, then alpha by source, then name
      if (a.source === "Llama AIO") {
        return -1;
      }
      if (b.source === "Llama AIO") {
        return 1;
      }
      return a.source.localeCompare(b.source) || path.basename(a.path).localeCompare(path.basename(b.path));
    }
    return path.basename(a.path).localeCompare(path.basename(b.path));
  });
}

/** Same roots as listLocalModelEntries, but only vision projector GGUFs. */
export function listMmprojEntries(config: ConfigAccessor): LocalModelEntry[] {
  return cachedScan("mmproj", config, () => {
    const map = new Map<string, LocalModelEntry>();
    for (const root of discoverModelRoots(config)) {
      walkGgufs(root, map, shouldSkipMmprojFile);
    }
    return [...map.values()].sort((a, b) => path.basename(a.path).localeCompare(path.basename(b.path)));
  });
}

/** Same roots as listLocalModelEntries, but only sidecar MTP draft GGUFs. */
export function listMtpDraftEntries(config: ConfigAccessor): LocalModelEntry[] {
  return cachedScan("mtp-draft", config, () => {
    const map = new Map<string, LocalModelEntry>();
    for (const root of discoverModelRoots(config)) {
      walkGgufs(root, map, shouldSkipMtpDraftFile);
    }
    return [...map.values()].sort((a, b) => path.basename(a.path).localeCompare(path.basename(b.path)));
  });
}

/**
 * For each source that currently has GGUFs, pick a folder to open in the OS
 * file manager (first existing scan root for that source).
 */
export function listActiveModelSourceDirs(
  config: ConfigAccessor,
  entries = listLocalModelEntries(config)
): Array<{ source: string; dir: string }> {
  const sources = [...new Set(entries.map((e) => e.source))];
  const roots = discoverModelRoots(config);
  const out: Array<{ source: string; dir: string }> = [];
  for (const source of sources) {
    const root = roots.find((r) => r.source === source && fs.existsSync(r.dir));
    if (root) {
      out.push({ source, dir: root.dir });
      continue;
    }
    // Fallback: parent of first model file for that source.
    const sample = entries.find((e) => e.source === source);
    if (sample) {
      out.push({ source, dir: path.dirname(sample.path) });
    }
  }
  return out;
}

export function formatModelSize(n: number): string {
  if (n < 1024 * 1024) {
    return `${(n / 1024).toFixed(1)} KB`;
  }
  if (n < 1024 * 1024 * 1024) {
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

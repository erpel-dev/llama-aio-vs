import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import type { ConfigAccessor } from "./config";
import { getModelsDir } from "./paths";

export interface LocalModelEntry {
  path: string;
  /** Display source, e.g. "LM Studio", "Llama AIO" */
  source: string;
  sizeBytes: number;
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

/** True for llama.cpp vision projector GGUFs (`mmproj-F16.gguf`, …). */
export function isMmprojFileName(name: string): boolean {
  return /mmproj/i.test(path.basename(name));
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
 * True for llama.cpp sidecar MTP drafters (`mtp-gemma-4-12B-it.gguf`,
 * `gemma-4-12B-it-Q4_0-MTP.gguf`). Not DFlash (`architecture = dflash`).
 */
export function isMtpDraftFileName(name: string): boolean {
  const base = path.basename(name);
  if (!base.toLowerCase().endsWith(".gguf") || isMmprojFileName(base)) {
    return false;
  }
  return /^mtp[-_]/i.test(base) || /-mtp\.gguf$/i.test(base);
}

/** MTP mode that loads a separate GGUF via `--model-draft` (Gemma 4), not baked-in next-n heads. */
export function usesSidecarMtp(settings: {
  speculativeMode?: string;
  draftModelPath?: string;
}): boolean {
  return settings.speculativeMode === "mtp" && isMtpDraftFileName(settings.draftModelPath || "");
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
    const base = path.basename(p).toLowerCase();
    const inMtpDir = path.basename(path.dirname(p)).toLowerCase() === "mtp";
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

function collectMtpDraftsInDir(dir: string, found: string[]): void {
  let names: string[];
  try {
    names = fs.readdirSync(dir);
  } catch {
    return;
  }
  for (const name of names) {
    if (!name.toLowerCase().endsWith(".gguf") || !isMtpDraftFileName(name)) {
      continue;
    }
    const full = path.join(dir, name);
    try {
      const st = fs.statSync(full);
      if (st.isFile() && st.size >= MIN_MTP_BYTES) {
        found.push(full);
      }
    } catch {
      // skip unreadable
    }
  }
}

/**
 * First usable sidecar MTP GGUF next to a language model, including an `MTP/`
 * subdirectory (Unsloth's extra Q4_0 / Q8_0 / BF16 drafters).
 */
export function findSiblingMtpDraft(modelPath: string): string | undefined {
  const trimmed = (modelPath || "").trim();
  if (!trimmed) {
    return undefined;
  }
  const dir = path.dirname(trimmed);
  const found: string[] = [];
  collectMtpDraftsInDir(dir, found);
  collectMtpDraftsInDir(path.join(dir, "MTP"), found);
  return preferMtpDraftPath(found);
}

/**
 * Sidecar MTP drafter to load with this language GGUF.
 * Switching models attaches a sibling `mtp-*.gguf` when one exists. Reloading
 * the same model keeps a manual pick (including a DFlash draft).
 */
export function resolveMtpDraftPath(
  modelPath: string,
  current: string | undefined,
  pathChanged: boolean,
  speculativeMode?: string
): string {
  const existing = (current || "").trim();
  if (!pathChanged) {
    if (existing && fs.existsSync(existing)) {
      return existing;
    }
    if (existing && isMtpDraftFileName(existing)) {
      return findSiblingMtpDraft(modelPath) || "";
    }
    return existing || "";
  }
  const sibling = findSiblingMtpDraft(modelPath);
  if (sibling) {
    return sibling;
  }
  if (
    speculativeMode === "dflash" &&
    existing &&
    !isMtpDraftFileName(existing) &&
    fs.existsSync(existing)
  ) {
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

function shouldSkipLanguageFile(name: string, size: number): boolean {
  if (!name.toLowerCase().endsWith(".gguf")) {
    return true;
  }
  if (SKIP_NAME_RE.test(name) || isMmprojFileName(name) || isMtpDraftFileName(name)) {
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

function shouldSkipMmprojFile(name: string, size: number): boolean {
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

function shouldSkipMtpDraftFile(name: string, size: number): boolean {
  if (!name.toLowerCase().endsWith(".gguf") || !isMtpDraftFileName(name)) {
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
  skip: (name: string, size: number) => boolean
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
        if (!st.isFile() || skip(ent.name, st.size)) {
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

/** Scan Llama AIO library + common third-party download folders for GGUF models. */
export function listLocalModelEntries(config: ConfigAccessor): LocalModelEntry[] {
  const map = new Map<string, LocalModelEntry>();
  for (const root of discoverModelRoots(config)) {
    walkGgufs(root, map, shouldSkipLanguageFile);
  }
  return [...map.values()].sort((a, b) => {
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
  const map = new Map<string, LocalModelEntry>();
  for (const root of discoverModelRoots(config)) {
    walkGgufs(root, map, shouldSkipMmprojFile);
  }
  return [...map.values()].sort((a, b) => path.basename(a.path).localeCompare(path.basename(b.path)));
}

/** Same roots as listLocalModelEntries, but only sidecar MTP draft GGUFs. */
export function listMtpDraftEntries(config: ConfigAccessor): LocalModelEntry[] {
  const map = new Map<string, LocalModelEntry>();
  for (const root of discoverModelRoots(config)) {
    walkGgufs(root, map, shouldSkipMtpDraftFile);
  }
  return [...map.values()].sort((a, b) => path.basename(a.path).localeCompare(path.basename(b.path)));
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

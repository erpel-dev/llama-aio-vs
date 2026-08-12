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
const SKIP_NAME_RE = /^(mmproj|ggml-vocab)/i;
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

function shouldSkipFile(name: string, size: number): boolean {
  if (!name.toLowerCase().endsWith(".gguf")) {
    return true;
  }
  if (SKIP_NAME_RE.test(name)) {
    return true;
  }
  if (name.toLowerCase().includes("mmproj")) {
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

function walkGgufs(root: ScanRoot, out: Map<string, LocalModelEntry>): void {
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
        if (!st.isFile() || shouldSkipFile(ent.name, st.size)) {
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
    walkGgufs(root, map);
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

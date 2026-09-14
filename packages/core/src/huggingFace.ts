/**
 * Hugging Face GGUF search + download (frontend-agnostic).
 * Progress uses the same ProgressReporter shape as LlamaInstaller.
 */
import * as https from "https";
import * as path from "path";
import { downloadHttpFile } from "./httpDownload";
import { shardFileNames } from "./ggufMetadata";
import {
  licenseFromModelDetail,
  licenseFromTags,
  type ModelLicenseInfo,
} from "./hfLicense";
import type { ProgressReporter } from "./llamaInstaller";
import { formatBytes } from "./memoryEstimate";
import {
  classifyGgufFile,
  invalidateModelLibraryCache,
  isMmprojFileName,
  languageRejectsSidecarMtp,
  listLocalModelEntries,
  listingBaseName,
  matchMtpDraftToLanguage,
  preferMmprojPath,
  preferMtpDraftPath,
  type GgufFileRole,
} from "./modelLibrary";
import { getModelsDir } from "./paths";
import type { SettingsStore } from "./settings";
import {
  recommendedMaxDraftTokens,
  speculativeUsesNgram,
  type HfFileHit,
  type HfModelHit,
} from "./types";

function fileRole(file: HfFileHit, files: HfFileHit[]): GgufFileRole {
  return classifyGgufFile(file, files);
}

/** Language GGUFs only — hide CLIP, imatrix dumps, sidecar MTP, and extra split shards. */
export function languageGgufFiles(files: HfFileHit[]): HfFileHit[] {
  const language = files.filter((f) => {
    const role = fileRole(f, files);
    return role === "language" || role === "mtp-baked";
  });
  return collapseSplitGgufFiles(language);
}

/** Vision projectors and sidecar MTP drafters — offered when a repo has no language GGUF. */
export function companionGgufFiles(files: HfFileHit[]): HfFileHit[] {
  const companions = files.filter((f) => {
    const role = fileRole(f, files);
    return role === "mmproj" || role === "mtp-sidecar";
  });
  return collapseSplitGgufFiles(companions);
}

export function downloadPickerFiles(files: HfFileHit[]): {
  files: HfFileHit[];
  companionOnly: boolean;
} {
  const language = languageGgufFiles(files);
  if (language.length) {
    return { files: language, companionOnly: false };
  }
  return { files: companionGgufFiles(files), companionOnly: true };
}

/** Attach a downloaded mmproj / MTP sidecar without replacing the selected language model. */
export async function attachDownloadedCompanion(
  store: SettingsStore,
  dest: string,
  remotePath: string,
  files: HfFileHit[]
): Promise<"mmproj" | "mtp-sidecar"> {
  const hit = files.find((f) => f.path === remotePath) || { path: remotePath, size: 0, url: "" };
  const role = fileRole(hit, files);
  if (role === "mtp-sidecar") {
    const cur = store.getState().loadSettings;
    const mode = speculativeUsesNgram(cur.speculativeMode) ? "ngram-mtp" : "mtp";
    await store.updateLoadSettings({
      draftModelPath: dest,
      speculativeMode: mode,
      maxDraftTokens: recommendedMaxDraftTokens(mode, cur.maxDraftTokens, true),
    });
    return "mtp-sidecar";
  }
  await store.updateLoadSettings({ mmprojPath: dest });
  return "mmproj";
}

/**
 * One picker row per split GGUF. `Foo-00001-of-00004.gguf` keeps the first
 * shard's path; `size` is the sum of every part.
 */
export function collapseSplitGgufFiles(files: HfFileHit[]): HfFileHit[] {
  const groups = new Map<string, HfFileHit[]>();
  const singles: HfFileHit[] = [];
  for (const f of files) {
    const key = splitGgufGroupKey(f.path);
    if (!key) {
      singles.push(f);
      continue;
    }
    const group = groups.get(key) || [];
    group.push(f);
    groups.set(key, group);
  }
  const collapsed = [...groups.values()].map((group) => {
    const sorted = [...group].sort((a, b) => listingBaseName(a.path).localeCompare(listingBaseName(b.path)));
    const first = sorted[0]!;
    return {
      ...first,
      size: group.reduce((sum, part) => sum + (part.size || 0), 0),
      shardCount: group.length,
    };
  });
  return [...singles, ...collapsed].sort((a, b) => a.path.localeCompare(b.path));
}

export function splitGgufGroupKey(filePath: string): string | undefined {
  const base = listingBaseName(filePath);
  const names = shardFileNames(base);
  if (!names) {
    return undefined;
  }
  const dir = posixDirname(filePath);
  const stem = /^(.*)-\d{5}-of-\d{5}\.gguf$/i.exec(base)?.[1] || base;
  return `${dir}/${stem.toLowerCase()}`.replace(/^\//, "");
}

export function posixDirname(filePath: string): string {
  const n = (filePath || "").replace(/\\/g, "/");
  const i = n.lastIndexOf("/");
  return i >= 0 ? n.slice(0, i) : "";
}

export function huggingfaceOrigin(): string {
  const raw = (
    process.env.HF_ENDPOINT ||
    process.env.HUGGING_FACE_HUB_ENDPOINT ||
    "https://huggingface.co"
  ).trim();
  return raw.replace(/\/$/, "") || "https://huggingface.co";
}

export function huggingfaceUrl(pathname: string): string {
  const pathPart = pathname.startsWith("/") ? pathname : `/${pathname}`;
  return `${huggingfaceOrigin()}${pathPart}`;
}

export function huggingfaceModelPage(modelId: string): string {
  return huggingfaceUrl(`/${modelId}`);
}

function lfsSha256(oid: string | undefined): string | undefined {
  if (!oid) {
    return undefined;
  }
  const hex = oid.replace(/^sha256:/i, "").trim().toLowerCase();
  return /^[0-9a-f]{64}$/.test(hex) ? hex : undefined;
}

export function ggufHitsFromTree(
  modelId: string,
  tree: Array<{
    path: string;
    type: string;
    size?: number;
    lfs?: { oid?: string; size?: number };
  }>
): HfFileHit[] {
  return tree
    .filter((f) => f.type === "file" && f.path.toLowerCase().endsWith(".gguf"))
    .map((f) => ({
      path: f.path,
      size: f.lfs?.size || f.size || 0,
      sha256: lfsSha256(f.lfs?.oid),
      url: huggingfaceUrl(`/${modelId}/resolve/main/${f.path}`),
    }))
    .sort((a, b) => a.path.localeCompare(b.path));
}

/** Remote paths for every shard of a split GGUF, or just `filePath`. */
export function remoteShardPaths(filePath: string): string[] {
  const base = listingBaseName(filePath);
  const names = shardFileNames(base);
  if (!names) {
    return [filePath];
  }
  const dir = posixDirname(filePath);
  return names.map((name) => (dir ? `${dir}/${name}` : name));
}

export function describeLanguageGgufFile(file: HfFileHit, all: HfFileHit[]): string {
  const role = fileRole(file, all);
  const bits = [formatBytes(file.size)];
  const shards = file.shardCount || shardFileNames(listingBaseName(file.path))?.length || 1;
  if (shards > 1) {
    bits.push(`${shards} shards`);
  }
  if (role === "mtp-baked") {
    bits.push("MTP head included");
  }
  if (role === "mmproj") {
    bits.push("vision projector");
  }
  if (role === "mtp-sidecar") {
    bits.push("MTP draft");
  }
  return bits.join(" · ");
}

/** Extra files auto-fetched with a language GGUF (mmproj + confirmed sidecars only). */
export function companionDownloadHint(files: HfFileHit[]): string {
  const extras: string[] = [];
  const mm = preferredMmprojFile(files);
  if (mm) {
    extras.push(listingBaseName(mm.path));
  }
  const sidecars = files.filter((f) => fileRole(f, files) === "mtp-sidecar");
  if (sidecars.length === 1) {
    extras.push(listingBaseName(sidecars[0]!.path));
  } else if (sidecars.length > 1) {
    extras.push("matching MTP sidecar");
  }
  return extras.join(", ");
}

/** Best mmproj in a Hugging Face file listing, if any. */
export function preferredMmprojFile(files: HfFileHit[]): HfFileHit | undefined {
  const mm = files.filter((f) => isMmprojFileName(f.path));
  const best = preferMmprojPath(mm.map((f) => f.path));
  return best ? mm.find((f) => f.path === best) : undefined;
}

/** Best sidecar MTP drafter in a Hugging Face file listing, if any. */
export function preferredMtpDraftFile(
  files: HfFileHit[],
  languagePath?: string
): HfFileHit | undefined {
  if (languagePath && languageRejectsSidecarMtp(languagePath)) {
    return undefined;
  }
  const mtp = files.filter((f) => fileRole(f, files) === "mtp-sidecar");
  if (!mtp.length) {
    return undefined;
  }
  const best = languagePath
    ? matchMtpDraftToLanguage(
        languagePath,
        mtp.map((f) => f.path)
      )
    : preferMtpDraftPath(mtp.map((f) => f.path));
  return best ? mtp.find((f) => f.path === best) : undefined;
}

const MAX_TREE_PAGES = 20;

/** RFC 8288 `Link: <url>; rel="next"` used by the Hugging Face tree API. */
export function parseNextLink(linkHeader: string | string[] | undefined): string | undefined {
  const header = Array.isArray(linkHeader) ? linkHeader.join(",") : linkHeader;
  if (!header) {
    return undefined;
  }
  for (const part of header.split(",")) {
    const m = /<([^>]+)>\s*;\s*rel=["']?next["']?/i.exec(part);
    if (m?.[1]) {
      return m[1];
    }
  }
  return undefined;
}

function hfHeaders(token?: string): Record<string, string> {
  const headers: Record<string, string> = {
    "User-Agent": "llama-aio-vs",
    Accept: "application/json",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

function requestJson<T>(url: string, token?: string): Promise<T> {
  return new Promise((resolve, reject) => {
    https
      .get(url, { headers: hfHeaders(token) }, (res) => {
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          requestJson<T>(res.headers.location, token).then(resolve, reject);
          res.resume();
          return;
        }
        const chunks: Buffer[] = [];
        res.on("data", (c) => chunks.push(c));
        res.on("end", () => {
          const body = Buffer.concat(chunks).toString("utf8");
          if (!res.statusCode || res.statusCode >= 400) {
            reject(new Error(`HF HTTP ${res.statusCode}: ${body.slice(0, 200)}`));
            return;
          }
          try {
            resolve(JSON.parse(body) as T);
          } catch (e) {
            reject(e);
          }
        });
      })
      .on("error", reject);
  });
}

/** Concatenate a Hugging Face paginated JSON array (`Link: rel="next"`). */
function requestJsonPages<T>(url: string, token?: string): Promise<T[]> {
  return new Promise((resolve, reject) => {
    const all: T[] = [];
    const go = (u: string, page: number) => {
      if (page > MAX_TREE_PAGES) {
        resolve(all);
        return;
      }
      https
        .get(u, { headers: hfHeaders(token) }, (res) => {
          if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            go(res.headers.location, page);
            res.resume();
            return;
          }
          const chunks: Buffer[] = [];
          res.on("data", (c) => chunks.push(c));
          res.on("end", () => {
            const body = Buffer.concat(chunks).toString("utf8");
            if (!res.statusCode || res.statusCode >= 400) {
              reject(new Error(`HF HTTP ${res.statusCode}: ${body.slice(0, 200)}`));
              return;
            }
            try {
              const parsed = JSON.parse(body) as T[];
              if (Array.isArray(parsed)) {
                all.push(...parsed);
              }
            } catch (e) {
              reject(e);
              return;
            }
            const next = parseNextLink(res.headers.link);
            if (next) {
              go(next, page + 1);
              return;
            }
            resolve(all);
          });
        })
        .on("error", reject);
    };
    go(url, 1);
  });
}

export class HuggingFaceClient {
  constructor(private readonly store: SettingsStore) {}

  private token(): string | undefined {
    const t = (this.store.getConfig().get<string>("hfToken") || "").trim();
    return t || undefined;
  }

  async searchGgufModels(query: string, limit = 25): Promise<HfModelHit[]> {
    const q = encodeURIComponent(query.trim() || "gguf");
    const url = huggingfaceUrl(`/api/models?search=${q}&filter=gguf&sort=downloads&direction=-1&limit=${limit}`);
    return requestJson<HfModelHit[]>(url, this.token());
  }

  async getModelLicense(modelId: string, tags?: string[]): Promise<ModelLicenseInfo> {
    try {
      const url = huggingfaceUrl(`/api/models/${modelId}`);
      const detail = await requestJson<{
        id?: string;
        tags?: string[];
        cardData?: { license?: string; license_name?: string; license_link?: string };
      }>(url, this.token());
      return licenseFromModelDetail(detail);
    } catch {
      return licenseFromTags(tags);
    }
  }

  async enrichLicenses(models: HfModelHit[]): Promise<Map<string, ModelLicenseInfo>> {
    const out = new Map<string, ModelLicenseInfo>();
    const needDetail: HfModelHit[] = [];
    for (const m of models) {
      const fromTag = licenseFromTags(m.tags);
      const tagId = (m.tags || [])
        .map((t) => /^license:(.+)$/i.exec(t)?.[1]?.toLowerCase())
        .find(Boolean);
      if (!tagId || tagId === "other" || fromTag.bucket === "unknown") {
        needDetail.push(m);
      } else {
        out.set(m.id, fromTag);
      }
    }
    const concurrency = 6;
    for (let i = 0; i < needDetail.length; i += concurrency) {
      const slice = needDetail.slice(i, i + concurrency);
      const settled = await Promise.all(
        slice.map(async (m) => {
          const info = await this.getModelLicense(m.id, m.tags);
          return [m.id, info] as const;
        })
      );
      for (const [id, info] of settled) {
        out.set(id, info);
      }
    }
    return out;
  }

  private async listRepoTree(
    modelId: string,
    options?: { recursive?: boolean; subpath?: string }
  ): Promise<Array<{ path: string; type: string; size?: number }>> {
    const suffix = options?.subpath ? `/${options.subpath}` : "";
    const query = options?.recursive ? "?recursive=true" : "";
    const url = huggingfaceUrl(`/api/models/${modelId}/tree/main${suffix}${query}`);
    return requestJsonPages(url, this.token());
  }

  async listGgufFiles(modelId: string): Promise<HfFileHit[]> {
    let tree: Array<{ path: string; type: string; size?: number }>;
    try {
      tree = await this.listRepoTree(modelId, { recursive: true });
    } catch {
      tree = await this.listRepoTree(modelId);
    }
    return ggufHitsFromTree(modelId, tree);
  }

  async downloadModelFile(
    modelId: string,
    filePath: string,
    progress?: ProgressReporter,
    files?: HfFileHit[]
  ): Promise<string> {
    const parts = remoteShardPaths(filePath);
    let primary = "";
    for (let i = 0; i < parts.length; i++) {
      const rel = parts[i]!;
      if (parts.length > 1) {
        progress?.report({
          message: `Downloading shard ${i + 1}/${parts.length} ${listingBaseName(rel)}…`,
        });
      }
      const meta = files?.find((f) => f.path === rel);
      const dest = await this.downloadOneFile(modelId, rel, progress, meta);
      if (!primary || listingBaseName(rel) === listingBaseName(filePath)) {
        primary = dest;
      }
    }
    return primary;
  }

  localDestFor(modelId: string, filePath: string): string {
    const modelsDir = getModelsDir(this.store.getConfig());
    const destDir = path.join(modelsDir, modelId.replace(/\//g, "__"));
    return path.join(destDir, path.basename(filePath));
  }

  private async downloadOneFile(
    modelId: string,
    filePath: string,
    progress?: ProgressReporter,
    meta?: Pick<HfFileHit, "size" | "sha256">
  ): Promise<string> {
    const dest = this.localDestFor(modelId, filePath);
    const url = huggingfaceUrl(`/${modelId}/resolve/main/${filePath}`);
    progress?.report({ message: `Downloading ${filePath}…` });
    await downloadHttpFile({
      url,
      dest,
      token: this.token(),
      expectedSize: meta?.size && meta.size > 0 ? meta.size : undefined,
      expectedSha256: meta?.sha256,
      gatedPageUrl: huggingfaceModelPage(modelId),
      onProgress: (info) => {
        const pct = info.total > 0 ? Math.floor((info.received / info.total) * 100) : 0;
        progress?.report({
          message: `Downloading ${path.basename(filePath)}… ${pct}% (${formatBytes(info.received)}${
            info.total ? " / " + formatBytes(info.total) : ""
          })`,
        });
      },
    });
    invalidateModelLibraryCache();
    return dest;
  }

  /**
   * Download the preferred sibling vision projector from a repo file listing.
   * No-op when the repo has no mmproj GGUF.
   */
  async downloadPreferredMmproj(
    modelId: string,
    files: HfFileHit[],
    progress?: ProgressReporter
  ): Promise<string | undefined> {
    const file = preferredMmprojFile(files);
    if (!file) {
      return undefined;
    }
    return this.downloadModelFile(modelId, file.path, progress, files);
  }

  /**
   * Download the preferred sibling MTP drafter from a repo file listing.
   * No-op when the repo has no `mtp-*.gguf` / `*-MTP.gguf`.
   */
  async downloadPreferredMtpDraft(
    modelId: string,
    files: HfFileHit[],
    progress?: ProgressReporter,
    languagePath?: string
  ): Promise<string | undefined> {
    const file = preferredMtpDraftFile(files, languagePath);
    if (!file) {
      return undefined;
    }
    return this.downloadModelFile(modelId, file.path, progress, files);
  }

  /** Paths only — prefer listLocalModelEntries() for source labels. */
  listLocalModels(): string[] {
    return listLocalModelEntries(this.store.getConfig()).map((e) => e.path);
  }
}

/** Curated first-run GGUF — small, public, widely supported. */
export const STARTER_MODEL = {
  repoId: "unsloth/Qwen3.5-2B-GGUF",
  filePath: "Qwen3.5-2B-Q4_K_M.gguf",
  label: "Qwen3.5-2B",
  approxSizeLabel: "~1.6 GB",
  detail: "~2B · Q4_K_M · good for first run",
} as const;

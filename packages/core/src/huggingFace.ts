/**
 * Hugging Face GGUF search + download (frontend-agnostic).
 * Progress uses the same ProgressReporter shape as LlamaInstaller.
 */
import * as fs from "fs";
import * as https from "https";
import * as path from "path";
import {
  licenseFromModelDetail,
  licenseFromTags,
  type ModelLicenseInfo,
} from "./hfLicense";
import type { ProgressReporter } from "./llamaInstaller";
import { formatBytes } from "./memoryEstimate";
import { listLocalModelEntries } from "./modelLibrary";
import { ensureDirs, getModelsDir } from "./paths";
import type { SettingsStore } from "./settings";
import type { HfFileHit, HfModelHit } from "./types";

function requestJson<T>(url: string, token?: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const headers: Record<string, string> = {
      "User-Agent": "llama-aio-vs",
      Accept: "application/json",
    };
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    https
      .get(url, { headers }, (res) => {
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

function downloadToFile(
  url: string,
  dest: string,
  token: string | undefined,
  onProgress?: (pct: number, received: number, total: number) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const headers: Record<string, string> = { "User-Agent": "llama-aio-vs" };
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    const go = (u: string) => {
      https
        .get(u, { headers }, (res) => {
          if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            go(res.headers.location);
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
            if (onProgress) {
              const pct = total > 0 ? Math.floor((received / total) * 100) : 0;
              onProgress(pct, received, total);
            }
          });
          res.pipe(out);
          out.on("finish", () => out.close(() => resolve()));
          out.on("error", reject);
        })
        .on("error", reject);
    };
    go(url);
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
    const url = `https://huggingface.co/api/models?search=${q}&filter=gguf&sort=downloads&direction=-1&limit=${limit}`;
    return requestJson<HfModelHit[]>(url, this.token());
  }

  async getModelLicense(modelId: string, tags?: string[]): Promise<ModelLicenseInfo> {
    try {
      const url = `https://huggingface.co/api/models/${modelId}`;
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

  async listGgufFiles(modelId: string): Promise<HfFileHit[]> {
    const url = `https://huggingface.co/api/models/${modelId}/tree/main`;
    const tree = await requestJson<Array<{ path: string; type: string; size?: number }>>(
      url,
      this.token()
    );
    return tree
      .filter((f) => f.type === "file" && f.path.toLowerCase().endsWith(".gguf"))
      .map((f) => ({
        path: f.path,
        size: f.size || 0,
        url: `https://huggingface.co/${modelId}/resolve/main/${f.path}`,
      }))
      .sort((a, b) => a.path.localeCompare(b.path));
  }

  async downloadModelFile(
    modelId: string,
    filePath: string,
    progress?: ProgressReporter
  ): Promise<string> {
    const modelsDir = getModelsDir(this.store.getConfig());
    const destDir = path.join(modelsDir, modelId.replace(/\//g, "__"));
    const dest = path.join(destDir, path.basename(filePath));
    if (fs.existsSync(dest) && fs.statSync(dest).size > 0) {
      return dest;
    }

    const url = `https://huggingface.co/${modelId}/resolve/main/${filePath}`;
    progress?.report({ message: `Downloading ${filePath}…` });
    await downloadToFile(url, dest + ".partial", this.token(), (pct, received, total) => {
      progress?.report({
        message: `Downloading ${path.basename(filePath)}… ${pct}% (${formatBytes(received)}${
          total ? " / " + formatBytes(total) : ""
        })`,
      });
    });
    fs.renameSync(dest + ".partial", dest);
    return dest;
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

import * as fs from "fs";
import * as https from "https";
import * as path from "path";
import * as vscode from "vscode";
import { listLocalModelEntries } from "./modelLibrary";
import { ensureDirs, getModelsDir } from "./paths";
import { SettingsStore } from "./settings";
import { HfFileHit, HfModelHit } from "./types";

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

function formatBytes(n: number): string {
  if (n < 1024) {
    return `${n} B`;
  }
  if (n < 1024 * 1024) {
    return `${(n / 1024).toFixed(1)} KB`;
  }
  if (n < 1024 * 1024 * 1024) {
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
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
    const results = await requestJson<HfModelHit[]>(url, this.token());
    return results;
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
    progress?: vscode.Progress<{ message?: string; increment?: number }>
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
  /** Short label for buttons. */
  label: "Qwen3.5-2B",
  /** Approx size for UI (Q4_K_M). */
  approxSizeLabel: "~1.6 GB",
  detail: "~2B · Q4_K_M · good for first run",
} as const;

/**
 * One-click download of the curated starter GGUF via the HF resolve URL
 * (no browse/search). Reuses an existing local copy if present.
 */
export async function downloadStarterModel(
  hf: HuggingFaceClient,
  store: SettingsStore
): Promise<string | undefined> {
  const dest = await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: `Llama AIO: Downloading starter ${STARTER_MODEL.label}`,
      cancellable: false,
    },
    async (progress) =>
      hf.downloadModelFile(STARTER_MODEL.repoId, STARTER_MODEL.filePath, progress)
  );

  await store.applySelectedModel(dest);
  return dest;
}

export async function browseAndDownloadModel(
  hf: HuggingFaceClient,
  store: SettingsStore
): Promise<string | undefined> {
  const query = await vscode.window.showInputBox({
    title: "Download a GGUF model from Hugging Face",
    prompt: "Search Hugging Face (GGUF). Example: qwen2.5-coder, llama-3.2, gpt-oss",
    placeHolder: "qwen2.5-coder",
    ignoreFocusOut: true,
  });
  if (query === undefined) {
    return undefined;
  }

  const models = await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Llama AIO: Searching Hugging Face…",
      cancellable: false,
    },
    async () => hf.searchGgufModels(query)
  );

  if (!models.length) {
    vscode.window.showWarningMessage("No GGUF models found for that query.");
    return undefined;
  }

  const pickedModel = await vscode.window.showQuickPick(
    models.map((m) => ({
      label: m.id,
      description: `${m.downloads?.toLocaleString?.() || m.downloads} downloads`,
      detail: (m.tags || []).slice(0, 8).join(", "),
      model: m,
    })),
    { title: "Select a Hugging Face model repo", matchOnDescription: true, matchOnDetail: true }
  );
  if (!pickedModel) {
    return undefined;
  }

  const files = await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Llama AIO: Listing GGUF files…",
      cancellable: false,
    },
    async () => hf.listGgufFiles(pickedModel.model.id)
  );

  if (!files.length) {
    vscode.window.showWarningMessage("No .gguf files found in that repo.");
    return undefined;
  }

  const pickedFile = await vscode.window.showQuickPick(
    files.map((f) => ({
      label: f.path,
      description: formatBytes(f.size),
      file: f,
    })),
    { title: "Select a GGUF file to download", matchOnDescription: true }
  );
  if (!pickedFile) {
    return undefined;
  }

  const dest = await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Llama AIO: Downloading model",
      cancellable: false,
    },
    async (progress) => hf.downloadModelFile(pickedModel.model.id, pickedFile.file.path, progress)
  );

  const state = await store.applySelectedModel(dest);
  const caps = state.modelCapabilities;
  vscode.window.showInformationMessage(
    `Model ready: ${caps?.name || dest}` +
      (caps
        ? ` (${caps.maxContextLength} max ctx, ${caps.blockCount} layers${caps.isMoe ? ", MoE" : ""})`
        : "")
  );
  return dest;
}

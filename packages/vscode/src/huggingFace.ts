/**
 * VS Code UI around the shared HuggingFaceClient (QuickPick / notifications).
 */
import * as vscode from "vscode";
import {
  formatLicenseQuickPick,
  licenseFromTags,
  resolveLicenseUrl,
  STARTER_MODEL,
  HuggingFaceClient,
  type ModelLicenseInfo,
  type SettingsStore,
} from "@llama-aio/core";

export { HuggingFaceClient, STARTER_MODEL };

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

  const licenses = await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Llama AIO: Resolving model licenses…",
      cancellable: false,
    },
    async () => hf.enrichLicenses(models)
  );

  const pickedModel = await vscode.window.showQuickPick(
    models.map((m) => {
      const license = licenses.get(m.id) || licenseFromTags(m.tags);
      const row = formatLicenseQuickPick(m.id, m.downloads, license);
      return {
        label: row.label,
        description: row.description,
        detail: (m.tags || []).filter((t) => !/^license:/i.test(t)).slice(0, 6).join(", "),
        model: m,
        license,
      };
    }),
    {
      title: "Select a Hugging Face model repo  ·  ✓ permissive · ⚠ limited · § custom",
      matchOnDescription: true,
      matchOnDetail: true,
    }
  );
  if (!pickedModel) {
    return undefined;
  }

  if (!(await confirmLicenseIfNeeded(pickedModel.model.id, pickedModel.license))) {
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
  const lic = pickedModel.license.badge;
  vscode.window.showInformationMessage(
    `Model ready: ${caps?.name || dest}` +
      (caps
        ? ` (${caps.maxContextLength} max ctx, ${caps.blockCount} layers${caps.isMoe ? ", MoE" : ""})`
        : "") +
      ` · ${lic}`
  );
  return dest;
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

/** Warn before download when the license is limited, custom, or unknown. */
async function confirmLicenseIfNeeded(
  modelId: string,
  license: ModelLicenseInfo
): Promise<boolean> {
  if (!license.needsConfirm) {
    return true;
  }
  const licenseUrl = resolveLicenseUrl(modelId, license);
  const choice = await vscode.window.showWarningMessage(
    `${modelId}\n\n${license.summary}\n\nThis is not legal advice — review the license before commercial use.`,
    { modal: true },
    "Download anyway",
    "View license"
  );
  if (choice === "View license") {
    if (licenseUrl) {
      await vscode.env.openExternal(vscode.Uri.parse(licenseUrl));
    }
    const again = await vscode.window.showWarningMessage(
      `Continue downloading ${modelId}?`,
      { modal: true },
      "Download anyway"
    );
    return again === "Download anyway";
  }
  return choice === "Download anyway";
}

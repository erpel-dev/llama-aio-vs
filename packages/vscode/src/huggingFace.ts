/**
 * VS Code UI around the shared HuggingFaceClient (QuickPick / notifications).
 */
import * as vscode from "vscode";
import {
  formatLicenseQuickPick,
  languageGgufFiles,
  licenseFromTags,
  preferredMmprojFile,
  preferredMtpDraftFile,
  isMtpDraftFileName,
  resolveLicenseUrl,
  STARTER_MODEL,
  HuggingFaceClient,
  type HfFileHit,
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
    async (progress) => {
      const modelDest = await hf.downloadModelFile(
        STARTER_MODEL.repoId,
        STARTER_MODEL.filePath,
        progress
      );
      await downloadCompanionMmproj(hf, STARTER_MODEL.repoId, progress);
      await downloadCompanionMtpDraft(hf, STARTER_MODEL.repoId, progress);
      return modelDest;
    }
  );

  await store.applySelectedModel(dest, { attachMmproj: true });
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

  const languageFiles = languageGgufFiles(files);
  if (!languageFiles.length) {
    vscode.window.showWarningMessage("That repo only has companion GGUFs (mmproj / MTP draft), not a language model.");
    return undefined;
  }

  const mmproj = preferredMmprojFile(files);
  const mtpDraft = preferredMtpDraftFile(files);
  const extras = [mmproj?.path, mtpDraft?.path].filter(Boolean).map((p) => (p as string).split("/").pop());
  const pickedFile = await vscode.window.showQuickPick(
    languageFiles.map((f) => ({
      label: f.path,
      description: formatBytes(f.size),
      file: f,
    })),
    {
      title: extras.length
        ? `Select a GGUF to download  ·  will also fetch ${extras.join(", ")}`
        : "Select a GGUF file to download",
      matchOnDescription: true,
    }
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
    async (progress) => {
      const modelDest = await hf.downloadModelFile(
        pickedModel.model.id,
        pickedFile.file.path,
        progress
      );
      await downloadCompanionMmproj(hf, pickedModel.model.id, progress, files);
      await downloadCompanionMtpDraft(hf, pickedModel.model.id, progress, files);
      return modelDest;
    }
  );

  const state = await store.applySelectedModel(dest, { attachMmproj: true });
  const caps = state.modelCapabilities;
  const vision = state.loadSettings.mmprojPath
    ? ` · vision ${state.loadSettings.mmprojPath.split(/[/\\]/).pop()}`
    : "";
  const mtp =
    isMtpDraftFileName(state.loadSettings.draftModelPath)
      ? ` · MTP ${state.loadSettings.draftModelPath.split(/[/\\]/).pop()}`
      : "";
  const lic = pickedModel.license.badge;
  vscode.window.showInformationMessage(
    `Model ready: ${caps?.name || dest}` +
      (caps
        ? ` (${caps.maxContextLength} max ctx, ${caps.blockCount} layers${caps.isMoe ? ", MoE" : ""})`
        : "") +
      vision +
      mtp +
      ` · ${lic}`
  );
  return dest;
}

async function downloadCompanionMmproj(
  hf: HuggingFaceClient,
  repoId: string,
  progress: vscode.Progress<{ message?: string; increment?: number }>,
  files?: HfFileHit[]
): Promise<string | undefined> {
  try {
    const listing = files ?? (await hf.listGgufFiles(repoId));
    const picked = preferredMmprojFile(listing);
    if (!picked) {
      return undefined;
    }
    progress.report({ message: `Downloading vision projector ${picked.path}…` });
    return await hf.downloadPreferredMmproj(repoId, listing, progress);
  } catch (e) {
    void vscode.window.showWarningMessage(
      `Model downloaded, but the vision projector failed: ${e instanceof Error ? e.message : String(e)}`
    );
    return undefined;
  }
}

async function downloadCompanionMtpDraft(
  hf: HuggingFaceClient,
  repoId: string,
  progress: vscode.Progress<{ message?: string; increment?: number }>,
  files?: HfFileHit[]
): Promise<string | undefined> {
  try {
    const listing = files ?? (await hf.listGgufFiles(repoId));
    const picked = preferredMtpDraftFile(listing);
    if (!picked) {
      return undefined;
    }
    progress.report({ message: `Downloading MTP drafter ${picked.path}…` });
    return await hf.downloadPreferredMtpDraft(repoId, listing, progress);
  } catch (e) {
    void vscode.window.showWarningMessage(
      `Model downloaded, but the MTP drafter failed: ${e instanceof Error ? e.message : String(e)}`
    );
    return undefined;
  }
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

/**
 * VS Code UI around the shared HuggingFaceClient (QuickPick / notifications).
 */
import * as vscode from "vscode";
import { DownloadPanel } from "./downloadPanel";
import {
  companionDownloadHint,
  describeLanguageGgufFile,
  downloadManager,
  formatLicenseQuickPick,
  huggingfaceModelPage,
  huggingfaceUrl,
  languageGgufFiles,
  licenseFromTags,
  listingBaseName,
  preferredMmprojFile,
  preferredMtpDraftFile,
  isMtpSidecarFile,
  remoteShardPaths,
  resolveLicenseUrl,
  STARTER_MODEL,
  HuggingFaceClient,
  type HfFileHit,
  type ModelLicenseInfo,
  type SettingsStore,
} from "@llama-aio/core";

export { HuggingFaceClient, STARTER_MODEL };

function hfToken(store: SettingsStore): string | undefined {
  const t = (store.getConfig().get<string>("hfToken") || "").trim();
  return t || undefined;
}

function enqueueHfFile(
  hf: HuggingFaceClient,
  store: SettingsStore,
  modelId: string,
  filePath: string,
  files?: HfFileHit[]
) {
  const meta = files?.find((f) => f.path === filePath);
  return downloadManager.enqueue({
    label: listingBaseName(filePath),
    dest: hf.localDestFor(modelId, filePath),
    url: meta?.url || huggingfaceUrl(`/${modelId}/resolve/main/${filePath}`),
    token: hfToken(store),
    expectedSize: meta?.size && meta.size > 0 ? meta.size : undefined,
    expectedSha256: meta?.sha256,
    gatedPageUrl: huggingfaceModelPage(modelId),
    modelId,
  });
}

async function enqueueModelAndCompanions(
  hf: HuggingFaceClient,
  store: SettingsStore,
  modelId: string,
  languagePath: string,
  files?: HfFileHit[]
): Promise<string> {
  DownloadPanel.show(store, downloadManager);
  const parts = remoteShardPaths(languagePath);
  const jobs = parts.map((part) => enqueueHfFile(hf, store, modelId, part, files));
  const listing = files ?? (await hf.listGgufFiles(modelId).catch(() => undefined));
  const extras = [];
  const mm = listing ? preferredMmprojFile(listing) : undefined;
  if (mm) {
    extras.push(enqueueHfFile(hf, store, modelId, mm.path, listing));
  }
  const mtp = listing ? preferredMtpDraftFile(listing, languagePath) : undefined;
  if (mtp) {
    extras.push(enqueueHfFile(hf, store, modelId, mtp.path, listing));
  }
  const dests = await Promise.all([...jobs, ...extras].map((j) => j.done));
  return dests[0] || hf.localDestFor(modelId, parts[0]!);
}

/**
 * One-click download of the curated starter GGUF via the HF resolve URL
 * (no browse/search). Reuses an existing local copy if present.
 */
export async function downloadStarterModel(
  hf: HuggingFaceClient,
  store: SettingsStore
): Promise<string | undefined> {
  const dest = await enqueueModelAndCompanions(
    hf,
    store,
    STARTER_MODEL.repoId,
    STARTER_MODEL.filePath
  );

  await store.applySelectedModel(dest, { attachMmproj: true });
  return dest;
}

export async function browseAndDownloadModel(
  hf: HuggingFaceClient,
  store: SettingsStore,
  initialQuery?: string
): Promise<string | undefined> {
  const query =
    initialQuery !== undefined
      ? initialQuery
      : await vscode.window.showInputBox({
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

  const extras = companionDownloadHint(files);
  const pickedFile = await vscode.window.showQuickPick(
    languageFiles.map((f) => ({
      label: f.path,
      description: describeLanguageGgufFile(f, files),
      file: f,
    })),
    {
      title: extras
        ? `Select a GGUF to download  ·  will also fetch ${extras}`
        : "Select a GGUF file to download",
      matchOnDescription: true,
    }
  );
  if (!pickedFile) {
    return undefined;
  }

  const dest = await enqueueModelAndCompanions(
    hf,
    store,
    pickedModel.model.id,
    pickedFile.file.path,
    files
  );

  const state = await store.applySelectedModel(dest, { attachMmproj: true });
  const caps = state.modelCapabilities;
  const vision = state.loadSettings.mmprojPath
    ? ` · vision ${state.loadSettings.mmprojPath.split(/[/\\]/).pop()}`
    : "";
  const mtp =
    isMtpSidecarFile({ path: state.loadSettings.draftModelPath })
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

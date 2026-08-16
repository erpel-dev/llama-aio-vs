import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";
import { HuggingFaceClient } from "./huggingFace";
import { formatModelSize, listLocalModelEntries } from "@llama-aio/core";
import { getModelsDir } from "@llama-aio/core";
import { SettingsStore } from "@llama-aio/core";

/** Open a native file dialog and select an existing .gguf model. */
export async function openModelFileDialog(
  store: SettingsStore
): Promise<string | undefined> {
  const modelsDir = getModelsDir(store.getConfig());
  const defaultUri = fs.existsSync(modelsDir)
    ? vscode.Uri.file(modelsDir)
    : vscode.Uri.file(os.homedir());

  const uris = await vscode.window.showOpenDialog({
    canSelectMany: false,
    canSelectFiles: true,
    canSelectFolders: false,
    openLabel: "Select GGUF model",
    title: "Select a GGUF model file",
    filters: {
      "GGUF models": ["gguf"],
      "All files": ["*"],
    },
    defaultUri,
  });

  const selected = uris?.[0]?.fsPath;
  if (!selected) {
    return undefined;
  }
  if (!selected.toLowerCase().endsWith(".gguf")) {
    const proceed = await vscode.window.showWarningMessage(
      "Selected file does not end with .gguf. Use it anyway?",
      "Use anyway",
      "Cancel"
    );
    if (proceed !== "Use anyway") {
      return undefined;
    }
  }

  await store.applySelectedModel(selected);
  return selected;
}

/** Pick from models in Llama AIO library and common tool download folders. */
export async function pickDownloadedModel(
  _hf: HuggingFaceClient,
  store: SettingsStore
): Promise<string | undefined> {
  const config = store.getConfig();
  const modelsDir = getModelsDir(config);
  const local = await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Llama AIO: Scanning local GGUF libraries…",
      cancellable: false,
    },
    async () => listLocalModelEntries(config)
  );

  if (!local.length) {
    const choice = await vscode.window.showInformationMessage(
      `No GGUF models found in ${modelsDir} or common tool folders (LM Studio, Unsloth, HF cache, …). Download one, or open a file.`,
      "Download from Hugging Face",
      "Open GGUF file…"
    );
    if (choice === "Download from Hugging Face") {
      await vscode.commands.executeCommand("llamaAio.browseModels");
      return store.getState().selectedModelPath || undefined;
    }
    if (choice === "Open GGUF file…") {
      return openModelFileDialog(store);
    }
    return store.getState().selectedModelPath || undefined;
  }

  const bySource = new Map<string, number>();
  for (const e of local) {
    bySource.set(e.source, (bySource.get(e.source) || 0) + 1);
  }
  const sourceSummary = [...bySource.entries()]
    .map(([s, n]) => `${s}: ${n}`)
    .join(" · ");

  type PickItem = vscode.QuickPickItem & { path?: string; openFile?: boolean };
  const items: PickItem[] = [
    {
      label: "$(folder-opened) Open GGUF file…",
      description: "Browse the filesystem for an existing .gguf",
      openFile: true,
    },
    ...local.map((e) => ({
      label: path.basename(e.path),
      description: e.source,
      detail: `${formatModelSize(e.sizeBytes)}  ·  ${e.path}`,
      path: e.path,
    })),
  ];

  const picked = await vscode.window.showQuickPick(items, {
    title: `Select a GGUF model (${local.length} found)`,
    placeHolder: sourceSummary || "Pick a model, or open a file…",
    matchOnDescription: true,
    matchOnDetail: true,
  });
  if (!picked) {
    return undefined;
  }
  if (picked.openFile) {
    return openModelFileDialog(store);
  }
  if (!picked.path) {
    return undefined;
  }

  await store.applySelectedModel(picked.path);
  return picked.path;
}

/**
 * Pick a draft GGUF for DFlash without changing the main selected model.
 * Prefers files whose architecture is `dflash` when metadata is readable.
 */
export async function pickDraftModelFile(
  store: SettingsStore
): Promise<string | undefined> {
  const modelsDir = getModelsDir(store.getConfig());
  const defaultUri = fs.existsSync(modelsDir)
    ? vscode.Uri.file(modelsDir)
    : vscode.Uri.file(os.homedir());

  const uris = await vscode.window.showOpenDialog({
    canSelectMany: false,
    canSelectFiles: true,
    canSelectFolders: false,
    openLabel: "Select draft GGUF",
    title: "Select a DFlash or MTP draft model (.gguf)",
    filters: {
      "GGUF models": ["gguf"],
      "All files": ["*"],
    },
    defaultUri,
  });

  return uris?.[0]?.fsPath;
}

/** Pick a draft GGUF from the local library (does not change the main model). */
export async function pickDraftModelFromLibrary(
  store: SettingsStore
): Promise<string | undefined> {
  const { readModelCapabilities, isDflashDraftArchitecture, isMtpDraftArchitecture, isMtpDraftFileName } =
    await import("@llama-aio/core");
  const { listMtpDraftEntries } = await import("@llama-aio/core");
  const config = store.getConfig();

  // Scan can take a while with many GGUFs — show progress so the sidebar click isn't a no-op.
  const scored = await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Llama AIO: Scanning for draft GGUFs…",
      cancellable: false,
    },
    async (progress) => {
      const local = [
        ...listLocalModelEntries(config),
        ...listMtpDraftEntries(config),
      ];
      const seen = new Set<string>();
      const unique = local.filter((e) => {
        if (seen.has(e.path)) {
          return false;
        }
        seen.add(e.path);
        return true;
      });
      const out: Array<{
        e: (typeof unique)[number];
        arch: string;
        dflash: boolean;
        mtp: boolean;
      }> = [];
      for (let i = 0; i < unique.length; i++) {
        const e = unique[i]!;
        if (i === 0 || (i + 1) % 5 === 0 || i + 1 === unique.length) {
          progress.report({
            message: `${i + 1}/${unique.length} · ${path.basename(e.path)}`,
          });
        }
        let arch = "";
        try {
          arch = readModelCapabilities(e.path).architecture || "";
        } catch {
          arch = "";
        }
        out.push({
          e,
          arch,
          dflash: isDflashDraftArchitecture(arch),
          mtp: isMtpDraftArchitecture(arch) || isMtpDraftFileName(e.path),
        });
      }
      out.sort((a, b) => Number(b.dflash || b.mtp) - Number(a.dflash || a.mtp) || Number(b.mtp) - Number(a.mtp));
      return out;
    }
  );

  if (!scored.length) {
    const choice = await vscode.window.showInformationMessage(
      "No local GGUF models found. Download a DFlash or sidecar MTP draft GGUF, then pick it here (or browse to the file).",
      "Open GGUF file…"
    );
    if (choice === "Open GGUF file…") {
      return pickDraftModelFile(store);
    }
    return undefined;
  }

  type PickItem = vscode.QuickPickItem & { path?: string; openFile?: boolean };
  const dflashCount = scored.filter((s) => s.dflash).length;
  const mtpCount = scored.filter((s) => s.mtp).length;
  const items: PickItem[] = [
    {
      label: "$(folder-opened) Open GGUF file…",
      description: "Browse for a DFlash or MTP draft .gguf",
      openFile: true,
    },
    ...scored.map(({ e, arch, dflash, mtp }) => ({
      label: (dflash || mtp ? "$(rocket) " : "") + path.basename(e.path),
      description: dflash
        ? `DFlash draft · ${e.source}`
        : mtp
          ? `MTP drafter · ${e.source}`
          : e.source + (arch ? ` · ${arch}` : ""),
      detail: `${formatModelSize(e.sizeBytes)}  ·  ${e.path}`,
      path: e.path,
    })),
  ];

  const picked = await vscode.window.showQuickPick(items, {
    title: "Select DFlash or MTP draft model",
    placeHolder:
      dflashCount + mtpCount > 0
        ? `${dflashCount} DFlash · ${mtpCount} MTP — rocket icons first`
        : "No draft GGUFs yet — pick Open GGUF file… or any local draft",
    matchOnDescription: true,
    matchOnDetail: true,
    ignoreFocusOut: true,
  });
  if (!picked) {
    return undefined;
  }
  if (picked.openFile) {
    return pickDraftModelFile(store);
  }
  return picked.path;
}

/**
 * Pick a vision projector GGUF without changing the main selected model.
 */
export async function pickMmprojFile(store: SettingsStore): Promise<string | undefined> {
  const modelsDir = getModelsDir(store.getConfig());
  const defaultUri = fs.existsSync(modelsDir)
    ? vscode.Uri.file(modelsDir)
    : vscode.Uri.file(os.homedir());

  const uris = await vscode.window.showOpenDialog({
    canSelectMany: false,
    canSelectFiles: true,
    canSelectFolders: false,
    openLabel: "Select mmproj GGUF",
    title: "Select a vision projector (.gguf)",
    filters: {
      "GGUF models": ["gguf"],
      "All files": ["*"],
    },
    defaultUri,
  });

  return uris?.[0]?.fsPath;
}

/** Pick a vision projector from the local library (does not change the main model). */
export async function pickMmprojFromLibrary(store: SettingsStore): Promise<string | undefined> {
  const { listMmprojEntries, formatModelSize, findSiblingMmproj } = await import("@llama-aio/core");
  const config = store.getConfig();
  const sibling = findSiblingMmproj(store.getState().selectedModelPath || "");

  const local = await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Llama AIO: Scanning for mmproj GGUFs…",
      cancellable: false,
    },
    async () => {
      const entries = listMmprojEntries(config);
      if (!sibling) {
        return entries;
      }
      return [...entries].sort((a, b) => Number(b.path === sibling) - Number(a.path === sibling));
    }
  );

  if (!local.length) {
    const choice = await vscode.window.showInformationMessage(
      "No mmproj GGUFs found next to your models. Download a multimodal GGUF (the projector is fetched automatically) or browse to mmproj-F16.gguf.",
      "Open GGUF file…"
    );
    if (choice === "Open GGUF file…") {
      return pickMmprojFile(store);
    }
    return undefined;
  }

  type PickItem = vscode.QuickPickItem & { path?: string; openFile?: boolean };
  const items: PickItem[] = [
    {
      label: "$(folder-opened) Open GGUF file…",
      description: "Browse for mmproj-F16.gguf (or similar)",
      openFile: true,
    },
    ...local.map((e) => {
      const isSibling = sibling ? e.path === sibling : false;
      return {
        label: (isSibling ? "$(eye) " : "") + path.basename(e.path),
        description: isSibling ? `Matches selected model · ${e.source}` : e.source,
        detail: `${formatModelSize(e.sizeBytes)}  ·  ${e.path}`,
        path: e.path,
      };
    }),
  ];

  const picked = await vscode.window.showQuickPick(items, {
    title: "Select vision projector (mmproj)",
    placeHolder: sibling
      ? `Sibling ${path.basename(sibling)} listed first with an eye icon`
      : "Pick an mmproj GGUF for llama-server --mmproj",
    matchOnDescription: true,
    matchOnDetail: true,
    ignoreFocusOut: true,
  });
  if (!picked) {
    return undefined;
  }
  if (picked.openFile) {
    return pickMmprojFile(store);
  }
  return picked.path;
}


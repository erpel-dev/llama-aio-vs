import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";
import { HuggingFaceClient } from "./huggingFace";
import {
  buildModelPickerHints,
  displayGgufTitle,
  formatBytes,
  formatPickerDetail,
  invalidateModelLibraryCache,
  listLocalModelEntries,
  pickerGpus,
  readPickerCapabilities,
  shortHomePath,
  type LlamaLoadSettings,
  type LocalModelEntry,
} from "@llama-aio/core";
import { getModelsDir } from "@llama-aio/core";
import { SettingsStore } from "@llama-aio/core";

export interface ModelPickerContext {
  cpuOnly?: boolean;
  llamaServerBinary?: string;
  loadSettings?: LlamaLoadSettings;
}

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

type PickAction = "model" | "openFile" | "hfSearch";
type PickItem = vscode.QuickPickItem & { action: PickAction; modelPath?: string };

function fitIcon(fit: string): string {
  if (fit === "fits") {
    return "$(pass)";
  }
  if (fit === "tight") {
    return "$(warning)";
  }
  if (fit === "wont-fit") {
    return "$(error)";
  }
  return "";
}

function fitLabel(fit: string): string {
  if (fit === "wont-fit") {
    return "won't fit";
  }
  return fit;
}

function isCurrentPath(current: string, filePath: string): boolean {
  return !!current && path.resolve(current) === path.resolve(filePath);
}

function cheapModelItem(entry: LocalModelEntry, current: string): PickItem {
  const currentMark = isCurrentPath(current, entry.path);
  const shards = entry.shardCount && entry.shardCount > 1 ? ` · ${entry.shardCount} shards` : "";
  return {
    action: "model",
    modelPath: entry.path,
    label: (currentMark ? "$(star-full) " : "") + displayGgufTitle(entry.path),
    description: `${currentMark ? "current · " : ""}${formatBytes(entry.sizeBytes)}${shards}`,
    detail: `${entry.source} · ${shortHomePath(path.dirname(entry.path))}`,
    picked: currentMark,
  };
}

function richModelItem(entry: LocalModelEntry, current: string, ctx?: ModelPickerContext): PickItem {
  const caps = readPickerCapabilities(entry.path);
  const hints = buildModelPickerHints(entry, {
    caps,
    settings: ctx?.loadSettings,
    gpus: pickerGpus(!!ctx?.cpuOnly, ctx?.llamaServerBinary),
    cpuOnly: ctx?.cpuOnly,
    ramTotal: os.totalmem(),
  });
  const currentMark = isCurrentPath(current, entry.path);
  const badges: string[] = [];
  if (currentMark) {
    badges.push("current");
  }
  if (hints.fit !== "unknown") {
    badges.push(`${fitIcon(hints.fit)} ${fitLabel(hints.fit)}`.trim());
  }
  if (hints.shardCount > 1) {
    badges.push(`${hints.shardCount} shards`);
  }
  return {
    action: "model",
    modelPath: entry.path,
    label: (currentMark ? "$(star-full) " : "") + hints.title,
    description: `${badges.join("  ·  ")}  ·  ${formatBytes(entry.sizeBytes)}`.replace(/^[ ·]+/, ""),
    detail: formatPickerDetail(hints),
    picked: currentMark,
  };
}

function hfSearchItem(query: string): PickItem {
  const q = query.trim();
  return {
    action: "hfSearch",
    label: q
      ? `$(cloud-download) Search Hugging Face for "${q}"…`
      : "$(cloud-download) Search Hugging Face…",
    description: "Download a GGUF",
    alwaysShow: true,
  };
}

/** Pick from models in Llama AIO library and common tool download folders. */
export async function pickDownloadedModel(
  _hf: HuggingFaceClient,
  store: SettingsStore,
  ctx?: ModelPickerContext
): Promise<string | undefined> {
  const config = store.getConfig();
  const modelsDir = getModelsDir(config);
  invalidateModelLibraryCache();
  const local = listLocalModelEntries(config);
  const current = store.getState().selectedModelPath || "";

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

  const qp = vscode.window.createQuickPick<PickItem>();
  qp.title = "Select model";
  qp.placeholder = "Filter models, or type to search Hugging Face…";
  qp.matchOnDescription = true;
  qp.matchOnDetail = true;
  qp.ignoreFocusOut = true;

  const openItem: PickItem = {
    action: "openFile",
    label: "$(folder-opened) Open GGUF file…",
    description: "Browse the filesystem for an existing .gguf",
    alwaysShow: true,
  };

  const byPath = new Map<string, PickItem>();
  for (const e of local) {
    byPath.set(e.path, cheapModelItem(e, current));
  }
  const rebuild = (filter: string) => {
    qp.items = [openItem, ...local.map((e) => byPath.get(e.path)!), hfSearchItem(filter)];
  };
  rebuild("");

  // Badges / fit lines are filled after the list is on screen (P-09).
  void (async () => {
    qp.busy = true;
    for (const e of local) {
      if (qp.items.length === 0) {
        break;
      }
      byPath.set(e.path, richModelItem(e, current, ctx));
      rebuild(qp.value);
      await new Promise((r) => setTimeout(r, 0));
    }
    qp.busy = false;
  })();

  return await new Promise<string | undefined>((resolve) => {
    let settled = false;
    const finish = (value: string | undefined) => {
      if (settled) {
        return;
      }
      settled = true;
      qp.dispose();
      resolve(value);
    };

    qp.onDidChangeValue((value) => rebuild(value));

    qp.onDidAccept(async () => {
      const picked = qp.selectedItems[0];
      if (!picked) {
        finish(undefined);
        return;
      }
      if (picked.action === "openFile") {
        finish(await openModelFileDialog(store));
        return;
      }
      if (picked.action === "hfSearch") {
        const q = qp.value.trim();
        qp.hide();
        await vscode.commands.executeCommand("llamaAio.browseModels", q || undefined);
        finish(store.getState().selectedModelPath || undefined);
        return;
      }
      if (!picked.modelPath) {
        finish(undefined);
        return;
      }
      await store.applySelectedModel(picked.modelPath);
      finish(picked.modelPath);
    });

    qp.onDidHide(() => finish(undefined));
    qp.show();
  });
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
  const { readModelCapabilities, isDflashDraftArchitecture, isMtpDraftArchitecture, isMtpSidecarFile } =
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
      invalidateModelLibraryCache();
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
          mtp: isMtpDraftArchitecture(arch) || isMtpSidecarFile({ path: e.path, size: e.sizeBytes }),
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

  type DraftPickItem = vscode.QuickPickItem & { path?: string; openFile?: boolean };
  const dflashCount = scored.filter((s) => s.dflash).length;
  const mtpCount = scored.filter((s) => s.mtp).length;
  const items: DraftPickItem[] = [
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
      detail: `${formatBytes(e.sizeBytes)}  ·  ${e.path}`,
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
      invalidateModelLibraryCache();
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

  type MmprojPickItem = vscode.QuickPickItem & { path?: string; openFile?: boolean };
  const items: MmprojPickItem[] = [
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

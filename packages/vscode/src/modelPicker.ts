import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";
import {
  findSiblingMmproj,
  formatBytes,
  formatModelSize,
  isDflashDraftArchitecture,
  isMtpDraftArchitecture,
  isMtpSidecarFile,
  listLocalModelEntries,
  listMmprojEntries,
  listMtpDraftEntries,
  readModelCapabilities,
} from "@llama-aio/core";
import { getModelsDir } from "@llama-aio/core";
import { SettingsStore } from "@llama-aio/core";

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

function yieldToUi(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
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
  const config = store.getConfig();
  type DraftPickItem = vscode.QuickPickItem & { path?: string; openFile?: boolean };
  type Scored = {
    path: string;
    source: string;
    sizeBytes: number;
    arch: string;
    dflash: boolean;
    mtp: boolean;
  };

  const qp = vscode.window.createQuickPick<DraftPickItem>();
  qp.title = "Select DFlash or MTP draft model";
  qp.placeholder = "Scanning library…";
  qp.busy = true;
  qp.matchOnDescription = true;
  qp.matchOnDetail = true;
  qp.ignoreFocusOut = true;

  const openItem: DraftPickItem = {
    label: "$(folder-opened) Open GGUF file…",
    description: "Browse for a DFlash or MTP draft .gguf",
    openFile: true,
  };

  let dismissed = false;
  const hideSub = qp.onDidHide(() => {
    dismissed = true;
  });
  qp.show();
  await yieldToUi();
  if (dismissed) {
    hideSub.dispose();
    qp.dispose();
    return undefined;
  }

  const listed = [...listLocalModelEntries(config), ...listMtpDraftEntries(config)];
  const seen = new Set<string>();
  const unique = listed.filter((e) => {
    if (seen.has(e.path)) {
      return false;
    }
    seen.add(e.path);
    return true;
  });
  await yieldToUi();
  hideSub.dispose();
  if (dismissed) {
    qp.dispose();
    return undefined;
  }

  if (!unique.length) {
    qp.dispose();
    const choice = await vscode.window.showInformationMessage(
      "No local GGUF models found. Download a DFlash or sidecar MTP draft GGUF, then pick it here (or browse to the file).",
      "Open GGUF file…"
    );
    if (choice === "Open GGUF file…") {
      return pickDraftModelFile(store);
    }
    return undefined;
  }

  const scored = new Map<string, Scored>();
  const cheap = (e: (typeof unique)[number]): DraftPickItem => ({
    label: path.basename(e.path),
    description: e.source,
    detail: `${formatBytes(e.sizeBytes)}  ·  ${e.path}`,
    path: e.path,
  });
  const rich = (s: Scored): DraftPickItem => ({
    label: (s.dflash || s.mtp ? "$(rocket) " : "") + path.basename(s.path),
    description: s.dflash
      ? `DFlash draft · ${s.source}`
      : s.mtp
        ? `MTP drafter · ${s.source}`
        : s.source + (s.arch ? ` · ${s.arch}` : ""),
    detail: `${formatBytes(s.sizeBytes)}  ·  ${s.path}`,
    path: s.path,
  });
  const paint = (filterHint: boolean) => {
    const rows = unique.map((e) => {
      const s = scored.get(e.path);
      return s ? rich(s) : cheap(e);
    });
    if (filterHint) {
      rows.sort((a, b) => {
        const as = scored.get(a.path || "");
        const bs = scored.get(b.path || "");
        return (
          Number(!!(bs?.dflash || bs?.mtp)) - Number(!!(as?.dflash || as?.mtp)) ||
          Number(!!bs?.mtp) - Number(!!as?.mtp)
        );
      });
    }
    const drafts = [...scored.values()].filter((s) => s.dflash || s.mtp).length;
    qp.placeholder =
      drafts > 0
        ? `${drafts} draft GGUF${drafts === 1 ? "" : "s"} — rocket icons first`
        : "Pick a draft GGUF, or Open GGUF file…";
    qp.items = [openItem, ...rows];
  };
  paint(false);

  void (async () => {
    try {
      qp.busy = true;
      for (const e of unique) {
        if (qp.items.length === 0) {
          break;
        }
        let arch = "";
        try {
          arch = readModelCapabilities(e.path).architecture || "";
        } catch {
          arch = "";
        }
        scored.set(e.path, {
          path: e.path,
          source: e.source,
          sizeBytes: e.sizeBytes,
          arch,
          dflash: isDflashDraftArchitecture(arch),
          mtp: isMtpDraftArchitecture(arch) || isMtpSidecarFile({ path: e.path, size: e.sizeBytes }),
        });
        paint(false);
        await yieldToUi();
      }
      paint(true);
      qp.busy = false;
    } catch {
      // Quick pick was dismissed while badges were filling in.
    }
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
    qp.onDidAccept(async () => {
      const picked = qp.selectedItems[0];
      if (!picked) {
        finish(undefined);
        return;
      }
      if (picked.openFile) {
        finish(await pickDraftModelFile(store));
        return;
      }
      finish(picked.path);
    });
    qp.onDidHide(() => finish(undefined));
  });
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
  const config = store.getConfig();
  const sibling = findSiblingMmproj(store.getState().selectedModelPath || "");

  const qp = vscode.window.createQuickPick<vscode.QuickPickItem & { path?: string; openFile?: boolean }>();
  qp.title = "Select vision projector (mmproj)";
  qp.placeholder = "Scanning library…";
  qp.busy = true;
  qp.ignoreFocusOut = true;
  let dismissed = false;
  const hideSub = qp.onDidHide(() => {
    dismissed = true;
  });
  qp.show();
  await yieldToUi();
  if (dismissed) {
    hideSub.dispose();
    qp.dispose();
    return undefined;
  }

  let local = listMmprojEntries(config);
  if (sibling) {
    local = [...local].sort((a, b) => Number(b.path === sibling) - Number(a.path === sibling));
  }
  await yieldToUi();
  hideSub.dispose();
  qp.dispose();
  if (dismissed) {
    return undefined;
  }

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

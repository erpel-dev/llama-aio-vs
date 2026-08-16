import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";
import { promptUseInCopilotChat } from "./copilotChatPrompt";
import { detectGpus } from "@llama-aio/core";
import { LlamaInstaller, UiBackend } from "@llama-aio/core";
import { estimateMemory, memoryEstimateInputs, mmprojFileSize, resolveDraftCapabilities } from "@llama-aio/core";
import { resolveModelModes } from "@llama-aio/core";
import { listActiveModelSourceDirs, listLocalModelEntries, findSiblingMtpDraft, isMtpDraftFileName } from "@llama-aio/core";
import { getModelsDir } from "@llama-aio/core";
import { PerfStats } from "@llama-aio/core";
import { LaunchToken, LAUNCH_IN_PROGRESS_MSG, ProcessManager } from "@llama-aio/core";
import { SettingsStore } from "@llama-aio/core";
import { resolveLaunchMode } from "@llama-aio/core";
import { DEFAULT_LOAD_SETTINGS, DEFAULT_REQUEST_SETTINGS, LlamaLoadSettings, RequestSettings } from "@llama-aio/core";
import { STARTER_MODEL } from "./huggingFace";

export type ModelActions = {
  downloadFromHuggingFace: () => Promise<void>;
  downloadStarter: () => Promise<void>;
  openGgufFile: () => Promise<void>;
  pickDownloaded: () => Promise<void>;
  pickDraftModel: () => Promise<void>;
  pickMmproj: () => Promise<void>;
  installLlamaCpp: (backend?: UiBackend) => Promise<void>;
  reinstallLlamaCpp: () => Promise<void>;
  installLlamaCppByTag: () => Promise<void>;
  installLlamaCppFromArchive: () => Promise<void>;
  switchBackend: (backend: UiBackend) => Promise<void>;
};

/**
 * Curated model modes overwrite temperature/top_p/top_k on every request, so the
 * Request defaults below them are inert for those models. Describe the active
 * mode set so the panel can say so instead of showing numbers that never ship.
 */
function describeModeSampling(
  caps: Parameters<typeof resolveModelModes>[0],
  modelPath: string
): { familyLabel: string; defaultMode: string; temperature: number; topP: number; topK: number } | null {
  const modeSet = resolveModelModes(caps, modelPath);
  if (!modeSet) {
    return null;
  }
  const params = modeSet.modes[modeSet.defaultMode] || {};
  return {
    familyLabel: modeSet.familyLabel,
    defaultMode: modeSet.defaultMode,
    temperature: params.temperature ?? 0,
    topP: params.top_p ?? 0,
    topK: params.top_k ?? 0,
  };
}

export class SettingsViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "llamaAio.settingsView";

  private view?: vscode.WebviewView;
  private updateCheckInFlight = false;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly store: SettingsStore,
    private readonly processManager: ProcessManager,
    private readonly installer: LlamaInstaller,
    private readonly perf: PerfStats,
    private readonly onReload: (token?: LaunchToken) => Promise<void>,
    private readonly modelActions: ModelActions,
    private readonly notifyChatModels: () => void = () => undefined
  ) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this.view = webviewView;
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this.extensionUri],
    };
    webviewView.webview.html = this.getHtml(webviewView.webview);
    webviewView.webview.onDidReceiveMessage(async (msg) => {
      try {
        switch (msg.type) {
          case "ready":
            await this.pushState();
            break;
          case "saveLoad":
            await this.store.updateLoadSettings(msg.payload as Partial<LlamaLoadSettings>);
            this.syncSpeculativeMode();
            if (msg.silent) {
              const status = this.processManager.getStatus();
              const httpReady = await this.processManager.isHttpReady();
              this.view?.webview.postMessage({
                type: "statusPatch",
                payload: {
                  configDirty: !!status.configDirty,
                  running: !!(status.running || httpReady),
                  starting: !!status.starting,
                  startMessage: status.startMessage || status.message,
                  perf: this.perf.get(),
                  perfLines: this.perf.detailLines(),
                },
              });
            } else {
              await this.pushState();
            }
            break;
          case "saveRequest":
            await this.store.updateRequestSettings(msg.payload as Partial<RequestSettings>);
            // Don't pushState — the webview already shows the edited values, and a
            // full state round-trip can overwrite in-progress typing.
            break;
          case "resetAdvancedLoad": {
            const d = DEFAULT_LOAD_SETTINGS;
            await this.store.updateLoadSettings({
              cpuThreads: d.cpuThreads,
              evalBatchSize: d.evalBatchSize,
              physicalBatchSize: d.physicalBatchSize,
              maxConcurrentPredictions: d.maxConcurrentPredictions,
              offloadKvCacheToGpu: d.offloadKvCacheToGpu,
              cacheTypeK: d.cacheTypeK,
              cacheTypeV: d.cacheTypeV,
              keepModelInMemory: d.keepModelInMemory,
              tryMmap: d.tryMmap,
              unifiedKvCache: d.unifiedKvCache,
              flashAttention: d.flashAttention,
              contextCheckpoints: d.contextCheckpoints,
              cacheReuse: d.cacheReuse,
              reasoningFormat: d.reasoningFormat,
              reasoningBudget: d.reasoningBudget,
              ropeFreqBase: d.ropeFreqBase,
              ropeFreqScale: d.ropeFreqScale,
              seed: d.seed,
              speculativeMode: d.speculativeMode,
              maxDraftTokens: d.maxDraftTokens,
              minDraftTokens: d.minDraftTokens,
              draftProbability: d.draftProbability,
              draftModelPath: d.draftModelPath,
              draftGpuOffload: d.draftGpuOffload,
            });
            this.syncSpeculativeMode();
            await this.pushState();
            break;
          }
          case "resetRequestDefaults":
            await this.store.updateRequestSettings({ ...DEFAULT_REQUEST_SETTINGS });
            await this.pushState();
            break;
          case "reload": {
            const token = this.processManager.claimLaunch("reload", "Reloading llama-server…");
            if (!token) {
              void vscode.window.showWarningMessage(LAUNCH_IN_PROGRESS_MSG);
              break;
            }
            this.postBootProgress("Reloading llama-server…");
            try {
              await this.store.updateLoadSettings(msg.payload as Partial<LlamaLoadSettings>);
              this.syncSpeculativeMode();
              if (!(await this.confirmIfMemorySpill())) {
                break;
              }
              await this.onReload(token);
            } finally {
              this.processManager.releaseLaunch(token);
              await this.pushState();
            }
            break;
          }
          case "start": {
            const token = this.processManager.claimLaunch("start", "Starting llama-server…");
            if (!token) {
              void vscode.window.showWarningMessage(LAUNCH_IN_PROGRESS_MSG);
              break;
            }
            this.postBootProgress("Starting llama-server…");
            try {
              if (!(await this.confirmIfMemorySpill())) {
                break;
              }
              const status = await vscode.window.withProgress(
                {
                  location: vscode.ProgressLocation.Notification,
                  title: "Llama AIO: Starting llama-server…",
                  cancellable: false,
                },
                async (progress) =>
                  this.processManager.start(
                    undefined,
                    (m) => {
                      progress.report({ message: m });
                      this.postBootProgress(m);
                    },
                    token
                  )
              );
              this.notifyChatModels();
              await promptUseInCopilotChat(this.store, status.message);
            } finally {
              this.processManager.releaseLaunch(token);
              await this.pushState();
            }
            break;
          }
          case "stop":
            await this.processManager.stop(true);
            await this.pushState();
            break;
          case "refresh":
            await this.pushState();
            break;
          case "setLaunchMode": {
            const mode = msg.payload === "background" ? "background" : "externalTerminal";
            await this.store
              .getConfig()
              .update("launchMode", mode);
            await this.pushState();
            break;
          }
          case "downloadModel":
            await this.modelActions.downloadFromHuggingFace();
            await this.pushState();
            break;
          case "downloadStarter":
            await this.modelActions.downloadStarter();
            await this.pushState();
            break;
          case "openModelFile":
            await this.modelActions.openGgufFile();
            await this.pushState();
            break;
          case "pickDownloadedModel":
            await this.modelActions.pickDownloaded();
            await this.pushState();
            break;
          case "pickDraftModel":
            // Prefer the command so QuickPick/file dialog get focus (webview clicks
            // can otherwise look like a no-op on Linux when the picker is hidden).
            try {
              await vscode.commands.executeCommand("llamaAio.selectDraftModel");
            } catch {
              await this.modelActions.pickDraftModel();
            }
            await this.pushState();
            break;
          case "clearDraftModel":
            await this.store.updateLoadSettings({ draftModelPath: "" });
            this.view?.webview.postMessage({ type: "draftModelSelected", path: "" });
            this.syncSpeculativeMode();
            await this.pushState();
            break;
          case "pickMmproj":
            try {
              await vscode.commands.executeCommand("llamaAio.selectMmproj");
            } catch {
              await this.modelActions.pickMmproj();
            }
            await this.pushState();
            break;
          case "clearMmproj":
            await this.store.updateLoadSettings({ mmprojPath: "" });
            this.view?.webview.postMessage({ type: "mmprojSelected", path: "" });
            this.notifyChatModels();
            await this.pushState();
            break;
          case "installLlamaCpp":
            await this.modelActions.installLlamaCpp(msg.payload as UiBackend | undefined);
            await this.pushState();
            break;
          case "reinstallLlamaCpp":
            await this.modelActions.reinstallLlamaCpp();
            await this.pushState();
            break;
          case "checkUpdates":
            await this.refreshUpdateCheck();
            break;
          case "viewLastCall":
            await this.openLastRequestContext();
            break;
          case "viewLastResponse":
            await this.openLastResponseTrace();
            break;
          case "setPromptReplacementsEnabled": {
            const enabled = !!(msg.payload && (msg.payload as { enabled?: boolean }).enabled);
            await this.store
              .getConfig()
              .update("promptReplacementsEnabled", enabled);
            await this.pushState();
            break;
          }
          case "installLlamaCppByTag":
            await this.modelActions.installLlamaCppByTag();
            await this.pushState();
            break;
          case "installLlamaCppFromArchive":
            await this.modelActions.installLlamaCppFromArchive();
            await this.pushState();
            break;
          case "switchBackend":
            await this.modelActions.switchBackend(msg.payload as UiBackend);
            await this.pushState();
            break;
          case "openExternal": {
            const url = typeof msg.url === "string" ? msg.url : "";
            if (/^https?:\/\//i.test(url)) {
              await vscode.env.openExternal(vscode.Uri.parse(url));
            }
            break;
          }
          case "revealInOs": {
            const targetPath = typeof msg.path === "string" ? msg.path : "";
            if (!targetPath) {
              break;
            }
            if (!fs.existsSync(targetPath)) {
              vscode.window.showWarningMessage(`Llama AIO: Path not found:\n${targetPath}`);
              break;
            }
            const uri = vscode.Uri.file(targetPath);
            try {
              const st = fs.statSync(targetPath);
              if (st.isDirectory()) {
                // Open the folder itself in the OS file manager.
                await vscode.env.openExternal(uri);
              } else {
                await vscode.commands.executeCommand("revealFileInOS", uri);
              }
            } catch (e) {
              vscode.window.showErrorMessage(
                `Llama AIO: Could not open path: ${e instanceof Error ? e.message : String(e)}`
              );
            }
            break;
          }
        }
      } catch (e) {
        vscode.window.showErrorMessage(
          `Llama AIO: ${e instanceof Error ? e.message : String(e)}`
        );
        await this.pushState();
      }
    });
  }

  private currentMemoryEstimate() {
    const state = this.store.getState();
    const cpuOnly =
      this.installer.resolveActiveUiBackend() === "cpu" || this.processManager.isCpuBackend();
    const gpus = cpuOnly ? [] : detectGpus(false, this.processManager.resolveBinary());
    return estimateMemory(
      state.modelCapabilities,
      state.loadSettings,
      cpuOnly ? undefined : gpus[0],
      { cpuOnly, gpus: cpuOnly ? undefined : gpus }
    );
  }

  /** Keep sidebar speculative line in sync with Load settings (clears stale % when off). */
  syncSpeculativeMode(): void {
    const mode = this.store.getState().loadSettings.speculativeMode || "off";
    this.perf.setSpeculativeMode(mode === "mtp" || mode === "dflash" ? mode : "off");
  }

  /** Ask before start/reload when estimated memory is likely to spill (VRAM or RAM). */
  async confirmIfMemorySpill(): Promise<boolean> {
    const est = this.currentMemoryEstimate();
    if (!est?.willSpill) {
      return true;
    }
    const choice = await vscode.window.showWarningMessage(
      est.warnings[0] ||
        "These settings leave too little memory headroom and may spill or thrash (much slower).",
      { modal: true },
      "Continue anyway",
      "Cancel"
    );
    return choice === "Continue anyway";
  }

  async pushState(): Promise<void> {
    if (!this.view) {
      return;
    }
    this.syncSpeculativeMode();
    let state = this.store.getState();
    // Refresh GGUF caps when older state lacks size / arch dims needed for estimates.
    if (
      state.selectedModelPath &&
      (!state.modelCapabilities?.fileSizeBytes ||
        state.modelCapabilities.path !== state.selectedModelPath ||
        // Stale caps from before SWA / per-layer KV support (e.g. Gemma 4).
        (state.modelCapabilities.architecture === "gemma4" &&
          !state.modelCapabilities.slidingWindowPattern) ||
        (state.modelCapabilities.architecture === "muse-glimmer" &&
          (!state.modelCapabilities.slidingWindowPattern ||
            state.modelCapabilities.slidingWindowPattern.length <
              (state.modelCapabilities.blockCount || 0))) ||
        // Stale caps from before hybrid full-attention interval (e.g. Qwen3.5).
        ((state.modelCapabilities.architecture === "qwen35" ||
          state.modelCapabilities.architecture === "qwen35moe") &&
          !state.modelCapabilities.fullAttentionInterval))
    ) {
      try {
        state = await this.store.applySelectedModel(state.selectedModelPath);
      } catch {
        // keep previous
      }
    }

    const status = this.processManager.getStatus();
    const httpReady = await this.processManager.isHttpReady();
    const binary = this.processManager.resolveBinary();
    const modelsDir = getModelsDir(this.store.getConfig());
    const localEntries = listLocalModelEntries(this.store.getConfig());
    const localModelCount = localEntries.length;
    const localSourceDirs = listActiveModelSourceDirs(this.store.getConfig(), localEntries);
    const localSources = localSourceDirs.map((s) => s.source);

    const caps = state.modelCapabilities;
    const build = this.installer.getInstalledInfo();
    const backendOptions = this.installer.getUiBackendOptions();
    // Active UI backend only — do not OR stale resolved/configured flags
    // (those stuck Load settings in CPU mode after switching back to Vulkan).
    const selectedUiBackend = (build.activeBackend ||
      build.resolvedBackend ||
      (build.configuredBackend === "auto" ? "vulkan" : build.configuredBackend)) as string;
    const cpuOnly = selectedUiBackend === "cpu";
    const gpus = cpuOnly ? [] : detectGpus(false, this.processManager.resolveBinary());
    const gpu = gpus[0];
    const draftCaps = resolveDraftCapabilities(state.loadSettings);
    const memory = estimateMemory(caps, state.loadSettings, gpu, {
      cpuOnly,
      draftCaps,
      gpus: cpuOnly ? undefined : gpus,
    });
    const updateCheck = this.installer.peekUpdateCheck();

    this.view.webview.postMessage({
      type: "state",
      payload: {
        state,
        status: { ...status, httpReady },
        perf: this.perf.get(),
        perfLines: this.perf.detailLines(),
        hasLastContext: this.perf.hasLastRequestContext(),
        hasLastResponse: this.perf.hasLastResponseTrace(),
        promptReplacementsEnabled: this.store.isPromptReplacementsEnabled(),
        endpoint: this.store.getEndpoint(),
        binary,
        binaryExists: fs.existsSync(binary),
        modelsDir,
        localModelCount,
        localSources,
        localSourceDirs,
        modelName: caps?.name || (state.selectedModelPath ? path.basename(state.selectedModelPath) : ""),
        build,
        backendOptions,
        selectedUiBackend,
        cpuOnly,
        launchMode: resolveLaunchMode(this.store.getConfig().get<string>("launchMode")),
        updateCheck,
        memory,
        modeSampling: describeModeSampling(caps, state.selectedModelPath),
        memInputs: memoryEstimateInputs(
          caps,
          draftCaps,
          mmprojFileSize(state.loadSettings.mmprojPath)
        ),
        systemRamTotalBytes: os.totalmem(),
        cpuCount: Math.max(1, os.cpus().length || 1),
        isWindows: process.platform === "win32",
        gpu: gpu
          ? { totalBytes: gpu.totalBytes, usedBytes: gpu.usedBytes, name: gpu.name }
          : null,
        gpus: gpus.map((g, i) => ({
          totalBytes: g.totalBytes,
          usedBytes: g.usedBytes,
          name: g.name,
          index: g.index ?? i,
          llamaDeviceId: g.llamaDeviceId,
        })),
        mtpSidecarPath:
          findSiblingMtpDraft(state.selectedModelPath || "") ||
          (isMtpDraftFileName(state.loadSettings.draftModelPath) ? state.loadSettings.draftModelPath : "") ||
          "",
        capabilities: caps
          ? {
              maxContextLength: caps.maxContextLength,
              blockCount: caps.blockCount,
              isMoe: caps.isMoe,
              expertCount: caps.expertCount,
              expertUsedCount: caps.expertUsedCount,
              architecture: caps.architecture,
              ropeFreqBase: caps.ropeFreqBase,
              nextnPredictLayers: caps.nextnPredictLayers,
              fileSizeBytes: caps.fileSizeBytes,
              embeddingLength: caps.embeddingLength,
              attentionHeadCount: caps.attentionHeadCount,
              attentionHeadCountKv: caps.attentionHeadCountKv,
              attentionHeadCountKvPerLayer: caps.attentionHeadCountKvPerLayer,
              keyLength: caps.keyLength,
              valueLength: caps.valueLength,
              keyLengthSwa: caps.keyLengthSwa,
              valueLengthSwa: caps.valueLengthSwa,
              slidingWindow: caps.slidingWindow,
              slidingWindowPattern: caps.slidingWindowPattern,
              fullAttentionInterval: caps.fullAttentionInterval,
              recurrentLayers: caps.recurrentLayers,
            }
          : null,
      },
    });

    // Resolve latest tag in the background when cache is cold (no GitHub API).
    // Patch the hint immediately — a full pushState re-probes the binary and
    // would leave "Checking for updates…" on screen for a long time.
    if (updateCheck.pending && !this.updateCheckInFlight) {
      this.updateCheckInFlight = true;
      void this.installer
        .getUpdateCheck(false)
        .then((check) => this.postUpdateCheck(check))
        .catch(() => undefined)
        .finally(() => {
          this.updateCheckInFlight = false;
        });
    }
  }

  /** Push live start/reload progress into the Server card without a full state rebuild. */
  postBootProgress(message: string): void {
    this.view?.webview.postMessage({
      type: "bootProgress",
      payload: {
        starting: true,
        message,
        running: false,
        configDirty: false,
      },
    });
  }

  /** Update the DFlash draft hint immediately (before a full pushState finishes). */
  postDraftModelSelected(draftPath: string): void {
    this.view?.webview.postMessage({
      type: "draftModelSelected",
      path: draftPath || "",
    });
  }

  /** Update the vision projector hint immediately (before a full pushState finishes). */
  postMmprojSelected(mmprojPath: string): void {
    this.view?.webview.postMessage({
      type: "mmprojSelected",
      path: mmprojPath || "",
    });
  }

  /** Swap the backend hint without rebuilding the rest of the sidebar. */
  postUpdateCheck(check: ReturnType<LlamaInstaller["peekUpdateCheck"]>): void {
    this.view?.webview.postMessage({
      type: "updateCheck",
      payload: check,
    });
  }

  /** Patch Performance tiles without a full sidebar rebuild (library scan, binary probe). */
  postPerf(): void {
    this.view?.webview.postMessage({
      type: "perfPatch",
      payload: {
        perf: this.perf.get(),
        perfLines: this.perf.detailLines(),
        hasLastContext: this.perf.hasLastRequestContext(),
        hasLastResponse: this.perf.hasLastResponseTrace(),
      },
    });
  }

  /** Force-refresh latest release tag and refresh the sidebar. */
  async refreshUpdateCheck(): Promise<void> {
    await this.installer.getUpdateCheck(true);
    await this.pushState();
  }

  /** Open the last Copilot → llama.cpp request dump in an editor. */
  async openLastRequestContext(): Promise<void> {
    const text = this.perf.formatLastRequestContext();
    if (!text) {
      vscode.window.showInformationMessage(
        "Llama AIO: No chat call captured yet. Send a Copilot Chat message using Llama AIO first."
      );
      return;
    }
    const doc = await vscode.workspace.openTextDocument({
      content: text,
      language: "markdown",
    });
    await vscode.window.showTextDocument(doc, { preview: false, viewColumn: vscode.ViewColumn.Active });
  }

  /** Open the last assistant stream dump (empty-response debugging). */
  async openLastResponseTrace(): Promise<void> {
    const text = this.perf.formatLastResponseTrace();
    if (!text) {
      vscode.window.showInformationMessage(
        "Llama AIO: No chat response captured yet. Send a Copilot Chat message using Llama AIO first."
      );
      return;
    }
    const doc = await vscode.workspace.openTextDocument({
      content: text,
      language: "markdown",
    });
    await vscode.window.showTextDocument(doc, { preview: false, viewColumn: vscode.ViewColumn.Active });
  }

  private getHtml(webview: vscode.Webview): string {
    const nonce = getNonce();
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Llama AIO</title>
  <style>
    :root {
      --bg: var(--vscode-sideBar-background);
      --fg: var(--vscode-foreground);
      --muted: var(--vscode-descriptionForeground);
      --input-bg: var(--vscode-input-background);
      --input-fg: var(--vscode-input-foreground);
      --border: var(--vscode-input-border, rgba(128,128,128,0.35));
      --accent: var(--vscode-button-background);
      --accent-fg: var(--vscode-button-foreground);
      --secondary: var(--vscode-button-secondaryBackground);
      --secondary-fg: var(--vscode-button-secondaryForeground);
      --warn-bg: color-mix(in srgb, #d29922 18%, transparent);
      --ok: #3fb950;
      --bad: #f85149;
      --warn: #d29922;
      --starting: #58a6ff;
      --link: var(--vscode-textLink-foreground);
    }
    body {
      font-family: var(--vscode-font-family);
      font-size: 12px;
      color: var(--fg);
      background: var(--bg);
      padding: 10px 12px 24px;
      margin: 0;
    }
    h2 { font-size: 13px; margin: 18px 0 8px; font-weight: 600; }
    h2:first-child { margin-top: 4px; }
    .card {
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--input-bg);
      margin-bottom: 10px;
      line-height: 1.45;
    }
    .card.server {
      border-color: color-mix(in srgb, var(--ok) 28%, var(--border));
    }
    .card.server.stopped { border-color: var(--border); }
    .card.server.dirty {
      border-color: color-mix(in srgb, var(--warn) 45%, var(--border));
    }
    .card.server.starting {
      border-color: color-mix(in srgb, var(--starting) 35%, var(--border));
    }
    .card.server.error {
      border-color: color-mix(in srgb, var(--bad) 45%, var(--border));
    }
    .srv-head { margin-bottom: 8px; }
    .srv-title { font-weight: 650; font-size: 13px; }
    .status-line {
      display: flex;
      align-items: center;
      gap: 7px;
      font-weight: 600;
      margin-bottom: 2px;
    }
    .status-line.ok { color: var(--ok); }
    .status-line.stopped { color: var(--muted); }
    .status-line.starting { color: var(--starting); }
    .status-line.error { color: var(--bad); }
    .status-line.dirty { color: var(--warn); }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      flex-shrink: 0;
    }
    .dot.ok { background: var(--ok); box-shadow: 0 0 0 2px color-mix(in srgb, var(--ok) 25%, transparent); }
    .dot.stopped { background: #6e7681; }
    .dot.starting {
      background: var(--starting);
      animation: srv-pulse 1.2s ease-in-out infinite;
    }
    .dot.error { background: var(--bad); box-shadow: 0 0 0 2px color-mix(in srgb, var(--bad) 25%, transparent); }
    .dot.dirty { background: var(--warn); box-shadow: 0 0 0 2px color-mix(in srgb, var(--warn) 25%, transparent); }
    @keyframes srv-pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.45; }
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 2px 9px 2px 7px;
      border-radius: 999px;
      border: 1px solid color-mix(in srgb, currentColor 45%, transparent);
      background: color-mix(in srgb, currentColor 12%, transparent);
      font-size: 11px;
      line-height: 1.6;
    }
    .meta-row {
      color: var(--muted);
      font-size: 11px;
      margin-top: 2px;
    }
    .meta-row a.endpoint {
      color: var(--link);
      text-decoration: underline;
      cursor: pointer;
    }
    .meta-row a.endpoint:hover {
      color: var(--vscode-textLink-activeForeground);
    }
    .meta-row .pid { color: var(--muted); }
    .dirty-hint {
      margin-top: 8px;
      padding: 6px 8px;
      border-radius: 5px;
      border: 1px solid color-mix(in srgb, var(--warn) 50%, var(--border));
      background: color-mix(in srgb, var(--warn) 12%, transparent);
      font-size: 11px;
      color: var(--fg);
      line-height: 1.4;
    }
    .dirty-hint .label {
      color: var(--warn);
      font-weight: 650;
      margin-right: 4px;
    }
    .actions-row {
      display: flex;
      gap: 6px;
      margin-top: 10px;
    }
    .actions-row button.primary,
    .actions-row button.warn-primary { flex: 1; text-align: center; }
    .actions-row button.secondary {
      flex: 0 0 auto;
      min-width: 72px;
      text-align: center;
    }
    .launch-row { margin-top: 8px; }
    .launch-row label {
      display: block;
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .mem-warn {
      margin-top: 8px;
      padding: 8px;
      border-radius: 6px;
      border: 1px solid color-mix(in srgb, #f85149 55%, var(--border));
      background: color-mix(in srgb, #f85149 16%, transparent);
      color: var(--fg);
      font-size: 11px;
      line-height: 1.45;
    }
    .mem-note {
      margin-top: 6px;
      padding: 6px 8px;
      border-radius: 6px;
      border: 1px solid color-mix(in srgb, #d29922 55%, var(--border));
      background: color-mix(in srgb, #d29922 14%, transparent);
      font-size: 11px;
      line-height: 1.45;
    }
    .ctx-chart-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-weight: 600;
      font-size: 11px;
      margin: 8px 0 4px;
    }
    .ctx-chart-title .sub { font-weight: 400; color: var(--muted); }
    .ctx-stack {
      display: flex;
      height: 14px;
      border-radius: 4px;
      overflow: hidden;
      background: color-mix(in srgb, var(--fg) 10%, transparent);
      border: 1px solid var(--border);
    }
    .ctx-stack > span { display: block; height: 100%; min-width: 0; }
    .ctx-stack.warn { outline: 1px solid #d29922; }
    .ctx-stack.critical { outline: 1px solid #f85149; }
    .ctx-stack .seg-tools { background: #3b82f6; }
    .ctx-stack .seg-system { background: #64748b; }
    .ctx-stack .seg-history { background: #22c55e; }
    .ctx-stack .seg-toolResults { background: #a855f7; }
    .ctx-stack .seg-request { background: #f59e0b; }
    .ctx-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      margin-top: 6px;
      font-size: 10px;
      color: var(--muted);
    }
    .ctx-legend i {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 2px;
      margin-right: 4px;
      vertical-align: middle;
    }
    .ctx-legend .seg-tools { background: #3b82f6; }
    .ctx-legend .seg-system { background: #64748b; }
    .ctx-legend .seg-history { background: #22c55e; }
    .ctx-legend .seg-toolResults { background: #a855f7; }
    .ctx-legend .seg-request { background: #f59e0b; }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      margin-top: 10px;
    }
    .stat {
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 6px 8px;
      background: color-mix(in srgb, var(--fg) 4%, transparent);
      min-width: 0;
    }
    .stat .k {
      font-size: 9.5px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .stat .v {
      font-size: 13px;
      font-weight: 650;
      margin-top: 1px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .stat .s {
      font-size: 10px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .stat.empty .v { color: var(--muted); font-weight: 500; }
    .stat.good .v { color: var(--ok); }
    .stat.warn .v { color: var(--warn); }
    .presets {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 0 0 4px;
    }
    .chip {
      border: 1px solid var(--border);
      border-radius: 999px;
      background: transparent;
      color: var(--fg);
      font-size: 11px;
      font-weight: 500;
      padding: 3px 10px;
      text-align: center;
      cursor: pointer;
    }
    .chip:hover { background: color-mix(in srgb, var(--fg) 8%, transparent); }
    .chip.active {
      border-color: color-mix(in srgb, var(--accent) 70%, var(--border));
      background: color-mix(in srgb, var(--accent) 22%, transparent);
    }
    .subgroup-title {
      font-size: 10px;
      font-weight: 650;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      margin: 16px 0 8px;
      padding-bottom: 4px;
      border-bottom: 1px solid var(--border);
    }
    .auto-field { display: flex; align-items: center; gap: 10px; }
    .auto-field .auto {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      font-size: 11px;
      font-weight: 400;
      color: var(--muted);
    }
    .auto-field input[type="number"] { width: 78px; }
    input:disabled, select:disabled { opacity: 0.5; }
    .mem-charts { margin-top: 10px; display: flex; flex-direction: column; gap: 10px; }
    .mem-chart-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-weight: 600;
      font-size: 11px;
      margin-bottom: 4px;
    }
    .mem-chart-title .sub { font-weight: 400; color: var(--muted); }
    .mem-stack {
      display: flex;
      height: 14px;
      border-radius: 4px;
      overflow: hidden;
      background: color-mix(in srgb, var(--fg) 10%, transparent);
      border: 1px solid var(--border);
    }
    .mem-stack > span { display: block; height: 100%; min-width: 0; }
    .mem-stack .seg-weights { background: #3b82f6; }
    .mem-stack .seg-vision { background: #f97316; }
    .mem-stack .seg-draft { background: #14b8a6; }
    .mem-stack .seg-kv { background: #a855f7; }
    .mem-stack .seg-overhead { background: #64748b; }
    .mem-stack.warn { outline: 1px solid #d29922; }
    .mem-stack.over { outline: 1px solid #f85149; }
    .mem-chart-title .sub.warn { color: #d29922; }
    .mem-chart-title .sub.over { color: #f85149; }
    .mem-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      margin-top: 6px;
      font-size: 10px;
      color: var(--muted);
    }
    .mem-legend i {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 2px;
      margin-right: 4px;
      vertical-align: middle;
    }
    .mem-legend .seg-weights { background: #3b82f6; }
    .mem-legend .seg-vision { background: #f97316; }
    .mem-legend .seg-draft { background: #14b8a6; }
    .mem-legend .seg-kv { background: #a855f7; }
    .mem-legend .seg-overhead { background: #64748b; }
    .setup {
      background: var(--warn-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      margin-bottom: 10px;
    }
    .setup ol { margin: 6px 0 0 18px; padding: 0; }
    .setup li { margin: 4px 0; }
    .row { margin: 10px 0 14px; }
    .label {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 4px;
    }
    .label span.name { font-weight: 500; }
    .label .badge {
      font-size: 10px;
      opacity: 0.75;
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 0 4px;
    }
    #tensorSplitPct {
      font-variant-numeric: tabular-nums;
      font-weight: 600;
      min-width: 3em;
      text-align: right;
    }
    .hint { color: var(--muted); font-size: 11px; margin-top: 2px; }
    /* Hover help: llama.cpp flag + short description */
    .tip {
      position: relative;
      border-bottom: 1px dotted color-mix(in srgb, var(--muted) 70%, transparent);
      cursor: help;
    }
    .tip::after {
      content: attr(data-flag) "\\A" attr(data-help);
      white-space: pre-wrap;
      position: absolute;
      left: 0;
      top: calc(100% + 6px);
      z-index: 40;
      width: max-content;
      max-width: min(260px, 70vw);
      padding: 8px 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--vscode-editorWidget-background, var(--input-bg));
      color: var(--fg);
      font-size: 11px;
      font-weight: 400;
      line-height: 1.45;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.28);
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
      transition: opacity 0.08s ease;
    }
    /* First line (flag) reads as code; description wraps below via \\A */
    .name.tip,
    .toggle > span.tip {
      font-weight: 500;
    }
    .tip:hover::after,
    .tip:focus-visible::after {
      opacity: 1;
      visibility: visible;
    }
    details.advanced,
    details.advanced > summary,
    .row,
    .toggle,
    .label {
      overflow: visible;
    }
    input[type="number"], input[type="text"],     select {
      width: 88px;
      background: var(--input-bg);
      color: var(--input-fg);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 2px 6px;
    }
    select.wide { width: 100%; box-sizing: border-box; padding: 6px 8px; }
    input[type="range"] { width: 100%; }
    .toggle {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin: 10px 0;
    }
    details.advanced {
      margin: 14px 0 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0 10px 4px;
      background: color-mix(in srgb, var(--input-bg) 55%, transparent);
    }
    details.advanced > summary {
      cursor: pointer;
      list-style: none;
      font-weight: 600;
      padding: 10px 0;
      color: var(--fg);
      user-select: none;
    }
    details.advanced > summary::-webkit-details-marker { display: none; }
    details.advanced > summary::before {
      content: '▸';
      display: inline-block;
      width: 1em;
      margin-right: 4px;
      color: var(--muted);
    }
    details.advanced[open] > summary::before { content: '▾'; }
    details.advanced > summary .sub {
      font-weight: 400;
      font-size: 11px;
      color: var(--muted);
      margin-left: 6px;
    }
    .btn-col {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 8px;
    }
    button {
      border: none;
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
      font-weight: 600;
      text-align: left;
      font-size: 12px;
      font-family: inherit;
    }
    button.primary { background: var(--accent); color: var(--accent-fg); }
    button.secondary { background: var(--secondary); color: var(--secondary-fg); }
    button.warn-primary {
      background: color-mix(in srgb, var(--warn) 75%, #8a6a10);
      color: #1a1a1a;
    }
    button:disabled {
      opacity: 0.55;
      cursor: default;
    }
    .model-title { font-weight: 600; margin-bottom: 4px; }
    .model-path, .meta {
      word-break: break-all;
      color: var(--muted);
      font-size: 11px;
    }
    a.model-path-link {
      color: var(--vscode-textLink-foreground);
      text-decoration: underline;
      cursor: pointer;
      word-break: break-all;
      font-size: 11px;
    }
    a.model-path-link:hover {
      color: var(--vscode-textLink-activeForeground);
    }
    .meta a.folder-link {
      color: var(--vscode-textLink-foreground);
      text-decoration: underline;
      cursor: pointer;
    }
    .meta a.folder-link:hover {
      color: var(--vscode-textLink-activeForeground);
    }
    .hidden { display: none !important; }
    .ok { color: var(--ok); }
    .caps {
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 11px;
      line-height: 1.5;
    }
  </style>
</head>
<body>
  <div class="card server stopped" id="serverCard">
    <div class="srv-head">
      <div class="srv-title">Server</div>
    </div>
    <div class="status-line stopped" id="statusLine">
      <span class="pill">
        <span class="dot stopped" id="statusDot"></span>
        <span id="statusText">Loading…</span>
      </span>
    </div>
    <div class="meta-row" id="statusMeta">—</div>
    <div class="dirty-hint hidden" id="dirtyHint">
      <span class="label">Settings changed</span>
      Reload to apply load settings and launch mode.
    </div>
    <div class="actions-row">
      <button class="primary" id="primaryBtn" data-action="start">Start</button>
      <button class="secondary" id="stopBtn" disabled>Stop</button>
    </div>
    <div class="launch-row">
      <label for="launchMode">Launch mode</label>
      <select id="launchMode" class="wide" title="How llama-server is started">
        <option value="externalTerminal">External terminal (logs visible)</option>
        <option value="background">Background (hidden process)</option>
      </select>
    </div>
  </div>

  <div class="card" id="perfCard">
    <div class="model-title">Performance</div>
    <div class="ctx-chart-title">
      <span id="ctxLabel">Context</span>
      <span class="sub" id="ctxSub">— (send a chat to measure)</span>
    </div>
    <div class="ctx-stack" id="ctxStack"></div>
    <div class="ctx-legend" id="ctxLegend">
      <span><i class="seg-tools"></i>Tools</span>
      <span><i class="seg-system"></i>System</span>
      <span><i class="seg-history"></i>History</span>
      <span><i class="seg-toolResults"></i>Tool results</span>
      <span><i class="seg-request"></i>Request</span>
    </div>
    <div class="stat-grid" id="perfStats"></div>
    <div class="meta" id="perfFoot" style="margin-top:6px">No generation yet</div>
    <div class="toggle" style="margin-top:10px">
      <span>Prompt replacements</span>
      <input type="checkbox" id="promptReplacementsEnabled" title="Strip Copilot system-prompt boilerplate before llama.cpp" />
    </div>
    <div class="meta" id="replacementStats" style="margin-top:4px">Last call: —</div>
    <div class="btn-col" style="margin-top:8px">
      <button class="secondary" id="viewContextBtn" disabled title="Open the last Copilot → llama.cpp request (messages + tools) in an editor">View last call</button>
      <button class="secondary" id="viewResponseBtn" disabled title="Open the last llama.cpp assistant stream (helps debug empty Chat replies)">View last response</button>
    </div>
  </div>

  <div class="setup hidden" id="setupBox">
    <strong>Get a model first</strong>
    <p class="hint" style="margin:8px 0">One-click starter: Unsloth ${STARTER_MODEL.label} (${STARTER_MODEL.approxSizeLabel}, ${STARTER_MODEL.detail}).</p>
    <div class="btn-col" style="margin-top:4px">
      <button class="primary" id="setupStarterBtn">⬇ Download starter (${STARTER_MODEL.label})</button>
    </div>
    <ol>
      <li>Install llama.cpp (once)</li>
      <li>Download the starter <em>or</em> pick/open any GGUF</li>
      <li>Start the server</li>
    </ol>
  </div>

  <h2>llama.cpp</h2>
  <div class="card">
    <div class="row" style="margin-top:0;margin-bottom:6px">
      <div class="label"><span class="name">Backend</span></div>
      <select id="backendSelect" class="wide"></select>
      <div class="hint" id="backendHint"></div>
    </div>
    <div class="btn-col">
      <button class="secondary hidden" id="installLlamaBtn">Upgrade to latest</button>
    </div>
    <details class="advanced" style="margin-top:10px">
      <summary>More install options<span class="sub">pin a tag, local archive, releases</span></summary>
      <div class="meta" id="llamaBinaryDetail" style="margin:6px 0 4px"></div>
      <div class="meta" id="llamaAssetDetail" style="margin:0 0 8px"></div>
      <div class="btn-col">
        <button class="secondary" id="checkUpdatesBtn">Check for updates</button>
        <button class="secondary" id="reinstallLlamaBtn">Reinstall current release</button>
        <button class="secondary" id="installByTagBtn">Install release tag…</button>
        <button class="secondary" id="installArchiveBtn">Install from archive…</button>
      </div>
      <div class="hint" style="margin-top:8px">
        Tag / archive installs skip the GitHub API (useful on shared IPs).
        <a href="https://github.com/ggml-org/llama.cpp/releases" id="releasesLink">Browse releases</a>
      </div>
    </details>
  </div>

  <h2>Model</h2>
  <div class="card">
    <div class="model-title" id="modelTitle">No model selected</div>
    <div class="model-path" id="modelPath"></div>
    <div class="caps hidden" id="modelCaps"></div>
    <div class="meta" id="modelsDirMeta"></div>
    <div class="btn-col">
      <button class="primary hidden" id="starterModelBtn">⬇ Download starter (${STARTER_MODEL.label})</button>
      <div class="hint hidden" id="starterModelHint" style="margin-top:0">${STARTER_MODEL.approxSizeLabel} · ${STARTER_MODEL.detail}</div>
      <button class="primary" id="downloadModelBtn">⬇ Download from Hugging Face…</button>
      <button class="secondary" id="openFileBtn">📂 Open GGUF file…</button>
      <button class="secondary" id="pickDownloadedBtn">📚 Choose from downloaded…</button>
    </div>
    <div class="row" style="margin-top:12px;margin-bottom:0">
      <div class="label"><span class="name tip" data-flag="-mm, --mmproj" data-help="Path to a multimodal projector GGUF. llama-server loads it with the language model so Copilot Chat can send images. Auto-attached when a sibling mmproj-*.gguf sits next to the model.">Vision projector</span></div>
      <div class="hint" id="mmprojPathHint" style="margin:4px 0 8px">No mmproj — text only.</div>
      <div class="btn-row" style="margin:0;gap:8px;flex-wrap:wrap">
        <button class="secondary" id="pickMmprojBtn" type="button">Choose mmproj…</button>
        <button class="secondary" id="clearMmprojBtn" type="button">Clear</button>
      </div>
      <div class="toggle" style="margin-top:8px"><span class="tip" data-flag="--mmproj-offload / --no-mmproj-offload" data-help="Whether to offload the CLIP vision projector to GPU (llama.cpp default: on). Uncheck to pass --no-mmproj-offload and keep the projector in system RAM. Frees VRAM on the Main GPU; image encode becomes CPU-bound.">Offload vision projector to GPU</span><input type="checkbox" id="mmprojOffloadToGpu" checked /></div>
    </div>
  </div>

  <h2>Load settings</h2>

  <div class="card" id="memCard">
    <div class="model-title">Memory estimate</div>
    <div class="hint" style="margin-top:0;margin-bottom:8px">Bars = estimate at <strong>full context</strong>. “Live GPU free” is current occupancy, not the bar.</div>
    <div class="mem-charts" id="memCharts">
      <div>
        <div class="mem-chart-title"><span id="vramChartTitle">VRAM · est. at full context</span><span class="sub" id="vramChartSub">—</span></div>
        <div class="mem-stack" id="vramStack"></div>
      </div>
      <div id="vram2ChartWrap" class="hidden">
        <div class="mem-chart-title"><span id="vram2ChartTitle">VRAM · GPU 1 · est. at full context</span><span class="sub" id="vram2ChartSub">—</span></div>
        <div class="mem-stack" id="vram2Stack"></div>
      </div>
      <div>
        <div class="mem-chart-title"><span id="ramChartTitle">System RAM · est. at full context</span><span class="sub" id="ramChartSub">—</span></div>
        <div class="mem-stack" id="ramStack"></div>
      </div>
      <div class="mem-legend">
        <span><i class="seg-weights"></i>Weights</span>
        <span><i class="seg-vision"></i>Vision (CLIP)</span>
        <span><i class="seg-draft"></i>Spec (MTP/DFlash)</span>
        <span><i class="seg-kv"></i>KV cache</span>
        <span><i class="seg-overhead"></i>Overhead</span>
      </div>
    </div>
    <div class="meta" id="memSummary" style="margin-top:8px">Select a model to estimate VRAM / RAM use.</div>
    <details class="advanced" id="memDetails" style="margin:8px 0 0">
      <summary>Details<span class="sub">capacity, weights, KV</span></summary>
      <div class="meta" id="memLines" style="padding-bottom:8px"></div>
    </details>
    <div class="mem-note hidden" id="memNotes"></div>
    <div class="mem-warn hidden" id="memWarn"></div>
  </div>

  <div class="row" style="margin-bottom:10px">
    <div class="label"><span class="name">Presets</span></div>
    <div class="presets" id="presetChips">
      <button class="chip" id="presetAgent" data-preset="agent" title="Coding agent: q8_0 K + q8_0 V, one slot, 64K context — near-lossless quality with room for tools + history">Coding agent</button>
      <button class="chip" id="presetContext" data-preset="context" title="Max context: q8_0 K + q4_0 V, largest context that still fits your VRAM. K stays at q8_0 because the key cache is far more sensitive to quantization than the value cache.">Max context</button>
      <button class="chip" id="presetQuality" data-preset="quality" title="Max quality: f16 K + q8_0 V at 64K context — spends VRAM on key precision instead of shrinking the context (truncated prompts cost more quality than q8_0 V does).">Max quality</button>
    </div>
    <div class="hint" id="presetHint">Sets context length, KV cache types, and slots together. Reload to apply.</div>
  </div>

  <div class="row">
    <div class="label"><span class="name tip" data-flag="-c, --ctx-size" data-help="Size of the prompt context (default: 0 = loaded from model).">Context Length</span><input type="number" id="contextLength" min="512" step="512" /></div>
    <input type="range" id="contextLengthRange" min="512" max="131072" step="512" />
    <div class="hint" id="ctxHint">Tokens for prompt + generation</div>
  </div>
  <div class="row" id="gpuOffloadRow">
    <div class="label"><span class="name tip" data-flag="-ngl, --n-gpu-layers" data-help="Max number of layers to store in VRAM (exact number, auto, or all).">GPU Offload</span><input type="number" id="gpuOffload" min="0" max="128" /></div>
    <input type="range" id="gpuOffloadRange" min="0" max="128" step="1" />
    <div class="hint" id="gpuOffloadHint">Max = all model layers.</div>
  </div>
  <div class="row hidden" id="dualGpuRow">
    <div class="label"><span class="name tip" data-flag="-mg, --main-gpu" data-help="GPU that holds the compute graph, scratch buffers, and the slider’s share of weights + KV. Index matches llama.cpp --list-devices (Vulkan0, Vulkan1, …), which is often not PCI / btop order.">Main GPU</span></div>
      <select id="mainGpu" class="wide"></select>
    <div class="label" style="margin-top:6px"><span class="name tip" data-flag="-ts, --tensor-split" data-help="Percent of model weights and KV cache on the Main GPU. The rest is split evenly across the other cards. llama.cpp receives this as --tensor-split in --list-devices order. Disabled when Split mode is None (the whole model stays on the Main GPU).">Weights on main GPU</span><span id="tensorSplitPct">75%</span></div>
    <input type="range" id="tensorSplitRange" min="10" max="90" step="1" />
    <div class="label" style="margin-top:6px"><span class="name tip" data-flag="-sm, --split-mode" data-help="How tensors are split. Layer (default) shares the model across cards. Row needs a fast x16 link. None keeps every GPU layer on the Main GPU and leaves the other cards free (--device).">Split mode</span></div>
      <select id="splitMode" class="wide">
        <option value="layer">Layer (default)</option>
        <option value="row">Row</option>
        <option value="none">None — Main GPU only</option>
      </select>
    <div class="hint" id="dualGpuHint">Two GPUs detected. Pick the faster card as Main, then raise the slider to give it more weights.</div>
  </div>
  <div class="row" id="moeRow">
    <div class="label"><span class="name tip" data-flag="-ncmoe, --n-cpu-moe" data-help="Keep the Mixture of Experts (MoE) weights of the first N layers in the CPU.">CPU MoE layers</span><span class="badge">MoE only</span><input type="number" id="nCpuMoe" min="0" max="256" /></div>
    <input type="range" id="nCpuMoeRange" min="0" max="128" step="1" />
    <div class="hint" id="moeHint">Only applies to MoE models.</div>
  </div>

  <details class="advanced">
    <summary>Advanced Settings<span class="sub">threads, batch, KV, RoPE, speculative…</span></summary>

  <div class="subgroup-title">Compute</div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="-t, --threads" data-help="Number of CPU threads to use during generation (default: -1).">CPU Thread Pool Size</span><input type="number" id="cpuThreads" min="1" max="64" /></div>
    <input type="range" id="cpuThreadsRange" min="1" max="64" step="1" />
    <div class="hint" id="cpuThreadsHint">Max = CPU logical cores.</div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="-b, --batch-size" data-help="Logical maximum batch size (default: 2048).">Evaluation Batch Size</span><input type="number" id="evalBatchSize" min="32" step="32" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="-ub, --ubatch-size" data-help="Physical maximum batch size (default: 512).">Physical Batch Size</span><input type="number" id="physicalBatchSize" min="32" step="32" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="-np, --parallel" data-help="Number of server slots (default: -1 = auto). Context is split across slots.">Max Concurrent Predictions</span><span class="badge">Splits context</span><input type="number" id="maxConcurrentPredictions" min="1" max="64" /></div>
    <div class="hint">Use <strong>1</strong> for Copilot Chat. Values &gt; 1 split Context Length across slots (e.g. 8192/4 = 2048 per request).</div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="-fa, --flash-attn" data-help="Flash Attention: 'on', 'off' or 'auto' (default: auto — enabled when the backend supports it).">Flash Attention</span>
      <select id="flashAttention">
        <option value="auto">Auto (default)</option>
        <option value="on">On</option>
        <option value="off">Off</option>
      </select>
    </div>
    <div class="hint">Only sent when not Auto. Forcing <strong>On</strong> can help quantized KV cache; older llama.cpp builds may reject the flag.</div>
  </div>

  <div class="subgroup-title">Memory &amp; KV cache</div>
  <div class="toggle"><span class="tip" data-flag="-nkvo, --no-kv-offload" data-help="Whether to enable KV cache offloading to GPU (default: enabled). Uncheck to keep KV in system RAM.">Offload KV Cache to GPU Memory</span><input type="checkbox" id="offloadKvCacheToGpu" /></div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="-ctk, --cache-type-k" data-help="KV cache data type for K. Allowed: f32, f16, bf16, q8_0, q4_0, … (default: q8_0). q8_0 halves KV size with little quality loss.">KV Cache Type (K)</span>
      <select id="cacheTypeK">
        <option value="q8_0">q8_0 (~½ size, default)</option>
        <option value="f16">f16</option>
        <option value="bf16">bf16</option>
        <option value="q4_0">q4_0 (~¼ size)</option>
      </select>
    </div>
  </div>
  <div class="toggle"><span class="tip" data-flag="-ctk / -ctv" data-help="Keep V on the same type as K. Mixed K/V types fall off the fast attention path and can cost ~40% of prompt and generation throughput.">Use same type for V</span><input type="checkbox" id="kvTypesLinked" checked /></div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="-ctv, --cache-type-v" data-help="KV cache data type for V. Same types as K (default: q8_0). q4_0 saves more VRAM but can hurt long-context prompt speed.">KV Cache Type (V)</span>
      <select id="cacheTypeV">
        <option value="q8_0">q8_0 (~½ size, default)</option>
        <option value="f16">f16</option>
        <option value="bf16">bf16</option>
        <option value="q4_0">q4_0 (~¼ size)</option>
      </select>
    </div>
    <div class="hint hidden" id="kvMismatchHint">V is stored more precisely than K. The key cache is the quantization-sensitive one, so this spends memory where it helps least — prefer K at the higher precision (e.g. q8_0 K with q4_0 V).</div>
    <div class="hint hidden" id="kvFlashAttnHint">A quantized V cache requires Flash Attention. With Flash Attention set to <strong>Off</strong>, llama-server will refuse to start — set V to f16 or put Flash Attention back to Auto/On.</div>
  </div>
  <div class="toggle"><span id="keepModelLabel" class="tip" data-flag="--load-mode mlock" data-help="Force the system to keep the model in RAM rather than swapping (load-mode mlock). On Windows this falls back to mmap.">Keep Model in Memory (--mlock)</span><input type="checkbox" id="keepModelInMemory" /></div>
  <div class="hint" id="keepModelHint" style="display:none">On Windows this uses mmap (--load-mode mmap); mlock is not reliable.</div>
  <div class="toggle"><span class="tip" data-flag="--load-mode mmap | none" data-help="Memory-map the model (mmap). If disabled and mlock is off, uses load-mode none (slower load, may reduce pageouts).">Try mmap()</span><input type="checkbox" id="tryMmap" /></div>
  <div class="toggle"><span class="tip" data-flag="-kvu, --kv-unified" data-help="Use a single unified KV buffer shared across all sequences (default: enabled if slot count is auto). Uncheck passes --no-kv-unified.">Unified KV Cache</span><input type="checkbox" id="unifiedKvCache" /></div>

  <div class="row">
    <div class="label"><span class="name tip" data-flag="-ctxcp, --ctx-checkpoints" data-help="Max number of context checkpoints to create per slot (default: 32). Checkpoints let a slot restore an earlier context state instead of reprocessing it.">Context Checkpoints</span><input type="number" id="contextCheckpoints" min="0" /></div>
    <div class="hint">Only sent when different from 32 (the llama.cpp default), so older builds keep working.</div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="--cache-reuse" data-help="Min chunk size to attempt reusing from the cache via KV shifting (requires prompt caching; llama.cpp default: 0 = off).">Cache Reuse (KV shift)</span><input type="number" id="cacheReuse" min="0" step="32" /></div>
    <div class="hint">Reuses cached chunks after the prompt prefix diverges — raises prompt reuse in long agent threads. 0 disables it.</div>
  </div>

  <div class="subgroup-title">Reasoning</div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="--reasoning-format" data-help="How thoughts are returned: deepseek-legacy keeps <think> tags in content and also fills reasoning_content; deepseek puts thoughts only in reasoning_content; none leaves the raw output untouched.">Reasoning Format</span>
      <select id="reasoningFormat">
        <option value="deepseek-legacy">deepseek-legacy (default)</option>
        <option value="deepseek">deepseek</option>
        <option value="none">none (raw)</option>
        <option value="auto">auto</option>
      </select>
    </div>
    <div class="hint">Copilot Chat only renders <code>content</code> — <strong>deepseek-legacy</strong> keeps thoughts visible there.</div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="--reasoning-budget" data-help="Token budget for thinking: -1 unrestricted, 0 ends thinking immediately, N > 0 caps thinking at N tokens.">Reasoning Budget</span>
      <span class="auto-field">
        <label class="auto"><input type="checkbox" id="reasoningBudgetUnlimited" /> Unlimited</label>
        <input type="number" id="reasoningBudget" min="0" step="128" />
      </span>
    </div>
    <div class="hint">Caps how many tokens a think model may spend before answering. Set <strong>0</strong> to skip thinking entirely.</div>
  </div>

  <div class="subgroup-title">Positional &amp; seed</div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="--rope-freq-base" data-help="RoPE base frequency, used by NTK-aware scaling (default: loaded from model).">RoPE Frequency Base</span>
      <span class="auto-field">
        <label class="auto"><input type="checkbox" id="ropeBaseAuto" /> Auto</label>
        <input type="number" id="ropeFreqBase" step="1" />
      </span>
    </div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="--rope-freq-scale" data-help="RoPE frequency scaling factor; expands context by a factor of 1/N.">RoPE Frequency Scale</span>
      <span class="auto-field">
        <label class="auto"><input type="checkbox" id="ropeScaleAuto" /> Auto</label>
        <input type="number" id="ropeFreqScale" step="0.01" />
      </span>
    </div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="-s, --seed" data-help="RNG seed (default: -1 = random).">Seed</span>
      <span class="auto-field">
        <label class="auto"><input type="checkbox" id="seedRandom" /> Random</label>
        <input type="number" id="seed" step="1" />
      </span>
    </div>
  </div>

  <div class="subgroup-title">Speculative decoding</div>
  <div class="row" id="specModeRow">
    <div class="label"><span class="name tip" data-flag="--spec-type" data-help="Speculative decoding type. MTP uses next-n layers in the main GGUF, or a sibling mtp-*.gguf (Gemma 4) via --model-draft. DFlash uses a separate draft GGUF with --spec-type draft-dflash.">Mode</span>
      <select id="speculativeMode">
        <option value="off">Off</option>
        <option value="mtp" id="specMtpOption">MTP (draft-mtp)</option>
        <option value="dflash" id="specDflashOption">DFlash (draft-dflash)</option>
      </select>
    </div>
    <div class="hint" id="specHint">MTP needs next-n layers in the main GGUF or a sibling mtp-*.gguf. DFlash needs a matching DFlash draft GGUF.</div>
  </div>
  <div class="row hidden" id="specDraftModelRow">
    <div class="label"><span class="name tip" data-flag="-md, --model-draft" data-help="Path to the DFlash draft GGUF (architecture = dflash) or Gemma 4 sidecar MTP GGUF (mtp-*.gguf / architecture = gemma4-assistant).">Draft model</span></div>
    <div class="hint" id="draftModelPathHint" style="margin:4px 0 8px">No draft model selected.</div>
    <div class="hint" id="draftModelKindHint" style="margin:0 0 8px">DFlash needs a <em>separate</em> draft GGUF (<code>architecture = dflash</code>) for your target — not the main model. Gemma 4 MTP uses a sibling <code>mtp-*.gguf</code>.</div>
    <div class="btn-row" style="margin:0 0 8px;gap:8px;flex-wrap:wrap">
      <button class="secondary" id="pickDraftModelBtn" type="button">Choose draft GGUF…</button>
      <button class="secondary" id="clearDraftModelBtn" type="button">Clear</button>
    </div>
  </div>
  <div class="row hidden" id="specDraftNglRow">
    <div class="label"><span class="name tip" data-flag="--spec-draft-ngl" data-help="Max draft-model layers in VRAM (exact number, auto, or all).">Draft GPU Offload</span><input type="number" id="draftGpuOffload" min="0" max="999" /></div>
    <div class="hint" id="draftNglHint">99 usually means all draft layers. DFlash draft KV cache is forced to f16 (quantized draft KV collapses acceptance).</div>
  </div>
  <div class="row" id="specDraftMaxRow">
    <div class="label"><span class="name tip" data-flag="--spec-draft-n-max" data-help="Number of tokens to draft for speculative decoding (default: 3). For DFlash this is clamped to the draft block size (try 8–15).">Max draft tokens</span><input type="number" id="maxDraftTokens" min="0" /></div>
  </div>
  <div class="row" id="specDraftMinRow">
    <div class="label"><span class="name tip" data-flag="--spec-draft-n-min" data-help="Minimum number of draft tokens to use for speculative decoding (default: 0).">Min draft tokens</span><input type="number" id="minDraftTokens" min="0" /></div>
  </div>
  <div class="row" id="specDraftPRow">
    <div class="label"><span class="name tip" data-flag="--spec-draft-p-min" data-help="Minimum speculative decoding probability / greedy threshold (default: 0.00). Mainly for MTP.">Draft probability</span><input type="number" id="draftProbability" min="0" max="1" step="0.01" /></div>
  </div>

  <div class="btn-col" style="margin:12px 0 8px">
    <button class="secondary" id="resetAdvancedBtn" title="Restore Advanced Settings to Llama AIO defaults (does not change Context Length / GPU Offload / MoE)">Reset advanced to defaults</button>
  </div>

  </details>

  <details class="advanced">
    <summary>Request defaults<span class="sub">temperature, top-p/k, max tokens</span></summary>
  <div class="hint hidden" id="modeOverrideHint" style="margin-bottom:10px"></div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="Chat / API request body" data-help="Sampling temperature for completions (extension request default, not a llama-server load flag).">Temperature</span><input type="number" id="temperature" min="0" max="2" step="0.05" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="Chat / API request body" data-help="Nucleus sampling top-p (extension request default).">Top P</span><input type="number" id="topP" min="0" max="1" step="0.01" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="Chat / API request body" data-help="Top-k sampling (extension request default).">Top K</span><input type="number" id="topK" min="0" step="1" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="Chat / API request body" data-help="Max tokens to generate per reply (extension request default / n_predict-style cap).">Max tokens</span><input type="number" id="maxTokens" min="16" step="16" /></div>
  </div>

  <div class="btn-col" style="margin:12px 0 8px">
    <button class="secondary" id="resetRequestBtn" title="Restore temperature, top-p/k, and max tokens to Llama AIO defaults">Reset request defaults</button>
  </div>
  </details>

  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const $ = (id) => document.getElementById(id);

    let memInputs = null;
    let gpuInfo = null;
    let gpuInfos = [];
    let systemRamTotalBytes = 0;
    let mtpSidecarPath = '';
    let backendOptionsCache = [];
    let modelIsMoe = false;
    let modelBlockCount = 128;
    let cpuLogicalCores = 64;
    let moeHintDefault = 'Number of layers to force experts onto CPU (--n-cpu-moe). Only applies to MoE models.';
    let suppressBackendChange = false;
    let activeBackendId = '';
    let serverRunning = false;
    let configDirty = false;
    let serverStarting = false;
    let saveLoadTimer = null;
    let updateCheck = { latestTag: undefined, installedTag: undefined, updateAvailable: false, checkFailed: false, pending: true };

    function setServerCardKind(kind) {
      const card = $('serverCard');
      if (!card) return;
      card.className = 'card server' + (kind && kind !== 'ok' ? ' ' + kind : '');
    }

    function renderStatusUi(opts) {
      const ready = !!opts.ready;
      const dirty = !!opts.dirty;
      const starting = !!opts.starting;
      const endpoint = opts.endpoint || '';
      const pid = opts.pid;
      const message = String(opts.message || '');
      const looksError = !ready && !starting && /error|fail|exit|crash/i.test(message);

      let kind = 'stopped';
      let label = 'Stopped';
      if (starting) {
        kind = 'starting';
        if (/Loading model/i.test(message)) {
          label = 'Loading model…';
        } else if (/Stopping/i.test(message)) {
          label = 'Stopping…';
        } else if (/Reload/i.test(message)) {
          label = 'Reloading…';
        } else {
          label = 'Starting…';
        }
      } else if (ready && dirty) {
        kind = 'dirty';
        label = 'Server ready';
      } else if (ready) {
        kind = 'ok';
        label = 'Server ready';
      } else if (looksError) {
        kind = 'error';
        label = 'Error';
      }

      setServerCardKind(kind);
      const line = $('statusLine');
      const dot = $('statusDot');
      const text = $('statusText');
      const meta = $('statusMeta');
      if (line) line.className = 'status-line ' + kind;
      if (dot) dot.className = 'dot ' + kind;
      if (text) text.textContent = label;

      if (meta) {
        meta.innerHTML = '';
        if (starting) {
          meta.textContent = message || 'Waiting for HTTP ready…';
        } else if (ready && endpoint) {
          const a = document.createElement('a');
          a.className = 'endpoint';
          a.href = endpoint;
          a.setAttribute('data-url', endpoint);
          a.title = 'Open llama-server web UI';
          a.textContent = endpoint;
          a.addEventListener('click', (e) => {
            e.preventDefault();
            const url = a.getAttribute('data-url');
            if (url) vscode.postMessage({ type: 'openExternal', url: url });
          });
          meta.appendChild(a);
          if (pid) {
            const pidEl = document.createElement('span');
            pidEl.className = 'pid';
            pidEl.textContent = ' · pid ' + pid;
            meta.appendChild(pidEl);
          }
        } else if (looksError && message) {
          meta.textContent = message;
        } else if (endpoint && !ready) {
          meta.textContent = endpoint + ' · not running';
        } else {
          meta.textContent = 'No endpoint · not running';
        }
      }

      const hint = $('dirtyHint');
      // Hide dirty while loading — settings are being applied right now.
      if (hint) hint.classList.toggle('hidden', starting || !(ready && dirty));
    }

    function updatePrimaryAction() {
      const primary = $('primaryBtn');
      const stop = $('stopBtn');
      if (!primary || !stop) return;
      stop.disabled = !serverRunning && !serverStarting;
      primary.classList.remove('warn-primary');
      primary.classList.add('primary');
      if (serverStarting) {
        primary.disabled = true;
        primary.textContent = 'Loading…';
        primary.dataset.action = '';
        primary.classList.remove('warn-primary');
        primary.classList.add('primary');
      } else if (!serverRunning) {
        primary.disabled = false;
        primary.textContent = 'Start';
        primary.dataset.action = 'start';
      } else if (configDirty) {
        primary.disabled = false;
        primary.textContent = 'Reload to apply';
        primary.dataset.action = 'reload';
        primary.classList.remove('primary');
        primary.classList.add('warn-primary');
      } else {
        // Status lives in the pill above; keep this slot as a usable action.
        primary.disabled = false;
        primary.textContent = 'Reload';
        primary.dataset.action = 'reload';
      }
    }

    function scheduleSaveLoad() {
      if (saveLoadTimer) clearTimeout(saveLoadTimer);
      highlightPreset();
      // Optimistic dirty UI while running — confirmed via silent save + statusPatch.
      if (serverRunning) {
        configDirty = true;
        updatePrimaryAction();
        const hint = $('dirtyHint');
        if (hint) hint.classList.remove('hidden');
        setServerCardKind('dirty');
        const line = $('statusLine');
        const dot = $('statusDot');
        if (line) line.className = 'status-line dirty';
        if (dot) dot.className = 'dot dirty';
      }
      saveLoadTimer = setTimeout(() => {
        vscode.postMessage({ type: 'saveLoad', payload: readLoad(), silent: true });
      }, 280);
    }

    function updateBackendUi() {
      const sel = $('backendSelect');
      const installBtn = $('installLlamaBtn');
      const hint = $('backendHint');
      const reinstallBtn = $('reinstallLlamaBtn');
      if (!sel || !installBtn || !hint) return;
      const selectedOpt = backendOptionsCache.find((o) => o.id === sel.value);
      const anyInstalled = backendOptionsCache.some((o) => o.installed);
      const installedTag = (selectedOpt && selectedOpt.installedTag) || updateCheck.installedTag;
      const latestTag = updateCheck.latestTag;
      const selectedIsActive = !!(selectedOpt && selectedOpt.active);
      const isPathBackend = sel.value === 'path';

      if (reinstallBtn) {
        reinstallBtn.classList.toggle(
          'hidden',
          isPathBackend || !(selectedOpt && selectedOpt.installed && selectedIsActive)
        );
        reinstallBtn.textContent = installedTag
          ? ('Reinstall ' + installedTag)
          : 'Reinstall current release';
      }

      if (isPathBackend) {
        if (selectedOpt && selectedOpt.installed) {
          hint.textContent =
            'Using llama-server from PATH' +
            (selectedOpt.installedTag ? (' (' + selectedOpt.installedTag + ')') : '') +
            '. Managed by your system package manager.';
        } else {
          hint.textContent =
            (selectedOpt && selectedOpt.reason) ||
            'llama-server not found on PATH. Install a system package (e.g. nixpkgs llama-cpp).';
        }
        installBtn.classList.add('hidden');
        installBtn.disabled = true;
        return;
      }

      if (selectedOpt && selectedOpt.reason && !selectedOpt.available) {
        hint.textContent = selectedOpt.reason;
        installBtn.classList.add('hidden');
        installBtn.disabled = true;
        return;
      }

      if (selectedOpt && !selectedOpt.installed) {
        hint.textContent = 'Not installed — selecting this backend will download it.';
        installBtn.textContent = 'Install ' + (selectedOpt.label || selectedOpt.id);
        installBtn.disabled = false;
        installBtn.classList.remove('hidden');
        installBtn.dataset.action = 'install';
        return;
      }

      if (!selectedOpt) {
        hint.textContent = anyInstalled ? '' : 'No llama.cpp backend installed yet.';
        installBtn.textContent = 'Install llama.cpp';
        installBtn.disabled = false;
        installBtn.classList.toggle('hidden', !!anyInstalled);
        installBtn.dataset.action = 'install';
        return;
      }

      // Installed backend selected
      if (!selectedIsActive) {
        hint.textContent = 'Switching applies immediately. Restart the server if it is running.';
        installBtn.classList.add('hidden');
        return;
      }

      if (updateCheck.pending) {
        hint.textContent = (installedTag ? (installedTag + ' · ') : '') + 'Checking for updates…';
        installBtn.classList.add('hidden');
        return;
      }

      if (updateCheck.checkFailed) {
        hint.textContent = (installedTag ? (installedTag + ' · ') : '') + 'Could not check for updates.';
        installBtn.textContent = 'Check for updates';
        installBtn.disabled = false;
        installBtn.classList.remove('hidden');
        installBtn.dataset.action = 'check';
        return;
      }

      if (updateCheck.updateAvailable && latestTag) {
        hint.textContent = (installedTag ? ('Installed ' + installedTag + ' · ') : '') + 'Update available.';
        installBtn.textContent = 'Upgrade to ' + latestTag;
        installBtn.disabled = false;
        installBtn.classList.remove('hidden');
        installBtn.dataset.action = 'upgrade';
        return;
      }

      hint.textContent = (installedTag || 'Installed') + ' · up to date' + (latestTag ? (' (latest ' + latestTag + ')') : '');
      installBtn.classList.add('hidden');
      installBtn.dataset.action = '';
    }

    function fmtBytes(bytes) {
      const GiB = 1024 ** 3, MiB = 1024 ** 2;
      if (!bytes || bytes <= 0) return '0 B';
      if (bytes >= GiB) return (bytes / GiB).toFixed(bytes >= 10 * GiB ? 1 : 2) + ' GiB';
      if (bytes >= MiB) return (bytes / MiB).toFixed(0) + ' MiB';
      return Math.round(bytes / 1024) + ' KiB';
    }

    function gpuLabel(gpu, index) {
      const name = (gpu && gpu.name) ? String(gpu.name).trim() : '';
      const pretty = name && !/^(amdgpu|nvidia|i915|xe)$/i.test(name) ? name : '';
      const id = (gpu && gpu.llamaDeviceId) ? String(gpu.llamaDeviceId) : ('GPU ' + index);
      return pretty ? (id + ' · ' + pretty) : id;
    }

    function parseTensorSplit(raw) {
      const s = String(raw || '').trim();
      if (!s) return [];
      const parts = s.split(/[,/;:\\s]+/).map(Number).filter((n) => isFinite(n) && n > 0).slice(0, 8);
      return parts.length >= 2 ? parts : [];
    }

    function clampMainGpu(mainGpu, n) {
      return Math.min(Math.max(0, Math.round(Number(mainGpu) || 0)), Math.max(0, n - 1));
    }

    function tensorSplitShares(raw, gpus) {
      const n = Math.max(1, (gpus && gpus.length) || 1);
      if (n === 1) return [1];
      const parsed = parseTensorSplit(raw);
      if (parsed.length >= 2) {
        const parts = parsed.slice(0, n);
        while (parts.length < n) parts.push(0);
        const sum = parts.reduce((a, b) => a + b, 0) || 1;
        return parts.map((p) => p / sum);
      }
      const totals = (gpus || []).map((g) => g.totalBytes || 0);
      while (totals.length < n) totals.push(0);
      const sum = totals.reduce((a, b) => a + b, 0);
      if (sum <= 0) return Array.from({ length: n }, () => 1 / n);
      return totals.map((b) => b / sum);
    }

    function effectiveTensorSplitShares(raw, gpus, splitMode, mainGpu) {
      const n = Math.max(1, (gpus && gpus.length) || 1);
      if (n === 1) return [1];
      if (splitMode === 'none') {
        const main = clampMainGpu(mainGpu, n);
        return Array.from({ length: n }, (_, i) => (i === main ? 1 : 0));
      }
      return tensorSplitShares(raw, gpus);
    }

    function gpuDisplayOrder(gpus, mainGpu) {
      const n = (gpus && gpus.length) || 0;
      const order = [];
      for (let i = 0; i < n; i++) order.push(i);
      if (n < 2) return order;
      const main = clampMainGpu(mainGpu, n);
      if (main > 0) {
        order.splice(main, 1);
        order.unshift(main);
      }
      return order;
    }

    function mainShareFromSplit(raw, mainGpu, gpus) {
      const shares = tensorSplitShares(raw, gpus);
      const n = (gpus && gpus.length) || 1;
      return shares[clampMainGpu(mainGpu, n)] || 1;
    }

    function tensorSplitForMainShare(mainShare, mainGpu, n) {
      n = Math.max(1, Math.round(n) || 1);
      if (n < 2) return '';
      const share = Math.min(0.9, Math.max(0.1, Number(mainShare) || 0.75));
      const main = clampMainGpu(mainGpu, n);
      const mainPct = Math.round(share * 100);
      const restPct = 100 - mainPct;
      const others = n - 1;
      const base = Math.floor(restPct / others);
      let rem = restPct - base * others;
      const percents = [];
      for (let i = 0; i < n; i++) {
        if (i === main) {
          percents.push(mainPct);
        } else {
          const extra = rem > 0 ? 1 : 0;
          if (rem > 0) rem -= 1;
          percents.push(base + extra);
        }
      }
      return percents.join(',');
    }

    function isLegacyGpu0FirstSplit(raw) {
      const n = String(raw || '').replace(/\s+/g, '');
      return n === '3,1' || n === '2,1' || n === '4,1' || n === '3,2';
    }

    function readMainGpuIndex() {
      const n = (gpuInfos && gpuInfos.length) || 1;
      return clampMainGpu(($('mainGpu') && $('mainGpu').value) || 0, n);
    }

    function readTensorSplitFromUi() {
      const n = (gpuInfos && gpuInfos.length) || 1;
      if (n < 2) return '';
      const pct = Number(($('tensorSplitRange') && $('tensorSplitRange').value) || 75);
      return tensorSplitForMainShare(pct / 100, readMainGpuIndex(), n);
    }

    function syncTensorSplitPctLabel() {
      const range = $('tensorSplitRange');
      const lbl = $('tensorSplitPct');
      if (!range || !lbl) return;
      if ($('splitMode') && $('splitMode').value === 'none') {
        lbl.textContent = '100%';
        return;
      }
      lbl.textContent = String(range.value) + '%';
    }

    function syncTensorSplitEnabled() {
      const ts = $('tensorSplitRange');
      const none = $('splitMode') && $('splitMode').value === 'none';
      const cpu = cpuOnlyLive();
      if (ts) {
        ts.disabled = cpu || none;
        ts.style.opacity = (cpu || none) ? '0.45' : '1';
      }
      syncTensorSplitPctLabel();
    }

    function fillMainGpuSelect(selected) {
      const sel = $('mainGpu');
      if (!sel) return;
      const gpus = gpuInfos || [];
      const n = Math.max(gpus.length, 1);
      const want = clampMainGpu(selected, n);
      sel.innerHTML = '';
      for (let i = 0; i < n; i++) {
        const opt = document.createElement('option');
        opt.value = String(i);
        const g = gpus[i];
        opt.textContent = g
          ? (gpuLabel(g, i) + (g.totalBytes ? ' · ' + fmtBytes(g.totalBytes) : ''))
          : ('GPU ' + i);
        sel.appendChild(opt);
      }
      sel.value = String(want);
    }

    function buildGpuChart(index, gpu, weights, kv, overhead, spec, specLabel, labeled, vision) {
      return {
        title: labeled && gpu
          ? ('VRAM · ' + gpuLabel(gpu, index) + ' · est. at full context')
          : 'VRAM · est. at full context',
        segments: [
          { key: 'weights', label: 'Weights', bytes: weights },
          { key: 'vision', label: 'Vision (CLIP)', bytes: vision || 0 },
          { key: 'draft', label: specLabel || 'Speculative', bytes: spec || 0 },
          { key: 'kv', label: 'KV cache (full ctx)', bytes: kv },
          { key: 'overhead', label: 'Overhead', bytes: overhead },
        ],
        totalBytes: weights + kv + overhead + (spec || 0) + (vision || 0),
        capacityBytes: gpu && gpu.totalBytes ? gpu.totalBytes : undefined,
      };
    }

    function buildCharts(gpuWeights, cpuWeights, kvBytes, kvOnGpu, gpuOverhead, cpuOverhead, totalGpu, totalCpu, draftGpu, draftCpu, specLabel, split, gpuVision, cpuVision) {
      const gpus = (!cpuOnlyLive() && gpuInfos && gpuInfos.length) ? gpuInfos : (gpuInfo ? [gpuInfo] : []);
      const shares = effectiveTensorSplitShares(
        split && split.tensorSplit,
        gpus,
        split && split.splitMode,
        split && split.mainGpu
      );
      const mainIdx = gpus.length ? clampMainGpu(split && split.mainGpu, gpus.length) : 0;
      const gpuKv = kvOnGpu ? kvBytes : 0;
      const labeled = gpus.length >= 2;
      const order = gpuDisplayOrder(gpus, mainIdx);
      const i0 = order[0];
      const i1 = order[1];
      const vram0 = gpus.length && i0 !== undefined && gpus[i0]
        ? buildGpuChart(
            i0,
            gpus[i0],
            gpuWeights * (shares[i0] || 0),
            gpuKv * (shares[i0] || 0),
            mainIdx === i0 ? gpuOverhead : 0,
            mainIdx === i0 ? (draftGpu || 0) : 0,
            specLabel,
            labeled,
            mainIdx === i0 ? (gpuVision || 0) : 0
          )
        : {
            title: 'VRAM · est. at full context',
            segments: [
              { key: 'weights', label: 'Weights', bytes: gpuWeights },
              { key: 'vision', label: 'Vision (CLIP)', bytes: gpuVision || 0 },
              { key: 'draft', label: specLabel || 'Speculative', bytes: draftGpu || 0 },
              { key: 'kv', label: 'KV cache (full ctx)', bytes: gpuKv },
              { key: 'overhead', label: 'Overhead', bytes: gpuOverhead },
            ],
            totalBytes: totalGpu,
            capacityBytes: gpuInfo && gpuInfo.totalBytes ? gpuInfo.totalBytes : undefined,
          };
      const vram2 = i1 !== undefined && gpus[i1]
        ? buildGpuChart(
            i1,
            gpus[i1],
            gpuWeights * (shares[i1] || 0),
            gpuKv * (shares[i1] || 0),
            mainIdx === i1 ? gpuOverhead : 0,
            mainIdx === i1 ? (draftGpu || 0) : 0,
            specLabel,
            labeled,
            mainIdx === i1 ? (gpuVision || 0) : 0
          )
        : undefined;
      return {
        vram: vram0,
        vram2,
        ram: {
          title: 'System RAM · est. at full context',
          segments: [
            { key: 'weights', label: 'Weights', bytes: cpuWeights },
            { key: 'vision', label: 'Vision (CLIP)', bytes: cpuVision || 0 },
            { key: 'draft', label: specLabel || 'Speculative', bytes: draftCpu || 0 },
            { key: 'kv', label: 'KV cache (full ctx)', bytes: kvOnGpu ? 0 : kvBytes },
            { key: 'overhead', label: 'Overhead', bytes: cpuOverhead },
          ],
          totalBytes: totalCpu,
          capacityBytes: systemRamTotalBytes || undefined,
        },
      };
    }

    function cpuOnlyLive() {
      const sel = $('backendSelect');
      return !!(sel && sel.value === 'cpu');
    }

    function moeExpertShareOf(inputs) {
      if (!inputs || !inputs.isMoe) return 0;
      if (inputs.moeExpertShare != null && isFinite(inputs.moeExpertShare)) {
        return Math.min(0.98, Math.max(0.05, Number(inputs.moeExpertShare)));
      }
      const n = Number(inputs.expertCount) || 0;
      if (n >= 128) return 0.9;
      if (n >= 64) return 0.85;
      if (n >= 16) return 0.8;
      return 0.75;
    }

    function renderContextStack(perf) {
      const stack = $('ctxStack');
      const label = $('ctxLabel');
      const sub = $('ctxSub');
      if (!stack || !label || !sub) return;

      const breakdown = perf && perf.contextBreakdown;
      const ctxPct = typeof perf?.contextPct === 'number' ? perf.contextPct : undefined;
      const ctxLevel = perf?.contextLevel || 'ok';
      const promptTokens = typeof perf?.promptTokens === 'number' ? perf.promptTokens : undefined;
      const contextLimit = typeof perf?.contextLimit === 'number' ? perf.contextLimit : undefined;
      const approx = perf?.contextEstimated ? '≈' : '';

      stack.className = 'ctx-stack' + (ctxLevel === 'warn' || ctxLevel === 'critical' ? ' ' + ctxLevel : '');
      label.className = '';
      if (ctxLevel === 'warn' || ctxLevel === 'critical') {
        label.style.color = ctxLevel === 'critical' ? '#f85149' : '#d29922';
      } else {
        label.style.color = '';
      }

      if (typeof promptTokens === 'number' && typeof contextLimit === 'number' && typeof ctxPct === 'number') {
        label.textContent = 'Context';
        sub.textContent =
          approx + promptTokens.toLocaleString() + ' / ' +
          contextLimit.toLocaleString() + ' (' + ctxPct + '%)' +
          (ctxLevel === 'critical' ? ' · nearly full' : ctxLevel === 'warn' ? ' · running low' : '');
      } else {
        label.textContent = 'Context';
        sub.textContent = '— (send a chat to measure)';
      }

      stack.innerHTML = '';
      if (!breakdown || !Array.isArray(breakdown.segments)) {
        return;
      }
      const scale = Math.max(1, breakdown.limitTokens || contextLimit || 1);
      for (const seg of breakdown.segments) {
        if (!seg || seg.key === 'free' || !(seg.tokens > 0)) continue;
        const el = document.createElement('span');
        el.className = 'seg-' + seg.key;
        el.style.width = Math.max(0.4, (seg.tokens / scale) * 100) + '%';
        const pct = Math.round((seg.tokens / scale) * 1000) / 10;
        el.title = seg.label + ': ≈' + Number(seg.tokens).toLocaleString() + ' tok (' + pct + '% of slot)';
        stack.appendChild(el);
      }
    }

    function fmtNum(n) {
      return typeof n === 'number' && isFinite(n) ? Number(n).toLocaleString() : '—';
    }

    function fmtRate(n) {
      if (typeof n !== 'number' || !isFinite(n) || n <= 0) return undefined;
      return n >= 100 ? n.toFixed(0) : n.toFixed(1);
    }

    function statTile(key, value, sub, kind) {
      return '<div class="stat' + (kind ? ' ' + kind : '') + '">' +
        '<div class="k">' + key + '</div>' +
        '<div class="v">' + value + '</div>' +
        (sub ? '<div class="s">' + sub + '</div>' : '<div class="s">&nbsp;</div>') +
        '</div>';
    }

    function renderPerfStats(perf, perfLines) {
      const grid = $('perfStats');
      const foot = $('perfFoot');
      if (!grid) return;
      const p = perf || {};
      const tiles = [];

      const gen = fmtRate(p.genTokPerSec);
      tiles.push(gen
        ? statTile('Generation', gen + ' tok/s', p.estimated ? 'estimated' : 'from server timings')
        : statTile('Generation', '—', p.generating ? 'measuring…' : 'no generation yet', 'empty'));

      const prompt = fmtRate(p.promptTokPerSec);
      tiles.push(prompt
        ? statTile('Prompt', prompt + ' tok/s', 'prompt processing')
        : statTile('Prompt', '—', 'prompt processing', 'empty'));

      if (typeof p.cacheHitPct === 'number') {
        tiles.push(statTile(
          'Prompt reuse',
          p.cacheHitPct + '%',
          fmtNum(p.cachedPromptTokens) + ' cached · ' + fmtNum(p.processedPromptTokens) + ' new',
          p.cacheHitPct >= 50 ? 'good' : undefined
        ));
      } else {
        tiles.push(statTile('Prompt reuse', '—', 'KV prefix cache (cache_n)', 'empty'));
      }

      if (p.speculativeMode === 'mtp' || p.speculativeMode === 'dflash') {
        const label = p.speculativeMode === 'dflash' ? 'DFlash accepted' : 'MTP accepted';
        if (typeof p.draftAcceptancePct === 'number') {
          tiles.push(statTile(
            label,
            p.draftAcceptancePct.toFixed(1) + '%',
            fmtNum(p.draftTokensAccepted) + ' / ' + fmtNum(p.draftTokens) + ' drafted',
            p.draftAcceptancePct >= 50 ? 'good' : undefined
          ));
        } else {
          tiles.push(statTile(label, '—', 'no draft tokens yet', 'empty'));
        }
      } else {
        tiles.push(statTile('Spec accepted', '—', 'speculative off', 'empty'));
      }

      grid.innerHTML = tiles.join('');
      grid.title = Array.isArray(perfLines) ? perfLines.join('\\n') : '';

      if (foot) {
        const parts = [];
        if (p.generating) parts.push('<span class="ok">● Generating…</span>');
        if (typeof p.completionTokens === 'number') {
          parts.push(fmtNum(p.completionTokens) + ' completion tokens');
        }
        if (!p.generating && p.finishedAt && p.startedAt) {
          parts.push(((p.finishedAt - p.startedAt) / 1000).toFixed(1) + 's');
        }
        foot.innerHTML = parts.length ? parts.join(' · ') : 'No generation yet';
      }
    }

    function renderStackedBar(stackId, subId, chart) {
      const stack = $(stackId);
      const sub = $(subId);
      if (!chart) {
        stack.innerHTML = '';
        stack.classList.remove('over', 'warn');
        sub.className = 'sub';
        sub.textContent = '—';
        return;
      }
      const capacity = chart.capacityBytes || chart.totalBytes || 1;
      const scale = Math.max(capacity, chart.totalBytes || 0) || 1;
      stack.innerHTML = '';
      for (const seg of chart.segments || []) {
        if (!seg.bytes || seg.bytes <= 0) continue;
        const el = document.createElement('span');
        el.className = 'seg-' + seg.key;
        el.style.width = Math.max(0.5, (seg.bytes / scale) * 100) + '%';
        el.title = seg.label + ': ~' + fmtBytes(seg.bytes);
        stack.appendChild(el);
      }
      const pct = chart.capacityBytes
        ? Math.round((chart.totalBytes / chart.capacityBytes) * 100)
        : undefined;
      // 92% is the usable ceiling used by the spill warnings (driver headroom).
      const over = pct !== undefined && pct > 92;
      const warn = !over && pct !== undefined && pct > 80;
      stack.classList.toggle('over', over);
      stack.classList.toggle('warn', warn);
      sub.className = 'sub' + (over ? ' over' : warn ? ' warn' : '');
      sub.textContent =
        '~' + fmtBytes(chart.totalBytes) +
        (chart.capacityBytes
          ? ' / ' + fmtBytes(chart.capacityBytes) + (pct !== undefined ? ' (' + pct + '%)' : '')
          : '');
    }

    function liveMemoryEstimate() {
      if (!memInputs || !memInputs.fileSizeBytes) return null;
      const L = readLoad();
      const nLayers = Math.max(1, memInputs.blockCount || 1);
      // Follow the dropdown (pending switch), not a sticky flag.
      const cpuOnly = $('backendSelect').value === 'cpu';
      let onGpu = cpuOnly ? 0 : (L.gpuOffload <= 0 ? 0 : (L.gpuOffload >= 99 ? nLayers : Math.min(L.gpuOffload, nLayers)));
      const expertShare = moeExpertShareOf(memInputs);
      let gpuWeights = memInputs.fileSizeBytes * (onGpu / nLayers);
      if (!cpuOnly && memInputs.isMoe && L.nCpuMoe > 0 && onGpu > 0 && expertShare > 0) {
        const moeCpu = Math.min(L.nCpuMoe, onGpu);
        gpuWeights = Math.max(0, gpuWeights - memInputs.fileSizeBytes * (moeCpu / nLayers) * expertShare);
      }
      let cpuWeights = Math.max(0, memInputs.fileSizeBytes - gpuWeights);
      const mmprojBytes = Math.max(0, Number(memInputs.mmprojFileSizeBytes) || 0);
      const gpuVisionBytes = !cpuOnly && onGpu > 0 && mmprojBytes > 0 && L.mmprojOffloadToGpu !== false ? mmprojBytes : 0;
      const cpuVisionBytes = gpuVisionBytes > 0 ? 0 : mmprojBytes;
      const heads = Math.max(1, memInputs.attentionHeadCount || 8);
      const defaultKvHeads = Math.max(1, memInputs.attentionHeadCountKv || heads);
      const defaultKeyDim = Math.max(1, memInputs.keyLength || Math.floor((memInputs.embeddingLength || heads * 128) / heads));
      const defaultValDim = Math.max(1, memInputs.valueLength || defaultKeyDim);
      const swa = memInputs.slidingWindow > 0 ? memInputs.slidingWindow : 0;
      const pattern = memInputs.slidingWindowPattern;
      const perKv = memInputs.attentionHeadCountKvPerLayer;
      const recurrent = memInputs.recurrentLayers;
      const fullInterval = memInputs.fullAttentionInterval > 1 ? memInputs.fullAttentionInterval : 0;
      function kvElemBytes(t) {
        if (t === 'q4_0') return 0.5;
        if (t === 'q8_0') return 1;
        return 2; // f16 / bf16
      }
      function kvAt(ctx) {
        const kBytes = kvElemBytes(L.cacheTypeK);
        const vBytes = kvElemBytes(L.cacheTypeV);
        let bytes = 0;
        let fullAttnLayers = 0;
        for (let i = 0; i < nLayers; i++) {
          const isRecurrent = (recurrent && recurrent.length === nLayers)
            ? !!recurrent[i]
            : !!(fullInterval && ((i + 1) % fullInterval !== 0));
          if (isRecurrent) continue;
          fullAttnLayers++;
          const isSwa = !!(swa && pattern && pattern.length && pattern[i % pattern.length]);
          const nKv = Math.max(1, (perKv && perKv[i]) || defaultKvHeads);
          const keyDim = (isSwa && memInputs.keyLengthSwa > 0) ? memInputs.keyLengthSwa : defaultKeyDim;
          const valDim = (isSwa && memInputs.valueLengthSwa > 0) ? memInputs.valueLengthSwa : defaultValDim;
          const tokens = isSwa ? Math.min(ctx, swa) : ctx;
          bytes += (nKv * keyDim * kBytes + nKv * valDim * vBytes) * tokens;
        }
        return { bytes, fullAttnLayers };
      }
      const fullKv = kvAt(L.contextLength);
      const warmCtx = Math.min(2048, Math.max(512, L.contextLength));
      const warmKv = kvAt(warmCtx);
      const kvBytes = fullKv.bytes;
      const kvBytesWarm = warmKv.bytes;
      const fullAttnLayers = fullKv.fullAttnLayers;
      const kvOnGpu = !cpuOnly && !!L.offloadKvCacheToGpu && onGpu > 0;
      // Mirrors computeOverheadBytes() in memoryEstimate.ts.
      const embedForOverhead = Math.max(2048, memInputs.embeddingLength || 4096);
      const ubatchForOverhead = Math.min(Math.max(32, L.physicalBatchSize || 512), 8192);
      const batchForOverhead = Math.min(Math.max(32, L.evalBatchSize || 2048), 8192);
      const overhead = Math.round(
        400 * 1024 * 1024 + ubatchForOverhead * embedForOverhead * 24 + batchForOverhead * 8 * 1024
      );
      const gpuOverhead = onGpu > 0 ? overhead : 0;
      const cpuOverhead = onGpu > 0 ? Math.round(overhead * 0.15) : Math.round(overhead * 0.5);

      const warnings = [];
      let willSpill = false;

      // DFlash draft weights + f16 KV (mirrors estimateMemory draft footprint).
      let draftGpuBundle = 0;
      let draftCpuBundle = 0;
      let draftGpuWarmBundle = 0;
      let draftCpuWarmBundle = 0;
      let draftLine = '';
      const sidecarMtp = L.speculativeMode === 'mtp' && memInputs.draft && memInputs.draft.fileSizeBytes && !(Number(memInputs.nextnPredictLayers) > 0);
      const draftIn = ((L.speculativeMode === 'dflash' || sidecarMtp) && memInputs.draft && memInputs.draft.fileSizeBytes)
        ? memInputs.draft
        : null;
      if (draftIn) {
        const dLayers = Math.max(1, draftIn.blockCount || 1);
        const dOff = Number(L.draftGpuOffload);
        const dOnGpu = cpuOnly ? 0 : (dOff <= 0 ? 0 : (dOff >= 99 ? dLayers : Math.min(dOff, dLayers)));
        const dGpuW = draftIn.fileSizeBytes * (dOnGpu / dLayers);
        const dCpuW = Math.max(0, draftIn.fileSizeBytes - dGpuW);
        function draftKvAt(ctx) {
          const kBytes = sidecarMtp ? kvElemBytes(L.cacheTypeK) : 2; // DFlash forces f16
          const vBytes = sidecarMtp ? kvElemBytes(L.cacheTypeV) : 2;
          const heads = Math.max(1, draftIn.attentionHeadCount || 8);
          const defaultKvHeads = Math.max(1, draftIn.attentionHeadCountKv || heads);
          const defaultKeyDim = Math.max(1, draftIn.keyLength || Math.floor((draftIn.embeddingLength || heads * 128) / heads));
          const defaultValDim = Math.max(1, draftIn.valueLength || defaultKeyDim);
          const swa = draftIn.slidingWindow > 0 ? draftIn.slidingWindow : 0;
          const pattern = draftIn.slidingWindowPattern;
          const perKv = draftIn.attentionHeadCountKvPerLayer;
          const recurrent = draftIn.recurrentLayers;
          const fullInterval = draftIn.fullAttentionInterval > 1 ? draftIn.fullAttentionInterval : 0;
          let bytes = 0;
          for (let i = 0; i < dLayers; i++) {
            const isRecurrent = (recurrent && recurrent.length === dLayers)
              ? !!recurrent[i]
              : !!(fullInterval && ((i + 1) % fullInterval !== 0));
            if (isRecurrent) continue;
            const isSwa = !!(swa && pattern && pattern.length && pattern[i % pattern.length]);
            const nKv = Math.max(1, (perKv && perKv[i]) || defaultKvHeads);
            const keyDim = (isSwa && draftIn.keyLengthSwa > 0) ? draftIn.keyLengthSwa : defaultKeyDim;
            const valDim = (isSwa && draftIn.valueLengthSwa > 0) ? draftIn.valueLengthSwa : defaultValDim;
            const tokens = isSwa ? Math.min(ctx, swa) : ctx;
            bytes += (nKv * keyDim * kBytes + nKv * valDim * vBytes) * tokens;
          }
          return bytes;
        }
        const dKv = draftKvAt(L.contextLength);
        const dKvWarm = draftKvAt(warmCtx);
        const dKvOnGpu = dOnGpu > 0;
        draftGpuBundle = dGpuW + (dKvOnGpu ? dKv : 0);
        draftCpuBundle = dCpuW + (dKvOnGpu ? 0 : dKv);
        draftGpuWarmBundle = dGpuW + (dKvOnGpu ? dKvWarm : 0);
        draftCpuWarmBundle = dCpuW + (dKvOnGpu ? 0 : dKvWarm);
        draftLine = (sidecarMtp ? 'MTP sidecar: ~' : 'DFlash draft: ~') + fmtBytes(dGpuW) + ' GPU / ~' + fmtBytes(dCpuW) + ' RAM weights (' +
          dOnGpu + '/' + dLayers + ' layers) · draft KV ~' + fmtBytes(dKv) + (sidecarMtp ? '' : ' f16') +
          (dKvOnGpu ? ' (GPU)' : ' (CPU RAM)');
        warnings.push(
          (sidecarMtp ? 'MTP sidecar included: ~' : 'DFlash draft included: ~') + fmtBytes(draftIn.fileSizeBytes) + ' weights (' +
          dOnGpu + '/' + dLayers + ' GPU layers) + ~' + fmtBytes(dKv) + ' draft KV' + (sidecarMtp ? '' : ' (f16)') + ' at full context.'
        );
      } else if (L.speculativeMode === 'dflash') {
        warnings.push('DFlash is on but no draft GGUF is selected — memory bars omit the draft; pick a draft model before starting.');
      } else if (L.speculativeMode === 'mtp' && !sidecarMtp) {
        const mtpLayers = Math.max(0, Math.floor(Number(memInputs.nextnPredictLayers) || 0));
        if (mtpLayers > 0) {
          const mtpWeights = memInputs.fileSizeBytes * (mtpLayers / nLayers);
          function mtpKvAt(ctx) {
            const kBytes = kvElemBytes(L.cacheTypeK);
            const vBytes = kvElemBytes(L.cacheTypeV);
            const heads = Math.max(1, memInputs.attentionHeadCount || 8);
            const nKv = Math.max(1, memInputs.attentionHeadCountKv || heads);
            const keyDim = Math.max(1, memInputs.keyLength || Math.floor((memInputs.embeddingLength || heads * 128) / heads));
            const valDim = Math.max(1, memInputs.valueLength || keyDim);
            return mtpLayers * (nKv * keyDim * kBytes + nKv * valDim * vBytes) * ctx;
          }
          const mtpKv = mtpKvAt(L.contextLength);
          const mtpKvWarm = mtpKvAt(warmCtx);
          const mtpWeightsOnGpu = !cpuOnly && onGpu > 0;
          const mtpKvOnGpu = kvOnGpu;
          draftGpuBundle = (mtpWeightsOnGpu ? mtpWeights : 0) + (mtpKvOnGpu ? mtpKv : 0);
          draftCpuBundle = (mtpWeightsOnGpu ? 0 : mtpWeights) + (mtpKvOnGpu ? 0 : mtpKv);
          draftGpuWarmBundle = (mtpWeightsOnGpu ? mtpWeights : 0) + (mtpKvOnGpu ? mtpKvWarm : 0);
          draftCpuWarmBundle = (mtpWeightsOnGpu ? 0 : mtpWeights) + (mtpKvOnGpu ? 0 : mtpKvWarm);
          draftLine = 'MTP: ~' + fmtBytes(mtpWeights) + ' next-n head (' + mtpLayers +
            ' layers) · MTP KV ~' + fmtBytes(mtpKv) + (mtpKvOnGpu ? ' (GPU)' : ' (CPU RAM)');
          warnings.push(
            'MTP overhead included: ~' + fmtBytes(mtpWeights) + ' next-n head (' + mtpLayers +
            ' layers) + ~' + fmtBytes(mtpKv) + ' MTP KV at full context.'
          );
        } else {
          warnings.push('MTP is on but this GGUF reports no nextn_predict_layers — speculative overhead omitted from the bars.');
        }
      }

      if (mmprojBytes > 0) {
        const gpusForVision = (!cpuOnly && gpuInfos && gpuInfos.length) ? gpuInfos : [];
        const mainIdxForVision = gpusForVision.length
          ? Math.min(Math.max(0, Number(L.mainGpu) || 0), gpusForVision.length - 1)
          : 0;
        const where = gpuVisionBytes > 0
          ? (gpusForVision.length
            ? ' on ' + gpuLabel(gpusForVision[mainIdxForVision], mainIdxForVision) + ' (CLIP / --mmproj, not tensor-split)'
            : ' in VRAM (--mmproj)')
          : (L.mmprojOffloadToGpu === false ? ' in system RAM (--no-mmproj-offload)' : ' in system RAM');
        warnings.push('Vision projector included: ~' + fmtBytes(mmprojBytes) + where + '.');
      }

      const totalGpu = gpuWeights + (kvOnGpu ? kvBytes : 0) + gpuOverhead + draftGpuBundle + gpuVisionBytes;
      const totalCpu = cpuWeights + (kvOnGpu ? 0 : kvBytes) + cpuOverhead + draftCpuBundle + cpuVisionBytes;
      const totalGpuWarm = gpuWeights + (kvOnGpu ? kvBytesWarm : 0) + gpuOverhead + draftGpuWarmBundle + gpuVisionBytes;
      const totalCpuWarm = cpuWeights + (kvOnGpu ? 0 : kvBytesWarm) + cpuOverhead + draftCpuWarmBundle + cpuVisionBytes;
      if (cpuOnly) {
        warnings.push('CPU backend: no GPU acceleration — weights, KV cache, and compute use system RAM (GPU Offload is ignored).');
      }
      if (!cpuOnly && onGpu > 0 && onGpu < nLayers) {
        warnings.push('Partial GPU offload: ' + (nLayers - onGpu) + '/' + nLayers + ' layers (~' + fmtBytes(cpuWeights) + ') stay in system RAM.');
      }
      if (!cpuOnly && onGpu === 0) warnings.push('GPU offload is 0 — weights run from system RAM.');
      if (!cpuOnly && !L.offloadKvCacheToGpu) warnings.push('KV cache (~' + fmtBytes(kvBytes) + ' at full context) is in system RAM.');
      if (!cpuOnly && memInputs.isMoe && L.nCpuMoe > 0) {
        warnings.push('CPU MoE layers = ' + L.nCpuMoe + ': ~' + Math.round(expertShare * 100) + '% of weights are experts; those layers’ experts stay in system RAM.');
      }
      if (!cpuOnly && gpuInfos && gpuInfos.length) {
        const shares = effectiveTensorSplitShares(L.tensorSplit, gpuInfos, L.splitMode, L.mainGpu);
        const mainIdx = Math.min(Math.max(0, Number(L.mainGpu) || 0), gpuInfos.length - 1);
        const gpuKv = kvOnGpu ? kvBytes : 0;
        for (let i = 0; i < gpuInfos.length; i++) {
          const g = gpuInfos[i];
          const share = shares[i] || 0;
          const used = gpuWeights * share + gpuKv * share + (i === mainIdx ? gpuOverhead : 0) + (i === mainIdx ? draftGpuBundle : 0) + (i === mainIdx ? gpuVisionBytes : 0);
          const cap = g.totalBytes;
          if (!cap) continue;
          const pct = Math.round((used / cap) * 100);
          const label = gpuLabel(g, i);
          if (used > cap) {
            willSpill = true;
            warnings.unshift('Estimated ' + label + ' at full context ~' + fmtBytes(used) + ' is over the full ' + fmtBytes(cap) + ' (' + pct + '%). Expect spill to system RAM. Lower Context or GPU Offload.');
          } else if (used > cap * 0.92) {
            willSpill = true;
            warnings.unshift('Tight on ' + label + ' at full context: ~' + fmtBytes(used) + ' of ' + fmtBytes(cap) + ' (' + pct + '%). Only ~' + fmtBytes(cap - cap * 0.92) + ' safe headroom for the driver — often spills to system RAM. Lower Context or GPU Offload.');
          } else if (used > cap * 0.8) {
            warnings.push('Getting full on ' + label + ' at full context: ~' + fmtBytes(used) + ' of ' + fmtBytes(cap) + ' VRAM (' + pct + '%).');
          }
        }
        if (gpuInfos.length >= 2 && L.splitMode !== 'none' && parseTensorSplit(L.tensorSplit).length < 2) {
          warnings.push('Tensor split is empty — llama.cpp will split by VRAM size (often 1:1). Pick the faster card as Main GPU and raise Weights on main GPU so that card gets more of the model.');
        }
      } else if (!cpuOnly && gpuInfo && gpuInfo.totalBytes) {
        const usable = gpuInfo.totalBytes * 0.92;
        const pct = Math.round((totalGpu / gpuInfo.totalBytes) * 100);
        if (totalGpu > gpuInfo.totalBytes) {
          willSpill = true;
          warnings.unshift('Estimated VRAM at full context ~' + fmtBytes(totalGpu) + ' is over the full ' + fmtBytes(gpuInfo.totalBytes) + ' GPU (' + pct + '%). Expect spill to system RAM. Lower Context or GPU Offload.');
        } else if (totalGpu > usable) {
          willSpill = true;
          warnings.unshift('Tight on VRAM at full context: ~' + fmtBytes(totalGpu) + ' of ' + fmtBytes(gpuInfo.totalBytes) + ' (' + pct + '%). Only ~' + fmtBytes(gpuInfo.totalBytes - usable) + ' safe headroom for the driver — often spills to system RAM. Lower Context or GPU Offload.');
        } else if (totalGpu > gpuInfo.totalBytes * 0.8) {
          warnings.push('Getting full at full context: ~' + fmtBytes(totalGpu) + ' of ' + fmtBytes(gpuInfo.totalBytes) + ' VRAM (' + pct + '%).');
        }
      }
      if (cpuOnly && systemRamTotalBytes && totalCpu > systemRamTotalBytes * 0.9) {
        willSpill = true;
        warnings.unshift('Estimated system RAM at full context ~' + fmtBytes(totalCpu) + ' is very high vs ' + fmtBytes(systemRamTotalBytes) + '. Lower Context Length or use a smaller model/quant.');
      }
      const lines = [];
      if (cpuOnly) {
        lines.push('Backend: CPU — GPU Offload / VRAM not used');
      } else if (gpuInfos && gpuInfos.length) {
        for (let i = 0; i < gpuInfos.length; i++) {
          const g = gpuInfos[i];
          lines.push(gpuLabel(g, i) + ' capacity: ' + fmtBytes(g.totalBytes));
          if (g.usedBytes != null) {
            const free = Math.max(0, g.totalBytes - g.usedBytes);
            lines.push('Live ' + gpuLabel(g, i) + ' free now: ~' + fmtBytes(free) + ' (current occupancy — not part of the estimate bars)');
          }
        }
        if (gpuInfos.length >= 2) {
          const mainIdx = clampMainGpu(L.mainGpu, gpuInfos.length);
          const mainG = gpuInfos[mainIdx];
          const mainLabel = mainG ? gpuLabel(mainG, mainIdx) : ('GPU ' + mainIdx);
          if (L.splitMode === 'none') {
            lines.push('No GPU split — all GPU layers on ' + mainLabel + ' (--split-mode none)');
          } else {
            const split = parseTensorSplit(L.tensorSplit);
            lines.push('Tensor split: ' + (split.length >= 2 ? L.tensorSplit : 'auto (by VRAM)') + ' · split-mode ' + (L.splitMode || 'layer') + ' · main ' + mainLabel);
          }
        }
      } else {
        lines.push('GPU VRAM: unknown');
      }
      if (systemRamTotalBytes) lines.push('System RAM capacity: ' + fmtBytes(systemRamTotalBytes));
      if (cpuOnly) {
        lines.push('Weights in RAM: ~' + fmtBytes(cpuWeights) + ' (' + nLayers + ' layers)');
        lines.push('KV @ full ' + Number(L.contextLength).toLocaleString() + ' ctx: ~' + fmtBytes(kvBytes) + ' (system RAM)' +
          (fullAttnLayers < nLayers ? (' · ' + fullAttnLayers + '/' + nLayers + ' full-attn layers') : ''));
        if (draftLine) lines.push(draftLine);
        if (mmprojBytes > 0) lines.push('Vision projector in RAM: ~' + fmtBytes(mmprojBytes));
        if (kvBytesWarm < kvBytes) {
        }
        lines.push('Est. total system RAM at full context: ~' + fmtBytes(totalCpu));
      } else {
        lines.push('Weights on GPU: ~' + fmtBytes(gpuWeights) + ' (' + onGpu + '/' + nLayers + ' layers)' + (cpuWeights > 1024*1024 ? ' · RAM: ~' + fmtBytes(cpuWeights) : '') +
          (memInputs.isMoe && expertShare > 0 ? (' · MoE experts ~' + Math.round(expertShare * 100) + '% of file') : ''));
        lines.push('KV @ full ' + Number(L.contextLength).toLocaleString() + ' ctx: ~' + fmtBytes(kvBytes) + (kvOnGpu ? ' (GPU)' : ' (CPU RAM)') +
          (fullAttnLayers < nLayers ? (' · ' + fullAttnLayers + '/' + nLayers + ' full-attn layers') : ''));
        if (draftLine) lines.push(draftLine);
        if (mmprojBytes > 0) {
          const mainIdx = (gpuInfos && gpuInfos.length) ? clampMainGpu(L.mainGpu, gpuInfos.length) : 0;
          const mainG = gpuInfos && gpuInfos[mainIdx];
          lines.push('Vision projector: ~' + fmtBytes(mmprojBytes) + (gpuVisionBytes > 0
            ? (mainG ? ' (' + gpuLabel(mainG, mainIdx) + ', CLIP / --mmproj)' : ' (GPU, --mmproj)')
            : (L.mmprojOffloadToGpu === false ? ' (CPU RAM, --no-mmproj-offload)' : ' (CPU RAM)')));
        }
        if (kvBytesWarm < kvBytes) {
        }
        lines.push('Est. total at full context — VRAM: ~' + fmtBytes(totalGpu) + (totalCpu > 1024*1024 ? ' · system RAM: ~' + fmtBytes(totalCpu) : ''));
      }
      lines.push('Bars show estimate at full context. Actual use varies by quant, MoE, and backend.');
      const specLabel = draftIn
        ? (sidecarMtp ? 'MTP draft (weights + KV)' : 'DFlash draft (weights + KV)')
        : (L.speculativeMode === 'mtp' && draftGpuBundle + draftCpuBundle > 0
          ? 'MTP head + KV'
          : 'Speculative');
      const charts = buildCharts(gpuWeights, cpuWeights, kvBytes, kvOnGpu, gpuOverhead, cpuOverhead, totalGpu, totalCpu, draftGpuBundle, draftCpuBundle, specLabel, { tensorSplit: L.tensorSplit, mainGpu: L.mainGpu, splitMode: L.splitMode }, gpuVisionBytes, cpuVisionBytes);
      if (cpuOnly) {
        charts.vram.capacityBytes = undefined;
      }
      let summary;
      const specBytes = draftGpuBundle + draftCpuBundle;
      const specSuffix = specBytes > 0
        ? (L.speculativeMode === 'dflash'
          ? (cpuOnly
            ? ' · DFlash +' + fmtBytes(specBytes)
            : ' · DFlash +' + fmtBytes(draftGpuBundle) + (draftCpuBundle > 1024 * 1024 ? ' (+' + fmtBytes(draftCpuBundle) + ' RAM)' : ''))
          : L.speculativeMode === 'mtp'
            ? (cpuOnly
              ? ' · MTP +' + fmtBytes(specBytes)
              : ' · MTP +' + fmtBytes(draftGpuBundle) + (draftCpuBundle > 1024 * 1024 ? ' (+' + fmtBytes(draftCpuBundle) + ' RAM)' : ''))
            : '')
        : '';
      if (cpuOnly) {
        summary = 'System RAM ~' + fmtBytes(totalCpu) +
          (systemRamTotalBytes ? ' of ' + fmtBytes(systemRamTotalBytes) : '') +
          ' · KV ~' + fmtBytes(kvBytes) + ' at full context' + specSuffix;
      } else if (gpuInfos && gpuInfos.length >= 2 && charts.vram2) {
        const p0 = charts.vram.capacityBytes ? Math.round((charts.vram.totalBytes / charts.vram.capacityBytes) * 100) : undefined;
        const p1 = charts.vram2.capacityBytes ? Math.round((charts.vram2.totalBytes / charts.vram2.capacityBytes) * 100) : undefined;
        const order = gpuDisplayOrder(gpuInfos, L.mainGpu);
        const g0 = gpuInfos[order[0]];
        const g1 = gpuInfos[order[1]];
        summary = 'VRAM ' + gpuLabel(g0, order[0]) + ' ~' + fmtBytes(charts.vram.totalBytes) +
          (charts.vram.capacityBytes ? ' of ' + fmtBytes(charts.vram.capacityBytes) + (p0 !== undefined ? ' (' + p0 + '%)' : '') : '') +
          ' · ' + gpuLabel(g1, order[1]) + ' ~' + fmtBytes(charts.vram2.totalBytes) +
          (charts.vram2.capacityBytes ? ' of ' + fmtBytes(charts.vram2.capacityBytes) + (p1 !== undefined ? ' (' + p1 + '%)' : '') : '') +
          ' · KV ~' + fmtBytes(kvBytes) + (kvOnGpu ? ' on GPU' : ' in RAM') +
          ' · ' + onGpu + '/' + nLayers + ' layers offloaded' + specSuffix;
      } else {
        const pct = gpuInfo && gpuInfo.totalBytes
          ? Math.round((totalGpu / gpuInfo.totalBytes) * 100)
          : undefined;
        summary = 'VRAM ~' + fmtBytes(totalGpu) +
          (gpuInfo && gpuInfo.totalBytes ? ' of ' + fmtBytes(gpuInfo.totalBytes) + (pct !== undefined ? ' (' + pct + '%)' : '') : '') +
          ' · KV ~' + fmtBytes(kvBytes) + (kvOnGpu ? ' on GPU' : ' in RAM') +
          ' · ' + onGpu + '/' + nLayers + ' layers offloaded' + specSuffix;
      }
      return {
        summary,
        lines,
        warnings,
        willSpill,
        charts,
        totalGpu,
        totalCpu,
      };
    }

    /** Say when a curated model mode replaces the sampling values below. */
    function renderModeOverrideHint(mode) {
      const hint = $('modeOverrideHint');
      if (!hint) return;
      if (!mode) {
        hint.classList.add('hidden');
        hint.textContent = '';
        return;
      }
      hint.classList.remove('hidden');
      hint.textContent =
        mode.familyLabel + ' model detected — the Model Mode picker in Copilot Chat sets sampling per request, ' +
        'so Temperature, Top P and Top K below are not used. "' + mode.defaultMode + '" sends temperature ' +
        mode.temperature + ', top_p ' + mode.topP + ', top_k ' + mode.topK + '. Max tokens still applies.';
    }

    function syncMmprojOffloadUi(cpuOnly) {
      const el = $('mmprojOffloadToGpu');
      if (!el) return;
      const hint = $('mmprojPathHint');
      const hasProj = !!(hint && (hint.dataset.path || '').trim());
      el.disabled = !!cpuOnly || !hasProj;
    }

    function applyCpuOnlyUi(cpuOnly) {
      $('gpuOffload').disabled = cpuOnly;
      $('gpuOffloadRange').disabled = cpuOnly;
      $('offloadKvCacheToGpu').disabled = cpuOnly;
      syncMmprojOffloadUi(cpuOnly);
      $('gpuOffloadHint').textContent = cpuOnly
        ? 'CPU backend installed — GPU Offload is ignored; everything runs in system RAM.'
        : ('Layers on GPU (-ngl). Range 0–' + modelBlockCount + '; max = all model layers.');
      $('gpuOffloadRow').style.opacity = cpuOnly ? '0.55' : '1';

      // --n-cpu-moe only splits experts GPU↔CPU; meaningless when everything is already on CPU.
      $('nCpuMoe').disabled = cpuOnly;
      $('nCpuMoeRange').disabled = cpuOnly;
      const showMoe = modelIsMoe && !cpuOnly;
      $('moeRow').classList.toggle('hidden', !showMoe);
      if (modelIsMoe) {
        $('moeHint').textContent = cpuOnly
          ? 'CPU backend — experts already run in system RAM; CPU MoE layers (--n-cpu-moe) does not apply.'
          : moeHintDefault;
      }

      const dual = $('dualGpuRow');
      if (dual) {
        const showDual = !cpuOnly && gpuInfos && gpuInfos.length >= 2;
        dual.classList.toggle('hidden', !showDual);
        const sm = $('splitMode');
        const mg = $('mainGpu');
        if (sm) sm.disabled = cpuOnly;
        if (mg) mg.disabled = cpuOnly;
        syncTensorSplitEnabled();
        const hint = $('dualGpuHint');
        if (hint && showDual) {
          const names = gpuInfos.map((g, i) => gpuLabel(g, i) + ' · ' + fmtBytes(g.totalBytes)).join('  ·  ');
          hint.textContent = splitModeIsNone()
            ? names + '. Split mode None keeps every GPU layer on Main GPU and leaves the other cards free. Reload to apply.'
            : names + '. Pick the faster card as Main GPU (Vulkan/CUDA order from llama.cpp, which may differ from btop). Then use the slider for how much of the model that card holds.';
        }
      }
    }

    function splitModeIsNone() {
      return !!( $('splitMode') && $('splitMode').value === 'none' );
    }

    function applyDualGpuUi(L) {
      fillMainGpuSelect(L.mainGpu ?? 0);
      const n = (gpuInfos && gpuInfos.length) || 0;
      if (n < 2) {
        syncTensorSplitPctLabel();
        return false;
      }
      let split = L.tensorSplit || '';
      if (isLegacyGpu0FirstSplit(split) && clampMainGpu(L.mainGpu ?? 0, n) > 0) {
        const shares = tensorSplitShares(split, gpuInfos);
        split = tensorSplitForMainShare(Math.max.apply(null, shares), L.mainGpu ?? 0, n);
      }
      const share = mainShareFromSplit(split, L.mainGpu ?? 0, gpuInfos);
      const pct = Math.min(90, Math.max(10, Math.round(share * 100)));
      const range = $('tensorSplitRange');
      if (range) range.value = String(pct);
      syncTensorSplitEnabled();
      return isLegacyGpu0FirstSplit(L.tensorSplit) && clampMainGpu(L.mainGpu ?? 0, n) > 0;
    }

    function renderMemory(est) {
      if (!est) {
        $('memSummary').textContent = 'Select a model to estimate VRAM / RAM use.';
        $('memLines').textContent = '';
        $('memNotes').classList.add('hidden');
        $('memWarn').classList.add('hidden');
        renderStackedBar('vramStack', 'vramChartSub', null);
        renderStackedBar('vram2Stack', 'vram2ChartSub', null);
        const wrap = $('vram2ChartWrap');
        if (wrap) wrap.classList.add('hidden');
        renderStackedBar('ramStack', 'ramChartSub', null);
        return;
      }
      if (est.charts) {
        if (est.charts.vram && est.charts.vram.title) {
          const t = $('vramChartTitle');
          if (t) t.textContent = est.charts.vram.title;
        }
        renderStackedBar('vramStack', 'vramChartSub', est.charts.vram);
        const wrap = $('vram2ChartWrap');
        if (wrap) {
          const show2 = !!(est.charts.vram2);
          wrap.classList.toggle('hidden', !show2);
          if (show2) {
            const t2 = $('vram2ChartTitle');
            if (t2 && est.charts.vram2.title) t2.textContent = est.charts.vram2.title;
            renderStackedBar('vram2Stack', 'vram2ChartSub', est.charts.vram2);
          } else {
            renderStackedBar('vram2Stack', 'vram2ChartSub', null);
          }
        }
        renderStackedBar('ramStack', 'ramChartSub', est.charts.ram);
      }
      $('memSummary').textContent = est.summary || '';
      $('memLines').innerHTML = (est.lines || []).map((l) => String(l)).join('<br/>');
      const soft = (est.warnings || []).filter((_, i) => !(est.willSpill && i === 0));
      if (soft.length) {
        $('memNotes').classList.remove('hidden');
        $('memNotes').innerHTML = soft.map((w) => String(w)).join('<br/>');
      } else {
        $('memNotes').classList.add('hidden');
      }
      if (est.willSpill && est.warnings && est.warnings.length) {
        $('memWarn').classList.remove('hidden');
        $('memWarn').textContent = est.warnings[0];
      } else {
        $('memWarn').classList.add('hidden');
      }
    }

    function refreshMemoryLive() {
      try {
        syncTensorSplitEnabled();
        renderMemory(liveMemoryEstimate());
      } catch (err) {
        const msg = err && err.message ? err.message : String(err);
        $('memSummary').textContent = 'Memory estimate failed: ' + msg;
      }
    }

    function bindRange(numId, rangeId) {
      const num = $(numId);
      const range = $(rangeId);
      if (!num || !range) return;
      num.addEventListener('input', () => { range.value = num.value; refreshMemoryLive(); });
      range.addEventListener('input', () => { num.value = range.value; refreshMemoryLive(); });
    }
    bindRange('contextLength', 'contextLengthRange');
    bindRange('gpuOffload', 'gpuOffloadRange');
    bindRange('cpuThreads', 'cpuThreadsRange');
    bindRange('nCpuMoe', 'nCpuMoeRange');
    const tsRange = $('tensorSplitRange');
    if (tsRange) tsRange.addEventListener('input', syncTensorSplitPctLabel);
    $('offloadKvCacheToGpu').addEventListener('change', refreshMemoryLive);
    $('evalBatchSize').addEventListener('input', refreshMemoryLive);
    $('physicalBatchSize').addEventListener('input', refreshMemoryLive);

    /** Higher = more precise. Used to flag lopsided K/V pairs. */
    const KV_PRECISION_RANK = { q4_0: 0, q8_0: 1, bf16: 2, f16: 2 };

    /** llama.cpp can only run a quantized V cache on the Flash Attention path. */
    function syncFlashAttentionWarning() {
      const hint = $('kvFlashAttnHint');
      if (!hint) return;
      const vQuantized = KV_PRECISION_RANK[$('cacheTypeV').value] < 2;
      hint.classList.toggle('hidden', !(vQuantized && $('flashAttention').value === 'off'));
    }

    /** Mirror K onto V while linked; warn when V is kept more precise than K. */
    function syncKvLink(propagate) {
      const link = $('kvTypesLinked');
      const k = $('cacheTypeK');
      const v = $('cacheTypeV');
      const hint = $('kvMismatchHint');
      if (!link || !k || !v) return;
      const linked = !!link.checked;
      if (linked && propagate !== false) {
        v.value = k.value;
      }
      v.disabled = linked;
      if (hint) {
        const lopsided = KV_PRECISION_RANK[v.value] > KV_PRECISION_RANK[k.value];
        hint.classList.toggle('hidden', linked || !lopsided);
      }
      syncFlashAttentionWarning();
    }

    $('cacheTypeK').addEventListener('change', () => {
      syncKvLink(true);
      refreshMemoryLive();
      scheduleSaveLoad();
    });
    $('cacheTypeV').addEventListener('change', () => {
      syncKvLink(false);
      refreshMemoryLive();
    });
    $('kvTypesLinked').addEventListener('change', () => {
      syncKvLink(true);
      refreshMemoryLive();
      scheduleSaveLoad();
    });
    $('flashAttention').addEventListener('change', syncFlashAttentionWarning);

    // 'fit' = largest context that still fits the detected VRAM (see fittingContext).
    const LOAD_PRESETS = {
      agent: { contextLength: 65536, cacheTypeK: 'q8_0', cacheTypeV: 'q8_0', slots: 1 },
      context: { contextLength: 'fit', cacheTypeK: 'q8_0', cacheTypeV: 'q4_0', slots: 1 },
      quality: { contextLength: 65536, cacheTypeK: 'f16', cacheTypeV: 'q8_0', slots: 1 },
    };

    const FIT_CONTEXT_STEPS = [
      262144, 196608, 163840, 131072, 98304, 65536, 49152, 32768, 24576, 16384, 8192,
    ];

    /**
     * Largest context from FIT_CONTEXT_STEPS whose estimate stays inside the
     * safe per-device budget (same 92% headroom as the memory bars / willSpill).
     * Dual GPU: each card is checked on its own — combined VRAM must not be
     * compared to a single card. Falls back to 8192 when nothing fits.
     */
    function fittingContext(maxCtx) {
      if (!memInputs) return maxCtx;
      const previous = $('contextLength').value;
      let best = 0;
      for (const step of FIT_CONTEXT_STEPS) {
        const ctx = Math.min(step, maxCtx);
        if (ctx < 8192 || ctx > maxCtx) continue;
        $('contextLength').value = ctx;
        const est = liveMemoryEstimate();
        if (est && !est.willSpill) { best = ctx; break; }
      }
      $('contextLength').value = previous;
      return best || Math.min(8192, maxCtx);
    }

    function currentPresetId() {
      const k = $('cacheTypeK').value;
      const v = $('cacheTypeV').value;
      if (Number($('maxConcurrentPredictions').value) !== 1) return '';
      const ctx = Number($('contextLength').value);
      const maxCtx = Number($('contextLengthRange').max) || 131072;
      for (const [id, p] of Object.entries(LOAD_PRESETS)) {
        if (p.cacheTypeK !== k || p.cacheTypeV !== v) continue;
        // 'fit' depends on live VRAM, so any context counts as a match once the
        // distinctive K/V pair lines up.
        if (p.contextLength === 'fit' || ctx === Math.min(p.contextLength, maxCtx)) return id;
      }
      return '';
    }

    function highlightPreset() {
      const active = currentPresetId();
      document.querySelectorAll('#presetChips .chip').forEach((chip) => {
        chip.classList.toggle('active', chip.dataset.preset === active);
      });
    }

    function applyPreset(id) {
      const p = LOAD_PRESETS[id];
      if (!p) return;
      const maxCtx = Number($('contextLengthRange').max) || 131072;
      $('maxConcurrentPredictions').value = p.slots;
      $('cacheTypeK').value = p.cacheTypeK;
      $('cacheTypeV').value = p.cacheTypeV;
      $('kvTypesLinked').checked = p.cacheTypeK === p.cacheTypeV;
      syncKvLink(false);
      // KV types must already be on the form — fittingContext measures with them.
      const ctx = p.contextLength === 'fit'
        ? fittingContext(maxCtx)
        : Math.min(p.contextLength, maxCtx);
      $('contextLength').value = ctx;
      $('contextLengthRange').value = ctx;
      highlightPreset();
      syncFlashAttentionWarning();
      refreshMemoryLive();
      scheduleSaveLoad();
    }

    document.querySelectorAll('#presetChips .chip').forEach((chip) => {
      chip.addEventListener('click', () => applyPreset(chip.dataset.preset));
    });

    // Persist load edits so dirty tracking / reload uses the form values.
    const loadFieldIds = [
      'contextLength', 'contextLengthRange', 'gpuOffload', 'gpuOffloadRange',
      'cpuThreads', 'cpuThreadsRange', 'evalBatchSize', 'physicalBatchSize',
      'maxConcurrentPredictions', 'nCpuMoe', 'nCpuMoeRange', 'offloadKvCacheToGpu',
      'mmprojOffloadToGpu',
      'cacheTypeK', 'cacheTypeV',
      'keepModelInMemory', 'tryMmap', 'unifiedKvCache', 'flashAttention',
      'contextCheckpoints', 'cacheReuse',
      'reasoningFormat', 'reasoningBudgetUnlimited', 'reasoningBudget',
      'ropeBaseAuto', 'ropeFreqBase', 'ropeScaleAuto', 'ropeFreqScale',
      'seedRandom', 'seed', 'speculativeMode', 'maxDraftTokens', 'minDraftTokens',
      'draftProbability', 'draftGpuOffload',
      'tensorSplitRange', 'splitMode', 'mainGpu'
    ];
    for (const id of loadFieldIds) {
      const el = $(id);
      if (!el) continue;
      el.addEventListener('change', () => { scheduleSaveLoad(); refreshMemoryLive(); });
      el.addEventListener('input', () => { scheduleSaveLoad(); refreshMemoryLive(); });
    }

    // Request defaults apply to the next chat call (no server reload). Persist on edit.
    let saveRequestTimer = null;
    function scheduleSaveRequest() {
      if (saveRequestTimer) clearTimeout(saveRequestTimer);
      saveRequestTimer = setTimeout(() => {
        vscode.postMessage({ type: 'saveRequest', payload: readRequest() });
      }, 250);
    }
    for (const id of ['temperature', 'topP', 'topK', 'maxTokens']) {
      const el = $(id);
      if (!el) continue;
      el.addEventListener('change', scheduleSaveRequest);
      el.addEventListener('input', scheduleSaveRequest);
    }

    function readLoad() {
      const ropeBaseAuto = $('ropeBaseAuto').checked;
      const ropeScaleAuto = $('ropeScaleAuto').checked;
      const seedRandom = $('seedRandom').checked;
      const reasoningUnlimited = $('reasoningBudgetUnlimited').checked;
      const modeSel = $('speculativeMode');
      const mtpOpt = $('specMtpOption');
      let speculativeMode = modeSel ? modeSel.value : 'off';
      // Never persist MTP when the option is unavailable for this GGUF.
      if (speculativeMode === 'mtp' && mtpOpt && (mtpOpt.disabled || mtpOpt.hidden)) {
        speculativeMode = 'off';
        if (modeSel) modeSel.value = 'off';
      }
      return {
        contextLength: Number($('contextLength').value),
        gpuOffload: Number($('gpuOffload').value),
        cpuThreads: Number($('cpuThreads').value),
        evalBatchSize: Number($('evalBatchSize').value),
        physicalBatchSize: Number($('physicalBatchSize').value),
        maxConcurrentPredictions: Number($('maxConcurrentPredictions').value),
        nCpuMoe: Number($('nCpuMoe').value),
        offloadKvCacheToGpu: $('offloadKvCacheToGpu').checked,
        mmprojOffloadToGpu: $('mmprojOffloadToGpu') ? $('mmprojOffloadToGpu').checked : true,
        cacheTypeK: $('cacheTypeK').value || 'q8_0',
        cacheTypeV: ($('kvTypesLinked').checked ? $('cacheTypeK').value : $('cacheTypeV').value) || 'q8_0',
        keepModelInMemory: $('keepModelInMemory').checked,
        tryMmap: $('tryMmap').checked,
        unifiedKvCache: $('unifiedKvCache').checked,
        flashAttention: $('flashAttention').value || 'auto',
        contextCheckpoints: Number($('contextCheckpoints').value),
        cacheReuse: Number($('cacheReuse').value),
        reasoningFormat: $('reasoningFormat').value || 'deepseek-legacy',
        reasoningBudget: reasoningUnlimited ? -1 : Number($('reasoningBudget').value),
        ropeFreqBase: ropeBaseAuto ? null : Number($('ropeFreqBase').value),
        ropeFreqScale: ropeScaleAuto ? null : Number($('ropeFreqScale').value),
        seed: seedRandom ? null : Number($('seed').value),
        speculativeMode,
        maxDraftTokens: Number($('maxDraftTokens').value),
        minDraftTokens: Number($('minDraftTokens').value),
        draftProbability: Number($('draftProbability').value),
        draftGpuOffload: Number(($('draftGpuOffload') && $('draftGpuOffload').value) || 99),
        tensorSplit: readTensorSplitFromUi(),
        splitMode: ($('splitMode') && $('splitMode').value) || 'layer',
        mainGpu: readMainGpuIndex(),
        // draftModelPath is owned by pick/clear handlers only. Never include it in
        // form autosave/reload payloads — an empty hint used to wipe a just-picked path.
      };
    }

    function setDraftModelHint(draftPath) {
      const hint = $('draftModelPathHint');
      if (!hint) return;
      const p = (draftPath || '').trim();
      hint.dataset.path = p;
      if (!p) {
        hint.textContent = 'No draft model selected.';
        return;
      }
      const base = p.split(/[/\\\\]/).pop() || p;
      hint.textContent = base + '  ·  ' + p;
    }

    function setMmprojHint(mmprojPath) {
      const hint = $('mmprojPathHint');
      if (!hint) return;
      const p = (mmprojPath || '').trim();
      hint.dataset.path = p;
      if (!p) {
        hint.textContent = 'No mmproj — text only. A sibling mmproj-*.gguf is attached automatically when you select a multimodal GGUF.';
        syncMmprojOffloadUi(cpuOnlyLive());
        return;
      }
      const base = p.split(/[/\\\\]/).pop() || p;
      hint.textContent = base + '  ·  Copilot Chat can send images. Reload the server to apply.';
      syncMmprojOffloadUi(cpuOnlyLive());
    }

    function readRequest() {
      return {
        temperature: Number($('temperature').value),
        topP: Number($('topP').value),
        topK: Number($('topK').value),
        maxTokens: Number($('maxTokens').value),
      };
    }

    function applyCapabilities(caps) {
      const maxCtx = (caps && caps.maxContextLength) ? caps.maxContextLength : 131072;
      const blocks = (caps && caps.blockCount) ? caps.blockCount : 128;
      const isMoe = !!(caps && caps.isMoe);
      modelIsMoe = isMoe;
      modelBlockCount = Math.max(1, blocks);

      $('contextLength').max = String(maxCtx);
      $('contextLengthRange').max = String(maxCtx);
      $('contextLengthRange').min = '512';
      // Slider max = actual layer count (legacy 99/"all" is shown as all layers).
      $('gpuOffload').max = String(modelBlockCount);
      $('gpuOffloadRange').max = String(modelBlockCount);
      $('nCpuMoe').max = String(modelBlockCount);
      $('nCpuMoeRange').max = String(modelBlockCount);

      // Visibility finalized in applyCpuOnlyUi (also hides on CPU backend).
      if (caps) {
        $('ctxHint').textContent = 'Model supports up to ' + maxCtx + ' tokens (from GGUF metadata)';
        $('moeHint').textContent = isMoe
          ? ('MoE model' + (caps.expertCount ? (' · ' + caps.expertCount + ' experts') : '') +
             (caps.expertUsedCount ? (' · ' + caps.expertUsedCount + ' used/token') : '') +
             '. Layers to force experts onto CPU (0–' + blocks + ').')
          : 'Not a MoE model — this setting is hidden.';
        if (isMoe) {
          moeHintDefault = $('moeHint').textContent;
        }
        $('modelCaps').classList.remove('hidden');
        $('modelCaps').innerHTML =
          'Architecture: <strong>' + (caps.architecture || '?') + '</strong><br/>' +
          'Max context: <strong>' + maxCtx + '</strong> · Layers: <strong>' + blocks + '</strong>' +
          (isMoe ? (' · MoE experts: <strong>' + (caps.expertCount || '?') + '</strong>') : ' · Dense (non-MoE)') +
          (caps.fullAttentionInterval > 1
            ? (' · hybrid full-attn every <strong>' + caps.fullAttentionInterval + '</strong> layers')
            : '') +
          (caps.nextnPredictLayers > 0
            ? (' · MTP next-n: <strong>' + caps.nextnPredictLayers + '</strong>')
            : (mtpSidecarPath
              ? (' · MTP sidecar: <strong>' + String(mtpSidecarPath).split(/[/\\\\]/).pop() + '</strong>')
              : ''));
        applySpecUi(!!(caps.nextnPredictLayers > 0), sidecarMtpAvailable());
      } else {
        $('modelCaps').classList.add('hidden');
        $('ctxHint').textContent = 'Tokens for prompt + generation';
        applySpecUi(false, sidecarMtpAvailable());
      }
    }

    function isMtpDraftName(p) {
      const n = String(p || '').split(/[/\\\\]/).pop() || '';
      return /^mtp[-_]/i.test(n) || /-mtp\\.gguf$/i.test(n);
    }

    function sidecarMtpAvailable() {
      const hint = $('draftModelPathHint');
      const fromHint = hint && hint.dataset ? hint.dataset.path : '';
      return !!(mtpSidecarPath || isMtpDraftName(fromHint));
    }

    /** Show MTP and/or DFlash controls based on mode + target capabilities. */
    function applySpecUi(bakedMtp, sidecarMtp) {
      const mtpCapable = !!(bakedMtp || sidecarMtp);
      const modeSel = $('speculativeMode');
      const mtpOpt = $('specMtpOption');
      const hint = $('specHint');
      if (mtpOpt) {
        mtpOpt.disabled = !mtpCapable;
        mtpOpt.hidden = !mtpCapable;
      }
      if (modeSel) {
        if (!mtpCapable && modeSel.value === 'mtp') {
          modeSel.value = 'off';
        }
      }
      const mode = modeSel ? modeSel.value : 'off';
      const isMtp = mode === 'mtp';
      const isDflash = mode === 'dflash';
      const showMtpKnobs = isMtp && mtpCapable;
      const showDraftKnobs = isMtp || isDflash;
      const showDraftPicker = isDflash || (isMtp && sidecarMtp && !bakedMtp);

      for (const id of ['maxDraftTokens']) {
        const el = $(id);
        if (el) el.disabled = !showDraftKnobs;
      }
      for (const id of ['minDraftTokens', 'draftProbability']) {
        const el = $(id);
        if (el) el.disabled = !showMtpKnobs;
      }
      for (const id of ['specDraftMaxRow']) {
        const row = $(id);
        if (row) {
          row.style.opacity = showDraftKnobs ? '1' : '0.55';
          row.classList.toggle('hidden', !showDraftKnobs);
        }
      }
      for (const id of ['specDraftMinRow', 'specDraftPRow']) {
        const row = $(id);
        if (row) {
          row.style.opacity = showMtpKnobs ? '1' : '0.55';
          row.classList.toggle('hidden', !showMtpKnobs);
        }
      }
      for (const id of ['specDraftModelRow', 'specDraftNglRow']) {
        const row = $(id);
        if (row) row.classList.toggle('hidden', !showDraftPicker);
      }

      const kindHint = $('draftModelKindHint');
      if (kindHint) {
        kindHint.textContent = isMtp
          ? 'Gemma 4 MTP is a sibling mtp-*.gguf (architecture gemma4-assistant), passed as --model-draft with --spec-type draft-mtp. llama.cpp ≥ 2026-06-07.'
          : 'DFlash needs a separate draft GGUF (architecture = dflash) for your target — not the main model. Download one first, then choose it here.';
      }
      const nglHint = $('draftNglHint');
      if (nglHint) {
        nglHint.textContent = isMtp
          ? '99 usually means all MTP draft layers (--spec-draft-ngl). Sidecar MTP can use the same KV cache types as the main model.'
          : '99 usually means all draft layers. DFlash draft KV cache is forced to f16 (quantized draft KV collapses acceptance).';
      }

      if (hint) {
        if (isDflash) {
          hint.textContent =
            'DFlash passes --spec-type draft-dflash -md <draft> --spec-draft-ngl … with draft KV forced to f16 and --fit off (llama.cpp auto-fit breaks DFlash). Flash Attention On is recommended.';
        } else if (sidecarMtp && !bakedMtp) {
          hint.textContent =
            'A sibling mtp-*.gguf was found. Mode MTP passes --spec-type draft-mtp --model-draft <mtp> and --fit off (llama.cpp auto-fit breaks Gemma 4 MTP). Needs llama.cpp ≥ 2026-06-07.';
        } else if (bakedMtp) {
          hint.textContent =
            'This model reports MTP next-n layers. Mode MTP passes --spec-type draft-mtp.';
        } else {
          hint.textContent =
            'This GGUF has no MTP / nextn_predict_layers and no sibling mtp-*.gguf — MTP is unavailable. Use DFlash with a separate draft GGUF, or an MTP-tagged main model.';
        }
      }
    }

    function applyState(payload) {
      const s = payload.state;
      const L = s.loadSettings;
      const R = s.requestSettings;
      const status = payload.status;
      const hasModel = !!(s.selectedModelPath);
      const caps = payload.capabilities;
      mtpSidecarPath = payload.mtpSidecarPath || '';

      applyCapabilities(caps);

      const cores = Math.max(1, Number(payload.cpuCount) || cpuLogicalCores);
      cpuLogicalCores = cores;
      $('cpuThreads').max = String(cores);
      $('cpuThreadsRange').max = String(cores);
      const cpuHint = $('cpuThreadsHint');
      if (cpuHint) {
        cpuHint.textContent = 'llama.cpp -t. Range 1–' + cores + ' (logical CPU cores).';
      }

      const build = payload.build || {};
      const starting = !!(status.starting || payload.starting);
      const ready = !starting && !!(status.httpReady || status.running);
      const endpoint = payload.endpoint || status.endpoint || '';
      const startMessage = status.startMessage || status.message || '';

      serverStarting = starting;
      serverRunning = ready;
      configDirty = starting ? false : !!status.configDirty;
      renderStatusUi({
        ready,
        dirty: configDirty,
        starting,
        endpoint,
        pid: status.pid,
        message: starting ? startMessage : (status.message || ''),
      });
      const lm = $('launchMode');
      if (lm && payload.launchMode) {
        lm.value = payload.launchMode === 'background' ? 'background' : 'externalTerminal';
      }
      updatePrimaryAction();

      const perfLines = Array.isArray(payload.perfLines) ? payload.perfLines : ['No generation yet'];
      const perf = payload.perf || {};
      renderContextStack(perf);
      renderPerfStats(perf, perfLines);
      const viewCtx = $('viewContextBtn');
      if (viewCtx) {
        viewCtx.disabled = !payload.hasLastContext;
        viewCtx.title = payload.hasLastContext
          ? 'Open the last Copilot → llama.cpp request (messages + tools) in an editor'
          : 'Send a Copilot Chat message first';
      }
      const viewResp = $('viewResponseBtn');
      if (viewResp) {
        viewResp.disabled = !payload.hasLastResponse;
        viewResp.title = payload.hasLastResponse
          ? 'Open the last llama.cpp → Copilot response stream in an editor'
          : 'Send a Copilot Chat message first';
      }

      const prToggle = $('promptReplacementsEnabled');
      if (prToggle) {
        prToggle.checked = !!payload.promptReplacementsEnabled;
      }
      const prStats = $('replacementStats');
      if (prStats) {
        const pr = perf.promptReplacements;
        if (!payload.promptReplacementsEnabled) {
          prStats.textContent = 'Last call: replacements off';
        } else if (!pr) {
          prStats.textContent = 'Last call: — (send a chat to measure)';
        } else if (!pr.enabled) {
          prStats.textContent = 'Last call: replacements were off';
        } else if (pr.tokensSaved > 0) {
          prStats.textContent =
            'Last call: saved ≈' + Number(pr.tokensSaved).toLocaleString() +
            ' tokens (' + pr.pctSaved + '% of request)';
        } else {
          prStats.textContent = 'Last call: no matching boilerplate (' +
            Number(pr.tokensBefore || 0).toLocaleString() + ' tok request)';
        }
      }

      const binaryDetail = $('llamaBinaryDetail');
      if (binaryDetail) {
        if (build.binaryRunnable === false) {
          binaryDetail.textContent =
            'Binary present but not runnable on this OS' +
            (build.nixOs ? ' (NixOS / missing FHS linker)' : '') +
            (build.binaryRunError ? (': ' + String(build.binaryRunError).slice(0, 120)) : '.');
        } else if (build.activeBackend === 'path' && build.pathBinary) {
          binaryDetail.textContent = build.binaryVersionDetail || ('PATH: ' + build.pathBinary);
        } else if (build.binaryVersionDetail || build.binaryVersion) {
          binaryDetail.textContent = build.binaryVersionDetail || build.binaryVersion;
        } else if (build.tag && payload.binaryExists) {
          binaryDetail.textContent = 'Installed release ' + build.tag;
        } else if (payload.binaryExists) {
          binaryDetail.textContent = 'Binary installed (version string unavailable).';
        } else {
          binaryDetail.textContent = 'No binary for this backend yet.';
        }
      }
      if (build.activeBackend === 'path') {
        $('llamaAssetDetail').textContent = build.pathBinary
          ? ('PATH binary: ' + build.pathBinary)
          : 'No llama-server on PATH.';
      } else {
        $('llamaAssetDetail').textContent = build.asset
          ? ('Asset: ' + build.asset + (build.configuredBackend ? (' · setting: ' + build.configuredBackend) : ''))
          : 'No archive recorded for this backend yet.';
      }

      updateCheck = payload.updateCheck || {
        latestTag: undefined,
        installedTag: build.tag,
        updateAvailable: false,
        checkFailed: false,
        pending: true,
      };

      const sel = $('backendSelect');
      const options = Array.isArray(payload.backendOptions) ? payload.backendOptions : [];
      backendOptionsCache = options;
      activeBackendId = payload.selectedUiBackend || (options.find((o) => o.active) || {}).id || '';
      const prev = sel.value;
      suppressBackendChange = true;
      sel.innerHTML = '';
      for (const opt of options) {
        const o = document.createElement('option');
        o.value = opt.id;
        let text = opt.label;
        if (!opt.available) {
          text += ' (unavailable)';
        } else if (opt.id === 'path') {
          text += opt.installed
            ? (opt.installedTag ? (' · ' + opt.installedTag) : ' · found')
            : ' · not on PATH';
          if (opt.active) text += ' ●';
        } else if (opt.installed) {
          text += opt.installedTag
            ? (' · installed ' + opt.installedTag)
            : ' · installed';
          if (opt.active) text += ' ●';
        } else {
          text += ' · not installed';
        }
        o.textContent = text;
        // PATH stays selectable even when missing so the user can switch to it
        // and see the install hint; download backends disable when unavailable.
        o.disabled = opt.id === 'path' ? false : !opt.available;
        if (opt.reason && (opt.id === 'path' ? !opt.installed : !opt.available)) {
          o.title = opt.reason;
        } else if (opt.id === 'path' && opt.installed && opt.installedTag) {
          o.title = 'llama-server on PATH';
        } else if (opt.installed && opt.installedTag) {
          o.title = 'Cached locally: ' + opt.installedTag;
        }
        sel.appendChild(o);
      }
      const want = activeBackendId || prev || 'vulkan';
      if ([...sel.options].some((o) => o.value === want && !o.disabled)) {
        sel.value = want;
      } else {
        const first = [...sel.options].find((o) => !o.disabled);
        if (first) sel.value = first.value;
      }
      suppressBackendChange = false;
      updateBackendUi();

      $('setupBox').classList.toggle('hidden', hasModel && payload.binaryExists);

      const localCount = payload.localModelCount || 0;
      const showStarter = !hasModel;
      const starterBtn = $('starterModelBtn');
      const starterHint = $('starterModelHint');
      const downloadBtn = $('downloadModelBtn');
      const pickBtn = $('pickDownloadedBtn');
      if (starterBtn) starterBtn.classList.toggle('hidden', !showStarter);
      if (starterHint) starterHint.classList.toggle('hidden', !showStarter);
      if (downloadBtn) {
        downloadBtn.className = showStarter && localCount === 0 ? 'secondary' : 'primary';
      }
      if (pickBtn) {
        // Prefer choosing an existing GGUF when the library already has files.
        pickBtn.className = showStarter && localCount > 0 ? 'primary' : 'secondary';
      }
      $('modelTitle').textContent = hasModel
        ? ('Selected: ' + (payload.modelName || 'model'))
        : 'No model selected';
      function escAttr(v) {
        return String(v).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
      }
      function escText(v) {
        return String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;');
      }
      function folderLink(label, dir, title) {
        return '<a class="folder-link" href="#" data-path="' + escAttr(dir) + '" title="' +
          escAttr(title || dir) + '">' + escText(label) + '</a>';
      }
      function bindFolderLinks(root) {
        root.querySelectorAll('a.folder-link, a.model-path-link').forEach((link) => {
          link.addEventListener('click', (e) => {
            e.preventDefault();
            const filePath = link.getAttribute('data-path');
            if (filePath) vscode.postMessage({ type: 'revealInOs', path: filePath });
          });
        });
      }

      if (hasModel && s.selectedModelPath) {
        const p = String(s.selectedModelPath);
        $('modelPath').innerHTML =
          '<a class="model-path-link" href="#" data-path="' + escAttr(p) +
          '" title="Reveal in File Explorer">' + escText(p) + '</a>';
        bindFolderLinks($('modelPath'));
      } else {
        $('modelPath').textContent = 'Choose one of the options below to install or select a GGUF model.';
      }

      const modelsDir = payload.modelsDir || '';
      const sourceDirs = Array.isArray(payload.localSourceDirs) ? payload.localSourceDirs : [];
      const sourceHtml = sourceDirs.length
        ? ' · sources: ' + sourceDirs.map((src) =>
            folderLink(src.source, src.dir, 'Open ' + src.source + ' folder')
          ).join(', ')
        : '';
      $('modelsDirMeta').innerHTML =
        'Llama AIO downloads: ' +
        (modelsDir ? folderLink(modelsDir, modelsDir, 'Open downloads folder') : '—') +
        ' · ' + (payload.localModelCount || 0) + ' GGUF found' +
        sourceHtml +
        ' (also scans LM Studio, Unsloth, HF cache, …)';
      bindFolderLinks($('modelsDirMeta'));

      const maxCtx = (caps && caps.maxContextLength) ? caps.maxContextLength : 131072;
      const blocks = (caps && caps.blockCount) ? caps.blockCount : 128;
      const ctx = Math.min(L.contextLength, maxCtx);
      // Normalize legacy 99/"all" sentinel to the model's layer count for the slider.
      const ngl =
        L.gpuOffload <= 0 ? 0 : L.gpuOffload >= 99 || L.gpuOffload >= blocks ? blocks : Math.min(L.gpuOffload, blocks);
      const moe = (caps && !caps.isMoe) ? 0 : Math.min(L.nCpuMoe, blocks);
      const threads = Math.min(Math.max(1, L.cpuThreads || 1), cpuLogicalCores);

      $('contextLength').value = ctx;
      $('contextLengthRange').value = ctx;
      $('gpuOffload').value = ngl;
      $('gpuOffloadRange').value = ngl;
      $('cpuThreads').value = threads;
      $('cpuThreadsRange').value = threads;
      $('evalBatchSize').value = L.evalBatchSize;
      $('physicalBatchSize').value = L.physicalBatchSize;
      $('maxConcurrentPredictions').value = L.maxConcurrentPredictions;
      $('nCpuMoe').value = moe;
      $('nCpuMoeRange').value = moe;
      $('offloadKvCacheToGpu').checked = !!L.offloadKvCacheToGpu;
      if ($('mmprojOffloadToGpu')) $('mmprojOffloadToGpu').checked = L.mmprojOffloadToGpu !== false;
      $('cacheTypeK').value = L.cacheTypeK || 'q8_0';
      $('cacheTypeV').value = L.cacheTypeV || 'q8_0';
      $('kvTypesLinked').checked = (L.cacheTypeK || 'q8_0') === (L.cacheTypeV || 'q8_0');
      syncKvLink(false);
      $('keepModelInMemory').checked = !!L.keepModelInMemory;
      if (payload.isWindows) {
        const label = $('keepModelLabel');
        if (label) label.textContent = 'Keep Model in Memory (mmap on Windows)';
        const hint = $('keepModelHint');
        if (hint) hint.style.display = 'block';
      }
      $('tryMmap').checked = !!L.tryMmap;
      $('unifiedKvCache').checked = !!L.unifiedKvCache;
      $('flashAttention').value = L.flashAttention || 'auto';
      $('contextCheckpoints').value = L.contextCheckpoints;
      $('cacheReuse').value = L.cacheReuse ?? 0;
      $('reasoningFormat').value = L.reasoningFormat || 'deepseek-legacy';
      const budget = L.reasoningBudget ?? -1;
      $('reasoningBudgetUnlimited').checked = budget < 0;
      $('reasoningBudget').value = budget < 0 ? 2048 : budget;
      $('reasoningBudget').disabled = budget < 0;
      $('ropeBaseAuto').checked = L.ropeFreqBase == null;
      $('ropeFreqBase').value = L.ropeFreqBase ?? 10000;
      $('ropeFreqBase').disabled = L.ropeFreqBase == null;
      $('ropeScaleAuto').checked = L.ropeFreqScale == null;
      $('ropeFreqScale').value = L.ropeFreqScale ?? 1;
      $('ropeFreqScale').disabled = L.ropeFreqScale == null;
      $('seedRandom').checked = L.seed == null;
      $('seed').value = L.seed ?? 0;
      $('seed').disabled = L.seed == null;
      $('speculativeMode').value = L.speculativeMode || 'off';
      $('maxDraftTokens').value = L.maxDraftTokens;
      $('minDraftTokens').value = L.minDraftTokens;
      $('draftProbability').value = L.draftProbability;
      if ($('draftGpuOffload')) $('draftGpuOffload').value = L.draftGpuOffload ?? 99;
      if ($('splitMode')) $('splitMode').value = L.splitMode || 'layer';
      setDraftModelHint(L.draftModelPath || '');
      setMmprojHint(L.mmprojPath || '');
      applySpecUi(!!(caps && caps.nextnPredictLayers > 0), sidecarMtpAvailable());
      $('temperature').value = R.temperature;
      $('topP').value = R.topP;
      $('topK').value = R.topK;
      $('maxTokens').value = R.maxTokens;
      renderModeOverrideHint(payload.modeSampling);
      syncFlashAttentionWarning();

      highlightPreset();

      memInputs = payload.memInputs || null;
      gpuInfo = payload.gpu || null;
      gpuInfos = Array.isArray(payload.gpus) ? payload.gpus : (gpuInfo ? [gpuInfo] : []);
      systemRamTotalBytes = payload.systemRamTotalBytes || 0;
      const splitDirty = applyDualGpuUi(L);
      applyCpuOnlyUi(!!payload.cpuOnly || payload.selectedUiBackend === 'cpu');
      if (gpuInfos.length >= 2) {
        refreshMemoryLive();
      } else if (payload.memory) {
        renderMemory(payload.memory);
      } else {
        refreshMemoryLive();
      }
      if (splitDirty) scheduleSaveLoad();
      }

    $('ropeBaseAuto').addEventListener('change', () => {
      $('ropeFreqBase').disabled = $('ropeBaseAuto').checked;
    });
    $('ropeScaleAuto').addEventListener('change', () => {
      $('ropeFreqScale').disabled = $('ropeScaleAuto').checked;
    });
    $('seedRandom').addEventListener('change', () => {
      $('seed').disabled = $('seedRandom').checked;
    });
    $('reasoningBudgetUnlimited').addEventListener('change', () => {
      $('reasoningBudget').disabled = $('reasoningBudgetUnlimited').checked;
    });
    const speculativeModeEl = $('speculativeMode');
    if (speculativeModeEl) {
      speculativeModeEl.addEventListener('change', () => {
        if (speculativeModeEl.value === 'dflash') {
          const maxEl = $('maxDraftTokens');
          if (maxEl && Number(maxEl.value) <= 2) {
            maxEl.value = '15';
          }
        } else if (speculativeModeEl.value === 'mtp') {
          const maxEl = $('maxDraftTokens');
          if (maxEl && sidecarMtpAvailable() && Number(maxEl.value) <= 0) {
            maxEl.value = '4';
          }
        }
        applySpecUi(!!(memInputs && memInputs.nextnPredictLayers > 0), sidecarMtpAvailable());
      });
    }
    const pickDraftBtn = $('pickDraftModelBtn');
    if (pickDraftBtn) {
      pickDraftBtn.addEventListener('click', () => {
        if (saveLoadTimer) clearTimeout(saveLoadTimer);
        vscode.postMessage({ type: 'pickDraftModel' });
      });
    }
    const clearDraftBtn = $('clearDraftModelBtn');
    if (clearDraftBtn) {
      clearDraftBtn.addEventListener('click', () => {
        if (saveLoadTimer) clearTimeout(saveLoadTimer);
        setDraftModelHint('');
        vscode.postMessage({ type: 'clearDraftModel' });
      });
    }
    const pickMmprojBtn = $('pickMmprojBtn');
    if (pickMmprojBtn) {
      pickMmprojBtn.addEventListener('click', () => {
        if (saveLoadTimer) clearTimeout(saveLoadTimer);
        vscode.postMessage({ type: 'pickMmproj' });
      });
    }
    const clearMmprojBtn = $('clearMmprojBtn');
    if (clearMmprojBtn) {
      clearMmprojBtn.addEventListener('click', () => {
        if (saveLoadTimer) clearTimeout(saveLoadTimer);
        setMmprojHint('');
        vscode.postMessage({ type: 'clearMmproj' });
      });
    }

    $('downloadModelBtn').addEventListener('click', () => vscode.postMessage({ type: 'downloadModel' }));
    const viewContextBtn = $('viewContextBtn');
    if (viewContextBtn) {
      viewContextBtn.addEventListener('click', () => vscode.postMessage({ type: 'viewLastCall' }));
    }
    const viewResponseBtn = $('viewResponseBtn');
    if (viewResponseBtn) {
      viewResponseBtn.addEventListener('click', () => vscode.postMessage({ type: 'viewLastResponse' }));
    }
    const resetAdvancedBtn = $('resetAdvancedBtn');
    if (resetAdvancedBtn) {
      resetAdvancedBtn.addEventListener('click', () => vscode.postMessage({ type: 'resetAdvancedLoad' }));
    }
    const resetRequestBtn = $('resetRequestBtn');
    if (resetRequestBtn) {
      resetRequestBtn.addEventListener('click', () => vscode.postMessage({ type: 'resetRequestDefaults' }));
    }
    const prToggle = $('promptReplacementsEnabled');
    if (prToggle) {
      prToggle.addEventListener('change', () => {
        vscode.postMessage({
          type: 'setPromptReplacementsEnabled',
          payload: { enabled: !!prToggle.checked },
        });
      });
    }
    const starterModelBtn = $('starterModelBtn');
    if (starterModelBtn) {
      starterModelBtn.addEventListener('click', () => vscode.postMessage({ type: 'downloadStarter' }));
    }
    const setupStarterBtn = $('setupStarterBtn');
    if (setupStarterBtn) {
      setupStarterBtn.addEventListener('click', () => vscode.postMessage({ type: 'downloadStarter' }));
    }
    $('openFileBtn').addEventListener('click', () => vscode.postMessage({ type: 'openModelFile' }));
    $('pickDownloadedBtn').addEventListener('click', () => vscode.postMessage({ type: 'pickDownloadedModel' }));
    $('installLlamaBtn').addEventListener('click', () => {
      const action = $('installLlamaBtn').dataset.action;
      if (action === 'check') {
        vscode.postMessage({ type: 'checkUpdates' });
        return;
      }
      vscode.postMessage({ type: 'installLlamaCpp', payload: $('backendSelect').value });
    });
    const checkUpdatesBtn = $('checkUpdatesBtn');
    if (checkUpdatesBtn) {
      checkUpdatesBtn.addEventListener('click', () => vscode.postMessage({ type: 'checkUpdates' }));
    }
    const reinstallLlamaBtn = $('reinstallLlamaBtn');
    if (reinstallLlamaBtn) {
      reinstallLlamaBtn.addEventListener('click', () => vscode.postMessage({ type: 'reinstallLlamaCpp' }));
    }
    $('installByTagBtn').addEventListener('click', () => vscode.postMessage({ type: 'installLlamaCppByTag' }));
    $('installArchiveBtn').addEventListener('click', () => vscode.postMessage({ type: 'installLlamaCppFromArchive' }));
    const releasesLink = $('releasesLink');
    if (releasesLink) {
      releasesLink.addEventListener('click', (e) => {
        e.preventDefault();
        vscode.postMessage({ type: 'openExternal', url: 'https://github.com/ggml-org/llama.cpp/releases' });
      });
    }
    $('backendSelect').addEventListener('change', () => {
      if (suppressBackendChange) return;
      updateBackendUi();
      applyCpuOnlyUi($('backendSelect').value === 'cpu');
      refreshMemoryLive();
      const next = $('backendSelect').value;
      if (!next || next === activeBackendId) return;
      const opt = backendOptionsCache.find((o) => o.id === next);
      // PATH stays selectable when missing so the user can switch and see the hint.
      if (opt && !opt.available && next !== 'path') return;
      vscode.postMessage({ type: 'switchBackend', payload: next });
    });

    $('primaryBtn').addEventListener('click', () => {
      const action = $('primaryBtn').dataset.action;
      if (action === 'reload') {
        serverStarting = true;
        updatePrimaryAction();
        renderStatusUi({
          ready: false,
          dirty: false,
          starting: true,
          endpoint: '',
          message: '',
        });
        vscode.postMessage({ type: 'reload', payload: readLoad() });
        vscode.postMessage({ type: 'saveRequest', payload: readRequest() });
      } else if (action === 'start') {
        serverStarting = true;
        updatePrimaryAction();
        renderStatusUi({
          ready: false,
          dirty: false,
          starting: true,
          endpoint: '',
          message: '',
        });
        vscode.postMessage({ type: 'saveLoad', payload: readLoad() });
        vscode.postMessage({ type: 'saveRequest', payload: readRequest() });
        vscode.postMessage({ type: 'start' });
      }
    });
    $('stopBtn').addEventListener('click', () => vscode.postMessage({ type: 'stop' }));
    $('launchMode').addEventListener('change', () => {
      const mode = $('launchMode').value === 'background' ? 'background' : 'externalTerminal';
      if (serverRunning) {
        configDirty = true;
        updatePrimaryAction();
        const hint = $('dirtyHint');
        if (hint) hint.classList.remove('hidden');
        setServerCardKind('dirty');
        const line = $('statusLine');
        const dot = $('statusDot');
        if (line) line.className = 'status-line dirty';
        if (dot) dot.className = 'dot dirty';
      }
      vscode.postMessage({ type: 'setLaunchMode', payload: mode });
    });

    window.addEventListener('message', (event) => {
      const msg = event.data;
      if (msg.type === 'state') applyState(msg.payload);
      if (msg.type === 'updateCheck' && msg.payload) {
        updateCheck = msg.payload;
        updateBackendUi();
      }
      if (msg.type === 'perfPatch' && msg.payload) {
        const perf = msg.payload.perf || {};
        renderContextStack(perf);
        renderPerfStats(perf, msg.payload.perfLines || []);
        const viewCtx = $('viewContextBtn');
        if (viewCtx) {
          viewCtx.disabled = !msg.payload.hasLastContext;
        }
        const viewResp = $('viewResponseBtn');
        if (viewResp) {
          viewResp.disabled = !msg.payload.hasLastResponse;
        }
      }
      if (msg.type === 'draftModelSelected') {
        setDraftModelHint(msg.path || '');
        if (isMtpDraftName(msg.path)) {
          mtpSidecarPath = mtpSidecarPath || msg.path;
        }
        applySpecUi(!!(memInputs && memInputs.nextnPredictLayers > 0), sidecarMtpAvailable());
        refreshMemoryLive();
      }
      if (msg.type === 'mmprojSelected') {
        setMmprojHint(msg.path || '');
        refreshMemoryLive();
      }
      if (msg.type === 'bootProgress' && msg.payload) {
        serverStarting = true;
        serverRunning = false;
        configDirty = false;
        updatePrimaryAction();
        renderStatusUi({
          ready: false,
          dirty: false,
          starting: true,
          endpoint: '',
          message: msg.payload.message || 'Starting…',
        });
      }
      if (msg.type === 'statusPatch' && msg.payload) {
        if (msg.payload.starting) {
          serverStarting = true;
          serverRunning = false;
          configDirty = false;
          updatePrimaryAction();
          renderStatusUi({
            ready: false,
            dirty: false,
            starting: true,
            endpoint: '',
            message: msg.payload.startMessage || msg.payload.message || 'Starting…',
          });
          return;
        }
        serverRunning = !!msg.payload.running;
        configDirty = !!msg.payload.configDirty;
        serverStarting = false;
        updatePrimaryAction();
        if (serverRunning) {
          const hint = $('dirtyHint');
          if (hint) hint.classList.toggle('hidden', !configDirty);
          setServerCardKind(configDirty ? 'dirty' : 'ok');
          const line = $('statusLine');
          const dot = $('statusDot');
          if (line) line.className = 'status-line ' + (configDirty ? 'dirty' : 'ok');
          if (dot) dot.className = 'dot ' + (configDirty ? 'dirty' : 'ok');
          const text = $('statusText');
          if (text) text.textContent = 'Server ready';
        }
        if (msg.payload.perf || Array.isArray(msg.payload.perfLines)) {
          renderPerfStats(msg.payload.perf || {}, msg.payload.perfLines);
        }
      }
    });

    vscode.postMessage({ type: 'ready' });
  </script>
</body>
</html>`;
  }
}

function getNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let nonce = "";
  for (let i = 0; i < 32; i++) {
    nonce += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return nonce;
}

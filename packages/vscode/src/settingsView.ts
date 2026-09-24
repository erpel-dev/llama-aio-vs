import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";
import { promptUseInCopilotChat } from "./copilotChatPrompt";
import { copyServerCommandLine, openServerLog, reportLaunchFailure } from "./serverDiagnostics";
import { detectGpus, activeInstallLock, type GpuMemoryInfo } from "@llama-aio/core";
import { LlamaInstaller, UiBackend } from "@llama-aio/core";
import { estimateMemory, memoryEstimateInputs, mmprojFileSize, resolveDraftCapabilities } from "@llama-aio/core";
import { resolveModelModes } from "@llama-aio/core";
import {
  displayModelTitle,
  listActiveModelSourceDirs,
  listLocalModelEntries,
  findSiblingMtpDraft,
  invalidateModelLibraryCache,
  isMtpSidecarFile,
} from "@llama-aio/core";
import { getModelsDir } from "@llama-aio/core";
import { PerfStats } from "@llama-aio/core";
import { LaunchToken, LAUNCH_IN_PROGRESS_MSG, ProcessManager } from "@llama-aio/core";
import { SettingsStore } from "@llama-aio/core";
import { resolveLaunchMode } from "@llama-aio/core";
import { capsMissDefaultSwaPattern, DEFAULT_LOAD_SETTINGS, DEFAULT_REQUEST_SETTINGS, effectiveServerUiState, isQwen4expArchitecture, LlamaLoadSettings, normalizeSpeculativeMode, RequestSettings } from "@llama-aio/core";
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
  showDownloads: () => Promise<void>;
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
    private readonly notifyChatModels: () => void = () => undefined,
    private readonly globalState?: vscode.Memento
  ) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this.view = webviewView;
    webviewView.onDidDispose(() => {
      if (this.view === webviewView) {
        this.view = undefined;
      }
    });
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
              await this.postStatusNow();
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
              lazyMode: d.lazyMode,
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
              ngramVariant: d.ngramVariant,
              ngramSizeN: d.ngramSizeN,
              ngramSizeM: d.ngramSizeM,
              ngramMinHits: d.ngramMinHits,
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
            let readyMessage: string | undefined;
            try {
              await this.store.updateLoadSettings(msg.payload as Partial<LlamaLoadSettings>);
              this.syncSpeculativeMode();
              if (!(await this.confirmIfMemorySpill())) {
                break;
              }
              await this.onReload(token);
              readyMessage = this.processManager.getStatus().message;
            } finally {
              this.processManager.releaseLaunch(token);
              await this.postStatusNow();
              await this.pushState();
            }
            if (readyMessage) {
              void promptUseInCopilotChat(this.store, readyMessage, this.globalState);
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
            let readyMessage: string | undefined;
            try {
              if (msg.payload) {
                await this.store.updateLoadSettings(msg.payload as Partial<LlamaLoadSettings>);
                this.syncSpeculativeMode();
              }
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
              readyMessage = status.message;
            } finally {
              this.processManager.releaseLaunch(token);
              await this.postStatusNow();
              await this.pushState();
            }
            if (readyMessage) {
              void promptUseInCopilotChat(this.store, readyMessage, this.globalState);
            }
            break;
          }
          case "stop":
            await this.processManager.stop(true);
            await this.pushState();
            break;
          case "openLog":
            await openServerLog();
            break;
          case "copyCommandLine":
            await copyServerCommandLine();
            break;
          case "refresh":
            invalidateModelLibraryCache();
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
          case "showDownloads":
            await this.modelActions.showDownloads();
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
          case "setWikipediaLookupEnabled": {
            const enabled = !!(msg.payload && (msg.payload as { enabled?: boolean }).enabled);
            await this.store.getConfig().update("wikipediaLookupEnabled", enabled);
            await this.pushState();
            break;
          }
          case "setDuplicateToolCallGuardEnabled": {
            const enabled = !!(msg.payload && (msg.payload as { enabled?: boolean }).enabled);
            await this.store.getConfig().update("duplicateToolCallGuardEnabled", enabled);
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
        if (msg.type === "start" || msg.type === "reload") {
          await reportLaunchFailure(msg.type === "reload" ? "Reload failed" : "Start failed", e);
        } else {
          vscode.window.showErrorMessage(
            `Llama AIO: ${e instanceof Error ? e.message : String(e)}`
          );
        }
        await this.pushState();
      }
    });
  }

  private detectGpusForEstimate(cpuOnly: boolean): GpuMemoryInfo[] {
    if (cpuOnly || activeInstallLock()) {
      return [];
    }
    return detectGpus(false, this.processManager.resolveBinary());
  }

  private currentMemoryEstimate() {
    const state = this.store.getState();
    const cpuOnly =
      this.installer.resolveActiveUiBackend() === "cpu" || this.processManager.isCpuBackend();
    const gpus = this.detectGpusForEstimate(cpuOnly);
    return estimateMemory(
      state.modelCapabilities,
      state.loadSettings,
      cpuOnly ? undefined : gpus[0],
      {
        cpuOnly,
        draftCaps: resolveDraftCapabilities(state.loadSettings),
        gpus: cpuOnly ? undefined : gpus,
      }
    );
  }

  /** Keep sidebar speculative line in sync with Load settings (clears stale % when off). */
  syncSpeculativeMode(): void {
    const mode = this.store.getState().loadSettings.speculativeMode || "off";
    this.perf.setSpeculativeMode(normalizeSpeculativeMode(mode));
  }

  /**
   * Ask before start/reload when core's memory estimate says the load will spill.
   * The webview's live chart is display-only and is not consulted here.
   */
  async confirmIfMemorySpill(): Promise<boolean> {
    const est = this.currentMemoryEstimate();
    if (!est?.willSpill) {
      return true;
    }
    const warning =
      est.warnings[0] ||
      "These settings leave too little memory headroom and may spill or thrash (much slower).";
    // Modal dialogs already include a localized Cancel — passing "Cancel" too
    // shows two Abbrechen buttons.
    const choice = await vscode.window.showWarningMessage(
      warning,
      { modal: true },
      "Continue anyway"
    );
    return choice === "Continue anyway";
  }

  async pushState(): Promise<void> {
    if (!this.view) {
      return;
    }
    // Paint Server ready/stopped before GPU probes and library scans. Reuse
    // that probe below instead of a second /health round-trip.
    const httpReady = await this.postStatusNow();
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
        // Stale caps from before llama.cpp's hard-coded SWA layouts were known
        // (Muse Glimmer, Gemma 2/3, gpt-oss, Llama 4, …).
        capsMissDefaultSwaPattern(state.modelCapabilities) ||
        // Stale caps from before hybrid full-attention interval (e.g. Qwen3.5 / qwen4exp).
        ((state.modelCapabilities.architecture === "qwen35" ||
          state.modelCapabilities.architecture === "qwen35moe" ||
          isQwen4expArchitecture(state.modelCapabilities.architecture)) &&
          !state.modelCapabilities.fullAttentionInterval) ||
        (isQwen4expArchitecture(state.modelCapabilities.architecture) &&
          (state.modelCapabilities.pleShare === undefined ||
            !Number.isFinite(state.modelCapabilities.pleShare) ||
            // Stale 40% heuristic from scanning only shard 1 (PLE often lives in shard 2).
            ((state.modelCapabilities.shardCount ?? 1) > 1 &&
              state.modelCapabilities.pleShare === 0.4))))
    ) {
      try {
        state = await this.store.applySelectedModel(state.selectedModelPath);
      } catch {
        // keep previous
      }
    }

    const status = this.processManager.getStatus();
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
    const gpus = this.detectGpusForEstimate(cpuOnly);
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
        wikipediaLookupEnabled: this.store.isWikipediaLookupEnabled(),
        duplicateToolCallGuardEnabled: this.store.isDuplicateToolCallGuardEnabled(),
        endpoint: this.store.getEndpoint(),
        binary,
        binaryExists: fs.existsSync(binary),
        modelsDir,
        localModelCount,
        localSources,
        localSourceDirs,
        modelName: displayModelTitle(caps?.name, state.selectedModelPath),
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
          (isMtpSidecarFile({ path: state.loadSettings.draftModelPath }) ? state.loadSettings.draftModelPath : "") ||
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
              ssmStateSize: caps.ssmStateSize,
              ssmInnerSize: caps.ssmInnerSize,
              ssmConvKernel: caps.ssmConvKernel,
              ssmGroupCount: caps.ssmGroupCount,
              pleShare: caps.pleShare,
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

  /**
   * Patch the Server card from live process state (does not wait on Copilot
   * dialogs). Returns the /health result so callers can reuse it.
   */
  async postStatusNow(knownHttpReady?: boolean): Promise<boolean> {
    if (!this.view) {
      return knownHttpReady ?? false;
    }
    const status = this.processManager.getStatus();
    const httpReady = knownHttpReady ?? (await this.processManager.isHttpReady());
    const ui = effectiveServerUiState({
      starting: status.starting,
      running: status.running,
      httpReady,
    });
    this.view.webview.postMessage({
      type: "statusPatch",
      payload: {
        configDirty: ui.starting ? false : !!status.configDirty,
        running: ui.ready,
        starting: ui.starting,
        httpReady,
        endpoint: status.endpoint,
        pid: status.pid,
        startMessage: status.startMessage || status.message,
        message: status.message,
        perf: this.perf.get(),
        perfLines: this.perf.detailLines(),
      },
    });
    return httpReady;
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
    .actions-row.diag-row { margin-top: 6px; }
    .actions-row.diag-row button.secondary {
      flex: 1;
      text-align: center;
      font-weight: 550;
      padding: 5px 8px;
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
    .ctx-legend.tight { gap: 6px 8px; font-size: 9.5px; margin-top: 4px; }
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
    .chart-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      font-size: 10px;
      color: var(--muted);
      margin: 8px 0 4px;
    }
    .chart-legend i {
      display: inline-block;
      width: 12px;
      height: 2px;
      border-radius: 1px;
      margin-right: 5px;
      vertical-align: middle;
    }
    .chart-legend .gen { background: var(--starting); }
    .chart-legend .prompt { background: var(--warn); }
    .perf-session { margin-top: 0; width: 100%; }
    .perf-chart {
      display: block;
      width: 100%;
      height: 96px;
      color: var(--muted);
    }
    .metric-line {
      display: flex;
      flex-wrap: wrap;
      gap: 2px 10px;
      margin-top: 6px;
      font-size: 11px;
      color: var(--fg);
    }
    .metric-line .muted { color: var(--muted); }
    .metric-line .ok { font-weight: 650; }
    .metric-line .gen { color: var(--starting); font-weight: 650; }
    .metric-line .prompt { color: var(--warn); font-weight: 650; }
    .perf-history {
      margin-top: 8px;
      border: 1px solid var(--border);
      border-radius: 5px;
      overflow: hidden;
    }
    .perf-history summary {
      list-style: none;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      padding: 5px 8px;
      font-size: 11px;
      cursor: pointer;
      user-select: none;
      color: var(--muted);
    }
    .perf-history summary::-webkit-details-marker { display: none; }
    .perf-history summary::before {
      content: '▸';
      color: var(--muted);
      font-size: 10px;
      flex-shrink: 0;
    }
    .perf-history[open] summary::before { content: '▾'; }
    .perf-history table {
      width: 100%;
      border-collapse: collapse;
      font-size: 10.5px;
      border-top: 1px solid var(--border);
    }
    .perf-history th, .perf-history td {
      padding: 3px 6px;
      text-align: right;
      white-space: nowrap;
    }
    .perf-history th:first-child, .perf-history td:first-child { text-align: left; }
    .perf-history th {
      font-size: 9px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 600;
      background: color-mix(in srgb, var(--fg) 5%, transparent);
    }
    .perf-history tr:nth-child(even) td {
      background: color-mix(in srgb, var(--fg) 3%, transparent);
    }
    .perf-history .muted { color: var(--muted); }
    .perf-history #perfHistoryTable { overflow-x: auto; }
    .perf-history .opt-list { padding: 0 8px 8px; }
    .perf-history .opt-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      padding: 4px 0;
      border-top: 1px solid var(--border);
      color: var(--fg);
      font-size: 11px;
    }
    .perf-history .opt-row .hint {
      font-size: 10px;
      color: var(--muted);
      font-weight: 400;
      margin: 0;
    }
    .perf-history .btn-row {
      display: flex;
      gap: 6px;
      margin-top: 6px;
    }
    .perf-history .btn-row button {
      flex: 1;
      padding: 6px 8px;
      font-size: 11px;
      text-align: center;
    }
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
    .spec-group {
      margin: 8px 0 4px;
      padding: 2px 0 2px 12px;
      border-left: 2px solid color-mix(in srgb, var(--accent) 40%, var(--border));
    }
    .spec-group-title {
      font-size: 10px;
      font-weight: 650;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      margin: 4px 0 2px;
    }
    .spec-group .row { margin: 8px 0 10px; }
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
    <div class="actions-row diag-row">
      <button class="secondary" id="openLogBtn" type="button">Open log</button>
      <button class="secondary" id="copyCmdBtn" type="button">Copy command</button>
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
    <div class="ctx-stack" id="ctxStack" role="img" aria-label="Context"></div>
    <div class="ctx-legend tight" id="ctxLegend">
      <span><i class="seg-tools"></i>Tools</span>
      <span><i class="seg-system"></i>Sys</span>
      <span><i class="seg-history"></i>Hist</span>
      <span><i class="seg-toolResults"></i>Results</span>
      <span><i class="seg-request"></i>Req</span>
    </div>
    <div class="chart-legend hidden" id="perfChartLegend">
      <span><i class="gen"></i>Generation</span>
      <span><i class="prompt"></i>Prompt processing</span>
    </div>
    <div class="perf-session hidden" id="perfSession"></div>
    <div class="metric-line" id="perfMetrics">No generation yet</div>
    <details class="perf-history" id="perfMore">
      <summary>History, options &amp; debug</summary>
      <div id="perfHistoryTable"></div>
      <div class="opt-list">
        <div class="opt-row">
          <span>Prompt replacements <span class="hint" id="replacementStats">—</span></span>
          <input type="checkbox" id="promptReplacementsEnabled" title="Strip Copilot system-prompt boilerplate before llama.cpp" />
        </div>
        <div class="opt-row">
          <span>Wikipedia lookup</span>
          <input type="checkbox" id="wikipediaLookupEnabled" title="Let the model call wikipedia_lookup for encyclopedic facts. Off by default." />
        </div>
        <div class="opt-row">
          <span>Skip duplicate tools</span>
          <input type="checkbox" id="duplicateToolCallGuardEnabled" title="Skip a tool that already ran with the same arguments in this turn. Off by default — can block a legitimate retry." />
        </div>
        <div class="btn-row">
          <button class="secondary" id="viewContextBtn" disabled title="Open the last Copilot → llama.cpp request (messages + tools) in an editor">Last call</button>
          <button class="secondary" id="viewResponseBtn" disabled title="Open the last llama.cpp assistant stream (helps debug empty Chat replies)">Last response</button>
        </div>
      </div>
    </details>
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
      <button class="secondary" id="showDownloadsBtn">⬇ Downloads</button>
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
    <div class="label"><span class="name tip" data-flag="-c, --ctx-size" data-help="Size of the prompt context (default: 0 = loaded from model).">Context Length</span><input type="number" id="contextLength" min="512" step="256" /></div>
    <input type="range" id="contextLengthRange" min="512" max="131072" step="1" />
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
    <div class="label" style="margin-top:6px"><span class="name tip" data-flag="-ts, --tensor-split" data-help="Percent of GPU-resident weights on the Main GPU after CPU MoE/FFN. llama.cpp --tensor-split still fills by layer count, so the emitted fractions can differ (cheap first layers get more of Main). The rest is split evenly across the other cards. Disabled when Split mode is None.">Weights on main GPU</span><span id="tensorSplitPct">75%</span></div>
    <input type="range" id="tensorSplitRange" min="10" max="90" step="1" />
    <div class="hint" id="tensorSplitHint"></div>
    <div class="label" style="margin-top:6px"><span class="name tip" data-flag="-sm, --split-mode" data-help="How tensors are split. Layer (default) shares the model across cards. Row needs a fast x16 link. Tensor splits every weight matrix across cards (experimental, fastest for multi-GPU inference). None keeps every GPU layer on the Main GPU and leaves the other cards free (--device).">Split mode</span></div>
      <select id="splitMode" class="wide">
        <option value="layer">Layer (default)</option>
        <option value="row">Row</option>
        <option value="tensor">Tensor (experimental)</option>
        <option value="none">None — Main GPU only</option>
      </select>
    <div class="hint" id="dualGpuHint">Two GPUs detected. Pick the faster card as Main, then raise the slider to give it more weights.</div>
  </div>
  <div class="row" id="moeRow">
    <div id="moeFields">
      <div class="label"><span class="name tip" data-flag="-ncmoe, --n-cpu-moe" data-help="Keep the Mixture of Experts (MoE) weights of the first N layers in the CPU.">CPU MoE layers</span><span class="badge">MoE only</span><input type="number" id="nCpuMoe" min="0" max="256" /></div>
      <input type="range" id="nCpuMoeRange" min="0" max="128" step="1" />
      <div class="hint" id="moeHint">Only applies to MoE models.</div>
    </div>
    <div id="ffnFields">
      <div class="label"><span class="name tip" data-flag="-ncffn, --n-cpu-ffn" data-help="Keep the dense FFN weights of the first N layers in the CPU (dense models; for MoE expert weights use --n-cpu-moe).">CPU FFN layers</span><span class="badge">Dense only</span><input type="number" id="nCpuFfn" min="0" max="256" /></div>
      <input type="range" id="nCpuFfnRange" min="0" max="128" step="1" />
      <div class="hint" id="ffnHint">Only applies to dense (non-MoE) models.</div>
    </div>
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
    <div class="label"><span class="name tip" data-flag="-ctk, --cache-type-k" data-help="KV cache data type for K. Allowed: f32, f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl (default: q8_0). q8_0 halves KV size with little quality loss.">KV Cache Type (K)</span>
      <select id="cacheTypeK">
        <option value="f32">f32 (full precision)</option>
        <option value="f16">f16</option>
        <option value="bf16">bf16</option>
        <option value="q8_0">q8_0 (~½ size, default)</option>
        <option value="q5_1">q5_1 (~5/8 size)</option>
        <option value="q5_0">q5_0 (~5/8 size)</option>
        <option value="q4_1">q4_1 (~9/16 size)</option>
        <option value="q4_0">q4_0 (~9/16 size)</option>
        <option value="iq4_nl">iq4_nl (~9/16 size, better quality)</option>
      </select>
    </div>
  </div>
  <div class="toggle"><span class="tip" data-flag="-ctk / -ctv" data-help="Keep V on the same type as K. Mixed K/V types fall off the fast attention path and can cost ~40% of prompt and generation throughput.">Use same type for V</span><input type="checkbox" id="kvTypesLinked" checked /></div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="-ctv, --cache-type-v" data-help="KV cache data type for V. Same types as K (default: q8_0). q4_0 saves more VRAM but can hurt long-context prompt speed.">KV Cache Type (V)</span>
      <select id="cacheTypeV">
        <option value="f32">f32 (full precision)</option>
        <option value="f16">f16</option>
        <option value="bf16">bf16</option>
        <option value="q8_0">q8_0 (~½ size, default)</option>
        <option value="q5_1">q5_1 (~5/8 size)</option>
        <option value="q5_0">q5_0 (~5/8 size)</option>
        <option value="q4_1">q4_1 (~9/16 size)</option>
        <option value="q4_0">q4_0 (~9/16 size)</option>
        <option value="iq4_nl">iq4_nl (~9/16 size, better quality)</option>
      </select>
    </div>
    <div class="hint hidden" id="kvMismatchHint">V is stored more precisely than K. The key cache is the quantization-sensitive one, so this spends memory where it helps least — prefer K at the higher precision (e.g. q8_0 K with q4_0 V).</div>
    <div class="hint hidden" id="kvFlashAttnHint">A quantized V cache requires Flash Attention. With Flash Attention set to <strong>Off</strong>, llama-server will refuse to start — set V to f16 or put Flash Attention back to Auto/On.</div>
  </div>
  <div class="toggle"><span id="keepModelLabel" class="tip" data-flag="--load-mode mlock" data-help="Force the system to keep the model in RAM rather than swapping (load-mode mlock). On Windows this falls back to mmap.">Keep Model in Memory (--mlock)</span><input type="checkbox" id="keepModelInMemory" /></div>
  <div class="hint" id="keepModelHint" style="display:none">On Windows this uses mmap (--load-mode mmap); mlock is not reliable.</div>
  <div class="toggle"><span class="tip" data-flag="--load-mode mmap | none" data-help="Memory-map the model (mmap). If disabled and mlock is off, uses load-mode none (slower load, may reduce pageouts). Lazy mode requires mmap.">Try mmap()</span><input type="checkbox" id="tryMmap" /></div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="-lzm, --lazy-mode" data-help="On-demand reading of large host tensors such as per-layer n-gram / PLE tables. auto = on for tensors larger than 4 GiB (llama.cpp default). Requires mmap.">Lazy tensor reads</span>
      <select id="lazyMode">
        <option value="auto">Auto (default — on if &gt; 4 GiB)</option>
        <option value="on">On — read from disk</option>
        <option value="off">Off — keep resident</option>
      </select>
    </div>
    <div class="hint">Only sent when not Auto. Keeps mmap even with CPU MoE/FFN so Flash-Next can leave the n-gram table on SSD.</div>
  </div>
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
    <div class="label"><span class="name tip" data-flag="--spec-type" data-help="Speculative decoding type. MTP uses next-n layers in the main GGUF, or a sibling mtp-*.gguf (Gemma 4) via --model-draft. DFlash uses a separate draft GGUF with --spec-type draft-dflash. N-gram drafts from n-grams seen in the prompt itself and can stack with MTP or DFlash.">Mode</span>
      <select id="speculativeMode">
        <option value="off">Off</option>
        <option value="mtp" id="specMtpOption">MTP (draft-mtp)</option>
        <option value="dflash" id="specDflashOption">DFlash (draft-dflash)</option>
        <option value="ngram">N-gram (no draft model)</option>
        <option value="ngram-mtp" id="specNgramMtpOption">N-gram + MTP</option>
        <option value="ngram-dflash">N-gram + DFlash</option>
      </select>
    </div>
    <div class="hint" id="specHint">MTP needs next-n layers in the main GGUF or a sibling mtp-*.gguf. DFlash needs a matching DFlash draft GGUF. N-gram works on any model and can stack with MTP or DFlash.</div>
  </div>
  <div class="spec-group hidden" id="specNgramGroup">
    <div class="spec-group-title">N-gram</div>
    <div class="row" id="specNgramVariantRow">
      <div class="label"><span class="name tip" data-flag="--spec-type ngram-*" data-help="N-gram lookup strategy. Simple keeps a rolling window of the prompt. Map-k indexes the prompt for faster lookup on long contexts. K4v additionally caches key/value states per n-gram. Mod is the classic prompt-lookup variant tuned for long drafts.">Variant</span>
        <select id="ngramVariant">
          <option value="simple">simple — rolling prompt window</option>
          <option value="map-k">map-k — indexed lookup (long prompts)</option>
          <option value="map-k4v">map-k4v — indexed + KV cache</option>
          <option value="mod">mod — classic prompt-lookup</option>
        </select>
      </div>
    </div>
    <div class="row" id="specNgramSizeRow">
      <div class="label"><span class="name tip" data-flag="--spec-ngram-*-size-n" data-help="Length of the lookup n-gram in tokens (default: 12; mod variant match length: 24). Longer = fewer false matches, shorter = more draft attempts.">Lookup size</span><input type="number" id="ngramSizeN" min="2" max="512" /></div>
      <input type="range" id="ngramSizeNRange" min="2" max="64" step="1" />
    </div>
    <div class="row" id="specNgramDraftRow">
      <div class="label"><span class="name tip" data-flag="--spec-ngram-*-size-m / --spec-ngram-mod-n-max" data-help="How many tokens n-gram may draft from a prompt match (default: 48; mod variant max: 64). This is not MTP/DFlash --spec-draft-n-max.">Draft length</span><input type="number" id="ngramSizeM" min="2" max="2048" /></div>
    </div>
    <div class="row" id="specNgramHitsRow">
      <div class="label"><span class="name tip" data-flag="--spec-ngram-*-min-hits" data-help="Minimum number of times an n-gram must appear before it is used to draft (default: 1). Raise to 2–3 to avoid drafting from coincidental matches. Not used by the mod variant.">Min hits</span><input type="number" id="ngramMinHits" min="1" max="64" /></div>
    </div>
  </div>
  <div class="spec-group hidden" id="specNeuralGroup">
    <div class="spec-group-title" id="specNeuralHeading">MTP</div>
    <div class="row hidden" id="specDraftModelRow">
      <div class="label"><span class="name tip" id="specDraftModelName" data-flag="-md, --model-draft" data-help="Path to the DFlash draft GGUF (architecture = dflash) or Gemma 4 sidecar MTP GGUF (mtp-*.gguf / architecture = gemma4-assistant).">Draft model</span></div>
      <div class="hint" id="draftModelPathHint" style="margin:4px 0 8px">No draft model selected.</div>
      <div class="hint" id="draftModelKindHint" style="margin:0 0 8px">DFlash needs a <em>separate</em> draft GGUF (<code>architecture = dflash</code>) for your target — not the main model. Gemma 4 MTP uses a sibling <code>mtp-*.gguf</code>.</div>
      <div class="btn-row" style="margin:0 0 8px;gap:8px;flex-wrap:wrap">
        <button class="secondary" id="pickDraftModelBtn" type="button">Choose draft GGUF…</button>
        <button class="secondary" id="clearDraftModelBtn" type="button">Clear</button>
      </div>
    </div>
    <div class="row hidden" id="specDraftNglRow">
      <div class="label"><span class="name tip" id="specDraftNglName" data-flag="--spec-draft-ngl" data-help="Max draft-model layers in VRAM (exact number, auto, or all).">GPU offload</span><input type="number" id="draftGpuOffload" min="0" max="999" /></div>
      <div class="hint" id="draftNglHint">99 usually means all draft layers. DFlash draft KV cache is forced to f16 (quantized draft KV collapses acceptance).</div>
    </div>
    <div class="row hidden" id="specDraftMaxRow">
      <div class="label"><span class="name tip" id="specDraftMaxName" data-flag="--spec-draft-n-max" data-help="Number of tokens the MTP or DFlash drafter proposes per step (--spec-draft-n-max). Separate from n-gram draft length. For DFlash try 8–15.">Max draft tokens</span><input type="number" id="maxDraftTokens" min="0" /></div>
    </div>
    <div class="row hidden" id="specDraftMinRow">
      <div class="label"><span class="name tip" id="specDraftMinName" data-flag="--spec-draft-n-min" data-help="Minimum number of draft tokens to use for speculative decoding (default: 0). MTP only.">Min draft tokens</span><input type="number" id="minDraftTokens" min="0" /></div>
    </div>
    <div class="row hidden" id="specDraftPRow">
      <div class="label"><span class="name tip" id="specDraftPName" data-flag="--spec-draft-p-min" data-help="Stop drafting when the next-token probability drops below this (default here: 0.75). llama.cpp's own default of 0.00 never stops and collapses acceptance. Used by MTP and DFlash.">Draft probability</span><input type="number" id="draftProbability" min="0" max="1" step="0.01" /></div>
    </div>
  </div>

  <div class="btn-col" style="margin:12px 0 8px">
    <button class="secondary" id="resetAdvancedBtn" title="Restore Advanced Settings to Llama AIO defaults (does not change Context Length / GPU Offload / MoE)">Reset advanced to defaults</button>
  </div>

  </details>

  <details class="advanced">
    <summary>Request defaults<span class="sub">temperature, top-p/k, min-p, penalties, max tokens</span></summary>
  <div class="hint hidden" id="modeOverrideHint" style="margin-bottom:10px"></div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="Chat / API request body" data-help="Sampling temperature for completions (extension request default, not a llama-server load flag).">Temperature</span><input type="number" id="temperature" min="0" max="2" step="0.05" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="Chat / API request body" data-help="Nucleus sampling top-p (extension request default).">Top P</span><input type="number" id="topP" min="0" max="1" step="0.01" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="--top-k / request body" data-help="Top-k sampling (extension request default).">Top K</span><input type="number" id="topK" min="0" step="1" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="--min-p / request body" data-help="Min-p sampling (0 = disabled). llama-server's built-in default of 0.05 is wrong for most current instruct/coder families — keep 0 unless a model card recommends otherwise.">Min P</span><input type="number" id="minP" min="0" max="1" step="0.01" /></div>
    <div class="hint">Also shipped as a server CLI default at start, so raw API clients inherit it.</div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="--repeat-penalty / request body" data-help="Repetition penalty (llama.cpp style; 1.0 = disabled). Values above 1 discourage repeating earlier tokens — useful for prose, harmful for code (breaks exact repetition like closing tags).">Repeat penalty</span><input type="number" id="repeatPenalty" min="0.5" max="2" step="0.01" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="--presence-penalty / request body" data-help="Presence penalty (OpenAI-style; 0 = disabled). Positive values push the model toward new topics. Qwen3 No-Think mode recommends 1.5.">Presence penalty</span><input type="number" id="presencePenalty" min="-2" max="2" step="0.05" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="--frequency-penalty / request body" data-help="Frequency penalty (OpenAI-style; 0 = disabled). Scaled by how often a token already appeared.">Frequency penalty</span><input type="number" id="frequencyPenalty" min="-2" max="2" step="0.05" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name tip" data-flag="Chat / API request body" data-help="Max tokens to generate per reply (extension request default / n_predict-style cap).">Max tokens</span><input type="number" id="maxTokens" min="16" step="16" /></div>
  </div>

  <div class="btn-col" style="margin:12px 0 8px">
    <button class="secondary" id="resetRequestBtn" title="Restore temperature, top-p/k, min-p, penalties, and max tokens to Llama AIO defaults">Reset request defaults</button>
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
    let ffnHintDefault = 'Number of layers to force dense FFN weights onto CPU (--n-cpu-ffn). Only applies to dense (non-MoE) models.';
    let suppressBackendChange = false;
    let activeBackendId = '';
    let serverRunning = false;
    let configDirty = false;
    let serverStarting = false;
    let lastEndpoint = '';
    let lastPid;
    let saveLoadTimer = null;
    let lastChartGen = [];
    let lastChartPrompt = [];
    let perfChartRO = null;
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

    const VRAM_HEADROOM = 1.5 * 1024 ** 3;
    const VRAM_SOFT_HEADROOM = 4 * 1024 ** 3;

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

    // Mirrors isIntegratedGpu() / gpuBarCapacityBytes() in core.
    function isIntegratedGpu(gpu) {
      if (!gpu || !(gpu.totalBytes > 0)) return false;
      const name = String(gpu.name || '') + ' ' + String(gpu.llamaDeviceId || '');
      if (/onboard|\bigd\b|integrated|iris|uhd graphics|hd graphics|radeon graphics|vega mobile|cezanne|renoir|lucienne|barcelo|mendocino|rembrandt|raphael|phoenix|hawk.?point|strix|gfx[0-9]+c\b/i.test(name)) {
        return true;
      }
      const gtt = gpu.gttTotalBytes || 0;
      return gpu.totalBytes <= 2 * 1024 ** 3 && gtt >= 4 * 1024 ** 3 && gtt >= gpu.totalBytes * 4;
    }
    function gpuBarCapacityBytes(gpu) {
      if (!gpu || !(gpu.totalBytes > 0)) return undefined;
      if (isIntegratedGpu(gpu) && (gpu.gttTotalBytes || 0) > gpu.totalBytes) return gpu.gttTotalBytes;
      return gpu.totalBytes;
    }

    function parseTensorSplit(raw) {
      const s = String(raw || '').trim();
      if (!s) return [];
      const parts = s.split(/[,/;:\\s]+/).map(Number).filter((n) => isFinite(n) && n >= 0).slice(0, 8);
      return parts.length >= 2 && parts.some((n) => n > 0) ? parts : [];
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
      if (sum <= 0 || totals.some((b) => b <= 0)) return Array.from({ length: n }, () => 1 / n);
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

    // Mirrors assignLayerDevices / layerAwareWeightShares in gpuSplit.ts.
    function layerAwareWeightSharesLive(nLayers, onGpu, shares, splitMode, mainGpu, opts) {
      const nDev = Math.max(1, (shares && shares.length) || 1);
      const fallback = shares && shares.length === nDev ? shares : Array.from({ length: nDev }, () => 1 / nDev);
      if (splitMode === 'row' || splitMode === 'tensor') return fallback;
      const n = Math.max(0, Math.round(nLayers) || 0);
      const gpuN = Math.min(n, Math.max(0, Math.round(onGpu) || 0));
      const start = n - gpuN;
      const assign = Array.from({ length: n }, () => -1);
      const main = clampMainGpu(mainGpu, nDev);
      if (gpuN <= 0) return fallback;
      if (splitMode === 'none' || nDev === 1) {
        for (let i = start; i < n; i++) assign[i] = main;
      } else {
        for (let k = 0; k < gpuN; k++) {
          const pos = (k + 1) / gpuN;
          let cum = 0;
          let dev = nDev - 1;
          for (let d = 0; d < nDev; d++) {
            cum += shares[d] || 0;
            if (pos <= cum + 1e-9) { dev = d; break; }
          }
          assign[start + k] = dev;
        }
      }
      const mass = Array.from({ length: nDev }, () => 0);
      let tot = 0;
      const moe = !!(opts && opts.isMoe);
      const nCpuMoe = Math.max(0, Math.round((opts && opts.nCpuMoe) || 0));
      const moeShare = Math.min(0.98, Math.max(0, (opts && opts.moeExpertShare) || 0));
      const nCpuFfn = Math.max(0, Math.round((opts && opts.nCpuFfn) || 0));
      const ffnShare = Math.min(0.95, Math.max(0, (opts && opts.denseFfnShare) || 0));
      for (let i = 0; i < assign.length; i++) {
        const d = assign[i];
        if (d < 0 || d >= nDev) continue;
        let m = 1;
        if (moe && i < nCpuMoe && moeShare > 0) m = Math.max(0.02, 1 - moeShare);
        else if (!moe && i < nCpuFfn && ffnShare > 0) m = Math.max(0.05, 1 - ffnShare);
        mass[d] += m;
        tot += m;
      }
      if (tot <= 0) return fallback;
      return mass.map((x) => x / tot);
    }

    function needsWeightAwareTensorSplitLive(splitMode, opts) {
      if (splitMode === 'row' || splitMode === 'tensor' || splitMode === 'none') return false;
      if (opts && opts.isMoe) return (opts.nCpuMoe || 0) > 0 && (opts.moeExpertShare || 0) > 0;
      return !!(opts && (opts.nCpuFfn || 0) > 0 && (opts.denseFfnShare || 0) > 0);
    }

    function liveLayersOnGpu(L) {
      const nLayers = Math.max(1, (memInputs && memInputs.blockCount) || 1);
      const ngl = L && Number.isFinite(Number(L.gpuOffload)) ? Number(L.gpuOffload) : Number($('gpuOffload') && $('gpuOffload').value);
      if (!Number.isFinite(ngl) || ngl >= 99) return nLayers;
      return Math.min(nLayers, Math.max(0, Math.round(ngl)));
    }

    function liveMassOpts(L) {
      const nCpuMoe = L && Number.isFinite(Number(L.nCpuMoe)) ? Number(L.nCpuMoe) : Number($('nCpuMoe') && $('nCpuMoe').value) || 0;
      const nCpuFfn = L && Number.isFinite(Number(L.nCpuFfn)) ? Number(L.nCpuFfn) : Number($('nCpuFfn') && $('nCpuFfn').value) || 0;
      return {
        isMoe: !!(memInputs && memInputs.isMoe),
        nCpuMoe: nCpuMoe,
        moeExpertShare: memInputs ? moeExpertShareOf(memInputs) : 0,
        nCpuFfn: nCpuFfn,
        denseFfnShare: memInputs ? denseFfnShareOf(memInputs) : 0,
      };
    }

    function mainWeightShareFromSplitLive(raw, mainGpu, gpus, splitMode, opts, nLayers, onGpu) {
      const shares = effectiveTensorSplitShares(raw, gpus, splitMode, mainGpu);
      const n = (gpus && gpus.length) || 1;
      const main = clampMainGpu(mainGpu, n);
      if (n < 2 || !needsWeightAwareTensorSplitLive(splitMode, opts)) return shares[main] || 1;
      const weight = layerAwareWeightSharesLive(nLayers, onGpu, shares, splitMode, mainGpu, opts);
      return weight[main] || shares[main] || 1;
    }

    function tensorSplitForTargetWeightShareLive(targetShare, mainGpu, n, nLayers, onGpu, splitMode, opts) {
      n = Math.max(1, Math.round(n) || 1);
      if (n < 2) return '';
      const target = Math.min(0.9, Math.max(0.1, Number(targetShare) || 0.5));
      if (nLayers < 1 || !needsWeightAwareTensorSplitLive(splitMode, opts)) {
        return tensorSplitForMainShare(target, mainGpu, n);
      }
      const main = clampMainGpu(mainGpu, n);
      const targetPct = Math.round(target * 100);
      let best = tensorSplitForMainShare(target, mainGpu, n);
      let bestErr = Infinity;
      let bestPct = targetPct;
      for (let pct = 10; pct <= 90; pct++) {
        const candidate = tensorSplitForMainShare(pct / 100, mainGpu, n);
        const shares = tensorSplitShares(candidate, gpuInfos);
        const weight = layerAwareWeightSharesLive(nLayers, onGpu, shares, splitMode || 'layer', mainGpu, opts);
        const err = Math.abs((weight[main] || 0) - target);
        if (err < bestErr - 1e-12) {
          best = candidate;
          bestErr = err;
          bestPct = pct;
        } else if (err <= bestErr + 1e-12) {
          const closer = Math.abs(pct - targetPct) < Math.abs(bestPct - targetPct);
          if (closer || (pct < bestPct && Math.abs(pct - targetPct) === Math.abs(bestPct - targetPct))) {
            best = candidate;
            bestErr = err;
            bestPct = pct;
          }
        }
      }
      return best;
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
      const raw = $('tensorSplitRange') && $('tensorSplitRange').value;
      const pct = Number(raw);
      const share = (Number.isFinite(pct) ? pct : 50) / 100;
      const splitMode = ($('splitMode') && $('splitMode').value) || 'layer';
      const nLayers = Math.max(1, (memInputs && memInputs.blockCount) || 1);
      return tensorSplitForTargetWeightShareLive(
        share,
        readMainGpuIndex(),
        n,
        nLayers,
        liveLayersOnGpu(),
        splitMode,
        liveMassOpts()
      );
    }

    function syncTensorSplitPctLabel() {
      const range = $('tensorSplitRange');
      const lbl = $('tensorSplitPct');
      const hint = $('tensorSplitHint');
      if (!range || !lbl) return;
      if ($('splitMode') && $('splitMode').value === 'none') {
        lbl.textContent = '100%';
        if (hint) hint.textContent = '';
        return;
      }
      lbl.textContent = String(range.value) + '%';
      if (hint) {
        const n = (gpuInfos && gpuInfos.length) || 1;
        const emitted = n >= 2 ? readTensorSplitFromUi() : '';
        hint.textContent = emitted ? ('GPU-resident weights · --tensor-split ' + emitted) : '';
      }
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
      if (!sel || document.activeElement === sel) return;
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
          ? ((isIntegratedGpu(gpu) ? 'iGPU GTT' : 'VRAM') + ' · ' + gpuLabel(gpu, index) + ' · est. at full context')
          : (isIntegratedGpu(gpu) ? 'iGPU GTT · est. at full context' : 'VRAM · est. at full context'),
        segments: [
          { key: 'weights', label: 'Weights', bytes: weights },
          { key: 'vision', label: 'Vision (CLIP)', bytes: vision || 0 },
          { key: 'draft', label: specLabel || 'Speculative', bytes: spec || 0 },
          { key: 'kv', label: 'KV cache (full ctx)', bytes: kv },
          { key: 'overhead', label: 'Overhead', bytes: overhead },
        ],
        totalBytes: weights + kv + overhead + (spec || 0) + (vision || 0),
        capacityBytes: gpuBarCapacityBytes(gpu),
      };
    }

    function overheadForGpu(i, mainIdx, share, gpuOverhead, peerOverhead) {
      if (!(share > 0)) return 0;
      return i === mainIdx ? gpuOverhead : (peerOverhead || 0);
    }

    function buildCharts(gpuWeights, cpuWeights, gpuKv, cpuKv, gpuOverhead, cpuOverhead, totalGpu, totalCpu, draftGpu, draftCpu, specLabel, split, gpuVision, cpuVision, peerOverhead, weightShares, kvPerGpu) {
      const gpus = (!cpuOnlyLive() && gpuInfos && gpuInfos.length) ? gpuInfos : (gpuInfo ? [gpuInfo] : []);
      const shares = effectiveTensorSplitShares(
        split && split.tensorSplit,
        gpus,
        split && split.splitMode,
        split && split.mainGpu
      );
      const wShares = (weightShares && weightShares.length === shares.length) ? weightShares : shares;
      const mainIdx = gpus.length ? clampMainGpu(split && split.mainGpu, gpus.length) : 0;
      const kvFor = (i) => (kvPerGpu && kvPerGpu[i] != null) ? kvPerGpu[i] : gpuKv * (shares[i] || 0);
      const labeled = gpus.length >= 2;
      const order = gpuDisplayOrder(gpus, mainIdx);
      const i0 = order[0];
      const i1 = order[1];
      const vram0 = gpus.length && i0 !== undefined && gpus[i0]
        ? buildGpuChart(
            i0,
            gpus[i0],
            gpuWeights * (wShares[i0] || 0),
            kvFor(i0),
            overheadForGpu(i0, mainIdx, shares[i0] || 0, gpuOverhead, peerOverhead),
            (draftGpu || 0) * (wShares[i0] || 0),
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
            capacityBytes: gpuBarCapacityBytes(gpuInfo),
          };
      const vram2 = i1 !== undefined && gpus[i1]
        ? buildGpuChart(
            i1,
            gpus[i1],
            gpuWeights * (wShares[i1] || 0),
            kvFor(i1),
            overheadForGpu(i1, mainIdx, shares[i1] || 0, gpuOverhead, peerOverhead),
            (draftGpu || 0) * (wShares[i1] || 0),
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
            { key: 'kv', label: 'KV cache (full ctx)', bytes: cpuKv },
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

    // Mirrors resolveDenseFfnShare in core/memoryEstimate.ts.
    function denseFfnShareOf(inputs) {
      if (!inputs || inputs.isMoe) return 0;
      if (inputs.denseFfnShare != null && isFinite(inputs.denseFfnShare)) {
        return Math.min(0.95, Math.max(0.05, Number(inputs.denseFfnShare)));
      }
      const ffn = Number(inputs.ffnLength) || 0;
      const embed = Number(inputs.embeddingLength) || 0;
      if (ffn <= 0 || embed <= 0) return 0.7;
      const ffnElems = 3 * ffn;
      const attnElems = 4 * embed;
      return Math.min(0.95, Math.max(0.05, ffnElems / (ffnElems + attnElems)));
    }

    function pleShareOf(inputs) {
      if (!inputs) return 0;
      if (inputs.pleShare != null && isFinite(inputs.pleShare)) {
        return Math.min(0.95, Math.max(0, Number(inputs.pleShare)));
      }
      return 0;
    }

    // Mirrors resolveLoadMode + lazyModeReadsFromDisk in core.
    function livePleFromDisk(L, pleBytes) {
      if (!(pleBytes > 0) || !L.tryMmap) return false;
      const lazy = L.lazyMode || 'auto';
      const wants = lazy === 'on' || (lazy === 'auto' && pleBytes > 4 * 1024 ** 3);
      if (!wants) return false;
      if (L.keepModelInMemory && !payload.isWindows) return false;
      return true;
    }

    function fmtNum(n) {
      return typeof n === 'number' && isFinite(n) ? Number(n).toLocaleString() : '—';
    }

    function fmtTokK(n) {
      if (typeof n !== 'number' || !isFinite(n)) return '—';
      const abs = Math.abs(n);
      if (abs >= 1000) {
        const digits = abs >= 10000 ? 0 : 1;
        const s = (n / 1000).toFixed(digits);
        return (s.endsWith('.0') ? s.slice(0, -2) : s) + 'k';
      }
      return Math.round(n).toLocaleString();
    }

    function fmtRate(n) {
      if (typeof n !== 'number' || !isFinite(n) || n <= 0) return undefined;
      return n >= 100 ? n.toFixed(0) : n.toFixed(1);
    }

    function niceCeil(n) {
      if (!(n > 0)) return 20;
      if (n <= 20) return 20;
      if (n <= 100) return Math.ceil(n / 10) * 10;
      return Math.ceil(n / 100) * 100;
    }

    function maxPositive(values) {
      let m = 0;
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (typeof v === 'number' && isFinite(v) && v > m) m = v;
      }
      return m;
    }

    function pickNum() {
      for (let i = 0; i < arguments.length; i++) {
        const n = arguments[i];
        if (typeof n === 'number' && isFinite(n)) return n;
      }
      return undefined;
    }

    function specAcceptLabel(mode) {
      if (mode === 'dflash') return 'DFlash';
      if (mode === 'ngram-dflash') return 'N-gram+DFlash';
      if (mode === 'ngram') return 'N-gram';
      if (mode === 'ngram-mtp') return 'N-gram+MTP';
      return 'MTP';
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
        const pct = Math.round(ctxPct * 10) / 10;
        sub.textContent =
          approx + fmtTokK(promptTokens) + ' / ' +
          fmtTokK(contextLimit) + ' · ' + pct + '%' +
          (ctxLevel === 'critical' ? ' · nearly full' : ctxLevel === 'warn' ? ' · running low' : '');
        stack.setAttribute('aria-label',
          'Context used: ' + approx + fmtTokK(promptTokens) + ' of ' + fmtTokK(contextLimit) + ' tokens');
      } else {
        label.textContent = 'Context';
        sub.textContent = '— (send a chat to measure)';
        stack.setAttribute('aria-label', 'Context');
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

    function chartWidth(el) {
      let w = el ? el.getBoundingClientRect().width : 0;
      if (!(w > 0)) {
        const card = $('perfCard');
        w = card ? card.getBoundingClientRect().width - 20 : 0;
      }
      return Math.max(200, Math.round(w) || 240);
    }

    function dualAxisSvg(genValues, promptValues, widthPx) {
      const n = Math.max(genValues.length, promptValues.length);
      if (!n) return '';
      const hasGen = maxPositive(genValues) > 0;
      const hasPrompt = maxPositive(promptValues) > 0;
      if (!hasGen && !hasPrompt) return '';

      const W = Math.max(200, Math.round(widthPx) || 240);
      const H = 96;
      const left = 28, right = 44, top = 12, bottom = 14;
      const x0 = left, y0 = top, plotW = W - left - right, plotH = H - top - bottom;
      const GEN = '#58a6ff';
      const PROMPT = '#d29922';
      const genMax = niceCeil(maxPositive(genValues));
      const promptMax = niceCeil(maxPositive(promptValues));

      function ptsOf(values, yMax) {
        const pts = [];
        for (let i = 0; i < n; i++) {
          const v = values[i];
          if (typeof v !== 'number' || !isFinite(v) || v <= 0) continue;
          const x = n === 1 ? x0 + plotW / 2 : x0 + (i / (n - 1)) * plotW;
          const y = y0 + plotH - (v / yMax) * plotH;
          pts.push({ x: x, y: y, v: v });
        }
        return pts;
      }

      function poly(pts) {
        return pts.map(function (p) { return p.x.toFixed(1) + ',' + p.y.toFixed(1); }).join(' ');
      }

      function fmtAxis(v) {
        if (v === 0) return '0';
        return v >= 100 || Math.round(v) === v ? String(Math.round(v)) : v.toFixed(1);
      }

      const genPts = hasGen ? ptsOf(genValues, genMax) : [];
      const promptPts = hasPrompt ? ptsOf(promptValues, promptMax) : [];
      const parts = [];
      const fracs = [0, 0.5, 1];
      for (let i = 0; i < fracs.length; i++) {
        const t = fracs[i];
        const y = y0 + plotH - t * plotH;
        parts.push('<line x1="' + x0 + '" x2="' + (x0 + plotW) + '" y1="' + y.toFixed(1) +
          '" y2="' + y.toFixed(1) + '" stroke="rgba(128,128,128,0.18)" stroke-width="1"/>');
        if (hasGen) {
          parts.push('<text x="' + (x0 - 4) + '" y="' + (y + 3).toFixed(1) +
            '" text-anchor="end" fill="' + GEN + '" font-size="8">' + fmtAxis(t * genMax) + '</text>');
        }
        if (hasPrompt) {
          parts.push('<text x="' + (x0 + plotW + 4) + '" y="' + (y + 3).toFixed(1) +
            '" fill="' + PROMPT + '" font-size="8">' + fmtAxis(t * promptMax) + '</text>');
        }
      }
      if (promptPts.length > 1) {
        parts.push('<polyline points="' + poly(promptPts) + '" fill="none" stroke="' + PROMPT + '" stroke-width="1.5"/>');
      }
      if (genPts.length > 1) {
        parts.push('<polyline points="' + poly(genPts) + '" fill="none" stroke="' + GEN + '" stroke-width="1.7"/>');
      }
      if (promptPts.length) {
        const last = promptPts[promptPts.length - 1];
        const label = fmtRate(last.v) || fmtAxis(last.v);
        parts.push('<circle cx="' + last.x.toFixed(1) + '" cy="' + last.y.toFixed(1) + '" r="2.3" fill="' + PROMPT + '"/>');
        parts.push('<text x="' + (last.x - 4).toFixed(1) + '" y="' + (last.y - 5).toFixed(1) +
          '" text-anchor="end" fill="' + PROMPT + '" font-size="9" font-weight="650">' + label + '</text>');
      }
      if (genPts.length) {
        const last = genPts[genPts.length - 1];
        const label = fmtRate(last.v) || fmtAxis(last.v);
        parts.push('<circle cx="' + last.x.toFixed(1) + '" cy="' + last.y.toFixed(1) + '" r="2.5" fill="' + GEN + '"/>');
        parts.push('<text x="' + (last.x + 5).toFixed(1) + '" y="' + (last.y + 12).toFixed(1) +
          '" fill="' + GEN + '" font-size="9" font-weight="650">' + label + '</text>');
      }
      parts.push('<text x="' + x0 + '" y="' + (H - 2) + '" fill="currentColor" font-size="8">oldest</text>');
      parts.push('<text x="' + (x0 + plotW) + '" y="' + (H - 2) +
        '" text-anchor="end" fill="currentColor" font-size="8">latest</text>');

      return '<svg xmlns="http://www.w3.org/2000/svg" class="perf-chart dual" viewBox="0 0 ' + W + ' ' + H +
        '" role="img" aria-label="Generation tok/s left axis, prompt tok/s right axis, oldest left">' +
        parts.join('') + '<' + '/svg>';
    }

    function renderPerfStats(perf, perfLines) {
      const sessEl = $('perfSession');
      const legendEl = $('perfChartLegend');
      const metricsEl = $('perfMetrics');
      const tableEl = $('perfHistoryTable');
      const p = perf || {};
      const stale = !!(p.generating && p.showingPreviousCall);
      const rows = Array.isArray(p.history) ? p.history.filter(function (h) { return h && typeof h === 'object'; }) : [];
      const last = rows[0] || {};

      if (tableEl) {
        if (!rows.length) {
          tableEl.innerHTML = '<div class="muted" style="padding:6px 8px">No completed calls yet</div>';
        } else {
          const body = rows.map(function (h, i) {
            const when = h.finishedAt ? new Date(h.finishedAt).toLocaleTimeString() : '';
            const genH = fmtRate(h.genTokPerSec);
            const promptH = fmtRate(h.promptTokPerSec);
            const reuse = typeof h.cacheHitPct === 'number' ? h.cacheHitPct.toFixed(0) + '%' : '—';
            return '<tr><td>' + (i === 0 ? when + ' <span class="muted">latest</span>' : when) +
              '</td><td>' + (genH || '—') +
              '</td><td>' + (promptH || '—') +
              '</td><td>' + reuse + '</td></tr>';
          }).join('');
          tableEl.innerHTML = '<table><thead><tr><th>Call</th><th>Gen</th><th>Prompt</th><th>Reuse</th></tr></thead><tbody>' +
            body + '</tbody></table>';
        }
      }

      const cacheHitPct = pickNum(p.cacheHitPct, last.cacheHitPct);
      const completionTokens = pickNum(p.completionTokens, last.completionTokens);
      const draftAcceptancePct = pickNum(p.draftAcceptancePct, last.draftAcceptancePct);
      const durationMs = (!p.generating && p.finishedAt && p.startedAt)
        ? (p.finishedAt - p.startedAt)
        : last.durationMs;
      const genNow = p.generating ? p.genTokPerSec : pickNum(p.genTokPerSec, last.genTokPerSec);
      const promptNow = pickNum(p.promptTokPerSec, last.promptTokPerSec);

      let svg = '';
      try {
        const chrono = rows.slice().reverse();
        const genValues = [];
        const promptValues = [];
        for (let i = 0; i < chrono.length; i++) {
          genValues.push(chrono[i].genTokPerSec);
          promptValues.push(chrono[i].promptTokPerSec);
        }
        lastChartGen = genValues;
        lastChartPrompt = promptValues;
        if (sessEl) {
          const empty = !maxPositive(genValues) && !maxPositive(promptValues);
          if (empty) {
            sessEl.classList.add('hidden');
            sessEl.textContent = '';
            svg = '';
          } else {
            sessEl.classList.remove('hidden');
            svg = dualAxisSvg(genValues, promptValues, chartWidth(sessEl)) || '';
            if (svg) {
              sessEl.innerHTML = svg;
              if (!perfChartRO && typeof ResizeObserver !== 'undefined') {
                let raf = 0;
                perfChartRO = new ResizeObserver(function () {
                  if (raf) return;
                  raf = requestAnimationFrame(function () {
                    raf = 0;
                    if (!lastChartGen.length && !lastChartPrompt.length) return;
                    const next = dualAxisSvg(lastChartGen, lastChartPrompt, chartWidth(sessEl));
                    if (next) sessEl.innerHTML = next;
                  });
                });
                perfChartRO.observe(sessEl);
              }
            } else {
              sessEl.classList.add('hidden');
              sessEl.textContent = '';
            }
          }
        }
        if (legendEl) legendEl.classList.toggle('hidden', !svg);
      } catch (err) {
        svg = '';
        if (sessEl) {
          sessEl.classList.add('hidden');
          sessEl.textContent = '';
        }
        if (legendEl) legendEl.classList.add('hidden');
      }

      const bits = [];
      if (p.generating) {
        const live = fmtRate(p.genTokPerSec);
        bits.push('<span class="ok">● Generating…' + (live ? ' ' + live + ' tok/s' : '') + '</span>');
      } else if (!svg) {
        const genLabel = fmtRate(genNow);
        const promptLabel = fmtRate(promptNow);
        if (genLabel) bits.push('<span class="gen">' + genLabel + ' gen</span>');
        if (promptLabel) bits.push('<span class="prompt">' + promptLabel + ' prompt</span>');
      }
      if (typeof cacheHitPct === 'number') {
        bits.push('<span' + (stale ? ' class="muted"' : '') + '>' + cacheHitPct + '% reuse</span>');
      }
      if (p.speculativeMode && p.speculativeMode !== 'off') {
        const specName = specAcceptLabel(p.speculativeMode);
        if (typeof draftAcceptancePct === 'number') {
          const cls = stale ? 'muted' : (draftAcceptancePct >= 50 ? 'ok' : '');
          bits.push('<span' + (cls ? ' class="' + cls + '"' : '') + '>' +
            draftAcceptancePct.toFixed(1) + '% ' + specName + '</span>');
        } else if (!p.generating) {
          bits.push('<span class="muted">' + specName + ' —</span>');
        }
      }
      if (typeof completionTokens === 'number') {
        const dur = typeof durationMs === 'number' && durationMs > 0 && !p.generating
          ? ' · ' + (durationMs / 1000).toFixed(1) + 's'
          : '';
        bits.push('<span class="muted">' + fmtNum(completionTokens) + ' tok' + dur + '</span>');
      }
      if (metricsEl) {
        metricsEl.innerHTML = bits.length ? bits.join('') : (rows.length ? 'Last ' + rows.length + ' call' + (rows.length === 1 ? '' : 's') : 'No generation yet');
        metricsEl.title = Array.isArray(perfLines) ? perfLines.join('\\n') : '';
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
      // Headroom only applies when the bar is big enough to leave that much.
      const remaining = chart.capacityBytes != null ? chart.capacityBytes - chart.totalBytes : undefined;
      const cap = chart.capacityBytes || 0;
      const over = remaining !== undefined && (remaining < 0 || (cap > VRAM_HEADROOM && remaining < VRAM_HEADROOM));
      const warn = !over && remaining !== undefined && cap > VRAM_SOFT_HEADROOM && remaining < VRAM_SOFT_HEADROOM;
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
      const ffnShare = denseFfnShareOf(memInputs);
      const pleShare = pleShareOf(memInputs);
      const pleBytes = memInputs.fileSizeBytes * pleShare;
      const pleFromDisk = livePleFromDisk(L, pleBytes);
      let gpuWeights = memInputs.fileSizeBytes * (onGpu / nLayers);
      if (!cpuOnly && onGpu > 0 && pleShare > 0) {
        gpuWeights = Math.max(0, gpuWeights - pleBytes);
      }
      if (!cpuOnly && memInputs.isMoe && L.nCpuMoe > 0 && onGpu > 0 && expertShare > 0) {
        const moeCpu = Math.min(L.nCpuMoe, onGpu);
        gpuWeights = Math.max(0, gpuWeights - memInputs.fileSizeBytes * (moeCpu / nLayers) * expertShare);
      }
      if (!cpuOnly && !memInputs.isMoe && L.nCpuFfn > 0 && onGpu > 0 && ffnShare > 0) {
        const ffnCpu = Math.min(L.nCpuFfn, onGpu);
        gpuWeights = Math.max(0, gpuWeights - memInputs.fileSizeBytes * (ffnCpu / nLayers) * ffnShare);
      }
      let cpuWeights = Math.max(0, memInputs.fileSizeBytes - gpuWeights - (pleFromDisk ? pleBytes : 0));
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
      // Recurrent (SSM / linear-attention) layers keep no growing KV, but hold a
      // fixed f32 state per layer per sequence slot. Mirrors
      // estimateRecurrentStateBytesPerLayer() in core/memoryEstimate.ts.
      const ssmStateSize = Math.max(0, Math.round(Number(memInputs.ssmStateSize) || 0));
      const ssmInnerSize = Math.max(0, Math.round(Number(memInputs.ssmInnerSize) || Number(memInputs.embeddingLength) || 0));
      const ssmConvKernel = Math.min(16, Math.max(2, Math.round(Number(memInputs.ssmConvKernel) || 4)));
      const ssmGroupCount = Math.max(1, Math.round(Number(memInputs.ssmGroupCount) || 1));
      const ssmPerLayerBytes = (ssmStateSize > 0 && ssmStateSize <= 4096 && ssmInnerSize > 0 && ssmInnerSize <= 1000000)
        ? (ssmInnerSize * ssmStateSize + (ssmConvKernel - 1) * (ssmInnerSize + 2 * ssmGroupCount * ssmStateSize)) * 4
        : 0;
      function kvElemBytes(t) {
        // Mirrors kvCacheTypeElemBytes in core/memoryEstimate.ts (block-quant scale overhead included)
        if (t === 'q4_0' || t === 'iq4_nl') return 0.5625;
        if (t === 'q4_1') return 0.625;
        if (t === 'q5_0') return 0.6875;
        if (t === 'q5_1') return 0.75;
        if (t === 'q8_0') return 1;
        if (t === 'f32') return 4;
        return 2; // f16 / bf16
      }
      // llama.cpp offloads the *last* onGpu layers; each layer's KV lives on the
      // device that owns the layer. Mirrors estimateMemory() in core.
      const firstGpuLayer = Math.max(0, nLayers - onGpu);
      function kvAt(ctx) {
        const kBytes = kvElemBytes(L.cacheTypeK);
        const vBytes = kvElemBytes(L.cacheTypeV);
        let bytes = 0;
        let gpuBytes = 0;
        let fullAttnLayers = 0;
        for (let i = 0; i < nLayers; i++) {
          const isRecurrent = (recurrent && recurrent.length === nLayers)
            ? !!recurrent[i]
            : (perKv && perKv.length === nLayers && Number(perKv[i]) <= 0)
              ? true
              : !!(fullInterval && ((i + 1) % fullInterval !== 0));
          if (isRecurrent) {
            // Fixed per-slot SSM state (context-independent, still device memory).
            bytes += ssmPerLayerBytes;
            if (i >= firstGpuLayer) gpuBytes += ssmPerLayerBytes;
            continue;
          }
          fullAttnLayers++;
          const isSwa = !!(swa && pattern && pattern.length && pattern[i % pattern.length]);
          let nKv;
          if (perKv && perKv.length === nLayers && Number.isFinite(perKv[i])) {
            nKv = Number(perKv[i]);
            if (nKv <= 0) continue;
          } else {
            nKv = Math.max(1, defaultKvHeads);
          }
          const keyDim = (isSwa && memInputs.keyLengthSwa > 0) ? memInputs.keyLengthSwa : defaultKeyDim;
          const valDim = (isSwa && memInputs.valueLengthSwa > 0) ? memInputs.valueLengthSwa : defaultValDim;
          const tokens = isSwa ? Math.min(ctx, swa) : ctx;
          const layerBytes = (nKv * keyDim * kBytes + nKv * valDim * vBytes) * tokens;
          bytes += layerBytes;
          if (i >= firstGpuLayer) gpuBytes += layerBytes;
        }
        return { bytes, gpuBytes, fullAttnLayers };
      }
      const slots = Math.max(1, Math.round(Number(L.maxConcurrentPredictions)) || 1);
      const slotMul = L.unifiedKvCache || slots <= 1 ? 1 : slots;
      const fullKv = kvAt(L.contextLength);
      const warmCtx = Math.min(2048, Math.max(512, L.contextLength));
      const warmKv = kvAt(warmCtx);
      const kvBytes = fullKv.bytes * slotMul;
      const kvBytesWarm = warmKv.bytes * slotMul;
      const fullAttnLayers = fullKv.fullAttnLayers;
      const kvOnGpu = !cpuOnly && !!L.offloadKvCacheToGpu && onGpu > 0;
      // Under partial offload only the offloaded layers' KV is in VRAM.
      const gpuKv = kvOnGpu ? fullKv.gpuBytes * slotMul : 0;
      const gpuKvWarm = kvOnGpu ? warmKv.gpuBytes * slotMul : 0;
      const cpuKv = Math.max(0, kvBytes - gpuKv);
      const cpuKvWarm = Math.max(0, kvBytesWarm - gpuKvWarm);
      const kvSplit = kvOnGpu && cpuKv > 1024 * 1024;
      // Mirrors computeOverheadBytes() / peerGpuOverheadBytes() in memoryEstimate.ts.
      const gpusForOh = (!cpuOnly && gpuInfos && gpuInfos.length) ? gpuInfos : (gpuInfo ? [gpuInfo] : []);
      const ohIds = gpusForOh.map((g) => String((g && g.llamaDeviceId) || '').toLowerCase());
      const ohBackend = cpuOnly
        ? 'cpu'
        : (ohIds.some((id) => id.indexOf('vulkan') === 0)
          ? 'vulkan'
          : (ohIds.some((id) => /^(cuda|rocm|hip)/.test(id))
            ? 'cuda'
            : (ohIds.some((id) => id.indexOf('metal') === 0) ? 'metal' : 'unknown')));
      const embedForOverhead = Math.max(2048, memInputs.embeddingLength || 4096);
      const ubatchForOverhead = Math.min(Math.max(32, L.physicalBatchSize || 512), 8192);
      const batchForOverhead = Math.min(Math.max(32, L.evalBatchSize || 2048), 8192);
      const ctxForOverhead = Math.min(Math.max(512, L.contextLength || 4096), 262144);
      const splitNone = L.splitMode === 'none';
      const vulkanDeviceCount = (onGpu > 0 && ohBackend === 'vulkan')
        ? ((gpusForOh.length >= 2 && !splitNone) ? gpusForOh.length : 1)
        : 0;
      // Mirrors vulkanDeviceReservedBytes() / vulkanDriverBytes(): 384 MiB
      // driver + 256 MiB slop on one Vulkan device. Split Flash-Next keeps
      // 768 MiB driver + 2.5 GiB reserve; Qwen3.5 RCO (and qwen4exp with
      // SSM geometry) keeps the 256 MiB slop even when split.
      // Linear-recurrent hybrids also run a smaller graph; full/SWA
      // Flash-Next without ssm.* keeps the dense graph.
      // Inline isLinearRecurrentHybrid(): this is webview JS, the TS import
      // above is not in scope here.
      const archNorm = String((memInputs && memInputs.architecture) || '').toLowerCase().replace(/[._-]/g, '');
      const ssmState = Number(memInputs && memInputs.ssmStateSize) || 0;
      const discountRecurrent = archNorm.indexOf('qwen35') === 0 || (ssmState > 0 && ssmState <= 4096);
      const ohTax = {
        vulkan: { driver: 768 * 1024 * 1024, peer: 512 * 1024 * 1024, graph: 12, reserved: 2.5 * 1024 ** 3 },
        cuda: { driver: 384 * 1024 * 1024, peer: 256 * 1024 * 1024, graph: 8, reserved: 0 },
        metal: { driver: 384 * 1024 * 1024, peer: 256 * 1024 * 1024, graph: 8, reserved: 0 },
        unknown: { driver: 512 * 1024 * 1024, peer: 384 * 1024 * 1024, graph: 10, reserved: 0 },
      }[ohBackend] || { driver: 512 * 1024 * 1024, peer: 384 * 1024 * 1024, graph: 10, reserved: 0 };
      const reservedBytes = ohBackend === 'vulkan'
        ? (vulkanDeviceCount <= 0 ? 0 : ((vulkanDeviceCount > 1 && !discountRecurrent) ? 2.5 * 1024 ** 3 : 256 * 1024 * 1024))
        : (ohTax.reserved || 0);
      const driverBytes = ohBackend === 'vulkan'
        ? (vulkanDeviceCount > 1 ? ohTax.driver : (vulkanDeviceCount === 1 ? 384 * 1024 * 1024 : 0))
        : ohTax.driver;
      const graphElem = L.flashAttention === 'off' ? Math.max(ohTax.graph, 36) : ohTax.graph;
      const fullAttnFrac = discountRecurrent && nLayers > 0
        ? Math.min(1, Math.max(0.05, fullAttnLayers / nLayers))
        : 1;
      const graphRaw = ctxForOverhead * embedForOverhead * graphElem * fullAttnFrac;
      const graphCap = L.flashAttention === 'off'
        ? 6 * 1024 * 1024 * 1024
        : (ohBackend === 'vulkan' ? 2.5 * 1024 ** 3 : (ohBackend === 'cuda' ? 1.75 * 1024 ** 3 : 2 * 1024 ** 3));
      const overhead = ohBackend === 'cpu'
        ? Math.round(256 * 1024 * 1024)
        : Math.round(
          driverBytes +
          reservedBytes +
          ubatchForOverhead * embedForOverhead * 96 +
          batchForOverhead * 8 * 1024 +
          Math.min(graphRaw, graphCap)
        );
      const gpuOverhead = onGpu > 0 ? overhead : 0;
      const peerReserved = ohBackend === 'vulkan'
        ? (discountRecurrent ? 256 * 1024 * 1024 : (ohTax.reserved || 0))
        : (ohTax.reserved || 0);
      const peerOverhead = (onGpu > 0 && gpusForOh.length >= 2 && !splitNone && ohBackend !== 'cpu')
        ? Math.round(ohTax.peer + peerReserved + ubatchForOverhead * embedForOverhead * 32)
        : 0;
      const cpuOverhead = onGpu > 0 ? Math.round(overhead * 0.15) : Math.round(overhead * 0.5);

      const warnings = [];
      let willSpill = false;

      // DFlash draft weights + f16 KV (mirrors estimateMemory draft footprint).
      let draftGpuBundle = 0;
      let draftCpuBundle = 0;
      let draftGpuWarmBundle = 0;
      let draftCpuWarmBundle = 0;
      let draftLine = '';
      const sidecarMtp = (L.speculativeMode === 'mtp' || L.speculativeMode === 'ngram-mtp') && memInputs.draft && memInputs.draft.fileSizeBytes && !(Number(memInputs.nextnPredictLayers) > 0);
      const draftIn = ((L.speculativeMode === 'dflash' || L.speculativeMode === 'ngram-dflash' || sidecarMtp) && memInputs.draft && memInputs.draft.fileSizeBytes)
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
          // Mirrors draftKvCaps() in core/memoryEstimate.ts: a sidecar mtp-*.gguf
          // carries the parent's block_count/interval/recurrent mask but holds
          // only the next-n heads, so its cache is one next-n layer, not the
          // parent's 16 full-attn layers (~5 GiB of cache that never exists).
          const nextn = Math.max(1, Math.floor(Number(draftIn.nextnPredictLayers) || 1));
          const sidecarReduced = sidecarMtp && nextn < dLayers;
          const kvLayers = sidecarReduced ? nextn : dLayers;
          const swa = !sidecarReduced && draftIn.slidingWindow > 0 ? draftIn.slidingWindow : 0;
          const pattern = sidecarReduced ? null : draftIn.slidingWindowPattern;
          const perKv = sidecarReduced ? null : draftIn.attentionHeadCountKvPerLayer;
          const recurrent = sidecarReduced ? null : draftIn.recurrentLayers;
          const fullInterval = !sidecarReduced && draftIn.fullAttentionInterval > 1 ? draftIn.fullAttentionInterval : 0;
          let bytes = 0;
          for (let i = 0; i < kvLayers; i++) {
            const isRecurrent = (recurrent && recurrent.length === kvLayers)
              ? !!recurrent[i]
              : (perKv && perKv.length === kvLayers && Number(perKv[i]) <= 0)
                ? true
                : !!(fullInterval && ((i + 1) % fullInterval !== 0));
            if (isRecurrent) continue;
            const isSwa = !!(swa && pattern && pattern.length && pattern[i % pattern.length]);
            let nKv;
            if (perKv && perKv.length === kvLayers && Number.isFinite(perKv[i])) {
              nKv = Number(perKv[i]);
              if (nKv <= 0) continue;
            } else {
              nKv = Math.max(1, defaultKvHeads);
            }
            const keyDim = (isSwa && draftIn.keyLengthSwa > 0) ? draftIn.keyLengthSwa : defaultKeyDim;
            const valDim = (isSwa && draftIn.valueLengthSwa > 0) ? draftIn.valueLengthSwa : defaultValDim;
            const tokens = isSwa ? Math.min(ctx, swa) : ctx;
            bytes += (nKv * keyDim * kBytes + nKv * valDim * vBytes) * tokens;
          }
          return bytes;
        }
        const dKv = draftKvAt(L.contextLength) * slotMul;
        const dKvWarm = draftKvAt(warmCtx) * slotMul;
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
      } else if (L.speculativeMode === 'dflash' || L.speculativeMode === 'ngram-dflash') {
        warnings.push('DFlash is on but no draft GGUF is selected — memory bars omit the draft; pick a draft model before starting.');
      } else if ((L.speculativeMode === 'mtp' || L.speculativeMode === 'ngram-mtp') && !sidecarMtp) {
        const mtpLayers = Math.max(0, Math.floor(Number(memInputs.nextnPredictLayers) || 0));
        if (mtpLayers > 0) {
          function mtpKvAt(ctx) {
            const kBytes = kvElemBytes(L.cacheTypeK);
            const vBytes = kvElemBytes(L.cacheTypeV);
            const heads = Math.max(1, memInputs.attentionHeadCount || 8);
            const nKv = Math.max(1, memInputs.attentionHeadCountKv || heads);
            const keyDim = Math.max(1, memInputs.keyLength || Math.floor((memInputs.embeddingLength || heads * 128) / heads));
            const valDim = Math.max(1, memInputs.valueLength || keyDim);
            return mtpLayers * (nKv * keyDim * kBytes + nKv * valDim * vBytes) * ctx;
          }
          const mtpKv = mtpKvAt(L.contextLength) * slotMul;
          const mtpKvWarm = mtpKvAt(warmCtx) * slotMul;
          const mtpKvOnGpu = kvOnGpu;
          draftGpuBundle = mtpKvOnGpu ? mtpKv : 0;
          draftCpuBundle = mtpKvOnGpu ? 0 : mtpKv;
          draftGpuWarmBundle = mtpKvOnGpu ? mtpKvWarm : 0;
          draftCpuWarmBundle = mtpKvOnGpu ? 0 : mtpKvWarm;
          draftLine = 'MTP: next-n heads already in GGUF · extra KV ~' + fmtBytes(mtpKv) +
            ' (' + mtpLayers + ' layers)' + (mtpKvOnGpu ? ' (GPU)' : ' (CPU RAM)');
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

      const liveShares = (!cpuOnly && gpusForOh.length)
        ? effectiveTensorSplitShares(L.tensorSplit, gpusForOh, L.splitMode, L.mainGpu)
        : [1];
      const liveMassOpts = {
        isMoe: !!memInputs.isMoe,
        nCpuMoe: L.nCpuMoe,
        moeExpertShare: expertShare,
        nCpuFfn: L.nCpuFfn,
        denseFfnShare: ffnShare,
      };
      const liveWeightShares = (!cpuOnly && gpusForOh.length >= 2)
        ? layerAwareWeightSharesLive(nLayers, onGpu, liveShares, L.splitMode, L.mainGpu, liveMassOpts)
        : liveShares;
      const liveKvShares = (!cpuOnly && gpusForOh.length >= 2)
        ? layerAwareWeightSharesLive(nLayers, onGpu, liveShares, L.splitMode, L.mainGpu, {})
        : liveShares;
      const liveKvPerGpu = liveShares.map((_, i) => (kvOnGpu ? gpuKv * (liveKvShares[i] || 0) : 0));
      const liveMainIdx = gpusForOh.length ? Math.min(Math.max(0, Number(L.mainGpu) || 0), gpusForOh.length - 1) : 0;
      const livePeerCount = gpusForOh.filter((_, i) => i !== liveMainIdx && (liveShares[i] || 0) > 0).length;
      const totalPeer = peerOverhead * livePeerCount;
      const totalGpu = gpuWeights + gpuKv + gpuOverhead + totalPeer + draftGpuBundle + gpuVisionBytes;
      const totalCpu = cpuWeights + cpuKv + cpuOverhead + draftCpuBundle + cpuVisionBytes;
      const totalGpuWarm = gpuWeights + gpuKvWarm + gpuOverhead + totalPeer + draftGpuWarmBundle + gpuVisionBytes;
      const totalCpuWarm = cpuWeights + cpuKvWarm + cpuOverhead + draftCpuWarmBundle + cpuVisionBytes;
      if (cpuOnly) {
        warnings.push('CPU backend: no GPU acceleration — weights, KV cache, and compute use system RAM (GPU Offload is ignored).');
      }
      if (!cpuOnly && onGpu > 0 && onGpu < nLayers) {
        warnings.push('Partial GPU offload: ' + (nLayers - onGpu) + '/' + nLayers + ' layers (~' + fmtBytes(cpuWeights) + ') stay in system RAM.');
      }
      const singleIgpu = !cpuOnly && gpuInfos && gpuInfos.length === 1 && isIntegratedGpu(gpuInfos[0]);
      if (!cpuOnly && onGpu === 0 && !singleIgpu) warnings.push('GPU offload is 0 — weights run from system RAM.');
      if (!cpuOnly && !L.offloadKvCacheToGpu) warnings.push('KV cache (~' + fmtBytes(kvBytes) + ' at full context) is in system RAM.');
      else if (kvSplit) warnings.push('KV cache follows the layers: ~' + fmtBytes(gpuKv) + ' in VRAM for the ' + onGpu + ' offloaded layers, ~' + fmtBytes(cpuKv) + ' in system RAM for the ' + (nLayers - onGpu) + ' CPU layers (at full context).');
      if (!cpuOnly && memInputs.isMoe && L.nCpuMoe > 0) {
        warnings.push('CPU MoE layers = ' + L.nCpuMoe + ': ~' + Math.round(expertShare * 100) + '% of weights are experts; those layers’ experts stay in system RAM.');
        if (
          gpusForOh.length >= 2 &&
          L.splitMode !== 'none' && L.splitMode !== 'row' && L.splitMode !== 'tensor' &&
          liveWeightShares.some((w, i) => (w || 0) > (liveShares[i] || 0) + 0.08)
        ) {
          warnings.push('Layer split + CPU MoE: llama.cpp assigns the first layers (cheap after --n-cpu-moe) to earlier GPUs, so later cards hold more expert weight than the tensor-split percentages suggest.');
        }
      }
      if (!cpuOnly && !memInputs.isMoe && L.nCpuFfn > 0) {
        warnings.push('CPU FFN layers = ' + L.nCpuFfn + ': ~' + Math.round(ffnShare * 100) + '% of per-layer weights are dense FFN; those layers’ FFN stays in system RAM.');
      }
      if (!cpuOnly && gpuInfos && gpuInfos.length) {
        const shares = liveShares;
        const mainIdx = liveMainIdx;
        if (singleIgpu) {
          const g = gpuInfos[0];
          const used = gpuWeights * (liveWeightShares[0] || 0) + (liveKvPerGpu[0] || 0) + overheadForGpu(0, mainIdx, shares[0] || 0, gpuOverhead, peerOverhead) + (draftGpuBundle * (liveWeightShares[0] || 0)) + gpuVisionBytes;
          const vram = fmtBytes(g.totalBytes);
          const gttCap = g.gttTotalBytes || 0;
          const gttLabel = gttCap > 0 ? '~' + fmtBytes(gttCap) : 'system RAM';
          if (onGpu <= 0) {
            warnings.unshift('CPU only — iGPU unused. The ' + vram + ' VRAM bar is the carve-out, not your budget.');
            warnings.push('To try the iGPU, set GPU offload to ' + nLayers + '. Weights go to GTT (' + gttLabel + '), not that ' + vram + '.');
          } else if (gttCap > 0 && used > gttCap) {
            willSpill = true;
            warnings.unshift('Too big for GTT (~' + fmtBytes(used) + ' of ' + fmtBytes(gttCap) + '). Lower context or keep offload at 0.');
          } else {
            warnings.unshift('iGPU via GTT — ~' + fmtBytes(gpuWeights) + ' weights in system RAM. Dedicated ' + vram + ' VRAM is unused; that is normal.');
          }
        } else {
        for (let i = 0; i < gpuInfos.length; i++) {
          const g = gpuInfos[i];
          const share = shares[i] || 0;
          const used = gpuWeights * (liveWeightShares[i] || 0) + (liveKvPerGpu[i] || 0) + overheadForGpu(i, mainIdx, share, gpuOverhead, peerOverhead) + (draftGpuBundle * (liveWeightShares[i] || 0)) + (i === mainIdx ? gpuVisionBytes : 0);
          const cap = g.totalBytes;
          if (!cap) continue;
          const pct = Math.round((used / cap) * 100);
          const label = gpuLabel(g, i);
          if (used > cap) {
            willSpill = true;
            warnings.unshift('Estimated ' + label + ' at full context ~' + fmtBytes(used) + ' is over the full ' + fmtBytes(cap) + ' (' + pct + '%). Expect spill to system RAM. Lower Context or GPU Offload.');
          } else if (cap > VRAM_HEADROOM && used > cap - VRAM_HEADROOM) {
            willSpill = true;
            warnings.unshift('Tight on ' + label + ' at full context: ~' + fmtBytes(used) + ' of ' + fmtBytes(cap) + ' (' + pct + '%). Only ~' + fmtBytes(cap - used) + ' left — target is ' + fmtBytes(VRAM_HEADROOM) + ' free. Lower Context or GPU Offload.');
          } else if (cap > VRAM_SOFT_HEADROOM && cap - used < VRAM_SOFT_HEADROOM) {
            warnings.push('Getting full on ' + label + ' at full context: ~' + fmtBytes(used) + ' of ' + fmtBytes(cap) + ' VRAM (' + pct + '%).');
          }
        }
        // Mirrors hostFallbackWarning() — skip APUs; only discrete GTT fallback.
        for (let i = 0; i < gpuInfos.length; i++) {
          const g = gpuInfos[i];
          if (isIntegratedGpu(g)) continue;
          const cap = g.totalBytes || 0;
          if (!cap) continue;
          const vramUsed = g.usedBytes || 0;
          const gttUsed = g.gttUsedBytes || 0;
          const visTotal = g.visVramTotalBytes || 0;
          const smallBar = visTotal > 0 && visTotal < cap / 2;
          const gttHeavy = gttUsed >= 1024 ** 3 && gttUsed > Math.max(vramUsed * 2, cap * 0.25);
          if (!gttHeavy && !(smallBar && gttUsed >= 1024 ** 3)) continue;
          warnings.unshift(
            gpuLabel(g, i) + ' is in GTT (~' + fmtBytes(gttUsed) + '), not VRAM.' +
            (smallBar ? ' Likely ReBAR off.' : '') +
            ' That card will be slow. Use one GPU, or fix ReBAR.'
          );
        }
        }
        if (gpuInfos.length >= 2 && L.splitMode !== 'none' && parseTensorSplit(L.tensorSplit).length < 2) {
          warnings.push('Tensor split is empty — llama.cpp will split by VRAM size (often 1:1). Pick the faster card as Main GPU and raise Weights on main GPU so that card gets more of the model.');
        }
      } else if (!cpuOnly && gpuInfo && gpuInfo.totalBytes) {
        if (isIntegratedGpu(gpuInfo)) {
          const vram = fmtBytes(gpuInfo.totalBytes);
          const gttCap = gpuInfo.gttTotalBytes || 0;
          const gttLabel = gttCap > 0 ? '~' + fmtBytes(gttCap) : 'system RAM';
          if (onGpu <= 0) {
            warnings.unshift('CPU only — iGPU unused. The ' + vram + ' VRAM bar is the carve-out, not your budget.');
            warnings.push('To try the iGPU, set GPU offload to ' + nLayers + '. Weights go to GTT (' + gttLabel + '), not that ' + vram + '.');
          } else if (gttCap > 0 && totalGpu > gttCap) {
            willSpill = true;
            warnings.unshift('Too big for GTT (~' + fmtBytes(totalGpu) + ' of ' + fmtBytes(gttCap) + '). Lower context or keep offload at 0.');
          } else {
            warnings.unshift('iGPU via GTT — ~' + fmtBytes(gpuWeights) + ' weights in system RAM. Dedicated ' + vram + ' VRAM is unused; that is normal.');
          }
        } else {
        const cap = gpuInfo.totalBytes;
        const pct = Math.round((totalGpu / cap) * 100);
        if (totalGpu > cap) {
          willSpill = true;
          warnings.unshift('Estimated VRAM at full context ~' + fmtBytes(totalGpu) + ' is over the full ' + fmtBytes(cap) + ' GPU (' + pct + '%). Expect spill to system RAM. Lower Context or GPU Offload.');
        } else if (cap > VRAM_HEADROOM && totalGpu > cap - VRAM_HEADROOM) {
          willSpill = true;
          warnings.unshift('Tight on VRAM at full context: ~' + fmtBytes(totalGpu) + ' of ' + fmtBytes(cap) + ' (' + pct + '%). Only ~' + fmtBytes(cap - totalGpu) + ' left — target is ' + fmtBytes(VRAM_HEADROOM) + ' free. Lower Context or GPU Offload.');
        } else if (cap > VRAM_SOFT_HEADROOM && cap - totalGpu < VRAM_SOFT_HEADROOM) {
          warnings.push('Getting full at full context: ~' + fmtBytes(totalGpu) + ' of ' + fmtBytes(cap) + ' VRAM (' + pct + '%).');
        }
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
          if ((g.gttUsedBytes || 0) > 0) {
            lines.push('Live ' + gpuLabel(g, i) + ' GTT now: ~' + fmtBytes(g.gttUsedBytes) + ' of host RAM mapped to the GPU (VRAM in use ~' + fmtBytes(g.usedBytes || 0) + ')');
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
          lines.push('KV @ ~' + warmCtx.toLocaleString() + ' ctx (mid-chat): ~' + fmtBytes(kvBytesWarm) + ' → total ~' + fmtBytes(totalCpuWarm));
        }
        lines.push('Est. total system RAM at full context: ~' + fmtBytes(totalCpu));
      } else {
        lines.push('Weights on GPU: ~' + fmtBytes(gpuWeights) + ' (' + onGpu + '/' + nLayers + ' layers)' + (cpuWeights > 1024*1024 ? ' · RAM: ~' + fmtBytes(cpuWeights) : '') +
          (memInputs.isMoe && expertShare > 0 ? (' · MoE experts ~' + Math.round(expertShare * 100) + '% of file') : ''));
        const kvPlacement = kvSplit
          ? ' (~' + fmtBytes(gpuKv) + ' GPU · ~' + fmtBytes(cpuKv) + ' CPU RAM)'
          : (kvOnGpu ? ' (GPU)' : ' (CPU RAM)');
        lines.push('KV @ full ' + Number(L.contextLength).toLocaleString() + ' ctx: ~' + fmtBytes(kvBytes) + kvPlacement +
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
          lines.push('KV @ ~' + warmCtx.toLocaleString() + ' ctx (mid-chat): ~' + fmtBytes(kvBytesWarm) +
            (kvSplit ? ' (~' + fmtBytes(gpuKvWarm) + ' GPU · ~' + fmtBytes(cpuKvWarm) + ' CPU RAM)' : (kvOnGpu ? ' (GPU)' : ' (CPU RAM)')) +
            ' → VRAM ~' + fmtBytes(totalGpuWarm));
        }
        lines.push('Est. total at full context — VRAM: ~' + fmtBytes(totalGpu) + (totalCpu > 1024*1024 ? ' · system RAM: ~' + fmtBytes(totalCpu) : ''));
      }
      lines.push('Bars show estimate at full context, including Vulkan/CUDA compute and per-GPU heaps. Actual use still varies by quant and driver.');
      const specLabel = draftIn
        ? (sidecarMtp ? 'MTP draft (weights + KV)' : 'DFlash draft (weights + KV)')
        : ((L.speculativeMode === 'mtp' || L.speculativeMode === 'ngram-mtp') && draftGpuBundle + draftCpuBundle > 0
          ? 'MTP head + KV'
          : 'Speculative');
      const charts = buildCharts(gpuWeights, cpuWeights, gpuKv, cpuKv, gpuOverhead, cpuOverhead, totalGpu, totalCpu, draftGpuBundle, draftCpuBundle, specLabel, { tensorSplit: L.tensorSplit, mainGpu: L.mainGpu, splitMode: L.splitMode }, gpuVisionBytes, cpuVisionBytes, peerOverhead, liveWeightShares, liveKvPerGpu);
      if (cpuOnly) {
        charts.vram.capacityBytes = undefined;
      }
      let summary;
      const specBytes = draftGpuBundle + draftCpuBundle;
      const specSuffix = specBytes > 0
        ? (L.speculativeMode === 'dflash' || L.speculativeMode === 'ngram-dflash'
          ? (cpuOnly
            ? ' · DFlash +' + fmtBytes(specBytes)
            : ' · DFlash +' + fmtBytes(draftGpuBundle) + (draftCpuBundle > 1024 * 1024 ? ' (+' + fmtBytes(draftCpuBundle) + ' RAM)' : ''))
          : (L.speculativeMode === 'mtp' || L.speculativeMode === 'ngram-mtp')
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
          ' · KV ~' + fmtBytes(kvBytes) + (kvSplit ? ' (~' + fmtBytes(gpuKv) + ' on GPU)' : (kvOnGpu ? ' on GPU' : ' in RAM')) +
          ' · ' + onGpu + '/' + nLayers + ' layers offloaded' + specSuffix;
      } else {
        const cap = gpuBarCapacityBytes(gpuInfo) || (gpuInfo && gpuInfo.totalBytes);
        const pct = cap ? Math.round((totalGpu / cap) * 100) : undefined;
        const kind = isIntegratedGpu(gpuInfo) ? 'iGPU GTT' : 'VRAM';
        summary = kind + ' ~' + fmtBytes(totalGpu) +
          (cap ? ' of ' + fmtBytes(cap) + (pct !== undefined ? ' (' + pct + '%)' : '') : '') +
          ' · KV ~' + fmtBytes(kvBytes) + (kvSplit ? ' (~' + fmtBytes(gpuKv) + ' on GPU)' : (kvOnGpu ? ' on GPU' : ' in RAM')) +
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
      $('moeFields').classList.toggle('hidden', !showMoe);
      if (modelIsMoe) {
        $('moeHint').textContent = cpuOnly
          ? 'CPU backend — experts already run in system RAM; CPU MoE layers (--n-cpu-moe) does not apply.'
          : moeHintDefault;
      }
      // --n-cpu-ffn is the dense counterpart: hidden for MoE models (experts use
      // --n-cpu-moe) and on the CPU backend.
      $('nCpuFfn').disabled = cpuOnly;
      $('nCpuFfnRange').disabled = cpuOnly;
      $('ffnFields').classList.toggle('hidden', modelIsMoe || cpuOnly);
      if (!modelIsMoe) {
        $('ffnHint').textContent = cpuOnly
          ? 'CPU backend — weights already run in system RAM; CPU FFN layers (--n-cpu-ffn) does not apply.'
          : ffnHintDefault;
      }
      $('moeRow').classList.toggle('hidden', cpuOnly);

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
            : names + '. Pick the faster card as Main GPU (Vulkan/CUDA order from llama.cpp, which may differ from btop). The slider is GPU-resident weight share on that card after CPU MoE/FFN.';
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
      const nLayers = Math.max(1, (memInputs && memInputs.blockCount) || 1);
      const share = mainWeightShareFromSplitLive(
        split,
        L.mainGpu ?? 0,
        gpuInfos,
        L.splitMode,
        liveMassOpts(L),
        nLayers,
        liveLayersOnGpu(L)
      );
      const pct = Math.min(90, Math.max(10, Math.round(share * 100)));
      const range = $('tensorSplitRange');
      if (range && document.activeElement !== range) range.value = String(pct);
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

    let liveMemRaf = 0;
    function scheduleLiveMemory() {
      if (liveMemRaf) return;
      liveMemRaf = requestAnimationFrame(() => {
        liveMemRaf = 0;
        refreshMemoryLive();
      });
    }

    /**
     * Keep the number input in sync with a range. align (context length) uses
     * step=1 on the range so the thumb tracks the pointer on a 256k-wide axis;
     * the saved value still snaps. Writing range.value during input fights
     * native dragging in Chromium.
     */
    function bindRange(numId, rangeId, align) {
      const num = $(numId);
      const range = $(rangeId);
      if (!num || !range) return;
      function snap(v) {
        const min = Number(range.min);
        const max = Number(range.max);
        let n = Number(v);
        if (!Number.isFinite(n)) n = Number.isFinite(min) ? min : 0;
        if (align > 0) n = Math.round(n / align) * align;
        if (Number.isFinite(min)) n = Math.max(min, n);
        if (Number.isFinite(max) && max >= min) n = Math.min(max, n);
        return n;
      }
      num.addEventListener('input', () => {
        range.value = String(snap(num.value));
        scheduleLiveMemory();
      });
      range.addEventListener('input', () => {
        num.value = String(snap(range.value));
        scheduleLiveMemory();
      });
      if (align > 0) {
        range.addEventListener('change', () => {
          const n = snap(range.value);
          num.value = String(n);
          range.value = String(n);
        });
      }
    }
    bindRange('contextLength', 'contextLengthRange', 256);
    bindRange('gpuOffload', 'gpuOffloadRange');
    bindRange('cpuThreads', 'cpuThreadsRange');
    bindRange('nCpuMoe', 'nCpuMoeRange');
    bindRange('nCpuFfn', 'nCpuFfnRange');
    bindRange('ngramSizeN', 'ngramSizeNRange');
    const tsRange = $('tensorSplitRange');
    if (tsRange) tsRange.addEventListener('input', syncTensorSplitPctLabel);
    $('offloadKvCacheToGpu').addEventListener('change', refreshMemoryLive);
    $('evalBatchSize').addEventListener('input', refreshMemoryLive);
    $('physicalBatchSize').addEventListener('input', refreshMemoryLive);

    /** Higher = more precise. Used to flag lopsided K/V pairs; < 2 = quantized. */
    const KV_PRECISION_RANK = {
      q4_0: 0,
      iq4_nl: 0.5,
      q4_1: 0.6,
      q5_0: 0.7,
      q5_1: 0.8,
      q8_0: 1,
      bf16: 2,
      f16: 2,
      f32: 3,
    };

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

    /**
     * Largest context (8192…model max, aligned to 256) whose live estimate
     * still leaves 1.5 GiB free on every card. Mirrors fittingContextLength in core.
     */
    function fittingContext(maxCtx) {
      if (!memInputs) return maxCtx;
      const previous = $('contextLength').value;
      const align = 256;
      const high = Math.min(maxCtx, Math.max(align, Math.floor(maxCtx / align) * align));
      const lowMin = Math.min(8192, high);
      function fits(ctx) {
        $('contextLength').value = ctx;
        const est = liveMemoryEstimate();
        return !!(est && !est.willSpill);
      }
      let best = lowMin;
      if (fits(high)) {
        $('contextLength').value = previous;
        return high;
      }
      if (!fits(lowMin)) {
        $('contextLength').value = previous;
        return lowMin;
      }
      let lo = lowMin;
      let hi = high;
      while (lo <= hi) {
        let mid = Math.max(align, Math.floor((lo + hi) / 2 / align) * align);
        if (mid < lo) mid = lo;
        if (mid > hi) break;
        if (fits(mid)) {
          best = mid;
          lo = mid + align;
        } else {
          hi = mid - align;
        }
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
      'maxConcurrentPredictions', 'nCpuMoe', 'nCpuMoeRange', 'nCpuFfn', 'nCpuFfnRange', 'offloadKvCacheToGpu',
      'mmprojOffloadToGpu',
      'cacheTypeK', 'cacheTypeV',
      'keepModelInMemory', 'tryMmap', 'lazyMode', 'unifiedKvCache', 'flashAttention',
      'contextCheckpoints', 'cacheReuse',
      'reasoningFormat', 'reasoningBudgetUnlimited', 'reasoningBudget',
      'ropeBaseAuto', 'ropeFreqBase', 'ropeScaleAuto', 'ropeFreqScale',
      'seedRandom', 'seed', 'speculativeMode', 'maxDraftTokens', 'minDraftTokens',
      'draftProbability', 'draftGpuOffload',
      'ngramVariant', 'ngramSizeN', 'ngramSizeNRange', 'ngramSizeM', 'ngramMinHits',
      'tensorSplitRange', 'splitMode', 'mainGpu'
    ];
    for (const id of loadFieldIds) {
      const el = $(id);
      if (!el) continue;
      el.addEventListener('change', () => { scheduleSaveLoad(); refreshMemoryLive(); });
      el.addEventListener('input', () => { scheduleSaveLoad(); scheduleLiveMemory(); syncTensorSplitPctLabel(); });
    }

    // Request defaults apply to the next chat call (no server reload). Persist on edit.
    let saveRequestTimer = null;
    function scheduleSaveRequest() {
      if (saveRequestTimer) clearTimeout(saveRequestTimer);
      saveRequestTimer = setTimeout(() => {
        vscode.postMessage({ type: 'saveRequest', payload: readRequest() });
      }, 250);
    }
    for (const id of ['temperature', 'topP', 'topK', 'minP', 'repeatPenalty', 'presencePenalty', 'frequencyPenalty', 'maxTokens']) {
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
      if ((speculativeMode === 'mtp' || speculativeMode === 'ngram-mtp') && mtpOpt && (mtpOpt.disabled || mtpOpt.hidden)) {
        speculativeMode = speculativeMode === 'ngram-mtp' ? 'ngram' : 'off';
        if (modeSel) modeSel.value = speculativeMode;
      }
      return {
        contextLength: Number($('contextLength').value),
        gpuOffload: Number($('gpuOffload').value),
        cpuThreads: Number($('cpuThreads').value),
        evalBatchSize: Number($('evalBatchSize').value),
        physicalBatchSize: Number($('physicalBatchSize').value),
        maxConcurrentPredictions: Number($('maxConcurrentPredictions').value),
        nCpuMoe: Number($('nCpuMoe').value),
        nCpuFfn: Number($('nCpuFfn').value),
        offloadKvCacheToGpu: $('offloadKvCacheToGpu').checked,
        mmprojOffloadToGpu: $('mmprojOffloadToGpu') ? $('mmprojOffloadToGpu').checked : true,
        cacheTypeK: $('cacheTypeK').value || 'q8_0',
        cacheTypeV: ($('kvTypesLinked').checked ? $('cacheTypeK').value : $('cacheTypeV').value) || 'q8_0',
        keepModelInMemory: $('keepModelInMemory').checked,
        tryMmap: $('tryMmap').checked,
        lazyMode: $('lazyMode') ? $('lazyMode').value || 'auto' : 'auto',
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
        ngramVariant: ($('ngramVariant') && $('ngramVariant').value) || 'simple',
        ngramSizeN: Number(($('ngramSizeN') && $('ngramSizeN').value) || 12),
        ngramSizeM: Number(($('ngramSizeM') && $('ngramSizeM').value) || 48),
        ngramMinHits: Number(($('ngramMinHits') && $('ngramMinHits').value) || 1),
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
        minP: $('minP') ? Number($('minP').value) : 0,
        repeatPenalty: $('repeatPenalty') ? Number($('repeatPenalty').value) : 1,
        presencePenalty: $('presencePenalty') ? Number($('presencePenalty').value) : 0,
        frequencyPenalty: $('frequencyPenalty') ? Number($('frequencyPenalty').value) : 0,
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
      $('contextLengthRange').step = '1';
      // Slider max = actual layer count (legacy 99/"all" is shown as all layers).
      $('gpuOffload').max = String(modelBlockCount);
      $('gpuOffloadRange').max = String(modelBlockCount);
      $('nCpuMoe').max = String(modelBlockCount);
      $('nCpuMoeRange').max = String(modelBlockCount);
      $('nCpuFfn').max = String(modelBlockCount);
      $('nCpuFfnRange').max = String(modelBlockCount);

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
        $('ffnHint').textContent = isMoe
          ? 'MoE model — expert weights are handled by CPU MoE layers (--n-cpu-moe).'
          : ('Dense model. Layers to force dense FFN weights onto CPU (0–' + blocks + ').');
        if (!isMoe) {
          ffnHintDefault = $('ffnHint').textContent;
        }
        $('modelCaps').classList.remove('hidden');
        $('modelCaps').innerHTML =
          'Architecture: <strong>' + (caps.architecture || '?') + '</strong><br/>' +
          'Max context: <strong>' + maxCtx + '</strong> · Layers: <strong>' + blocks + '</strong>' +
          (isMoe ? (' · MoE experts: <strong>' + (caps.expertCount || '?') + '</strong>') : ' · Dense (non-MoE)') +
          (caps.fullAttentionInterval > 1
            ? (' · hybrid full-attn every <strong>' + caps.fullAttentionInterval + '</strong> layers')
            : '') +
          (caps.pleShare > 0
            ? (' · PLE table ~<strong>' + Math.round(caps.pleShare * 100) + '%</strong>')
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
      const ngramMtpOpt = $('specNgramMtpOption');
      if (mtpOpt) {
        mtpOpt.disabled = !mtpCapable;
        mtpOpt.hidden = !mtpCapable;
      }
      if (ngramMtpOpt) {
        ngramMtpOpt.disabled = !mtpCapable;
        ngramMtpOpt.hidden = !mtpCapable;
      }
      if (modeSel) {
        if (!mtpCapable && modeSel.value === 'mtp') {
          modeSel.value = 'off';
        } else if (!mtpCapable && modeSel.value === 'ngram-mtp') {
          modeSel.value = 'ngram';
        }
      }
      const mode = modeSel ? modeSel.value : 'off';
      const isMtp = mode === 'mtp' || mode === 'ngram-mtp';
      const isDflash = mode === 'dflash' || mode === 'ngram-dflash';
      const isNgram = mode === 'ngram' || mode === 'ngram-mtp' || mode === 'ngram-dflash';
      const showMtpKnobs = isMtp && mtpCapable;
      const showDraftKnobs = isMtp || isDflash;
      const showDraftPicker = isDflash || (isMtp && sidecarMtp && !bakedMtp);
      const ngramGroup = $('specNgramGroup');
      if (ngramGroup) ngramGroup.classList.toggle('hidden', !isNgram);
      const neuralGroup = $('specNeuralGroup');
      if (neuralGroup) neuralGroup.classList.toggle('hidden', !showDraftKnobs);
      const neuralHead = $('specNeuralHeading');
      if (neuralHead) neuralHead.textContent = isMtp ? 'MTP' : 'DFlash';
      const maxName = $('specDraftMaxName');
      if (maxName) maxName.textContent = isMtp ? 'MTP max draft tokens' : 'DFlash max draft tokens';
      const minName = $('specDraftMinName');
      if (minName) minName.textContent = 'MTP min draft tokens';
      const pName = $('specDraftPName');
      if (pName) pName.textContent = isMtp ? 'MTP draft probability' : 'DFlash draft probability';
      const modelName = $('specDraftModelName');
      if (modelName) modelName.textContent = isMtp ? 'MTP draft model' : 'DFlash draft model';
      const nglName = $('specDraftNglName');
      if (nglName) nglName.textContent = isMtp ? 'MTP GPU offload' : 'DFlash GPU offload';

      for (const id of ['maxDraftTokens']) {
        const el = $(id);
        if (el) el.disabled = !showDraftKnobs;
      }
      for (const id of ['minDraftTokens']) {
        const el = $(id);
        if (el) el.disabled = !showMtpKnobs;
      }
      for (const id of ['draftProbability']) {
        const el = $(id);
        if (el) el.disabled = !showDraftKnobs;
      }
      for (const id of ['specDraftMaxRow', 'specDraftPRow']) {
        const row = $(id);
        if (row) {
          row.style.opacity = showDraftKnobs ? '1' : '0.55';
          row.classList.toggle('hidden', !showDraftKnobs);
        }
      }
      for (const id of ['specDraftMinRow']) {
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
      // N-gram knobs stay visible in stacked modes (ngram-mtp / ngram-dflash) too.
      const ngramVariant = $('ngramVariant') ? $('ngramVariant').value : 'simple';
      for (const id of ['specNgramVariantRow', 'specNgramSizeRow', 'specNgramDraftRow']) {
        const row = $(id);
        if (row) row.classList.toggle('hidden', !isNgram);
      }
      const hitsRow = $('specNgramHitsRow');
      if (hitsRow) hitsRow.classList.toggle('hidden', !isNgram || ngramVariant === 'mod');

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
        if (mode === 'ngram-mtp') {
          hint.textContent =
            'Stacks --spec-type ngram-<variant> with draft-mtp. N-gram drafts from prompt repeats; MTP fills the rest. Needs an MTP-capable GGUF (or sidecar mtp-*.gguf).';
        } else if (mode === 'ngram-dflash') {
          hint.textContent =
            'Stacks --spec-type ngram-<variant> with draft-dflash. N-gram is a cheap lookup; DFlash drafts when there is no n-gram hit. Needs a DFlash draft GGUF.';
        } else if (isNgram) {
          hint.textContent =
            'N-gram passes --spec-type ngram-<variant> with lookup/draft sizes. Drafts come from n-grams in the prompt itself — great for code (edits repeat the prompt) and the only speculative mode that needs no draft model.';
        } else if (isDflash) {
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
            'This GGUF has no MTP / nextn_predict_layers and no sibling mtp-*.gguf — MTP is unavailable. Use DFlash with a separate draft GGUF, N-gram (no draft model), or an MTP-tagged main model.';
        }
      }
    }

    function fieldFocused(id) {
      const el = $(id);
      return !!(el && document.activeElement === el);
    }
    /** Leave the control the user is editing alone. Background pushes must not revert it. */
    function setField(id, value) {
      const el = $(id);
      if (!el || document.activeElement === el) return;
      const next = value == null ? '' : String(value);
      if (el.value !== next) el.value = next;
    }
    function setPair(id, rangeId, value) {
      if (fieldFocused(id) || fieldFocused(rangeId)) return;
      setField(id, value);
      setField(rangeId, value);
    }
    function setChecked(id, on) {
      const el = $(id);
      if (!el || document.activeElement === el) return;
      el.checked = !!on;
    }
    function setDisabled(id, disabled) {
      const el = $(id);
      if (!el || document.activeElement === el) return;
      el.disabled = !!disabled;
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
      const generating = !!(payload.perf && payload.perf.generating);
      const httpUp = !!(status.httpReady || status.running || generating);
      const starting = !httpUp && !!(status.starting || payload.starting);
      const ready = httpUp;
      const endpoint = payload.endpoint || status.endpoint || lastEndpoint;
      const startMessage = status.startMessage || status.message || '';
      lastEndpoint = endpoint;
      if (status.pid) lastPid = status.pid;

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
      if (lm && payload.launchMode && document.activeElement !== lm) {
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

      setChecked('promptReplacementsEnabled', !!payload.promptReplacementsEnabled);
      setChecked('wikipediaLookupEnabled', !!payload.wikipediaLookupEnabled);
      setChecked('duplicateToolCallGuardEnabled', !!payload.duplicateToolCallGuardEnabled);
      const prStats = $('replacementStats');
      if (prStats) {
        const pr = perf.promptReplacements;
        const hist = Array.isArray(perf.history) ? perf.history : [];
        const sessionSaved = hist.reduce(function (sum, h) {
          return sum + (typeof h.tokensSaved === 'number' ? h.tokensSaved : 0);
        }, 0);
        const sessionTitle = sessionSaved > 0
          ? 'Session: ≈' + Number(sessionSaved).toLocaleString() + ' tok across ' + hist.length + ' call' + (hist.length === 1 ? '' : 's')
          : '';
        if (!payload.promptReplacementsEnabled) {
          prStats.textContent = sessionSaved > 0 ? '−' + fmtTokK(sessionSaved) + ' session' : 'off';
          prStats.title = 'Prompt replacements are off' + (sessionTitle ? '. ' + sessionTitle : '');
        } else if (!pr) {
          prStats.textContent = sessionSaved > 0 ? '−' + fmtTokK(sessionSaved) + ' session' : '—';
          prStats.title = (sessionTitle ? sessionTitle + '. ' : '') + 'Send a chat to measure';
        } else if (!pr.enabled) {
          prStats.textContent = 'off last call';
          prStats.title = (sessionTitle ? sessionTitle + '. ' : '') + 'Last call: replacements were off';
        } else if (pr.tokensSaved > 0) {
          prStats.textContent = '−' + fmtTokK(pr.tokensSaved) + ' last call';
          prStats.title = (sessionTitle ? sessionTitle + '. ' : '') +
            'Last call: saved ≈' + Number(pr.tokensSaved).toLocaleString() +
            ' tokens (' + pr.pctSaved + '% of request)';
        } else {
          prStats.textContent = sessionSaved > 0 ? '−' + fmtTokK(sessionSaved) + ' session' : 'no match';
          prStats.title = (sessionTitle ? sessionTitle + '. ' : '') +
            'Last call: no matching boilerplate (' +
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
      const backendFocused = document.activeElement === sel;
      suppressBackendChange = true;
      if (!backendFocused) {
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
      const ffn = (caps && caps.isMoe) ? 0 : Math.min(L.nCpuFfn ?? 0, blocks);
      const threads = Math.min(Math.max(1, L.cpuThreads || 1), cpuLogicalCores);

      setPair('contextLength', 'contextLengthRange', ctx);
      setPair('gpuOffload', 'gpuOffloadRange', ngl);
      setPair('cpuThreads', 'cpuThreadsRange', threads);
      setField('evalBatchSize', L.evalBatchSize);
      setField('physicalBatchSize', L.physicalBatchSize);
      setField('maxConcurrentPredictions', L.maxConcurrentPredictions);
      setPair('nCpuMoe', 'nCpuMoeRange', moe);
      setPair('nCpuFfn', 'nCpuFfnRange', ffn);
      setChecked('offloadKvCacheToGpu', !!L.offloadKvCacheToGpu);
      setChecked('mmprojOffloadToGpu', L.mmprojOffloadToGpu !== false);
      if (!fieldFocused('cacheTypeK') && !fieldFocused('cacheTypeV') && !fieldFocused('kvTypesLinked')) {
        setField('cacheTypeK', L.cacheTypeK || 'q8_0');
        setField('cacheTypeV', L.cacheTypeV || 'q8_0');
        setChecked('kvTypesLinked', (L.cacheTypeK || 'q8_0') === (L.cacheTypeV || 'q8_0'));
        syncKvLink(false);
      }
      setChecked('keepModelInMemory', !!L.keepModelInMemory);
      if (payload.isWindows) {
        const label = $('keepModelLabel');
        if (label) label.textContent = 'Keep Model in Memory (mmap on Windows)';
        const hint = $('keepModelHint');
        if (hint) hint.style.display = 'block';
      }
      setChecked('tryMmap', !!L.tryMmap);
      setField('lazyMode', L.lazyMode || 'auto');
      setChecked('unifiedKvCache', !!L.unifiedKvCache);
      setField('flashAttention', L.flashAttention || 'auto');
      setField('contextCheckpoints', L.contextCheckpoints);
      setField('cacheReuse', L.cacheReuse ?? 0);
      setField('reasoningFormat', L.reasoningFormat || 'deepseek-legacy');
      const budget = L.reasoningBudget ?? -1;
      const budgetFocused = fieldFocused('reasoningBudget') || fieldFocused('reasoningBudgetUnlimited');
      if (!budgetFocused) {
        setChecked('reasoningBudgetUnlimited', budget < 0);
        setField('reasoningBudget', budget < 0 ? 2048 : budget);
        setDisabled('reasoningBudget', budget < 0);
      }
      const ropeBaseFocused = fieldFocused('ropeFreqBase') || fieldFocused('ropeBaseAuto');
      if (!ropeBaseFocused) {
        setChecked('ropeBaseAuto', L.ropeFreqBase == null);
        setField('ropeFreqBase', L.ropeFreqBase ?? 10000);
        setDisabled('ropeFreqBase', L.ropeFreqBase == null);
      }
      const ropeScaleFocused = fieldFocused('ropeFreqScale') || fieldFocused('ropeScaleAuto');
      if (!ropeScaleFocused) {
        setChecked('ropeScaleAuto', L.ropeFreqScale == null);
        setField('ropeFreqScale', L.ropeFreqScale ?? 1);
        setDisabled('ropeFreqScale', L.ropeFreqScale == null);
      }
      const seedFocused = fieldFocused('seed') || fieldFocused('seedRandom');
      if (!seedFocused) {
        setChecked('seedRandom', L.seed == null);
        setField('seed', L.seed ?? 0);
        setDisabled('seed', L.seed == null);
      }
      setField('speculativeMode', L.speculativeMode || 'off');
      setField('maxDraftTokens', L.maxDraftTokens);
      setField('minDraftTokens', L.minDraftTokens);
      setField('draftProbability', L.draftProbability);
      setField('ngramVariant', L.ngramVariant || 'simple');
      if (!fieldFocused('ngramSizeN') && !fieldFocused('ngramSizeNRange')) {
        setField('ngramSizeN', L.ngramSizeN ?? 12);
        setField('ngramSizeNRange', Math.min(64, L.ngramSizeN ?? 12));
      }
      setField('ngramSizeM', L.ngramSizeM ?? 48);
      setField('ngramMinHits', L.ngramMinHits ?? 1);
      setField('draftGpuOffload', L.draftGpuOffload ?? 99);
      setField('splitMode', L.splitMode || 'layer');
      setDraftModelHint(L.draftModelPath || '');
      setMmprojHint(L.mmprojPath || '');
      applySpecUi(!!(caps && caps.nextnPredictLayers > 0), sidecarMtpAvailable());
      setField('temperature', R.temperature);
      setField('topP', R.topP);
      setField('topK', R.topK);
      setField('maxTokens', R.maxTokens);
      setField('minP', R.minP ?? 0);
      setField('repeatPenalty', R.repeatPenalty ?? 1);
      setField('presencePenalty', R.presencePenalty ?? 0);
      setField('frequencyPenalty', R.frequencyPenalty ?? 0);
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
    function recommendedMaxDraftTokens(mode, current, sidecarMtp) {
      const mtpDefault = sidecarMtp ? 4 : 2;
      const n = Number(current) || 0;
      const dflash = mode === 'dflash' || mode === 'ngram-dflash';
      const mtp = mode === 'mtp' || mode === 'ngram-mtp';
      if (dflash) return n <= 4 ? 15 : n;
      if (mtp) {
        if (n <= 0 || n >= 8) return mtpDefault;
        if (sidecarMtp && n === 2) return 4;
        return n;
      }
      return n;
    }
    function recommendedNgramSizes(variant, n, m) {
      const simpleDefault = n === 12 && m === 48;
      const modDefault = n === 24 && m === 64;
      if (variant === 'mod') return simpleDefault ? { n: 24, m: 64 } : { n: n, m: m };
      return modDefault ? { n: 12, m: 48 } : { n: n, m: m };
    }
    const speculativeModeEl = $('speculativeMode');
    if (speculativeModeEl) {
      speculativeModeEl.addEventListener('change', () => {
        const maxEl = $('maxDraftTokens');
        if (maxEl) {
          maxEl.value = String(recommendedMaxDraftTokens(
            speculativeModeEl.value,
            Number(maxEl.value),
            sidecarMtpAvailable()
          ));
        }
        applySpecUi(!!(memInputs && memInputs.nextnPredictLayers > 0), sidecarMtpAvailable());
      });
    }
    const ngramVariantEl = $('ngramVariant');
    if (ngramVariantEl) {
      ngramVariantEl.addEventListener('change', () => {
        const nEl = $('ngramSizeN');
        const mEl = $('ngramSizeM');
        const rangeEl = $('ngramSizeNRange');
        if (nEl && mEl) {
          const next = recommendedNgramSizes(ngramVariantEl.value, Number(nEl.value), Number(mEl.value));
          nEl.value = String(next.n);
          mEl.value = String(next.m);
          if (rangeEl) rangeEl.value = String(Math.min(64, next.n));
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
    const showDownloadsBtn = $('showDownloadsBtn');
    if (showDownloadsBtn) {
      showDownloadsBtn.addEventListener('click', () => vscode.postMessage({ type: 'showDownloads' }));
    }
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
    const wikiToggle = $('wikipediaLookupEnabled');
    if (wikiToggle) {
      wikiToggle.addEventListener('change', () => {
        vscode.postMessage({
          type: 'setWikipediaLookupEnabled',
          payload: { enabled: !!wikiToggle.checked },
        });
      });
    }
    const dupToggle = $('duplicateToolCallGuardEnabled');
    if (dupToggle) {
      dupToggle.addEventListener('change', () => {
        vscode.postMessage({
          type: 'setDuplicateToolCallGuardEnabled',
          payload: { enabled: !!dupToggle.checked },
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
        vscode.postMessage({
          type: 'reload',
          payload: readLoad(),
        });
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
        vscode.postMessage({
          type: 'start',
          payload: readLoad(),
        });
        vscode.postMessage({ type: 'saveRequest', payload: readRequest() });
      }
    });
    $('stopBtn').addEventListener('click', () => vscode.postMessage({ type: 'stop' }));
    $('openLogBtn').addEventListener('click', () => vscode.postMessage({ type: 'openLog' }));
    $('copyCmdBtn').addEventListener('click', () => vscode.postMessage({ type: 'copyCommandLine' }));
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
        if (perf.generating && serverStarting) {
          serverStarting = false;
          serverRunning = true;
          updatePrimaryAction();
          renderStatusUi({
            ready: true,
            dirty: configDirty,
            starting: false,
            endpoint: lastEndpoint,
            pid: lastPid,
            message: '',
          });
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
        const p = msg.payload;
        const httpUp = !!(p.running || p.httpReady);
        const starting = !!p.starting && !httpUp;
        if (p.endpoint) lastEndpoint = p.endpoint;
        if (p.pid) lastPid = p.pid;
        serverStarting = starting;
        serverRunning = httpUp;
        configDirty = starting ? false : !!p.configDirty;
        updatePrimaryAction();
        renderStatusUi({
          ready: httpUp,
          dirty: configDirty,
          starting,
          endpoint: lastEndpoint,
          pid: lastPid,
          message: starting ? (p.startMessage || p.message || 'Starting…') : (p.message || ''),
        });
        if (p.perf || Array.isArray(p.perfLines)) {
          renderPerfStats(p.perf || {}, p.perfLines || []);
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

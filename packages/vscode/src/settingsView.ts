import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";
import { promptUseInCopilotChat } from "./copilotChatPrompt";
import { copyServerCommandLine, openServerLog, reportLaunchFailure } from "./serverDiagnostics";
import { detectGpus, activeInstallLock, type GpuMemoryInfo } from "@llama-aio/core";
import { LlamaInstaller, UiBackend } from "@llama-aio/core";
import { computeMemoryView, diffLoadSettings, fittingContextLength, memoryEstimateInputs, mmprojFileSize, resolveDraftCapabilities, shortGpuName, type SettingChange } from "@llama-aio/core";
import { resolveModelModes } from "@llama-aio/core";
import {
  displayModelTitle,
  friendlyModelTitle,
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
  /** GPUs from the last full state push; live estimates reuse them instead of re-probing. */
  private lastGpus: GpuMemoryInfo[] = [];

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
          case "estimate": {
            // Live estimate for unsaved form values; the webview only renders it.
            const draft = {
              ...this.store.getState().loadSettings,
              ...((msg.payload || {}) as Partial<LlamaLoadSettings>),
            };
            const cpuOnly = !!msg.cpuOnly;
            const gpus = cpuOnly ? [] : this.lastGpus;
            const { view } = computeMemoryView(this.store.getState().modelCapabilities, draft, gpus, {
              cpuOnly,
              withFixes: true,
            });
            this.view?.webview.postMessage({ type: "memoryEstimate", seq: msg.seq, view: view ?? null });
            break;
          }
          case "applyLoadPatch": {
            const patch = { ...((msg.payload || {}) as Partial<LlamaLoadSettings>) };
            if (msg.fitContext) {
              const caps = this.store.getState().modelCapabilities;
              if (caps) {
                patch.contextLength = fittingContextLength(
                  caps,
                  { ...this.store.getState().loadSettings, ...patch },
                  { cpuOnly: !!msg.cpuOnly, gpus: msg.cpuOnly ? [] : this.lastGpus }
                );
              }
            }
            await this.store.updateLoadSettings(patch);
            this.syncSpeculativeMode();
            await this.pushState();
            break;
          }
          case "discardChanges":
            await this.discardUnappliedChanges();
            await this.pushState();
            break;
          case "changeModel":
            await this.modelActions.pickDownloaded();
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

  /**
   * The one memory path for the sidebar: core estimate + fit verdict for the
   * saved settings. Spill confirmation, the state push and live edits all go
   * through `computeMemoryView`, so the bars and the dialog cannot disagree.
   */
  private memoryForSaved(cpuOnly: boolean) {
    const state = this.store.getState();
    const gpus = this.detectGpusForEstimate(cpuOnly);
    this.lastGpus = gpus;
    return {
      gpus,
      ...computeMemoryView(state.modelCapabilities, state.loadSettings, gpus, {
        cpuOnly,
        draftCaps: resolveDraftCapabilities(state.loadSettings),
        withFixes: true,
      }),
    };
  }

  private cpuOnlyBackend(): boolean {
    return this.installer.resolveActiveUiBackend() === "cpu" || this.processManager.isCpuBackend();
  }

  /** Model / load settings / launch mode edits that the running server has not picked up. */
  private pendingChanges(): SettingChange[] {
    const launched = this.processManager.getLaunchedConfig();
    if (!launched) {
      return [];
    }
    const state = this.store.getState();
    const changes = diffLoadSettings(launched.loadSettings, state.loadSettings);
    if (launched.modelPath && state.selectedModelPath && path.resolve(launched.modelPath) !== path.resolve(state.selectedModelPath)) {
      changes.unshift({
        key: "model",
        label: "model",
        from: path.basename(launched.modelPath),
        to: path.basename(state.selectedModelPath),
      });
    }
    const mode = resolveLaunchMode(this.store.getConfig().get<string>("launchMode"));
    if (launched.launchMode && launched.launchMode !== mode) {
      changes.push({ key: "launchMode", label: "launch mode", from: launched.launchMode, to: mode });
    }
    return changes;
  }

  /** Put the sidebar back to what the running server was started with. */
  private async discardUnappliedChanges(): Promise<void> {
    const launched = this.processManager.getLaunchedConfig();
    if (!launched) {
      return;
    }
    const state = this.store.getState();
    if (launched.modelPath && launched.modelPath !== state.selectedModelPath) {
      await this.store.applySelectedModel(launched.modelPath);
    }
    if (launched.loadSettings) {
      await this.store.updateLoadSettings(launched.loadSettings);
    }
    const mode = resolveLaunchMode(this.store.getConfig().get<string>("launchMode"));
    if (launched.launchMode && launched.launchMode !== mode) {
      await this.store.getConfig().update("launchMode", launched.launchMode);
    }
    this.syncSpeculativeMode();
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
    const { estimate: est, view } = this.memoryForSaved(this.cpuOnlyBackend());
    if (!est?.willSpill) {
      return true;
    }
    const warning =
      view?.headline ||
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
    const { gpus, view: memoryView } = this.memoryForSaved(cpuOnly);
    const draftCaps = resolveDraftCapabilities(state.loadSettings);
    const launched = this.processManager.getLaunchedConfig();
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
        modelName: friendlyModelTitle(caps?.name, state.selectedModelPath),
        modelNameRaw: displayModelTitle(caps?.name, state.selectedModelPath),
        changes: this.pendingChanges(),
        startedAt: launched?.startedAt,
        launchedSettings: launched?.loadSettings ?? null,
        defaults: DEFAULT_LOAD_SETTINGS,
        build,
        backendOptions,
        selectedUiBackend,
        cpuOnly,
        launchMode: resolveLaunchMode(this.store.getConfig().get<string>("launchMode")),
        updateCheck,
        memoryView: memoryView ?? null,
        modeSampling: describeModeSampling(caps, state.selectedModelPath),
        memInputs: memoryEstimateInputs(
          caps,
          draftCaps,
          mmprojFileSize(state.loadSettings.mmprojPath)
        ),
        cpuCount: Math.max(1, os.cpus().length || 1),
        isWindows: process.platform === "win32",
        gpus: gpus.map((g, i) => ({
          totalBytes: g.totalBytes,
          usedBytes: g.usedBytes,
          name: g.name,
          shortName: shortGpuName(g, g.index ?? i),
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
        changes: !ui.starting && status.configDirty ? this.pendingChanges() : [],
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
      margin: 2px 0 4px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.5;
    }
    /* ---- Header ---- */
    .sticky-head {
      position: sticky;
      top: 0;
      z-index: 30;
      box-shadow: 0 3px 8px rgba(0, 0, 0, 0.25);
    }
    .srv-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }
    .srv-top .status-line { flex-wrap: wrap; margin: 0; }
    .srv-top .meta-row { margin: 0; font-weight: 400; }
    .srv-icons { display: flex; gap: 4px; align-items: center; flex-shrink: 0; }
    .icon-btn {
      width: 24px;
      height: 24px;
      padding: 0;
      border-radius: 4px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: var(--secondary);
      color: var(--secondary-fg);
      font-size: 12px;
      line-height: 1;
      cursor: pointer;
      list-style: none;
    }
    .icon-btn::-webkit-details-marker { display: none; }
    .menu { position: relative; }
    .menu-body {
      position: absolute;
      right: 0;
      top: 28px;
      z-index: 50;
      min-width: 210px;
      padding: 4px 0 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--vscode-editorWidget-background, var(--input-bg));
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
    }
    .menu-item {
      display: block;
      width: 100%;
      background: transparent;
      color: var(--fg);
      font-weight: 500;
      padding: 6px 12px;
      border-radius: 0;
    }
    .menu-item:hover { background: color-mix(in srgb, var(--fg) 10%, transparent); }
    .menu-sep { border-top: 1px solid var(--border); margin: 4px 0 6px; }
    .menu-label { display: block; font-size: 11px; color: var(--muted); padding: 0 12px 4px; }
    .menu-body select { margin: 0 12px; width: calc(100% - 24px); }
    .srv-model { font-weight: 650; margin-top: 6px; overflow-wrap: anywhere; }
    .srv-sub { color: var(--muted); font-size: 11px; margin-top: 1px; }
    /* ---- Cards & folds ---- */
    .card-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 6px; }
    .card-title { font-weight: 650; font-size: 12.5px; }
    .card-head .sub { color: var(--muted); font-size: 11px; font-weight: 400; }
    button.small { padding: 3px 10px; font-size: 11px; border-radius: 4px; }
    details.fold { border-top: 1px solid var(--border); margin-top: 8px; }
    details.fold.top { border-top: 0; margin-top: 0; }
    details.fold > summary {
      list-style: none;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
      padding: 7px 0 5px;
      cursor: pointer;
      user-select: none;
      font-weight: 600;
    }
    details.fold > summary::-webkit-details-marker { display: none; }
    details.fold > summary > span:first-child::before { content: '▸'; color: var(--muted); display: inline-block; width: 1.1em; }
    details.fold[open] > summary > span:first-child::before { content: '▾'; }
    details.fold > summary .sum {
      font-weight: 400;
      font-size: 11px;
      color: var(--muted);
      text-align: right;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      min-width: 0;
    }
    .fold-body { padding: 2px 0 6px; }
    .btn-row { display: flex; gap: 6px; flex-wrap: wrap; }
    .btn-row button { flex: 1; text-align: center; padding: 6px 8px; font-size: 11px; }
    .opt-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; padding: 4px 0; font-size: 11px; }
    .opt-row .hint { font-size: 10px; margin: 0; }
    .section-sep { border-top: 1px solid var(--border); margin: 10px 0 10px; }
    /* ---- Memory verdict ---- */
    .verdict { border-radius: 5px; padding: 6px 9px; font-size: 12px; margin: 2px 0 8px; line-height: 1.4; border: 1px solid var(--border); }
    .verdict.good { border-color: color-mix(in srgb, var(--ok) 50%, var(--border)); background: color-mix(in srgb, var(--ok) 13%, transparent); }
    .verdict.tight { border-color: color-mix(in srgb, var(--warn) 55%, var(--border)); background: color-mix(in srgb, var(--warn) 14%, transparent); }
    .verdict.spill { border-color: color-mix(in srgb, var(--bad) 55%, var(--border)); background: color-mix(in srgb, var(--bad) 15%, transparent); }
    .verdict.unknown { color: var(--muted); }
    .mem-dev { margin: 6px 0; }
    .mem-stack { position: relative; overflow: visible; }
    .mem-stack > span:first-child { border-radius: 3px 0 0 3px; }
    .mem-stack .target {
      position: absolute;
      top: -3px;
      bottom: -3px;
      width: 2px;
      margin-left: -1px;
      background: var(--fg);
      opacity: 0.75;
      border-radius: 1px;
    }
    .fix-row { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; margin: 6px 0 2px; }
    .fix-row .hint { margin: 0 2px 0 0; }
    .chip.fix { border-color: color-mix(in srgb, var(--ok) 50%, var(--border)); background: color-mix(in srgb, var(--ok) 12%, transparent); }
    .chip.fix.tight { border-color: color-mix(in srgb, var(--warn) 50%, var(--border)); background: color-mix(in srgb, var(--warn) 10%, transparent); }
    #memNotes > div, #memLines > div { margin: 2px 0; }
    #memNotes { margin-bottom: 6px; color: var(--fg); }
    /* ---- Load controls ---- */
    .chip.custom { cursor: default; opacity: 0.6; }
    .chip.custom.active { opacity: 1; }
    .num-pair { display: inline-flex; align-items: center; gap: 6px; }
    .num-pair .sub { color: var(--muted); font-size: 11px; }
    .ticks { position: relative; height: 12px; margin: -2px 7px 2px; font-size: 9.5px; color: var(--muted); }
    .ticks span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
    .seg { display: inline-flex; border: 1px solid var(--border); border-radius: 5px; overflow: hidden; }
    .seg-btn {
      background: transparent;
      color: var(--fg);
      font-weight: 500;
      font-size: 11px;
      padding: 2px 9px;
      border-radius: 0;
      text-align: center;
    }
    .seg-btn + .seg-btn { border-left: 1px solid var(--border); }
    .seg-btn.active { background: color-mix(in srgb, var(--accent) 35%, transparent); color: var(--fg); }
    /* ---- Performance ---- */
    .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 5px; margin: 2px 0 8px; }
    .kpi { border: 1px solid var(--border); border-radius: 5px; padding: 4px 6px; background: color-mix(in srgb, var(--fg) 3%, transparent); }
    .kpi b { display: block; font-size: 14px; font-variant-numeric: tabular-nums; }
    .kpi b.muted { color: var(--muted); }
    .kpi i { font-style: normal; font-size: 10px; color: var(--muted); white-space: nowrap; }
    /* ---- Advanced ---- */
    .adv-tools { display: flex; gap: 8px; align-items: center; margin: 2px 0 6px; }
    .adv-tools input[type="text"] { flex: 1; width: auto; padding: 4px 6px; }
    .adv-tools .auto { display: inline-flex; gap: 4px; align-items: center; font-size: 11px; color: var(--muted); white-space: nowrap; }
    .row.changed > .label > .name::before,
    .toggle.changed > span:first-child::before { content: '●'; color: var(--warn); font-size: 9px; margin-right: 4px; vertical-align: 1px; }
    .filtered-out { display: none !important; }
  </style>
</head>
<body>
  <div class="card server stopped sticky-head" id="serverCard">
    <div class="srv-top">
      <div class="status-line stopped" id="statusLine">
        <span class="pill">
          <span class="dot stopped" id="statusDot"></span>
          <span id="statusText">Loading…</span>
        </span>
        <span class="meta-row" id="statusMeta">—</span>
      </div>
      <div class="srv-icons">
        <button class="icon-btn hidden" id="reloadIconBtn" type="button" title="Reload llama-server (restart with the same settings)" aria-label="Reload server">⟳</button>
        <button class="icon-btn" id="stopBtn" type="button" title="Stop llama-server" aria-label="Stop server" disabled>■</button>
        <details class="menu" id="srvMenu">
          <summary class="icon-btn" title="More server actions" aria-label="More server actions">⋯</summary>
          <div class="menu-body">
            <button class="menu-item" id="openLogBtn" type="button">Open log</button>
            <button class="menu-item" id="copyCmdBtn" type="button">Copy command line</button>
            <button class="menu-item" id="openWebUiBtn" type="button">Open llama.cpp web UI</button>
            <div class="menu-sep"></div>
            <label class="menu-label" for="launchMode">Launch mode</label>
            <select id="launchMode" class="wide" title="How llama-server is started">
              <option value="externalTerminal">External terminal (logs visible)</option>
              <option value="background">Background (hidden process)</option>
            </select>
          </div>
        </details>
      </div>
    </div>
    <div class="srv-model" id="srvModel">No model selected</div>
    <div class="srv-sub" id="srvSummary"></div>
    <div class="srv-sub" id="srvPerf"></div>
    <div class="dirty-hint hidden" id="dirtyHint">
      <span class="label">Not applied yet:</span> <span id="dirtyList">load settings or launch mode changed.</span>
    </div>
    <div class="actions-row" id="primaryRow">
      <button class="primary" id="primaryBtn" data-action="start">Start</button>
      <button class="secondary hidden" id="discardBtn" type="button" title="Put the settings back to what the running server uses">Discard</button>
    </div>
  </div>

  <div class="setup hidden" id="setupBox">
    <strong>Get a model first</strong>
    <p class="hint" style="margin:8px 0">One-click starter: Unsloth ${STARTER_MODEL.label} (${STARTER_MODEL.approxSizeLabel}, ${STARTER_MODEL.detail}).</p>
    <div class="btn-col" style="margin-top:4px">
      <button class="primary" id="setupStarterBtn">Download starter (${STARTER_MODEL.label})</button>
    </div>
    <ol>
      <li>Install llama.cpp (once)</li>
      <li>Download the starter <em>or</em> pick/open any GGUF</li>
      <li>Start the server</li>
    </ol>
  </div>

  <div class="card" id="modelCard">
    <div class="card-head">
      <span class="card-title">Model</span>
      <button class="secondary small" id="changeModelBtn" type="button" title="Pick a downloaded GGUF, open a file, or search Hugging Face">Change…</button>
    </div>
    <div class="model-title" id="modelTitle">No model selected</div>
    <div class="caps hidden" id="modelCaps"></div>
    <div class="model-path" id="modelPath"></div>
    <div class="btn-col hidden" id="starterCol">
      <button class="primary hidden" id="starterModelBtn">Download starter (${STARTER_MODEL.label})</button>
      <div class="hint hidden" id="starterModelHint" style="margin-top:0">${STARTER_MODEL.approxSizeLabel} · ${STARTER_MODEL.detail}</div>
    </div>
    <details class="fold" id="visionFold">
      <summary><span class="tip" data-flag="-mm, --mmproj" data-help="Path to a multimodal projector GGUF. llama-server loads it with the language model so Copilot Chat can send images. Auto-attached when a sibling mmproj-*.gguf sits next to the model.">Vision projector</span><span class="sum" id="visionSum">none · text only</span></summary>
      <div class="fold-body">
        <div class="hint" id="mmprojPathHint" style="margin:0 0 8px">No mmproj — text only.</div>
        <div class="btn-row">
          <button class="secondary" id="pickMmprojBtn" type="button">Choose mmproj…</button>
          <button class="secondary" id="clearMmprojBtn" type="button">Clear</button>
        </div>
        <div class="toggle hidden" id="mmprojOffloadRow"><span class="tip" data-flag="--mmproj-offload / --no-mmproj-offload" data-help="Whether to offload the CLIP vision projector to GPU (llama.cpp default: on). Uncheck to pass --no-mmproj-offload and keep the projector in system RAM. Frees VRAM on the Main GPU; image encode becomes CPU-bound.">Offload vision projector to GPU</span><input type="checkbox" id="mmprojOffloadToGpu" checked /></div>
      </div>
    </details>
    <details class="fold" id="libraryFold">
      <summary><span>Library &amp; downloads</span><span class="sum" id="librarySum">—</span></summary>
      <div class="fold-body">
        <div class="meta" id="modelsDirMeta"></div>
        <div class="btn-row">
          <button class="secondary" id="downloadModelBtn" type="button">Hugging Face…</button>
          <button class="secondary" id="openFileBtn" type="button">Open GGUF…</button>
          <button class="secondary" id="showDownloadsBtn" type="button">Downloads</button>
        </div>
      </div>
    </details>
  </div>

  <div class="card" id="memCard">
    <div class="card-head">
      <span class="card-title">Memory &amp; load</span>
      <span class="sub" id="memCtxNote" title="Bars are the estimate at full context. Live use is what the driver reports now.">at full context</span>
    </div>
    <div class="verdict unknown" id="memVerdict">Select a model to estimate VRAM / RAM use.</div>
    <div id="memDevices"></div>
    <div class="mem-legend hidden" id="memLegend"></div>
    <div class="hint" id="memFootnote"></div>
    <div class="fix-row hidden" id="memFixes"></div>
    <details class="fold" id="memDetails">
      <summary><span>Breakdown</span><span class="sum">capacity · weights · KV · notes</span></summary>
      <div class="fold-body">
        <div class="meta" id="memNotes"></div>
        <div class="meta" id="memLines"></div>
      </div>
    </details>

    <div class="section-sep"></div>
    <div class="presets" id="presetChips">
      <button class="chip" id="presetAgent" data-preset="agent" title="Coding agent: q8_0 K + q8_0 V, one slot, 64K context — near-lossless quality with room for tools + history">Coding agent</button>
      <button class="chip" id="presetContext" data-preset="context" title="Max context: q8_0 K + q4_0 V, largest context that still fits your VRAM. K stays at q8_0 because the key cache is far more sensitive to quantization than the value cache.">Max context</button>
      <button class="chip" id="presetQuality" data-preset="quality" title="Max quality: f16 K + q8_0 V at 64K context — spends VRAM on key precision instead of shrinking the context (truncated prompts cost more quality than q8_0 V does).">Max quality</button>
      <span class="chip custom" id="presetCustom" title="Your own mix of context, KV cache types and slots">Custom</span>
    </div>
    <div class="hint" id="presetHint">Sets context length, KV cache types, and slots together.</div>

    <div class="row">
      <div class="label"><span class="name tip" data-flag="-c, --ctx-size" data-help="Size of the prompt context (default: 0 = loaded from model). The slider snaps to 8k, 16k, 32k, 64k, 128k, 256k; type any value in the box.">Context</span><span class="num-pair"><span class="sub" id="ctxK"></span><input type="number" id="contextLength" min="512" step="256" /></span></div>
      <input type="range" id="contextLengthRange" min="0" max="1000" step="1" />
      <div class="ticks" id="ctxTicks"></div>
      <div class="hint" id="ctxHint">Tokens for prompt + generation</div>
    </div>
    <div class="row" id="gpuOffloadRow">
      <div class="label"><span class="name tip" data-flag="-ngl, --n-gpu-layers" data-help="Max number of layers to store in VRAM (exact number, auto, or all).">GPU layers</span>
        <span class="seg" id="nglSeg"><button class="seg-btn" type="button" id="nglAll" data-ngl="all">All</button><button class="seg-btn" type="button" id="nglCustom" data-ngl="custom">Custom</button></span>
      </div>
      <div class="hidden" id="nglCustomRow">
        <div class="label"><span class="hint" style="margin:0">Layers on GPU</span><input type="number" id="gpuOffload" min="0" max="128" /></div>
        <input type="range" id="gpuOffloadRange" min="0" max="128" step="1" />
      </div>
      <div class="hint" id="gpuOffloadHint">Max = all model layers.</div>
    </div>
    <details class="fold" id="offloadFold">
      <summary><span>Multi-GPU &amp; CPU offload</span><span class="sum" id="offloadSum">—</span></summary>
      <div class="fold-body">
        <div class="row hidden" id="dualGpuRow">
          <div class="label"><span class="name tip" data-flag="-sm, --split-mode" data-help="How tensors are split. Layer (default) shares the model across cards. Row needs a fast x16 link. Tensor splits every weight matrix across cards (experimental, fastest for multi-GPU inference). None keeps every GPU layer on the Main GPU and leaves the other cards free (--device).">Split</span>
            <span class="seg" id="splitSeg"><button class="seg-btn" type="button" data-split="none">None</button><button class="seg-btn" type="button" data-split="layer">Layer</button><button class="seg-btn" type="button" data-split="row">Row</button><button class="seg-btn" type="button" data-split="tensor">Tensor</button></span>
          </div>
          <select id="splitMode" class="wide hidden" aria-label="Split mode">
            <option value="layer">Layer (default)</option>
            <option value="row">Row</option>
            <option value="tensor">Tensor (experimental)</option>
            <option value="none">None — Main GPU only</option>
          </select>
          <div class="label" style="margin-top:8px"><span class="name tip" data-flag="-mg, --main-gpu" data-help="GPU that holds the compute graph, scratch buffers, and the slider’s share of weights + KV. Index matches llama.cpp --list-devices (Vulkan0, Vulkan1, …), which is often not PCI / btop order.">Main GPU</span></div>
          <select id="mainGpu" class="wide"></select>
          <div id="splitDetails">
            <div class="label" style="margin-top:8px"><span class="name tip" data-flag="-ts, --tensor-split" data-help="Percent of GPU-resident weights on the Main GPU after CPU MoE/FFN. llama.cpp --tensor-split still fills by layer count, so the emitted fractions can differ (cheap first layers get more of Main). The rest is split evenly across the other cards.">Weights on main GPU</span><span id="tensorSplitPct">75%</span></div>
            <input type="range" id="tensorSplitRange" min="10" max="90" step="1" />
            <div class="hint" id="tensorSplitHint"></div>
          </div>
          <div class="hint" id="dualGpuHint"></div>
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
      </div>
    </details>
  <details class="fold" id="advancedFold">
    <summary><span>Advanced</span><span class="sum" id="advSum">threads, batch, KV, RoPE, speculative</span></summary>
  <div class="fold-body" id="advancedBody">
  <div class="adv-tools"><input type="text" id="advFilter" placeholder="Filter by label or flag (e.g. -ub, rope)" aria-label="Filter advanced settings" /><label class="auto"><input type="checkbox" id="advChangedOnly" /> Changed only</label></div>

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

  </div>
  </details>
  </div>

  <div class="card" id="perfCard">
    <div class="card-head">
      <span class="card-title">Performance</span>
      <span class="sub" id="perfWindow"></span>
    </div>
    <div class="kpis" id="perfKpis">
      <div class="kpi"><b id="kpiGen">—</b><i>gen tok/s</i></div>
      <div class="kpi"><b id="kpiPrompt">—</b><i>prompt tok/s</i></div>
      <div class="kpi"><b id="kpiReuse">—</b><i>cache reuse</i></div>
      <div class="kpi"><b id="kpiSpec">—</b><i id="kpiSpecLabel">draft accept</i></div>
    </div>
    <div class="chart-legend hidden" id="perfChartLegend">
      <span><i class="gen"></i>gen tok/s (left)</span>
      <span><i class="prompt"></i>prompt tok/s (right)</span>
    </div>
    <div class="perf-session hidden" id="perfSession"></div>
    <div class="ctx-chart-title">
      <span id="ctxLabel">Context</span>
      <span class="sub" id="ctxSub">— (send a chat to measure)</span>
    </div>
    <div class="ctx-stack" id="ctxStack" role="img" aria-label="Context"></div>
    <div class="ctx-legend" id="ctxLegend"></div>
    <div class="metric-line" id="perfMetrics">No generation yet</div>
    <details class="fold" id="perfMore">
      <summary><span>History</span><span class="sum" id="perfHistSum">per-request table</span></summary>
      <div class="fold-body perf-history"><div id="perfHistoryTable"></div></div>
    </details>
  </div>

  <div class="card" id="copilotCard">
    <details class="fold top" id="copilotFold">
      <summary><span class="card-title">Copilot Chat</span><span class="sum" id="copilotSum">—</span></summary>
      <div class="fold-body">
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
        </div>
        <div class="btn-row" style="margin:8px 0 4px">
          <button class="secondary" id="viewContextBtn" disabled title="Open the last Copilot → llama.cpp request (messages + tools) in an editor">Last call</button>
          <button class="secondary" id="viewResponseBtn" disabled title="Open the last llama.cpp assistant stream (helps debug empty Chat replies)">Last response</button>
        </div>
  <details class="fold" id="requestFold">
    <summary><span>Request defaults</span><span class="sum">temperature, top-p/k, min-p, penalties, max tokens</span></summary>
  <div class="fold-body">
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
  </div>
  </details>

      </div>
    </details>
  </div>

  <div class="card" id="backendCard">
    <div class="card-head">
      <span><span class="card-title">llama.cpp</span> <span class="sub" id="backendLine">—</span></span>
      <button class="secondary small hidden" id="installLlamaBtn" type="button">Upgrade to latest</button>
    </div>
    <div class="hint" id="backendHint"></div>
    <details class="fold" id="backendFold">
      <summary><span>Backends &amp; install options</span><span class="sum" id="backendSum">—</span></summary>
      <div class="fold-body">
        <select id="backendSelect" class="wide" aria-label="Backend"></select>
        <div class="meta" id="llamaBinaryDetail" style="margin:8px 0 4px"></div>
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
      </div>
    </details>
  </div>


  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const $ = (id) => document.getElementById(id);

    let memInputs = null;
    let gpuInfos = [];
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
    /** Settings the running server has not picked up yet ({ label, from, to }). */
    let pendingChanges = [];
    let lastPayload = null;
    let lastMemoryView = null;
    let loadDefaults = {};
    /** "Custom" GPU layers stays open while the user works with the slider. */
    let nglCustomOpen = false;

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

    /**
     * The big button only appears when there is something to do: Start,
     * Loading…, or Reload to apply. A clean running server gets the small ⟳.
     */
    function updatePrimaryAction() {
      const primary = $('primaryBtn');
      const stop = $('stopBtn');
      const row = $('primaryRow');
      const reloadIcon = $('reloadIconBtn');
      const discard = $('discardBtn');
      if (!primary || !stop) return;
      stop.disabled = !serverRunning && !serverStarting;
      primary.classList.remove('warn-primary');
      primary.classList.add('primary');
      let showRow = true;
      let showDiscard = false;
      if (serverStarting) {
        primary.disabled = true;
        primary.textContent = 'Loading…';
        primary.dataset.action = '';
      } else if (!serverRunning) {
        primary.disabled = false;
        primary.textContent = 'Start';
        primary.dataset.action = 'start';
      } else if (configDirty) {
        primary.disabled = false;
        const n = pendingChanges.length;
        primary.textContent = n ? ('Reload to apply (' + n + ' change' + (n === 1 ? '' : 's') + ')') : 'Reload to apply';
        primary.dataset.action = 'reload';
        primary.classList.remove('primary');
        primary.classList.add('warn-primary');
        showDiscard = n > 0;
      } else {
        showRow = false;
        primary.dataset.action = 'reload';
      }
      if (row) row.classList.toggle('hidden', !showRow);
      if (discard) discard.classList.toggle('hidden', !showDiscard);
      if (reloadIcon) reloadIcon.classList.toggle('hidden', !(serverRunning && !serverStarting && !configDirty));
      renderDirtyList();
    }

    function renderDirtyList() {
      const list = $('dirtyList');
      if (!list) return;
      list.textContent = pendingChanges.length
        ? pendingChanges.map((c) => c.label + ' ' + c.from + ' → ' + c.to).join(' · ')
        : 'load settings or launch mode changed.';
    }

    function fmtTokShort(n) {
      if (!(n > 0)) return '—';
      if (n >= 1024 && n % 1024 === 0) return (n / 1024) + 'k';
      return n >= 1000 ? Math.round(n / 1000) + 'k' : String(n);
    }

    function fmtUptime(iso) {
      const t = Date.parse(iso || '');
      if (!isFinite(t)) return '';
      const min = Math.max(0, Math.round((Date.now() - t) / 60000));
      if (min < 60) return 'up ' + min + ' min';
      const h = Math.floor(min / 60);
      return 'up ' + h + ' h' + (min % 60 ? ' ' + (min % 60) + ' min' : '');
    }

    /** Header: what is loaded, with which settings, and how it is doing. */
    function renderHeader() {
      const p = lastPayload;
      const modelEl = $('srvModel');
      const sumEl = $('srvSummary');
      const perfEl = $('srvPerf');
      if (!p || !modelEl) return;
      const hasModel = !!(p.state && p.state.selectedModelPath);
      modelEl.textContent = hasModel ? (p.modelName || 'model') : 'No model selected';
      modelEl.title = p.modelNameRaw || '';
      // Describe the running server; pending edits are listed separately below.
      const L = (serverRunning && p.launchedSettings) || (p.state && p.state.loadSettings) || {};
      const caps = p.capabilities || {};
      const blocks = caps.blockCount || 0;
      const bits = [];
      if (hasModel) {
        bits.push('ctx ' + fmtTokShort(L.contextLength));
        if (p.cpuOnly) {
          bits.push('CPU backend');
        } else {
          const ngl = L.gpuOffload >= 99 || (blocks && L.gpuOffload >= blocks) ? blocks : L.gpuOffload;
          const mainIdx = clampMainGpu(L.mainGpu, (gpuInfos || []).length || 1);
          const mainG = gpuInfos && gpuInfos[mainIdx];
          const where = (gpuInfos || []).length >= 2 && L.splitMode !== 'none'
            ? gpuInfos.map((g, i) => gpuLabel(g, i)).join(' + ')
            : (mainG ? gpuLabel(mainG, mainIdx) : 'GPU');
          bits.push((blocks ? ngl + '/' + blocks : ngl) + ' layers on ' + where);
        }
        if (L.speculativeMode && L.speculativeMode !== 'off') bits.push(specAcceptLabel(L.speculativeMode));
        const backend = (backendOptionsCache.find((o) => o.id === activeBackendId) || {});
        if (backend.label) bits.push(backend.label + (backend.installedTag ? ' ' + backend.installedTag : ''));
      }
      sumEl.textContent = bits.join(' · ');
      const perfBits = [];
      const perf = p.perf || {};
      const rows = Array.isArray(perf.history) ? perf.history : [];
      const gen = pickNum(perf.genTokPerSec, rows[0] && rows[0].genTokPerSec);
      if (serverRunning && fmtRate(gen)) perfBits.push(fmtRate(gen) + ' tok/s');
      if (serverRunning && p.startedAt) perfBits.push(fmtUptime(p.startedAt));
      if (!configDirty && lastMemoryView && lastMemoryView.devices && lastMemoryView.devices.length) {
        const d = lastMemoryView.devices[0];
        if (d.capacityBytes) perfBits.push(fmtBytes(d.usedBytes) + ' / ' + fmtBytes(d.capacityBytes) + ' ' + (d.kind === 'ram' ? 'RAM' : d.label));
      }
      perfEl.textContent = perfBits.join(' · ');
      perfEl.classList.toggle('hidden', !perfBits.length);
    }

    function markDirtyIfRunning() {
      // Optimistic dirty UI while running — confirmed via silent save + statusPatch.
      if (!serverRunning) return;
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

    function scheduleSaveLoad() {
      if (saveLoadTimer) clearTimeout(saveLoadTimer);
      highlightPreset();
      markDirtyIfRunning();
      refreshAdvancedMarks();
      saveLoadTimer = setTimeout(() => {
        vscode.postMessage({ type: 'saveLoad', payload: readLoad(), silent: true });
      }, 280);
    }

    function updateBackendUi() {
      updateBackendControls();
      renderBackendLine();
    }

    function updateBackendControls() {
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
        hint.textContent = '';
        installBtn.textContent = 'Upgrade to ' + latestTag;
        installBtn.disabled = false;
        installBtn.classList.remove('hidden');
        installBtn.dataset.action = 'upgrade';
        return;
      }

      hint.textContent = '';
      installBtn.classList.add('hidden');
      installBtn.dataset.action = '';
    }

    /** "Vulkan · b11163" once, plus the alternatives as a summary of the fold. */
    function renderBackendLine() {
      const line = $('backendLine');
      const sum = $('backendSum');
      const active = backendOptionsCache.find((o) => o.id === activeBackendId) || backendOptionsCache.find((o) => o.active);
      if (line) {
        const tag = (active && active.installedTag) || updateCheck.installedTag || '';
        const upToDate = !updateCheck.pending && !updateCheck.checkFailed && !updateCheck.updateAvailable && tag;
        line.textContent = active
          ? active.label + (tag ? ' · ' + tag : '') + (upToDate ? ' · ✓ latest' : '')
          : 'not installed';
      }
      if (sum) {
        sum.textContent = backendOptionsCache
          .filter((o) => o.available && o.id !== 'path')
          .map((o) => o.label + (o.id === activeBackendId ? ' ✓' : ''))
          .join(' · ');
      }
      const hint = $('backendHint');
      if (hint) hint.classList.toggle('hidden', !hint.textContent);
    }

    /** One unit style everywhere: GiB with one decimal, MiB below 1 GiB. */
    function fmtBytes(bytes) {
      const GiB = 1024 ** 3, MiB = 1024 ** 2;
      if (!bytes || bytes <= 0) return '0 B';
      if (bytes >= GiB) return (bytes / GiB).toFixed(1) + ' GiB';
      if (bytes >= MiB) return (bytes / MiB).toFixed(0) + ' MiB';
      return Math.round(bytes / 1024) + ' KiB';
    }

    /** Short device name ("RX 9070"); the full name and llama.cpp id live in tooltips. */
    function gpuLabel(gpu, index) {
      if (gpu && gpu.shortName) return String(gpu.shortName);
      const id = (gpu && gpu.llamaDeviceId) ? String(gpu.llamaDeviceId) : ('GPU ' + index);
      return id;
    }
    function gpuFullLabel(gpu, index) {
      const name = (gpu && gpu.name) ? String(gpu.name).trim() : '';
      const id = (gpu && gpu.llamaDeviceId) ? String(gpu.llamaDeviceId) : ('GPU ' + index);
      return name ? (id + ' · ' + name) : id;
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
      if (ts) ts.disabled = cpu || none;
      // Weight share means nothing with split None — hide it instead of greying it out.
      const details = $('splitDetails');
      if (details) details.classList.toggle('hidden', !!(cpu || none));
      syncSplitSeg();
      syncTensorSplitPctLabel();
      renderOffloadSum();
    }

    function syncSplitSeg() {
      const mode = ($('splitMode') && $('splitMode').value) || 'layer';
      document.querySelectorAll('#splitSeg .seg-btn').forEach((b) => {
        b.classList.toggle('active', b.dataset.split === mode);
      });
    }

    /** One-line state of the Multi-GPU & CPU offload fold, visible while collapsed. */
    function renderOffloadSum() {
      const el = $('offloadSum');
      if (!el) return;
      const bits = [];
      const n = (gpuInfos && gpuInfos.length) || 0;
      const mode = ($('splitMode') && $('splitMode').value) || 'layer';
      if (cpuOnlyLive()) {
        bits.push('CPU backend');
      } else if (n >= 2) {
        const main = clampMainGpu(readMainGpuIndex(), n);
        bits.push(mode === 'none'
          ? gpuLabel(gpuInfos[main], main) + ' only'
          : 'split ' + mode + ' · main ' + gpuLabel(gpuInfos[main], main) + ' ' + (($('tensorSplitPct') && $('tensorSplitPct').textContent) || ''));
      } else if (n === 1) {
        bits.push(gpuLabel(gpuInfos[0], 0));
      }
      if (modelIsMoe) bits.push('CPU MoE ' + (Number($('nCpuMoe').value) || 0));
      else bits.push('CPU FFN ' + (Number($('nCpuFfn').value) || 0));
      el.textContent = bits.join(' · ');
    }

    /** GPU layers: "All" hides the slider; "Custom" shows it. */
    function syncNglMode() {
      const blocks = Math.max(1, modelBlockCount || 1);
      const ngl = Number($('gpuOffload').value);
      const all = !nglCustomOpen && (ngl >= blocks || ngl >= 99);
      $('nglAll').classList.toggle('active', all);
      $('nglAll').textContent = 'All (' + blocks + ')';
      $('nglCustom').classList.toggle('active', !all);
      $('nglCustomRow').classList.toggle('hidden', all);
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
      const legend = $('ctxLegend');
      if (legend) legend.textContent = '';
      if (!breakdown || !Array.isArray(breakdown.segments)) {
        return;
      }
      const scale = Math.max(1, breakdown.limitTokens || contextLimit || 1);
      const used = [];
      for (const seg of breakdown.segments) {
        if (!seg || seg.key === 'free' || !(seg.tokens > 0)) continue;
        used.push(seg);
        const el = document.createElement('span');
        el.className = 'seg-' + seg.key;
        el.style.width = Math.max(0.4, (seg.tokens / scale) * 100) + '%';
        const pct = Math.round((seg.tokens / scale) * 1000) / 10;
        el.title = seg.label + ': ≈' + Number(seg.tokens).toLocaleString() + ' tok (' + pct + '% of slot)';
        stack.appendChild(el);
      }
      // Largest first, with counts, so "tools + results eat ⅔" is readable without hovering.
      if (legend) {
        used.sort((a, b) => b.tokens - a.tokens);
        for (const seg of used) {
          const span = document.createElement('span');
          const swatch = document.createElement('i');
          swatch.className = 'seg-' + seg.key;
          span.appendChild(swatch);
          span.appendChild(document.createTextNode((seg.label || seg.key) + ' ' + fmtTokK(seg.tokens)));
          legend.appendChild(span);
        }
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
      // Latest values are in the KPI tiles; end labels here collided with the right axis.
      if (promptPts.length) {
        const last = promptPts[promptPts.length - 1];
        parts.push('<circle cx="' + last.x.toFixed(1) + '" cy="' + last.y.toFixed(1) + '" r="2.3" fill="' + PROMPT + '"/>');
      }
      if (genPts.length) {
        const last = genPts[genPts.length - 1];
        parts.push('<circle cx="' + last.x.toFixed(1) + '" cy="' + last.y.toFixed(1) + '" r="2.5" fill="' + GEN + '"/>');
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

      const setKpi = (id, text, muted) => {
        const el = $(id);
        if (!el) return;
        el.textContent = text;
        el.classList.toggle('muted', !!muted);
      };
      setKpi('kpiGen', fmtRate(genNow) || '—', stale);
      setKpi('kpiPrompt', fmtRate(promptNow) || '—', stale);
      setKpi('kpiReuse', typeof cacheHitPct === 'number' ? Math.round(cacheHitPct) + '%' : '—', stale);
      const specOn = !!(p.speculativeMode && p.speculativeMode !== 'off');
      setKpi('kpiSpec', specOn && typeof draftAcceptancePct === 'number' ? Math.round(draftAcceptancePct) + '%' : (specOn ? '—' : 'off'), stale);
      const specLabel = $('kpiSpecLabel');
      if (specLabel) specLabel.textContent = specOn ? specAcceptLabel(p.speculativeMode) + ' accept' : 'speculative';
      const windowEl = $('perfWindow');
      if (windowEl) windowEl.textContent = rows.length ? 'last ' + rows.length + ' request' + (rows.length === 1 ? '' : 's') : '';
      const histSum = $('perfHistSum');
      if (histSum) histSum.textContent = rows.length ? rows.length + ' request' + (rows.length === 1 ? '' : 's') : 'per-request table';

      const bits = [];
      if (p.generating) {
        const live = fmtRate(p.genTokPerSec);
        bits.push('<span class="ok">● Generating…' + (live ? ' ' + live + ' tok/s' : '') + '</span>');
      }
      if (typeof completionTokens === 'number') {
        const dur = typeof durationMs === 'number' && durationMs > 0 && !p.generating
          ? ' in ' + (durationMs / 1000).toFixed(1) + ' s'
          : '';
        bits.push('<span class="muted">Last reply: ' + fmtNum(completionTokens) + ' tok' + dur + '</span>');
      }
      if (metricsEl) {
        metricsEl.innerHTML = bits.length ? bits.join('') : (rows.length ? '' : 'No generation yet');
        metricsEl.title = Array.isArray(perfLines) ? perfLines.join('\\n') : '';
      }
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
      const projPath = (hint && (hint.dataset.path || '').trim()) || '';
      el.disabled = !!cpuOnly || !projPath;
      // The offload toggle does nothing without a projector, so it only shows with one.
      const row = $('mmprojOffloadRow');
      if (row) row.classList.toggle('hidden', !projPath);
      const sum = $('visionSum');
      if (sum) {
        sum.textContent = projPath
          ? (projPath.split(/[/\\\\]/).pop() + (el.checked && !cpuOnly ? ' · GPU' : ' · RAM'))
          : 'none · text only';
      }
    }

    function applyCpuOnlyUi(cpuOnly) {
      $('gpuOffload').disabled = cpuOnly;
      $('gpuOffloadRange').disabled = cpuOnly;
      $('offloadKvCacheToGpu').disabled = cpuOnly;
      syncMmprojOffloadUi(cpuOnly);
      $('gpuOffloadHint').textContent = cpuOnly
        ? 'CPU backend — GPU layers are ignored; everything runs in system RAM.'
        : '';
      $('gpuOffloadHint').classList.toggle('hidden', !cpuOnly);
      $('nglAll').disabled = cpuOnly;
      $('nglCustom').disabled = cpuOnly;
      $('gpuOffloadRow').style.opacity = cpuOnly ? '0.55' : '1';
      syncNglMode();

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
          hint.textContent = splitModeIsNone()
            ? 'Every GPU layer stays on the Main GPU; the other cards stay free.'
            : 'Pick the faster card as Main GPU, then give it more of the weights.';
          hint.title = gpuInfos.map((g, i) => gpuFullLabel(g, i) + ' · ' + fmtBytes(g.totalBytes)).join('\\n');
        }
      }
      renderOffloadSum();
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

    // Memory: the extension runs core's estimator on the form values and sends
    // back a verdict. The webview has no estimator of its own.
    let estimateSeq = 0;
    let renderedSeq = 0;
    let estimateTimer = null;
    function requestEstimate() {
      if (estimateTimer) clearTimeout(estimateTimer);
      estimateTimer = setTimeout(() => {
        estimateTimer = null;
        estimateSeq += 1;
        vscode.postMessage({ type: 'estimate', seq: estimateSeq, payload: readLoad(), cpuOnly: cpuOnlyLive() });
      }, 90);
    }
    function refreshMemoryLive() { requestEstimate(); }
    function scheduleLiveMemory() { requestEstimate(); }

    const SEG_LABEL = { weights: 'Weights', vision: 'Vision', draft: 'Spec', kv: 'KV', overhead: 'Overhead' };

    function memBar(device, targetFree) {
      const bar = document.createElement('div');
      bar.className = 'mem-stack';
      const scale = Math.max(device.capacityBytes || 0, device.usedBytes || 0) || 1;
      for (const seg of device.segments || []) {
        if (!(seg.bytes > 0)) continue;
        const el = document.createElement('span');
        el.className = 'seg-' + seg.key;
        el.style.width = Math.max(0.5, (seg.bytes / scale) * 100) + '%';
        el.title = seg.label + ': ~' + fmtBytes(seg.bytes);
        bar.appendChild(el);
      }
      if (device.kind !== 'ram' && device.capacityBytes > targetFree && targetFree > 0) {
        const mark = document.createElement('i');
        mark.className = 'target';
        mark.style.left = (((device.capacityBytes - targetFree) / scale) * 100) + '%';
        mark.title = 'Target: keep ' + fmtBytes(targetFree) + ' free';
        bar.appendChild(mark);
      }
      return bar;
    }

    function renderMemoryView(view) {
      const verdict = $('memVerdict');
      const devicesEl = $('memDevices');
      const legend = $('memLegend');
      const fixes = $('memFixes');
      devicesEl.textContent = '';
      fixes.textContent = '';
      legend.textContent = '';
      $('memNotes').textContent = '';
      $('memLines').textContent = '';
      if (!view) {
        verdict.className = 'verdict unknown';
        verdict.textContent = 'Select a model to estimate VRAM / RAM use.';
        legend.classList.add('hidden');
        fixes.classList.add('hidden');
        $('memFootnote').textContent = '';
        renderHeader();
        return;
      }
      lastMemoryView = view;
      verdict.className = 'verdict ' + view.level;
      const icon = view.level === 'good' ? '✓ ' : view.level === 'spill' ? '✕ ' : view.level === 'tight' ? '⚠ ' : '';
      verdict.textContent = icon + view.headline;

      const totals = {};
      for (const d of view.devices || []) {
        const row = document.createElement('div');
        row.className = 'mem-dev';
        const head = document.createElement('div');
        head.className = 'mem-chart-title';
        const name = document.createElement('span');
        name.textContent = d.label + (d.capacityBytes && d.kind !== 'ram' ? ' · ' + fmtBytes(d.capacityBytes) : '');
        name.title = d.fullLabel;
        const sub = document.createElement('span');
        sub.className = 'sub' + (d.level === 'spill' ? ' over' : d.level === 'tight' ? ' warn' : '');
        let text = '~' + fmtBytes(d.usedBytes) + (d.capacityBytes ? ' / ' + fmtBytes(d.capacityBytes) : '');
        if (serverRunning && d.kind !== 'ram' && d.liveUsedBytes > 0) {
          text += ' · live ' + fmtBytes(d.liveUsedBytes);
        }
        sub.textContent = text;
        sub.title = 'Estimate at full context' + (d.liveUsedBytes > 0 ? '. Live = what the driver reports now.' : '');
        head.appendChild(name);
        head.appendChild(sub);
        row.appendChild(head);
        row.appendChild(memBar(d, view.targetFreeBytes));
        devicesEl.appendChild(row);
        for (const s of d.segments || []) {
          if (s.bytes > 0) totals[s.key] = (totals[s.key] || 0) + s.bytes;
        }
      }
      for (const key of ['weights', 'vision', 'draft', 'kv', 'overhead']) {
        if (!totals[key]) continue;
        const span = document.createElement('span');
        const swatch = document.createElement('i');
        swatch.className = 'seg-' + key;
        span.appendChild(swatch);
        span.appendChild(document.createTextNode(SEG_LABEL[key] + ' ' + fmtBytes(totals[key])));
        legend.appendChild(span);
      }
      legend.classList.toggle('hidden', !Object.keys(totals).length);
      $('memFootnote').textContent = view.footnote || '';

      const fixList = Array.isArray(view.fixes) ? view.fixes : [];
      fixes.classList.toggle('hidden', !fixList.length);
      if (fixList.length) {
        const lead = document.createElement('span');
        lead.className = 'hint';
        lead.textContent = view.level === 'spill' ? 'Make it fit:' : 'Get to ' + fmtBytes(view.targetFreeBytes) + ' free:';
        fixes.appendChild(lead);
        for (const f of fixList) {
          const b = document.createElement('button');
          b.type = 'button';
          b.className = 'chip fix ' + f.level;
          b.textContent = f.label + ' → ' + f.detail;
          b.title = 'Apply ' + Object.entries(f.patch).map(([k, v]) => k + ' = ' + (v === '' ? 'auto' : v)).join(', ');
          b.addEventListener('click', () => applyLoadPatch(f.patch));
          fixes.appendChild(b);
        }
      }
      for (const [targetId, list] of [['memNotes', view.notes], ['memLines', view.lines]]) {
        const el = $(targetId);
        for (const line of list || []) {
          const div = document.createElement('div');
          div.textContent = String(line);
          el.appendChild(div);
        }
      }
      renderHeader();
    }

    /** Apply a set of load settings in one round-trip (fix chips, "fit" preset). */
    function applyLoadPatch(patch, opts) {
      if (saveLoadTimer) {
        clearTimeout(saveLoadTimer);
        saveLoadTimer = null;
      }
      markDirtyIfRunning();
      vscode.postMessage({
        type: 'applyLoadPatch',
        payload: Object.assign({}, readLoad(), patch),
        fitContext: !!(opts && opts.fitContext),
        cpuOnly: cpuOnlyLive(),
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
    // Context: logarithmic slider (0–1000) that snaps to the common sizes; the
    // number box keeps any exact value.
    const CTX_SNAPS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
    let ctxMin = 2048;
    let ctxMax = 131072;
    function ctxToSlider(ctx) {
      const c = Math.min(ctxMax, Math.max(ctxMin, Number(ctx) || ctxMin));
      return Math.round((1000 * Math.log(c / ctxMin)) / Math.log(ctxMax / ctxMin));
    }
    function sliderToCtx(v) {
      const raw = ctxMin * Math.pow(ctxMax / ctxMin, Number(v) / 1000);
      for (const s of CTX_SNAPS.concat([ctxMax])) {
        if (s <= ctxMax && Math.abs(raw - s) / s < 0.06) return s;
      }
      return Math.max(ctxMin, Math.min(ctxMax, Math.round(raw / 256) * 256));
    }
    function syncCtxK() {
      const el = $('ctxK');
      if (el) el.textContent = '≈' + fmtTokShort(Number($('contextLength').value));
    }
    function setContextField(ctx) {
      if (!fieldFocused('contextLength')) setField('contextLength', ctx);
      if (!fieldFocused('contextLengthRange')) $('contextLengthRange').value = String(ctxToSlider(ctx));
      syncCtxK();
    }
    function renderCtxTicks() {
      const ticks = $('ctxTicks');
      if (!ticks) return;
      ticks.textContent = '';
      const marks = CTX_SNAPS.filter((s) => s >= ctxMin && s < ctxMax * 0.97).concat([ctxMax]);
      for (const s of marks) {
        const t = document.createElement('span');
        t.textContent = fmtTokShort(s);
        t.style.left = (ctxToSlider(s) / 10) + '%';
        ticks.appendChild(t);
      }
    }
    (function bindContext() {
      const num = $('contextLength');
      const range = $('contextLengthRange');
      num.addEventListener('input', () => {
        range.value = String(ctxToSlider(num.value));
        syncCtxK();
        scheduleLiveMemory();
      });
      range.addEventListener('input', () => {
        num.value = String(sliderToCtx(range.value));
        syncCtxK();
        scheduleLiveMemory();
      });
    })();
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

    // 'fit' = largest context that still fits the detected VRAM (core fittingContextLength).
    const LOAD_PRESETS = {
      agent: { contextLength: 65536, cacheTypeK: 'q8_0', cacheTypeV: 'q8_0', slots: 1 },
      context: { contextLength: 'fit', cacheTypeK: 'q8_0', cacheTypeV: 'q4_0', slots: 1 },
      quality: { contextLength: 65536, cacheTypeK: 'f16', cacheTypeV: 'q8_0', slots: 1 },
    };

    function currentPresetId() {
      const k = $('cacheTypeK').value;
      const v = $('cacheTypeV').value;
      if (Number($('maxConcurrentPredictions').value) !== 1) return '';
      const ctx = Number($('contextLength').value);
      const maxCtx = ctxMax;
      for (const [id, p] of Object.entries(LOAD_PRESETS)) {
        if (p.cacheTypeK !== k || p.cacheTypeV !== v) continue;
        // 'fit' depends on live VRAM, so any context counts as a match once the
        // distinctive K/V pair lines up.
        if (p.contextLength === 'fit' || ctx === Math.min(p.contextLength, maxCtx)) return id;
      }
      return '';
    }

    /** Highlight the matching preset, or "Custom" with what the mix is. */
    function highlightPreset() {
      const active = currentPresetId();
      document.querySelectorAll('#presetChips .chip[data-preset]').forEach((chip) => {
        chip.classList.toggle('active', chip.dataset.preset === active);
      });
      const custom = $('presetCustom');
      if (custom) custom.classList.toggle('active', !active);
      const hint = $('presetHint');
      if (hint) {
        const k = $('cacheTypeK').value;
        const v = $('kvTypesLinked').checked ? k : $('cacheTypeV').value;
        const slots = Number($('maxConcurrentPredictions').value) || 1;
        const mix = 'ctx ' + fmtTokShort(Number($('contextLength').value)) + ' · KV ' + k + '/' + v + ' · ' + slots + ' slot' + (slots === 1 ? '' : 's');
        hint.textContent = active
          ? mix + '. Presets set context, KV cache types and slots together.'
          : 'Custom: ' + mix + '.';
      }
    }

    function applyPreset(id) {
      const p = LOAD_PRESETS[id];
      if (!p) return;
      $('maxConcurrentPredictions').value = p.slots;
      $('cacheTypeK').value = p.cacheTypeK;
      $('cacheTypeV').value = p.cacheTypeV;
      $('kvTypesLinked').checked = p.cacheTypeK === p.cacheTypeV;
      syncKvLink(false);
      syncFlashAttentionWarning();
      if (p.contextLength === 'fit') {
        // The extension searches with core's estimator and pushes the result.
        applyLoadPatch({ cacheTypeK: p.cacheTypeK, cacheTypeV: p.cacheTypeV, maxConcurrentPredictions: p.slots }, { fitContext: true });
        return;
      }
      setContextField(Math.min(p.contextLength, ctxMax));
      highlightPreset();
      refreshMemoryLive();
      scheduleSaveLoad();
    }

    document.querySelectorAll('#presetChips .chip[data-preset]').forEach((chip) => {
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

    // Advanced: mark values that differ from Llama AIO defaults, filter by label or flag.
    const ADV_KEYS = new Set([
      'cpuThreads', 'evalBatchSize', 'physicalBatchSize', 'maxConcurrentPredictions', 'flashAttention',
      'offloadKvCacheToGpu', 'cacheTypeK', 'cacheTypeV', 'keepModelInMemory', 'tryMmap', 'lazyMode',
      'unifiedKvCache', 'contextCheckpoints', 'cacheReuse', 'reasoningFormat', 'reasoningBudget',
      'ropeFreqBase', 'ropeFreqScale', 'seed', 'speculativeMode', 'ngramVariant', 'ngramSizeN',
      'ngramSizeM', 'ngramMinHits', 'maxDraftTokens', 'minDraftTokens', 'draftProbability', 'draftGpuOffload',
    ]);
    const ADV_ALIAS = {
      reasoningBudgetUnlimited: 'reasoningBudget', ropeBaseAuto: 'ropeFreqBase', ropeScaleAuto: 'ropeFreqScale',
      seedRandom: 'seed', cpuThreadsRange: 'cpuThreads', ngramSizeNRange: 'ngramSizeN',
    };
    function advRowKey(row) {
      for (const el of row.querySelectorAll('input[id], select[id]')) {
        const key = ADV_ALIAS[el.id] || el.id;
        if (ADV_KEYS.has(key)) return key;
      }
      return '';
    }
    function refreshAdvancedMarks() {
      const body = $('advancedBody');
      if (!body) return;
      const L = readLoad();
      const query = String(($('advFilter') && $('advFilter').value) || '').trim().toLowerCase();
      const changedOnly = !!($('advChangedOnly') && $('advChangedOnly').checked);
      const filtering = !!query || changedOnly;
      let changedCount = 0;
      const rows = [...body.querySelectorAll('.row, .toggle')];
      for (const row of rows) {
        const key = advRowKey(row);
        const changed = !!key && key in loadDefaults && JSON.stringify(L[key]) !== JSON.stringify(loadDefaults[key]);
        row.classList.toggle('changed', changed);
        // Settings hidden for this mode (e.g. n-gram variant with MTP) do not count.
        if (changed && !row.closest('.hidden')) changedCount++;
        const flags = [...row.querySelectorAll('[data-flag]')].map((n) => n.getAttribute('data-flag')).join(' ');
        const text = (row.textContent + ' ' + flags).toLowerCase();
        const visible = (!query || text.includes(query)) && (!changedOnly || changed);
        row.classList.toggle('filtered-out', filtering && !visible);
      }
      // Hide sub-headings (and spec groups) that have nothing left to show.
      for (const title of body.querySelectorAll('.subgroup-title, .spec-group-title')) {
        let n = title.nextElementSibling;
        let any = false;
        while (n && !n.classList.contains('subgroup-title')) {
          if (n.matches('.row, .toggle, .spec-group') && !n.classList.contains('hidden')) {
            const inner = n.matches('.spec-group') ? [...n.querySelectorAll('.row')] : [n];
            if (inner.some((r) => !r.classList.contains('filtered-out') && !r.classList.contains('hidden'))) any = true;
          }
          n = n.nextElementSibling;
        }
        title.classList.toggle('filtered-out', filtering && !any);
      }
      for (const hint of body.querySelectorAll(':scope > .hint')) hint.classList.toggle('filtered-out', filtering);
      const sum = $('advSum');
      if (sum) sum.textContent = changedCount
        ? changedCount + ' changed from defaults'
        : 'threads, batch, KV, RoPE, speculative';
    }
    if ($('advFilter')) $('advFilter').addEventListener('input', refreshAdvancedMarks);
    if ($('advChangedOnly')) $('advChangedOnly').addEventListener('change', refreshAdvancedMarks);

    // Remember open sections and scroll position while the view is hidden (G-15).
    const savedUi = vscode.getState() || {};
    let scrollRestored = false;
    function saveUiState() {
      const open = [...document.querySelectorAll('details[id]')]
        .filter((d) => d.open && d.id !== 'srvMenu')
        .map((d) => d.id);
      vscode.setState({ open: open, scrollY: window.scrollY });
    }
    for (const id of Array.isArray(savedUi.open) ? savedUi.open : []) {
      const d = $(id);
      if (d && d.tagName === 'DETAILS') d.open = true;
    }
    document.querySelectorAll('details[id]').forEach((d) => d.addEventListener('toggle', saveUiState));
    let scrollSaveTimer = null;
    window.addEventListener('scroll', () => {
      if (scrollSaveTimer) clearTimeout(scrollSaveTimer);
      scrollSaveTimer = setTimeout(saveUiState, 200);
    });

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
      ctxMax = Math.max(512, maxCtx);
      ctxMin = Math.min(2048, ctxMax);
      renderCtxTicks();
      // Slider max = actual layer count (legacy 99/"all" is shown as all layers).
      $('gpuOffload').max = String(modelBlockCount);
      $('gpuOffloadRange').max = String(modelBlockCount);
      $('nCpuMoe').max = String(modelBlockCount);
      $('nCpuMoeRange').max = String(modelBlockCount);
      $('nCpuFfn').max = String(modelBlockCount);
      $('nCpuFfnRange').max = String(modelBlockCount);

      // Visibility finalized in applyCpuOnlyUi (also hides on CPU backend).
      if (caps) {
        $('ctxHint').textContent = 'Model max ' + fmtTokShort(maxCtx) + ' (' + Number(maxCtx).toLocaleString() + ' tokens).';
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
        const capBits = [
          caps.architecture || '?',
          isMoe ? ('MoE ' + (caps.expertCount || '?') + ' experts') : 'dense',
          blocks + ' layers',
          'max ctx ' + fmtTokShort(maxCtx),
        ];
        if (caps.fullAttentionInterval > 1) capBits.push('full attn every ' + caps.fullAttentionInterval);
        if (caps.pleShare > 0) capBits.push('PLE ~' + Math.round(caps.pleShare * 100) + '%');
        if (caps.nextnPredictLayers > 0) capBits.push('MTP ✓');
        else if (mtpSidecarPath) capBits.push('MTP sidecar');
        const modelCaps = $('modelCaps');
        modelCaps.classList.remove('hidden');
        modelCaps.textContent = capBits.join(' · ');
        modelCaps.title = mtpSidecarPath ? 'MTP sidecar: ' + String(mtpSidecarPath).split(/[/\\\\]/).pop() : '';
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
      lastPayload = payload;
      pendingChanges = Array.isArray(payload.changes) ? payload.changes : [];
      loadDefaults = payload.defaults || {};
      gpuInfos = Array.isArray(payload.gpus) ? payload.gpus : [];

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
      const changeBtn = $('changeModelBtn');
      $('starterCol').classList.toggle('hidden', !showStarter);
      if (starterBtn) starterBtn.classList.toggle('hidden', !showStarter);
      if (starterHint) starterHint.classList.toggle('hidden', !showStarter);
      if (changeBtn) {
        changeBtn.textContent = hasModel ? 'Change…' : 'Choose…';
        changeBtn.className = (showStarter && localCount > 0 ? 'primary' : 'secondary') + ' small';
      }
      $('modelTitle').textContent = hasModel ? (payload.modelName || 'model') : 'No model selected';
      $('modelTitle').title = hasModel && payload.modelNameRaw ? 'general.name: ' + payload.modelNameRaw : '';
      const libSum = $('librarySum');
      if (libSum) {
        const sources = (Array.isArray(payload.localSourceDirs) ? payload.localSourceDirs : []).map((d) => d.source);
        libSum.textContent = localCount + ' GGUF' + (sources.length ? ' · ' + sources.slice(0, 3).join(', ') + (sources.length > 3 ? '…' : '') : '');
      }
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
        // Middle-ellipsis keeps the folder and the file name; the full path is the tooltip.
        const p = String(s.selectedModelPath);
        const parts = p.split(/[/\\\\]/);
        const file = parts.pop() || p;
        const dir = parts.pop() || '';
        const short = (dir ? '…/' + (dir.length > 28 ? dir.slice(0, 26) + '…' : dir) + '/' : '') + file;
        $('modelPath').innerHTML =
          '<a class="model-path-link" href="#" data-path="' + escAttr(p) +
          '" title="' + escAttr(p) + ' — reveal in file manager">' + escText(short) + '</a>';
        bindFolderLinks($('modelPath'));
      } else {
        $('modelPath').textContent = 'Pick a downloaded GGUF, open a file, or download one from Hugging Face.';
      }

      const modelsDir = payload.modelsDir || '';
      const sourceDirs = Array.isArray(payload.localSourceDirs) ? payload.localSourceDirs : [];
      const sourceHtml = sourceDirs.length
        ? ' · sources: ' + sourceDirs.map((src) =>
            folderLink(src.source, src.dir, 'Open ' + src.source + ' folder')
          ).join(', ')
        : '';
      $('modelsDirMeta').innerHTML =
        'Downloads go to ' +
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

      setContextField(ctx);
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
      const splitDirty = applyDualGpuUi(L);
      applyCpuOnlyUi(!!payload.cpuOnly || payload.selectedUiBackend === 'cpu');
      // Core already estimated the saved settings; newer live requests still win.
      renderedSeq = estimateSeq;
      renderMemoryView(payload.memoryView || null);
      renderCopilotSummary(payload);
      refreshAdvancedMarks();
      updatePrimaryAction();
      renderHeader();
      if (splitDirty) {
        scheduleSaveLoad();
        requestEstimate();
      }
      if (!scrollRestored) {
        scrollRestored = true;
        if (savedUi.scrollY > 0) window.scrollTo(0, savedUi.scrollY);
      }
    }

    function renderCopilotSummary(payload) {
      const el = $('copilotSum');
      if (!el) return;
      const R = (payload.state && payload.state.requestSettings) || {};
      const bits = [];
      if (payload.modeSampling) bits.push(payload.modeSampling.familyLabel + ' modes');
      else if (typeof R.temperature === 'number') bits.push('temp ' + R.temperature);
      bits.push('replacements ' + (payload.promptReplacementsEnabled ? 'on' : 'off'));
      if (payload.wikipediaLookupEnabled) bits.push('wiki on');
      if (payload.duplicateToolCallGuardEnabled) bits.push('dup-guard on');
      el.textContent = bits.join(' · ');
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
    $('changeModelBtn').addEventListener('click', () => vscode.postMessage({ type: 'changeModel' }));
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

    function launch(action) {
      if (action !== 'reload' && action !== 'start') return;
      serverStarting = true;
      updatePrimaryAction();
      renderStatusUi({
        ready: false,
        dirty: false,
        starting: true,
        endpoint: '',
        message: '',
      });
      vscode.postMessage({ type: action, payload: readLoad() });
      vscode.postMessage({ type: 'saveRequest', payload: readRequest() });
    }
    $('primaryBtn').addEventListener('click', () => launch($('primaryBtn').dataset.action));
    $('reloadIconBtn').addEventListener('click', () => launch('reload'));
    $('discardBtn').addEventListener('click', () => {
      if (saveLoadTimer) {
        clearTimeout(saveLoadTimer);
        saveLoadTimer = null;
      }
      vscode.postMessage({ type: 'discardChanges' });
    });
    $('stopBtn').addEventListener('click', () => vscode.postMessage({ type: 'stop' }));
    const closeMenu = () => {
      const menu = $('srvMenu');
      if (menu) menu.open = false;
    };
    $('openLogBtn').addEventListener('click', () => { closeMenu(); vscode.postMessage({ type: 'openLog' }); });
    $('copyCmdBtn').addEventListener('click', () => { closeMenu(); vscode.postMessage({ type: 'copyCommandLine' }); });
    $('openWebUiBtn').addEventListener('click', () => {
      closeMenu();
      if (lastEndpoint) vscode.postMessage({ type: 'openExternal', url: lastEndpoint });
    });
    document.addEventListener('click', (e) => {
      const menu = $('srvMenu');
      if (menu && menu.open && !menu.contains(e.target)) menu.open = false;
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeMenu();
    });

    $('nglAll').addEventListener('click', () => {
      nglCustomOpen = false;
      const blocks = Math.max(1, modelBlockCount || 1);
      $('gpuOffload').value = String(blocks);
      $('gpuOffloadRange').value = String(blocks);
      syncNglMode();
      scheduleSaveLoad();
      refreshMemoryLive();
    });
    $('nglCustom').addEventListener('click', () => {
      nglCustomOpen = true;
      syncNglMode();
      $('gpuOffloadRange').focus();
    });
    document.querySelectorAll('#splitSeg .seg-btn').forEach((b) => {
      b.addEventListener('click', () => {
        const sel = $('splitMode');
        if (!sel || sel.disabled || sel.value === b.dataset.split) return;
        sel.value = b.dataset.split;
        sel.dispatchEvent(new Event('change'));
        applyCpuOnlyUi(cpuOnlyLive());
      });
    });
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
      if (msg.type === 'memoryEstimate') {
        // Drop replies to superseded requests (typing or dragging fires many).
        if (typeof msg.seq === 'number' && msg.seq >= renderedSeq && msg.seq === estimateSeq) {
          renderedSeq = msg.seq;
          renderMemoryView(msg.view || null);
        }
      }
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
        pendingChanges = Array.isArray(p.changes) ? p.changes : [];
        updatePrimaryAction();
        renderStatusUi({
          ready: httpUp,
          dirty: configDirty,
          starting,
          endpoint: lastEndpoint,
          pid: lastPid,
          message: starting ? (p.startMessage || p.message || 'Starting…') : (p.message || ''),
        });
        renderHeader();
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

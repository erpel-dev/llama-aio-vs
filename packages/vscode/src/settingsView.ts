import { randomBytes } from "crypto";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";
import { isCopilotChatAvailable, promptUseInCopilotChat } from "./copilotChatPrompt";
import { copyServerCommandLine, openServerLog, reportLaunchFailure } from "./serverDiagnostics";
import type { ToHostMessage, ToWebviewMessage } from "./webview/protocol";
import {
  connectionHint,
  detectGpusAsync,
  discoverLoopbackLlamaServersOnThisMachineAsync,
  activeInstallLock,
  formatRemoteDiagnostics,
  isPublicPlainHttp,
  normalizeServerUrl,
  originOf,
  probeRemoteServer,
  remoteLifecycleMessage,
  type GpuMemoryInfo,
  type DiscoveredLlamaServer,
} from "@llama-aio/core";
import { LlamaInstaller, UiBackend } from "@llama-aio/core";
import { clampLoadSettingsToModel, computeMemoryView, diffLoadSettings, fittingContextLength, memoryEstimateInputs, mmprojFileSize, resolveDraftCapabilities, shortGpuName, type SettingChange } from "@llama-aio/core";
import { resolveModelModes } from "@llama-aio/core";
import {
  buildLibraryGroupsAsync,
  canonicalModelPath,
  displayModelTitle,
  formatModelSize,
  friendlyModelTitle,
  hiddenModelPaths,
  isOwnedModelPath,
  listActiveModelSourceDirs,
  listLocalModelEntries,
  listPartialDownloads,
  listShardPaths,
  findSiblingMtpDraft,
  invalidateModelLibraryCache,
  isMtpSidecarFile,
  readPickerCapabilities,
  removalPathsForQuant,
  sameModelFile,
  withoutHiddenModels,
  type LibraryGroup,
} from "@llama-aio/core";
import { getModelsDir } from "@llama-aio/core";
import { PerfStats } from "@llama-aio/core";
import { LaunchToken, LAUNCH_IN_PROGRESS_MSG, ProcessManager } from "@llama-aio/core";
import { SettingsStore } from "@llama-aio/core";
import { resolveLaunchMode } from "@llama-aio/core";
import { BUILTIN_CHAT_TEMPLATES, capsMissDefaultSwaPattern, chatTemplateFileProblem, COMMON_CHAT_TEMPLATES, DEFAULT_LOAD_SETTINGS, DEFAULT_REQUEST_SETTINGS, effectiveServerUiState, isQwen4expArchitecture, LlamaLoadSettings, normalizeSpeculativeMode, RequestSettings } from "@llama-aio/core";
import { isDflashDraftArchitecture, recommendedMaxDraftTokens, speculativeUsesMtp, speculativeUsesNgram } from "@llama-aio/core";
import {
  companionDownloadHint,
  describeLanguageGgufFile,
  DownloadAbortError,
  downloadManager,
  downloadPickerFiles,
  GatedDownloadError,
  licenseFromTags,
  listingBaseName,
  resolveLicenseUrl,
  type DownloadJobSnapshot,
} from "@llama-aio/core";
import { downloadGgufSelection, HuggingFaceClient, STARTER_MODEL } from "./huggingFace";

/** Secret Storage keyed by server origin, with a sync cache for request headers. */
export interface ApiKeyStore {
  /** Cached key; may miss a key stored in another window or before activation. */
  get(origin: string): string | undefined;
  /** Read Secret Storage into the cache, then return the key. */
  load(origin: string): Promise<string | undefined>;
  set(origin: string, value: string): Promise<void>;
  delete(origin: string): Promise<void>;
}

/** Hugging Face token in Secret Storage, with a sync cache for request headers. */
export interface HfTokenStore {
  get(): string | undefined;
  load(): Promise<string | undefined>;
  set(value: string): Promise<void>;
  delete(): Promise<void>;
}

export type ModelActions = {
  downloadStarter: () => Promise<void>;
  openGgufFile: () => Promise<void>;
  pickDraftModel: () => Promise<void>;
  pickMmproj: () => Promise<void>;
  pickChatTemplate: () => Promise<void>;
  installLlamaCpp: (backend?: UiBackend) => Promise<void>;
  reinstallLlamaCpp: () => Promise<void>;
  installLlamaCppByTag: () => Promise<void>;
  installLlamaCppFromArchive: () => Promise<void>;
  cancelLlamaInstall: () => void;
  retryLlamaInstall: () => Promise<void>;
  switchBackend: (backend: UiBackend) => Promise<void>;
  selectModel: (filePath: string) => Promise<void>;
};

/**
 * Curated model modes overwrite temperature/top_p/top_k on every request, so the
 * Request defaults below them are inert for those models. Describe the active
 * mode set so the panel can say so instead of showing numbers that never ship.
 */
function chatTemplateNameOptions(): string {
  const common = new Set<string>(COMMON_CHAT_TEMPLATES);
  const opt = (name: string) => `<option value="${name}">${name}</option>`;
  const other = BUILTIN_CHAT_TEMPLATES.filter((name) => !common.has(name))
    .slice()
    .sort((a, b) => a.localeCompare(b))
    .map(opt)
    .join("");
  return (
    `<optgroup label="Common">${COMMON_CHAT_TEMPLATES.map(opt).join("")}</optgroup>` +
    `<optgroup label="Other">${other}</optgroup>`
  );
}

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
  /** Drops fit-badge work from an older library scan when a newer state arrives. */
  private libraryFitGen = 0;
  /** What the last posted fit badges were computed from (see postLibraryFits). */
  private libraryFitsPosted: { signature: string; at: number } | undefined;
  private hfClient?: HuggingFaceClient;
  private downloadsBound = false;
  private latestJobs: DownloadJobSnapshot[] = [];
  private postedJobState = "";
  private downloadPostTimer: ReturnType<typeof setTimeout> | undefined;
  private libraryFocus: { filter?: boolean; downloads?: boolean; query?: string; search?: boolean } | null = null;
  private libraryReady = false;
  private wantConnectForm = false;
  private pushInFlight: Promise<void> | undefined;
  private pushTrailing: Promise<void> | undefined;
  /** Last loopback scan. The 5s status poll reuses this instead of running `ss` every tick. */
  private localServerScan?: { at: number; servers: DiscoveredLlamaServer[] };
  private localServerScanRun?: Promise<DiscoveredLlamaServer[]>;
  private apiKeys?: ApiKeyStore;
  private hfTokens?: HfTokenStore;

  /** Typed send to the sidebar; a no-op while the view is closed. */
  private post(msg: ToWebviewMessage): void {
    void this.view?.webview.postMessage(msg);
  }

  /** Secret Storage cache owned by the extension. Keys never go in config.json. */
  attachApiKeyStore(store: ApiKeyStore): void {
    this.apiKeys = store;
  }

  /** Hugging Face token keeps the same treatment as remote API keys. */
  attachHfTokenStore(store: HfTokenStore): void {
    this.hfTokens = store;
  }

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
        this.libraryReady = false;
      }
    });
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this.extensionUri],
    };
    webviewView.webview.html = this.getHtml(webviewView.webview);
    this.bindDownloads();
    webviewView.webview.onDidReceiveMessage(async (msg: ToHostMessage) => {
      try {
        switch (msg.type) {
          case "ready":
            this.libraryReady = true;
            // A (re)created webview starts without badges.
            this.libraryFitsPosted = undefined;
            await this.pushState();
            this.flushDownloadPost();
            this.flushLibraryFocus();
            if (this.wantConnectForm) {
              this.wantConnectForm = false;
              this.post({ type: "openConnect" });
            }
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
              chatTemplateSource: d.chatTemplateSource,
              chatTemplateName: d.chatTemplateName,
              chatTemplateFile: d.chatTemplateFile,
              chatTemplateJinja: d.chatTemplateJinja,
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
            this.post({ type: "memoryEstimate", seq: msg.seq, view: view ?? null });
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
          case "reload": {
            if (await this.refuseRemote("reload")) {
              break;
            }
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
            if (await this.refuseRemote("start")) {
              break;
            }
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
            if (await this.refuseRemote("stop")) {
              break;
            }
            await this.processManager.stop(true);
            await this.pushState();
            break;
          case "openLog":
            await openServerLog();
            break;
          case "copyCommandLine":
            await copyServerCommandLine();
            break;
          case "refreshRemote":
            await this.processManager.remote.refresh(true);
            this.rememberRemoteModel();
            this.notifyChatModels();
            await this.pushState();
            break;
          case "testRemote":
            await this.testRemote(String(msg.url || ""), String(msg.apiKey || ""));
            break;
          case "connectRemote":
            await this.connectRemote(String(msg.url || ""), String(msg.label || ""), String(msg.apiKey || ""));
            break;
          case "switchToLocal":
            await this.switchToLocal();
            break;
          case "setApiKey":
            await this.promptApiKey();
            break;
          case "forgetApiKey":
            await this.forgetApiKey();
            break;
          case "copyDiagnostics":
            await this.copyDiagnostics();
            break;
          case "setLaunchMode": {
            const mode = msg.payload === "background" ? "background" : "externalTerminal";
            await this.store
              .getConfig()
              .update("launchMode", mode);
            await this.pushState();
            break;
          }
          case "downloadStarter":
            await this.modelActions.downloadStarter();
            await this.pushState();
            break;
          case "openModelFile":
            await this.modelActions.openGgufFile();
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
            this.post({ type: "draftModelSelected", path: "" });
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
            this.post({ type: "mmprojSelected", path: "" });
            this.notifyChatModels();
            await this.pushState();
            break;
          case "pickChatTemplate":
            try {
              await vscode.commands.executeCommand("llamaAio.selectChatTemplate");
            } catch {
              await this.modelActions.pickChatTemplate();
            }
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
          case "cancelLlamaInstall":
            this.modelActions.cancelLlamaInstall();
            break;
          case "retryLlamaInstall":
            await this.modelActions.retryLlamaInstall();
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
          case "useLibraryModel": {
            const filePath = typeof msg.path === "string" ? msg.path : "";
            if (filePath) {
              await this.modelActions.selectModel(filePath);
            }
            break;
          }
          case "useLibraryDraft": {
            const filePath = typeof msg.path === "string" ? msg.path : "";
            if (filePath) {
              await this.useDflashDraft(filePath);
            }
            break;
          }
          case "copyPath": {
            const filePath = typeof msg.path === "string" ? msg.path : "";
            if (filePath) {
              await vscode.env.clipboard.writeText(filePath);
              vscode.window.setStatusBarMessage("Llama AIO: path copied", 2000);
            }
            break;
          }
          case "hideLibraryModel": {
            await this.hideLibraryPath(typeof msg.path === "string" ? msg.path : "");
            break;
          }
          case "unhideLibraryModel": {
            await this.unhideLibraryPath(typeof msg.path === "string" ? msg.path : "");
            break;
          }
          case "removeLibraryQuant": {
            const quantPath = typeof msg.path === "string" ? msg.path : "";
            const companions = Array.isArray(msg.companionPaths)
              ? msg.companionPaths.filter((p: unknown): p is string => typeof p === "string")
              : [];
            await this.removeOwnedQuant(quantPath, companions);
            break;
          }
          case "cleanPartialDownloads":
            await this.cleanPartialDownloads();
            break;
          case "searchLibraryHf":
            await this.searchLibraryHf(typeof msg.query === "string" ? msg.query : "", msg.requestId);
            break;
          case "listLibraryHfFiles":
            await this.listLibraryHfFiles(typeof msg.modelId === "string" ? msg.modelId : "", msg.requestId);
            break;
          case "downloadLibraryHf":
            void this.runLibraryDownload(
              typeof msg.modelId === "string" ? msg.modelId : "",
              typeof msg.filePath === "string" ? msg.filePath : ""
            );
            break;
          case "pauseDownload":
            downloadManager.pause(typeof msg.id === "string" ? msg.id : "");
            break;
          case "resumeDownload":
            downloadManager.resume(typeof msg.id === "string" ? msg.id : "");
            break;
          case "cancelDownload":
            downloadManager.cancel(typeof msg.id === "string" ? msg.id : "");
            break;
          case "setHfToken": {
            const token = await vscode.window.showInputBox({
              title: "Hugging Face access token",
              prompt:
                "Paste a token with access to gated repos (stored in VS Code Secret Storage; leave empty to clear).",
              password: true,
              ignoreFocusOut: true,
            });
            if (token !== undefined) {
              const trimmed = token.trim();
              if (this.hfTokens) {
                if (trimmed) {
                  await this.hfTokens.set(trimmed);
                } else {
                  await this.hfTokens.delete();
                }
              } else {
                // No secret store available (e.g. tests): keep the legacy path.
                await this.store.getConfig().update("hfToken", trimmed);
              }
              vscode.window.setStatusBarMessage(
                trimmed ? "Llama AIO: Hugging Face token saved" : "Llama AIO: Hugging Face token cleared",
                4000
              );
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

  private async detectGpusForEstimate(cpuOnly: boolean): Promise<GpuMemoryInfo[]> {
    if (cpuOnly || activeInstallLock()) {
      return [];
    }
    // Async: nvidia-smi / rocm-smi / --list-devices must not block the extension host.
    return detectGpusAsync(false, this.processManager.resolveBinary());
  }

  /**
   * The one memory path for the sidebar: core estimate + fit verdict for the
   * saved settings. Spill confirmation, the state push and live edits all go
   * through `computeMemoryView`, so the bars and the dialog cannot disagree.
   */
  private async memoryForSaved(cpuOnly: boolean) {
    const gpus = await this.detectGpusForEstimate(cpuOnly);
    const state = this.store.getState();
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
    if (this.store.isRemote()) {
      this.perf.setSpeculativeMode(undefined);
      return;
    }
    const mode = this.store.getState().loadSettings.speculativeMode || "off";
    this.perf.setSpeculativeMode(normalizeSpeculativeMode(mode));
  }

  /** Ask to switch back when a local lifecycle action arrives in remote mode. */
  private async refuseRemote(action: string): Promise<boolean> {
    if (!this.store.isRemote()) {
      return false;
    }
    const choice = await vscode.window.showWarningMessage(
      remoteLifecycleMessage(action, this.store.getServerTarget().label),
      "Switch to this machine"
    );
    if (choice === "Switch to this machine") {
      await this.switchToLocal();
    }
    return true;
  }

  /**
   * Hand-started llama-server processes on loopback, while this machine's own
   * server is stopped. Empty while a server we own is running or remote mode is on.
   * Cached for 20s so status polling does not rescan; the scan itself is async.
   */
  private async localServersForSidebar(
    status: ReturnType<ProcessManager["getStatus"]>,
    httpReady = false
  ): Promise<DiscoveredLlamaServer[]> {
    if (this.store.isRemote() || status.running || status.starting || httpReady) {
      this.localServerScan = undefined;
      return [];
    }
    const now = Date.now();
    if (this.localServerScan && now - this.localServerScan.at < 20_000) {
      return this.localServerScan.servers;
    }
    if (!this.localServerScanRun) {
      const skip = status.ownedByThisExtension && status.pid ? [status.pid] : [];
      this.localServerScanRun = discoverLoopbackLlamaServersOnThisMachineAsync(skip)
        .catch(() => [])
        .then((servers) => {
          this.localServerScan = { at: Date.now(), servers };
          return servers;
        })
        .finally(() => {
          this.localServerScanRun = undefined;
        });
    }
    return this.localServerScanRun;
  }

  /** Fields the sidebar needs to draw remote mode. Included in state and status patches. */
  private remoteFields(): Record<string, unknown> {
    const remote = this.store.isRemote();
    const target = this.store.getServerTarget();
    const info = remote ? this.processManager.remote.current() : undefined;
    return {
      serverMode: remote ? "remote" : "local",
      remoteUrl: this.store.getServerPreferences().remoteUrl,
      remoteLabel: remote ? target.label : "",
      hasApiKey: this.currentHasApiKey(),
      remote: info,
      nextRetryAt: remote ? this.processManager.remote.nextRetryAt() : undefined,
      remoteHint: info?.error ? connectionHint(info.error.kind, target.baseUrl) : "",
    };
  }

  private currentHasApiKey(): boolean {
    if (!this.apiKeys || !this.store.isRemote()) {
      return false;
    }
    const origin = originOf(this.store.getEndpoint());
    return !!origin && !!this.apiKeys.get(origin);
  }

  private rememberRemoteModel(): void {
    const id = this.processManager.remote.current()?.modelId;
    if (id) {
      void this.store.setLastRemoteModelId(id).catch(() => undefined);
    }
  }

  /** Show the connect form. If the sidebar is not open yet, it opens on the next ready. */
  openConnectForm(): void {
    if (this.view) {
      this.post({ type: "openConnect" });
    } else {
      this.wantConnectForm = true;
    }
  }

  private async testRemote(rawUrl: string, apiKey: string): Promise<void> {
    let url: string;
    try {
      url = normalizeServerUrl(rawUrl);
    } catch (e) {
      this.post({
        type: "remoteProbe",
        error: e instanceof Error ? e.message : String(e),
      });
      return;
    }
    const origin = originOf(url);
    const key = apiKey.trim() || (origin ? await this.apiKeys?.load(origin) : undefined);
    try {
      const info = await probeRemoteServer(url, { apiKey: key });
      this.post({ type: "remoteProbe", payload: info });
    } catch (e) {
      this.post({
        type: "remoteProbe",
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }

  /** Save a remote URL and probe it. Confirms plain HTTP to a public address once. */
  async connectRemote(rawUrl: string, label: string, apiKey: string): Promise<void> {
    let url: string;
    try {
      url = normalizeServerUrl(rawUrl);
    } catch (e) {
      void vscode.window.showErrorMessage(e instanceof Error ? e.message : String(e));
      return;
    }
    if (isPublicPlainHttp(url) && !this.store.isPlainHttpConfirmed(url)) {
      const choice = await vscode.window.showWarningMessage(
        `${url} is plain HTTP on a public address. The API key and chat will travel unencrypted. Connect anyway?`,
        { modal: true },
        "Connect"
      );
      if (choice !== "Connect") {
        return;
      }
      await this.store.confirmPlainHttp(url);
    }
    const typedKey = apiKey.trim();
    if (typedKey) {
      if (!this.apiKeys) {
        void vscode.window.showWarningMessage("API key storage is not available in this window.");
      } else {
        await this.apiKeys.set(originOf(url)!, typedKey);
      }
    } else {
      // Reconnecting to a server used before: its key may only be in Secret Storage.
      await this.apiKeys?.load(originOf(url)!);
    }
    if (!this.store.isRemote() && this.processManager.getStatus().running) {
      const choice = await vscode.window.showWarningMessage(
        "A local llama-server is still running. Stop it, or leave it running in the background?",
        { modal: true },
        "Stop",
        "Keep running"
      );
      if (!choice) {
        return;
      }
      if (choice === "Stop") {
        await this.processManager.stop(true);
      }
    }
    await this.store.setRemoteServer(url, label);
    this.processManager.remote.reset();
    this.perf.setSpeculativeMode(undefined);
    await this.processManager.remote.refresh(true);
    this.rememberRemoteModel();
    this.notifyChatModels();
    await this.pushState();
    // Closes the connect form once the new target is live, including when the
    // user edited an existing connection (the mode does not change then).
    this.post({ type: "remoteConnected" });
  }

  async switchToLocal(): Promise<void> {
    await this.store.setServerMode("local");
    this.processManager.remote.reset();
    this.syncSpeculativeMode();
    this.notifyChatModels();
    await this.pushState();
  }

  async promptApiKey(): Promise<void> {
    if (!this.store.isRemote()) {
      return;
    }
    const target = this.store.getServerTarget();
    const value = await vscode.window.showInputBox({
      title: `API key for ${target.label}`,
      prompt: "Stored in VS Code Secret Storage. Leave empty to forget the key. It is never written to config.json.",
      password: true,
      ignoreFocusOut: true,
    });
    if (value === undefined) {
      return;
    }
    const origin = originOf(target.baseUrl);
    if (!origin) {
      return;
    }
    if (!value.trim()) {
      await this.apiKeys?.delete(origin);
    } else if (this.apiKeys) {
      await this.apiKeys.set(origin, value.trim());
    } else {
      void vscode.window.showWarningMessage("API key storage is not available in this window.");
      return;
    }
    this.processManager.remote.reset();
    await this.processManager.remote.refresh(true);
    this.rememberRemoteModel();
    this.notifyChatModels();
    await this.pushState();
  }

  async forgetApiKey(): Promise<void> {
    if (!this.store.isRemote()) {
      return;
    }
    const origin = originOf(this.store.getEndpoint());
    if (!origin) {
      return;
    }
    await this.apiKeys?.delete(origin);
    this.processManager.remote.reset();
    await this.processManager.remote.refresh(true);
    this.notifyChatModels();
    await this.pushState();
  }

  async copyDiagnostics(): Promise<void> {
    const target = this.store.getServerTarget();
    const text = formatRemoteDiagnostics(this.processManager.remote.current(), {
      url: target.baseUrl,
      label: target.label,
      hasApiKey: this.currentHasApiKey(),
    });
    await vscode.env.clipboard.writeText(text);
    void vscode.window.showInformationMessage("Connection diagnostics copied. The API key is not included.");
  }

  /**
   * Ask before start/reload when core's memory estimate says the load will spill.
   * The webview's live chart is display-only and is not consulted here.
   */
  async confirmIfMemorySpill(): Promise<boolean> {
    const { estimate: est, view } = await this.memoryForSaved(this.cpuOnlyBackend());
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

  /**
   * Rebuild and post the full sidebar state. Calls are coalesced: while one
   * push runs, every further caller shares a single trailing push, so a burst
   * of events (save + config echo + remote change) costs two rebuilds, not N,
   * and an older rebuild can never land after a newer one.
   */
  pushState(): Promise<void> {
    if (!this.pushInFlight) {
      this.pushInFlight = this.pushStateNow().finally(() => {
        this.pushInFlight = undefined;
      });
      return this.pushInFlight;
    }
    if (!this.pushTrailing) {
      this.pushTrailing = this.pushInFlight
        .catch(() => undefined)
        .then(() => {
          this.pushTrailing = undefined;
          return this.pushState();
        });
    }
    return this.pushTrailing;
  }

  private async pushStateNow(): Promise<void> {
    if (!this.view) {
      return;
    }
    // Paint Server ready/stopped before GPU probes and library scans. Reuse
    // that probe below instead of a second /health round-trip.
    const httpReady = await this.postStatusNow();
    this.syncSpeculativeMode();
    const remoteMode = this.store.isRemote();
    let state = this.store.getState();
    // Refresh GGUF caps when older state lacks size / arch dims needed for estimates.
    if (
      !remoteMode &&
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
    const localEntries = remoteMode ? [] : listLocalModelEntries(this.store.getConfig());
    const localModelCount = localEntries.length;
    const library = remoteMode ? null : await this.libraryPayload(state.selectedModelPath, status);
    const localSourceDirs = listActiveModelSourceDirs(this.store.getConfig(), localEntries);
    const localSources = localSourceDirs.map((s) => s.source);

    const caps = state.modelCapabilities;
    const build = this.installer.getInstalledInfoCached();
    const backendOptions = this.installer.getUiBackendOptions();
    // Active UI backend only — do not OR stale resolved/configured flags
    // (those stuck Load settings in CPU mode after switching back to Vulkan).
    const selectedUiBackend = (build.activeBackend ||
      build.resolvedBackend ||
      (build.configuredBackend === "auto" ? "vulkan" : build.configuredBackend)) as string;
    const cpuOnly = selectedUiBackend === "cpu";
    const { gpus, view: memoryView } = remoteMode
      ? { gpus: [] as GpuMemoryInfo[], view: undefined }
      : await this.memoryForSaved(cpuOnly);
    const draftCaps = resolveDraftCapabilities(state.loadSettings);
    const launched = this.processManager.getLaunchedConfig();
    const updateCheck = this.installer.peekUpdateCheck();

    this.post({
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
        copilotChatAvailable: isCopilotChatAvailable(),
        endpoint: this.store.getEndpoint(),
        localServers: await this.localServersForSidebar(status, httpReady),
        ...this.remoteFields(),
        binary,
        binaryExists: fs.existsSync(binary),
        modelsDir,
        localModelCount,
        library,
        localSources,
        localSourceDirs,
        modelName: friendlyModelTitle(caps?.name, state.selectedModelPath),
        modelNameRaw: displayModelTitle(caps?.name, state.selectedModelPath),
        changes: this.pendingChanges(),
        startedAt: launched?.startedAt,
        launchedSettings: launched?.loadSettings ?? null,
        defaults: DEFAULT_LOAD_SETTINGS,
        chatTemplateProblem: chatTemplateFileProblem(state.loadSettings) || "",
        build,
        backendOptions,
        selectedUiBackend,
        cpuOnly,
        launchMode: resolveLaunchMode(this.store.getConfig().get<string>("launchMode")),
        updateCheck,
        memoryView: memoryView ?? null,
        modeSampling: remoteMode
          ? describeModeSampling(
              undefined,
              [
                this.processManager.remote.current()?.modelId,
                this.processManager.remote.current()?.modelPath
                  ? path.basename(this.processManager.remote.current()?.modelPath || "")
                  : "",
              ]
                .filter(Boolean)
                .join(" ")
            )
          : describeModeSampling(caps, state.selectedModelPath),
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
    if (library) {
      void this.postLibraryFits(
        [...library.groups, ...library.hiddenGroups].flatMap((group) => group.quants.map((quant) => quant.path)),
        cpuOnly
      );
    }

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

  /** Live llama.cpp install/upgrade progress for the sidebar card. */
  postLlamaInstall(payload: {
    status: "idle" | "running" | "done" | "error" | "cancelled";
    phase?: string;
    message?: string;
    error?: string;
    percent?: number;
    received?: number;
    total?: number;
    bytesPerSec?: number;
    etaSeconds?: number;
    fileIndex?: number;
    fileCount?: number;
    assetName?: string;
    releaseTag?: string;
  }): void {
    this.post({ type: "llamaInstall", payload });
  }

  /** Push live start/reload progress into the Server card without a full state rebuild. */
  postBootProgress(message: string): void {
    this.post({
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
    const status = await this.processManager.refreshStatus();
    const httpReady = knownHttpReady ?? (await this.processManager.isHttpReady());
    const ui = effectiveServerUiState({
      starting: status.starting,
      running: status.running,
      httpReady,
    });
    this.post({
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
        localServers: await this.localServersForSidebar(status, httpReady),
        perf: this.perf.get(),
        perfLines: this.perf.detailLines(),
        ...this.remoteFields(),
      },
    });
    return httpReady;
  }

  /** Update the DFlash draft hint immediately (before a full pushState finishes). */
  postDraftModelSelected(draftPath: string): void {
    this.post({
      type: "draftModelSelected",
      path: draftPath || "",
    });
  }

  /** Update the vision projector hint immediately (before a full pushState finishes). */
  postMmprojSelected(mmprojPath: string): void {
    this.post({
      type: "mmprojSelected",
      path: mmprojPath || "",
    });
  }

  /** Swap the backend hint without rebuilding the rest of the sidebar. */
  postUpdateCheck(check: ReturnType<LlamaInstaller["peekUpdateCheck"]>): void {
    this.post({
      type: "updateCheck",
      payload: check,
    });
  }

  /** Patch Performance tiles without a full sidebar rebuild (library scan, binary probe). */
  postPerf(): void {
    this.post({
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

  /** Attach a library DFlash GGUF as the draft for the current model instead of replacing it. */
  private async useDflashDraft(filePath: string): Promise<void> {
    const cur = this.store.getState().loadSettings;
    const mode = speculativeUsesNgram(cur.speculativeMode)
      ? "ngram-dflash"
      : cur.speculativeMode === "off" || speculativeUsesMtp(cur.speculativeMode)
        ? "dflash"
        : cur.speculativeMode;
    await this.store.updateLoadSettings({
      draftModelPath: filePath,
      speculativeMode: mode,
      maxDraftTokens: recommendedMaxDraftTokens(mode, cur.maxDraftTokens),
    });
    this.postDraftModelSelected(filePath);
    this.syncSpeculativeMode();
    await this.pushState();
    void vscode.window.showInformationMessage(`DFlash draft set to ${path.basename(filePath)}`);
  }

  private async libraryPayload(selectedPath: string, status: { running: boolean; starting?: boolean; modelPath?: string }) {
    const config = this.store.getConfig();
    const modelsDir = getModelsDir(config);
    const hidden = hiddenModelPaths(config);
    const present = hidden.filter((filePath) => fs.existsSync(filePath));
    if (present.length !== hidden.length) {
      void config.update("hiddenModels", present);
    }
    const hiddenSet = new Set(present);
    const all = listLocalModelEntries(config);
    const visible = withoutHiddenModels(all, present);
    const hiddenEntries = all.filter((entry) => hiddenSet.has(canonicalModelPath(entry.path)));
    const decorate = (groups: LibraryGroup[]) =>
      groups.map((group) => ({
        id: group.id,
        title: group.title,
        source: group.source,
        owned: group.owned,
        detail: group.detail,
        companions: group.companions.map((companion) => ({
          path: companion.path,
          name: companion.name,
          role: companion.role,
          sizeBytes: companion.sizeBytes,
          sizeLabel: formatModelSize(companion.sizeBytes),
          shared: companion.shared,
        })),
        quants: group.quants.map((quant) => ({
          path: quant.path,
          label: quant.label,
          sizeBytes: quant.sizeBytes,
          sizeLabel: formatModelSize(quant.sizeBytes),
          shardCount: quant.shardCount,
          source: quant.source,
          owned: quant.owned,
          current: sameModelFile(quant.path, selectedPath),
          loaded: !!(
            status.running &&
            (sameModelFile(quant.path, status.modelPath) ||
              (!status.modelPath && sameModelFile(quant.path, selectedPath)))
          ),
        })),
      }));
    const [groups, hiddenGroups] = await Promise.all([
      buildLibraryGroupsAsync(visible, modelsDir),
      buildLibraryGroupsAsync(hiddenEntries, modelsDir),
    ]);
    const partials = listPartialDownloads(modelsDir);
    const partialBytes = partials.reduce((sum, file) => sum + file.sizeBytes, 0);
    return {
      groups: decorate(groups),
      hiddenGroups: decorate(hiddenGroups),
      partials: partials.map((file) => ({
        path: file.path,
        sizeBytes: file.sizeBytes,
        name: path.basename(file.path),
        sizeLabel: formatModelSize(file.sizeBytes),
      })),
      partialBytes,
      partialSizeLabel: formatModelSize(partialBytes),
      serverStarting: !!status.starting,
      recent: [selectedPath, ...(config.get<string[]>("recentModels") || [])].filter(
        (p, i, all) => !!p && fs.existsSync(p) && all.findIndex((q) => sameModelFile(q, p)) === i
      ),
    };
  }

  private async postLibraryFits(paths: string[], cpuOnly: boolean): Promise<void> {
    // Every state push used to re-run the fit pass for every quant. Skip it
    // when nothing the verdicts depend on changed (the webview keeps the last
    // badges). Used VRAM is bucketed so normal jitter does not count.
    const state = this.store.getState();
    const bucket = 512 * 1024 ** 2;
    const signature = JSON.stringify([
      paths,
      state.loadSettings,
      state.selectedModelPath,
      cpuOnly,
      cpuOnly ? [] : this.lastGpus.map((g) => [g.totalBytes, Math.round((g.usedBytes || 0) / bucket)]),
    ]);
    const now = Date.now();
    if (this.libraryFitsPosted && this.libraryFitsPosted.signature === signature && now - this.libraryFitsPosted.at < 60_000) {
      return;
    }
    const gen = ++this.libraryFitGen;
    const fits: Record<string, string> = {};
    const ratios: Record<string, number> = {};
    const drafts: Record<string, boolean> = {};
    let capacityBytes = 0;
    const settings = this.store.getState().loadSettings;
    for (const filePath of paths) {
      if (gen !== this.libraryFitGen) {
        return;
      }
      try {
        const caps = readPickerCapabilities(filePath);
        if (!caps) {
          continue;
        }
        if (isDflashDraftArchitecture(caps.architecture)) {
          drafts[filePath] = true;
        }
        const current = sameModelFile(filePath, this.store.getState().selectedModelPath);
        const load = clampLoadSettingsToModel(settings, caps);
        const { view } = computeMemoryView(
          caps,
          current ? load : { ...load, draftModelPath: "", mmprojPath: "" },
          cpuOnly ? [] : this.lastGpus,
          { cpuOnly }
        );
        // Same verdict as the Memory card: tight means under the 1.5 GiB
        // headroom, not merely under 4 GiB free.
        const level = view?.level;
        if (level === "good") {
          fits[filePath] = "fits";
        } else if (level === "tight") {
          fits[filePath] = "tight";
        } else if (level === "spill") {
          fits[filePath] = "wont-fit";
        }
        const devices = (view?.devices || []).filter((d) => (d.capacityBytes || 0) > 0);
        const cap = devices.reduce((sum, d) => sum + (d.capacityBytes || 0), 0);
        if (cap > 0) {
          ratios[filePath] = devices.reduce((sum, d) => sum + d.usedBytes, 0) / cap;
          capacityBytes = Math.max(capacityBytes, cap);
        }
      } catch {
        // Leave the badge blank when the GGUF header cannot be read.
      }
      await new Promise((resolve) => setImmediate(resolve));
    }
    if (gen !== this.libraryFitGen || !this.view) {
      return;
    }
    this.libraryFitsPosted = { signature, at: now };
    this.post({
      type: "libraryFits",
      fits,
      ratios,
      drafts,
      budgetLabel: capacityBytes > 0 ? `${formatModelSize(capacityBytes)} ${cpuOnly ? "RAM" : "VRAM"}` : "",
    });
  }

  /** Palette and the status-bar chip land in this list instead of a second picker. */
  focusLibrary(opts: { filter?: boolean; downloads?: boolean; query?: string; search?: boolean } = {}): void {
    this.libraryFocus = opts;
    void vscode.commands.executeCommand("llamaAio.settingsView.focus").then(() => {
      if (this.libraryReady) {
        this.flushLibraryFocus();
      }
    });
  }

  private flushLibraryFocus(): void {
    if (!this.view || !this.libraryFocus) {
      return;
    }
    this.post({ type: "focusLibrary", ...this.libraryFocus });
    this.libraryFocus = null;
  }

  private hf(): HuggingFaceClient {
    if (!this.hfClient) {
      this.hfClient = new HuggingFaceClient(this.store);
    }
    return this.hfClient;
  }

  private bindDownloads(): void {
    if (this.downloadsBound) {
      return;
    }
    this.downloadsBound = true;
    downloadManager.subscribe((jobs) => this.postLibraryDownloads(jobs));
  }

  private postLibraryDownloads(jobs: DownloadJobSnapshot[]): void {
    this.latestJobs = jobs;
    const sig = jobs.map((job) => job.id + ":" + job.state).join("|");
    if (sig !== this.postedJobState) {
      this.postedJobState = sig;
      if (this.downloadPostTimer) {
        clearTimeout(this.downloadPostTimer);
        this.downloadPostTimer = undefined;
      }
      this.flushDownloadPost();
      return;
    }
    if (this.downloadPostTimer) {
      return;
    }
    this.downloadPostTimer = setTimeout(() => {
      this.downloadPostTimer = undefined;
      this.flushDownloadPost();
    }, 300);
  }

  private flushDownloadPost(): void {
    if (!this.view) {
      return;
    }
    const jobs = this.latestJobs
      .filter((job) => job.state !== "done" && job.state !== "cancelled")
      .map((job) => ({
        id: job.id,
        label: job.label,
        received: job.received,
        total: job.total,
        bytesPerSec: job.bytesPerSec,
        etaSeconds: job.etaSeconds,
        state: job.state,
        error: job.error || "",
        pageUrl: job.pageUrl || "",
      }));
    this.post({ type: "libraryDownloads", jobs });
  }

  private async searchLibraryHf(query: string, requestId: number): Promise<void> {
    const q = query.trim();
    if (q.length < 2 || !this.view) {
      return;
    }
    try {
      const models = await this.hf().searchGgufModels(q, 8);
      const licenses = await this.hf().enrichLicenses(models);
      const repos = models.map((model) => {
        const license = licenses.get(model.id) || licenseFromTags(model.tags);
        return {
          id: model.id,
          downloads: model.downloads,
          badge: license.badge,
          needsConfirm: license.needsConfirm,
          summary: license.summary,
          licenseUrl: resolveLicenseUrl(model.id, license) || "",
        };
      });
      this.post({ type: "libraryHfResults", requestId, query: q, repos });
    } catch (e) {
      this.post({
        type: "libraryHfError",
        requestId,
        query: q,
        message: e instanceof Error ? e.message : String(e),
      });
    }
  }

  private async listLibraryHfFiles(modelId: string, requestId: number): Promise<void> {
    if (!modelId || !this.view) {
      return;
    }
    try {
      const files = await this.hf().listGgufFiles(modelId);
      const picker = downloadPickerFiles(files);
      this.post({
        type: "libraryHfFiles",
        requestId,
        modelId,
        companionOnly: picker.companionOnly,
        companionHint: picker.companionOnly ? "" : companionDownloadHint(files),
        files: picker.files.map((file) => ({
          path: file.path,
          label: listingBaseName(file.path),
          detail: describeLanguageGgufFile(file, files),
        })),
      });
    } catch (e) {
      this.post({
        type: "libraryHfError",
        requestId,
        query: "",
        message: e instanceof Error ? e.message : String(e),
      });
    }
  }

  private async runLibraryDownload(modelId: string, filePath: string): Promise<void> {
    if (!modelId || !filePath) {
      return;
    }
    try {
      const dest = await downloadGgufSelection(this.hf(), this.store, modelId, filePath);
      invalidateModelLibraryCache();
      if (dest) {
        await this.modelActions.selectModel(dest);
      } else {
        await this.pushState();
      }
    } catch (e) {
      if (e instanceof DownloadAbortError || e instanceof GatedDownloadError) {
        return;
      }
      vscode.window.showErrorMessage(
        `Download failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  }

  private async trashOwnedFile(filePath: string): Promise<void> {
    const uri = vscode.Uri.file(filePath);
    try {
      await vscode.workspace.fs.delete(uri, { useTrash: true });
    } catch {
      await fs.promises.rm(filePath, { force: true });
    }
  }

  private async removeEmptyOwnedDir(dir: string, modelsDir: string): Promise<void> {
    const dirReal = canonicalModelPath(dir);
    const rootReal = canonicalModelPath(modelsDir);
    if (!dirReal || dirReal === rootReal || !isOwnedModelPath(dir, modelsDir)) {
      return;
    }
    let names: string[] = [];
    try {
      names = fs.readdirSync(dirReal);
    } catch {
      return;
    }
    if (names.length) {
      return;
    }
    try {
      await fs.promises.rmdir(dirReal);
    } catch {
      // Another file appeared, or the directory is busy.
    }
  }

  async hideLibraryPath(filePath: string): Promise<void> {
    const canon = canonicalModelPath(filePath);
    if (!canon || !fs.existsSync(canon)) {
      vscode.window.showWarningMessage("Llama AIO: That model file is not on disk.");
      return;
    }
    const config = this.store.getConfig();
    const hidden = hiddenModelPaths(config);
    if (!hidden.includes(canon)) {
      await config.update("hiddenModels", [...hidden, canon]);
    }
    await this.pushState();
  }

  async unhideLibraryPath(filePath: string): Promise<void> {
    const canon = canonicalModelPath(filePath);
    if (!canon) {
      return;
    }
    const config = this.store.getConfig();
    const hidden = hiddenModelPaths(config).filter((entry) => entry !== canon);
    await config.update("hiddenModels", hidden);
    await this.pushState();
  }

  async hideSelectedModel(): Promise<void> {
    const selected = this.store.getState().selectedModelPath;
    if (!selected) {
      vscode.window.showInformationMessage("Llama AIO: Select a model first.");
      return;
    }
    await this.hideLibraryPath(selected);
    void vscode.window.showInformationMessage(
      "Hidden from the library. Use Show hidden in the sidebar to bring it back."
    );
  }

  /**
   * Delete an owned quant (and any checked companions). The webview confirm
   * dialog is the prompt; this re-checks every path and stops the server when
   * a mapped file is in the set.
   */
  async removeOwnedQuant(quantPath: string, companionPaths: string[]): Promise<void> {
    const trimmed = quantPath.trim();
    if (!trimmed) {
      return;
    }
    const modelsDir = getModelsDir(this.store.getConfig());
    const plan = removalPathsForQuant(trimmed, modelsDir, companionPaths);
    const shardSet = new Set(listShardPaths(trimmed).map((filePath) => canonicalModelPath(filePath)));
    const shardRejected = plan.rejected.some((filePath) => shardSet.has(canonicalModelPath(filePath)));
    if (shardRejected || !plan.files.some((filePath) => shardSet.has(filePath))) {
      vscode.window.showWarningMessage(
        "Llama AIO only removes files inside its own models folder. Use Hide for models found in other apps, or delete them in your file manager."
      );
      return;
    }
    const status = this.processManager.getStatus();
    if (status.starting) {
      vscode.window.showWarningMessage("Llama AIO: the server is still starting. Wait, then remove the model.");
      return;
    }
    const state = this.store.getState();
    const hits = (candidate: string | undefined) =>
      !!candidate && plan.files.some((filePath) => sameModelFile(filePath, canonicalModelPath(candidate)));
    // Resolve aliases before unlinking: afterward realpath cannot resolve a
    // selected path reached through a symlinked models directory.
    const clearModel = hits(state.selectedModelPath);
    const clearMm = hits(state.loadSettings.mmprojPath);
    const clearDraft = hits(state.loadSettings.draftModelPath);
    const mapped =
      hits(status.modelPath) ||
      clearModel || clearMm || clearDraft;
    if (status.running && mapped) {
      await this.processManager.stop(true);
    }
    for (const filePath of plan.files) {
      await this.trashOwnedFile(filePath);
    }
    await this.removeEmptyOwnedDir(path.dirname(plan.files[0] || trimmed), modelsDir);
    if (clearModel || clearMm || clearDraft) {
      await this.store.setState({
        ...(clearModel
          ? { selectedModelPath: "", modelCapabilities: undefined, modelMaxContext: undefined }
          : {}),
        loadSettings: {
          ...state.loadSettings,
          ...(clearMm ? { mmprojPath: "" } : {}),
          ...(clearDraft ? { draftModelPath: "" } : {}),
        },
      });
    }
    invalidateModelLibraryCache();
    await this.pushState();
  }

  async removeSelectedModel(): Promise<void> {
    const state = this.store.getState();
    const selected = state.selectedModelPath;
    if (!selected) {
      vscode.window.showInformationMessage("Llama AIO: Select a model first.");
      return;
    }
    const modelsDir = getModelsDir(this.store.getConfig());
    const plan = removalPathsForQuant(selected, modelsDir, []);
    if (!plan.files.length || plan.rejected.length) {
      vscode.window.showWarningMessage(
        "Llama AIO only removes files inside its own models folder. Use Hide for models found in other apps, or delete them in your file manager."
      );
      return;
    }
    const status = this.processManager.getStatus();
    const loaded =
      status.running &&
      (sameModelFile(status.modelPath, selected) ||
        (!status.modelPath && sameModelFile(state.selectedModelPath, selected)));
    const extra = plan.files.length > 1 ? ` and ${plan.files.length - 1} other shard${plan.files.length > 2 ? "s" : ""}` : "";
    const choice = await vscode.window.showWarningMessage(
      `Remove ${path.basename(selected)}${extra}? Files go to the trash when the system allows it.`,
      { modal: true },
      loaded ? "Stop and remove" : "Remove"
    );
    if (choice !== "Remove" && choice !== "Stop and remove") {
      return;
    }
    await this.removeOwnedQuant(selected, []);
  }

  async cleanPartialDownloads(): Promise<void> {
    const modelsDir = getModelsDir(this.store.getConfig());
    const partials = listPartialDownloads(modelsDir).filter((file) =>
      isOwnedModelPath(file.path, modelsDir) && !downloadManager.isPartialInUse(file.path)
    );
    if (!partials.length) {
      vscode.window.showInformationMessage("No unused incomplete downloads in the Llama AIO models folder.");
      return;
    }
    const bytes = partials.reduce((sum, file) => sum + file.sizeBytes, 0);
    const choice = await vscode.window.showWarningMessage(
      `Remove ${partials.length} incomplete download${partials.length === 1 ? "" : "s"} (${formatModelSize(bytes)})? They go to the trash when the system allows it.`,
      { modal: true },
      "Clean"
    );
    if (choice !== "Clean") {
      return;
    }
    for (const file of partials) {
      await downloadManager.removePartialIfIdle(file.path, async () => {
        // Ownership and job activity may have changed while confirmation was open.
        if (isOwnedModelPath(file.path, modelsDir)) await this.trashOwnedFile(file.path);
      });
    }
    invalidateModelLibraryCache();
    await this.pushState();
  }

  /**
   * Sidebar page: src/webview/sidebar.{html,css,ts}, built into out/webview by
   * esbuild.mjs. The markup is a static template; the few values that come
   * from core (starter model, built-in chat templates) are filled in here.
   */
  getHtml(webview: vscode.Webview): string {
    const dir = vscode.Uri.joinPath(this.extensionUri, "out", "webview");
    const nonce = getNonce();
    const values: Record<string, string> = {
      cspSource: webview.cspSource,
      nonce,
      styleUri: webview.asWebviewUri(vscode.Uri.joinPath(dir, "sidebar.css")).toString(),
      scriptUri: webview.asWebviewUri(vscode.Uri.joinPath(dir, "sidebar.js")).toString(),
      starterLabel: escapeHtml(STARTER_MODEL.label),
      starterSize: escapeHtml(STARTER_MODEL.approxSizeLabel),
      starterDetail: escapeHtml(STARTER_MODEL.detail),
      chatTemplateOptions: chatTemplateNameOptions(),
    };
    return sidebarTemplate(vscode.Uri.joinPath(dir, "sidebar.html").fsPath).replace(
      /\{\{(\w+)\}\}/g,
      (match, key: string) => values[key] ?? match
    );
  }
}

let cachedTemplate: { file: string; text: string } | undefined;

function sidebarTemplate(file: string): string {
  if (cachedTemplate?.file !== file) {
    cachedTemplate = { file, text: fs.readFileSync(file, "utf8") };
  }
  return cachedTemplate.text;
}

function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function getNonce(): string {
  return randomBytes(16).toString("base64");
}

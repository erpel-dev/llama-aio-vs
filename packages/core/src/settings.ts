import * as path from "path";
import {
  AppConfig,
  ConfigAccessor,
  ConfigFile,
  DEFAULT_SERVER_PREFERENCES,
  ServerMode,
  ServerPreferences,
  toExtensionState,
} from "./config";
import { isLoopbackHost, isWildcardHost, urlHostname } from "./serverHttp";
import { Event } from "./events";
import { clampLoadSettingsToModel, isQwen4expArchitecture, readModelCapabilities } from "./ggufMetadata";
import { detectGpuMemory, detectGpus } from "./gpuInfo";
import { LlamaInstaller } from "./llamaInstaller";
import { resolveMmprojPath, resolveMtpDraftPath, findSiblingMtpDraft, isMtpSidecarFile } from "./modelLibrary";
import { recommendLoadSettings } from "./recommendSettings";
import {
  findLlamaServerOnPath,
  getDefaultPromptReplacementsPath,
  getLlamaServerBinary,
} from "./paths";
import {
  ChatTemplateChoice,
  DEFAULT_CHAT_TEMPLATE,
  ExtensionState,
  isDefaultChatTemplate,
  LlamaLoadSettings,
  normalizeChatTemplate,
  normalizeLoadSettings,
  normalizeRequestSettings,
  RequestSettings,
  speculativeUsesMtp,
} from "./types";

const RECENT_MODELS_MAX = 8;

/**
 * Everything the frontends read and write, backed by the shared config file so
 * the extension and the TUI always agree on model, tuning, and server address.
 */
export class SettingsStore {
  constructor(private readonly config: ConfigFile) {}

  /** Fires when any frontend (this one included) changes the configuration. */
  get onDidChange(): Event<AppConfig> {
    return this.config.onDidChange;
  }

  /** Fires only for changes written by another process. */
  get onDidChangeExternally(): Event<AppConfig> {
    return this.config.onDidChangeExternally;
  }

  get configPath(): string {
    return this.config.path;
  }

  getState(): ExtensionState {
    const raw = toExtensionState(this.config.getState());
    let loadSettings = normalizeLoadSettings(raw.loadSettings);
    const caps = raw.modelCapabilities;
    if (caps?.maxContextLength) {
      loadSettings = clampLoadSettingsToModel(loadSettings, caps);
    }
    return {
      selectedModelPath: raw.selectedModelPath || "",
      loadSettings,
      requestSettings: normalizeRequestSettings(raw.requestSettings),
      modelMaxContext: caps?.maxContextLength ?? raw.modelMaxContext,
      modelCapabilities: caps,
    };
  }

  /**
   * Re-read GGUF caps when MoE expert-share, PLE share, or other newer fields
   * are missing from a previously persisted modelCapabilities blob.
   */
  async refreshCapabilitiesIfStale(): Promise<boolean> {
    const state = this.getState();
    const path = (state.selectedModelPath || "").trim();
    const caps = state.modelCapabilities;
    if (!path || !caps) {
      return false;
    }
    const moeNeedsShare = caps.isMoe && (caps.moeExpertShare === undefined || !Number.isFinite(caps.moeExpertShare));
    const qwen4NeedsPle =
      isQwen4expArchitecture(caps.architecture) &&
      (caps.pleShare === undefined || !Number.isFinite(caps.pleShare));
    if (!moeNeedsShare && !qwen4NeedsPle) {
      return false;
    }
    try {
      const fresh = readModelCapabilities(path);
      await this.setState({
        modelCapabilities: fresh,
        modelMaxContext: fresh.maxContextLength,
        loadSettings: clampLoadSettingsToModel(state.loadSettings, fresh),
      });
      return true;
    } catch {
      return false;
    }
  }

  /** Effective tokens per parallel slot (what llama-server actually allows per request). */
  getSlotContextSize(load = this.getState().loadSettings): number {
    return Math.floor(load.contextLength / Math.max(1, load.maxConcurrentPredictions));
  }

  async setState(patch: Partial<ExtensionState>): Promise<ExtensionState> {
    await this.config.setState(patch);
    return this.getState();
  }

  async updateLoadSettings(patch: Partial<LlamaLoadSettings>): Promise<ExtensionState> {
    const state = this.getState();
    let loadSettings = normalizeLoadSettings({ ...state.loadSettings, ...patch });
    if (
      speculativeUsesMtp(patch.speculativeMode) &&
      !isMtpSidecarFile({ path: loadSettings.draftModelPath || "" }) &&
      !(state.modelCapabilities?.nextnPredictLayers && state.modelCapabilities.nextnPredictLayers > 0)
    ) {
      const sibling = findSiblingMtpDraft(state.selectedModelPath);
      if (sibling) {
        loadSettings = { ...loadSettings, draftModelPath: sibling };
      }
    }
    if (state.modelCapabilities) {
      loadSettings = clampLoadSettingsToModel(loadSettings, state.modelCapabilities);
    }
    const next = await this.setState({ loadSettings });
    if (patchTouchesChatTemplate(patch) && next.selectedModelPath) {
      await this.writeChatTemplateForModel(next.selectedModelPath, next.loadSettings);
    }
    return this.getState();
  }

  async updateRequestSettings(patch: Partial<RequestSettings>): Promise<ExtensionState> {
    return this.setState({
      requestSettings: normalizeRequestSettings({ ...this.getState().requestSettings, ...patch }),
    });
  }

  /**
   * Read GGUF metadata for a model, clamp settings, and persist capabilities.
   * When the selected path changes (or `recommendDefaults` is true), also apply
   * MoE/dense/CPU recommended context + offload defaults, and enable MTP when
   * the GGUF reports nextn_predict_layers > 0 or a sibling `mtp-*.gguf` is attached.
   */
  async applySelectedModel(
    modelPath: string,
    options?: { recommendDefaults?: boolean; cpuOnly?: boolean; attachMmproj?: boolean }
  ): Promise<ExtensionState> {
    const prev = this.getState();
    const caps = readModelCapabilities(modelPath);
    const pathChanged = !!modelPath && modelPath !== prev.selectedModelPath;
    const recommend = options?.recommendDefaults ?? pathChanged;

    let loadSettings = prev.loadSettings;
    const attachCompanions = options?.attachMmproj || pathChanged;
    loadSettings = {
      ...loadSettings,
      mmprojPath: resolveMmprojPath(modelPath, loadSettings.mmprojPath, attachCompanions),
      draftModelPath: resolveMtpDraftPath(
        modelPath,
        loadSettings.draftModelPath,
        attachCompanions,
        loadSettings.speculativeMode
      ),
    };
    if (recommend) {
      const installer = new LlamaInstaller(this);
      const cpuOnly = options?.cpuOnly ?? installer.resolveActiveUiBackend() === "cpu";
      const llamaBinary = cpuOnly
        ? undefined
        : installer.resolveActiveUiBackend() === "path"
          ? findLlamaServerOnPath()
          : getLlamaServerBinary(installer.getActiveInstallDir());
      loadSettings = recommendLoadSettings(loadSettings, caps, {
        cpuOnly,
        gpu: cpuOnly ? undefined : detectGpuMemory(),
        gpus: cpuOnly ? undefined : detectGpus(false, llamaBinary),
      });
    }
    loadSettings = clampLoadSettingsToModel(loadSettings, caps);
    if (pathChanged) {
      const choice = this.savedChatTemplate(modelPath);
      loadSettings = {
        ...loadSettings,
        chatTemplateSource: choice.source,
        chatTemplateName: choice.name,
        chatTemplateFile: choice.file,
        chatTemplateJinja: choice.jinja,
      };
    }

    const next = await this.setState({
      selectedModelPath: modelPath,
      modelCapabilities: caps,
      modelMaxContext: caps.maxContextLength,
      loadSettings,
    });
    if (pathChanged) {
      await this.rememberRecentModels([modelPath, prev.selectedModelPath]);
    }
    return next;
  }

  private savedChatTemplate(modelPath: string): ChatTemplateChoice {
    const map = this.config.get().app.chatTemplates || {};
    const key = path.resolve(modelPath);
    const raw = map[key] ?? map[modelPath];
    return raw ? normalizeChatTemplate(raw) : { ...DEFAULT_CHAT_TEMPLATE };
  }

  /** Remember the active choice for this model file. Default choices are removed. */
  private async writeChatTemplateForModel(
    modelPath: string,
    loadSettings: LlamaLoadSettings
  ): Promise<void> {
    const key = path.resolve(modelPath);
    const map = { ...(this.config.get().app.chatTemplates || {}) };
    const choice = normalizeChatTemplate({
      source: loadSettings.chatTemplateSource,
      name: loadSettings.chatTemplateName,
      file: loadSettings.chatTemplateFile,
      jinja: loadSettings.chatTemplateJinja,
    });
    if (isDefaultChatTemplate(choice)) {
      delete map[key];
    } else {
      map[key] = choice;
    }
    await this.getConfig().update("chatTemplates", map);
  }

  /** Newest first, one entry per file. */
  private async rememberRecentModels(newest: string[]): Promise<void> {
    const list = [...newest, ...(this.getConfig().get<string[]>("recentModels") || [])].filter(Boolean);
    const seen = new Set<string>();
    const out = list.filter((p) => {
      const key = path.resolve(p);
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
    await this.getConfig().update("recentModels", out.slice(0, RECENT_MODELS_MAX));
  }

  getConfig(): ConfigAccessor {
    return this.config.accessor();
  }

  getPort(): number {
    return this.getConfig().get<number>("port", 8742);
  }

  getHost(): string {
    return this.getConfig().get<string>("host", "127.0.0.1");
  }

  getServerPreferences(): ServerPreferences {
    return this.getConfig().get<ServerPreferences>("server", DEFAULT_SERVER_PREFERENCES);
  }

  isRemote(): boolean {
    return this.getServerTarget().mode === "remote";
  }

  /** The one switch every request goes through: which server, and how to label it. */
  getServerTarget(): ServerTarget {
    const server = this.getServerPreferences();
    if (server.mode === "remote" && server.remoteUrl) {
      return {
        mode: "remote",
        baseUrl: server.remoteUrl,
        label: server.remoteLabel || urlHostname(server.remoteUrl) || server.remoteUrl,
      };
    }
    return { mode: "local", baseUrl: localEndpoint(this.getHost(), this.getPort()), label: "this machine" };
  }

  getEndpoint(): string {
    return this.getServerTarget().baseUrl;
  }

  private async updateServer(patch: Partial<ServerPreferences>): Promise<void> {
    await this.getConfig().update("server", { ...this.getServerPreferences(), ...patch });
  }

  /** Switch to a remote server. `url` must already be normalized. */
  async setRemoteServer(url: string, label = ""): Promise<void> {
    const prev = this.getServerPreferences();
    await this.updateServer({
      mode: "remote",
      remoteUrl: url,
      remoteLabel: label.trim(),
      lastRemoteModelId: prev.remoteUrl === url ? prev.lastRemoteModelId : "",
    });
  }

  async setServerMode(mode: ServerMode): Promise<void> {
    const prev = this.getServerPreferences();
    await this.updateServer({ mode: mode === "remote" && prev.remoteUrl ? "remote" : "local" });
  }

  async setLastRemoteModelId(id: string): Promise<void> {
    if (id && this.getServerPreferences().lastRemoteModelId !== id) {
      await this.updateServer({ lastRemoteModelId: id });
    }
  }

  isPlainHttpConfirmed(url: string): boolean {
    return this.getServerPreferences().plainHttpConfirmed.includes(url);
  }

  async confirmPlainHttp(url: string): Promise<void> {
    if (!this.isPlainHttpConfirmed(url)) {
      const prev = this.getServerPreferences();
      await this.updateServer({ plainHttpConfirmed: [...prev.plainHttpConfirmed, url] });
    }
  }

  /**
   * Older configs pointed `host` at another machine to reach a remote server.
   * Turn that into remote mode once. Returns the remote URL when it migrated.
   */
  async migrateLegacyRemoteHost(localAddresses: string[]): Promise<string | undefined> {
    const remoteUrl = legacyRemoteHostUrl(this.getHost(), this.getPort(), localAddresses);
    if (!remoteUrl || this.getServerPreferences().remoteUrl) {
      return undefined;
    }
    await this.getConfig().update("host", "127.0.0.1");
    await this.updateServer({ mode: "remote", remoteUrl, remoteLabel: "" });
    return remoteUrl;
  }

  /** Put `host` back after the user chose Undo on the migration toast. */
  async undoLegacyRemoteHostMigration(remoteUrl: string): Promise<void> {
    const u = new URL(remoteUrl);
    await this.updateServer({ ...DEFAULT_SERVER_PREFERENCES });
    await this.getConfig().update("host", u.hostname.replace(/^\[|\]$/g, ""));
  }

  isPromptReplacementsEnabled(): boolean {
    return this.getConfig().get<boolean>("promptReplacementsEnabled", true);
  }

  isWikipediaLookupEnabled(): boolean {
    return this.getConfig().get<boolean>("wikipediaLookupEnabled", false);
  }

  isDuplicateToolCallGuardEnabled(): boolean {
    return this.getConfig().get<boolean>("duplicateToolCallGuardEnabled", false);
  }

  /** Configured replacements file, falling back to the bundled defaults. */
  getPromptReplacementsFile(): string {
    const override = (this.getConfig().get<string>("promptReplacementsFile", "") || "").trim();
    return override || getDefaultPromptReplacementsPath();
  }
}

export interface ServerTarget {
  mode: ServerMode;
  /** Base URL without a trailing slash (may carry a path prefix). */
  baseUrl: string;
  /** "gpu-box" for remotes, "this machine" locally. */
  label: string;
}

function bracketIpv6(host: string): string {
  return host.includes(":") && !host.startsWith("[") ? `[${host}]` : host;
}

/** Where to connect for a local server bound to `host`. A wildcard bind is reached via loopback. */
export function localEndpoint(host: string, port: number): string {
  const h = host.trim().replace(/^\[|\]$/g, "");
  const connect = h === "::" ? "::1" : isWildcardHost(h) ? "127.0.0.1" : h;
  return `http://${bracketIpv6(connect)}:${port}`;
}

/**
 * Remote URL implied by a legacy `host`, or undefined when that host is this
 * machine (loopback, wildcard bind, or one of `localAddresses`).
 */
export function legacyRemoteHostUrl(host: string, port: number, localAddresses: string[]): string | undefined {
  const h = (host || "").trim().replace(/^\[|\]$/g, "").toLowerCase();
  if (!h || isLoopbackHost(h) || isWildcardHost(h)) {
    return undefined;
  }
  const local = new Set(localAddresses.map((a) => a.toLowerCase().replace(/%.*$/, "")));
  if (local.has(h)) {
    return undefined;
  }
  return `http://${bracketIpv6(h)}:${port}`;
}

export {
  buildServerArgs,
  chatTemplateFileProblem,
  serverConfigFingerprint,
  normalizeLoadSettingsForCpuBackend,
} from "./serverArgs";

const CHAT_TEMPLATE_PATCH_KEYS = [
  "chatTemplateSource",
  "chatTemplateName",
  "chatTemplateFile",
  "chatTemplateJinja",
] as const;

function patchTouchesChatTemplate(patch: Partial<LlamaLoadSettings>): boolean {
  return CHAT_TEMPLATE_PATCH_KEYS.some((key) => Object.prototype.hasOwnProperty.call(patch, key));
}

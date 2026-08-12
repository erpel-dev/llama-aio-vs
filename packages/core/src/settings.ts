import { AppConfig, ConfigAccessor, ConfigFile, toExtensionState } from "./config";
import { Event } from "./events";
import { clampLoadSettingsToModel, readModelCapabilities } from "./ggufMetadata";
import { detectGpuMemory } from "./gpuInfo";
import { LlamaInstaller } from "./llamaInstaller";
import { recommendLoadSettings } from "./recommendSettings";
import { getDefaultPromptReplacementsPath } from "./paths";
import {
  ExtensionState,
  LlamaLoadSettings,
  normalizeLoadSettings,
  normalizeRequestSettings,
  RequestSettings,
} from "./types";

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
   * Re-read GGUF caps when MoE expert-share (or other newer fields) are missing
   * from a previously persisted modelCapabilities blob.
   */
  async refreshCapabilitiesIfStale(): Promise<boolean> {
    const state = this.getState();
    const path = (state.selectedModelPath || "").trim();
    const caps = state.modelCapabilities;
    if (!path || !caps?.isMoe) {
      return false;
    }
    if (caps.moeExpertShare !== undefined && Number.isFinite(caps.moeExpertShare)) {
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
    if (state.modelCapabilities) {
      loadSettings = clampLoadSettingsToModel(loadSettings, state.modelCapabilities);
    }
    return this.setState({ loadSettings });
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
   * the GGUF reports nextn_predict_layers > 0.
   */
  async applySelectedModel(
    modelPath: string,
    options?: { recommendDefaults?: boolean; cpuOnly?: boolean }
  ): Promise<ExtensionState> {
    const prev = this.getState();
    const caps = readModelCapabilities(modelPath);
    const pathChanged = !!modelPath && modelPath !== prev.selectedModelPath;
    const recommend = options?.recommendDefaults ?? pathChanged;

    let loadSettings = prev.loadSettings;
    if (recommend) {
      const cpuOnly =
        options?.cpuOnly ?? new LlamaInstaller(this).resolveActiveUiBackend() === "cpu";
      loadSettings = recommendLoadSettings(loadSettings, caps, {
        cpuOnly,
        gpu: cpuOnly ? undefined : detectGpuMemory(),
      });
    }
    loadSettings = clampLoadSettingsToModel(loadSettings, caps);

    return this.setState({
      selectedModelPath: modelPath,
      modelCapabilities: caps,
      modelMaxContext: caps.maxContextLength,
      loadSettings,
    });
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

  getEndpoint(): string {
    return `http://${this.getHost()}:${this.getPort()}`;
  }

  isPromptReplacementsEnabled(): boolean {
    return this.getConfig().get<boolean>("promptReplacementsEnabled", true);
  }

  /** Configured replacements file, falling back to the bundled defaults. */
  getPromptReplacementsFile(): string {
    const override = (this.getConfig().get<string>("promptReplacementsFile", "") || "").trim();
    return override || getDefaultPromptReplacementsPath();
  }
}

export { buildServerArgs, serverConfigFingerprint, normalizeLoadSettingsForCpuBackend } from "./serverArgs";

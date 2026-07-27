import * as vscode from "vscode";
import { clampLoadSettingsToModel, readModelCapabilities } from "./ggufMetadata";
import { detectGpuMemory } from "./gpuInfo";
import { LlamaInstaller } from "./llamaInstaller";
import { recommendLoadSettings } from "./recommendSettings";
import {
  DEFAULT_LOAD_SETTINGS,
  DEFAULT_REQUEST_SETTINGS,
  ExtensionState,
  LlamaLoadSettings,
  RequestSettings,
} from "./types";

const STATE_KEY = "llamaAio.state";

export class SettingsStore {
  constructor(private readonly context: vscode.ExtensionContext) {}

  getState(): ExtensionState {
    const raw = this.context.globalState.get<Partial<ExtensionState>>(STATE_KEY) || {};
    let loadSettings = { ...DEFAULT_LOAD_SETTINGS, ...(raw.loadSettings || {}) };
    const caps = raw.modelCapabilities;
    if (caps?.maxContextLength) {
      loadSettings = clampLoadSettingsToModel(loadSettings, caps);
    }
    return {
      selectedModelPath: raw.selectedModelPath || "",
      loadSettings,
      requestSettings: { ...DEFAULT_REQUEST_SETTINGS, ...(raw.requestSettings || {}) },
      modelMaxContext: caps?.maxContextLength ?? raw.modelMaxContext,
      modelCapabilities: caps,
    };
  }

  /** Effective tokens per parallel slot (what llama-server actually allows per request). */
  getSlotContextSize(load = this.getState().loadSettings): number {
    return Math.floor(load.contextLength / Math.max(1, load.maxConcurrentPredictions));
  }

  /**
   * One-time migration: older defaults used -np 4 with 8k ctx → 2k/slot, which
   * breaks Copilot Chat agent prompts. Bump to a single large slot.
   */
  async migrateChatContextIfNeeded(): Promise<boolean> {
    const migrated = this.context.globalState.get<boolean>("llamaAio.migratedChatContextV1");
    if (migrated) {
      return false;
    }
    const state = this.getState();
    const slot = this.getSlotContextSize(state.loadSettings);
    let changed = false;
    const patch: Partial<typeof state.loadSettings> = {};
    if (state.loadSettings.maxConcurrentPredictions > 1) {
      patch.maxConcurrentPredictions = 1;
      changed = true;
    }
    if (slot < 32768 || (state.loadSettings.contextLength || 0) < 32768) {
      patch.contextLength = Math.max(65536, state.loadSettings.contextLength || 0);
      changed = true;
    }
    if (changed) {
      await this.updateLoadSettings(patch);
    }
    await this.context.globalState.update("llamaAio.migratedChatContextV1", true);
    return changed;
  }

  async setState(patch: Partial<ExtensionState>): Promise<ExtensionState> {
    const next: ExtensionState = {
      ...this.getState(),
      ...patch,
      loadSettings: {
        ...this.getState().loadSettings,
        ...(patch.loadSettings || {}),
      },
      requestSettings: {
        ...this.getState().requestSettings,
        ...(patch.requestSettings || {}),
      },
    };
    await this.context.globalState.update(STATE_KEY, next);
    return next;
  }

  async updateLoadSettings(patch: Partial<LlamaLoadSettings>): Promise<ExtensionState> {
    const state = this.getState();
    let loadSettings = { ...state.loadSettings, ...patch };
    if (state.modelCapabilities) {
      loadSettings = clampLoadSettingsToModel(loadSettings, state.modelCapabilities);
    }
    return this.setState({ loadSettings });
  }

  async updateRequestSettings(patch: Partial<RequestSettings>): Promise<ExtensionState> {
    return this.setState({
      requestSettings: { ...this.getState().requestSettings, ...patch },
    });
  }

  /**
   * Read GGUF metadata for a model, clamp settings, and persist capabilities.
   * When the selected path changes (or `recommendDefaults` is true), also apply
   * MoE/dense/CPU recommended context + offload defaults.
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

  getConfig(): vscode.WorkspaceConfiguration {
    return vscode.workspace.getConfiguration("llamaAio");
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
}

export { buildServerArgs } from "./serverArgs";

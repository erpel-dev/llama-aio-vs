import * as fs from "fs";
import * as path from "path";
import { Emitter } from "./events";
import { getConfigPath, getDataRoot } from "./paths";
import {
  ChatTemplateChoice,
  ExtensionState,
  isDefaultChatTemplate,
  LlamaLoadSettings,
  normalizeChatTemplate,
  normalizeLoadSettings,
  normalizeRequestSettings,
  RequestSettings,
} from "./types";
import type { ModelCapabilities } from "./ggufMetadata";

export const CONFIG_VERSION = 1;

export type ServerMode = "local" | "remote";

/** Which llama-server the frontends talk to. Local `host`/`port` stay the bind address. */
export interface ServerPreferences {
  mode: ServerMode;
  /** Normalized `http(s)://host[:port][/prefix]`; empty until a remote is set. */
  remoteUrl: string;
  /** Short name shown in the sidebar and status bar; empty = URL host. */
  remoteLabel: string;
  /** Last model id the remote reported, so the Copilot picker stays stable while offline. */
  lastRemoteModelId: string;
  /** Plain-http URLs to public addresses the user confirmed once. */
  plainHttpConfirmed: string[];
}

export const DEFAULT_SERVER_PREFERENCES: ServerPreferences = {
  mode: "local",
  remoteUrl: "",
  remoteLabel: "",
  lastRemoteModelId: "",
  plainHttpConfirmed: [],
};

/**
 * Frontend-agnostic preferences. These used to live in VS Code settings under
 * `llamaAio.*`; they now live in the shared config file so the TUI and the
 * extension drive the same server.
 */
export interface AppPreferences {
  port: number;
  host: string;
  /** Empty means `<dataRoot>/llama.cpp`. */
  installDir: string;
  /** Empty means `<dataRoot>/models`. */
  modelsDir: string;
  extraModelDirs: string[];
  hfToken: string;
  autoStart: boolean;
  backend: string;
  launchMode: string;
  promptReplacementsEnabled: boolean;
  /** Empty means the defaults bundled with the core package. */
  promptReplacementsFile: string;
  /** Off by default: Copilot may call wikipedia_lookup for encyclopedic facts. */
  wikipediaLookupEnabled: boolean;
  /**
   * Off by default. When true, skip a Copilot tool that already ran with the
   * same arguments in this turn. Can block a legitimate retry (e.g. Wikipedia).
   */
  duplicateToolCallGuardEnabled: boolean;
  /**
   * Real paths omitted from the model library and the quick pick.
   * The files stay on disk. Discovered models (LM Studio, HF cache) are hidden
   * this way instead of deleted.
   */
  hiddenModels: string[];
  /** Most recently selected model paths, newest first (quick switch in the sidebar). */
  recentModels: string[];
  /**
   * Chat-template override per model file (`path.resolve` key).
   * Missing entry means the GGUF's own Jinja template.
   */
  chatTemplates: Record<string, ChatTemplateChoice>;
  server: ServerPreferences;
}

/** Model selection plus the tuning that gets turned into llama-server flags. */
export interface ModelState {
  selectedModelPath: string;
  loadSettings: LlamaLoadSettings;
  requestSettings: RequestSettings;
  modelMaxContext?: number;
  modelCapabilities?: ModelCapabilities;
}

export interface AppConfig {
  version: number;
  app: AppPreferences;
  state: ModelState;
}

export const DEFAULT_APP_PREFERENCES: AppPreferences = {
  port: 8742,
  host: "127.0.0.1",
  installDir: "",
  modelsDir: "",
  extraModelDirs: [],
  hfToken: "",
  autoStart: false,
  backend: "auto",
  launchMode: "externalTerminal",
  promptReplacementsEnabled: true,
  promptReplacementsFile: "",
  wikipediaLookupEnabled: false,
  duplicateToolCallGuardEnabled: false,
  hiddenModels: [],
  recentModels: [],
  chatTemplates: {},
  server: DEFAULT_SERVER_PREFERENCES,
};

function normalizeServerPreferences(raw: unknown): ServerPreferences {
  const src = (raw && typeof raw === "object" ? raw : {}) as Partial<ServerPreferences>;
  const str = (v: unknown) => (typeof v === "string" ? v.trim() : "");
  const remoteUrl = str(src.remoteUrl);
  return {
    mode: src.mode === "remote" && remoteUrl ? "remote" : "local",
    remoteUrl,
    remoteLabel: str(src.remoteLabel),
    lastRemoteModelId: str(src.lastRemoteModelId),
    plainHttpConfirmed: Array.isArray(src.plainHttpConfirmed)
      ? [...new Set(src.plainHttpConfirmed.filter((u): u is string => typeof u === "string" && !!u.trim()))]
      : [],
  };
}

/**
 * Read/write access to {@link AppPreferences}, shaped like
 * `vscode.WorkspaceConfiguration` so path and installer helpers work with
 * either without a translation layer.
 */
export interface ConfigAccessor {
  get<T>(key: string): T | undefined;
  get<T>(key: string, fallback: T): T;
  update(key: string, value: unknown): Promise<void>;
}

function normalizePreferences(raw: Partial<AppPreferences> | undefined): AppPreferences {
  const src = raw || {};
  const str = (value: unknown, fallback: string): string =>
    typeof value === "string" ? value : fallback;
  const port = Number(src.port);
  return {
    port: Number.isFinite(port) && port > 0 && port < 65536 ? Math.round(port) : 8742,
    host: str(src.host, DEFAULT_APP_PREFERENCES.host) || DEFAULT_APP_PREFERENCES.host,
    installDir: str(src.installDir, ""),
    modelsDir: str(src.modelsDir, ""),
    extraModelDirs: Array.isArray(src.extraModelDirs)
      ? src.extraModelDirs.filter((d): d is string => typeof d === "string")
      : [],
    hfToken: str(src.hfToken, ""),
    autoStart: src.autoStart === true,
    backend: str(src.backend, DEFAULT_APP_PREFERENCES.backend) || DEFAULT_APP_PREFERENCES.backend,
    launchMode:
      str(src.launchMode, DEFAULT_APP_PREFERENCES.launchMode) || DEFAULT_APP_PREFERENCES.launchMode,
    promptReplacementsEnabled: src.promptReplacementsEnabled !== false,
    promptReplacementsFile: str(src.promptReplacementsFile, ""),
    wikipediaLookupEnabled: src.wikipediaLookupEnabled === true,
    duplicateToolCallGuardEnabled: src.duplicateToolCallGuardEnabled === true,
    hiddenModels: Array.isArray(src.hiddenModels)
      ? src.hiddenModels.filter((p): p is string => typeof p === "string" && p.trim().length > 0)
      : [],
    recentModels: Array.isArray(src.recentModels)
      ? src.recentModels.filter((p): p is string => typeof p === "string" && p.trim().length > 0)
      : [],
    chatTemplates: normalizeChatTemplateMap(src.chatTemplates),
    server: normalizeServerPreferences(src.server),
  };
}

/** Drop default and invalid entries. Keys are absolute paths. */
function normalizeChatTemplateMap(raw: unknown): Record<string, ChatTemplateChoice> {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return {};
  }
  const out: Record<string, ChatTemplateChoice> = {};
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    const trimmed = key.trim();
    if (!trimmed) {
      continue;
    }
    const choice = normalizeChatTemplate(value);
    if (isDefaultChatTemplate(choice)) {
      continue;
    }
    out[path.resolve(trimmed)] = choice;
  }
  return out;
}

export function normalizeConfig(raw: Partial<AppConfig> | undefined): AppConfig {
  const src = raw || {};
  const state: Partial<ModelState> = src.state || {};
  return {
    version: CONFIG_VERSION,
    app: normalizePreferences(src.app),
    state: {
      selectedModelPath:
        typeof state.selectedModelPath === "string" ? state.selectedModelPath : "",
      loadSettings: normalizeLoadSettings(state.loadSettings),
      requestSettings: normalizeRequestSettings(state.requestSettings),
      modelMaxContext: state.modelCapabilities?.maxContextLength ?? state.modelMaxContext,
      modelCapabilities: state.modelCapabilities,
    },
  };
}

/**
 * The shared `~/.llama-aio-vs/config.json`.
 *
 * Writes go through a temp file + rename so a crashed or concurrent writer can
 * never leave a half-written config behind, and the directory is watched so a
 * change made in one frontend shows up in the other without a restart.
 */
export class ConfigFile {
  private readonly file: string;
  private cache: AppConfig;
  /** Exact text of our own last write, so the watcher can ignore the echo. */
  private lastWritten = "";
  private watcher: fs.FSWatcher | undefined;
  private debounce: NodeJS.Timeout | undefined;

  private readonly _onDidChangeExternally = new Emitter<AppConfig>();
  /** Fires when *another* process changed the file on disk. */
  readonly onDidChangeExternally = this._onDidChangeExternally.event;

  private readonly _onDidChange = new Emitter<AppConfig>();
  /** Fires on every change, local writes included. */
  readonly onDidChange = this._onDidChange.event;

  constructor(file = getConfigPath()) {
    this.file = file;
    this.cache = this.readFromDisk();
  }

  get path(): string {
    return this.file;
  }

  private readFromDisk(): AppConfig {
    try {
      const text = fs.readFileSync(this.file, "utf8");
      this.lastWritten = text;
      return normalizeConfig(JSON.parse(text) as Partial<AppConfig>);
    } catch (err) {
      const code = (err as NodeJS.ErrnoException)?.code;
      if (code !== "ENOENT") {
        console.warn(
          `Llama AIO: could not read ${this.file}, falling back to defaults:`,
          err instanceof Error ? err.message : err
        );
      }
      return normalizeConfig(undefined);
    }
  }

  get(): AppConfig {
    return this.cache;
  }

  getPreferences(): AppPreferences {
    return this.cache.app;
  }

  getState(): ModelState {
    return this.cache.state;
  }

  async setPreferences(patch: Partial<AppPreferences>): Promise<AppConfig> {
    const absorbed = this.syncFromDisk();
    return this.announceAbsorbed(absorbed, await this.write({ ...this.cache, app: { ...this.cache.app, ...patch } }));
  }

  async setState(patch: Partial<ModelState>): Promise<AppConfig> {
    const absorbed = this.syncFromDisk();
    const current = this.cache.state;
    return this.announceAbsorbed(absorbed, await this.write({
      ...this.cache,
      state: {
        ...current,
        ...patch,
        loadSettings: { ...current.loadSettings, ...(patch.loadSettings || {}) },
        requestSettings: { ...current.requestSettings, ...(patch.requestSettings || {}) },
      },
    }));
  }

  /**
   * A write that first pulled in a peer's newer file (TUI, other window) must
   * still announce that edit: the debounced watcher will find nothing new.
   */
  private announceAbsorbed(absorbed: boolean, config: AppConfig): AppConfig {
    if (absorbed) {
      this._onDidChangeExternally.fire(config);
    }
    return config;
  }

  private async write(next: AppConfig): Promise<AppConfig> {
    const normalized = normalizeConfig(next);
    const text = `${JSON.stringify(normalized, null, 2)}\n`;
    this.cache = normalized;

    if (text !== this.lastWritten) {
      fs.mkdirSync(path.dirname(this.file), { recursive: true });
      const tmp = `${this.file}.tmp-${process.pid}-${Date.now()}`;
      try {
        fs.writeFileSync(tmp, text, { encoding: "utf8", mode: 0o600 });
        fs.renameSync(tmp, this.file);
        this.lastWritten = text;
      } catch (err) {
        fs.rmSync(tmp, { force: true });
        throw err;
      }
    }

    this._onDidChange.fire(normalized);
    return normalized;
  }

  /** Create the file on disk if it is missing, so it is discoverable/editable. */
  async ensureExists(): Promise<void> {
    if (!fs.existsSync(this.file)) {
      await this.write(this.cache);
    }
  }

  /**
   * Watch the containing directory (not the file): an atomic rename replaces
   * the inode, which a file watch would stop following after the first write.
   */
  watch(): Disposable {
    if (this.watcher) {
      return { dispose: () => this.unwatch() };
    }
    const dir = path.dirname(this.file);
    const name = path.basename(this.file);
    try {
      fs.mkdirSync(dir, { recursive: true });
      this.watcher = fs.watch(dir, (_event, changed) => {
        if (changed && changed !== name) {
          return;
        }
        clearTimeout(this.debounce);
        this.debounce = setTimeout(() => this.reloadIfChanged(), 120);
      });
    } catch (err) {
      console.warn(
        `Llama AIO: cannot watch ${dir} for config changes:`,
        err instanceof Error ? err.message : err
      );
    }
    return { dispose: () => this.unwatch() };
  }

  /** Pull a newer on-disk config into the cache without firing events. */
  private syncFromDisk(): boolean {
    let text: string;
    try {
      text = fs.readFileSync(this.file, "utf8");
    } catch {
      return false;
    }
    if (text === this.lastWritten) {
      return false;
    }
    try {
      this.cache = normalizeConfig(JSON.parse(text) as Partial<AppConfig>);
      this.lastWritten = text;
      return true;
    } catch (err) {
      // A peer may be mid-write with a non-atomic editor; the next event wins.
      console.warn(
        "Llama AIO: ignoring unparseable config change:",
        err instanceof Error ? err.message : err
      );
      return false;
    }
  }

  private reloadIfChanged(): void {
    if (!this.syncFromDisk()) {
      return;
    }
    this._onDidChangeExternally.fire(this.cache);
    this._onDidChange.fire(this.cache);
  }

  private unwatch(): void {
    clearTimeout(this.debounce);
    this.debounce = undefined;
    this.watcher?.close();
    this.watcher = undefined;
  }

  dispose(): void {
    this.unwatch();
    this._onDidChange.dispose();
    this._onDidChangeExternally.dispose();
  }

  /** `WorkspaceConfiguration`-shaped view over {@link AppPreferences}. */
  accessor(): ConfigAccessor {
    const self = this;
    return {
      get<T>(key: string, fallback?: T): T | undefined {
        const value = (self.cache.app as unknown as Record<string, unknown>)[key];
        return (value === undefined ? fallback : value) as T | undefined;
      },
      async update(key: string, value: unknown): Promise<void> {
        await self.setPreferences({ [key]: value } as Partial<AppPreferences>);
      },
    };
  }
}

interface Disposable {
  dispose(): void;
}

/** Where the shared configuration lives, for logs and "reveal in folder". */
export function describeConfigLocation(): { root: string; file: string } {
  return { root: getDataRoot(), file: getConfigPath() };
}

/** Convert the file's model state into the legacy {@link ExtensionState} shape. */
export function toExtensionState(state: ModelState): ExtensionState {
  return {
    selectedModelPath: state.selectedModelPath,
    loadSettings: state.loadSettings,
    requestSettings: state.requestSettings,
    modelMaxContext: state.modelMaxContext,
    modelCapabilities: state.modelCapabilities,
  };
}

/**
 * Shared services for the TUI — one ConfigFile / SettingsStore / ProcessManager
 * shared with the VS Code extension via ~/.llama-aio-vs/config.json.
 */
import {
  ConfigFile,
  HuggingFaceClient,
  LlamaInstaller,
  ProcessManager,
  SettingsStore,
  type Disposable,
} from "@llama-aio/core";

export interface AppServices {
  config: ConfigFile;
  store: SettingsStore;
  processManager: ProcessManager;
  installer: LlamaInstaller;
  hf: HuggingFaceClient;
  dispose: () => void;
}

export async function createServices(): Promise<AppServices> {
  const config = new ConfigFile();
  await config.ensureExists();
  const watch: Disposable = config.watch();
  const store = new SettingsStore(config);
  const processManager = new ProcessManager(store);
  const installer = new LlamaInstaller(store);
  const hf = new HuggingFaceClient(store);
  return {
    config,
    store,
    processManager,
    installer,
    hf,
    dispose: () => {
      watch.dispose();
      config.dispose();
    },
  };
}

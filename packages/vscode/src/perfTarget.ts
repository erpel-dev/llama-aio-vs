import * as path from "path";
import type { PerfStats, ProcessManager, SettingsStore } from "@llama-aio/core";

/**
 * Identity of the server + model that performance numbers describe, or
 * undefined while nothing is serving (stopping keeps the last numbers).
 * Local: endpoint + the model the running server was launched with.
 * Remote: endpoint + the model id the server reports.
 */
export function perfTargetKey(store: SettingsStore, processManager: ProcessManager): string | undefined {
  const endpoint = store.getEndpoint();
  if (store.isRemote()) {
    const info = processManager.remote.current();
    return info?.modelId ? `remote|${endpoint}|${info.modelId}` : undefined;
  }
  const launched = processManager.getLaunchedConfig();
  return launched?.modelPath ? `local|${endpoint}|${path.resolve(launched.modelPath)}` : undefined;
}

/**
 * Point `perf` at the current target when one is serving. Switching between
 * this machine and a remote server always starts fresh, even while nothing
 * serves yet, so one side's numbers never show on the other.
 */
export function syncPerfTarget(perf: PerfStats, store: SettingsStore, processManager: ProcessManager): void {
  const key = perfTargetKey(store, processManager);
  if (key !== undefined) {
    perf.setTarget(key);
    return;
  }
  const mode = store.isRemote() ? "remote" : "local";
  const current = perf.getTarget();
  if (current !== undefined && !current.startsWith(mode + "|")) {
    perf.setTarget(`${mode}|${store.getEndpoint()}|`);
  }
}

import * as fs from "fs";
import * as http from "http";
import * as path from "path";
import { execFileSync, spawn } from "child_process";
import {
  ensureDirs,
  findLlamaServerOnPath,
  getLlamaServerBinary,
  getLockDir,
  getLockPath,
  getLogPath,
} from "./paths";
import {
  captureChildError,
  childSpawnError,
  resolveLaunchMode,
  spawnInExternalTerminal,
} from "./externalTerminal";
import { clampLoadSettingsToModel, readModelCapabilities } from "./ggufMetadata";
import { detectGpus } from "./gpuInfo";
import { LlamaInstaller } from "./llamaInstaller";
import {
  hasUsableFhsDynamicLinker,
  isNixOS,
  LaunchPlan,
  nixOsIncompatibilityHint,
  resolveLaunchPlan,
} from "./nixCompat";
import {
  isLlamaServerProcess,
  isPidAlive,
  sameModelFile,
  stopProcessTree,
  uniquePids,
} from "./processIdentity";
import { buildServerArgs, serverConfigFingerprint, SettingsStore } from "./settings";
import { formatCommandDisplay, normalizeLoadSettingsForCpuBackend } from "./serverArgs";
import { LlamaLoadSettings, ServerStatus } from "./types";
import { activeInstallLock } from "./installSwap";
import { acquireLaunchLock, activeLaunchLock, LaunchLockHandle } from "./launchLock";

interface LockFile {
  pid: number;
  port: number;
  host: string;
  modelPath: string;
  startedAt: string;
  binary: string;
  args: string[];
  launchMode?: string;
  /** Fingerprint of model + load settings (+ launch mode) applied at start. */
  configFingerprint?: string;
  /** Load settings the server was started with (sidebar "what changed" / Discard). */
  loadSettings?: LlamaLoadSettings;
}

/** Model + settings of the running server, as recorded in the lock at start. */
export interface LaunchedConfig {
  modelPath: string;
  loadSettings?: LlamaLoadSettings;
  launchMode?: string;
  startedAt: string;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Tail a log from the last byte offset. The start-wait loop used to
 * `readFileSync` the whole file (and run `ss`/`netstat`) on every tick.
 */
class IncrementalLog {
  private offset = 0;
  private remainder = "";

  constructor(private readonly logPath: string) {}

  async poll(): Promise<{ fatal?: string; sawLoading: boolean }> {
    let chunk = "";
    try {
      const fh = await fs.promises.open(this.logPath, "r");
      try {
        const st = await fh.stat();
        if (st.size < this.offset) {
          this.offset = 0;
          this.remainder = "";
        }
        const len = st.size - this.offset;
        if (len > 0) {
          const buf = Buffer.alloc(len);
          const { bytesRead } = await fh.read(buf, 0, len, this.offset);
          this.offset += bytesRead;
          chunk = buf.subarray(0, bytesRead).toString("utf8");
        }
      } finally {
        await fh.close();
      }
    } catch {
      return { sawLoading: false };
    }

    const text = this.remainder + chunk;
    const lines = text.split(/\r?\n/);
    this.remainder = lines.pop() ?? "";
    let sawLoading = false;
    let fatal: string | undefined;
    const consider = (line: string) => {
      if (/loading model/i.test(line)) {
        sawLoading = true;
      }
      if (FATAL_LOG_RE.test(line)) {
        fatal = line.trim();
      }
    };
    for (const line of lines) {
      consider(line);
    }
    if (this.remainder) {
      consider(this.remainder);
    }
    return { fatal, sawLoading };
  }
}

/** Fatal / terminal failure lines from llama-server logs. */
const FATAL_LOG_RE =
  /error loading model|failed to load|ErrorDeviceLost|out of memory|insufficient memory|CUDA error|VK_ERROR|ggml_vulkan.*failed|llama_init_from_model.*failed|common_init_from_params.*failed|srv\s+load_model:\s+failed/i;

export type StartProgress = (message: string) => void;

/** Opaque handle from {@link ProcessManager.claimLaunch}; pass into start/reload. */
export type LaunchToken = { readonly id: number };

export const LAUNCH_IN_PROGRESS_MSG = "A server start/reload is already in progress.";

export class ProcessManager {
  /** In-flight start/reload — drives sidebar "Loading model…" until HTTP ready. */
  private boot:
    | {
        kind: "start" | "reload";
        message: string;
      }
    | undefined;

  /** Exclusive launch lock — prevents overlapping start/reload (and dual progress toasts). */
  private launchToken: LaunchToken | undefined;
  private nextLaunchId = 1;

  /**
   * PIDs this ProcessManager spawned itself (launchers / servers). They are
   * trusted without a command-line check even when the OS hides it from us.
   */
  private readonly spawnedPids = new Set<number>();

  /**
   * Identity verdicts for lock-file PIDs, keyed by pid → lock.startedAt. A PID
   * cannot be reused while the process is alive, so a verdict stays valid
   * until the lock is rewritten for a new launch.
   */
  private readonly pidIdentity = new Map<number, { startedAt: string; ours: boolean }>();

  constructor(private readonly store: SettingsStore) {}

  /**
   * True when `lock.pid` is (still) a llama-server — or a launcher we spawned —
   * rather than an unrelated process that inherited the number after a crash.
   * Result is cached per pid/launch so status polling stays cheap.
   */
  private lockPidIsOurs(lock: LockFile): boolean {
    if (!isPidAlive(lock.pid)) {
      this.spawnedPids.delete(lock.pid);
      return false;
    }
    if (this.spawnedPids.has(lock.pid)) {
      return true;
    }
    const cached = this.pidIdentity.get(lock.pid);
    if (cached && cached.startedAt === lock.startedAt) {
      return cached.ours;
    }
    const ours = isLlamaServerProcess(lock.pid);
    this.pidIdentity.set(lock.pid, { startedAt: lock.startedAt, ours });
    return ours;
  }

  private rememberSpawned(pid: number | undefined): void {
    if (pid && pid > 0) {
      this.spawnedPids.add(pid);
    }
  }

  isStarting(): boolean {
    return !!this.boot || !!this.launchToken;
  }

  /**
   * Claim exclusive start/reload before any awaits (confirm dialogs, withProgress).
   * Returns undefined if another launch is already in flight.
   */
  claimLaunch(kind: "start" | "reload", message?: string): LaunchToken | undefined {
    if (this.launchToken) {
      return undefined;
    }
    const token: LaunchToken = { id: this.nextLaunchId++ };
    this.launchToken = token;
    this.beginBoot(
      kind,
      message || (kind === "reload" ? "Reloading llama-server…" : "Starting llama-server…")
    );
    return token;
  }

  /** Mark boot UI immediately (before awaits / pushState races). Prefer {@link claimLaunch}. */
  markStarting(kind: "start" | "reload", message?: string): void {
    this.claimLaunch(kind, message);
  }

  /** Clear boot UI after cancel / error paths that never call start/reload. */
  clearStarting(): void {
    this.releaseLaunch(this.launchToken);
  }

  /** Release a claim when canceling before start/reload, or after they finish. */
  releaseLaunch(token: LaunchToken | undefined): void {
    if (!token || this.launchToken !== token) {
      return;
    }
    this.launchToken = undefined;
    this.endBoot();
  }

  private beginBoot(kind: "start" | "reload", message: string): void {
    this.boot = { kind, message };
  }

  private updateBootMessage(message: string): void {
    if (this.boot) {
      this.boot = { ...this.boot, message };
    }
  }

  private endBoot(): void {
    this.boot = undefined;
  }

  /**
   * Ensure this call owns the launch lock. If `token` was claimed by the UI, verifies it;
   * otherwise claims a new one. Throws if another launch owns the lock.
   */
  private takeLaunch(kind: "start" | "reload", token?: LaunchToken): LaunchToken {
    if (token) {
      if (this.launchToken !== token) {
        throw new Error(LAUNCH_IN_PROGRESS_MSG);
      }
      this.beginBoot(
        kind,
        kind === "reload" ? "Reloading llama-server…" : "Starting llama-server…"
      );
      return token;
    }
    const claimed = this.claimLaunch(kind);
    if (!claimed) {
      throw new Error(LAUNCH_IN_PROGRESS_MSG);
    }
    return claimed;
  }

  getStatus(): ServerStatus {
    const host = this.store.getHost();
    const port = this.store.getPort();
    const endpoint = this.store.getEndpoint();
    const lock = this.readLock();
    const bootFields = this.boot
      ? { starting: true as const, startMessage: this.boot.message }
      : { starting: false as const, startMessage: undefined };

    if (lock && this.lockPidIsOurs(lock)) {
      return {
        running: !this.boot, // still booting = not yet "ready" for UI purposes
        pid: lock.pid,
        port: lock.port,
        host: lock.host,
        modelPath: lock.modelPath,
        endpoint: `http://${lock.host}:${lock.port}`,
        ownedByThisExtension: true,
        message: this.boot?.message || `Running (pid ${lock.pid})`,
        // Don't show dirty while a reload/start is applying the new fingerprint.
        configDirty: this.boot ? false : this.isConfigDirty(lock),
        ...bootFields,
      };
    }

    // Recover after external-terminal launchers exit (common on Windows `start` /
    // Windows Terminal) while llama-server still owns the port.
    if (lock) {
      // Only adopt a port occupant that is actually a llama-server; otherwise
      // an unrelated listener would be reported as our running server.
      const portPids = this.findPidsOnPort(lock.port).filter(isLlamaServerProcess);
      if (portPids.length) {
        const pid = portPids[0];
        this.writeLock({ ...lock, pid });
        return {
          running: !this.boot,
          pid,
          port: lock.port,
          host: lock.host,
          modelPath: lock.modelPath,
          endpoint: `http://${lock.host}:${lock.port}`,
          ownedByThisExtension: true,
          message: this.boot?.message || `Running (pid ${pid})`,
          configDirty: this.boot ? false : this.isConfigDirty(lock),
          ...bootFields,
        };
      }
      // Dead (or reused) pid and nothing of ours on the port: the lock is
      // stale. Drop it so status polling stops scanning the port every tick —
      // unless a launch is still in flight here or in another window, whose
      // provisional lock this may be.
      if (!this.boot && !activeLaunchLock()) {
        this.clearLock();
      }
    }

    return {
      running: false,
      port,
      host,
      endpoint,
      ownedByThisExtension: false,
      message: this.boot?.message || "Not running",
      configDirty: false,
      ...bootFields,
    };
  }

  /** Current model + load + launch fingerprint (CPU-normalized like start). */
  private currentConfigFingerprint(modelPath?: string): string {
    const state = this.store.getState();
    const model = (modelPath ?? state.selectedModelPath ?? "").trim();
    let loadSettings = state.loadSettings;
    if (this.isCpuBackend()) {
      loadSettings = normalizeLoadSettingsForCpuBackend(loadSettings);
    }
    const launchMode = resolveLaunchMode(this.store.getConfig().get<string>("launchMode"));
    return serverConfigFingerprint(model, loadSettings, launchMode);
  }

  /** What the running server was launched with, or undefined when none of ours is up. */
  getLaunchedConfig(): LaunchedConfig | undefined {
    const lock = this.readLock();
    if (!lock || !this.lockPidIsOurs(lock)) {
      return undefined;
    }
    return {
      modelPath: lock.modelPath,
      loadSettings: lock.loadSettings,
      launchMode: lock.launchMode,
      startedAt: lock.startedAt,
    };
  }

  private isConfigDirty(lock: LockFile): boolean {
    return !this.lockMatchesDesired(lock);
  }

  /** True when the lock describes the server we would start now. */
  private lockMatchesDesired(lock: LockFile, modelPath?: string): boolean {
    if (lock.configFingerprint) {
      return lock.configFingerprint === this.currentConfigFingerprint(modelPath);
    }
    const selected = (modelPath ?? this.store.getState().selectedModelPath ?? "").trim();
    return sameModelFile(lock.modelPath, selected);
  }

  private canReuseRunningServer(
    modelPath: string,
    live: { nCtx: number; slots: number; modelPath?: string } | undefined
  ): boolean {
    const lock = this.readLock();
    if (lock?.configFingerprint) {
      return this.lockMatchesDesired(lock, modelPath);
    }
    // Pre-fingerprint lock or foreign occupant: slot size + model file only.
    if (!live) {
      return false;
    }
    const state = this.store.getState();
    const desiredSlot = this.store.getSlotContextSize(state.loadSettings);
    const modelMatches = !live.modelPath || sameModelFile(live.modelPath, modelPath);
    return (
      live.nCtx >= desiredSlot * 0.9 &&
      live.slots <= state.loadSettings.maxConcurrentPredictions &&
      modelMatches
    );
  }

  async isHttpReady(endpoint?: string): Promise<boolean> {
    const base = endpoint || this.store.getEndpoint();
    return new Promise((resolve) => {
      const req = http.get(`${base}/health`, { timeout: 1500 }, (res) => {
        res.resume();
        resolve((res.statusCode || 500) < 500);
      });
      req.setTimeout(1500, () => {
        req.destroy();
        resolve(false);
      });
      req.on("error", () => resolve(false));
      req.on("timeout", () => {
        req.destroy();
        resolve(false);
      });
    });
  }

  /** How the current backend setting should launch llama-server. */
  resolveLaunch(): LaunchPlan {
    const installer = new LlamaInstaller(this.store);
    const usePath = installer.resolveActiveUiBackend() === "path";
    const installed = usePath
      ? undefined
      : getLlamaServerBinary(installer.getActiveInstallDir());
    return resolveLaunchPlan({ installedBinary: installed, usePath });
  }

  /**
   * Command line the next start would run, for the sidebar “Copy command” button.
   * Does not launch anything and does not write the log.
   */
  describeCommandLine(): string {
    const launch = this.resolveLaunch();
    const state = this.store.getState();
    const model = (state.selectedModelPath || "").trim();
    if (!model) {
      throw new Error("No model selected. Choose a GGUF before copying the command line.");
    }
    let loadSettings = state.loadSettings;
    const caps = state.modelCapabilities;
    if (caps) {
      loadSettings = clampLoadSettingsToModel(loadSettings, caps);
    }
    if (this.isCpuBackend()) {
      loadSettings = normalizeLoadSettingsForCpuBackend(loadSettings);
    }
    const gpus = this.isCpuBackend() ? [] : detectGpus(false, launch.binary);
    const args = buildServerArgs(model, this.store.getHost(), this.store.getPort(), loadSettings, {
      gpus,
      requestSampling: state.requestSettings,
      caps,
    });
    return formatCommandDisplay([launch.command, ...launch.prefixArgs, ...args]);
  }

  /** Raw tail of the server log (unfiltered), for the Output channel after a failed start. */
  readRecentLogLines(maxLines = 20): string {
    try {
      const lines = fs
        .readFileSync(getLogPath(), "utf8")
        .split(/\r?\n/)
        .filter((line) => line.length > 0);
      return lines.slice(-maxLines).join("\n");
    } catch {
      return "";
    }
  }

  resolveBinary(): string {
    try {
      return this.resolveLaunch().binary;
    } catch {
      const installer = new LlamaInstaller(this.store);
      if (installer.resolveActiveUiBackend() === "path") {
        return findLlamaServerOnPath() || "llama-server";
      }
      return getLlamaServerBinary(installer.getActiveInstallDir());
    }
  }

  /** True when the active install / setting is a CPU-only binary (no GPU). */
  isCpuBackend(): boolean {
    return new LlamaInstaller(this.store).resolveActiveUiBackend() === "cpu";
  }

  async start(
    modelPath?: string,
    onProgress?: StartProgress,
    token?: LaunchToken
  ): Promise<ServerStatus> {
    const owned = this.takeLaunch("start", token);
    const report = (msg: string) => {
      this.updateBootMessage(msg);
      onProgress?.(msg);
    };
    let fileLock: LaunchLockHandle | undefined;
    try {
      fileLock = acquireLaunchLock("start");
      return await this.startInner(modelPath, report);
    } finally {
      fileLock?.release();
      this.releaseLaunch(owned);
    }
  }

  private async startInner(modelPath: string | undefined, report: StartProgress): Promise<ServerStatus> {
    const installing = activeInstallLock();
    if (installing) {
      throw new Error(
        "llama.cpp is being updated" +
          (installing.pid && installing.pid !== process.pid
            ? ` in another Llama AIO window (pid ${installing.pid})`
            : "") +
          ". Wait for that install to finish, then start the server."
      );
    }

    const state = this.store.getState();
    const model = (modelPath || state.selectedModelPath || "").trim();
    if (!model) {
      throw new Error("No model selected. Browse or select a local GGUF first.");
    }
    if (!fs.existsSync(model)) {
      throw new Error(`Model file not found: ${model}`);
    }

    // Reuse only when the live process matches the fingerprint we would start
    // with. Slot-ctx heuristics miss KV type, ngl, and speculative flags.
    if (await this.isHttpReady()) {
      const live = await this.fetchLiveSlotContext();
      if (this.canReuseRunningServer(model, live)) {
        const existing = this.getStatus();
        return {
          ...existing,
          running: true,
          starting: false,
          startMessage: undefined,
          modelPath: model,
          message: live
            ? `Reusing server on port ${this.store.getPort()} (${live.nCtx} ctx/slot, ${live.slots} slot(s))`
            : `Reusing server on port ${this.store.getPort()}`,
        };
      }
      report("Stopping previous server…");
      await this.stop(true);
      await sleep(400);
    }

    const lock = this.readLock();
    if (lock && this.lockPidIsOurs(lock)) {
      if (lock.configFingerprint && !this.lockMatchesDesired(lock, model)) {
        report("Stopping previous server…");
        await this.stop(true);
      } else {
        // Process alive but HTTP not ready yet — wait briefly, then re-check.
        report("Waiting for existing server…");
        let becameReady = false;
        for (let i = 0; i < 20; i++) {
          await sleep(250);
          if (await this.isHttpReady(`http://${lock.host}:${lock.port}`)) {
            becameReady = true;
            break;
          }
        }
        if (becameReady) {
          const live = await this.fetchLiveSlotContext();
          if (this.canReuseRunningServer(model, live)) {
            return this.getStatus();
          }
        }
        report("Stopping previous server…");
        await this.stop(true);
      }
    } else if (lock) {
      this.clearLock();
    }

    const launch = this.resolveLaunch();
    const binary = launch.binary;

    const host = this.store.getHost();
    const port = this.store.getPort();

    // Fail fast instead of waiting out the readiness timeout when something
    // that is not ours already holds the port.
    const blockers = (this.queryPidsOnPort(port) ?? []).filter((pid) => !isLlamaServerProcess(pid));
    if (blockers.length) {
      throw new Error(
        `Port ${port} is already in use by another process (pid ${blockers.join(", ")}). ` +
          `Close it, or change the port in ~/.llama-aio-vs/config.json.`
      );
    }

    let loadSettings = state.loadSettings;
    let caps = state.modelCapabilities;
    try {
      caps = state.modelCapabilities || readModelCapabilities(model);
      loadSettings = clampLoadSettingsToModel(loadSettings, caps);
      if (!state.modelCapabilities || state.modelCapabilities.path !== model) {
        await this.store.setState({
          modelCapabilities: caps,
          modelMaxContext: caps.maxContextLength,
          loadSettings,
        });
      }
    } catch {
      // If metadata cannot be read, proceed with stored settings.
    }
    if (this.isCpuBackend()) {
      // CPU builds ignore -ngl / --n-cpu-moe; keep args honest so logs/estimate match reality.
      loadSettings = normalizeLoadSettingsForCpuBackend(loadSettings);
    }
    const gpus = this.isCpuBackend() ? [] : detectGpus(false, binary);
    const args = buildServerArgs(model, host, port, loadSettings, {
      gpus,
      // Ship request sampling defaults as CLI flags so raw clients pointing at
      // the endpoint inherit them too (our frontends send per-request values).
      requestSampling: state.requestSettings,
      caps,
    });
    const launchMode = resolveLaunchMode(this.store.getConfig().get<string>("launchMode"));
    const configFingerprint = serverConfigFingerprint(model, loadSettings, launchMode);

    ensureDirs(getLockDir(), path.dirname(getLogPath()));
    // Fresh log per start so failures are easy to read.
    const logPath = getLogPath();
    const spawnArgv = [...launch.prefixArgs, ...args];
    fs.writeFileSync(
      logPath,
      `# llama-aio-vs start ${new Date().toISOString()}\n` +
        `# mode=${launchMode} method=${launch.method}\n` +
        `# ${launch.command} ${spawnArgv.join(" ")}\n` +
        (launch.note ? `# note: ${launch.note}\n` : ""),
      "utf8"
    );

    const childEnv = launch.env;

    report(
      launch.method === "steam-run"
        ? `Starting llama-server via steam-run (${path.basename(model)})…`
        : launch.method === "path"
          ? `Starting llama-server from PATH (${path.basename(model)})…`
          : `Starting llama-server (${path.basename(model)})…`
    );
    let launcherPid: number | undefined;
    let launched: ReturnType<typeof captureChildError> | undefined;
    if (launchMode === "externalTerminal") {
      const child = spawnInExternalTerminal({
        binary,
        args,
        env: childEnv,
        logPath,
        command: launch.method === "direct" ? undefined : launch.command,
        prefixArgs: launch.method === "direct" ? undefined : launch.prefixArgs,
      });
      launched = child;
      await sleep(0);
      launcherPid = child.pid;
      const failed = childSpawnError(child);
      if (failed || !launcherPid) {
        throw new Error(
          failed
            ? `Failed to open external terminal for llama-server: ${failed.message}`
            : "Failed to open external terminal for llama-server."
        );
      }
      this.rememberSpawned(launcherPid);
    } else {
      const logFd = fs.openSync(logPath, "a");
      const child = captureChildError(
        spawn(launch.command, spawnArgv, {
          detached: true,
          stdio: ["ignore", logFd, logFd],
          windowsHide: true,
          cwd: path.dirname(binary),
          env: childEnv,
        })
      );
      launched = child;
      fs.closeSync(logFd);
      await sleep(0);
      launcherPid = child.pid;
      const failed = childSpawnError(child);
      if (failed || !launcherPid) {
        throw new Error(
          failed
            ? `Failed to spawn llama-server: ${failed.message}`
            : "Failed to spawn llama-server (no pid)."
        );
      }
      this.rememberSpawned(launcherPid);
      child.unref();
    }

    // Provisional lock (launcher pid). Updated to llama-server pid once the port is up.
    this.writeLock({
      pid: launcherPid,
      port,
      host,
      modelPath: model,
      startedAt: new Date().toISOString(),
      binary,
      args,
      launchMode,
      configFingerprint,
      loadSettings,
    });

    // Wait for HTTP readiness. Log is tailed from the last offset; port scans
    // (`ss` / `netstat`) stay out of this loop — one lookup once /health answers.
    const maxWaitMs = 180_000; // large MoE models can take a while to mmap/load
    const startedAt = Date.now();
    let sawLoading = false;
    let lastProgressAt = 0;
    const logTail = new IncrementalLog(logPath);

    while (Date.now() - startedAt < maxWaitMs) {
      const failed = childSpawnError(launched);
      if (failed) {
        await this.stop(true).catch(() => undefined);
        throw new Error(
          this.formatStartFailure(`Failed to spawn llama-server (${failed.message}).`)
        );
      }

      const log = await logTail.poll();
      if (log.sawLoading) {
        sawLoading = true;
      }
      if (log.fatal) {
        await this.stop(true).catch(() => undefined);
        throw new Error(this.formatStartFailure("llama-server failed while loading.", log.fatal));
      }

      if (await this.isHttpReady()) {
        const serverPids = this.findPidsOnPort(port);
        const pid = serverPids[0] || launcherPid;
        const finalLock: LockFile = {
          pid,
          port,
          host,
          modelPath: model,
          startedAt: new Date().toISOString(),
          binary,
          args,
          launchMode,
          configFingerprint,
          loadSettings,
        };
        this.writeLock(finalLock);
        // We just watched this pid answer on our port — no need to re-inspect it.
        this.pidIdentity.set(pid, { startedAt: finalLock.startedAt, ours: true });
        return {
          running: true,
          pid,
          port,
          host,
          modelPath: model,
          endpoint: `http://${host}:${port}`,
          ownedByThisExtension: true,
          configDirty: false,
          message:
            launchMode === "externalTerminal"
              ? `Started in external terminal (pid ${pid}). Close that window to stop the server.`
              : `Started in background (pid ${pid})`,
        };
      }

      if (launchMode === "background" && launcherPid && !isPidAlive(launcherPid)) {
        await this.stop(true).catch(() => undefined);
        throw new Error(this.formatStartFailure("llama-server exited early."));
      }

      const elapsedSec = Math.floor((Date.now() - startedAt) / 1000);
      if (Date.now() - lastProgressAt > 2500) {
        lastProgressAt = Date.now();
        report(
          sawLoading
            ? `Loading model into memory… ${elapsedSec}s`
            : `Waiting for llama-server… ${elapsedSec}s`
        );
      }

      await sleep(400);
    }

    await this.stop(true).catch(() => undefined);
    throw new Error(
      this.formatStartFailure(
        `llama-server did not become ready within ${Math.round(maxWaitMs / 1000)}s.`
      )
    );
  }

  private formatStartFailure(prefix: string, fatalLine?: string): string {
    const logPath = getLogPath();
    const snippet = this.readLogSnippet(logPath, 20);
    const blob = `${fatalLine || ""}\n${snippet}`;
    let hint = "";
    if (/ErrorDeviceLost|VK_ERROR_DEVICE_LOST|vk::Queue::submit/i.test(blob)) {
      hint =
        "\n\nVulkan GPU lost while loading (common with large MoE + full offload on AMD).\n" +
        "Try in Llama AIO sidebar:\n" +
        "• Lower GPU Offload (e.g. 20–40) or raise CPU MoE layers\n" +
        "• Lower Context Length\n" +
        "• Turn off Try mmap()\n" +
        "• Use a smaller quant (Q4_K_M) or denser/smaller model\n" +
        "Then Reload.";
    } else if (/openvino/i.test(blob)) {
      hint =
        "\n\nInstalled build looks like OpenVINO — set backend to Vulkan (AMD) or CPU, then re-run Install / Upgrade llama.cpp.";
    } else if (/out of memory|insufficient memory/i.test(blob)) {
      hint =
        "\n\nOut of memory while loading. Lower Context Length / GPU Offload, or use a smaller quant.";
    } else if (
      (isNixOS() || (process.platform === "linux" && !hasUsableFhsDynamicLinker())) &&
      /no such file or directory|error while loading shared libraries|cannot open shared object file/i.test(
        blob
      )
    ) {
      hint = `\n\n${nixOsIncompatibilityHint()}`;
    }
    const fatal = fatalLine ? `\n\nDetected: ${fatalLine}` : "";
    return `${prefix}${fatal}${hint}\n\nLast log lines (${logPath}):\n${snippet || "(empty)"}`;
  }

  private readLogSnippet(logPath: string, maxLines = 18): string {
    try {
      const lines = fs
        .readFileSync(logPath, "utf8")
        .split(/\r?\n/)
        .filter((l) => l.trim().length > 0);
      // Prefer the abort / error lines over gdb noise.
      const interesting = lines.filter(
        (l) =>
          /error|abort|failed|cannot|GGML|OPENVINO|CUDA|Vulkan|loading model|srv/i.test(l) &&
          !/LWP |gdb|Debuginfod|Thread debugging|Inferior /i.test(l)
      );
      const picked = (interesting.length ? interesting : lines).slice(-maxLines);
      return picked.join("\n");
    } catch {
      return "";
    }
  }

  async stop(force = false): Promise<void> {
    const lock = this.readLock();
    const port = lock?.port ?? this.store.getPort();
    const pids = new Set<number>();
    // Never signal a PID just because it is written in the lock file: after a
    // crash the OS may have handed that number to an unrelated process.
    if (lock?.pid && isPidAlive(lock.pid)) {
      if (this.lockPidIsOurs(lock)) {
        pids.add(lock.pid);
      } else {
        console.warn(
          `Llama AIO: lock pid ${lock.pid} is no longer llama-server (pid reused) — not killing it.`
        );
      }
    }
    // Also clean up stale llama-servers holding our port (missed locks, crashed
    // launchers). Anything else on the port belongs to another application and
    // is left running — the caller is told instead.
    const occupants = this.queryPidsOnPort(port) ?? [];
    const foreign: number[] = [];
    for (const pid of occupants) {
      if (pids.has(pid) || isLlamaServerProcess(pid)) {
        pids.add(pid);
      } else {
        foreign.push(pid);
      }
    }
    if (foreign.length) {
      console.warn(
        `Llama AIO: leaving non-llama process(es) ${foreign.join(", ")} on port ${port} alone.`
      );
    }
    for (const pid of pids) {
      await stopProcessTree(pid, force);
    }
    // Wait briefly for the port to free (Windows especially).
    for (let i = 0; i < 20; i++) {
      if (!this.findPidsOnPort(port).length) {
        break;
      }
      await sleep(100);
    }
    this.clearLock();
  }

  /**
   * PIDs listening on `port`, or undefined when we could not find out (no
   * tooling, permission denied). Callers must treat undefined as "unknown"
   * rather than "nothing there" — the difference decides whether we kill.
   */
  private queryPidsOnPort(port: number): number[] | undefined {
    try {
      if (process.platform === "win32") {
        const out = execFileSync("netstat", ["-ano", "-p", "tcp"], {
          encoding: "utf8",
          stdio: ["ignore", "pipe", "ignore"],
          windowsHide: true,
          timeout: 8000,
        });
        const pids = new Set<number>();
        for (const line of out.split(/\r?\n/)) {
          if (!/LISTENING/i.test(line)) {
            continue;
          }
          const parts = line.trim().split(/\s+/);
          // Proto  LocalAddress  ForeignAddress  State  PID
          if (parts.length < 5) {
            continue;
          }
          const local = parts[1];
          if (!local.endsWith(`:${port}`)) {
            continue;
          }
          const pid = Number(parts[parts.length - 1]);
          if (Number.isFinite(pid) && pid > 0) {
            pids.add(pid);
          }
        }
        return [...pids];
      }

      if (process.platform === "darwin") {
        // macOS has no `ss`; lsof prints one PID per line with -t.
        const out = execFileSync(
          "lsof",
          ["-nP", `-iTCP:${port}`, "-sTCP:LISTEN", "-t"],
          { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 8000 }
        );
        return uniquePids(out.split(/\s+/));
      }

      const out = execFileSync("ss", ["-ltnp", `sport = :${port}`], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
        timeout: 8000,
      });
      return uniquePids([...out.matchAll(/pid=(\d+)/g)].map((m) => m[1]));
    } catch (err) {
      // `lsof -t` exits 1 with no output when nothing is listening — that is a
      // real "port is free", not a tooling failure.
      const e = err as { status?: number; stdout?: string };
      if (process.platform === "darwin" && e?.status === 1 && !e?.stdout) {
        return [];
      }
      console.warn(`Llama AIO: could not determine what is listening on port ${port}:`, err);
      return undefined;
    }
  }

  /** Best-effort variant for callers that only need a list to iterate. */
  private findPidsOnPort(port: number): number[] {
    return this.queryPidsOnPort(port) ?? [];
  }

  private async fetchLiveSlotContext(): Promise<
    { nCtx: number; slots: number; modelPath?: string } | undefined
  > {
    try {
      const data = await new Promise<string>((resolve, reject) => {
        const req = http.get(`${this.store.getEndpoint()}/props`, { timeout: 2000 }, (res) => {
          const chunks: Buffer[] = [];
          res.on("data", (c) => chunks.push(c));
          res.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
        });
        req.on("error", reject);
        req.on("timeout", () => {
          req.destroy();
          reject(new Error("timeout"));
        });
      });
      const props = JSON.parse(data) as {
        default_generation_settings?: { n_ctx?: number };
        total_slots?: number;
        model_path?: string;
      };
      const nCtx = props.default_generation_settings?.n_ctx;
      if (!nCtx) {
        return undefined;
      }
      return {
        nCtx,
        slots: props.total_slots || 1,
        modelPath: props.model_path,
      };
    } catch {
      return undefined;
    }
  }

  async reload(onProgress?: StartProgress, token?: LaunchToken): Promise<ServerStatus> {
    const owned = this.takeLaunch("reload", token);
    const report = (msg: string) => {
      this.updateBootMessage(msg);
      onProgress?.(msg);
    };
    let fileLock: LaunchLockHandle | undefined;
    try {
      fileLock = acquireLaunchLock("reload");
      report("Stopping server for reload…");
      await this.stop(true);
      await sleep(400);
      // Keep boot active across inner start — don't nest beginBoot/endBoot via start().
      return await this.startInner(undefined, report);
    } finally {
      fileLock?.release();
      this.releaseLaunch(owned);
    }
  }

  private readLock(): LockFile | undefined {
    const p = getLockPath();
    try {
      if (!fs.existsSync(p)) {
        return undefined;
      }
      return JSON.parse(fs.readFileSync(p, "utf8")) as LockFile;
    } catch {
      return undefined;
    }
  }

  private writeLock(data: LockFile): void {
    ensureDirs(getLockDir());
    // Other windows poll this file; write it atomically so they never read a
    // half-written JSON document.
    const target = getLockPath();
    const tmp = `${target}.${process.pid}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(data, null, 2), "utf8");
    try {
      fs.renameSync(tmp, target);
    } catch {
      fs.writeFileSync(target, JSON.stringify(data, null, 2), "utf8");
      try {
        fs.unlinkSync(tmp);
      } catch {
        // ignore
      }
    }
  }

  private clearLock(): void {
    try {
      fs.unlinkSync(getLockPath());
    } catch {
      // ignore
    }
  }
}

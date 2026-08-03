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
  withBinaryDirEnv,
} from "./paths";
import { resolveLaunchMode, spawnInExternalTerminal } from "./externalTerminal";
import { clampLoadSettingsToModel, readModelCapabilities } from "./ggufMetadata";
import { LlamaInstaller } from "./llamaInstaller";
import { buildServerArgs, serverConfigFingerprint, SettingsStore } from "./settings";
import { ServerStatus } from "./types";

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
}

function isPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Fatal / terminal failure lines from llama-server logs. */
const FATAL_LOG_RE =
  /error loading model|failed to load|ErrorDeviceLost|out of memory|insufficient memory|CUDA error|VK_ERROR|ggml_vulkan.*failed|llama_init_from_model.*failed|common_init_from_params.*failed|srv\s+load_model:\s+failed/i;

export type StartProgress = (message: string) => void;

export class ProcessManager {
  /** In-flight start/reload — drives sidebar "Loading model…" until HTTP ready. */
  private boot:
    | {
        kind: "start" | "reload";
        message: string;
      }
    | undefined;

  constructor(private readonly store: SettingsStore) {}

  isStarting(): boolean {
    return !!this.boot;
  }

  /** Mark boot UI immediately (before awaits / pushState races). */
  markStarting(kind: "start" | "reload", message?: string): void {
    this.beginBoot(
      kind,
      message || (kind === "reload" ? "Reloading llama-server…" : "Starting llama-server…")
    );
  }

  /** Clear boot UI after cancel / error paths that never call start/reload. */
  clearStarting(): void {
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

  getStatus(): ServerStatus {
    const host = this.store.getHost();
    const port = this.store.getPort();
    const endpoint = this.store.getEndpoint();
    const lock = this.readLock();
    const bootFields = this.boot
      ? { starting: true as const, startMessage: this.boot.message }
      : { starting: false as const, startMessage: undefined };

    if (lock && isPidAlive(lock.pid)) {
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
      const portPids = this.findPidsOnPort(lock.port);
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
      loadSettings = {
        ...loadSettings,
        gpuOffload: 0,
        offloadKvCacheToGpu: false,
      };
    }
    const launchMode = resolveLaunchMode(this.store.getConfig().get<string>("launchMode"));
    return serverConfigFingerprint(model, loadSettings, launchMode);
  }

  private isConfigDirty(lock: LockFile): boolean {
    const current = this.currentConfigFingerprint();
    if (lock.configFingerprint) {
      return lock.configFingerprint !== current;
    }
    // Pre-fingerprint locks: dirty if model path changed.
    const selected = (this.store.getState().selectedModelPath || "").trim();
    return (lock.modelPath || "").trim() !== selected;
  }

  async isHttpReady(endpoint?: string): Promise<boolean> {
    const base = endpoint || this.store.getEndpoint();
    return new Promise((resolve) => {
      const req = http.get(`${base}/v1/models`, { timeout: 1500 }, (res) => {
        res.resume();
        resolve((res.statusCode || 500) < 500);
      });
      req.on("error", () => resolve(false));
      req.on("timeout", () => {
        req.destroy();
        resolve(false);
      });
    });
  }

  resolveBinary(): string {
    const installer = new LlamaInstaller(this.store);
    const installed = getLlamaServerBinary(installer.getActiveInstallDir());
    if (fs.existsSync(installed)) {
      return installed;
    }
    const onPath = findLlamaServerOnPath();
    if (onPath) {
      return onPath;
    }
    return installed;
  }

  /** True when the active install / setting is a CPU-only binary (no GPU). */
  isCpuBackend(): boolean {
    return new LlamaInstaller(this.store).resolveActiveUiBackend() === "cpu";
  }

  async start(modelPath?: string, onProgress?: StartProgress): Promise<ServerStatus> {
    this.beginBoot("start", "Starting llama-server…");
    const report = (msg: string) => {
      this.updateBootMessage(msg);
      onProgress?.(msg);
    };
    try {
      return await this.startInner(modelPath, report);
    } finally {
      this.endBoot();
    }
  }

  private async startInner(modelPath: string | undefined, report: StartProgress): Promise<ServerStatus> {
    const state = this.store.getState();
    const model = (modelPath || state.selectedModelPath || "").trim();
    if (!model) {
      throw new Error("No model selected. Browse or select a local GGUF first.");
    }
    if (!fs.existsSync(model)) {
      throw new Error(`Model file not found: ${model}`);
    }

    // Reuse existing healthy instance only if its slot context matches settings.
    // (llama.cpp splits --ctx-size across -np slots; a 8k/4 setup yields 2k/request.)
    if (await this.isHttpReady()) {
      const desiredSlot = this.store.getSlotContextSize(state.loadSettings);
      const live = await this.fetchLiveSlotContext();
      const modelMatches =
        !live?.modelPath ||
        path.basename(live.modelPath) === path.basename(model) ||
        live.modelPath === model;
      if (live && live.nCtx >= desiredSlot * 0.9 && live.slots <= state.loadSettings.maxConcurrentPredictions && modelMatches) {
        const existing = this.getStatus();
        return {
          ...existing,
          running: true,
          starting: false,
          startMessage: undefined,
          modelPath: model,
          message: `Reusing server on port ${this.store.getPort()} (${live.nCtx} ctx/slot, ${live.slots} slot(s))`,
        };
      }
      // Wrong context/slots/model — restart with current settings.
      report("Stopping previous server…");
      await this.stop(true);
      await sleep(400);
    }

    const lock = this.readLock();
    if (lock && isPidAlive(lock.pid)) {
      // Process alive but HTTP not ready yet — wait briefly.
      report("Waiting for existing server…");
      for (let i = 0; i < 20; i++) {
        await sleep(250);
        if (await this.isHttpReady(`http://${lock.host}:${lock.port}`)) {
          return this.getStatus();
        }
      }
      // Stale / stuck — kill and restart.
      await this.stop(true);
    } else if (lock) {
      this.clearLock();
    }

    const binary = this.resolveBinary();
    if (!fs.existsSync(binary)) {
      throw new Error(
        `llama-server binary not found at ${binary}. Run "Llama AIO: Install / Upgrade llama.cpp" first.`
      );
    }

    const host = this.store.getHost();
    const port = this.store.getPort();
    let loadSettings = state.loadSettings;
    try {
      const caps = state.modelCapabilities || readModelCapabilities(model);
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
      loadSettings = {
        ...loadSettings,
        gpuOffload: 0,
        offloadKvCacheToGpu: false,
        nCpuMoe: 0,
      };
    }
    const args = buildServerArgs(model, host, port, loadSettings);
    const launchMode = resolveLaunchMode(this.store.getConfig().get<string>("launchMode"));
    const configFingerprint = serverConfigFingerprint(model, loadSettings, launchMode);

    ensureDirs(getLockDir(), path.dirname(getLogPath()));
    // Fresh log per start so failures are easy to read.
    const logPath = getLogPath();
    fs.writeFileSync(
      logPath,
      `# llama-aio-vs start ${new Date().toISOString()}\n# mode=${launchMode}\n# ${binary} ${args.join(" ")}\n`,
      "utf8"
    );

    const childEnv = withBinaryDirEnv(binary, process.env);

    report(`Starting llama-server (${path.basename(model)})…`);
    let launcherPid: number | undefined;
    if (launchMode === "externalTerminal") {
      const child = spawnInExternalTerminal({
        binary,
        args,
        env: childEnv,
        logPath,
      });
      launcherPid = child.pid;
      if (!launcherPid) {
        throw new Error("Failed to open external terminal for llama-server.");
      }
    } else {
      const logFd = fs.openSync(logPath, "a");
      const child = spawn(binary, args, {
        detached: true,
        stdio: ["ignore", logFd, logFd],
        windowsHide: true,
        cwd: path.dirname(binary),
        env: childEnv,
      });
      fs.closeSync(logFd);
      launcherPid = child.pid;
      if (!launcherPid) {
        throw new Error("Failed to spawn llama-server (no pid).");
      }
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
    });

    // Wait for readiness — fail fast on log fatals / dead process instead of hanging ~45s.
    const maxWaitMs = 180_000; // large MoE models can take a while to mmap/load
    const startedAt = Date.now();
    let sawLoading = false;
    let sawServerPid = false;
    let lastProgressAt = 0;

    while (Date.now() - startedAt < maxWaitMs) {
      await sleep(400);

      const fatal = this.findFatalLogLine(logPath);
      if (fatal) {
        await this.stop(true).catch(() => undefined);
        throw new Error(this.formatStartFailure("llama-server failed while loading.", fatal));
      }

      if (await this.isHttpReady()) {
        const serverPids = this.findPidsOnPort(port);
        const pid = serverPids[0] || launcherPid;
        this.writeLock({
          pid,
          port,
          host,
          modelPath: model,
          startedAt: new Date().toISOString(),
          binary,
          args,
          launchMode,
          configFingerprint,
        });
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

      const portPids = this.findPidsOnPort(port);
      if (portPids.length) {
        sawServerPid = true;
      } else if (sawServerPid) {
        // Was bound, then vanished — crash during init.
        await this.stop(true).catch(() => undefined);
        throw new Error(
          this.formatStartFailure("llama-server exited while starting (port closed).")
        );
      }

      if (launchMode === "background" && launcherPid && !isPidAlive(launcherPid)) {
        await this.stop(true).catch(() => undefined);
        throw new Error(this.formatStartFailure("llama-server exited early."));
      }

      const logTail = this.readLogSnippet(logPath, 8);
      if (/loading model/i.test(logTail)) {
        sawLoading = true;
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
    }

    await this.stop(true).catch(() => undefined);
    throw new Error(
      this.formatStartFailure(
        `llama-server did not become ready within ${Math.round(maxWaitMs / 1000)}s.`
      )
    );
  }

  private findFatalLogLine(logPath: string): string | undefined {
    try {
      const text = fs.readFileSync(logPath, "utf8");
      const lines = text.split(/\r?\n/);
      for (let i = lines.length - 1; i >= 0; i--) {
        const line = lines[i];
        if (FATAL_LOG_RE.test(line)) {
          return line.trim();
        }
      }
    } catch {
      // ignore
    }
    return undefined;
  }

  private formatStartFailure(prefix: string, fatalLine?: string): string {
    const logPath = getLogPath();
    const snippet = this.readLogSnippet(logPath);
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
    if (lock?.pid) {
      pids.add(lock.pid);
    }
    // Also kill whatever currently owns our port (stale processes / missed locks).
    for (const pid of this.findPidsOnPort(port)) {
      pids.add(pid);
    }
    for (const pid of pids) {
      if (!isPidAlive(pid)) {
        continue;
      }
      try {
        if (process.platform === "win32") {
          try {
            execFileSync("taskkill", ["/PID", String(pid), "/T", "/F"], {
              stdio: "ignore",
              windowsHide: true,
              timeout: 15_000,
            });
          } catch {
            // Process may already be gone.
          }
          for (let i = 0; i < 25; i++) {
            if (!isPidAlive(pid)) {
              break;
            }
            await sleep(100);
          }
        } else {
          process.kill(pid, force ? "SIGKILL" : "SIGTERM");
          await sleep(500);
          if (isPidAlive(pid)) {
            process.kill(pid, "SIGKILL");
          }
        }
      } catch {
        // ignore
      }
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

  private findPidsOnPort(port: number): number[] {
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

      const out = execFileSync("ss", ["-ltnp", `sport = :${port}`], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
      });
      const pids = new Set<number>();
      for (const match of out.matchAll(/pid=(\d+)/g)) {
        pids.add(Number(match[1]));
      }
      return [...pids];
    } catch {
      return [];
    }
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

  async reload(onProgress?: StartProgress): Promise<ServerStatus> {
    this.beginBoot("reload", "Reloading llama-server…");
    const report = (msg: string) => {
      this.updateBootMessage(msg);
      onProgress?.(msg);
    };
    try {
      report("Stopping server for reload…");
      await this.stop(true);
      await sleep(400);
      // Keep boot active across inner start — don't nest beginBoot/endBoot via start().
      return await this.startInner(undefined, report);
    } finally {
      this.endBoot();
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
    fs.writeFileSync(getLockPath(), JSON.stringify(data, null, 2), "utf8");
  }

  private clearLock(): void {
    try {
      fs.unlinkSync(getLockPath());
    } catch {
      // ignore
    }
  }
}

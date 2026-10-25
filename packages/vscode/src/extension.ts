import {
  clearOverthinkingTokenCacheForEndpoint,
  ConfigFile,
  connectionHint,
  DownloadAbortError,
  downloadManager,
  describeConfigLocation,
  ensureDirs,
  formatRemoteDiagnostics,
  getInstallDir,
  getLockDir,
  getModelsDir,
  InstallCancelledError,
  installNotificationPercent,
  LaunchToken,
  LAUNCH_IN_PROGRESS_MSG,
  LlamaInstaller,
  type InstallPhase,
  type InstallProgress,
  type ProgressReporter,
  nixOsIncompatibilityHint,
  originOf,
  remoteLifecycleMessage,
  remoteStateLabel,
  remoteStateUsable,
  rocmRuntimeMissingReason,
  setHfTokenProvider,
  setServerApiKeyProvider,
  shortBuild,
  syclRuntimeMissingReason,
  PerfStats,
  ProcessManager,
  SettingsStore,
  chatTemplateFileProblem,
  recommendedMaxDraftTokens,
  speculativeUsesMtp,
  speculativeUsesNgram,
  UiBackend,
} from "@llama-aio/core";
import * as os from "os";
import { formatDownloadStatusLine } from "./downloadStatus";
import { openModelFileDialog, pickDraftModelFromLibrary, pickMmprojFromLibrary } from "./modelPicker";
import { SettingsViewProvider } from "./settingsView";
import * as path from "path";
import * as vscode from "vscode";
import { LlamaAioChatProvider } from "./chatProvider";
import { promptUseInCopilotChat } from "./copilotChatPrompt";
import {
  copyServerCommandLine,
  initServerDiagnostics,
  openServerLog,
  reportLaunchFailure,
  showServerOutput,
} from "./serverDiagnostics";
import { downloadStarterModel, HuggingFaceClient } from "./huggingFace";
import { WikipediaLookupTool, syncWikipediaLookupContext, WIKIPEDIA_LOOKUP_TOOL_NAME } from "./wikipediaTool";

let chatProvider: LlamaAioChatProvider | undefined;

export function activate(context: vscode.ExtensionContext): void {
  // Configuration lives in ~/.llama-aio-vs/config.json, shared with the TUI.
  const configFile = new ConfigFile();
  void configFile.ensureExists().catch(() => undefined);
  context.subscriptions.push(configFile.watch(), { dispose: () => configFile.dispose() });

  const store = new SettingsStore(configFile);
  const config = store.getConfig();
  ensureDirs(getInstallDir(config), getModelsDir(config), getLockDir());

  const processManager = new ProcessManager(store);
  context.subscriptions.push(initServerDiagnostics(processManager));
  const perf = new PerfStats();
  perf.setSpeculativeMode(
    store.isRemote()
      ? undefined
      : (() => {
          const mode = store.getState().loadSettings.speculativeMode;
          return mode === "off" ? "off" : mode;
        })()
  );
  context.subscriptions.push(perf);
  // Refresh GGUF capability limits for the already-selected model.
  const existingModel = store.getState().selectedModelPath;
  if (existingModel) {
    void store.applySelectedModel(existingModel).catch(() => {
      // ignore unreadable models at activate
    });
  }
  void store.refreshCapabilitiesIfStale().catch(() => undefined);

  const installer = new LlamaInstaller(store);
  const hf = new HuggingFaceClient(store);

  let settingsView!: SettingsViewProvider;
  /** Set below, once {@link runLaunch} exists; the pickers run only after activation. */
  let afterModelSelected: (selected: string | undefined) => Promise<void> = async () => undefined;

  const downloadStarter = async () => {
    try {
      const selected = await downloadStarterModel(hf, store);
      await afterModelSelected(selected);
    } catch (e) {
      if (e instanceof DownloadAbortError) {
        return;
      }
      vscode.window.showErrorMessage(
        `Starter download failed: ${e instanceof Error ? e.message : String(e)}\n` +
          `Try “Download from Hugging Face…” instead.`
      );
    }
  };

  const openGgufFile = async () => {
    try {
      const selected = await openModelFileDialog(store);
      await afterModelSelected(selected);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Open file failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  const pickDraftModel = async () => {
    try {
      const selected = await pickDraftModelFromLibrary(store);
      if (!selected) {
        return;
      }
      const {
        readModelCapabilities,
        isDflashDraftArchitecture,
        isMtpDraftArchitecture,
        isMtpSidecarFile,
        isMtpBakedInFile,
      } = await import("@llama-aio/core");
      let warn = "";
      let kind: "dflash" | "mtp" | "unknown" = "unknown";
      try {
        const caps = readModelCapabilities(selected);
        if (isDflashDraftArchitecture(caps.architecture)) {
          kind = "dflash";
        } else if (isMtpDraftArchitecture(caps.architecture) || isMtpSidecarFile({ path: selected })) {
          kind = "mtp";
        } else if (
          isMtpBakedInFile({ path: selected }) ||
          (caps.nextnPredictLayers && caps.nextnPredictLayers > 0 && (caps.blockCount || 0) >= 16)
        ) {
          warn =
            `${path.basename(selected)} looks like a full language GGUF with MTP baked in, not a sidecar drafter. ` +
            `Select it as the main model instead. Use as draft anyway?`;
        } else {
          warn =
            `Selected draft GGUF architecture is "${caps.architecture || "unknown"}" ` +
            `(DFlash expects general.architecture = dflash; Gemma 4 MTP is a sibling mtp-*.gguf). Continue anyway?`;
        }
      } catch {
        if (isMtpSidecarFile({ path: selected })) {
          kind = "mtp";
        } else if (isMtpBakedInFile({ path: selected })) {
          warn =
            `${path.basename(selected)} looks like a full language GGUF with MTP baked in, not a sidecar drafter. ` +
            `Select it as the main model instead. Use as draft anyway?`;
        } else {
          warn = "Could not read draft GGUF metadata. Use it anyway?";
        }
      }
      if (warn) {
        const proceed = await vscode.window.showWarningMessage(warn, "Use anyway", "Cancel");
        if (proceed !== "Use anyway") {
          return;
        }
      }
      const cur = store.getState().loadSettings;
      if (kind === "mtp") {
        const mode = speculativeUsesNgram(cur.speculativeMode) ? "ngram-mtp" : "mtp";
        await store.updateLoadSettings({
          draftModelPath: selected,
          speculativeMode: mode,
          maxDraftTokens: recommendedMaxDraftTokens(mode, cur.maxDraftTokens, true),
        });
        settingsView.postDraftModelSelected(selected);
        settingsView.syncSpeculativeMode();
        await settingsView.pushState();
        void vscode.window.showInformationMessage(`MTP drafter set to ${path.basename(selected)}`);
        return;
      }
      const mode = speculativeUsesNgram(cur.speculativeMode)
        ? "ngram-dflash"
        : cur.speculativeMode === "off" || speculativeUsesMtp(cur.speculativeMode)
          ? "dflash"
          : cur.speculativeMode;
      await store.updateLoadSettings({
        draftModelPath: selected,
        speculativeMode: mode,
        maxDraftTokens: recommendedMaxDraftTokens(mode, cur.maxDraftTokens),
      });
      // Immediate sidebar update — full pushState can lag (local library scan) so
      // the hint used to stay on "No draft model selected" after a successful pick.
      settingsView.postDraftModelSelected(selected);
      settingsView.syncSpeculativeMode();
      await settingsView.pushState();
      void vscode.window.showInformationMessage(`DFlash draft set to ${path.basename(selected)}`);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Select draft model failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  const pickMmproj = async () => {
    try {
      const selected = await pickMmprojFromLibrary(store);
      if (!selected) {
        return;
      }
      const { isMmprojFileName } = await import("@llama-aio/core");
      if (!isMmprojFileName(selected)) {
        const proceed = await vscode.window.showWarningMessage(
          "Selected file name does not look like an mmproj projector. Use it anyway?",
          "Use anyway",
          "Cancel"
        );
        if (proceed !== "Use anyway") {
          return;
        }
      }
      await store.updateLoadSettings({ mmprojPath: selected });
      settingsView.postMmprojSelected(selected);
      await settingsView.pushState();
      chatProvider?.notifyChanged();
      void vscode.window.showInformationMessage(`Vision projector set to ${path.basename(selected)}`);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Select vision projector failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  const pickChatTemplate = async () => {
    try {
      const modelsDir = getModelsDir(store.getConfig());
      const uris = await vscode.window.showOpenDialog({
        canSelectMany: false,
        canSelectFiles: true,
        canSelectFolders: false,
        openLabel: "Use template",
        title: "Select a Jinja chat template",
        filters: {
          "Jinja templates": ["jinja", "txt"],
          "All files": ["*"],
        },
        defaultUri: vscode.Uri.file(modelsDir || os.homedir()),
      });
      const selected = uris?.[0]?.fsPath;
      if (!selected) {
        return;
      }
      const problem = chatTemplateFileProblem({
        ...store.getState().loadSettings,
        chatTemplateSource: "file",
        chatTemplateFile: selected,
      });
      if (problem) {
        void vscode.window.showErrorMessage(problem);
        return;
      }
      await store.updateLoadSettings({
        chatTemplateSource: "file",
        chatTemplateFile: selected,
        chatTemplateJinja: true,
      });
      await settingsView.pushState();
      void vscode.window.showInformationMessage(`Chat template set to ${path.basename(selected)}`);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Select chat template failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  /**
   * The one start / reload flow behind every entry point (commands, model
   * pick, post-install, backend switch, auto-start): launch lock, optional
   * spill confirmation, progress toast mirrored into the sidebar, then chat /
   * sidebar / status-bar refresh and the Copilot hint. Returns true when the
   * server came up. A reload stops the running server first, so it doubles
   * as "restart".
   */
  const runLaunch = async (
    kind: "start" | "reload",
    opts: { confirmSpill?: boolean; progress?: boolean; failurePrefix?: string } = {}
  ): Promise<boolean> => {
    const title = kind === "reload" ? "Reloading llama-server…" : "Starting llama-server…";
    const token = processManager.claimLaunch(kind, title);
    if (!token) {
      void vscode.window.showWarningMessage(LAUNCH_IN_PROGRESS_MSG);
      return false;
    }
    let message: string;
    try {
      if (opts.confirmSpill && !(await settingsView.confirmIfMemorySpill())) {
        return false;
      }
      const run = (report?: (msg: string) => void) =>
        kind === "reload" ? processManager.reload(report, token) : processManager.start(undefined, report, token);
      const status =
        opts.progress === false
          ? await run()
          : await vscode.window.withProgress(
              { location: vscode.ProgressLocation.Notification, title: `Llama AIO: ${title}`, cancellable: false },
              async (progress) =>
                run((msg) => {
                  progress.report({ message: msg });
                  settingsView.postBootProgress(msg);
                })
            );
      message = status.message;
    } catch (e) {
      await reportLaunchFailure(opts.failurePrefix ?? (kind === "reload" ? "Reload failed" : "Start failed"), e);
      return false;
    } finally {
      processManager.releaseLaunch(token);
      chatProvider?.notifyChanged();
      await settingsView.pushState();
      void refreshStatusBar();
    }
    await promptUseInCopilotChat(store, message, context.globalState);
    return true;
  };

  /** Ask before restarting a server that is up, then restart it through {@link runLaunch}. */
  const offerRestart = async (question: string, action: string) => {
    if ((await vscode.window.showInformationMessage(question, action, "Later")) === action) {
      await runLaunch("reload", { failurePrefix: "Restart failed" });
    }
  };

  afterModelSelected = async (selected: string | undefined) => {
    await settingsView.pushState();
    chatProvider?.notifyChanged();
    if (!selected || store.isRemote()) {
      return;
    }
    const choice = await vscode.window.showInformationMessage(
      `Model selected:\n${selected}`,
      "Start / reload server",
      "Later"
    );
    if (choice === "Start / reload server") {
      await runLaunch((await processManager.isHttpReady()) ? "reload" : "start");
    }
  };

  const afterBackendInstall = async (wasReady: boolean) => {
    const info = installer.getInstalledInfoCached(0);
    void refreshStatusBar(true);
    const label = info.binaryVersion || info.tag || processManager.resolveBinary();
    if (info.binaryRunnable === false) {
      const hint =
        info.activeBackend === "sycl"
          ? syclRuntimeMissingReason()
          : info.activeBackend === "rocm"
            ? rocmRuntimeMissingReason()
            : nixOsIncompatibilityHint().replace(/\n+/g, " ");
      void vscode.window.showWarningMessage(
        `llama.cpp installed (${label}) but the binary cannot run on this system yet. ${hint}`
      );
    } else {
      vscode.window.showInformationMessage(
        `llama.cpp ready: ${label}` + (info.asset ? ` (${info.asset})` : "")
      );
    }
    chatProvider?.notifyChanged();
    await settingsView.pushState();
    if (info.binaryRunnable === false) {
      return;
    }
    // The installer stopped the server right before swapping the binaries.
    if (wasReady && store.getState().selectedModelPath) {
      const restart = await vscode.window.showInformationMessage(
        "Backend ready. Start the server again?",
        "Start server",
        "Later"
      );
      if (restart === "Start server") {
        await runLaunch("start");
      }
    }
  };

  let installBusy = false;
  let installAbort: AbortController | undefined;
  let replayInstall: (() => Promise<void>) | undefined;

  const postInstall = (payload: InstallProgress & {
    status: "idle" | "running" | "done" | "error" | "cancelled";
    error?: string;
  }) => {
    settingsView.postLlamaInstall(payload);
  };

  type InstallOutcome = "ok" | "cancelled" | "error";

  const executeInstall = async (
    title: string,
    run: (progress: ProgressReporter, signal: AbortSignal) => Promise<unknown>
  ): Promise<InstallOutcome> => {
    const ac = new AbortController();
    installAbort = ac;
    let phase: InstallPhase = "resolve";
    const reporter: ProgressReporter = {
      report(value) {
        if (!value.message) {
          return;
        }
        postInstall({ status: "running", phase, message: value.message });
      },
      onInstall(update) {
        if (update.phase) {
          phase = update.phase;
        }
        postInstall({ status: "running", phase, ...update });
      },
    };
    try {
      postInstall({ status: "running", phase: "resolve", message: "Starting…" });
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title,
          cancellable: true,
        },
        async (progress, token) => {
          const sub = token.onCancellationRequested(() => ac.abort());
          let noted = 0;
          const wrapped: ProgressReporter = {
            report(value) {
              reporter.report(value);
              if (value.message) {
                progress.report({ message: value.message });
              }
            },
            onInstall(update) {
              reporter.onInstall?.(update);
              const pct = installNotificationPercent(update);
              if (pct != null && pct > noted) {
                progress.report({ message: update.message, increment: pct - noted });
                noted = pct;
              } else if (update.message) {
                progress.report({ message: update.message });
              }
            },
          };
          try {
            await run(wrapped, ac.signal);
          } finally {
            sub.dispose();
          }
        }
      );
      postInstall({ status: "done" });
      return "ok";
    } catch (e) {
      if (ac.signal.aborted || e instanceof InstallCancelledError) {
        postInstall({ status: "cancelled", message: "Install cancelled." });
        return "cancelled";
      }
      const message = e instanceof Error ? e.message : String(e);
      postInstall({ status: "error", error: message, message });
      vscode.window.showErrorMessage(`Install failed: ${message}`);
      return "error";
    } finally {
      installAbort = undefined;
    }
  };

  /** What an install entry point wants to run once its own checks and dialogs are done. */
  type InstallJob = {
    title: string;
    run: (progress: ProgressReporter, signal: AbortSignal) => Promise<unknown>;
    /** How the sidebar's Retry repeats this install. */
    replay?: () => Promise<void>;
  };

  /**
   * The frame every install entry point shares: one install at a time,
   * `prepare` (checks, pickers) inside the busy guard so a double click cannot
   * open two dialogs, remembered Retry, error reporting, and the restart offer
   * when the server was up before. `prepare` returns undefined to stop early.
   */
  const runInstall = async (failureLabel: string, prepare: () => Promise<InstallJob | undefined>): Promise<void> => {
    if (installBusy) {
      return;
    }
    installBusy = true;
    try {
      const job = await prepare();
      if (!job) {
        return;
      }
      // Keep the server up during the download. The installer takes a cross-window
      // lock, then stops llama-server immediately before replacing the bin dir.
      const wasReady = !store.isRemote() && (await processManager.isHttpReady());
      if (job.replay) {
        replayInstall = job.replay;
      }
      if ((await executeInstall(job.title, job.run)) === "ok") {
        await afterBackendInstall(wasReady);
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      postInstall({ status: "error", error: message, message });
      vscode.window.showErrorMessage(`${failureLabel}: ${message}`);
    } finally {
      installBusy = false;
    }
  };

  /** Guard for tag / archive installs, which need a downloadable backend. */
  const refusePathBackend = (what: string): boolean => {
    if (installer.resolveActiveUiBackend() !== "path") {
      return false;
    }
    postInstall({ status: "idle" });
    vscode.window.showErrorMessage(
      `Switch to Vulkan, CUDA, or CPU before installing ${what}. System (PATH) uses a package-managed binary.`
    );
    return true;
  };

  const installLlamaCpp = (backendOverride?: UiBackend): Promise<void> =>
    runInstall("Install failed", async () => {
      if (backendOverride === "path") {
        await installer.setBackend("path");
        await settingsView.pushState();
        postInstall({ status: "idle" });
        if (!installer.hasBackendInstalled("path")) {
          vscode.window.showErrorMessage(
            "llama-server not found on PATH. Install a system package (e.g. nixpkgs llama-cpp) first."
          );
        } else {
          vscode.window.showInformationMessage(`Using system llama-server: ${processManager.resolveBinary()}`);
        }
        return undefined;
      }
      if (backendOverride) {
        await installer.setBackend(backendOverride);
      }
      const check = await installer.getUpdateCheck(true);
      // Only skip when we *know* latest and it is not newer. A failed refresh
      // used to look like "already on b#####" while the sidebar showed an update.
      if (
        installer.hasBackendInstalled(installer.resolveActiveUiBackend()) &&
        !check.checkFailed &&
        check.latestTag &&
        check.installedTag &&
        !check.updateAvailable
      ) {
        vscode.window.showInformationMessage(
          `Llama AIO: already on ${check.installedTag} (latest ${check.latestTag}).`
        );
        await settingsView.pushState();
        postInstall({ status: "idle" });
        return undefined;
      }
      return {
        title: backendOverride ? `Llama AIO: Installing ${backendOverride} build…` : "Llama AIO: Installing llama.cpp",
        run: (progress, signal) => installer.installOrUpgrade(progress, backendOverride, { signal }),
        replay: () => installLlamaCpp(backendOverride),
      };
    });

  const reinstallLlamaCpp = async (): Promise<void> => {
    const backend = installer.resolveActiveUiBackend();
    if (backend === "path") {
      postInstall({ status: "idle" });
      vscode.window.showInformationMessage(
        "System (PATH) backend has nothing to reinstall — update llama.cpp via your package manager."
      );
      return;
    }
    const tag = installer.readBackendVersion(backend).tag;
    if (!tag) {
      return installLlamaCpp(backend);
    }
    return runInstall("Reinstall failed", async () => ({
      title: `Llama AIO: Reinstalling ${tag} (${backend})…`,
      run: (progress, signal) => installer.installByTag(tag, progress, backend, { signal }),
      replay: reinstallLlamaCpp,
    }));
  };

  const installTagJob = (tag: string, backend: UiBackend): InstallJob => ({
    title: `Llama AIO: Installing ${tag} (${backend})…`,
    run: (progress, signal) => installer.installByTag(tag, progress, backend, { signal }),
    replay: (): Promise<void> => runInstall("Install by tag failed", async () => installTagJob(tag, backend)),
  });

  const installLlamaCppByTag = () =>
    runInstall("Install by tag failed", async () => {
      if (refusePathBackend("a release tag")) {
        return undefined;
      }
      const backend = installer.resolveActiveUiBackend();
      const tagInput = await vscode.window.showInputBox({
        title: "Install llama.cpp by release tag",
        prompt: `Uses backend “${backend}”. Paste a nightly (b10587), a stable tag (v0.2.0), or a releases URL.`,
        placeHolder: "b10587",
        ignoreFocusOut: true,
        validateInput: (v) => ((v || "").trim() ? undefined : "Enter a tag or releases URL"),
      });
      if (!tagInput) {
        postInstall({ status: "idle" });
        return undefined;
      }
      return installTagJob(tagInput.trim(), backend);
    });

  const installArchiveJob = (archivePath: string, backend: UiBackend, cudartPath?: string): InstallJob => ({
    title: `Llama AIO: Installing from ${path.basename(archivePath)}…`,
    run: (progress, signal) =>
      installer.installFromArchive(archivePath, { uiBackend: backend, cudartArchivePath: cudartPath, progress, signal }),
    replay: (): Promise<void> =>
      runInstall("Install from archive failed", async () => installArchiveJob(archivePath, backend, cudartPath)),
  });

  const installLlamaCppFromArchive = () =>
    runInstall("Install from archive failed", async () => {
      if (refusePathBackend("from an archive")) {
        return undefined;
      }
      const backend = installer.resolveActiveUiBackend();
      const picked = await vscode.window.showOpenDialog({
        canSelectMany: false,
        openLabel: "Install llama.cpp archive",
        filters: { Archives: ["zip", "gz", "tgz"], "All files": ["*"] },
        title: `Select llama.cpp binary archive for ${backend}`,
      });
      if (!picked?.length) {
        postInstall({ status: "idle" });
        return undefined;
      }
      const archivePath = picked[0].fsPath;
      let cudartPath: string | undefined;
      const base = path.basename(archivePath).toLowerCase();
      if (process.platform === "win32" && (/cuda/.test(base) || backend === "cuda")) {
        const addCudart = await vscode.window.showInformationMessage(
          "Windows CUDA builds usually need a matching cudart-*.zip. Select one now?",
          "Select cudart…",
          "Skip"
        );
        if (addCudart === "Select cudart…") {
          const cudart = await vscode.window.showOpenDialog({
            canSelectMany: false,
            openLabel: "Use cudart archive",
            filters: { Archives: ["zip"], "All files": ["*"] },
            title: "Select cudart-llama-bin-win-cuda-*.zip",
          });
          if (cudart?.length) {
            cudartPath = cudart[0].fsPath;
          }
        }
      }
      return installArchiveJob(archivePath, backend, cudartPath);
    });

  const cancelLlamaInstall = () => {
    installAbort?.abort();
  };

  const retryLlamaInstall = async () => {
    if (installBusy || !replayInstall) {
      if (!replayInstall) {
        postInstall({ status: "idle" });
      }
      return;
    }
    await replayInstall();
  };

  const switchBackend = async (backend: UiBackend) => {
    if (installBusy) {
      return;
    }
    const opt = installer.getUiBackendOptions().find((o) => o.id === backend);
    if (opt && !opt.available && backend !== "path") {
      postInstall({ status: "idle" });
      vscode.window.showErrorMessage(
        `Backend ${backend} is unavailable: ${opt.reason || "not supported on this machine"}`
      );
      return;
    }
    const previous = installer.resolveActiveUiBackend();
    if (previous === backend && installer.hasBackendInstalled(backend)) {
      postInstall({ status: "idle" });
      await settingsView.pushState();
      return;
    }

    await installer.setBackend(backend);
    void refreshStatusBar(true);
    const wasReady = !store.isRemote() && (await processManager.isHttpReady());

    if (backend !== "path" && !installer.hasBackendInstalled(backend)) {
      await installLlamaCpp(backend);
      return;
    }
    postInstall({ status: "idle" });
    chatProvider?.notifyChanged();
    await settingsView.pushState();
    if (backend === "path" && !installer.hasBackendInstalled("path")) {
      vscode.window.showErrorMessage(
        "llama-server not found on PATH. Install a system package (e.g. nixpkgs llama-cpp), " +
          "or enable nix-ld / steam-run and use a downloaded backend."
      );
      return;
    }
    if (wasReady && store.getState().selectedModelPath) {
      await offerRestart(
        backend === "path"
          ? `Switched to System (PATH). Restart the server to use ${processManager.resolveBinary()}?`
          : `Switched to ${backend}. Restart the server to use this binary?`,
        "Restart server"
      );
    } else {
      vscode.window.setStatusBarMessage(
        backend === "path"
          ? `Llama AIO: using PATH llama-server (${processManager.resolveBinary()})`
          : `Llama AIO: using ${backend} backend`,
        backend === "path" ? 5000 : 4000
      );
    }
  };

  settingsView = new SettingsViewProvider(
    context.extensionUri,
    store,
    processManager,
    installer,
    perf,
    async (token?: LaunchToken) => {
      // Sidebar may already hold `token`; commands claim inside reload when omitted.
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: "Llama AIO: Reloading llama-server…",
          cancellable: false,
        },
        async (progress) =>
          processManager.reload((msg) => {
            progress.report({ message: msg });
            settingsView.postBootProgress(msg);
          }, token)
      );
      chatProvider?.notifyChanged();
      await settingsView.pushState();
    },
    {
      downloadStarter,
      openGgufFile,
      pickDraftModel,
      pickMmproj,
      pickChatTemplate,
      installLlamaCpp: (backend?: UiBackend) => installLlamaCpp(backend),
      reinstallLlamaCpp,
      installLlamaCppByTag,
      installLlamaCppFromArchive,
      cancelLlamaInstall,
      retryLlamaInstall,
      switchBackend,
      selectModel: async (filePath: string) => {
        const selected = filePath.trim();
        if (!selected) {
          return;
        }
        await store.applySelectedModel(selected);
        await settingsView.pushState();
        chatProvider?.notifyChanged();
      },
    },
    () => chatProvider?.notifyChanged(),
    context.globalState
  );

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(SettingsViewProvider.viewType, settingsView)
  );

  chatProvider = new LlamaAioChatProvider(store, processManager, perf, context.extensionPath);
  context.subscriptions.push(
    vscode.lm.registerLanguageModelChatProvider("llama-aio", chatProvider)
  );
  try {
    context.subscriptions.push(
      vscode.lm.registerTool(WIKIPEDIA_LOOKUP_TOOL_NAME, new WikipediaLookupTool(store))
    );
  } catch (e) {
    console.warn(
      "Llama AIO: Wikipedia lookup tool was not registered:",
      e instanceof Error ? e.message : e
    );
  }
  syncWikipediaLookupContext(store);
  // Copilot Chat can be installed or removed while we run (sideload in
  // VSCodium / code-server); refresh the sidebar's "no Copilot Chat" hint.
  context.subscriptions.push(
    vscode.extensions.onDidChange(() => void settingsView.pushState())
  );
  context.subscriptions.push(store.onDidChange(() => syncWikipediaLookupContext(store)));

  context.subscriptions.push(
    store.onDidChangeExternally(() => {
      settingsView.syncSpeculativeMode();
      void settingsView.pushState();
      chatProvider?.notifyChanged();
    })
  );

  const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.command = "llamaAio.showStatus";
  statusBar.tooltip = "Llama AIO server status";
  context.subscriptions.push(statusBar);

  const downloadBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 90);
  downloadBar.command = "llamaAio.showDownloads";
  downloadBar.tooltip = "Show the download in the model library";
  context.subscriptions.push(downloadBar);
  context.subscriptions.push({
    dispose: downloadManager.subscribe((jobs) => {
      const line = formatDownloadStatusLine(jobs);
      if (!line) {
        downloadBar.hide();
        return;
      }
      downloadBar.text = line;
      downloadBar.show();
    }),
  });

  let serverReadyCache = false;
  // Shared with the sidebar: the installer caches the build summary and
  // invalidates it itself on install / backend switch.
  const currentBuildInfo = (force = false) => installer.getInstalledInfoCached(force ? 0 : undefined);
  const remoteModelLabel = () =>
    (processManager.remote.current()?.modelId || store.getServerPreferences().lastRemoteModelId || "")
      .split(/[/\\]/)
      .pop()
      ?.replace(/\.gguf$/i, "") || "";
  const paintRemoteStatusBar = () => {
    const info = processManager.remote.current();
    const target = store.getServerTarget();
    const usable = remoteStateUsable(info?.state);
    const model = remoteModelLabel();
    const rate = perf.get().genTokPerSec;
    const rateText = typeof rate === "number" && rate > 0 ? `${rate.toFixed(1)} tok/s` : "";
    statusBar.text = usable
      ? ["$(remote)", target.label, model, rateText].filter(Boolean).join(" · ")
      : `$(remote) ${target.label} · ${remoteStateLabel(info?.state)}`;
    const slots =
      typeof info?.slotsTotal === "number" && info.slotsTotal > 0
        ? `${info.slotsBusy ?? 0}/${info.slotsTotal} slots busy${info.requestsDeferred ? ` · ${info.requestsDeferred} queued` : ""}`
        : "";
    statusBar.tooltip = [
      info?.isLlamaCpp === false
        ? `OpenAI-compatible server at ${target.baseUrl}`
        : `Remote llama.cpp at ${target.baseUrl}`,
      [model, info?.buildInfo ? shortBuild(info.buildInfo) : "", info?.rttMs != null ? `${info.rttMs} ms` : ""]
        .filter(Boolean)
        .join(" · "),
      slots,
      info?.error ? `${info.error.message} ${connectionHint(info.error.kind, target.baseUrl)}` : "",
      "Click to open the sidebar",
    ]
      .filter(Boolean)
      .join("\n");
    return usable;
  };
  const refreshStatusBar = async (forceBuildInfo = false) => {
    if (store.isRemote()) {
      await processManager.remote.refresh();
      serverReadyCache = paintRemoteStatusBar();
      await settingsView.postStatusNow(serverReadyCache);
      statusBar.show();
      return;
    }
    const status = await processManager.refreshStatus();
    serverReadyCache = status.running || (await processManager.isHttpReady());
    currentBuildInfo(forceBuildInfo);
    statusBar.text = perf.statusBarText(serverReadyCache);
    // Same probe feeds the sidebar so a crash or a TUI stop clears "Server ready".
    await settingsView.postStatusNow(serverReadyCache);
    statusBar.tooltip = localTooltip();
    statusBar.show();
  };
  /** Local-mode tooltip. One builder, so perf updates do not drop the build lines. */
  const localTooltip = () => {
    const build = currentBuildInfo();
    return [
      "Llama AIO",
      `Endpoint: ${store.getEndpoint()}`,
      build.binaryVersion || build.tag ? `llama.cpp: ${build.binaryVersion || build.tag}` : "llama.cpp: not installed",
      build.resolvedBackend ? `Backend: ${build.resolvedBackend}` : undefined,
      build.asset ? `Asset: ${build.asset}` : undefined,
      ...perf.detailLines(),
    ]
      .filter(Boolean)
      .join("\n");
  };
  const API_PREFIX = "llama-aio.apiKey:";
  // Request headers need a sync lookup; Secret Storage is async. The cache is
  // refreshed on connect, on remote URL changes, and on Secret Storage events.
  const keyCache = new Map<string, string>();
  setServerApiKeyProvider((origin) => keyCache.get(origin));
  const loadKey = async (origin: string): Promise<string | undefined> => {
    try {
      const stored = await context.secrets.get(API_PREFIX + origin);
      if (stored) {
        keyCache.set(origin, stored);
      } else {
        keyCache.delete(origin);
      }
      return stored;
    } catch {
      // Secret Storage can be unavailable in some hosts.
      return keyCache.get(origin);
    }
  };
  settingsView.attachApiKeyStore({
    get(origin) {
      return keyCache.get(origin);
    },
    load: loadKey,
    async set(origin, value) {
      keyCache.set(origin, value);
      await context.secrets.store(API_PREFIX + origin, value);
    },
    async delete(origin) {
      keyCache.delete(origin);
      await context.secrets.delete(API_PREFIX + origin);
    },
  });

  // Hugging Face token: Secret Storage first, with the plaintext config value
  // migrated once so existing users keep their gated-repo access.
  const HF_TOKEN_SECRET = "llama-aio.hfToken";
  let hfTokenCache = "";
  setHfTokenProvider(() => hfTokenCache || undefined);
  const loadHfToken = async (): Promise<string | undefined> => {
    try {
      hfTokenCache = ((await context.secrets.get(HF_TOKEN_SECRET)) || "").trim();
    } catch {
      // Secret Storage can be unavailable; fall through to the legacy value.
      hfTokenCache = "";
    }
    if (!hfTokenCache) {
      const legacy = (store.getConfig().get<string>("hfToken") || "").trim();
      if (legacy) {
        hfTokenCache = legacy;
        try {
          await context.secrets.store(HF_TOKEN_SECRET, legacy);
          await store.getConfig().update("hfToken", "");
        } catch {
          // Keep working from the in-memory value even if storage fails.
        }
      }
    }
    return hfTokenCache || undefined;
  };
  settingsView.attachHfTokenStore({
    get: () => hfTokenCache || undefined,
    load: loadHfToken,
    async set(value) {
      hfTokenCache = value;
      await context.secrets.store(HF_TOKEN_SECRET, value);
    },
    async delete() {
      hfTokenCache = "";
      await context.secrets.delete(HF_TOKEN_SECRET);
    },
  });
  void loadHfToken();
  context.subscriptions.push(
    context.secrets.onDidChange((e) => {
      if (e.key === HF_TOKEN_SECRET) {
        void loadHfToken();
      }
    })
  );

  const preloadKey = async (url: string | undefined) => {
    const origin = originOf(url);
    if (origin) {
      await loadKey(origin);
    }
  };
  // A key set or forgotten in another window, or a remote URL switched by the
  // TUI, must not leave this window probing without (or with a stale) key.
  const reprobeIfCurrent = async (origin: string) => {
    if (store.isRemote() && originOf(store.getEndpoint()) === origin) {
      processManager.remote.reset();
      await processManager.remote.refresh(true);
      chatProvider?.notifyChanged();
      await settingsView.pushState();
    }
  };
  context.subscriptions.push(
    context.secrets.onDidChange(async (e) => {
      if (!e.key.startsWith(API_PREFIX)) {
        return;
      }
      const origin = e.key.slice(API_PREFIX.length);
      const before = keyCache.get(origin);
      if ((await loadKey(origin)) !== before) {
        await reprobeIfCurrent(origin);
      }
    })
  );
  let knownRemoteUrl = store.getServerPreferences().remoteUrl;
  context.subscriptions.push(
    store.onDidChange(() => {
      const url = store.getServerPreferences().remoteUrl;
      if (url === knownRemoteUrl) {
        return;
      }
      knownRemoteUrl = url;
      const origin = originOf(url);
      if (origin) {
        void loadKey(origin).then(() => reprobeIfCurrent(origin));
      }
    })
  );
  let seenRemoteModel = store.getServerPreferences().lastRemoteModelId;
  // What Copilot's model picker shows. Slot load and server tok/s also fire
  // remote changes; re-announcing the model on those made VS Code re-query
  // /v1/models and /props every few seconds while a shared server was busy.
  let remoteModelInfoKey = "";
  context.subscriptions.push(
    processManager.remote.onDidChange((info) => {
      if (info.modelId && seenRemoteModel && info.modelId !== seenRemoteModel) {
        clearOverthinkingTokenCacheForEndpoint(info.url);
      }
      if (info.modelId && info.modelId !== seenRemoteModel) {
        seenRemoteModel = info.modelId;
        void store.setLastRemoteModelId(info.modelId);
      }
      const key = [
        info.url,
        remoteStateUsable(info.state),
        info.modelId,
        // The served label carries the model family (e.g. Qwen3.8-Flash-Next),
        // which decides the picker's thinking modes.
        info.modelName,
        info.modelAlias,
        info.modelPath,
        JSON.stringify(info.modelAliases),
        info.slotContext,
        info.servedContext,
        info.lastKnownModel,
        info.vision,
        info.isLlamaCpp,
      ].join("|");
      if (key !== remoteModelInfoKey) {
        remoteModelInfoKey = key;
        chatProvider?.notifyChanged();
        void settingsView.pushState();
      } else {
        // Load-only change: patch the server card, skip the full rebuild.
        void settingsView.postStatusNow();
      }
    })
  );
  void (async () => {
    const configured = store.getServerPreferences().remoteUrl;
    if (configured) {
      await preloadKey(configured);
    }
    const migrated = await store.migrateLegacyRemoteHost(machineAddresses());
    if (migrated) {
      const choice = await vscode.window.showInformationMessage(
        `Llama AIO now treats ${migrated} as a remote server. The local host was reset to 127.0.0.1.`,
        "Undo"
      );
      if (choice === "Undo") {
        await store.undoLegacyRemoteHostMigration(migrated);
        processManager.remote.reset();
        chatProvider?.notifyChanged();
        await settingsView.pushState();
      }
    }
    void refreshStatusBar();
    // The first model query can land while the probe still lacks the served
    // label, which hides the picker's thinking modes. Once the API key is
    // loaded, probe and announce so Copilot re-reads the model information.
    if (store.isRemote()) {
      await processManager.remote.refresh();
      chatProvider?.notifyChanged();
    }
  })();
  // Skip a tick while the previous one is still waiting on a slow probe, so
  // an unreachable remote cannot stack up overlapping refreshes.
  let statusTickBusy = false;
  const interval = setInterval(() => {
    if (statusTickBusy) {
      return;
    }
    statusTickBusy = true;
    void refreshStatusBar()
      .catch(() => undefined)
      .finally(() => {
        statusTickBusy = false;
      });
  }, 5000);
  context.subscriptions.push({ dispose: () => clearInterval(interval) });

  let sidebarPerfTimer: NodeJS.Timeout | undefined;
  context.subscriptions.push(
    perf.onDidChange(() => {
      if (store.isRemote()) {
        paintRemoteStatusBar();
      } else {
        statusBar.text = perf.statusBarText(serverReadyCache);
      }
      const p = perf.get();
      statusBar.backgroundColor =
        p.contextLevel === "critical"
          ? new vscode.ThemeColor("statusBarItem.errorBackground")
          : p.contextLevel === "warn"
            ? new vscode.ThemeColor("statusBarItem.warningBackground")
            : undefined;
      if (!store.isRemote()) {
        statusBar.tooltip = localTooltip();
      }

      const alert = perf.consumeContextAlert();
      if (alert) {
        const msg =
          alert.level === "critical"
            ? store.isRemote()
            ? `Llama AIO: this chat fills ${alert.pct}% of the context window (${alert.used.toLocaleString()} of ${alert.limit.toLocaleString()} tokens). The next large request may fail — start a new chat. A larger window needs a server restart with a bigger --ctx-size.`
            : `Llama AIO: this chat fills ${alert.pct}% of the context window (${alert.used.toLocaleString()} of ${alert.limit.toLocaleString()} tokens). The next large request may fail — start a new chat or raise the Context window.`
            : `Llama AIO: this chat uses ${alert.pct}% of the context window (${alert.used.toLocaleString()} of ${alert.limit.toLocaleString()} tokens).`;
        if (alert.level === "critical") {
          void vscode.window.showWarningMessage(msg, "Open settings").then((c) => {
            if (c === "Open settings") {
              void vscode.commands.executeCommand("llamaAio.openSettings");
            }
          });
        } else {
          void vscode.window.showInformationMessage(msg);
        }
      }

      // Throttle sidebar updates while streaming. A full pushState re-probes
      // the binary and scans the model library, so live tok/s would lag or stall.
      if (sidebarPerfTimer) {
        return;
      }
      sidebarPerfTimer = setTimeout(() => {
        sidebarPerfTimer = undefined;
        settingsView.postPerf();
      }, 250);
    }),
    { dispose: () => clearTimeout(sidebarPerfTimer) }
  );

  const offerSwitchIfRemote = async (action: string): Promise<boolean> => {
    if (!store.isRemote()) {
      return false;
    }
    const choice = await vscode.window.showWarningMessage(
      remoteLifecycleMessage(action, store.getServerTarget().label),
      "Switch to this machine"
    );
    if (choice === "Switch to this machine") {
      await settingsView.switchToLocal();
    }
    return true;
  };

  context.subscriptions.push(
    vscode.commands.registerCommand("llamaAio.openSettings", async () => {
      await vscode.commands.executeCommand("llamaAio.settingsView.focus");
    }),
    vscode.commands.registerCommand("llamaAio.openConfigFile", async () => {
      await configFile.ensureExists();
      const uri = vscode.Uri.file(describeConfigLocation().file);
      await vscode.window.showTextDocument(uri);
    }),

    vscode.commands.registerCommand("llamaAio.installLlamaCpp", installLlamaCpp),
    vscode.commands.registerCommand("llamaAio.installLlamaCppByTag", installLlamaCppByTag),
    vscode.commands.registerCommand("llamaAio.installLlamaCppFromArchive", installLlamaCppFromArchive),
    vscode.commands.registerCommand("llamaAio.browseModels", (query?: string) => {
      const q = typeof query === "string" ? query.trim() : "";
      settingsView.focusLibrary({ filter: true, query: q, search: q.length >= 2 });
    }),
    vscode.commands.registerCommand("llamaAio.showDownloads", () =>
      settingsView.focusLibrary({ downloads: true })
    ),
    vscode.commands.registerCommand("llamaAio.openModelFile", openGgufFile),
    vscode.commands.registerCommand("llamaAio.selectLocalModel", () =>
      settingsView.focusLibrary({ filter: true })
    ),
    vscode.commands.registerCommand("llamaAio.removeSelectedModel", () =>
      settingsView.removeSelectedModel()
    ),
    vscode.commands.registerCommand("llamaAio.hideModel", () => settingsView.hideSelectedModel()),
    vscode.commands.registerCommand("llamaAio.cleanPartialDownloads", () =>
      settingsView.cleanPartialDownloads()
    ),
    vscode.commands.registerCommand("llamaAio.selectDraftModel", pickDraftModel),
    vscode.commands.registerCommand("llamaAio.selectMmproj", pickMmproj),
    vscode.commands.registerCommand("llamaAio.selectChatTemplate", pickChatTemplate),

    vscode.commands.registerCommand("llamaAio.startServer", async () => {
      if (!(await offerSwitchIfRemote("start"))) {
        await runLaunch("start", { confirmSpill: true });
      }
    }),

    vscode.commands.registerCommand("llamaAio.stopServer", async () => {
      if (await offerSwitchIfRemote("stop")) {
        return;
      }
      await processManager.stop(true);
      chatProvider?.notifyChanged();
      await settingsView.pushState();
      await refreshStatusBar();
      vscode.window.showInformationMessage("Llama AIO server stopped.");
    }),

    vscode.commands.registerCommand("llamaAio.reloadServer", async () => {
      if (!(await offerSwitchIfRemote("reload"))) {
        await runLaunch("reload", { confirmSpill: true });
      }
    }),

    vscode.commands.registerCommand("llamaAio.openServerLog", () => openServerLog()),
    vscode.commands.registerCommand("llamaAio.copyServerCommand", () => copyServerCommandLine()),
    vscode.commands.registerCommand("llamaAio.showServerOutput", () => {
      showServerOutput();
    }),

    vscode.commands.registerCommand("llamaAio.connectRemote", async () => {
      await vscode.commands.executeCommand("llamaAio.settingsView.focus");
      settingsView.openConnectForm();
    }),
    vscode.commands.registerCommand("llamaAio.switchToLocal", () => settingsView.switchToLocal()),
    vscode.commands.registerCommand("llamaAio.setApiKey", () => settingsView.promptApiKey()),
    vscode.commands.registerCommand("llamaAio.forgetApiKey", () => settingsView.forgetApiKey()),
    vscode.commands.registerCommand("llamaAio.copyConnectionDiagnostics", () => settingsView.copyDiagnostics()),

    vscode.commands.registerCommand("llamaAio.showStatus", async () => {
      if (store.isRemote()) {
        await processManager.remote.refresh(true);
        const target = store.getServerTarget();
        const text = formatRemoteDiagnostics(processManager.remote.current(), {
          url: target.baseUrl,
          label: target.label,
          hasApiKey: !!keyCache.get(originOf(target.baseUrl) || ""),
        });
        void vscode.window.showInformationMessage(text.split("\n").filter(Boolean).slice(0, 8).join(" | "));
        await settingsView.pushState();
        return;
      }
      const status = processManager.getStatus();
      const ready = await processManager.isHttpReady();
      const state = store.getState();
      const build = installer.getInstalledInfoCached();
      const lines = [
        `HTTP ready: ${ready}`,
        `Status: ${status.message}`,
        `Endpoint: ${store.getEndpoint()}`,
        `PID: ${status.pid ?? "—"}`,
        `Model: ${state.selectedModelPath || "—"}`,
        `llama.cpp: ${build.binaryVersion || build.tag || "unknown"}`,
        `Backend: ${build.resolvedBackend || build.configuredBackend}`,
        `Asset: ${build.asset || "—"}`,
        `Binary: ${processManager.resolveBinary()}`,
        `Models dir: ${getModelsDir(store.getConfig())}`,
        `n-cpu-moe: ${state.loadSettings.nCpuMoe}`,
        `ctx: ${state.loadSettings.contextLength}, ngl: ${state.loadSettings.gpuOffload}`,
        ...perf.detailLines(),
      ];
      vscode.window.showInformationMessage(lines.join(" | "));
      await settingsView.pushState();
    }),

    vscode.commands.registerCommand("llamaAio.viewLastCall", async () => {
      await settingsView.openLastRequestContext();
    }),
    // Keep old id as alias for any saved keybindings.
    vscode.commands.registerCommand("llamaAio.viewLastContext", async () => {
      await settingsView.openLastRequestContext();
    }),

    vscode.commands.registerCommand("llamaAio.viewLastResponse", async () => {
      await settingsView.openLastResponseTrace();
    })
  );

  if (
    !store.isRemote() &&
    store.getConfig().get<boolean>("autoStart", false) &&
    store.getState().selectedModelPath
  ) {
    // No toast at window open; failures still go to the output channel.
    void runLaunch("start", { progress: false, failurePrefix: "Auto-start failed" });
  }
}

function machineAddresses(): string[] {
  const out = ["127.0.0.1", "::1", "localhost"];
  for (const infos of Object.values(os.networkInterfaces())) {
    for (const info of infos || []) {
      if (info.address) {
        out.push(info.address);
      }
    }
  }
  return out;
}

export async function deactivate(): Promise<void> {
  // Intentionally do NOT stop the shared external llama-server.
  // It must survive VS Code window/folder switches and multi-window use.
}

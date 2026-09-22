import {
  ConfigFile,
  DownloadAbortError,
  downloadManager,
  describeConfigLocation,
  ensureDirs,
  getInstallDir,
  getLockDir,
  getModelsDir,
  LaunchToken,
  LAUNCH_IN_PROGRESS_MSG,
  LlamaInstaller,
  nixOsIncompatibilityHint,
  PerfStats,
  ProcessManager,
  SettingsStore,
  recommendedMaxDraftTokens,
  speculativeUsesMtp,
  speculativeUsesNgram,
  UiBackend,
} from "@llama-aio/core";
import { DownloadPanel, formatDownloadStatusLine } from "./downloadPanel";
import { openModelFileDialog, pickDownloadedModel, pickDraftModelFromLibrary, pickMmprojFromLibrary } from "./modelPicker";
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
import { browseAndDownloadModel, downloadStarterModel, HuggingFaceClient } from "./huggingFace";
import { WikipediaLookupTool, syncWikipediaLookupContext, WIKIPEDIA_LOOKUP_TOOL_NAME } from "./wikipediaTool";

let chatProvider: LlamaAioChatProvider | undefined;

async function afterModelSelected(
  selected: string | undefined,
  settingsView: SettingsViewProvider,
  processManager: ProcessManager,
  store: SettingsStore,
  globalState: vscode.Memento
): Promise<void> {
  await settingsView.pushState();
  chatProvider?.notifyChanged();
  if (!selected) {
    return;
  }

  const choice = await vscode.window.showInformationMessage(
    `Model selected:\n${selected}`,
    "Start / reload server",
    "Later"
  );
  if (choice === "Start / reload server") {
    const ready = await processManager.isHttpReady();
    const kind = ready ? "reload" : "start";
    const token = processManager.claimLaunch(
      kind,
      kind === "reload" ? "Reloading llama-server…" : "Starting llama-server…"
    );
    if (!token) {
      void vscode.window.showWarningMessage(LAUNCH_IN_PROGRESS_MSG);
      return;
    }
    try {
      const status = await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title:
            kind === "reload"
              ? "Llama AIO: Reloading llama-server…"
              : "Llama AIO: Starting llama-server…",
          cancellable: false,
        },
        async (progress) => {
          const report = (msg: string) => progress.report({ message: msg });
          if (ready) {
            return processManager.reload(report, token);
          }
          return processManager.start(undefined, report, token);
        }
      );
      chatProvider?.notifyChanged();
      await settingsView.pushState();
      await promptUseInCopilotChat(store, status.message, globalState);
    } catch (e) {
      await reportLaunchFailure(kind === "reload" ? "Reload failed" : "Start failed", e);
    } finally {
      processManager.releaseLaunch(token);
    }
  }
}

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
    (() => {
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

  const downloadFromHuggingFace = async (query?: string) => {
    try {
      const selected = await browseAndDownloadModel(hf, store, query);
      await afterModelSelected(selected, settingsView, processManager, store, context.globalState);
    } catch (e) {
      if (e instanceof DownloadAbortError) {
        return;
      }
      vscode.window.showErrorMessage(
        `Browse/download failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  const downloadStarter = async () => {
    try {
      const selected = await downloadStarterModel(hf, store);
      await afterModelSelected(selected, settingsView, processManager, store, context.globalState);
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
      await afterModelSelected(selected, settingsView, processManager, store, context.globalState);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Open file failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  const pickDownloaded = async () => {
    try {
      const selected = await pickDownloadedModel(hf, store, {
        cpuOnly: installer.resolveActiveUiBackend() === "cpu" || processManager.isCpuBackend(),
        llamaServerBinary: processManager.resolveBinary(),
        loadSettings: store.getState().loadSettings,
      });
      await afterModelSelected(selected, settingsView, processManager, store, context.globalState);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Select model failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  const showDownloads = async () => {
    DownloadPanel.show(store);
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

  const afterBackendInstall = async (wasReady: boolean) => {
    const info = installer.getInstalledInfo();
    void refreshStatusBar(true);
    const label = info.binaryVersion || info.tag || processManager.resolveBinary();
    if (info.binaryRunnable === false) {
      void vscode.window.showWarningMessage(
        `llama.cpp installed (${label}) but the binary cannot run on this system yet. ` +
          nixOsIncompatibilityHint().replace(/\n+/g, " ")
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
    if (wasReady && store.getState().selectedModelPath) {
      const restart = await vscode.window.showInformationMessage(
        "Backend ready. Start the server again?",
        "Start server",
        "Later"
      );
      if (restart === "Start server") {
        try {
          const status = await processManager.start();
          chatProvider?.notifyChanged();
          await settingsView.pushState();
          await promptUseInCopilotChat(store, status.message, context.globalState);
        } catch (e) {
          await reportLaunchFailure("Start failed", e);
        }
      }
    }
  };

  const installLlamaCpp = async (backendOverride?: UiBackend) => {
    try {
      if (backendOverride === "path") {
        await installer.setBackend("path");
        await settingsView.pushState();
        if (!installer.hasBackendInstalled("path")) {
          vscode.window.showErrorMessage(
            "llama-server not found on PATH. Install a system package (e.g. nixpkgs llama-cpp) first."
          );
        } else {
          vscode.window.showInformationMessage(
            `Using system llama-server: ${processManager.resolveBinary()}`
          );
        }
        return;
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
        return;
      }
      const wasReady = await processManager.isHttpReady();
      // Keep the server up during the download. The installer takes a cross-window
      // lock, then stops llama-server immediately before replacing vulkan\bin.
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: backendOverride
            ? `Llama AIO: Installing ${backendOverride} build…`
            : "Llama AIO: Installing llama.cpp",
          cancellable: false,
        },
        async (progress) => installer.installOrUpgrade(progress, backendOverride)
      );
      await afterBackendInstall(wasReady);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Install failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  const reinstallLlamaCpp = async () => {
    try {
      const backend = installer.resolveActiveUiBackend();
      if (backend === "path") {
        vscode.window.showInformationMessage(
          "System (PATH) backend has nothing to reinstall — update llama.cpp via your package manager."
        );
        return;
      }
      const tag = installer.readBackendVersion(backend).tag;
      if (!tag) {
        await installLlamaCpp(backend);
        return;
      }
      const wasReady = await processManager.isHttpReady();
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Llama AIO: Reinstalling ${tag} (${backend})…`,
          cancellable: false,
        },
        async (progress) => installer.installByTag(tag, progress, backend)
      );
      await afterBackendInstall(wasReady);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Reinstall failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  const installLlamaCppByTag = async () => {
    try {
      const backend = installer.resolveActiveUiBackend();
      if (backend === "path") {
        vscode.window.showErrorMessage(
          "Switch to Vulkan, CUDA, or CPU before installing a release tag. " +
            "System (PATH) uses a package-managed binary."
        );
        return;
      }
      const tagInput = await vscode.window.showInputBox({
        title: "Install llama.cpp by release tag",
        prompt: `Uses backend “${backend}”. Paste a nightly (b10587), a stable tag (v0.2.0), or a releases URL.`,
        placeHolder: "b10587",
        ignoreFocusOut: true,
        validateInput: (v) => {
          const t = (v || "").trim();
          if (!t) {
            return "Enter a tag or releases URL";
          }
          return undefined;
        },
      });
      if (!tagInput) {
        return;
      }

      const wasReady = await processManager.isHttpReady();
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Llama AIO: Installing ${tagInput.trim()} (${backend})…`,
          cancellable: false,
        },
        async (progress) => installer.installByTag(tagInput, progress, backend)
      );
      await afterBackendInstall(wasReady);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Install by tag failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  const installLlamaCppFromArchive = async () => {
    try {
      const backend = installer.resolveActiveUiBackend();
      if (backend === "path") {
        vscode.window.showErrorMessage(
          "Switch to Vulkan, CUDA, or CPU before installing from an archive. " +
            "System (PATH) uses a package-managed binary."
        );
        return;
      }
      const picked = await vscode.window.showOpenDialog({
        canSelectMany: false,
        openLabel: "Install llama.cpp archive",
        filters: {
          Archives: ["zip", "gz", "tgz"],
          "All files": ["*"],
        },
        title: `Select llama.cpp binary archive for ${backend}`,
      });
      if (!picked?.length) {
        return;
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

      const wasReady = await processManager.isHttpReady();
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Llama AIO: Installing from ${path.basename(archivePath)}…`,
          cancellable: false,
        },
        async (progress) =>
          installer.installFromArchive(archivePath, {
            uiBackend: backend,
            cudartArchivePath: cudartPath,
            progress,
          })
      );
      await afterBackendInstall(wasReady);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Install from archive failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  const switchBackend = async (backend: UiBackend) => {
    const opt = installer.getUiBackendOptions().find((o) => o.id === backend);
    if (opt && !opt.available && backend !== "path") {
      vscode.window.showErrorMessage(
        `Backend ${backend} is unavailable: ${opt.reason || "not supported on this machine"}`
      );
      return;
    }
    const previous = installer.resolveActiveUiBackend();
    if (previous === backend && installer.hasBackendInstalled(backend)) {
      await settingsView.pushState();
      return;
    }

    await installer.setBackend(backend);
    void refreshStatusBar(true);
    const wasReady = await processManager.isHttpReady();

    if (backend === "path") {
      chatProvider?.notifyChanged();
      await settingsView.pushState();
      if (!installer.hasBackendInstalled("path")) {
        vscode.window.showErrorMessage(
          "llama-server not found on PATH. Install a system package (e.g. nixpkgs llama-cpp), " +
            "or enable nix-ld / steam-run and use a downloaded backend."
        );
        return;
      }
      if (wasReady && store.getState().selectedModelPath) {
        const restart = await vscode.window.showInformationMessage(
          `Switched to System (PATH). Restart the server to use ${processManager.resolveBinary()}?`,
          "Restart server",
          "Later"
        );
        if (restart === "Restart server") {
          try {
            await processManager.stop(true);
            const status = await processManager.start();
            chatProvider?.notifyChanged();
            await settingsView.pushState();
            await promptUseInCopilotChat(store, status.message, context.globalState);
          } catch (e) {
            await reportLaunchFailure("Restart failed", e);
          }
        }
      } else {
        vscode.window.setStatusBarMessage(
          `Llama AIO: using PATH llama-server (${processManager.resolveBinary()})`,
          5000
        );
      }
      return;
    }

    if (installer.hasBackendInstalled(backend)) {
      chatProvider?.notifyChanged();
      await settingsView.pushState();
      if (wasReady && store.getState().selectedModelPath) {
        const restart = await vscode.window.showInformationMessage(
          `Switched to ${backend}. Restart the server to use this binary?`,
          "Restart server",
          "Later"
        );
        if (restart === "Restart server") {
          try {
            await processManager.stop(true);
            const status = await processManager.start();
            chatProvider?.notifyChanged();
            await settingsView.pushState();
            await promptUseInCopilotChat(store, status.message, context.globalState);
          } catch (e) {
            await reportLaunchFailure("Restart failed", e);
          }
        }
      } else {
        vscode.window.setStatusBarMessage(`Llama AIO: using ${backend} backend`, 4000);
      }
      return;
    }

    await installLlamaCpp(backend);
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
      downloadFromHuggingFace,
      downloadStarter,
      openGgufFile,
      pickDownloaded,
      pickDraftModel,
      pickMmproj,
      installLlamaCpp: (backend?: UiBackend) => installLlamaCpp(backend),
      reinstallLlamaCpp,
      installLlamaCppByTag,
      installLlamaCppFromArchive,
      switchBackend,
      showDownloads,
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
  downloadBar.tooltip = "Llama AIO downloads";
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
  // The installed-build summary only changes on install / backend switch. Do
  // not re-probe the binary and re-detect GPUs on every 5 s tick — refresh it
  // on demand (after installs) and at most once a minute otherwise.
  const BUILD_INFO_TTL_MS = 60_000;
  let buildInfoAt = 0;
  let buildInfo: ReturnType<LlamaInstaller["getInstalledInfo"]> | undefined;
  const currentBuildInfo = (force = false) => {
    if (force || !buildInfo || Date.now() - buildInfoAt > BUILD_INFO_TTL_MS) {
      buildInfo = installer.getInstalledInfo();
      buildInfoAt = Date.now();
    }
    return buildInfo;
  };
  const refreshStatusBar = async (forceBuildInfo = false) => {
    const status = processManager.getStatus();
    serverReadyCache = status.running || (await processManager.isHttpReady());
    const build = currentBuildInfo(forceBuildInfo);
    statusBar.text = perf.statusBarText(serverReadyCache);
    statusBar.tooltip = [
      "Llama AIO",
      `Endpoint: ${store.getEndpoint()}`,
      build.binaryVersion || build.tag
        ? `llama.cpp: ${build.binaryVersion || build.tag}`
        : "llama.cpp: not installed",
      build.resolvedBackend ? `Backend: ${build.resolvedBackend}` : undefined,
      build.asset ? `Asset: ${build.asset}` : undefined,
      ...perf.detailLines(),
    ]
      .filter(Boolean)
      .join("\n");
    statusBar.show();
  };
  void refreshStatusBar();
  const interval = setInterval(() => void refreshStatusBar(), 5000);
  context.subscriptions.push({ dispose: () => clearInterval(interval) });

  let sidebarPerfTimer: NodeJS.Timeout | undefined;
  context.subscriptions.push(
    perf.onDidChange(() => {
      statusBar.text = perf.statusBarText(serverReadyCache);
      const p = perf.get();
      statusBar.backgroundColor =
        p.contextLevel === "critical"
          ? new vscode.ThemeColor("statusBarItem.errorBackground")
          : p.contextLevel === "warn"
            ? new vscode.ThemeColor("statusBarItem.warningBackground")
            : undefined;
      statusBar.tooltip = [
        "Llama AIO",
        `Endpoint: ${store.getEndpoint()}`,
        ...perf.detailLines(),
      ].join("\n");

      const alert = perf.consumeContextAlert();
      if (alert) {
        const msg =
          alert.level === "critical"
            ? `Llama AIO context nearly full: ${alert.used.toLocaleString()} / ${alert.limit.toLocaleString()} tokens (${alert.pct}%). The next large request may fail — start a new chat or raise Context Length.`
            : `Llama AIO context running low: ${alert.used.toLocaleString()} / ${alert.limit.toLocaleString()} tokens (${alert.pct}%).`;
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
    vscode.commands.registerCommand("llamaAio.browseModels", (query?: string) =>
      downloadFromHuggingFace(typeof query === "string" ? query : undefined)
    ),
    vscode.commands.registerCommand("llamaAio.showDownloads", showDownloads),
    vscode.commands.registerCommand("llamaAio.openModelFile", openGgufFile),
    vscode.commands.registerCommand("llamaAio.selectLocalModel", pickDownloaded),
    vscode.commands.registerCommand("llamaAio.selectDraftModel", pickDraftModel),
    vscode.commands.registerCommand("llamaAio.selectMmproj", pickMmproj),

    vscode.commands.registerCommand("llamaAio.startServer", async () => {
      const token = processManager.claimLaunch("start", "Starting llama-server…");
      if (!token) {
        void vscode.window.showWarningMessage(LAUNCH_IN_PROGRESS_MSG);
        return;
      }
      try {
        if (!(await settingsView.confirmIfMemorySpill())) {
          return;
        }
        const status = await vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Notification,
            title: "Llama AIO: Starting llama-server…",
            cancellable: false,
          },
          async (progress) =>
            processManager.start(
              undefined,
              (msg) => progress.report({ message: msg }),
              token
            )
        );
        chatProvider?.notifyChanged();
        await settingsView.pushState();
        await refreshStatusBar();
        await promptUseInCopilotChat(store, status.message, context.globalState);
      } catch (e) {
        await reportLaunchFailure("Start failed", e);
      } finally {
        processManager.releaseLaunch(token);
      }
    }),

    vscode.commands.registerCommand("llamaAio.stopServer", async () => {
      await processManager.stop(true);
      chatProvider?.notifyChanged();
      await settingsView.pushState();
      await refreshStatusBar();
      vscode.window.showInformationMessage("Llama AIO server stopped.");
    }),

    vscode.commands.registerCommand("llamaAio.reloadServer", async () => {
      const token = processManager.claimLaunch("reload", "Reloading llama-server…");
      if (!token) {
        void vscode.window.showWarningMessage(LAUNCH_IN_PROGRESS_MSG);
        return;
      }
      try {
        if (!(await settingsView.confirmIfMemorySpill())) {
          return;
        }
        const status = await vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Notification,
            title: "Llama AIO: Reloading llama-server…",
            cancellable: false,
          },
          async (progress) =>
            processManager.reload((msg) => progress.report({ message: msg }), token)
        );
        chatProvider?.notifyChanged();
        await settingsView.pushState();
        await refreshStatusBar();
        await promptUseInCopilotChat(store, status.message, context.globalState);
      } catch (e) {
        await reportLaunchFailure("Reload failed", e);
      } finally {
        processManager.releaseLaunch(token);
      }
    }),

    vscode.commands.registerCommand("llamaAio.openServerLog", () => openServerLog()),
    vscode.commands.registerCommand("llamaAio.copyServerCommand", () => copyServerCommandLine()),
    vscode.commands.registerCommand("llamaAio.showServerOutput", () => {
      showServerOutput();
    }),

    vscode.commands.registerCommand("llamaAio.showStatus", async () => {
      const status = processManager.getStatus();
      const ready = await processManager.isHttpReady();
      const state = store.getState();
      const build = installer.getInstalledInfo();
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

  if (store.getConfig().get<boolean>("autoStart", false) && store.getState().selectedModelPath) {
    void processManager
      .start()
      .then(async (status) => {
        chatProvider?.notifyChanged();
        void refreshStatusBar();
        await promptUseInCopilotChat(store, status.message, context.globalState);
      })
      .catch((err) => {
        void refreshStatusBar();
        void reportLaunchFailure("Auto-start failed", err);
      });
  }
}

export async function deactivate(): Promise<void> {
  // Intentionally do NOT stop the shared external llama-server.
  // It must survive VS Code window/folder switches and multi-window use.
}

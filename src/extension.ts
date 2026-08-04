import { LlamaInstaller, UiBackend } from "./llamaInstaller";
import { openModelFileDialog, pickDownloadedModel } from "./modelPicker";
import { ensureDirs, getInstallDir, getLockDir, getModelsDir } from "./paths";
import { PerfStats } from "./perfStats";
import { LaunchToken, LAUNCH_IN_PROGRESS_MSG, ProcessManager } from "./processManager";
import { SettingsStore } from "./settings";
import { SettingsViewProvider } from "./settingsView";
import * as path from "path";
import * as vscode from "vscode";
import { LlamaAioChatProvider } from "./chatProvider";
import { promptUseInCopilotChat } from "./copilotChatPrompt";
import { browseAndDownloadModel, downloadStarterModel, HuggingFaceClient } from "./huggingFace";

let chatProvider: LlamaAioChatProvider | undefined;

async function afterModelSelected(
  selected: string | undefined,
  settingsView: SettingsViewProvider,
  processManager: ProcessManager,
  store: SettingsStore
): Promise<void> {
  if (!selected) {
    return;
  }
  await settingsView.pushState();
  chatProvider?.notifyChanged();

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
      await promptUseInCopilotChat(store, status.message);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Start failed: ${e instanceof Error ? e.message : String(e)}`
      );
    } finally {
      processManager.releaseLaunch(token);
    }
  }
}

export function activate(context: vscode.ExtensionContext): void {
  const store = new SettingsStore(context);
  const config = store.getConfig();
  ensureDirs(getInstallDir(config), getModelsDir(config), getLockDir());

  const processManager = new ProcessManager(store);
  const perf = new PerfStats();
  perf.setSpeculativeMode(
    store.getState().loadSettings.speculativeMode === "mtp" ? "mtp" : "off"
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

  void store.migrateChatContextIfNeeded().then(async (changed) => {
    if (!changed) {
      return;
    }
    const slot = store.getSlotContextSize();
    const reload = await vscode.window.showInformationMessage(
      `Llama AIO updated chat defaults (single slot, larger context → ${slot} tokens/request). Reload the server to apply?`,
      "Reload server",
      "Later"
    );
    if (reload === "Reload server" && store.getState().selectedModelPath) {
      try {
        const status = await processManager.reload();
        chatProvider?.notifyChanged();
        await promptUseInCopilotChat(
          store,
          status.message || "Llama AIO server reloaded with chat-friendly context."
        );
      } catch (e) {
        vscode.window.showWarningMessage(
          `Reload failed: ${e instanceof Error ? e.message : String(e)}. Use Reload in the Llama AIO sidebar.`
        );
      }
    }
  });
  const installer = new LlamaInstaller(store);
  const hf = new HuggingFaceClient(store);

  let settingsView!: SettingsViewProvider;

  const downloadFromHuggingFace = async () => {
    try {
      const selected = await browseAndDownloadModel(hf, store);
      await afterModelSelected(selected, settingsView, processManager, store);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Browse/download failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  const downloadStarter = async () => {
    try {
      const selected = await downloadStarterModel(hf, store);
      await afterModelSelected(selected, settingsView, processManager, store);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Starter download failed: ${e instanceof Error ? e.message : String(e)}\n` +
          `Try “Download from Hugging Face…” instead.`
      );
    }
  };

  const openGgufFile = async () => {
    try {
      const selected = await openModelFileDialog(store);
      await afterModelSelected(selected, settingsView, processManager, store);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Open file failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  const pickDownloaded = async () => {
    try {
      const selected = await pickDownloadedModel(hf, store);
      await afterModelSelected(selected, settingsView, processManager, store);
    } catch (e) {
      vscode.window.showErrorMessage(
        `Select model failed: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  };

  const afterBackendInstall = async (wasReady: boolean) => {
    const info = installer.getInstalledInfo();
    const label = info.binaryVersion || info.tag || processManager.resolveBinary();
    vscode.window.showInformationMessage(
      `llama.cpp ready: ${label}` + (info.asset ? ` (${info.asset})` : "")
    );
    chatProvider?.notifyChanged();
    await settingsView.pushState();
    if (wasReady && store.getState().selectedModelPath) {
      const restart = await vscode.window.showInformationMessage(
        "Backend ready. Start the server again?",
        "Start server",
        "Later"
      );
      if (restart === "Start server") {
        const status = await processManager.start();
        chatProvider?.notifyChanged();
        await settingsView.pushState();
        await promptUseInCopilotChat(store, status.message);
      }
    }
  };

  const installLlamaCpp = async (backendOverride?: UiBackend) => {
    try {
      if (backendOverride) {
        await installer.setBackend(backendOverride);
      }
      const check = await installer.getUpdateCheck(true);
      if (
        installer.hasBackendInstalled(installer.resolveActiveUiBackend()) &&
        !check.updateAvailable &&
        check.installedTag
      ) {
        vscode.window.showInformationMessage(
          `Llama AIO: already on ${check.installedTag}` +
            (check.latestTag ? ` (latest ${check.latestTag}).` : ".")
        );
        await settingsView.pushState();
        return;
      }
      const wasReady = await processManager.isHttpReady();
      if (wasReady) {
        await processManager.stop(true);
      }
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
      const tag = installer.readBackendVersion(backend).tag;
      if (!tag) {
        await installLlamaCpp(backend);
        return;
      }
      const wasReady = await processManager.isHttpReady();
      if (wasReady) {
        await processManager.stop(true);
      }
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
      const tagInput = await vscode.window.showInputBox({
        title: "Install llama.cpp by release tag",
        prompt: `Direct download (no GitHub API). Uses backend “${backend}”. Paste a tag (b10154) or releases URL.`,
        placeHolder: "b10154",
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
      if (wasReady) {
        await processManager.stop(true);
      }
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
      if (wasReady) {
        await processManager.stop(true);
      }
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
    if (opt && !opt.available) {
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
    const wasReady = await processManager.isHttpReady();

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
            await promptUseInCopilotChat(store, status.message);
          } catch (e) {
            vscode.window.showErrorMessage(
              `Restart failed: ${e instanceof Error ? e.message : String(e)}`
            );
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
      const status = await vscode.window.withProgress(
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
      // Keep the Copilot prompt outside withProgress — awaiting the info dialog
      // inside would leave the "Reloading…" toast up while the server is already ready.
      chatProvider?.notifyChanged();
      await promptUseInCopilotChat(store, status.message);
    },
    {
      downloadFromHuggingFace,
      downloadStarter,
      openGgufFile,
      pickDownloaded,
      installLlamaCpp: (backend?: UiBackend) => installLlamaCpp(backend),
      reinstallLlamaCpp,
      installLlamaCppByTag,
      installLlamaCppFromArchive,
      switchBackend,
    },
    () => chatProvider?.notifyChanged()
  );

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(SettingsViewProvider.viewType, settingsView)
  );

  chatProvider = new LlamaAioChatProvider(store, processManager, perf, context.extensionPath);
  context.subscriptions.push(
    vscode.lm.registerLanguageModelChatProvider("llama-aio", chatProvider)
  );

  const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.command = "llamaAio.showStatus";
  statusBar.tooltip = "Llama AIO server status";
  context.subscriptions.push(statusBar);

  let serverReadyCache = false;
  const refreshStatusBar = async () => {
    const status = processManager.getStatus();
    serverReadyCache = status.running || (await processManager.isHttpReady());
    const build = installer.getInstalledInfo();
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

      // Throttle sidebar updates while streaming.
      if (sidebarPerfTimer) {
        return;
      }
      sidebarPerfTimer = setTimeout(() => {
        sidebarPerfTimer = undefined;
        void settingsView.pushState();
      }, 400);
    }),
    { dispose: () => clearTimeout(sidebarPerfTimer) }
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("llamaAio.openSettings", async () => {
      await vscode.commands.executeCommand("llamaAio.settingsView.focus");
    }),

    vscode.commands.registerCommand("llamaAio.installLlamaCpp", installLlamaCpp),
    vscode.commands.registerCommand("llamaAio.installLlamaCppByTag", installLlamaCppByTag),
    vscode.commands.registerCommand("llamaAio.installLlamaCppFromArchive", installLlamaCppFromArchive),
    vscode.commands.registerCommand("llamaAio.browseModels", downloadFromHuggingFace),
    vscode.commands.registerCommand("llamaAio.openModelFile", openGgufFile),
    vscode.commands.registerCommand("llamaAio.selectLocalModel", pickDownloaded),

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
        await promptUseInCopilotChat(store, status.message);
      } catch (e) {
        vscode.window.showErrorMessage(
          `Start failed: ${e instanceof Error ? e.message : String(e)}`
        );
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
        await promptUseInCopilotChat(store, status.message);
      } catch (e) {
        vscode.window.showErrorMessage(
          `Reload failed: ${e instanceof Error ? e.message : String(e)}`
        );
      } finally {
        processManager.releaseLaunch(token);
      }
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
    void processManager.start().then(async (status) => {
      chatProvider?.notifyChanged();
      void refreshStatusBar();
      await promptUseInCopilotChat(store, status.message);
    });
  }
}

export async function deactivate(): Promise<void> {
  // Intentionally do NOT stop the shared external llama-server.
  // It must survive VS Code window/folder switches and multi-window use.
}

import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";
import { promptUseInCopilotChat } from "./copilotChatPrompt";
import { detectGpuMemory } from "./gpuInfo";
import { LlamaInstaller, UiBackend } from "./llamaInstaller";
import { estimateMemory, memoryEstimateInputs } from "./memoryEstimate";
import { listActiveModelSourceDirs, listLocalModelEntries } from "./modelLibrary";
import { getModelsDir } from "./paths";
import { PerfStats } from "./perfStats";
import { ProcessManager } from "./processManager";
import { SettingsStore } from "./settings";
import { resolveLaunchMode } from "./externalTerminal";
import { LlamaLoadSettings, RequestSettings } from "./types";
import { STARTER_MODEL } from "./huggingFace";

export type ModelActions = {
  downloadFromHuggingFace: () => Promise<void>;
  downloadStarter: () => Promise<void>;
  openGgufFile: () => Promise<void>;
  pickDownloaded: () => Promise<void>;
  installLlamaCpp: (backend?: UiBackend) => Promise<void>;
  reinstallLlamaCpp: () => Promise<void>;
  installLlamaCppByTag: () => Promise<void>;
  installLlamaCppFromArchive: () => Promise<void>;
  switchBackend: (backend: UiBackend) => Promise<void>;
};

export class SettingsViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "llamaAio.settingsView";

  private view?: vscode.WebviewView;
  private updateCheckInFlight = false;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly store: SettingsStore,
    private readonly processManager: ProcessManager,
    private readonly installer: LlamaInstaller,
    private readonly perf: PerfStats,
    private readonly onReload: () => Promise<void>,
    private readonly modelActions: ModelActions,
    private readonly notifyChatModels: () => void = () => undefined
  ) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this.view = webviewView;
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this.extensionUri],
    };
    webviewView.webview.html = this.getHtml(webviewView.webview);
    webviewView.webview.onDidReceiveMessage(async (msg) => {
      try {
        switch (msg.type) {
          case "ready":
            await this.pushState();
            break;
          case "saveLoad":
            await this.store.updateLoadSettings(msg.payload as Partial<LlamaLoadSettings>);
            if (msg.silent) {
              const status = this.processManager.getStatus();
              const httpReady = await this.processManager.isHttpReady();
              this.view?.webview.postMessage({
                type: "statusPatch",
                payload: {
                  configDirty: !!status.configDirty,
                  running: !!(status.running || httpReady),
                },
              });
            } else {
              await this.pushState();
            }
            break;
          case "saveRequest":
            await this.store.updateRequestSettings(msg.payload as Partial<RequestSettings>);
            await this.pushState();
            break;
          case "reload":
            await this.store.updateLoadSettings(msg.payload as Partial<LlamaLoadSettings>);
            if (!(await this.confirmIfMemorySpill())) {
              await this.pushState();
              break;
            }
            await this.onReload();
            await this.pushState();
            break;
          case "start":
            if (!(await this.confirmIfMemorySpill())) {
              await this.pushState();
              break;
            }
            {
              const status = await vscode.window.withProgress(
                {
                  location: vscode.ProgressLocation.Notification,
                  title: "Llama AIO: Starting llama-server…",
                  cancellable: false,
                },
                async (progress) =>
                  this.processManager.start(undefined, (m) => progress.report({ message: m }))
              );
              this.notifyChatModels();
              await this.pushState();
              await promptUseInCopilotChat(this.store, status.message);
            }
            break;
          case "stop":
            await this.processManager.stop(true);
            await this.pushState();
            break;
          case "refresh":
            await this.pushState();
            break;
          case "setLaunchMode": {
            const mode = msg.payload === "background" ? "background" : "externalTerminal";
            await this.store
              .getConfig()
              .update("launchMode", mode, vscode.ConfigurationTarget.Global);
            await this.pushState();
            break;
          }
          case "downloadModel":
            await this.modelActions.downloadFromHuggingFace();
            await this.pushState();
            break;
          case "downloadStarter":
            await this.modelActions.downloadStarter();
            await this.pushState();
            break;
          case "openModelFile":
            await this.modelActions.openGgufFile();
            await this.pushState();
            break;
          case "pickDownloadedModel":
            await this.modelActions.pickDownloaded();
            await this.pushState();
            break;
          case "installLlamaCpp":
            await this.modelActions.installLlamaCpp(msg.payload as UiBackend | undefined);
            await this.pushState();
            break;
          case "reinstallLlamaCpp":
            await this.modelActions.reinstallLlamaCpp();
            await this.pushState();
            break;
          case "checkUpdates":
            await this.refreshUpdateCheck();
            break;
          case "installLlamaCppByTag":
            await this.modelActions.installLlamaCppByTag();
            await this.pushState();
            break;
          case "installLlamaCppFromArchive":
            await this.modelActions.installLlamaCppFromArchive();
            await this.pushState();
            break;
          case "switchBackend":
            await this.modelActions.switchBackend(msg.payload as UiBackend);
            await this.pushState();
            break;
          case "openExternal": {
            const url = typeof msg.url === "string" ? msg.url : "";
            if (/^https?:\/\//i.test(url)) {
              await vscode.env.openExternal(vscode.Uri.parse(url));
            }
            break;
          }
          case "revealInOs": {
            const targetPath = typeof msg.path === "string" ? msg.path : "";
            if (!targetPath) {
              break;
            }
            if (!fs.existsSync(targetPath)) {
              vscode.window.showWarningMessage(`Llama AIO: Path not found:\n${targetPath}`);
              break;
            }
            const uri = vscode.Uri.file(targetPath);
            try {
              const st = fs.statSync(targetPath);
              if (st.isDirectory()) {
                // Open the folder itself in the OS file manager.
                await vscode.env.openExternal(uri);
              } else {
                await vscode.commands.executeCommand("revealFileInOS", uri);
              }
            } catch (e) {
              vscode.window.showErrorMessage(
                `Llama AIO: Could not open path: ${e instanceof Error ? e.message : String(e)}`
              );
            }
            break;
          }
        }
      } catch (e) {
        vscode.window.showErrorMessage(
          `Llama AIO: ${e instanceof Error ? e.message : String(e)}`
        );
        await this.pushState();
      }
    });
  }

  private currentMemoryEstimate() {
    const state = this.store.getState();
    const cpuOnly =
      this.installer.resolveActiveUiBackend() === "cpu" || this.processManager.isCpuBackend();
    return estimateMemory(
      state.modelCapabilities,
      state.loadSettings,
      cpuOnly ? undefined : detectGpuMemory(),
      { cpuOnly }
    );
  }

  /** Ask before start/reload when estimated memory is likely to spill (VRAM or RAM). */
  async confirmIfMemorySpill(): Promise<boolean> {
    const est = this.currentMemoryEstimate();
    if (!est?.willSpill) {
      return true;
    }
    const choice = await vscode.window.showWarningMessage(
      est.warnings[0] ||
        "These settings leave too little memory headroom and may spill or thrash (much slower).",
      { modal: true },
      "Continue anyway",
      "Cancel"
    );
    return choice === "Continue anyway";
  }

  async pushState(): Promise<void> {
    if (!this.view) {
      return;
    }
    let state = this.store.getState();
    // Refresh GGUF caps when older state lacks size / arch dims needed for estimates.
    if (
      state.selectedModelPath &&
      (!state.modelCapabilities?.fileSizeBytes ||
        state.modelCapabilities.path !== state.selectedModelPath ||
        // Stale caps from before SWA / per-layer KV support (e.g. Gemma 4).
        (state.modelCapabilities.architecture === "gemma4" &&
          !state.modelCapabilities.slidingWindowPattern) ||
        // Stale caps from before hybrid full-attention interval (e.g. Qwen3.5).
        ((state.modelCapabilities.architecture === "qwen35" ||
          state.modelCapabilities.architecture === "qwen35moe") &&
          !state.modelCapabilities.fullAttentionInterval))
    ) {
      try {
        state = await this.store.applySelectedModel(state.selectedModelPath);
      } catch {
        // keep previous
      }
    }

    const status = this.processManager.getStatus();
    const httpReady = await this.processManager.isHttpReady();
    const binary = this.processManager.resolveBinary();
    const modelsDir = getModelsDir(this.store.getConfig());
    const localEntries = listLocalModelEntries(this.store.getConfig());
    const localModelCount = localEntries.length;
    const localSourceDirs = listActiveModelSourceDirs(this.store.getConfig(), localEntries);
    const localSources = localSourceDirs.map((s) => s.source);

    const caps = state.modelCapabilities;
    const build = this.installer.getInstalledInfo();
    const backendOptions = this.installer.getUiBackendOptions();
    // Active UI backend only — do not OR stale resolved/configured flags
    // (those stuck Load settings in CPU mode after switching back to Vulkan).
    const selectedUiBackend = (build.activeBackend ||
      build.resolvedBackend ||
      (build.configuredBackend === "auto" ? "vulkan" : build.configuredBackend)) as string;
    const cpuOnly = selectedUiBackend === "cpu";
    const gpu = cpuOnly ? undefined : detectGpuMemory();
    const memory = estimateMemory(caps, state.loadSettings, gpu, { cpuOnly });
    const updateCheck = this.installer.peekUpdateCheck();

    this.view.webview.postMessage({
      type: "state",
      payload: {
        state,
        status: { ...status, httpReady },
        perf: this.perf.get(),
        perfLines: this.perf.detailLines(),
        endpoint: this.store.getEndpoint(),
        binary,
        binaryExists: fs.existsSync(binary),
        modelsDir,
        localModelCount,
        localSources,
        localSourceDirs,
        modelName: caps?.name || (state.selectedModelPath ? path.basename(state.selectedModelPath) : ""),
        build,
        backendOptions,
        selectedUiBackend,
        cpuOnly,
        launchMode: resolveLaunchMode(this.store.getConfig().get<string>("launchMode")),
        updateCheck,
        memory,
        memInputs: memoryEstimateInputs(caps),
        systemRamTotalBytes: os.totalmem(),
        isWindows: process.platform === "win32",
        gpu: gpu
          ? { totalBytes: gpu.totalBytes, usedBytes: gpu.usedBytes, name: gpu.name }
          : null,
        capabilities: caps
          ? {
              maxContextLength: caps.maxContextLength,
              blockCount: caps.blockCount,
              isMoe: caps.isMoe,
              expertCount: caps.expertCount,
              expertUsedCount: caps.expertUsedCount,
              architecture: caps.architecture,
              ropeFreqBase: caps.ropeFreqBase,
              nextnPredictLayers: caps.nextnPredictLayers,
              fileSizeBytes: caps.fileSizeBytes,
              embeddingLength: caps.embeddingLength,
              attentionHeadCount: caps.attentionHeadCount,
              attentionHeadCountKv: caps.attentionHeadCountKv,
              attentionHeadCountKvPerLayer: caps.attentionHeadCountKvPerLayer,
              keyLength: caps.keyLength,
              valueLength: caps.valueLength,
              keyLengthSwa: caps.keyLengthSwa,
              valueLengthSwa: caps.valueLengthSwa,
              slidingWindow: caps.slidingWindow,
              slidingWindowPattern: caps.slidingWindowPattern,
            }
          : null,
      },
    });

    // Resolve latest tag in the background when cache is cold (no GitHub API).
    if (updateCheck.pending && !this.updateCheckInFlight) {
      this.updateCheckInFlight = true;
      void this.installer
        .getUpdateCheck(false)
        .then(() => this.pushState())
        .catch(() => undefined)
        .finally(() => {
          this.updateCheckInFlight = false;
        });
    }
  }

  /** Force-refresh latest release tag and refresh the sidebar. */
  async refreshUpdateCheck(): Promise<void> {
    await this.installer.getUpdateCheck(true);
    await this.pushState();
  }

  private getHtml(webview: vscode.Webview): string {
    const nonce = getNonce();
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Llama AIO</title>
  <style>
    :root {
      --bg: var(--vscode-sideBar-background);
      --fg: var(--vscode-foreground);
      --muted: var(--vscode-descriptionForeground);
      --input-bg: var(--vscode-input-background);
      --input-fg: var(--vscode-input-foreground);
      --border: var(--vscode-input-border, rgba(128,128,128,0.35));
      --accent: var(--vscode-button-background);
      --accent-fg: var(--vscode-button-foreground);
      --secondary: var(--vscode-button-secondaryBackground);
      --secondary-fg: var(--vscode-button-secondaryForeground);
      --warn-bg: color-mix(in srgb, #d29922 18%, transparent);
    }
    body {
      font-family: var(--vscode-font-family);
      font-size: 12px;
      color: var(--fg);
      background: var(--bg);
      padding: 10px 12px 24px;
      margin: 0;
    }
    h2 { font-size: 13px; margin: 18px 0 8px; font-weight: 600; }
    h2:first-child { margin-top: 4px; }
    .status, .card {
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--input-bg);
      margin-bottom: 10px;
      line-height: 1.45;
    }
    .status .ok { color: #3fb950; }
    .status .bad { color: #f85149; }
    .status a.endpoint {
      color: var(--vscode-textLink-foreground);
      text-decoration: underline;
      cursor: pointer;
    }
    .status a.endpoint:hover {
      color: var(--vscode-textLink-activeForeground);
    }
    .status .endpoint-hint {
      color: var(--muted);
      font-size: 11px;
      margin-top: 2px;
    }
    .mem-warn {
      margin-top: 8px;
      padding: 8px;
      border-radius: 6px;
      border: 1px solid color-mix(in srgb, #f85149 55%, var(--border));
      background: color-mix(in srgb, #f85149 16%, transparent);
      color: var(--fg);
      font-size: 11px;
      line-height: 1.45;
    }
    .mem-note {
      margin-top: 6px;
      padding: 6px 8px;
      border-radius: 6px;
      border: 1px solid color-mix(in srgb, #d29922 55%, var(--border));
      background: color-mix(in srgb, #d29922 14%, transparent);
      font-size: 11px;
      line-height: 1.45;
    }
    .ctx-meter {
      margin: 8px 0 6px;
      height: 8px;
      border-radius: 4px;
      background: color-mix(in srgb, var(--fg) 12%, transparent);
      overflow: hidden;
    }
    .ctx-meter > span {
      display: block;
      height: 100%;
      width: 0%;
      border-radius: 4px;
      background: #3fb950;
      transition: width 0.2s ease;
    }
    .ctx-meter.warn > span { background: #d29922; }
    .ctx-meter.critical > span { background: #f85149; }
    .ctx-label { font-size: 11px; margin-bottom: 4px; }
    .ctx-label.warn { color: #d29922; }
    .ctx-label.critical { color: #f85149; }
    .mem-charts { margin-top: 10px; display: flex; flex-direction: column; gap: 10px; }
    .mem-chart-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-weight: 600;
      font-size: 11px;
      margin-bottom: 4px;
    }
    .mem-chart-title .sub { font-weight: 400; color: var(--muted); }
    .mem-stack {
      display: flex;
      height: 14px;
      border-radius: 4px;
      overflow: hidden;
      background: color-mix(in srgb, var(--fg) 10%, transparent);
      border: 1px solid var(--border);
    }
    .mem-stack > span { display: block; height: 100%; min-width: 0; }
    .mem-stack .seg-weights { background: #3b82f6; }
    .mem-stack .seg-kv { background: #a855f7; }
    .mem-stack .seg-overhead { background: #64748b; }
    .mem-stack.over { outline: 1px solid #f85149; }
    .mem-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      margin-top: 6px;
      font-size: 10px;
      color: var(--muted);
    }
    .mem-legend i {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 2px;
      margin-right: 4px;
      vertical-align: middle;
    }
    .mem-legend .seg-weights { background: #3b82f6; }
    .mem-legend .seg-kv { background: #a855f7; }
    .mem-legend .seg-overhead { background: #64748b; }
    .setup {
      background: var(--warn-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      margin-bottom: 10px;
    }
    .setup ol { margin: 6px 0 0 18px; padding: 0; }
    .setup li { margin: 4px 0; }
    .row { margin: 10px 0 14px; }
    .label {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 4px;
    }
    .label span.name { font-weight: 500; }
    .label .badge {
      font-size: 10px;
      opacity: 0.75;
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 0 4px;
    }
    .hint { color: var(--muted); font-size: 11px; margin-top: 2px; }
    input[type="number"], input[type="text"],     select {
      width: 88px;
      background: var(--input-bg);
      color: var(--input-fg);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 2px 6px;
    }
    select.wide { width: 100%; box-sizing: border-box; padding: 6px 8px; }
    input[type="range"] { width: 100%; }
    .toggle {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin: 10px 0;
    }
    details.advanced {
      margin: 14px 0 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0 10px 4px;
      background: color-mix(in srgb, var(--input-bg) 55%, transparent);
    }
    details.advanced > summary {
      cursor: pointer;
      list-style: none;
      font-weight: 600;
      padding: 10px 0;
      color: var(--fg);
      user-select: none;
    }
    details.advanced > summary::-webkit-details-marker { display: none; }
    details.advanced > summary::before {
      content: '▸';
      display: inline-block;
      width: 1em;
      margin-right: 4px;
      color: var(--muted);
    }
    details.advanced[open] > summary::before { content: '▾'; }
    details.advanced > summary .sub {
      font-weight: 400;
      font-size: 11px;
      color: var(--muted);
      margin-left: 6px;
    }
    .btn-col {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 8px;
    }
    .actions {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 16px;
      position: sticky;
      bottom: 0;
      padding-top: 8px;
      background: linear-gradient(transparent, var(--bg) 30%);
    }
    button {
      border: none;
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
      font-weight: 600;
      text-align: left;
    }
    button.primary { background: var(--accent); color: var(--accent-fg); }
    button.secondary { background: var(--secondary); color: var(--secondary-fg); }
    button:disabled {
      opacity: 0.55;
      cursor: default;
    }
    .actions .row { margin: 0 0 4px; }
    .actions select.wide { width: 100%; }
    .model-title { font-weight: 600; margin-bottom: 4px; }
    .model-path, .meta {
      word-break: break-all;
      color: var(--muted);
      font-size: 11px;
    }
    a.model-path-link {
      color: var(--vscode-textLink-foreground);
      text-decoration: underline;
      cursor: pointer;
      word-break: break-all;
      font-size: 11px;
    }
    a.model-path-link:hover {
      color: var(--vscode-textLink-activeForeground);
    }
    .meta a.folder-link {
      color: var(--vscode-textLink-foreground);
      text-decoration: underline;
      cursor: pointer;
    }
    .meta a.folder-link:hover {
      color: var(--vscode-textLink-activeForeground);
    }
    .hidden { display: none !important; }
    .caps {
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 11px;
      line-height: 1.5;
    }
  </style>
</head>
<body>
  <div class="status" id="status">Loading…</div>
  <div class="card" id="perfCard" style="margin-top:8px">
    <div class="model-title">Performance</div>
    <div class="ctx-label" id="ctxLabel">Context: — (send a chat to measure)</div>
    <div class="ctx-meter" id="ctxMeter"><span id="ctxMeterFill"></span></div>
    <div class="meta" id="perfLines">No generation yet</div>
  </div>

  <div class="setup hidden" id="setupBox">
    <strong>Get a model first</strong>
    <p class="hint" style="margin:8px 0">One-click starter: Unsloth ${STARTER_MODEL.label} (${STARTER_MODEL.approxSizeLabel}, ${STARTER_MODEL.detail}).</p>
    <div class="btn-col" style="margin-top:4px">
      <button class="primary" id="setupStarterBtn">⬇ Download starter (${STARTER_MODEL.label})</button>
    </div>
    <ol>
      <li>Install llama.cpp (once)</li>
      <li>Download the starter <em>or</em> pick/open any GGUF</li>
      <li>Start the server</li>
    </ol>
  </div>

  <h2>llama.cpp</h2>
  <div class="card">
    <div class="row" style="margin-top:0;margin-bottom:6px">
      <div class="label"><span class="name">Backend</span></div>
      <select id="backendSelect" class="wide"></select>
      <div class="hint" id="backendHint"></div>
    </div>
    <div class="btn-col">
      <button class="secondary hidden" id="installLlamaBtn">Upgrade to latest</button>
    </div>
    <details class="advanced" style="margin-top:10px">
      <summary>More install options<span class="sub">pin a tag, local archive, releases</span></summary>
      <div class="meta" id="llamaBinaryDetail" style="margin:6px 0 4px"></div>
      <div class="meta" id="llamaAssetDetail" style="margin:0 0 8px"></div>
      <div class="btn-col">
        <button class="secondary" id="checkUpdatesBtn">Check for updates</button>
        <button class="secondary" id="reinstallLlamaBtn">Reinstall current release</button>
        <button class="secondary" id="installByTagBtn">Install release tag…</button>
        <button class="secondary" id="installArchiveBtn">Install from archive…</button>
      </div>
      <div class="hint" style="margin-top:8px">
        Tag / archive installs skip the GitHub API (useful on shared IPs).
        <a href="https://github.com/ggml-org/llama.cpp/releases" id="releasesLink">Browse releases</a>
      </div>
    </details>
  </div>

  <h2>Model</h2>
  <div class="card">
    <div class="model-title" id="modelTitle">No model selected</div>
    <div class="model-path" id="modelPath"></div>
    <div class="caps hidden" id="modelCaps"></div>
    <div class="meta" id="modelsDirMeta"></div>
    <div class="btn-col">
      <button class="primary hidden" id="starterModelBtn">⬇ Download starter (${STARTER_MODEL.label})</button>
      <div class="hint hidden" id="starterModelHint" style="margin-top:0">${STARTER_MODEL.approxSizeLabel} · ${STARTER_MODEL.detail}</div>
      <button class="primary" id="downloadModelBtn">⬇ Download from Hugging Face…</button>
      <button class="secondary" id="openFileBtn">📂 Open GGUF file…</button>
      <button class="secondary" id="pickDownloadedBtn">📚 Choose from downloaded…</button>
    </div>
  </div>

  <h2>Load settings</h2>

  <div class="card" id="memCard">
    <div class="model-title">Memory estimate</div>
    <div class="hint" style="margin-top:0;margin-bottom:8px">Bars = estimate at <strong>full context</strong>. “Live GPU free” is current occupancy, not the bar.</div>
    <div class="mem-charts" id="memCharts">
      <div>
        <div class="mem-chart-title"><span id="vramChartTitle">VRAM · est. at full context</span><span class="sub" id="vramChartSub">—</span></div>
        <div class="mem-stack" id="vramStack"></div>
      </div>
      <div>
        <div class="mem-chart-title"><span id="ramChartTitle">System RAM · est. at full context</span><span class="sub" id="ramChartSub">—</span></div>
        <div class="mem-stack" id="ramStack"></div>
      </div>
      <div class="mem-legend">
        <span><i class="seg-weights"></i>Weights</span>
        <span><i class="seg-kv"></i>KV cache</span>
        <span><i class="seg-overhead"></i>Overhead</span>
      </div>
    </div>
    <div class="meta" id="memLines" style="margin-top:8px">Select a model to estimate VRAM / RAM use.</div>
    <div class="mem-note hidden" id="memNotes"></div>
    <div class="mem-warn hidden" id="memWarn"></div>
  </div>

  <div class="row">
    <div class="label"><span class="name">Context Length</span><input type="number" id="contextLength" min="512" step="512" /></div>
    <input type="range" id="contextLengthRange" min="512" max="131072" step="512" />
    <div class="hint" id="ctxHint">Tokens for prompt + generation</div>
  </div>
  <div class="row" id="gpuOffloadRow">
    <div class="label"><span class="name">GPU Offload</span><input type="number" id="gpuOffload" min="0" max="999" /></div>
    <input type="range" id="gpuOffloadRange" min="0" max="999" step="1" />
    <div class="hint" id="gpuOffloadHint">Layers on GPU (-ngl). Use 99/-1 style high values for “all”.</div>
  </div>
  <div class="row" id="moeRow">
    <div class="label"><span class="name">CPU MoE layers</span><span class="badge">MoE only</span><input type="number" id="nCpuMoe" min="0" max="256" /></div>
    <input type="range" id="nCpuMoeRange" min="0" max="128" step="1" />
    <div class="hint" id="moeHint">Number of layers to force experts onto CPU (--n-cpu-moe). Only applies to MoE models.</div>
  </div>

  <details class="advanced">
    <summary>Advanced Settings<span class="sub">threads, batch, KV, RoPE, speculative…</span></summary>

  <div class="row">
    <div class="label"><span class="name">CPU Thread Pool Size</span><input type="number" id="cpuThreads" min="1" max="256" /></div>
    <input type="range" id="cpuThreadsRange" min="1" max="64" step="1" />
  </div>
  <div class="row">
    <div class="label"><span class="name">Evaluation Batch Size</span><input type="number" id="evalBatchSize" min="32" step="32" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name">Physical Batch Size</span><input type="number" id="physicalBatchSize" min="32" step="32" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name">Max Concurrent Predictions</span><span class="badge">Splits context</span><input type="number" id="maxConcurrentPredictions" min="1" max="64" /></div>
    <div class="hint">Use <strong>1</strong> for Copilot Chat. Values &gt; 1 split Context Length across slots (e.g. 8192/4 = 2048 per request).</div>
  </div>

  <div class="toggle"><span>Offload KV Cache to GPU Memory</span><input type="checkbox" id="offloadKvCacheToGpu" /></div>
  <div class="toggle"><span id="keepModelLabel">Keep Model in Memory (--mlock)</span><input type="checkbox" id="keepModelInMemory" /></div>
  <div class="hint" id="keepModelHint" style="display:none">On Windows this uses mmap (--load-mode mmap); mlock is not reliable.</div>
  <div class="toggle"><span>Try mmap()</span><input type="checkbox" id="tryMmap" /></div>
  <div class="toggle"><span>Unified KV Cache</span><input type="checkbox" id="unifiedKvCache" /></div>

  <div class="row">
    <div class="label"><span class="name">Context Checkpoints</span><input type="number" id="contextCheckpoints" min="0" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name">RoPE Frequency Base</span>
      <label><input type="checkbox" id="ropeBaseAuto" /> Auto</label>
    </div>
    <input type="number" id="ropeFreqBase" step="1" />
  </div>
  <div class="row">
    <div class="label"><span class="name">RoPE Frequency Scale</span>
      <label><input type="checkbox" id="ropeScaleAuto" /> Auto</label>
    </div>
    <input type="number" id="ropeFreqScale" step="0.01" />
  </div>
  <div class="row">
    <div class="label"><span class="name">Seed</span>
      <label><input type="checkbox" id="seedRandom" /> Random</label>
    </div>
    <input type="number" id="seed" step="1" />
  </div>

  <h2>Speculative decoding</h2>
  <div class="row">
    <div class="label"><span class="name">Mode</span>
      <select id="speculativeMode">
        <option value="off">Off</option>
        <option value="mtp">MTP (draft-mtp)</option>
      </select>
    </div>
    <div class="hint" id="specHint">MTP needs a model with next-n / MTP layers (e.g. Ornith MTP). Uses --spec-type draft-mtp.</div>
  </div>
  <div class="row">
    <div class="label"><span class="name">Max draft tokens</span><input type="number" id="maxDraftTokens" min="0" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name">Min draft tokens</span><input type="number" id="minDraftTokens" min="0" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name">Draft probability</span><input type="number" id="draftProbability" min="0" max="1" step="0.01" /></div>
  </div>

  </details>

  <details class="advanced">
    <summary>Request defaults<span class="sub">temperature, top-p/k, max tokens</span></summary>
  <div class="row">
    <div class="label"><span class="name">Temperature</span><input type="number" id="temperature" min="0" max="2" step="0.05" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name">Top P</span><input type="number" id="topP" min="0" max="1" step="0.01" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name">Top K</span><input type="number" id="topK" min="0" step="1" /></div>
  </div>
  <div class="row">
    <div class="label"><span class="name">Max tokens</span><input type="number" id="maxTokens" min="16" step="16" /></div>
  </div>
  </details>

  <div class="actions">
    <div class="row">
      <div class="label"><span class="name">Launch</span></div>
      <select id="launchMode" class="wide" title="How llama-server is started">
        <option value="externalTerminal">External terminal (logs visible)</option>
        <option value="background">Background (hidden process)</option>
      </select>
    </div>
    <button class="primary" id="primaryBtn" data-action="start">Start server</button>
    <button class="secondary hidden" id="stopBtn">Stop server</button>
  </div>

  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const $ = (id) => document.getElementById(id);

    let memInputs = null;
    let gpuInfo = null;
    let systemRamTotalBytes = 0;
    let backendOptionsCache = [];
    let modelIsMoe = false;
    let moeHintDefault = 'Number of layers to force experts onto CPU (--n-cpu-moe). Only applies to MoE models.';
    let suppressBackendChange = false;
    let activeBackendId = '';
    let serverRunning = false;
    let configDirty = false;
    let saveLoadTimer = null;
    let updateCheck = { latestTag: undefined, installedTag: undefined, updateAvailable: false, checkFailed: false, pending: true };

    function updatePrimaryAction() {
      const primary = $('primaryBtn');
      const stop = $('stopBtn');
      if (!primary || !stop) return;
      stop.classList.toggle('hidden', !serverRunning);
      if (!serverRunning) {
        primary.disabled = false;
        primary.textContent = 'Start server';
        primary.dataset.action = 'start';
      } else if (configDirty) {
        primary.disabled = false;
        primary.textContent = '↻ Reload to apply';
        primary.dataset.action = 'reload';
      } else {
        primary.disabled = true;
        primary.textContent = 'Running';
        primary.dataset.action = '';
      }
    }

    function scheduleSaveLoad() {
      if (saveLoadTimer) clearTimeout(saveLoadTimer);
      // Optimistic dirty UI while running — confirmed via silent save + statusPatch.
      if (serverRunning) {
        configDirty = true;
        updatePrimaryAction();
      }
      saveLoadTimer = setTimeout(() => {
        vscode.postMessage({ type: 'saveLoad', payload: readLoad(), silent: true });
      }, 280);
    }

    function updateBackendUi() {
      const sel = $('backendSelect');
      const installBtn = $('installLlamaBtn');
      const hint = $('backendHint');
      const reinstallBtn = $('reinstallLlamaBtn');
      if (!sel || !installBtn || !hint) return;
      const selectedOpt = backendOptionsCache.find((o) => o.id === sel.value);
      const anyInstalled = backendOptionsCache.some((o) => o.installed);
      const installedTag = (selectedOpt && selectedOpt.installedTag) || updateCheck.installedTag;
      const latestTag = updateCheck.latestTag;
      const selectedIsActive = !!(selectedOpt && selectedOpt.active);

      if (reinstallBtn) {
        reinstallBtn.classList.toggle('hidden', !(selectedOpt && selectedOpt.installed && selectedIsActive));
        reinstallBtn.textContent = installedTag
          ? ('Reinstall ' + installedTag)
          : 'Reinstall current release';
      }

      if (selectedOpt && selectedOpt.reason && !selectedOpt.available) {
        hint.textContent = selectedOpt.reason;
        installBtn.classList.add('hidden');
        installBtn.disabled = true;
        return;
      }

      if (selectedOpt && !selectedOpt.installed) {
        hint.textContent = 'Not installed — selecting this backend will download it.';
        installBtn.textContent = 'Install ' + (selectedOpt.label || selectedOpt.id);
        installBtn.disabled = false;
        installBtn.classList.remove('hidden');
        installBtn.dataset.action = 'install';
        return;
      }

      if (!selectedOpt) {
        hint.textContent = anyInstalled ? '' : 'No llama.cpp backend installed yet.';
        installBtn.textContent = 'Install llama.cpp';
        installBtn.disabled = false;
        installBtn.classList.toggle('hidden', !!anyInstalled);
        installBtn.dataset.action = 'install';
        return;
      }

      // Installed backend selected
      if (!selectedIsActive) {
        hint.textContent = 'Switching applies immediately. Restart the server if it is running.';
        installBtn.classList.add('hidden');
        return;
      }

      if (updateCheck.pending) {
        hint.textContent = (installedTag ? (installedTag + ' · ') : '') + 'Checking for updates…';
        installBtn.classList.add('hidden');
        return;
      }

      if (updateCheck.checkFailed) {
        hint.textContent = (installedTag ? (installedTag + ' · ') : '') + 'Could not check for updates.';
        installBtn.textContent = 'Check for updates';
        installBtn.disabled = false;
        installBtn.classList.remove('hidden');
        installBtn.dataset.action = 'check';
        return;
      }

      if (updateCheck.updateAvailable && latestTag) {
        hint.textContent = (installedTag ? ('Installed ' + installedTag + ' · ') : '') + 'Update available.';
        installBtn.textContent = 'Upgrade to ' + latestTag;
        installBtn.disabled = false;
        installBtn.classList.remove('hidden');
        installBtn.dataset.action = 'upgrade';
        return;
      }

      hint.textContent = (installedTag || 'Installed') + ' · up to date' + (latestTag ? (' (latest ' + latestTag + ')') : '');
      installBtn.classList.add('hidden');
      installBtn.dataset.action = '';
    }

    function fmtBytes(bytes) {
      const GiB = 1024 ** 3, MiB = 1024 ** 2;
      if (!bytes || bytes <= 0) return '0 B';
      if (bytes >= GiB) return (bytes / GiB).toFixed(bytes >= 10 * GiB ? 1 : 2) + ' GiB';
      if (bytes >= MiB) return (bytes / MiB).toFixed(0) + ' MiB';
      return Math.round(bytes / 1024) + ' KiB';
    }

    function buildCharts(gpuWeights, cpuWeights, kvBytes, kvOnGpu, gpuOverhead, cpuOverhead, totalGpu, totalCpu) {
      return {
        vram: {
          title: 'VRAM · est. at full context',
          segments: [
            { key: 'weights', label: 'Weights', bytes: gpuWeights },
            { key: 'kv', label: 'KV cache (full ctx)', bytes: kvOnGpu ? kvBytes : 0 },
            { key: 'overhead', label: 'Overhead', bytes: gpuOverhead },
          ],
          totalBytes: totalGpu,
          capacityBytes: gpuInfo && gpuInfo.totalBytes ? gpuInfo.totalBytes : undefined,
        },
        ram: {
          title: 'System RAM · est. at full context',
          segments: [
            { key: 'weights', label: 'Weights', bytes: cpuWeights },
            { key: 'kv', label: 'KV cache (full ctx)', bytes: kvOnGpu ? 0 : kvBytes },
            { key: 'overhead', label: 'Overhead', bytes: cpuOverhead },
          ],
          totalBytes: totalCpu,
          capacityBytes: systemRamTotalBytes || undefined,
        },
      };
    }

    function moeExpertShareOf(inputs) {
      if (!inputs || !inputs.isMoe) return 0;
      if (inputs.moeExpertShare != null && isFinite(inputs.moeExpertShare)) {
        return Math.min(0.98, Math.max(0.05, Number(inputs.moeExpertShare)));
      }
      const n = Number(inputs.expertCount) || 0;
      if (n >= 128) return 0.9;
      if (n >= 64) return 0.85;
      if (n >= 16) return 0.8;
      return 0.75;
    }

    function renderStackedBar(stackId, subId, chart) {
      const stack = $(stackId);
      const sub = $(subId);
      if (!chart) {
        stack.innerHTML = '';
        stack.classList.remove('over');
        sub.textContent = '—';
        return;
      }
      const capacity = chart.capacityBytes || chart.totalBytes || 1;
      const scale = Math.max(capacity, chart.totalBytes || 0) || 1;
      stack.innerHTML = '';
      for (const seg of chart.segments || []) {
        if (!seg.bytes || seg.bytes <= 0) continue;
        const el = document.createElement('span');
        el.className = 'seg-' + seg.key;
        el.style.width = Math.max(0.5, (seg.bytes / scale) * 100) + '%';
        el.title = seg.label + ': ~' + fmtBytes(seg.bytes);
        stack.appendChild(el);
      }
      const over = chart.capacityBytes && chart.totalBytes > chart.capacityBytes;
      stack.classList.toggle('over', !!over);
      const pct = chart.capacityBytes
        ? Math.round((chart.totalBytes / chart.capacityBytes) * 100)
        : undefined;
      sub.textContent =
        '~' + fmtBytes(chart.totalBytes) +
        (chart.capacityBytes
          ? ' / ' + fmtBytes(chart.capacityBytes) + (pct !== undefined ? ' (' + pct + '%)' : '')
          : '');
    }

    function liveMemoryEstimate() {
      if (!memInputs || !memInputs.fileSizeBytes) return null;
      const L = readLoad();
      const nLayers = Math.max(1, memInputs.blockCount || 1);
      // Follow the dropdown (pending switch), not a sticky flag.
      const cpuOnly = $('backendSelect').value === 'cpu';
      let onGpu = cpuOnly ? 0 : (L.gpuOffload <= 0 ? 0 : (L.gpuOffload >= 99 ? nLayers : Math.min(L.gpuOffload, nLayers)));
      const expertShare = moeExpertShareOf(memInputs);
      let gpuWeights = memInputs.fileSizeBytes * (onGpu / nLayers);
      if (!cpuOnly && memInputs.isMoe && L.nCpuMoe > 0 && onGpu > 0 && expertShare > 0) {
        const moeCpu = Math.min(L.nCpuMoe, onGpu);
        gpuWeights = Math.max(0, gpuWeights - memInputs.fileSizeBytes * (moeCpu / nLayers) * expertShare);
      }
      const cpuWeights = Math.max(0, memInputs.fileSizeBytes - gpuWeights);
      const heads = Math.max(1, memInputs.attentionHeadCount || 8);
      const defaultKvHeads = Math.max(1, memInputs.attentionHeadCountKv || heads);
      const defaultKeyDim = Math.max(1, memInputs.keyLength || Math.floor((memInputs.embeddingLength || heads * 128) / heads));
      const defaultValDim = Math.max(1, memInputs.valueLength || defaultKeyDim);
      const swa = memInputs.slidingWindow > 0 ? memInputs.slidingWindow : 0;
      const pattern = memInputs.slidingWindowPattern;
      const perKv = memInputs.attentionHeadCountKvPerLayer;
      const recurrent = memInputs.recurrentLayers;
      const fullInterval = memInputs.fullAttentionInterval > 1 ? memInputs.fullAttentionInterval : 0;
      function kvAt(ctx) {
        let bytes = 0;
        let fullAttnLayers = 0;
        for (let i = 0; i < nLayers; i++) {
          const isRecurrent = (recurrent && recurrent.length === nLayers)
            ? !!recurrent[i]
            : !!(fullInterval && ((i + 1) % fullInterval !== 0));
          if (isRecurrent) continue;
          fullAttnLayers++;
          const isSwa = !!(swa && pattern && pattern[i]);
          const nKv = Math.max(1, (perKv && perKv[i]) || defaultKvHeads);
          const keyDim = isSwa ? Math.max(1, memInputs.keyLengthSwa || Math.floor(defaultKeyDim / 2) || defaultKeyDim) : defaultKeyDim;
          const valDim = isSwa ? Math.max(1, memInputs.valueLengthSwa || Math.floor(defaultValDim / 2) || defaultValDim) : defaultValDim;
          const tokens = isSwa ? Math.min(ctx, swa) : ctx;
          bytes += (nKv * keyDim + nKv * valDim) * tokens * 2;
        }
        return { bytes, fullAttnLayers };
      }
      const fullKv = kvAt(L.contextLength);
      const warmCtx = Math.min(2048, Math.max(512, L.contextLength));
      const warmKv = kvAt(warmCtx);
      const kvBytes = fullKv.bytes;
      const kvBytesWarm = warmKv.bytes;
      const fullAttnLayers = fullKv.fullAttnLayers;
      const kvOnGpu = !cpuOnly && !!L.offloadKvCacheToGpu && onGpu > 0;
      const overhead = Math.round(400 * 1024 * 1024 + Math.min(L.evalBatchSize || 512, 4096) * 64 * 1024);
      const gpuOverhead = onGpu > 0 ? overhead : 0;
      const cpuOverhead = onGpu > 0 ? Math.round(overhead * 0.15) : Math.round(overhead * 0.5);
      const totalGpu = gpuWeights + (kvOnGpu ? kvBytes : 0) + gpuOverhead;
      const totalCpu = cpuWeights + (kvOnGpu ? 0 : kvBytes) + cpuOverhead;
      const totalGpuWarm = gpuWeights + (kvOnGpu ? kvBytesWarm : 0) + gpuOverhead;
      const totalCpuWarm = cpuWeights + (kvOnGpu ? 0 : kvBytesWarm) + cpuOverhead;
      const warnings = [];
      let willSpill = false;
      if (cpuOnly) {
        warnings.push('CPU backend: no GPU acceleration — weights, KV cache, and compute use system RAM (GPU Offload is ignored).');
      }
      if (!cpuOnly && onGpu > 0 && onGpu < nLayers) {
        warnings.push('Partial GPU offload: ' + (nLayers - onGpu) + '/' + nLayers + ' layers (~' + fmtBytes(cpuWeights) + ') stay in system RAM.');
      }
      if (!cpuOnly && onGpu === 0) warnings.push('GPU offload is 0 — weights run from system RAM.');
      if (!cpuOnly && !L.offloadKvCacheToGpu) warnings.push('KV cache (~' + fmtBytes(kvBytes) + ' at full context) is in system RAM.');
      if (!cpuOnly && memInputs.isMoe && L.nCpuMoe > 0) {
        warnings.push('CPU MoE layers = ' + L.nCpuMoe + ': ~' + Math.round(expertShare * 100) + '% of weights are experts; those layers’ experts stay in system RAM.');
      }
      if (!cpuOnly && gpuInfo && gpuInfo.totalBytes) {
        const usable = gpuInfo.totalBytes * 0.92;
        const pct = Math.round((totalGpu / gpuInfo.totalBytes) * 100);
        if (totalGpu > gpuInfo.totalBytes) {
          willSpill = true;
          warnings.unshift('Estimated VRAM at full context ~' + fmtBytes(totalGpu) + ' is over the full ' + fmtBytes(gpuInfo.totalBytes) + ' GPU (' + pct + '%). Expect spill to system RAM. Lower Context or GPU Offload.');
        } else if (totalGpu > usable) {
          willSpill = true;
          warnings.unshift('Tight on VRAM at full context: ~' + fmtBytes(totalGpu) + ' of ' + fmtBytes(gpuInfo.totalBytes) + ' (' + pct + '%). Only ~' + fmtBytes(gpuInfo.totalBytes - usable) + ' safe headroom for the driver — often spills to system RAM. Lower Context or GPU Offload.');
        } else if (totalGpu > gpuInfo.totalBytes * 0.8) {
          warnings.push('Getting full at full context: ~' + fmtBytes(totalGpu) + ' of ' + fmtBytes(gpuInfo.totalBytes) + ' VRAM (' + pct + '%).');
        }
      }
      if (cpuOnly && systemRamTotalBytes && totalCpu > systemRamTotalBytes * 0.9) {
        willSpill = true;
        warnings.unshift('Estimated system RAM at full context ~' + fmtBytes(totalCpu) + ' is very high vs ' + fmtBytes(systemRamTotalBytes) + '. Lower Context Length or use a smaller model/quant.');
      }
      const lines = [];
      if (cpuOnly) {
        lines.push('Backend: CPU — GPU Offload / VRAM not used');
      } else if (gpuInfo && gpuInfo.totalBytes) {
        lines.push('GPU capacity: ' + fmtBytes(gpuInfo.totalBytes) + (gpuInfo.name ? ' (' + gpuInfo.name + ')' : ''));
        if (gpuInfo.usedBytes != null) {
          const free = Math.max(0, gpuInfo.totalBytes - gpuInfo.usedBytes);
          lines.push('Live GPU free now: ~' + fmtBytes(free) + ' (current occupancy — not part of the estimate bars)');
        }
      } else {
        lines.push('GPU VRAM: unknown');
      }
      if (systemRamTotalBytes) lines.push('System RAM capacity: ' + fmtBytes(systemRamTotalBytes));
      if (cpuOnly) {
        lines.push('Weights in RAM: ~' + fmtBytes(cpuWeights) + ' (' + nLayers + ' layers)');
        lines.push('KV @ full ' + Number(L.contextLength).toLocaleString() + ' ctx: ~' + fmtBytes(kvBytes) + ' (system RAM)' +
          (fullAttnLayers < nLayers ? (' · ' + fullAttnLayers + '/' + nLayers + ' full-attn layers') : ''));
        if (kvBytesWarm < kvBytes) {
          lines.push('KV @ ~' + Number(warmCtx).toLocaleString() + ' ctx (mid-chat): ~' + fmtBytes(kvBytesWarm) + ' → total ~' + fmtBytes(totalCpuWarm));
        }
        lines.push('Est. total system RAM at full context: ~' + fmtBytes(totalCpu));
      } else {
        lines.push('Weights on GPU: ~' + fmtBytes(gpuWeights) + ' (' + onGpu + '/' + nLayers + ' layers)' + (cpuWeights > 1024*1024 ? ' · RAM: ~' + fmtBytes(cpuWeights) : '') +
          (memInputs.isMoe && expertShare > 0 ? (' · MoE experts ~' + Math.round(expertShare * 100) + '% of file') : ''));
        lines.push('KV @ full ' + Number(L.contextLength).toLocaleString() + ' ctx: ~' + fmtBytes(kvBytes) + (kvOnGpu ? ' (GPU)' : ' (CPU RAM)') +
          (fullAttnLayers < nLayers ? (' · ' + fullAttnLayers + '/' + nLayers + ' full-attn layers') : ''));
        if (kvBytesWarm < kvBytes) {
          lines.push('KV @ ~' + Number(warmCtx).toLocaleString() + ' ctx (mid-chat): ~' + fmtBytes(kvBytesWarm) + (kvOnGpu ? ' (GPU)' : ' (CPU RAM)') + ' → VRAM ~' + fmtBytes(totalGpuWarm));
        }
        lines.push('Est. total at full context — VRAM: ~' + fmtBytes(totalGpu) + (totalCpu > 1024*1024 ? ' · system RAM: ~' + fmtBytes(totalCpu) : ''));
      }
      lines.push('Bars show estimate at full context. Actual use varies by quant, MoE, and backend.');
      const charts = buildCharts(gpuWeights, cpuWeights, kvBytes, kvOnGpu, gpuOverhead, cpuOverhead, totalGpu, totalCpu);
      if (cpuOnly) {
        charts.vram.capacityBytes = undefined;
      }
      return {
        lines,
        warnings,
        willSpill,
        charts,
      };
    }

    function applyCpuOnlyUi(cpuOnly) {
      $('gpuOffload').disabled = cpuOnly;
      $('gpuOffloadRange').disabled = cpuOnly;
      $('offloadKvCacheToGpu').disabled = cpuOnly;
      $('gpuOffloadHint').textContent = cpuOnly
        ? 'CPU backend installed — GPU Offload is ignored; everything runs in system RAM.'
        : 'Layers on GPU (-ngl). Use 99/-1 style high values for “all”.';
      $('gpuOffloadRow').style.opacity = cpuOnly ? '0.55' : '1';

      // --n-cpu-moe only splits experts GPU↔CPU; meaningless when everything is already on CPU.
      $('nCpuMoe').disabled = cpuOnly;
      $('nCpuMoeRange').disabled = cpuOnly;
      const showMoe = modelIsMoe && !cpuOnly;
      $('moeRow').classList.toggle('hidden', !showMoe);
      if (modelIsMoe) {
        $('moeHint').textContent = cpuOnly
          ? 'CPU backend — experts already run in system RAM; CPU MoE layers (--n-cpu-moe) does not apply.'
          : moeHintDefault;
      }
    }

    function renderMemory(est) {
      if (!est) {
        $('memLines').textContent = 'Select a model to estimate VRAM / RAM use.';
        $('memNotes').classList.add('hidden');
        $('memWarn').classList.add('hidden');
        renderStackedBar('vramStack', 'vramChartSub', null);
        renderStackedBar('ramStack', 'ramChartSub', null);
        return;
      }
      if (est.charts) {
        renderStackedBar('vramStack', 'vramChartSub', est.charts.vram);
        renderStackedBar('ramStack', 'ramChartSub', est.charts.ram);
      }
      $('memLines').innerHTML = (est.lines || []).map((l) => String(l)).join('<br/>');
      const soft = (est.warnings || []).filter((_, i) => !(est.willSpill && i === 0));
      if (soft.length) {
        $('memNotes').classList.remove('hidden');
        $('memNotes').innerHTML = soft.map((w) => String(w)).join('<br/>');
      } else {
        $('memNotes').classList.add('hidden');
      }
      if (est.willSpill && est.warnings && est.warnings.length) {
        $('memWarn').classList.remove('hidden');
        $('memWarn').textContent = est.warnings[0];
      } else {
        $('memWarn').classList.add('hidden');
      }
    }

    function refreshMemoryLive() {
      renderMemory(liveMemoryEstimate());
    }

    function bindRange(numId, rangeId) {
      const num = $(numId);
      const range = $(rangeId);
      if (!num || !range) return;
      num.addEventListener('input', () => { range.value = num.value; refreshMemoryLive(); });
      range.addEventListener('input', () => { num.value = range.value; refreshMemoryLive(); });
    }
    bindRange('contextLength', 'contextLengthRange');
    bindRange('gpuOffload', 'gpuOffloadRange');
    bindRange('cpuThreads', 'cpuThreadsRange');
    bindRange('nCpuMoe', 'nCpuMoeRange');
    $('offloadKvCacheToGpu').addEventListener('change', refreshMemoryLive);
    $('evalBatchSize').addEventListener('input', refreshMemoryLive);

    // Persist load edits so dirty tracking / reload uses the form values.
    const loadFieldIds = [
      'contextLength', 'contextLengthRange', 'gpuOffload', 'gpuOffloadRange',
      'cpuThreads', 'cpuThreadsRange', 'evalBatchSize', 'physicalBatchSize',
      'maxConcurrentPredictions', 'nCpuMoe', 'nCpuMoeRange', 'offloadKvCacheToGpu',
      'keepModelInMemory', 'tryMmap', 'unifiedKvCache', 'contextCheckpoints',
      'ropeBaseAuto', 'ropeFreqBase', 'ropeScaleAuto', 'ropeFreqScale',
      'seedRandom', 'seed', 'speculativeMode', 'maxDraftTokens', 'minDraftTokens',
      'draftProbability'
    ];
    for (const id of loadFieldIds) {
      const el = $(id);
      if (!el) continue;
      el.addEventListener('change', scheduleSaveLoad);
      el.addEventListener('input', scheduleSaveLoad);
    }

    function readLoad() {
      const ropeBaseAuto = $('ropeBaseAuto').checked;
      const ropeScaleAuto = $('ropeScaleAuto').checked;
      const seedRandom = $('seedRandom').checked;
      return {
        contextLength: Number($('contextLength').value),
        gpuOffload: Number($('gpuOffload').value),
        cpuThreads: Number($('cpuThreads').value),
        evalBatchSize: Number($('evalBatchSize').value),
        physicalBatchSize: Number($('physicalBatchSize').value),
        maxConcurrentPredictions: Number($('maxConcurrentPredictions').value),
        nCpuMoe: Number($('nCpuMoe').value),
        offloadKvCacheToGpu: $('offloadKvCacheToGpu').checked,
        keepModelInMemory: $('keepModelInMemory').checked,
        tryMmap: $('tryMmap').checked,
        unifiedKvCache: $('unifiedKvCache').checked,
        contextCheckpoints: Number($('contextCheckpoints').value),
        ropeFreqBase: ropeBaseAuto ? null : Number($('ropeFreqBase').value),
        ropeFreqScale: ropeScaleAuto ? null : Number($('ropeFreqScale').value),
        seed: seedRandom ? null : Number($('seed').value),
        speculativeMode: $('speculativeMode').value,
        maxDraftTokens: Number($('maxDraftTokens').value),
        minDraftTokens: Number($('minDraftTokens').value),
        draftProbability: Number($('draftProbability').value),
      };
    }

    function readRequest() {
      return {
        temperature: Number($('temperature').value),
        topP: Number($('topP').value),
        topK: Number($('topK').value),
        maxTokens: Number($('maxTokens').value),
      };
    }

    function applyCapabilities(caps) {
      const maxCtx = (caps && caps.maxContextLength) ? caps.maxContextLength : 131072;
      const blocks = (caps && caps.blockCount) ? caps.blockCount : 128;
      const isMoe = !!(caps && caps.isMoe);
      modelIsMoe = isMoe;

      $('contextLength').max = String(maxCtx);
      $('contextLengthRange').max = String(maxCtx);
      $('contextLengthRange').min = '512';
      // Allow 99 as "all layers" sentinel above block count.
      $('gpuOffload').max = String(Math.max(blocks, 99));
      $('gpuOffloadRange').max = String(Math.max(blocks, 99));
      $('nCpuMoe').max = String(blocks);
      $('nCpuMoeRange').max = String(blocks);

      // Visibility finalized in applyCpuOnlyUi (also hides on CPU backend).
      if (caps) {
        $('ctxHint').textContent = 'Model supports up to ' + maxCtx + ' tokens (from GGUF metadata)';
        $('moeHint').textContent = isMoe
          ? ('MoE model' + (caps.expertCount ? (' · ' + caps.expertCount + ' experts') : '') +
             (caps.expertUsedCount ? (' · ' + caps.expertUsedCount + ' used/token') : '') +
             '. Layers to force experts onto CPU (0–' + blocks + ').')
          : 'Not a MoE model — this setting is hidden.';
        if (isMoe) {
          moeHintDefault = $('moeHint').textContent;
        }
        $('modelCaps').classList.remove('hidden');
        $('modelCaps').innerHTML =
          'Architecture: <strong>' + (caps.architecture || '?') + '</strong><br/>' +
          'Max context: <strong>' + maxCtx + '</strong> · Layers: <strong>' + blocks + '</strong>' +
          (isMoe ? (' · MoE experts: <strong>' + (caps.expertCount || '?') + '</strong>') : ' · Dense (non-MoE)') +
          (caps.fullAttentionInterval > 1
            ? (' · hybrid full-attn every <strong>' + caps.fullAttentionInterval + '</strong> layers')
            : '') +
          (caps.nextnPredictLayers > 0
            ? (' · MTP next-n: <strong>' + caps.nextnPredictLayers + '</strong>')
            : '');
        const specHint = $('specHint');
        if (specHint) {
          specHint.textContent = caps.nextnPredictLayers > 0
            ? ('This model reports MTP next-n = ' + caps.nextnPredictLayers + '. Mode MTP passes --spec-type draft-mtp.')
            : 'This GGUF has no nextn_predict_layers metadata — MTP may be ignored or fail. Prefer an MTP-tagged model (e.g. Ornith MTP).';
        }
      } else {
        $('modelCaps').classList.add('hidden');
        $('ctxHint').textContent = 'Tokens for prompt + generation';
      }
    }

    function applyState(payload) {
      const s = payload.state;
      const L = s.loadSettings;
      const R = s.requestSettings;
      const status = payload.status;
      const hasModel = !!(s.selectedModelPath);
      const caps = payload.capabilities;

      applyCapabilities(caps);

      const build = payload.build || {};
      const ready = !!(status.httpReady || status.running);
      const endpoint = payload.endpoint || '';
      const endpointHtml = endpoint
        ? (ready
            ? '<a class="endpoint" href="' + endpoint + '" data-url="' + endpoint + '" title="Open llama-server web UI">' + endpoint + '</a>' +
              '<div class="endpoint-hint">Open in browser to chat with the model directly (llama.cpp web UI).</div>'
            : '<span>' + endpoint + '</span>' +
              '<div class="endpoint-hint">Start the server, then click the link to chat in the browser.</div>')
        : '';
      $('status').innerHTML =
        (ready
          ? '<span class="ok">● Server ready</span>'
          : '<span class="bad">● Server stopped</span>') +
        '<br/>' + (status.message || '') +
        (endpointHtml ? '<br/>' + endpointHtml : '');
      const ep = $('status').querySelector('a.endpoint');
      if (ep) {
        ep.addEventListener('click', (e) => {
          e.preventDefault();
          const url = ep.getAttribute('data-url');
          if (url) vscode.postMessage({ type: 'openExternal', url: url });
        });
      }

      serverRunning = ready;
      configDirty = !!status.configDirty;
      const lm = $('launchMode');
      if (lm && payload.launchMode) {
        lm.value = payload.launchMode === 'background' ? 'background' : 'externalTerminal';
      }
      updatePrimaryAction();

      const perfLines = Array.isArray(payload.perfLines) ? payload.perfLines : ['No generation yet'];
      const generating = payload.perf && payload.perf.generating;
      const perf = payload.perf || {};
      const ctxPct = typeof perf.contextPct === 'number' ? perf.contextPct : undefined;
      const ctxLevel = perf.contextLevel || 'ok';
      const ctxMeter = $('ctxMeter');
      const ctxLabel = $('ctxLabel');
      ctxMeter.className = 'ctx-meter' + (ctxLevel === 'warn' || ctxLevel === 'critical' ? ' ' + ctxLevel : '');
      ctxLabel.className = 'ctx-label' + (ctxLevel === 'warn' || ctxLevel === 'critical' ? ' ' + ctxLevel : '');
      if (typeof ctxPct === 'number' && typeof perf.promptTokens === 'number' && typeof perf.contextLimit === 'number') {
        const approx = perf.contextEstimated ? '≈' : '';
        ctxLabel.textContent =
          'Context: ' + approx + perf.promptTokens.toLocaleString() + ' / ' +
          perf.contextLimit.toLocaleString() + ' (' + ctxPct + '%)' +
          (ctxLevel === 'critical' ? ' · nearly full' : ctxLevel === 'warn' ? ' · running low' : '');
        $('ctxMeterFill').style.width = Math.min(100, ctxPct) + '%';
      } else {
        ctxLabel.textContent = 'Context: — (send a chat to measure)';
        $('ctxMeterFill').style.width = '0%';
      }
      $('perfLines').innerHTML =
        (generating ? '<span class="ok">● Generating</span><br/>' : '') +
        perfLines.map((l) => String(l)).join('<br/>');

      const binaryDetail = $('llamaBinaryDetail');
      if (binaryDetail) {
        if (build.binaryVersionDetail || build.binaryVersion) {
          binaryDetail.textContent = build.binaryVersionDetail || build.binaryVersion;
        } else if (build.tag && payload.binaryExists) {
          binaryDetail.textContent = 'Installed release ' + build.tag;
        } else if (payload.binaryExists) {
          binaryDetail.textContent = 'Binary installed (version string unavailable).';
        } else {
          binaryDetail.textContent = 'No binary for this backend yet.';
        }
      }
      $('llamaAssetDetail').textContent = build.asset
        ? ('Asset: ' + build.asset + (build.configuredBackend ? (' · setting: ' + build.configuredBackend) : ''))
        : 'No archive recorded for this backend yet.';

      updateCheck = payload.updateCheck || {
        latestTag: undefined,
        installedTag: build.tag,
        updateAvailable: false,
        checkFailed: false,
        pending: true,
      };

      const sel = $('backendSelect');
      const options = Array.isArray(payload.backendOptions) ? payload.backendOptions : [];
      backendOptionsCache = options;
      activeBackendId = payload.selectedUiBackend || (options.find((o) => o.active) || {}).id || '';
      const prev = sel.value;
      suppressBackendChange = true;
      sel.innerHTML = '';
      for (const opt of options) {
        const o = document.createElement('option');
        o.value = opt.id;
        let text = opt.label;
        if (!opt.available) {
          text += ' (unavailable)';
        } else if (opt.installed) {
          text += opt.installedTag
            ? (' · installed ' + opt.installedTag)
            : ' · installed';
          if (opt.active) text += ' ●';
        } else {
          text += ' · not installed';
        }
        o.textContent = text;
        o.disabled = !opt.available;
        if (opt.reason && !opt.available) {
          o.title = opt.reason;
        } else if (opt.installed && opt.installedTag) {
          o.title = 'Cached locally: ' + opt.installedTag;
        }
        sel.appendChild(o);
      }
      const want = activeBackendId || prev || 'vulkan';
      if ([...sel.options].some((o) => o.value === want && !o.disabled)) {
        sel.value = want;
      } else {
        const first = [...sel.options].find((o) => !o.disabled);
        if (first) sel.value = first.value;
      }
      suppressBackendChange = false;
      updateBackendUi();

      $('setupBox').classList.toggle('hidden', hasModel && payload.binaryExists);

      const localCount = payload.localModelCount || 0;
      const showStarter = !hasModel;
      const starterBtn = $('starterModelBtn');
      const starterHint = $('starterModelHint');
      const downloadBtn = $('downloadModelBtn');
      const pickBtn = $('pickDownloadedBtn');
      if (starterBtn) starterBtn.classList.toggle('hidden', !showStarter);
      if (starterHint) starterHint.classList.toggle('hidden', !showStarter);
      if (downloadBtn) {
        downloadBtn.className = showStarter && localCount === 0 ? 'secondary' : 'primary';
      }
      if (pickBtn) {
        // Prefer choosing an existing GGUF when the library already has files.
        pickBtn.className = showStarter && localCount > 0 ? 'primary' : 'secondary';
      }
      $('modelTitle').textContent = hasModel
        ? ('Selected: ' + (payload.modelName || 'model'))
        : 'No model selected';
      function escAttr(v) {
        return String(v).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
      }
      function escText(v) {
        return String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;');
      }
      function folderLink(label, dir, title) {
        return '<a class="folder-link" href="#" data-path="' + escAttr(dir) + '" title="' +
          escAttr(title || dir) + '">' + escText(label) + '</a>';
      }
      function bindFolderLinks(root) {
        root.querySelectorAll('a.folder-link, a.model-path-link').forEach((link) => {
          link.addEventListener('click', (e) => {
            e.preventDefault();
            const filePath = link.getAttribute('data-path');
            if (filePath) vscode.postMessage({ type: 'revealInOs', path: filePath });
          });
        });
      }

      if (hasModel && s.selectedModelPath) {
        const p = String(s.selectedModelPath);
        $('modelPath').innerHTML =
          '<a class="model-path-link" href="#" data-path="' + escAttr(p) +
          '" title="Reveal in File Explorer">' + escText(p) + '</a>';
        bindFolderLinks($('modelPath'));
      } else {
        $('modelPath').textContent = 'Choose one of the options below to install or select a GGUF model.';
      }

      const modelsDir = payload.modelsDir || '';
      const sourceDirs = Array.isArray(payload.localSourceDirs) ? payload.localSourceDirs : [];
      const sourceHtml = sourceDirs.length
        ? ' · sources: ' + sourceDirs.map((src) =>
            folderLink(src.source, src.dir, 'Open ' + src.source + ' folder')
          ).join(', ')
        : '';
      $('modelsDirMeta').innerHTML =
        'Llama AIO downloads: ' +
        (modelsDir ? folderLink(modelsDir, modelsDir, 'Open downloads folder') : '—') +
        ' · ' + (payload.localModelCount || 0) + ' GGUF found' +
        sourceHtml +
        ' (also scans LM Studio, Unsloth, HF cache, …)';
      bindFolderLinks($('modelsDirMeta'));

      const maxCtx = (caps && caps.maxContextLength) ? caps.maxContextLength : 131072;
      const blocks = (caps && caps.blockCount) ? caps.blockCount : 128;
      const ctx = Math.min(L.contextLength, maxCtx);
      const ngl = L.gpuOffload >= 99 ? L.gpuOffload : Math.min(L.gpuOffload, blocks);
      const moe = (caps && !caps.isMoe) ? 0 : Math.min(L.nCpuMoe, blocks);

      $('contextLength').value = ctx;
      $('contextLengthRange').value = ctx;
      $('gpuOffload').value = ngl;
      $('gpuOffloadRange').value = Math.min(ngl, Number($('gpuOffloadRange').max));
      $('cpuThreads').value = L.cpuThreads;
      $('cpuThreadsRange').value = L.cpuThreads;
      $('evalBatchSize').value = L.evalBatchSize;
      $('physicalBatchSize').value = L.physicalBatchSize;
      $('maxConcurrentPredictions').value = L.maxConcurrentPredictions;
      $('nCpuMoe').value = moe;
      $('nCpuMoeRange').value = moe;
      $('offloadKvCacheToGpu').checked = !!L.offloadKvCacheToGpu;
      $('keepModelInMemory').checked = !!L.keepModelInMemory;
      if (payload.isWindows) {
        const label = $('keepModelLabel');
        if (label) label.textContent = 'Keep Model in Memory (mmap on Windows)';
        const hint = $('keepModelHint');
        if (hint) hint.style.display = 'block';
      }
      $('tryMmap').checked = !!L.tryMmap;
      $('unifiedKvCache').checked = !!L.unifiedKvCache;
      $('contextCheckpoints').value = L.contextCheckpoints;
      $('ropeBaseAuto').checked = L.ropeFreqBase == null;
      $('ropeFreqBase').value = L.ropeFreqBase ?? 10000;
      $('ropeFreqBase').disabled = L.ropeFreqBase == null;
      $('ropeScaleAuto').checked = L.ropeFreqScale == null;
      $('ropeFreqScale').value = L.ropeFreqScale ?? 1;
      $('ropeFreqScale').disabled = L.ropeFreqScale == null;
      $('seedRandom').checked = L.seed == null;
      $('seed').value = L.seed ?? 0;
      $('seed').disabled = L.seed == null;
      $('speculativeMode').value = L.speculativeMode || 'off';
      $('maxDraftTokens').value = L.maxDraftTokens;
      $('minDraftTokens').value = L.minDraftTokens;
      $('draftProbability').value = L.draftProbability;
      $('temperature').value = R.temperature;
      $('topP').value = R.topP;
      $('topK').value = R.topK;
      $('maxTokens').value = R.maxTokens;

      memInputs = payload.memInputs || null;
      gpuInfo = payload.gpu || null;
      systemRamTotalBytes = payload.systemRamTotalBytes || 0;
      applyCpuOnlyUi(!!payload.cpuOnly || payload.selectedUiBackend === 'cpu');
      if (payload.memory) {
        renderMemory(payload.memory);
      } else {
        refreshMemoryLive();
      }
      }

    $('ropeBaseAuto').addEventListener('change', () => {
      $('ropeFreqBase').disabled = $('ropeBaseAuto').checked;
    });
    $('ropeScaleAuto').addEventListener('change', () => {
      $('ropeFreqScale').disabled = $('ropeScaleAuto').checked;
    });
    $('seedRandom').addEventListener('change', () => {
      $('seed').disabled = $('seedRandom').checked;
    });

    $('downloadModelBtn').addEventListener('click', () => vscode.postMessage({ type: 'downloadModel' }));
    const starterModelBtn = $('starterModelBtn');
    if (starterModelBtn) {
      starterModelBtn.addEventListener('click', () => vscode.postMessage({ type: 'downloadStarter' }));
    }
    const setupStarterBtn = $('setupStarterBtn');
    if (setupStarterBtn) {
      setupStarterBtn.addEventListener('click', () => vscode.postMessage({ type: 'downloadStarter' }));
    }
    $('openFileBtn').addEventListener('click', () => vscode.postMessage({ type: 'openModelFile' }));
    $('pickDownloadedBtn').addEventListener('click', () => vscode.postMessage({ type: 'pickDownloadedModel' }));
    $('installLlamaBtn').addEventListener('click', () => {
      const action = $('installLlamaBtn').dataset.action;
      if (action === 'check') {
        vscode.postMessage({ type: 'checkUpdates' });
        return;
      }
      vscode.postMessage({ type: 'installLlamaCpp', payload: $('backendSelect').value });
    });
    const checkUpdatesBtn = $('checkUpdatesBtn');
    if (checkUpdatesBtn) {
      checkUpdatesBtn.addEventListener('click', () => vscode.postMessage({ type: 'checkUpdates' }));
    }
    const reinstallLlamaBtn = $('reinstallLlamaBtn');
    if (reinstallLlamaBtn) {
      reinstallLlamaBtn.addEventListener('click', () => vscode.postMessage({ type: 'reinstallLlamaCpp' }));
    }
    $('installByTagBtn').addEventListener('click', () => vscode.postMessage({ type: 'installLlamaCppByTag' }));
    $('installArchiveBtn').addEventListener('click', () => vscode.postMessage({ type: 'installLlamaCppFromArchive' }));
    const releasesLink = $('releasesLink');
    if (releasesLink) {
      releasesLink.addEventListener('click', (e) => {
        e.preventDefault();
        vscode.postMessage({ type: 'openExternal', url: 'https://github.com/ggml-org/llama.cpp/releases' });
      });
    }
    $('backendSelect').addEventListener('change', () => {
      if (suppressBackendChange) return;
      updateBackendUi();
      applyCpuOnlyUi($('backendSelect').value === 'cpu');
      refreshMemoryLive();
      const next = $('backendSelect').value;
      if (!next || next === activeBackendId) return;
      const opt = backendOptionsCache.find((o) => o.id === next);
      if (opt && !opt.available) return;
      vscode.postMessage({ type: 'switchBackend', payload: next });
    });

    $('primaryBtn').addEventListener('click', () => {
      const action = $('primaryBtn').dataset.action;
      if (action === 'reload') {
        vscode.postMessage({ type: 'reload', payload: readLoad() });
        vscode.postMessage({ type: 'saveRequest', payload: readRequest() });
      } else if (action === 'start') {
        vscode.postMessage({ type: 'saveLoad', payload: readLoad() });
        vscode.postMessage({ type: 'saveRequest', payload: readRequest() });
        vscode.postMessage({ type: 'start' });
      }
    });
    $('stopBtn').addEventListener('click', () => vscode.postMessage({ type: 'stop' }));
    $('launchMode').addEventListener('change', () => {
      const mode = $('launchMode').value === 'background' ? 'background' : 'externalTerminal';
      if (serverRunning) {
        configDirty = true;
        updatePrimaryAction();
      }
      vscode.postMessage({ type: 'setLaunchMode', payload: mode });
    });

    window.addEventListener('message', (event) => {
      const msg = event.data;
      if (msg.type === 'state') applyState(msg.payload);
      if (msg.type === 'statusPatch' && msg.payload) {
        serverRunning = !!msg.payload.running;
        configDirty = !!msg.payload.configDirty;
        updatePrimaryAction();
      }
    });

    vscode.postMessage({ type: 'ready' });
  </script>
</body>
</html>`;
  }
}

function getNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let nonce = "";
  for (let i = 0; i < 32; i++) {
    nonce += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return nonce;
}

#!/usr/bin/env node
/*
 * Serve the real Llama AIO sidebar (SettingsViewProvider.getHtml + its message
 * handler) in a normal browser, with fake VS Code services. Used for visual and
 * behavioural checks of the webview without launching VS Code.
 *
 *   npx tsc -p packages/vscode/tsconfig.json --outDir packages/vscode/out-harness
 *   node packages/vscode/scripts/sidebar-harness.cjs [--out out-harness] [--port 4817]
 *
 * Seeds state read-only from ~/.llama-aio-vs/config.json when present. GPUs come
 * from the real detector (sysfs / nvidia-smi); nothing is written to the user config.
 *
 * POST /scenario {"name": "running"|"stopped"|"dirty"|"spill"|"cpu"} resets state.
 * GET  /debug returns the last messages exchanged.
 */
const Module = require("module");
const http = require("http");
const path = require("path");
const fs = require("fs");
const os = require("os");

const args = process.argv.slice(2);
const argOf = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
};
const outDir = path.resolve(__dirname, "..", argOf("--out", "out-harness"));
const port = Number(argOf("--port", "4817"));

const vscodeMock = {
  window: {
    showWarningMessage: async () => undefined,
    showInformationMessage: async () => undefined,
    showErrorMessage: async (m) => console.error("[showErrorMessage]", m),
    withProgress: async (_o, fn) => fn({ report() {} }),
    showTextDocument: async () => undefined,
    createOutputChannel: () => ({ appendLine() {}, append() {}, show() {}, clear() {}, dispose() {} }),
  },
  workspace: { openTextDocument: async () => ({}), getConfiguration: () => ({ get: () => undefined }) },
  commands: { executeCommand: async () => undefined },
  env: { openExternal: async () => true, clipboard: { writeText: async () => undefined } },
  Uri: { parse: (s) => ({ toString: () => s }), file: (s) => ({ fsPath: s }), joinPath: () => ({}) },
  ProgressLocation: { Notification: 15 },
  ViewColumn: { Active: -1 },
  EventEmitter: class {
    constructor() {
      this.event = () => ({ dispose() {} });
    }
    fire() {}
    dispose() {}
  },
};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === "vscode") {
    return "vscode";
  }
  return origResolve.call(this, request, ...rest);
};
require.cache.vscode = { id: "vscode", filename: "vscode", loaded: true, exports: vscodeMock };

const core = require("@llama-aio/core");
const { SettingsViewProvider } = require(path.join(outDir, "settingsView.js"));

const home = os.homedir();
let seed = {};
try {
  seed = JSON.parse(fs.readFileSync(path.join(home, ".llama-aio-vs", "config.json"), "utf8")).state || {};
} catch {
  seed = {};
}
const modelPath = seed.selectedModelPath || "";
const vulkanBinary = path.join(home, ".llama-aio-vs", "llama.cpp", "vulkan", "bin", "llama-server");
const realBinary = fs.existsSync(vulkanBinary) ? vulkanBinary : "";

function freshState(overrides = {}) {
  let caps;
  try {
    caps = modelPath ? core.readModelCapabilities(modelPath) : undefined;
  } catch {
    caps = undefined;
  }
  return {
    selectedModelPath: modelPath,
    modelCapabilities: caps,
    modelMaxContext: caps?.maxContextLength,
    loadSettings: { ...core.DEFAULT_LOAD_SETTINGS, ...(seed.loadSettings || {}), ...overrides },
    requestSettings: { ...core.DEFAULT_REQUEST_SETTINGS, ...(seed.requestSettings || {}) },
  };
}

let state = freshState();
const config = new Map([["launchMode", "externalTerminal"]]);
let running = true;
let cpuBackend = false;
let launched = snapshot();

function snapshot() {
  return {
    modelPath: state.selectedModelPath,
    loadSettings: JSON.parse(JSON.stringify(state.loadSettings)),
    launchMode: config.get("launchMode"),
    startedAt: new Date(Date.now() - 42 * 60_000).toISOString(),
  };
}

const store = {
  getState: () => state,
  getConfig: () => ({ get: (k) => config.get(k), update: async (k, v) => void config.set(k, v) }),
  updateLoadSettings: async (p) => {
    state = { ...state, loadSettings: { ...state.loadSettings, ...p } };
  },
  updateRequestSettings: async (p) => {
    state = { ...state, requestSettings: { ...state.requestSettings, ...p } };
  },
  applySelectedModel: async (p) => {
    state = { ...state, selectedModelPath: p, modelCapabilities: core.readModelCapabilities(p) };
    return state;
  },
  isPromptReplacementsEnabled: () => true,
  isWikipediaLookupEnabled: () => false,
  isDuplicateToolCallGuardEnabled: () => false,
  getEndpoint: () => "http://127.0.0.1:8742",
  getSlotContextSize: (l) => l.contextLength,
};

const sameConfig = () =>
  JSON.stringify(launched.loadSettings) === JSON.stringify(state.loadSettings) &&
  launched.launchMode === config.get("launchMode") &&
  launched.modelPath === state.selectedModelPath;

const processManager = {
  getStatus: () => ({
    running,
    starting: false,
    pid: running ? 8992 : undefined,
    endpoint: "http://127.0.0.1:8742",
    message: running ? "Running (pid 8992)" : "Stopped",
    configDirty: running && !sameConfig(),
  }),
  isHttpReady: async () => running,
  // The real binary gives llama.cpp's device order via --list-devices (read-only probe).
  resolveBinary: () => realBinary,
  isCpuBackend: () => cpuBackend,
  getLaunchedConfig: () => (running ? launched : undefined),
  claimLaunch: () => ({ id: 1 }),
  releaseLaunch() {},
  start: async () => {
    running = true;
    launched = snapshot();
    return { message: "Server ready" };
  },
  stop: async () => {
    running = false;
  },
};

const installer = {
  getInstalledInfo: () => ({ activeBackend: cpuBackend ? "cpu" : "vulkan", tag: "b11163", binaryVersion: "version: 11163" }),
  getUiBackendOptions: () => [
    { id: "vulkan", label: "Vulkan", installed: true, installedTag: "b11163", active: !cpuBackend, available: true },
    { id: "rocm", label: "ROCm", installed: false, active: false, available: true },
    { id: "cpu", label: "CPU", installed: cpuBackend, installedTag: cpuBackend ? "b11163" : undefined, active: cpuBackend, available: true },
    { id: "path", label: "PATH", installed: false, active: false, available: true },
  ],
  resolveActiveUiBackend: () => (cpuBackend ? "cpu" : "vulkan"),
  peekUpdateCheck: () => ({ latestTag: "b11165", installedTag: "b11163", updateAvailable: true, checkFailed: false, pending: false }),
  getUpdateCheck: async () => ({ latestTag: "b11165", installedTag: "b11163", updateAvailable: true, checkFailed: false, pending: false }),
};

const now = Date.now();
const gens = [39.8, 38.9, 37.5, 36.9, 37.8, 41.2, 37.4, 36.2, 35.9];
const prompts = [150, 190, 120, 130, 125, 100, 160, 330, 365];
const history = gens
  .map((g, i) => ({
    genTokPerSec: g,
    promptTokPerSec: prompts[i],
    cacheHitPct: 90 + (i % 5),
    finishedAt: now - (gens.length - i) * 90_000,
    completionTokens: 300 + i * 20,
    durationMs: 60_000,
  }))
  .reverse();
const perfSnap = {
  generating: false,
  genTokPerSec: 35.9,
  promptTokPerSec: 365,
  cacheHitPct: 94.6,
  draftAcceptancePct: 96.7,
  completionTokens: 419,
  startedAt: now - 162_600,
  finishedAt: now,
  speculativeMode: "mtp",
  promptTokens: 61000,
  contextLimit: 81664,
  contextPct: 74.7,
  contextLevel: "ok",
  contextBreakdown: {
    segments: [
      { key: "tools", label: "Tools", tokens: 22400 },
      { key: "system", label: "System", tokens: 3100 },
      { key: "history", label: "History", tokens: 4200 },
      { key: "toolResults", label: "Tool results", tokens: 30700 },
      { key: "request", label: "Request", tokens: 600 },
      { key: "free", label: "Free", tokens: 20664 },
    ],
    usedTokens: 61000,
    limitTokens: 81664,
  },
  history,
};
const perf = {
  get: () => perfSnap,
  detailLines: () => ["Last: 35.9 tok/s"],
  hasLastRequestContext: () => true,
  hasLastResponseTrace: () => true,
  setSpeculativeMode() {},
  formatLastRequestContext: () => "",
  formatLastResponseTrace: () => "",
};

const modelActions = new Proxy(
  {},
  {
    get: (_t, name) => async () => {
      log.push({ dir: "action", name: String(name) });
    },
  }
);

const log = [];
const clients = new Set();
let handler = async () => undefined;
const webview = {
  options: {},
  html: "",
  cspSource: "",
  asWebviewUri: (u) => u,
  postMessage: (m) => {
    log.push({ dir: "toWebview", type: m.type, seq: m.seq });
    const data = `data: ${JSON.stringify(m)}\n\n`;
    for (const res of clients) {
      res.write(data);
    }
    return Promise.resolve(true);
  },
  onDidReceiveMessage: (h) => {
    handler = h;
    return { dispose() {} };
  },
};

const provider = new SettingsViewProvider(
  {},
  store,
  processManager,
  installer,
  perf,
  async () => {
    running = true;
    launched = snapshot();
  },
  modelActions,
  () => undefined,
  { get: () => undefined, update: async () => undefined }
);
provider.resolveWebviewView({ webview, onDidDispose: () => ({ dispose() {} }), visible: true }, {}, {});

const shim = `<script>
  window.__sent = [];
  window.__errs = [];
  window.addEventListener('error', (e) => window.__errs.push(e.message + ' @' + e.lineno + ':' + e.colno));
  window.acquireVsCodeApi = () => ({
    postMessage: (m) => { window.__sent.push(m); fetch('/msg', { method: 'POST', body: JSON.stringify(m) }); },
    getState: () => { try { return JSON.parse(sessionStorage.getItem('vsstate') || 'null'); } catch { return null; } },
    setState: (s) => { sessionStorage.setItem('vsstate', JSON.stringify(s)); return s; },
  });
  window.__received = [];
  const es = new EventSource('/events');
  es.onmessage = (e) => { const data = JSON.parse(e.data); window.__received.push(data.type); window.dispatchEvent(new MessageEvent('message', { data })); };
</script>`;

function pageHtml() {
  let html = webview.html;
  html = html.replace(/<meta http-equiv="Content-Security-Policy"[^>]*>/, "");
  html = html.replace("<head>", `<head>${shim}<style>body{max-width:${argOf("--width", "360")}px;margin:0 auto;border-left:1px solid #333;border-right:1px solid #333;min-height:100vh}</style>`);
  // VS Code theme variables the webview relies on.
  html = html.replace(
    "<style>",
    `<style>:root{--vscode-sideBar-background:#181818;--vscode-foreground:#cccccc;--vscode-descriptionForeground:#9d9d9d;--vscode-input-background:#252526;--vscode-input-foreground:#cccccc;--vscode-input-border:#3c3c3c;--vscode-button-background:#0e639c;--vscode-button-foreground:#fff;--vscode-button-secondaryBackground:#3a3d41;--vscode-button-secondaryForeground:#fff;--vscode-textLink-foreground:#3794ff;--vscode-font-family:"Segoe UI",system-ui,sans-serif}</style><style>`
  );
  return html;
}

function applyScenario(name) {
  cpuBackend = false;
  config.set("launchMode", "externalTerminal");
  state = freshState();
  running = name !== "stopped";
  launched = snapshot();
  if (name === "dirty") {
    state = { ...state, loadSettings: { ...state.loadSettings, contextLength: 65536, cacheTypeV: "q4_0" } };
  } else if (name === "spill") {
    state = freshState({ contextLength: 262144, cacheTypeK: "f16", cacheTypeV: "f16" });
    launched = snapshot();
  } else if (name === "cpu") {
    cpuBackend = true;
    state = freshState({ contextLength: 32768 });
    launched = snapshot();
  }
}

const server = http.createServer(async (req, res) => {
  if (req.url === "/" || req.url?.startsWith("/?")) {
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    res.end(pageHtml());
    return;
  }
  if (req.url === "/events") {
    res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
    res.write(": ok\n\n");
    clients.add(res);
    req.on("close", () => clients.delete(res));
    // "ready" usually arrives before the stream is open; push once it is.
    setTimeout(() => void provider.pushState(), 50);
    return;
  }
  if (req.method === "POST" && (req.url === "/msg" || req.url === "/scenario")) {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", async () => {
      try {
        const msg = JSON.parse(body || "{}");
        if (req.url === "/scenario") {
          applyScenario(msg.name || "running");
          await provider.pushState();
        } else {
          log.push({ dir: "fromWebview", type: msg.type, seq: msg.seq });
          await handler(msg);
        }
        res.writeHead(204);
        res.end();
      } catch (e) {
        console.error(e);
        res.writeHead(500);
        res.end(String(e));
      }
    });
    return;
  }
  if (req.method === "POST" && req.url === "/coreEstimate") {
    // Core estimate for arbitrary settings, to compare against what the page shows.
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      const patch = JSON.parse(body || "{}");
      const settings = { ...state.loadSettings, ...patch };
      const gpus = cpuBackend ? [] : core.detectGpus(false, realBinary);
      const { estimate, view } = core.computeMemoryView(state.modelCapabilities, settings, gpus, { cpuOnly: cpuBackend });
      const perGpu = (estimate?.charts.gpus || []).map((c) => ({ gpu: c.gpuIndex, total: c.totalBytes }));
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ perGpu, ram: estimate?.charts.ram.totalBytes, willSpill: estimate?.willSpill, level: view?.level }));
    });
    return;
  }
  if (req.url === "/debug") {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ state: state.loadSettings, launched: launched.loadSettings, running, log: log.slice(-60) }, null, 1));
    return;
  }
  res.writeHead(404);
  res.end();
});
server.listen(port, "127.0.0.1", () => {
  console.log(`sidebar harness on http://127.0.0.1:${port}/ (model: ${path.basename(modelPath) || "none"})`);
});

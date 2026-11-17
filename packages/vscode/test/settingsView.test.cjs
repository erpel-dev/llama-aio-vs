// Message-level tests for the sidebar provider (estimate, fit preset, discard, spill
// confirmation). Runs the compiled provider with a stub `vscode` module.
//   npm test -w llama-aio-vs
const assert = require("node:assert/strict");
const { describe, it, beforeEach } = require("node:test");
const Module = require("module");
const path = require("path");
const fs = require("node:fs");
const os = require("node:os");
const http = require("node:http");

const shown = [];
let copilotChatInstalled = true;
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  return request === "vscode" ? "vscode" : origResolve.call(this, request, ...rest);
};
let lastPanel;
require.cache.vscode = {
  id: "vscode",
  filename: "vscode",
  loaded: true,
  exports: {
    window: {
      showWarningMessage: async (msg, opts) => {
        shown.push({ msg, modal: !!(opts && opts.modal) });
        return undefined;
      },
      showInformationMessage: async () => undefined,
      showErrorMessage: async () => undefined,
      showOpenDialog: async () => undefined,
      showInputBox: async () => undefined,
      // Attachment pickers list library files first; by default tests take "Browse…".
      showQuickPick: async (items) => items.find((i) => i.browse),
      withProgress: async (_o, fn) => fn({ report() {} }),
      // Advanced settings tab: a fake panel whose messages tests can drive.
      createWebviewPanel: () => {
        const panel = {
          posted: [],
          webview: {
            options: {}, html: "", cspSource: "",
            asWebviewUri: (u) => ({ toString: () => u.fsPath }),
            postMessage: (m) => { panel.posted.push(m); return Promise.resolve(true); },
            onDidReceiveMessage: (h) => { panel.handler = h; return { dispose() {} }; },
          },
          onDidDispose: (h) => { panel.dispose = h; return { dispose() {} }; },
          reveal() { panel.revealed = (panel.revealed || 0) + 1; },
        };
        lastPanel = panel;
        return panel;
      },
    },
    workspace: {},
    commands: { executeCommand: async () => undefined },
    env: {},
    UIKind: { Desktop: 1, Web: 2 },
    extensions: { getExtension: (id) => (copilotChatInstalled && id === "GitHub.copilot-chat" ? {} : undefined) },
    QuickPickItemKind: { Separator: -1, Default: 0 },
    Uri: { file: fsPath => ({ fsPath }), joinPath: (base, ...parts) => ({ fsPath: path.join(base.fsPath, ...parts) }) },
    ProgressLocation: {},
    lm: undefined,
    LanguageModelChatMessage: undefined,
    LanguageModelTextPart: undefined,
    CancellationTokenSource: undefined,
    ViewColumn: {},
    EventEmitter: class {},
  },
};

const core = require("@llama-aio/core");
const { SettingsViewProvider } = require(path.join(__dirname, "..", "out-harness", "settingsView.js"));

it("an open managed gateway stops forwarding in remote mode and resumes in local mode", async (t) => {
  const seen = [];
  const server = http.createServer((req, res) => { seen.push(req.headers.authorization); res.end("ok"); });
  await new Promise(r => server.listen(0, "127.0.0.1", r));
  let remote = false;
  const provider = Object.create(SettingsViewProvider.prototype);
  provider.store = { isRemote: () => remote, getEndpoint: () => `http://127.0.0.1:${server.address().port}` };
  t.after(() => { provider.dispose(); server.close(); });
  const url = await provider.ensureWebUiGateway();
  // The test uses its own key, never the user's managed credential.
  provider.webUiGateway.options.apiKey = () => "synthetic-test-key";
  const request = () => new Promise((resolve, reject) => {
    http.get(url + "props", res => { res.resume(); res.on("end", () => resolve(res.statusCode)); }).on("error", reject);
  });
  assert.equal(await request(), 200);
  remote = true;
  assert.equal(await request(), 502);
  assert.deepEqual(seen, ["Bearer synthetic-test-key"]);
  remote = false;
  assert.equal(await request(), 200);
  assert.equal(seen.length, 2);
});

describe("removal through a symlinked models directory", () => {
  for (const role of ["main", "mmproj", "draft"]) {
    it(`stops before deleting and clears the ${role} path after unlink`, async (t) => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-remove-alias-"));
      t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
      const real = path.join(dir, "real"); fs.mkdirSync(real);
      const alias = path.join(dir, "models"); fs.symlinkSync(real, alias, "junction");
      const target = path.join(real, "target.gguf"); fs.writeFileSync(target, "GGUF");
      const selected = path.join(alias, "target.gguf");
      let state = {
        selectedModelPath: role === "main" ? selected : "/other/model.gguf",
        modelCapabilities: { path: selected }, modelMaxContext: 8192,
        loadSettings: { ...core.DEFAULT_LOAD_SETTINGS,
          mmprojPath: role === "mmproj" ? selected : "",
          draftModelPath: role === "draft" ? selected : "" },
      };
      const events = [];
      await SettingsViewProvider.prototype.removeOwnedQuant.call({
        store: { getConfig: () => ({ get: key => key === "modelsDir" ? alias : undefined }),
          getState: () => state, setState: async patch => { state = { ...state, ...patch }; } },
        processManager: { getStatus: () => ({ running: true, modelPath: state.selectedModelPath }),
          stop: async force => { assert.equal(force, true); events.push("stop"); } },
        trashOwnedFile: async file => { events.push("delete"); fs.unlinkSync(file); },
        removeEmptyOwnedDir: async () => undefined, pushState: async () => undefined,
      }, selected, []);
      assert.deepEqual(events, ["stop", "delete"]);
      assert.equal(fs.existsSync(selected), false);
      if (role === "main") {
        assert.equal(state.selectedModelPath, "");
        assert.equal(state.modelCapabilities, undefined);
        assert.equal(state.modelMaxContext, undefined);
      } else {
        assert.equal(state.selectedModelPath, "/other/model.gguf");
        assert.equal(state.loadSettings[role === "mmproj" ? "mmprojPath" : "draftModelPath"], "");
      }
    });
  }
});

describe("partial download cleanup", () => {
  for (const startDuringConfirmation of [false, true]) {
    it(`preserves a transfer${startDuringConfirmation ? " started during confirmation" : " already running"}`, async (t) => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-clean-active-"));
      let response;
      const server = http.createServer((_req, res) => {
        response = res;
        res.writeHead(200, { "Content-Length": 2 }); res.write("a");
      });
      await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
      const dest = path.join(dir, "active.gguf");
      const orphan = path.join(dir, "orphan.gguf.partial");
      const spec = { dest, label: "active", expectedSize: 2, url: `http://127.0.0.1:${server.address().port}` };
      let job;
      t.after(async () => {
        if (job) core.downloadManager.cancel(job.job.id);
        server.closeAllConnections();
        await new Promise(resolve => server.close(resolve));
        core.downloadManager.clearFinished();
        fs.rmSync(dir, { recursive: true, force: true });
      });
      const waitFor = async check => {
        const until = Date.now() + 2000;
        while (!check()) {
          assert.ok(Date.now() < until, "transfer did not start");
          await new Promise(resolve => setTimeout(resolve, 5));
        }
      };
      if (startDuringConfirmation) fs.writeFileSync(dest + ".partial", "z");
      else {
        job = core.downloadManager.enqueue(spec);
        await waitFor(() => fs.existsSync(dest + ".partial") && fs.statSync(dest + ".partial").size === 1);
        fs.writeFileSync(orphan, "unused");
      }
      const removed = [];
      t.mock.method(require("vscode").window, "showWarningMessage", async () => {
        if (startDuringConfirmation) job = core.downloadManager.enqueue(spec);
        return "Clean";
      });
      await SettingsViewProvider.prototype.cleanPartialDownloads.call({
        store: { getConfig: () => ({ get: key => key === "modelsDir" ? dir : undefined }) },
        trashOwnedFile: async file => { removed.push(file); await fs.promises.rm(file); },
        pushState: async () => undefined,
      });
      assert.deepEqual(removed, startDuringConfirmation ? [] : [orphan]);
      assert.equal(fs.existsSync(dest + ".partial"), true);
      await waitFor(() => !!response);
      response.end("b");
      await job.done;
      assert.equal(fs.readFileSync(dest, "utf8"), "ab");
    });
  }
});

const GiB = 1024 ** 3;
const caps = {
  path: "/m/Test-27B-Q4_K_M.gguf",
  name: "Test 27B",
  architecture: "qwen3",
  fileSizeBytes: 11 * GiB,
  blockCount: 48,
  embeddingLength: 5120,
  attentionHeadCount: 40,
  attentionHeadCountKv: 8,
  keyLength: 128,
  valueLength: 128,
  maxContextLength: 262144,
  isMoe: false,
};
const gpus = [
  { totalBytes: 16 * GiB, usedBytes: 1 * GiB, name: "AMD Radeon RX 9070", index: 0, llamaDeviceId: "Vulkan0", source: "test" },
  { totalBytes: 16 * GiB, usedBytes: 0, name: "AMD Radeon RX 9060 XT", index: 1, llamaDeviceId: "Vulkan1", source: "test" },
];

function setup(loadOverrides = {}, remoteInfo, serverFeatures = core.NO_SERVER_FEATURES, modelCaps = caps, opts = {}) {
  let state = {
    selectedModelPath: caps.path,
    modelCapabilities: modelCaps,
    loadSettings: { ...core.DEFAULT_LOAD_SETTINGS, contextLength: 65536, splitMode: "none", ...loadOverrides },
    requestSettings: { ...core.DEFAULT_REQUEST_SETTINGS },
  };
  const config = new Map([["launchMode", "externalTerminal"]]);
  if (opts.modelsDir) config.set("modelsDir", opts.modelsDir);
  const loraApplied = [];
  const launched = {
    modelPath: caps.path,
    loadSettings: JSON.parse(JSON.stringify(opts.cpu ? core.normalizeLoadSettingsForCpuBackend(state.loadSettings) : state.loadSettings)),
    launchMode: "externalTerminal",
    startedAt: new Date().toISOString(),
  };
  const store = {
    getState: () => state,
    getConfig: () => ({ get: (k) => config.get(k), update: async (k, v) => void config.set(k, v) }),
    updateLoadSettings: async (p) => {
      state = { ...state, loadSettings: { ...state.loadSettings, ...p } };
    },
    updateRequestSettings: async () => undefined,
    applySelectedModel: async () => state,
    isPromptReplacementsEnabled: () => false,
    isWikipediaLookupEnabled: () => false,
    isDuplicateToolCallGuardEnabled: () => false,
    getEndpoint: () => "http://127.0.0.1:8742",
    isRemote: () => !!remoteInfo,
    getServerPreferences: () => ({
      mode: remoteInfo ? "remote" : "local",
      remoteUrl: "",
      remoteLabel: "",
      lastRemoteModelId: "",
      plainHttpConfirmed: [],
    }),
    getServerTarget: () => ({ mode: remoteInfo ? "remote" : "local", baseUrl: "http://127.0.0.1:8742", label: "this machine" }),
  };
  const pm = {
    getStatus: () => ({ running: true, starting: false, pid: 1, endpoint: "http://127.0.0.1:8742", message: "Running", configDirty: JSON.stringify(state.loadSettings) !== JSON.stringify(launched.loadSettings) }),
    refreshStatus: async () => pm.getStatus(),
    isHttpReady: async () => true,
    resolveBinary: () => "",
    getServerFeatures: async () => ({ ...serverFeatures }),
    applyLoraScales: async () => {
      if (opts.loraFails) throw new Error("server said no");
      const wanted = state.loadSettings.loraAdapters || [];
      loraApplied.push(JSON.parse(JSON.stringify(wanted)));
      launched.loadSettings.loraAdapters = launched.loadSettings.loraAdapters.map(a => wanted.find(w => w.path === a.path) ?? a);
    },
    isCpuBackend: () => !!opts.cpu,
    effectiveLaunchMode: () => config.get("launchMode") === "background" ? "background" : "externalTerminal",
    launchModeFallbackReason: () => undefined,
    getLaunchedConfig: () => launched,
    claimLaunch: () => ({}),
    releaseLaunch() {},
    remote: { current: () => remoteInfo, nextRetryAt: () => undefined },
  };
  const installer = {
    getInstalledInfo: () => ({ activeBackend: "vulkan", tag: "b1" }),
    getInstalledInfoCached() {
      return this.getInstalledInfo();
    },
    getUiBackendOptions: () => [],
    resolveActiveUiBackend: () => "vulkan",
    peekUpdateCheck: () => ({ pending: false }),
    getUpdateCheck: async () => ({ pending: false }),
  };
  const perf = { get: () => ({}), detailLines: () => [], hasLastRequestContext: () => false, hasLastResponseTrace: () => false, setSpeculativeMode() {}, setTarget() {}, getTarget() { return undefined; } };
  const posted = [];
  let handler;
  const provider = new SettingsViewProvider(
    { fsPath: path.join(__dirname, "..") },
    store,
    pm,
    installer,
    perf,
    async () => undefined,
    {},
    () => undefined
  );
  // Deterministic GPUs instead of probing the test machine.
  provider.detectGpusForEstimate = () => gpus;
  provider.resolveWebviewView(
    {
      webview: {
        options: {},
        html: "",
        cspSource: "",
        asWebviewUri: (u) => ({ toString: () => u.fsPath }),
        postMessage: (m) => {
          posted.push(m);
          return Promise.resolve(true);
        },
        onDidReceiveMessage: (h) => {
          handler = h;
          return { dispose() {} };
        },
      },
      onDidDispose: () => ({ dispose() {} }),
    },
    {},
    {}
  );
  return { send: (m) => handler(m), posted, getState: () => state, launched, provider, loraApplied };
}

describe("PLE attachment controls", () => {
  it("renders the real sidebar bundle through pick, clear, missing-file and old-build states", async t => {
    const { JSDOM } = require("jsdom");
    const html = fs.readFileSync(path.join(__dirname, "../src/webview/sidebar.html"), "utf8").replace(/\{\{\w+\}\}/g, "");
    const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true, url: "https://preview.invalid" });
    t.after(() => dom.window.close());
    const sent = [];
    dom.window.acquireVsCodeApi = () => ({ postMessage: m => sent.push(m), getState: () => null, setState() {} });
    dom.window.eval(fs.readFileSync(path.join(__dirname, "../out/webview/sidebar.js"), "utf8"));
    const ui = setup({}, undefined, { ...core.NO_SERVER_FEATURES, externalPle: "--shard" }, { ...caps, pleSupported: true });
    await ui.provider.pushState();
    const payload = ui.posted.filter(m => m.type === "state").at(-1).payload;
    const paint = patch => dom.window.dispatchEvent(new dom.window.MessageEvent("message", { data: { type: "state", payload: { ...payload, ...patch } } }));
    const el = id => dom.window.document.getElementById(id);
    paint({});
    assert.equal(el("pleName").textContent, "Built-in");
    assert.equal(el("pickPleBtn").disabled, false);
    el("pickPleBtn").click();
    assert.equal(sent.at(-1).type, "pickPle");
    paint({ pleTable: { supported: true, path: "/models/very-long-name-ple.gguf", bytes: 1024 } });
    assert.equal(el("pleName").textContent, "very-long-name-ple.gguf");
    assert.equal(el("clearPleBtn").classList.contains("hidden"), false);
    el("clearPleBtn").click();
    assert.equal(sent.at(-1).type, "clearPle");
    paint({ pleTable: { supported: true, path: "/missing.gguf", problem: "File not found" }, serverFeatures: core.NO_SERVER_FEATURES });
    assert.equal(el("pleHint").textContent, "File not found");
    assert.equal(el("pickPleBtn").disabled, true);
    assert.equal(el("clearPleBtn").disabled, false, "can detach after a binary downgrade");
    paint({ pleTable: { supported: true, missing: true }, memoryView: null });
    assert.equal(el("pleName").textContent, "Not attached");
    assert.equal(el("memVerdict").textContent, "Attach a PLE table to estimate memory.");
    paint({ pleTable: { supported: false, path: "" } });
    assert.equal(el("pleAttach").classList.contains("hidden"), true);
  });
  it("picks a table without replacing the model, lists Reload changes, and clears it", async t => {
    t.mock.method(require("vscode").window, "showOpenDialog", async () => [{ fsPath: "/m/ple-q8.gguf" }]);
    const ui = setup({}, undefined, { ...core.NO_SERVER_FEATURES, probed: true, externalPle: "--shard" }, { ...caps, pleSupported: true });
    await ui.send({ type: "pickPle" });
    assert.equal(ui.getState().selectedModelPath, caps.path);
    assert.equal(ui.getState().loadSettings.plePath, "/m/ple-q8.gguf");
    let payload = ui.posted.filter(m => m.type === "state").at(-1).payload;
    assert.ok(payload.changes.some(c => c.key === "plePath" && c.to === "ple-q8.gguf"));
    assert.equal(payload.pleTable.path, "/m/ple-q8.gguf");
    await ui.send({ type: "clearPle" });
    assert.equal(ui.getState().loadSettings.plePath, "");
    payload = ui.posted.filter(m => m.type === "state").at(-1).payload;
    assert.equal(payload.changes.some(c => c.key === "plePath"), false);
  });
  it("refuses unsupported and remote picks before opening a dialog", async t => {
    const dialog = t.mock.method(require("vscode").window, "showOpenDialog");
    await assert.rejects(setup().provider.pickPle(), /Update llama.cpp/);
    await assert.rejects(setup({}, { state: "ready" }).provider.pickPle(), /remote server/);
    assert.equal(dialog.mock.callCount(), 0);
  });
  it("rejects a pick when the connection changes while the file dialog is open", async t => {
    const ui = setup({}, undefined, { ...core.NO_SERVER_FEATURES, externalPle: "--shard" });
    t.mock.method(require("vscode").window, "showOpenDialog", async () => {
      ui.provider.store.isRemote = () => true;
      return [{ fsPath: "/m/ple.gguf" }];
    });
    await assert.rejects(ui.provider.pickPle(), /connection changed/);
    assert.equal(ui.getState().loadSettings.plePath, "");
  });
  it("leaves a selection unchanged when the dialog is cancelled", async () => {
    const ui = setup({ plePath: "/m/existing.gguf" }, undefined, { ...core.NO_SERVER_FEATURES, externalPle: "--ple" });
    await ui.provider.pickPle();
    assert.equal(ui.getState().loadSettings.plePath, "/m/existing.gguf");
  });
  it("discarding a pre-feature launch snapshot restores the built-in table", async () => {
    const ui = setup();
    delete ui.launched.loadSettings.plePath;
    await ui.send({ type: "saveLoad", payload: { plePath: "/m/new.gguf" }, silent: true });
    await ui.send({ type: "discardChanges" });
    assert.equal(ui.getState().loadSettings.plePath, "");
  });
});

describe("SettingsViewProvider memory path", () => {
  for (const fail of [false, true]) {
    it(`finishes a library rescan and restores the refresh control (failure=${fail})`, async () => {
      const { provider, send, posted } = setup();
      let scans = 0;
      provider.pushState = async () => { scans++; if (fail) throw new Error("scan failed"); };
      await send({ type: "refreshLibrary" });
      assert.equal(scans, 1);
      assert.ok(posted.some(message => message.type === "libraryRefreshComplete"));
    });
  }
  beforeEach(() => {
    shown.length = 0;
    copilotChatInstalled = true;
  });

  it("state push says whether Copilot Chat is available", async () => {
    let t = setup();
    await t.send({ type: "ready" });
    assert.equal(t.posted.find((m) => m.type === "state").payload.copilotChatAvailable, true);
    copilotChatInstalled = false;
    t = setup();
    await t.send({ type: "ready" });
    assert.equal(t.posted.find((m) => m.type === "state").payload.copilotChatAvailable, false);
  });

  for (const [identity, family, defaultMode] of [
    [{ modelId: "opaque", modelName: "Spark-X2.6-4B" }, "Spark X2.6", "Think (High)"],
    [{ modelId: "opaque", modelAlias: "gpt-oss-20b" }, "GPT-OSS", "Think (Medium)"],
    [{ modelId: "opaque", modelAliases: ["glm-5.3-flash"] }, "GLM-5.3-Flash", "Think (Max)"],
    [{ modelId: "opaque", modelPath: "C:\\models\\GLM-5.3-Flash.gguf" }, "GLM-5.3-Flash", "Think (Max)"],
    [{ modelId: "LFM2.5-2.6B", modelPath: "/models/opaque.gguf" }, "LFM2.5 2.6B", "Think"],
  ]) {
    it(`shows the remote ${family} preset without borrowing the local selection`, async () => {
      const t = setup({}, { state: "limited", isLlamaCpp: false, ...identity });
      await t.send({ type: "ready" });
      const payload = t.posted.find(m => m.type === "state").payload;
      assert.equal(payload.modeSampling.familyLabel, family);
      assert.equal(payload.modeSampling.defaultMode, defaultMode);
      assert.equal(payload.memoryView, null);
    });
  }

  it("keeps free sampling for an unknown remote model despite a recognized local model", async () => {
    const t = setup({}, { state: "limited", isLlamaCpp: false, modelId: "opaque" });
    await t.send({ type: "ready" });
    assert.equal(t.posted.find(m => m.type === "state").payload.modeSampling, null);
  });

  it("shows the loaded LFM3 recipe while Spark is pending reload", async () => {
    const t = setup();
    t.launched.modelPath = "/models/LFM3-1.7B.gguf";
    const state = t.getState();
    state.selectedModelPath = "/models/Spark-X2.6-1.7B.gguf";
    state.modelCapabilities = { ...caps, path: state.selectedModelPath, name: "Dense F32", architecture: "spark2_5" };
    await t.send({ type: "ready" });
    const mode = t.posted.find(m => m.type === "state").payload.modeSampling;
    assert.equal(mode.familyLabel, "LFM3 1.7B");
    assert.equal(mode.defaultMode, "Think (General)");
  });

  it("state push carries core's verdict for the saved settings", async () => {
    const t = setup();
    await t.send({ type: "ready" });
    const state = t.posted.find((m) => m.type === "state");
    const expected = core.computeMemoryView(caps, t.getState().loadSettings, gpus).view;
    assert.equal(state.payload.memoryView.level, expected.level);
    assert.equal(state.payload.memoryView.headline, expected.headline);
    assert.equal(state.payload.memory, undefined, "no second estimate object for the webview");
    assert.equal(state.payload.gpus[0].shortName, "RX 9070");
  });

  it("renders expert cache as a separate colored segment and legend, hiding it when off", async t => {
    const { JSDOM } = require("jsdom");
    const html = fs.readFileSync(path.join(__dirname, "../src/webview/sidebar.html"), "utf8").replace(/\{\{\w+\}\}/g, "");
    const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true, url: "https://preview.invalid" });
    t.after(() => dom.window.close());
    dom.window.acquireVsCodeApi = () => ({ postMessage() {}, getState: () => null, setState() {} });
    dom.window.eval(fs.readFileSync(path.join(__dirname, "../out/webview/sidebar.js"), "utf8"));
    const model = { ...caps, isMoe: true, expertCount: 128, expertUsedCount: 8, moeExpertShare: 0.9 };
    const ui = setup({ nCpuMoe: 48, moeCacheMiB: 4096 }, undefined,
      { ...core.NO_SERVER_FEATURES, probed: true, moeCache: true }, model);
    await ui.send({ type: "ready" });
    const payload = ui.posted.find(m => m.type === "state").payload;
    const paint = memoryView => dom.window.dispatchEvent(new dom.window.MessageEvent("message", {
      data: { type: "state", payload: { ...payload, memoryView } }
    }));
    paint(payload.memoryView);
    const doc = dom.window.document;
    assert.ok(doc.querySelector("#memDevices .seg-expertCache"));
    assert.match(doc.querySelector("#memDevices .seg-expertCache").title, /GPU expert cache/);
    assert.match(doc.getElementById("memLegend").textContent, /Expert cache 4/);
    const css = fs.readFileSync(path.join(__dirname, "../src/webview/sidebar.css"), "utf8");
    const style = doc.createElement("style"); style.textContent = css; doc.head.appendChild(style);
    assert.notEqual(dom.window.getComputedStyle(doc.querySelector(".mem-stack .seg-expertCache")).backgroundColor,
      dom.window.getComputedStyle(doc.querySelector(".mem-stack .seg-weights")).backgroundColor);
    paint(core.computeMemoryView(model, { ...ui.getState().loadSettings, moeCacheMiB: 0 }, gpus).view);
    assert.equal(doc.querySelector("#memDevices .seg-expertCache"), null);
    assert.doesNotMatch(doc.getElementById("memLegend").textContent, /Expert cache/);
  });

  for (const supported of [false, true]) {
    it(`accounts for expert caching only on a supporting binary (${supported}) and preserves saved choices`, async () => {
      const features = { probed: true, moeCache: supported, draftSampling: supported, draftUbatch: supported };
      const model = { ...caps, isMoe: true, expertCount: 128, expertUsedCount: 8, moeExpertShare: 0.9 };
      const t = setup({ nCpuMoe: 48, moeCacheMiB: 2048, draftSampling: "probabilistic", draftBatchSize: 64 },
        undefined, features, model);
      await t.send({ type: "ready" });
      const state = t.posted.find(m => m.type === "state").payload;
      assert.deepEqual(state.serverFeatures, features);
      const expected = core.computeMemoryView(model,
        { ...t.getState().loadSettings, moeCacheMiB: supported ? 2048 : 0 }, gpus, { withFixes: true }).view;
      assert.deepEqual(state.memoryView, expected);
      await t.send({ type: "estimate", seq: 15, payload: { moeCacheMiB: 4096 } });
      const live = t.posted.find(m => m.type === "memoryEstimate" && m.seq === 15);
      assert.deepEqual(live.view, core.computeMemoryView(model,
        { ...t.getState().loadSettings, moeCacheMiB: supported ? 4096 : 0 }, gpus, { withFixes: true }).view);
      assert.equal(t.getState().loadSettings.moeCacheMiB, 2048);
      assert.equal(t.getState().loadSettings.draftSampling, "probabilistic");
      assert.equal(t.getState().loadSettings.draftBatchSize, 64);
    });
  }

  it("resets the new controls with advanced settings", async () => {
    const t = setup({ moeCacheMiB: 4096, draftSampling: "greedy", draftBatchSize: 64,
      cacheRamMiB: 1024, mmprojDevice: "Vulkan1", draftDevice: "Vulkan1" });
    await t.send({ type: "resetAdvancedLoad" });
    assert.equal(t.getState().loadSettings.moeCacheMiB, 0);
    assert.equal(t.getState().loadSettings.draftSampling, "auto");
    assert.equal(t.getState().loadSettings.draftBatchSize, 0);
    assert.equal(t.getState().loadSettings.cacheRamMiB, undefined);
    assert.equal(t.getState().loadSettings.mmprojDevice, "");
    assert.equal(t.getState().loadSettings.draftDevice, "");
  });

  it("answers a live estimate for unsaved values with the same seq, without saving them", async () => {
    const t = setup();
    await t.send({ type: "ready" });
    await t.send({ type: "estimate", seq: 7, payload: { contextLength: 262144, cacheTypeK: "f16", cacheTypeV: "f16" } });
    const reply = t.posted.find((m) => m.type === "memoryEstimate");
    assert.equal(reply.seq, 7);
    const expected = core.computeMemoryView(
      caps,
      { ...t.getState().loadSettings, contextLength: 262144, cacheTypeK: "f16", cacheTypeV: "f16" },
      gpus,
      { withFixes: true }
    ).view;
    assert.equal(reply.view.level, expected.level);
    assert.deepEqual(reply.view.fixes.map((f) => f.id), expected.fixes.map((f) => f.id));
    assert.equal(t.getState().loadSettings.contextLength, 65536, "estimate must not persist");
  });

  it("fit preset searches with core fittingContextLength", async () => {
    const t = setup();
    await t.send({ type: "ready" });
    await t.send({ type: "applyLoadPatch", payload: { cacheTypeK: "q8_0", cacheTypeV: "q4_0" }, fitContext: true });
    const s = t.getState().loadSettings;
    assert.equal(s.cacheTypeV, "q4_0");
    const expected = core.fittingContextLength(caps, { ...s }, { gpus });
    assert.equal(s.contextLength, expected);
  });

  it("lists pending changes and discards them back to the launched settings", async () => {
    const t = setup();
    await t.send({ type: "ready" });
    await t.send({ type: "saveLoad", payload: { contextLength: 32768 }, silent: true });
    const patch = t.posted.filter((m) => m.type === "statusPatch").pop();
    assert.deepEqual(patch.payload.changes, [{ key: "contextLength", label: "context window", from: "64k", to: "32k" }]);
    await t.send({ type: "discardChanges" });
    assert.deepEqual(t.getState().loadSettings, core.normalizeLoadSettings(t.launched.loadSettings));
  });

  it("CPU backend: a context edit lists only the context, and discard keeps GPU preferences", async () => {
    const t = setup({ gpuOffload: 999, offloadKvCacheToGpu: true, mmprojOffloadToGpu: true }, undefined, core.NO_SERVER_FEATURES, caps, { cpu: true });
    await t.send({ type: "ready" });
    await t.send({ type: "saveLoad", payload: { contextLength: 16384 }, silent: true });
    const patch = t.posted.filter((m) => m.type === "statusPatch").pop();
    assert.deepEqual(patch.payload.changes.map((c) => c.key), ["contextLength"]);
    await t.send({ type: "discardChanges" });
    const s = t.getState().loadSettings;
    assert.equal(s.contextLength, 65536);
    assert.equal(s.gpuOffload, 999);
    assert.equal(s.offloadKvCacheToGpu, true);
    assert.equal(s.mmprojOffloadToGpu, true);
  });

  it("GPU backend still lists real GPU changes", async () => {
    const t = setup();
    await t.send({ type: "ready" });
    await t.send({ type: "saveLoad", payload: { gpuOffload: 10 }, silent: true });
    const patch = t.posted.filter((m) => m.type === "statusPatch").pop();
    assert.ok(patch.payload.changes.some((c) => c.key === "gpuOffload"));
  });

  it("Advanced tab: a silent save there refreshes the sidebar only; reopening reveals the same tab", async () => {
    const t = setup();
    await t.send({ type: "ready" });
    t.provider.openAdvancedPanel();
    const panel = lastPanel;
    assert.ok(panel.webview.html.includes("view-advanced"));
    assert.ok(t.posted.some((m) => m.type === "advancedPanel" && m.open === true));
    t.posted.length = 0;
    panel.posted.length = 0;
    await panel.handler({ type: "saveLoad", payload: { cpuThreads: 3 }, silent: true });
    assert.equal(t.getState().loadSettings.cpuThreads, 3);
    assert.ok(t.posted.some((m) => m.type === "state"), "sidebar gets the new values");
    assert.ok(!panel.posted.some((m) => m.type === "state"), "the tab being edited is not re-filled");
    t.provider.openAdvancedPanel();
    assert.equal(panel.revealed, 1);
    panel.dispose();
    assert.ok(t.posted.some((m) => m.type === "advancedPanel" && m.open === false));
  });

  it("partial saves merge instead of resetting other settings", async () => {
    const t = setup({ cacheTypeK: "q8_0" });
    await t.send({ type: "ready" });
    await t.send({ type: "saveLoad", payload: { contextLength: 16384 }, silent: true });
    assert.equal(t.getState().loadSettings.contextLength, 16384);
    assert.equal(t.getState().loadSettings.cacheTypeK, "q8_0");
  });

  it("spill confirmation uses the same verdict as the sidebar", async () => {
    const t = setup({ contextLength: 262144, cacheTypeK: "f16", cacheTypeV: "f16" });
    const ok = await t.provider.confirmIfMemorySpill();
    assert.equal(ok, false);
    const expected = core.computeMemoryView(caps, t.getState().loadSettings, gpus).view;
    assert.equal(shown[0].msg, expected.headline);
    assert.ok(shown[0].modal);
  });

  it("coalesces overlapping state pushes into one trailing push", async () => {
    const t = setup();
    await t.send({ type: "ready" });
    t.posted.length = 0;
    // Five callers while a push is running: the first push plus one trailing.
    await Promise.all([1, 2, 3, 4, 5].map(() => t.provider.pushState()));
    assert.equal(t.posted.filter((m) => m.type === "state").length, 2);
    // The trailing push sees state changed after the first push started.
    t.posted.length = 0;
    const first = t.provider.pushState();
    await t.send({ type: "saveLoad", payload: { contextLength: 16384 }, silent: true });
    const second = t.provider.pushState();
    await Promise.all([first, second]);
    const states = t.posted.filter((m) => m.type === "state");
    assert.equal(states.at(-1).payload.state.loadSettings.contextLength, 16384);
  });
});

describe("LoRA adapter controls", () => {
  const lora = { ...core.NO_SERVER_FEATURES, probed: true, lora: true };
  const adapters = [
    { path: "/l/sql-lora.gguf", scale: 0.8, enabled: true, bytes: 1024, fit: "match", message: "", activated: false, loaded: true },
    { path: "/l/new-lora.gguf", scale: 1, enabled: true, bytes: 2048, fit: "unknown-base", message: "The adapter does not name its base model.", activated: true, loaded: false },
  ];
  it("renders adapter rows and sends strength, on/off, remove and add", async t => {
    const { JSDOM } = require("jsdom");
    const html = fs.readFileSync(path.join(__dirname, "../src/webview/sidebar.html"), "utf8").replace(/\{\{\w+\}\}/g, "");
    const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true, url: "https://preview.invalid" });
    t.after(() => dom.window.close());
    const sent = [];
    dom.window.acquireVsCodeApi = () => ({ postMessage: m => sent.push(m), getState: () => null, setState() {} });
    dom.window.eval(fs.readFileSync(path.join(__dirname, "../out/webview/sidebar.js"), "utf8"));
    const ui = setup({}, undefined, lora);
    await ui.provider.pushState();
    const payload = ui.posted.filter(m => m.type === "state").at(-1).payload;
    const paint = patch => dom.window.dispatchEvent(new dom.window.MessageEvent("message", { data: { type: "state", payload: { ...payload, ...patch } } }));
    const el = id => dom.window.document.getElementById(id);
    const fire = (node, type) => node.dispatchEvent(new dom.window.Event(type, { bubbles: true }));
    paint({});
    assert.equal(el("loraAttach").classList.contains("hidden"), false);
    assert.equal(el("loraSum").textContent, "None");
    assert.equal(el("modelLineage").classList.contains("hidden"), true);
    el("pickLoraBtn").click();
    assert.equal(sent.at(-1).type, "pickLora");

    paint({ lora: { adapters, builtIn: null, lineage: null }, status: { ...payload.status, running: false } });
    assert.equal(el("loraSum").textContent, "2 of 2 on", "stopped server: nothing is active yet");
    paint({ lora: { adapters, builtIn: null, lineage: null }, status: { ...payload.status, running: true } });
    const rows = [...el("loraList").querySelectorAll(".lora-item")];
    assert.equal(rows.length, 2);
    assert.match(rows[0].textContent, /sql-lora\.gguf.*Active/);
    assert.match(rows[1].textContent, /Queued for reload/);
    assert.match(rows[1].textContent, /does not name its base model/);
    assert.match(rows[1].textContent, /trigger phrase/);
    assert.equal(el("loraSum").textContent, "1 active · 1 after reload");
    assert.match(el("loraHint").textContent, /Queued adapters take effect on reload/);

    const slider = rows[0].querySelector("input[type=range]");
    slider.value = "0.35";
    fire(slider, "input");
    assert.equal(rows[0].querySelector("output").textContent, "0.35");
    assert.notEqual(sent.at(-1).type, "setLora", "dragging does not post");
    fire(slider, "change");
    assert.deepEqual({ ...sent.at(-1) }, { type: "setLora", path: "/l/sql-lora.gguf", scale: 0.35 });
    const toggle = rows[0].querySelector("input[type=checkbox]");
    toggle.checked = false;
    fire(toggle, "change");
    assert.deepEqual({ ...sent.at(-1) }, { type: "setLora", path: "/l/sql-lora.gguf", enabled: false });
    slider.focus();
    paint({ lora: { adapters, builtIn: null, lineage: null }, status: { ...payload.status, running: true } });
    assert.equal(dom.window.document.activeElement, slider);
    assert.equal(slider.value, "0.8");
    assert.equal(rows[0].querySelector("output").textContent, "0.80");
    assert.equal(toggle.checked, true);
    rows[1].querySelector("[data-act=lora-remove]").click();
    assert.deepEqual({ ...sent.at(-1) }, { type: "removeLora", path: "/l/new-lora.gguf" });

    paint({ lora: { adapters: [], pendingRemovals: [{ path: adapters[0].path, scale: 0.8 }] } });
    assert.equal(el("loraSum").textContent, "1 active · 1 pending removal");
    assert.match(el("loraList").textContent, /sql-lora\.gguf.*Active at 0\.80 · Removed on reload/);
    assert.equal(el("loraList").querySelector("input,button"), null);
    paint({ lora: { adapters: [], pendingRemovals: [{ path: adapters[0].path, scale: 0 }] } });
    assert.equal(el("loraSum").textContent, "0 active · 1 pending removal");
    assert.match(el("loraList").textContent, /Off on server · Removed on reload/);

    paint({ lora: { adapters: [], builtIn: { count: 12, rank: 32 }, lineage: { finetune: "Storywriter", baseModelName: "Granite 8B" } } });
    assert.match(el("modelLineage").textContent, /12 built-in adapters · rank 32/);
    assert.match(el("modelLineage").textContent, /Based on Granite 8B \(Storywriter\).*cannot be detected/);
    paint({ serverFeatures: { ...core.NO_SERVER_FEATURES, probed: true } });
    assert.equal(el("pickLoraBtn").disabled, true);
    assert.match(el("loraHint").textContent, /no --lora option/);
    paint({ serverMode: "remote" });
    assert.equal(el("loraAttach").classList.contains("hidden"), true);
  });
  it("applies strength and on/off live, and lists only file changes for Reload", async () => {
    const saved = [{ path: "/l/a.gguf", scale: 1, enabled: true }];
    const ui = setup({ loraAdapters: saved }, undefined, lora);
    await ui.send({ type: "setLora", path: "/l/a.gguf", scale: 0.4 });
    await ui.send({ type: "setLora", path: "/l/a.gguf", enabled: false });
    assert.deepEqual(ui.getState().loadSettings.loraAdapters, [{ path: "/l/a.gguf", scale: 0.4, enabled: false }]);
    assert.deepEqual(ui.loraApplied.at(-1), [{ path: "/l/a.gguf", scale: 0.4, enabled: false }]);
    let payload = ui.posted.filter(m => m.type === "state").at(-1).payload;
    assert.equal(payload.changes.some(c => c.key === "loraAdapters"), false);
    assert.equal(payload.lora.adapters[0].loaded, true);
    await ui.send({ type: "removeLora", path: "/l/a.gguf" });
    payload = ui.posted.filter(m => m.type === "state").at(-1).payload;
    assert.deepEqual(payload.changes.filter(c => c.key === "loraAdapters"), [
      { key: "loraAdapters", label: "LoRA adapters", from: "a.gguf", to: "none" },
    ]);
    assert.deepEqual(payload.lora.pendingRemovals, [{ path: "/l/a.gguf", scale: 0 }]);
    await ui.send({ type: "discardChanges" });
    assert.deepEqual(ui.getState().loadSettings.loraAdapters, [{ path: "/l/a.gguf", scale: 0.4, enabled: false }]);
    assert.deepEqual(ui.loraApplied.at(-1), [], "Discard should not reapply launch-time strengths");
  });
  it("reports a removed adapter that still applies at its running strength", async () => {
    const saved = [{ path: "/l/a.gguf", scale: 0.45, enabled: true }];
    const ui = setup({ loraAdapters: saved }, undefined, lora);
    await ui.send({ type: "removeLora", path: "/l/a.gguf" });
    let state = ui.posted.findLast(m => m.type === "state").payload.lora;
    assert.deepEqual(state.adapters, []);
    assert.deepEqual(state.pendingRemovals, [{ path: "/l/a.gguf", scale: 0.45 }]);
    await ui.send({ type: "discardAttachments" });
    state = ui.posted.findLast(m => m.type === "state").payload.lora;
    assert.equal(state.adapters[0].scale, 0.45);
    assert.deepEqual(state.pendingRemovals, []);
  });
  it("discards attachment edits without reverting pending context or thread settings", async () => {
    const saved = [{ path: "/l/a.gguf", scale: 0.4, enabled: true }];
    const ui = setup({ loraAdapters: saved, mmprojPath: "/m/projector.gguf", plePath: "/m/ple.gguf" }, undefined, lora);
    await ui.send({ type: "removeLora", path: "/l/a.gguf" });
    await ui.provider.store.updateLoadSettings({ contextLength: 32768, mmprojPath: "", plePath: "" });
    await ui.send({ type: "discardAttachments", payload: { cpuThreads: 5 } });
    const settings = ui.getState().loadSettings;
    assert.deepEqual(settings.loraAdapters, saved);
    assert.deepEqual(ui.loraApplied.at(-1), saved);
    assert.equal(settings.mmprojPath, "/m/projector.gguf");
    assert.equal(settings.plePath, "/m/ple.gguf");
    assert.equal(settings.contextLength, 32768);
    assert.equal(settings.cpuThreads, 5);
  });

  it("undoes a change the running server did not take, and says so", async t => {
    const errors = t.mock.method(require("vscode").window, "showErrorMessage", async () => undefined);
    const saved = [{ path: "/l/a.gguf", scale: 1, enabled: true }];
    const ui = setup({ loraAdapters: saved }, undefined, lora, caps, { loraFails: true });
    await ui.send({ type: "setLora", path: "/l/a.gguf", enabled: false });
    assert.deepEqual(ui.getState().loadSettings.loraAdapters, saved, "the sidebar must not show an unapplied state");
    assert.match(errors.mock.calls.at(-1).arguments[0], /did not take the LoRA change, so it was undone: server said no/);
    const payload = ui.posted.filter(m => m.type === "state").at(-1).payload;
    assert.equal(payload.lora.adapters[0].enabled, true);
  });
  it("adds picked files once, and refuses unsupported builds and remote servers", async t => {
    const dialog = t.mock.method(require("vscode").window, "showOpenDialog", async () => [{ fsPath: "/l/a.gguf" }, { fsPath: "/l/b.gguf" }]);
    const ui = setup({ loraAdapters: [{ path: "/l/a.gguf", scale: 0.5, enabled: true }] }, undefined, lora);
    await ui.send({ type: "pickLora" });
    assert.deepEqual(ui.getState().loadSettings.loraAdapters, [
      { path: "/l/a.gguf", scale: 0.5, enabled: true }, { path: "/l/b.gguf", scale: 1, enabled: true },
    ]);
    const payload = ui.posted.filter(m => m.type === "state").at(-1).payload;
    assert.equal(payload.lora.adapters[1].loaded, false);
    assert.equal(payload.lora.adapters[1].fit, "invalid", "a missing file is reported on its row");
    const calls = dialog.mock.callCount();
    await assert.rejects(setup().provider.pickLora(), /no --lora option/);
    await assert.rejects(setup({}, { state: "ready" }).provider.pickLora(), /remote server/);
    assert.equal(dialog.mock.callCount(), calls);
  });
});

describe("more ports to look on", () => {
  it("saves typed ports from the sidebar link and looks again", async t => {
    const vs = require("vscode").window;
    let asked;
    t.mock.method(vs, "showInputBox", async (o) => { asked = o; return "9000, 8001 9000"; });
    const ui = setup();
    await ui.send({ type: "extraServerPorts" });
    assert.deepEqual(ui.provider.store.getConfig().get("extraServerPorts"), [9000, 8001]);
    assert.equal(asked.validateInput("8001, x"), '"x" is not a port number (1 to 65535).');
    assert.equal(asked.validateInput("8001"), undefined);
    t.mock.method(vs, "showInputBox", async () => undefined);
    await ui.send({ type: "extraServerPorts" });
    assert.deepEqual(ui.provider.store.getConfig().get("extraServerPorts"), [9000, 8001], "cancel keeps the list");
    t.mock.method(vs, "showInputBox", async () => "");
    await ui.send({ type: "extraServerPorts" });
    assert.deepEqual(ui.provider.store.getConfig().get("extraServerPorts"), []);
  });
});

it("falls back to labeled total RAM when free GPU memory is unavailable", async t => {
  const ui = setup();
  t.mock.method(os, "totalmem", () => 16 * 1024 ** 3);
  t.mock.method(os, "freemem", () => 0);
  ui.provider.hf = () => ({
    listGgufFiles: async () => [{ path: "model-Q4_K_M.gguf", size: 2 * 1024 ** 3, url: "" }],
  });
  await ui.send({ type: "listLibraryHfFiles", modelId: "owner/model", requestId: 1 });
  const reply = ui.posted.findLast(m => m.type === "libraryHfFiles");
  assert.equal(reply.usesRam, true);
  assert.match(reply.capacityLabel, /total system RAM/);
  assert.equal(reply.files[0].fit, "fits");
});

it("preserves a connection draft and shows explicit scan results while the local server runs", async t => {
  const { JSDOM } = require("jsdom");
  const html = fs.readFileSync(path.join(__dirname, "../src/webview/sidebar.html"), "utf8").replace(/\{\{\w+\}\}/g, "");
  const dom = new JSDOM(html, { runScripts: "outside-only" });
  t.after(() => dom.window.close());
  const sent = [];
  dom.window.acquireVsCodeApi = () => ({ postMessage: m => sent.push(m), getState: () => null, setState() {} });
  dom.window.eval(fs.readFileSync(path.join(__dirname, "../out/webview/sidebar.js"), "utf8"));
  const el = id => dom.window.document.getElementById(id);
  const post = data => dom.window.dispatchEvent(new dom.window.MessageEvent("message", { data }));
  const ui = setup();
  await ui.provider.pushState();
  post(ui.posted.findLast(m => m.type === "state"));
  post({ type: "openConnect" });
  el("remoteUrlInput").value = "http://127.0.0.1:9123";
  el("remoteKeyInput").value = "temporary-test-key";
  el("remoteKeyInput").focus();
  post({ type: "openConnect" });
  assert.equal(el("remoteUrlInput").value, "http://127.0.0.1:9123");
  assert.equal(el("remoteKeyInput").value, "temporary-test-key");
  el("extraPortsBtn").click();
  const requestId = sent.at(-1).requestId;
  assert.equal(sent.at(-1).type, "extraServerPorts");
  post({ type: "connectScan", requestId, servers: [{ url: "http://127.0.0.1:9124" }] });
  assert.equal(el("foundServers").classList.contains("hidden"), false);
  el("foundServersList").querySelector("button").click();
  assert.equal(el("remoteUrlInput").value, "http://127.0.0.1:9124");
  assert.equal(el("remoteKeyInput").value, "");

  el("connectCancelBtn").click();
  post({
    type: "statusPatch",
    payload: { starting: true, changes: [{ key: "model", from: "first.gguf", to: "second.gguf" }] },
  });
  assert.equal(el("modelReloadBar").classList.contains("hidden"), false);
  assert.match(el("modelReloadText").textContent, /loading first\.gguf/);
  assert.equal(el("modelReloadBtn").disabled, true);
});

it("keeps draft API keys bound to their origin during preview and connect", async t => {
  const { JSDOM } = require("jsdom");
  const html = fs.readFileSync(path.join(__dirname, "../src/webview/sidebar.html"), "utf8").replace(/\{\{\w+\}\}/g, "");
  const dom = new JSDOM(html, { runScripts: "outside-only" });
  t.after(() => dom.window.close());
  const sent = [];
  dom.window.acquireVsCodeApi = () => ({ postMessage: m => sent.push(m), getState: () => null, setState() {} });
  const setTimeout = dom.window.setTimeout.bind(dom.window);
  let scheduledProbe;
  t.mock.method(dom.window, "setTimeout", (callback, delay, ...args) => {
    if (delay === 600) {
      scheduledProbe = callback;
      return 0;
    }
    return setTimeout(callback, delay, ...args);
  });
  dom.window.eval(fs.readFileSync(path.join(__dirname, "../out/webview/sidebar.js"), "utf8"));
  const el = id => dom.window.document.getElementById(id);
  const post = data => dom.window.dispatchEvent(new dom.window.MessageEvent("message", { data }));
  const input = (id, value) => {
    const field = el(id);
    field.value = value;
    field.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
  };
  post({ type: "openConnect" });
  const url = "remoteUrlInput";
  const key = "remoteKeyInput";
  assert.equal(el(key).disabled, true);

  input(url, "LOCALHOST:18080/path");
  input(key, "disposable-fixture-key");
  scheduledProbe();
  assert.equal(sent.at(-1).apiKey, "disposable-fixture-key");

  input(url, "http://localhost:18080/other-path");
  scheduledProbe();
  assert.equal(sent.at(-1).apiKey, "disposable-fixture-key");

  input(url, "http://localhost:18087");
  assert.equal(el(key).value, "");
  assert.match(el("remoteKeyHint").textContent, /Server address changed/);
  scheduledProbe();
  assert.equal(sent.at(-1).apiKey, "");
  el("connectBtn").click();
  assert.equal(sent.at(-1).type, "connectRemote");
  assert.equal(sent.at(-1).apiKey, "");

  input(key, "new-origin-key");
  input(url, "http://[");
  assert.equal(el(key).disabled, true);
  assert.equal(el(key).value, "");

  input(url, "https://localhost:18087");
  input(key, "https-key");
  el("connectBtn").click();
  assert.equal(sent.at(-1).apiKey, "https-key");
});

describe("local server scan and the port list", () => {
  function held(ui) {
    const calls = [];
    ui.provider.discoverLocalServers = (_skip, ports) => {
      let resolve;
      const promise = new Promise((r) => { resolve = r; });
      calls.push({ ports: [...ports], resolve });
      return promise;
    };
    return calls;
  }
  const stopped = { running: false, starting: false };
  const srv = (port) => [{ url: `http://127.0.0.1:${port}`, port, name: "OpenAI-compatible server" }];

  for (const order of ["old scan finishes first", "new scan finishes first"]) {
    it(`a scan started before the ports changed never fills the cache (${order})`, async () => {
      const ui = setup();
      const calls = held(ui);
      const config = ui.provider.store.getConfig();
      const first = ui.provider.localServersForSidebar(stopped);
      await config.update("extraServerPorts", [9123]);
      const second = ui.provider.localServersForSidebar(stopped);
      assert.deepEqual(calls.map((c) => c.ports), [[], [9123]], "the new list is looked at at once");
      if (order === "old scan finishes first") {
        calls[0].resolve(srv(1111)); calls[1].resolve(srv(9123));
      } else {
        calls[1].resolve(srv(9123)); await second; calls[0].resolve(srv(1111));
      }
      await Promise.all([first, second]);
      assert.deepEqual(await ui.provider.localServersForSidebar(stopped), srv(9123));
      assert.equal(calls.length, 2, "the cached result for the current list is reused");
    });
  }

  it("allows explicit discovery while in remote mode without scanning on status polls", async () => {
    const ui = setup({}, { state: "ready" });
    await ui.provider.store.getConfig().update("extraServerPorts", [9123]);
    const calls = held(ui);
    assert.deepEqual(await ui.provider.localServersForSidebar(stopped), []);
    assert.equal(calls.length, 0);
    const explicit = ui.provider.localServersForSidebar(stopped, false, true);
    assert.deepEqual(calls.map(c => c.ports), [[9123]]);
    calls[0].resolve(srv(9123));
    assert.deepEqual(await explicit, srv(9123));
  });

  it("removing every added port drops what they found", async () => {
    const ui = setup();
    const calls = held(ui);
    const config = ui.provider.store.getConfig();
    await config.update("extraServerPorts", [9123]);
    const first = ui.provider.localServersForSidebar(stopped);
    calls[0].resolve(srv(9123));
    await first;
    await config.update("extraServerPorts", []);
    const after = ui.provider.localServersForSidebar(stopped);
    assert.deepEqual(calls.map((c) => c.ports), [[9123], []]);
    calls[1].resolve([]);
    assert.deepEqual(await after, []);
  });
});

describe("attachment pickers search the library", () => {
  const lora = { ...core.NO_SERVER_FEATURES, probed: true, lora: true };
  function library(t) {
    const dir = fs.mkdtempSync(path.join(require("os").tmpdir(), "llama-aio-pick-"));
    core.invalidateModelLibraryCache();
    t.after(() => { core.invalidateModelLibraryCache(); fs.rmSync(dir, { recursive: true, force: true }); });
    // Minimal GGUF v3 header with string metadata only.
    const gguf = kv => {
      const u32 = n => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; };
      const u64 = n => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; };
      const text = v => Buffer.concat([u64(Buffer.byteLength(v)), Buffer.from(v)]);
      const entries = Object.entries(kv);
      return Buffer.concat([Buffer.from("GGUF"), u32(3), u64(0), u64(entries.length),
        ...entries.flatMap(([k, v]) => [text(k), u32(8), text(v)])]);
    };
    const adapter = (name, arch) => {
      const file = path.join(dir, name);
      fs.writeFileSync(file, gguf({ "general.architecture": arch, "general.type": "adapter", "adapter.type": "lora" }));
      return file;
    };
    return { dir, adapter };
  }
  const mine = (items, dir) => items.filter(i => i.candidate && i.candidate.path.startsWith(dir));

  it("offers adapters found in the model folders and adds the chosen one without a file dialog", async t => {
    const { dir, adapter } = library(t);
    const good = adapter("sql-lora.gguf", caps.architecture);
    const wrong = adapter("other-family-lora.gguf", "some-other-arch");
    const dialog = t.mock.method(require("vscode").window, "showOpenDialog");
    let offered;
    t.mock.method(require("vscode").window, "showQuickPick", async (items, options) => {
      offered = { items, options };
      return items.find(i => i.candidate && i.candidate.path === good);
    });
    const ui = setup({}, undefined, lora, caps, { modelsDir: dir });
    await ui.send({ type: "pickLora" });
    assert.deepEqual(ui.getState().loadSettings.loraAdapters, [{ path: good, scale: 1, enabled: true }]);
    assert.equal(dialog.mock.callCount(), 0);
    const rows = mine(offered.items, dir);
    assert.deepEqual(rows.map(i => [i.label, i.candidate.usable]), [["sql-lora.gguf", true], ["other-family-lora.gguf", false]]);
    assert.ok(offered.items.indexOf(rows[1]) > offered.items.findIndex(i => i.label === "Cannot be used with this model"));
    assert.equal(offered.items.at(-1).browse, true, "Browse stays available");
    assert.match(offered.options.placeHolder, /found in your model folders/);
    assert.ok(wrong);

    // Opening the picker again marks the adapter as already added; choosing it changes nothing.
    t.mock.method(require("vscode").window, "showQuickPick", async items => {
      offered = { items };
      return items.find(i => i.candidate && i.candidate.path === good);
    });
    await ui.send({ type: "pickLora" });
    assert.match(mine(offered.items, dir)[0].label, /^\$\(check\) sql-lora\.gguf$/);
    assert.match(mine(offered.items, dir)[0].description, /Already added/);
    assert.equal(ui.getState().loadSettings.loraAdapters.length, 1);
  });

  it("explains instead of adding when an unusable file is chosen, and does nothing when dismissed", async t => {
    const { dir, adapter } = library(t);
    const wrong = adapter("other-family-lora.gguf", "some-other-arch");
    const ui = setup({}, undefined, lora, caps, { modelsDir: dir });
    t.mock.method(require("vscode").window, "showQuickPick", async items => items.find(i => i.candidate && i.candidate.path === wrong));
    const before = shown.length;
    await ui.send({ type: "pickLora" });
    assert.deepEqual(ui.getState().loadSettings.loraAdapters, []);
    assert.match(shown.slice(before).at(-1).msg, /other-family-lora\.gguf cannot be used with this model.*some-other-arch/);
    t.mock.method(require("vscode").window, "showQuickPick", async () => undefined);
    const dialog = t.mock.method(require("vscode").window, "showOpenDialog");
    await ui.send({ type: "pickLora" });
    assert.equal(dialog.mock.callCount(), 0);
    assert.deepEqual(ui.getState().loadSettings.loraAdapters, []);
  });

  it("says when nothing was found and still offers Browse, for LoRA and PLE", async t => {
    const { dir } = library(t);
    const prompts = [];
    t.mock.method(require("vscode").window, "showQuickPick", async (items, options) => {
      prompts.push({ options, mine: mine(items, dir), browse: items.some(i => i.browse) });
      return undefined;
    });
    await setup({}, undefined, lora, caps, { modelsDir: dir }).provider.pickLora();
    await setup({}, undefined, { ...core.NO_SERVER_FEATURES, externalPle: "--ple" }, caps, { modelsDir: dir }).provider.pickPle();
    assert.equal(prompts.length, 2);
    for (const p of prompts) {
      assert.deepEqual(p.mine, []);
      assert.equal(p.browse, true);
    }
  });
});

describe("onboarding test message", () => {
  const vscode = require("vscode");

  function mockTestModel(t, select) {
    t.mock.property(vscode, "lm", { selectChatModels: select });
    t.mock.property(vscode, "LanguageModelChatMessage", { User: text => text });
    t.mock.property(vscode, "LanguageModelTextPart", class { constructor(value) { this.value = value; } });
    t.mock.property(vscode, "CancellationTokenSource", class {
      listeners = new Set();
      token = {
        isCancellationRequested: false,
        onCancellationRequested: listener => {
          this.listeners.add(listener);
          return { dispose: () => this.listeners.delete(listener) };
        },
      };
      cancel() {
        this.token.isCancellationRequested = true;
        for (const listener of this.listeners) listener();
      }
      dispose() { this.listeners.clear(); }
    });
  }

  function captureResults() {
    const results = [];
    const provider = Object.create(SettingsViewProvider.prototype);
    provider.post = result => results.push(result);
    provider.pushState = async () => {};
    return { provider, results };
  }

  it("offers retry when the stream returns no displayable text", async t => {
    mockTestModel(t, async () => [{
      sendRequest: async () => ({ stream: [new vscode.LanguageModelTextPart("   ")] }),
    }]);
    const { provider, results } = captureResults();
    await provider.runTestMessage("Hello");
    assert.deepEqual(results.map(r => r.status), ["running", "error"]);
    assert.match(results.at(-1).error, /returned no text/);
  });

  for (const action of ["cancel", "timeout"]) {
    it(`can retry after ${action} while an older model lookup is still pending`, async t => {
      let resolveOld;
      const old = new Promise(resolve => { resolveOld = resolve; });
      const sendRequest = t.mock.fn(async () => ({
        stream: [new vscode.LanguageModelTextPart("New reply")],
      }));
      const model = { sendRequest };
      let lookups = 0;
      mockTestModel(t, async () => lookups++ === 0 ? old : [model]);
      const { provider, results } = captureResults();
      if (action === "timeout") t.mock.timers.enable({ apis: ["setTimeout"] });

      const first = provider.runTestMessage("First");
      if (action === "timeout") t.mock.timers.tick(180_000);
      else provider.testCancel.cancel();
      await first;
      assert.equal(results.at(-1).status, action === "timeout" ? "error" : "cancelled");
      if (action === "timeout") assert.match(results.at(-1).error, /within 3 minutes/);

      await provider.runTestMessage("Second");
      assert.equal(results.at(-1).text, "New reply");
      resolveOld([model]);
      await old;
      assert.equal(sendRequest.mock.callCount(), 1, "cancelled lookup must not send the old prompt");
    });
  }

  it("waits for the old stream to end before starting a retry", async t => {
    let finishStream;
    const oldStream = new Promise(resolve => { finishStream = resolve; });
    const sendRequest = t.mock.fn(async ([prompt]) => ({
      stream: prompt === "First" ? (async function* () {
        await oldStream;
        yield new vscode.LanguageModelTextPart("Old reply");
      })() : [new vscode.LanguageModelTextPart("New reply")],
    }));
    mockTestModel(t, async () => [{ sendRequest }]);
    const { provider, results } = captureResults();
    const first = provider.runTestMessage("First");
    await new Promise(resolve => setImmediate(resolve));
    provider.testCancel.cancel();
    await first;
    const retry = provider.runTestMessage("Retry");
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(sendRequest.mock.callCount(), 1);
    finishStream();
    await retry;
    assert.equal(sendRequest.mock.callCount(), 2);
    assert.equal(results.at(-1).text, "New reply");
  });

  it("does not bring onboarding back when a dismissed test finishes", async t => {
    let select;
    const lookup = new Promise(resolve => { select = resolve; });
    mockTestModel(t, () => lookup);
    const { provider, results } = captureResults();
    const test = provider.runTestMessage("First");
    await provider.handleMessage({ type: "dismissSetup" });
    select([]);
    await test;
    assert.deepEqual(results.map(r => r.status), ["running"]);
    assert.equal(provider.lastTestResult, undefined);
  });

  it("shows the reply without an editable prompt and can send another test after onboarding completes", async t => {
    const { JSDOM } = require("jsdom");
    const html = fs.readFileSync(path.join(__dirname, "../src/webview/sidebar.html"), "utf8").replace(/\{\{\w+\}\}/g, "");
    const dom = new JSDOM(html, { runScripts: "outside-only" });
    t.after(() => dom.window.close());
    const sent = [];
    dom.window.acquireVsCodeApi = () => ({ postMessage: m => sent.push(m), getState: () => null, setState() {} });
    dom.window.eval(fs.readFileSync(path.join(__dirname, "../out/webview/sidebar.js"), "utf8"));
    const el = id => dom.window.document.getElementById(id);
    const post = data => dom.window.dispatchEvent(new dom.window.MessageEvent("message", { data }));
    const result = (status, extra) => post({ type: "testResult", status, ...extra });
    const ui = setup();
    await ui.provider.pushState();
    const state = ui.posted.findLast(m => m.type === "state");
    post(state);

    const primary = el("setupPrimaryBtn");
    const input = el("setupTestInput");
    assert.equal(primary.dataset.action, "test");
    primary.click();
    assert.equal(primary.dataset.action, "cancelTest");
    assert.equal(input.disabled, true);
    post({ ...state, payload: { ...state.payload, onboardingDone: true } });
    result("done", { text: "Hello!" });
    assert.equal(el("setupTestPrompt").classList.contains("hidden"), true);
    assert.match(el("setupTestReply").textContent, /Hello!/);
    assert.equal(primary.dataset.action, "openChat");

    el("setupSecondary").querySelector('[data-setup="composeTest"]').click();
    assert.equal(el("setupTestPrompt").classList.contains("hidden"), false);
    assert.equal(dom.window.document.activeElement, input);
    input.value = "   ";
    input.dispatchEvent(new dom.window.Event("input"));
    assert.equal(primary.disabled, true);
    input.value = "Try again";
    input.dispatchEvent(new dom.window.Event("input"));
    primary.click();
    assert.equal(sent.at(-1).text, "Try again");
    result("error", { error: "No reply" });
    assert.equal(primary.textContent, "Try again");
    assert.equal(input.disabled, false);
    primary.click();
    result("cancelled");
    assert.equal(primary.dataset.action, "test");
    assert.equal(input.disabled, false);
    assert.match(el("setupTestReply").textContent, /Test cancelled/);
    result("done", { text: "Second reply" });
    el("setupSecondary").querySelector('[data-setup="dismiss"]').click();
    result("done", { text: "Result sent before the host handled dismiss" });
    assert.equal(el("setupBox").classList.contains("hidden"), true);
  });
});

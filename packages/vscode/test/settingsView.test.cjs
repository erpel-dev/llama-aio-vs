// Message-level tests for the sidebar provider (estimate, fit preset, discard, spill
// confirmation). Runs the compiled provider with a stub `vscode` module.
//   npm test -w llama-aio-vs
const assert = require("node:assert/strict");
const { describe, it, beforeEach } = require("node:test");
const Module = require("module");
const path = require("path");

const shown = [];
let copilotChatInstalled = true;
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  return request === "vscode" ? "vscode" : origResolve.call(this, request, ...rest);
};
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
      withProgress: async (_o, fn) => fn({ report() {} }),
    },
    workspace: {},
    commands: { executeCommand: async () => undefined },
    env: {},
    extensions: { getExtension: (id) => (copilotChatInstalled && id === "GitHub.copilot-chat" ? {} : undefined) },
    Uri: { joinPath: (base, ...parts) => ({ fsPath: path.join(base.fsPath, ...parts) }) },
    ProgressLocation: {},
    ViewColumn: {},
    EventEmitter: class {},
  },
};

const core = require("@llama-aio/core");
const { SettingsViewProvider } = require(path.join(__dirname, "..", "out-harness", "settingsView.js"));

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

function setup(loadOverrides = {}) {
  let state = {
    selectedModelPath: caps.path,
    modelCapabilities: caps,
    loadSettings: { ...core.DEFAULT_LOAD_SETTINGS, contextLength: 65536, splitMode: "none", ...loadOverrides },
    requestSettings: { ...core.DEFAULT_REQUEST_SETTINGS },
  };
  const config = new Map([["launchMode", "externalTerminal"]]);
  const launched = {
    modelPath: caps.path,
    loadSettings: JSON.parse(JSON.stringify(state.loadSettings)),
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
    isRemote: () => false,
    getServerPreferences: () => ({
      mode: "local",
      remoteUrl: "",
      remoteLabel: "",
      lastRemoteModelId: "",
      plainHttpConfirmed: [],
    }),
    getServerTarget: () => ({ mode: "local", baseUrl: "http://127.0.0.1:8742", label: "this machine" }),
  };
  const pm = {
    getStatus: () => ({ running: true, starting: false, pid: 1, endpoint: "http://127.0.0.1:8742", message: "Running", configDirty: JSON.stringify(state.loadSettings) !== JSON.stringify(launched.loadSettings) }),
    refreshStatus: async () => pm.getStatus(),
    isHttpReady: async () => true,
    resolveBinary: () => "",
    isCpuBackend: () => false,
    getLaunchedConfig: () => launched,
    claimLaunch: () => ({}),
    releaseLaunch() {},
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
  const perf = { get: () => ({}), detailLines: () => [], hasLastRequestContext: () => false, hasLastResponseTrace: () => false, setSpeculativeMode() {} };
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
  return { send: (m) => handler(m), posted, getState: () => state, launched, provider };
}

describe("SettingsViewProvider memory path", () => {
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
    assert.deepEqual(t.getState().loadSettings, t.launched.loadSettings);
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

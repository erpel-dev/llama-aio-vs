// Request-level tests for the Copilot chat provider against a fake llama-server:
// message conversion, system-hint placement, quiet cancellation, empty replies.
//   npm test -w llama-aio-vs
const assert = require("node:assert/strict");
const http = require("node:http");
const { after, before, beforeEach, describe, it } = require("node:test");
const Module = require("module");
const path = require("path");

class TextPart {
  constructor(value) {
    this.value = value;
  }
}
class ThinkingPart {
  constructor(value) { this.value = value; }
}
class ToolCallPart {
  constructor(callId, name, input) {
    Object.assign(this, { callId, name, input });
  }
}
class ToolResultPart {
  constructor(callId, content) {
    Object.assign(this, { callId, content });
  }
}
class LanguageModelError extends Error {}
const Role = { User: 1, Assistant: 2 };

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  return request === "vscode" ? "vscode" : origResolve.call(this, request, ...rest);
};
require.cache.vscode = {
  id: "vscode",
  filename: "vscode",
  loaded: true,
  exports: {
    LanguageModelTextPart: TextPart,
    LanguageModelThinkingPart: ThinkingPart,
    LanguageModelToolCallPart: ToolCallPart,
    LanguageModelToolResultPart: ToolResultPart,
    LanguageModelChatMessageRole: Role,
    LanguageModelChatToolMode: { Auto: 1, Required: 2 },
    LanguageModelError,
    ThemeIcon: class {},
    EventEmitter: class {
      event() {}
      fire() {}
    },
    window: { showWarningMessage: async () => undefined },
  },
};

const core = require("@llama-aio/core");
const catalogs = require("../../core/test/fixtures/remote-model-catalogs.json");
const { LlamaAioChatProvider } = require(path.join(__dirname, "..", "out-harness", "chatProvider.js"));

let server;
let endpoint = "";
let lastBody;
let servedModelId;
let servedProps;
let servedMetadata;
let propsStatus;
let servedNative;
let expectedApiKey;
let seenRequests = [];
/** How /v1/chat/completions answers: array of SSE data objects, optional per-event delay. */
let reply = { events: [], delayMs: 0 };

before(async () => {
  server = http.createServer((req, res) => {
    seenRequests.push({ url: req.url, authorization: req.headers.authorization });
    if (expectedApiKey && req.headers.authorization !== `Bearer ${expectedApiKey}`) {
      res.writeHead(401); res.end("unauthorized"); return;
    }
    req.url = req.url.replace(/^\/llama-proxy(?=\/)/, "");
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", async () => {
      if (req.url === "/health") {
        res.writeHead(200);
        res.end("{}");
        return;
      }
      if (req.url === "/props") {
        res.writeHead(propsStatus || 200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ default_generation_settings: { n_ctx: 32768 }, total_slots: 1, ...servedProps }));
        return;
      }
      if (req.url === "/v1/models") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ data: [{ id: servedModelId || "test-model", ...servedMetadata }] }));
        return;
      }
      if (req.url === "/api/native") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(servedNative || {}));
        return;
      }
      if (req.method === "GET") {
        res.writeHead(404);
        res.end("missing");
        return;
      }
      lastBody = JSON.parse(body);
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      for (const ev of reply.events) {
        if (res.destroyed) return;
        res.write(`data: ${JSON.stringify(ev)}\n\n`);
        if (reply.delayMs) await new Promise((r) => setTimeout(r, reply.delayMs));
      }
      res.end("data: [DONE]\n\n");
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  endpoint = `http://127.0.0.1:${server.address().port}`;
});
after(() => new Promise((resolve) => server.close(() => resolve())));

function setup({ guard = false, remote = false, remoteInfo = {} } = {}) {
  const perfCalls = [];
  const state = {
    selectedModelPath: "/m/test.gguf",
    loadSettings: { ...core.DEFAULT_LOAD_SETTINGS },
    requestSettings: { ...core.DEFAULT_REQUEST_SETTINGS, overthinkingPenalty: false },
  };
  const store = {
    getEndpoint: () => endpoint,
    isRemote: () => remote,
    getState: () => state,
    getServerTarget: () => ({ mode: remote ? "remote" : "local", baseUrl: endpoint, label: "this machine" }),
    getPromptReplacementsFile: () => "",
    isPromptReplacementsEnabled: () => false,
    isWikipediaLookupEnabled: () => false,
    isDuplicateToolCallGuardEnabled: () => guard,
    getSlotContextSize: () => 32768,
    getServerPreferences: () => ({}),
    setLastRemoteModelId: async () => undefined,
  };
  const pm = {
    isHttpReady: async () => true,
    getStatus: () => ({ starting: false, configDirty: false }),
    remote: {
      current: () => remote ? { state: "ready", ...remoteInfo } : undefined,
      refresh: async () => ({ state: "ready", ...remoteInfo }),
    },
  };
  const perf = new Proxy({}, { get: (_t, name) => (...args) => perfCalls.push([name, args]) });
  const provider = new LlamaAioChatProvider(store, pm, perf, path.join(__dirname, ".."));
  return { provider, perfCalls, state };
}

function cancelToken() {
  const listeners = [];
  const token = {
    isCancellationRequested: false,
    onCancellationRequested: (fn) => {
      listeners.push(fn);
      return { dispose() {} };
    },
  };
  return {
    token,
    cancel() {
      token.isCancellationRequested = true;
      listeners.forEach((fn) => fn());
    },
  };
}

const msg = (role, ...content) => ({ role, content, name: undefined });
const delta = (d) => ({ choices: [{ delta: d }] });

describe("LlamaAioChatProvider requests", () => {
  beforeEach(() => {
    reply = { events: [delta({ content: "ok" })], delayMs: 0 };
    lastBody = undefined;
    servedModelId = undefined;
    servedProps = undefined;
    servedMetadata = undefined;
    propsStatus = undefined;
    servedNative = undefined;
    expectedApiKey = undefined;
    seenRequests = [];
  });

  for (const row of catalogs.rows) {
    for (const source of row.provider === "strata" ? ["desktop", "standalone"] : ["standalone"]) {
      it(`selects the remote ${row.provider}/${row.installId} ${source} profile and preserves its served ID`, async () => {
        servedModelId = source === "desktop" ? `${row.installId}--2` : row.servedId;
        servedMetadata = source === "desktop" ? { owned_by: "strata", meta: { label: row.label } } : {};
        propsStatus = source === "desktop" ? 404 : 200;
        servedProps = source === "standalone" ? { model_alias: row.servedId,
          model_path: "C:\\models\\pack\\full", default_generation_settings: { n_ctx: 262144 } } : undefined;
        const { provider, state } = setup({ remote: true });
        state.modelCapabilities = { architecture: "gemma4", path: "/local/gemma-4-12B-it.gguf" };
        const [model] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
        const profile = provider.modeSetFor(model.id, source === "standalone" ? servedProps : undefined);
        assert.equal(profile.familyLabel.replace(/ 1017$/, ""), row.profile);
        const efforts = model.configurationSchema.properties.reasoningEffort.enum;
        for (const effort of efforts) {
          await provider.provideLanguageModelChatResponse(model, [msg(Role.User, new TextPart("hello"))],
            { modelConfiguration: { reasoningEffort: effort } }, { report() {} }, cancelToken().token);
          assert.equal(lastBody.model, servedModelId);
          if (row.family === "deepseek" || row.family === "deepseek41") {
            assert.equal(lastBody.top_p, 1);
            assert.equal(lastBody.chat_template_kwargs.enable_thinking, effort !== "none");
          }
        }
      });
    }
  }

  for (const metadata of [
    { owned_by: "project-maya", meta: { label: "GLM-5.3-Flash" } },
    { alias_of: "glm-5.3-flash" },
    { aliases: ["glm-5.3-flash"] },
    { owned_by: "strata", meta: { aliases: ["glm-5.3-flash"] } },
  ]) {
    it(`recovers an opaque remote alias from ${JSON.stringify(metadata)}`, async () => {
      servedModelId = "private-model";
      servedMetadata = metadata;
      propsStatus = 404;
      const { provider } = setup({ remote: true });
      const [model] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
      assert.deepEqual(model.configurationSchema.properties.reasoningEffort.enum, ["max", "high", "low"]);
      await provider.provideLanguageModelChatResponse(model, [msg(Role.User, new TextPart("hello"))],
        { modelConfiguration: { reasoningEffort: "low" } }, { report() {} }, cancelToken().token);
      assert.equal(lastBody.model, "private-model");
      assert.equal(lastBody.chat_template_kwargs.reasoning_effort, "low");
    });
  }

  for (const row of catalogs.legacyRows) {
    it(`uses the family label of legacy Strata entry ${row.id}`, async () => {
      servedModelId = row.id;
      servedMetadata = { owned_by: "strata", meta: { label: row.label } };
      propsStatus = 404;
      const { provider } = setup({ remote: true });
      const [model] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
      assert.equal(provider.modeSetFor(model.id, undefined).familyLabel, row.profile);
      await provider.provideLanguageModelChatResponse(model, [msg(Role.User, new TextPart("hello"))],
        {}, { report() {} }, cancelToken().token);
      assert.equal(lastBody.model, row.id);
      if (row.id.startsWith("qwen2.5")) {
        assert.deepEqual([lastBody.temperature, lastBody.top_p, lastBody.top_k, lastBody.repeat_penalty], [0.7, 0.8, 20, 1.1]);
        assert.deepEqual(model.configurationSchema.properties.reasoningEffort.enum, ["none"]);
      }
    });
  }

  it("clears earlier GLM reasoning on the client while retaining the current tool turn", async () => {
    servedModelId = "glm-5.3-flash";
    propsStatus = 404;
    const { provider } = setup({ remote: true });
    const [model] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    await provider.provideLanguageModelChatResponse(model, [
      msg(Role.User, new TextPart("old question")),
      msg(Role.Assistant, new ThinkingPart("Old private reasoning"), new TextPart("<think>Legacy reasoning</think>Answer")),
      msg(Role.User, new TextPart("new question")),
      msg(Role.Assistant, new ThinkingPart("Current reasoning"), new ToolCallPart("call_1", "read_file", { path: "README.md" })),
      msg(Role.User, new ToolResultPart("call_1", [new TextPart("file text")])),
    ], {}, { report() {} }, cancelToken().token);
    const assistants = lastBody.messages.filter(message => message.role === "assistant");
    assert.equal(assistants[0].reasoning_content, undefined);
    assert.equal(assistants[0].content, "Answer");
    assert.equal(assistants[1].reasoning_content, "Current reasoning");
    assert.equal(assistants[1].tool_calls[0].id, "call_1");
  });

  it("refreshes reused remote IDs from new aliases and does not resurrect an old profile after ambiguity", async () => {
    servedModelId = "private-model";
    servedMetadata = { owned_by: "strata", meta: { label: "Qwen3.8-Flash-Next" } };
    propsStatus = 404;
    const { provider } = setup({ remote: true });
    const [first] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    assert.equal(first.configurationSchema.properties.reasoningEffort.default, "xhigh");
    servedMetadata = { meta: { aliases: ["glm-5.3-flash"] } };
    provider.notifyChanged();
    const [changed] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    assert.equal(changed.configurationSchema.properties.reasoningEffort.default, "max");
    servedMetadata = { meta: { aliases: ["glm-5.3-flash", "qwen3.8-flash-next"] } };
    provider.notifyChanged();
    const [ambiguous] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    assert.equal(ambiguous.configurationSchema, undefined);
    servedMetadata = undefined;
    provider.notifyChanged();
    const [unidentified] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    assert.equal(unidentified.configurationSchema, undefined);
  });

  it("keeps a remote SDK effort selection through an authenticated proxy prefix and tool history", async (t) => {
    const originalEndpoint = endpoint;
    endpoint += "/llama-proxy";
    expectedApiKey = "fixture-remote-key";
    core.setServerApiKeyProvider(origin => origin === new URL(endpoint).origin ? expectedApiKey : undefined);
    t.after(() => { endpoint = originalEndpoint; expectedApiKey = undefined; core.setServerApiKeyProvider(undefined); });
    servedModelId = "opaque-instance";
    servedMetadata = { owned_by: "strata", meta: { label: "Spark-X2.6-4B" } };
    propsStatus = 404;
    const { provider, state } = setup({ remote: true, remoteInfo: { isLlamaCpp: false, servedContext: 262144 } });
    state.modelCapabilities = { architecture: "gemma4", path: "/local/gemma-4-12B-it.gguf" };
    const [model] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    assert.deepEqual(model.configurationSchema.properties.reasoningEffort.enum, ["high", "low", "none"]);
    assert.equal(model.maxInputTokens + model.maxOutputTokens, 262144);
    const parts = [];
    await provider.provideLanguageModelChatResponse(model, [
      msg(Role.User, new TextPart("read a file")),
      msg(Role.Assistant, new ToolCallPart("call_1", "read_file", { path: "README.md" })),
      msg(Role.User, new ToolResultPart("call_1", [new TextPart("Example project")])),
    ], { modelConfiguration: { reasoningEffort: "none" },
      tools: [{ name: "read_file", description: "read", inputSchema: { type: "object" } }] },
    { report: part => parts.push(part) }, cancelToken().token);
    assert.deepEqual(lastBody.chat_template_kwargs, { enable_thinking: false });
    assert.equal(lastBody.model, "opaque-instance");
    assert.ok(lastBody.messages.some(m => m.role === "tool" && m.tool_call_id === "call_1"));
    assert.ok(lastBody.messages.some(m => m.tool_calls?.[0].function.arguments === '{"path":"README.md"}'));
    assert.equal(parts.filter(p => p instanceof TextPart).map(p => p.value).join(""), "ok");
    for (const suffix of ["/v1/models", "/v1/chat/completions"]) {
      assert.ok(seenRequests.some(r => r.url === `/llama-proxy${suffix}` && r.authorization === "Bearer fixture-remote-key"));
    }
  });

  for (const source of ["models", "runtime"]) {
    it(`advertises Strata's served 256K context from ${source} and accepts a prompt beyond the 32K fallback`, async () => {
      servedModelId = "gyro-gyro-s";
      servedMetadata = { owned_by: "strata", meta: {
        label: "Qwen3.8-Flash-Next Gyro-S", instanceId: "gyro-instance",
        ...(source === "models" ? { context: 262144 } : {}),
      } };
      servedNative = { instances: [{ id: "gyro-instance", apiModelId: servedModelId, status: "ready",
        context: 65536, metrics: { engine: { max_context: 262144 } } }] };
      propsStatus = 404;
      const info = await core.probeRemoteServer(endpoint);
      assert.equal(core.remoteSlotContext(info), 262144);
      const { provider } = setup({ remote: true, remoteInfo: info });
      const [model] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
      assert.equal(model.maxInputTokens + model.maxOutputTokens, 262144);
      assert.match(model.tooltip, /slot context 262144/);
      await provider.provideLanguageModelChatResponse(model,
        [msg(Role.User, new TextPart("word ".repeat(40000)))], {},
        { report() {} }, cancelToken().token);
      assert.equal(lastBody.model, "gyro-gyro-s");
      assert.equal(lastBody.messages.at(-1).content.length, 200000);
    });
  }

  for (const [id, label, family, defaultMode] of [
    ["q2_0", "Qwen3.8-Flash-Next Q2_0", "Qwen3.8 Flash-Next", "xhigh"],
    ["unsloth-ud-iq4_xs", "Qwen3.8-Flash-Next (Unsloth) UD-IQ4_XS", "Qwen3.8 Flash-Next", "xhigh"],
    ["swift-iq3_s", "Swift 1.5 IQ3_S", "Qwen3.8 Flash-Next", "xhigh"],
    ["gyro-gyro-m", "Qwen3.8-Flash-Next Gyro-M", "Qwen3.8 Flash-Next", "xhigh"],
    ["qwen36-ud-q4_k_xl", "Qwen3.6-35B-A3B UD-Q4_K_XL", "Qwen3.5 / 3.6", "medium"],
  ]) {
    for (const source of ["models", "remote probe"]) {
      it(`maps Desktop ID ${id} through the ${source} label and preserves its request ID`, async () => {
        servedModelId = id;
        servedMetadata = source === "models" ? { owned_by: "strata", meta: { label } } : undefined;
        propsStatus = 404;
        const { provider } = setup({ remote: true, remoteInfo: {
          isLlamaCpp: false, modelId: id,
          ...(source === "remote probe" ? { modelName: label } : {}),
        } });
        const [model] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
        assert.equal(model.id, id);
        assert.equal(model.configurationSchema.properties.reasoningEffort.default, defaultMode);
        assert.equal(provider.modeSetFor(id, undefined).familyLabel, family);
        await provider.provideLanguageModelChatResponse(model, [msg(Role.User, new TextPart("hello"))],
          {}, { report() {} }, cancelToken().token);
        assert.equal(lastBody.model, id);
        assert.equal(lastBody.chat_template_kwargs.reasoning_effort, family === "Qwen3.5 / 3.6" ? undefined : "xhigh");
      });
    }
  }

  for (const [effort, templateEffort, thinking] of [
    ["xhigh", "xhigh", true], ["medium", "medium", true],
    ["low", "low", true], ["none", undefined, false],
  ]) {
    it(`passes the Copilot CLI ${effort} selection to Strata's template`, async () => {
      servedModelId = "unsloth-ud-iq4_xs";
      servedMetadata = { owned_by: "strata", meta: { label: "Qwen3.8-Flash-Next (Unsloth) UD-IQ4_XS" } };
      propsStatus = 404;
      const { provider } = setup({ remote: true, remoteInfo: { modelId: servedModelId } });
      const [model] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
      const schema = model.configurationSchema.properties.reasoningEffort;
      // The CLI bridge retains standard values only, discarding display names.
      const allowed = new Set(["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]);
      assert.deepEqual(schema.enum.filter(value => allowed.has(value)), ["xhigh", "medium", "low", "none"]);
      await provider.provideLanguageModelChatResponse(model, [msg(Role.User, new TextPart("hello"))],
        { modelConfiguration: { reasoningEffort: effort } }, { report() {} }, cancelToken().token);
      assert.equal(lastBody.model, servedModelId);
      assert.equal(lastBody.chat_template_kwargs.reasoning_effort, templateEffort);
      assert.equal(lastBody.chat_template_kwargs.enable_thinking, thinking);
      assert.equal(lastBody.reasoning_effort, undefined);
      assert.equal(lastBody.temperature, thinking ? 1 : 0.7);
    });
  }

  for (const [label, efforts, sampling, kwargs] of [
    ["MiniCPM5-2B", ["high", "none"], [1, 0.95, -1, 1],
      [{ enable_thinking: true }, { enable_thinking: false }]],
    ["Mellum2.1-12B-A2.5B-Thinking", ["high"], [0.6, 0.95, 20, 1], [{}]],
    ["MiMo-V2.6-Distill-Qwen-9B", ["high"], [0.6, 0.95, 20, 1], [{}]],
    ["Flint", ["high"], [1, 0.95, 20, 1], [{}]],
    ["Sage-A1.5", ["high"], [1, 0.95, 20, 1], [{ enable_thinking: true, preserve_thinking: true }]],
    ["GLM-5.3", ["max", "high", "low"], [1, 0.95, -1, 1],
      [{ reasoning_effort: "max", clear_thinking: true }, { reasoning_effort: "high", clear_thinking: true }, { reasoning_effort: "low", clear_thinking: true }]],
    ["Spark-X2.5-4B", ["high", "none"], [1, 0.95, -1, 1],
      [{ enable_thinking: true }, { enable_thinking: false }]],
    ["Spark-X2.6-4B", ["high", "low", "none"], [1, 0.95, -1, 1],
      [{ enable_thinking: true, reasoning_effort: "high" }, { enable_thinking: true, reasoning_effort: "low" }, { enable_thinking: false }]],
    ["gpt-oss-20b", ["high", "medium", "low"], [1, 1, -1, 1],
      [{ reasoning_effort: "high" }, { reasoning_effort: "medium" }, { reasoning_effort: "low" }]],
    ["GLM-5.3-Flash", ["max", "high", "low"], [1, 0.95, -1, 1],
      [{ reasoning_effort: "max", clear_thinking: true }, { reasoning_effort: "high", clear_thinking: true }, { reasoning_effort: "low", clear_thinking: true }]],
    ["LFM2.5-2.6B", ["high"], [0.1, 1, 50, 1.1], [{}]],
    ["LFM2.5-24B-A2B", ["high"], [0.1, 1, 50, 1.05], [{}]],
    ["LFM2.5-1.2B-Instruct", ["none"], [0.1, 1, 50, 1.05], [{}]],
    ["Qwen3-4B-Instruct-2507", ["none"], [0.7, 0.8, 20, 1], [{}]],
    ["Qwen3-4B-Thinking-2507", ["high"], [0.6, 0.95, 20, 1], [{}]],
    ["Qwen3.8-35B-A3B-Distill", ["high"], [0.6, 0.95, 20, 1], [{ enable_thinking: true }]],
    ["Qwen3-Coder-30B-A3B-Instruct", ["none"], [0.7, 0.8, 20, 1.05], [{}]],
    ["Qwen3-Coder-Next", ["none"], [1, 0.95, 40, 1], [{}]],
    ["NVIDIA-Nemotron-3.5-Bolt-E4B", ["high", "none"], [1, 0.95, 64, 1],
      [{ enable_thinking: true }, { enable_thinking: false }]],
  ]) {
    it(`sends every ${label} picker option with its author settings through an opaque remote ID`, async () => {
      servedModelId = "opaque-q4-instance";
      servedMetadata = { owned_by: "strata", meta: { label } };
      propsStatus = 404;
      const { provider, state } = setup({ remote: true });
      Object.assign(state.requestSettings, { temperature: 0, topP: 0.1, topK: 1,
        minP: 0.9, repeatPenalty: 2, presencePenalty: 2, frequencyPenalty: 2 });
      const [model] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
      assert.deepEqual(model.configurationSchema.properties.reasoningEffort.enum, efforts);
      for (const [index, effort] of efforts.entries()) {
        await provider.provideLanguageModelChatResponse(model, [msg(Role.User, new TextPart("hello"))],
          { modelConfiguration: { reasoningEffort: effort },
            tools: [{ name: "read_file", description: "read", inputSchema: { type: "object" } }], toolMode: 2 },
          { report() {} }, cancelToken().token);
        assert.equal(lastBody.model, servedModelId);
        assert.deepEqual([lastBody.temperature, lastBody.top_p, lastBody.top_k, lastBody.repeat_penalty], sampling);
        assert.deepEqual(lastBody.chat_template_kwargs, kwargs[index]);
        assert.equal(lastBody.frequency_penalty, 0);
        assert.equal(lastBody.min_p, 0);
        assert.equal(lastBody.reasoning_effort, undefined);
        assert.equal(lastBody.tool_choice, "required");
        assert.equal(lastBody.tools[0].function.name, "read_file");
      }
    });
  }

  for (const [label, effort, temperature, topP, topK, presence, minP, kwargs] of [
    ["Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL", "high", 0.6, 0.95, 20, 0, 0, { enable_thinking: true }],
    ["Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL", "none", 0.7, 0.8, 20, 1.5, 0, { enable_thinking: false }],
    ["Underdog-Saluki-27B-1.0", "high", 0.6, 0.95, 20, 0, 0, { enable_thinking: true }],
    ["Underdog-Saluki-27B-1.0", "none", 0, 0.95, 20, 0, 0, { enable_thinking: false }],
    ["Ornith-1.5-9B", "high", 1, 0.95, 20, 1.5, 0, { enable_thinking: true, preserve_thinking: true }],
    ["NVIDIA-Nemotron-3-Nano-4B", "high", 1, 0.95, -1, 0, 0, { enable_thinking: true }],
    ["NVIDIA-Nemotron-3-Nano-30B-A3B", "high", 1, 1, -1, 0, 0, { enable_thinking: true }],
    ["Ternary-Bonsai-2-27B", "xhigh", 1, 0.95, 20, 0, 0.05, { enable_thinking: true, reasoning_effort: "xhigh" }],
    ["Ternary-Bonsai-2-27B", "none", 0.7, 0.8, 20, 1.5, 0, { enable_thinking: false }],
  ]) {
    it(`sends ${label} ${effort} with its distinct penalties and template controls`, async () => {
      servedModelId = "opaque-q4-instance";
      servedMetadata = { owned_by: "strata", meta: { label } };
      propsStatus = 404;
      const { provider } = setup({ remote: true });
      const [model] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
      await provider.provideLanguageModelChatResponse(model, [msg(Role.User, new TextPart("hello"))],
        { modelConfiguration: { reasoningEffort: effort } }, { report() {} }, cancelToken().token);
      assert.equal(lastBody.model, servedModelId);
      assert.deepEqual([lastBody.temperature, lastBody.top_p, lastBody.top_k, lastBody.presence_penalty, lastBody.min_p],
        [temperature, topP, topK, presence, minP]);
      assert.deepEqual(lastBody.chat_template_kwargs, kwargs);
    });
  }

  for (const label of ["DeepSeek-V4.1-Flash", "DeepSeek-V4.1-Flash-1017"]) {
    it(`maps every ${label} SDK effort to the numeric budget without changing the remote request ID`, async () => {
      servedModelId = "opaque-instance";
      servedMetadata = { owned_by: "strata", meta: { label } };
      propsStatus = 404;
      const { provider } = setup({ remote: true });
      const [model] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
      assert.deepEqual(model.configurationSchema.properties.reasoningEffort.enum, ["max", "high", "low", "none"]);
      for (const [effort, budget] of [["max", 100], ["high", 75], ["low", 50], ["none", undefined]]) {
        await provider.provideLanguageModelChatResponse(model, [msg(Role.User, new TextPart("hello"))],
          { modelConfiguration: { reasoningEffort: effort } }, { report() {} }, cancelToken().token);
        assert.equal(lastBody.model, "opaque-instance");
        assert.equal(lastBody.top_p, 1);
        assert.deepEqual(lastBody.chat_template_kwargs, budget === undefined
          ? { enable_thinking: false } : { enable_thinking: true, reasoning_effort: budget });
        assert.equal(lastBody.reasoning_effort, undefined);
      }
    });
  }

  for (const remote of [false, true]) {
    for (const label of ["Flint", "DeepSeek-V4.1-Flash-1017"]) {
      it(`round-trips ${label} SDK thinking parts separately from text across tool turns (remote=${remote})`, async () => {
        servedModelId = remote ? "opaque-instance" : label;
        servedMetadata = { owned_by: "strata", meta: { label } };
        propsStatus = 404;
        const { provider } = setup({ remote });
        const [model] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
        const options = { tools: [{ name: "read_file", description: "read", inputSchema: { type: "object" } }] };
        reply = { events: [delta({ reasoning_content: "I need " }), delta({ reasoning_content: "the file." }),
          delta({ tool_calls: [{ index: 0, id: "call_1", function: { name: "read_file", arguments: '{"path":"README.md"}' } }] })], delayMs: 0 };
        const parts = [];
        await provider.provideLanguageModelChatResponse(model, [msg(Role.User, new TextPart("Read it"))],
          options, { report: part => parts.push(part) }, cancelToken().token);
        assert.equal(parts.filter(p => p instanceof ThinkingPart).map(p => p.value).join(""), "I need the file.");
        assert.equal(parts.filter(p => p instanceof TextPart).length, 0);
        assert.equal(parts.filter(p => p instanceof ToolCallPart).length, 1);
        reply = { events: [delta({ content: "done" })], delayMs: 0 };
        await provider.provideLanguageModelChatResponse(model, [msg(Role.User, new TextPart("Read it")),
          msg(Role.Assistant, ...parts), msg(Role.User, new ToolResultPart("call_1", [new TextPart("Project text")]))],
          options, { report() {} }, cancelToken().token);
        const assistant = lastBody.messages.find(m => m.role === "assistant");
        assert.equal(assistant.reasoning_content, "I need the file.");
        assert.equal(assistant.content, null);
        assert.equal(assistant.tool_calls[0].id, "call_1");
      });
    }
  }

  it("keeps the legacy text fallback when the host has no thinking-part API", async (t) => {
    const api = require.cache.vscode.exports;
    api.LanguageModelThinkingPart = undefined;
    t.after(() => { api.LanguageModelThinkingPart = ThinkingPart; });
    servedModelId = "Flint";
    const { provider } = setup();
    reply = { events: [delta({ reasoning_content: "Working." })], delayMs: 0 };
    const parts = [];
    await provider.provideLanguageModelChatResponse({ id: "Flint" }, [msg(Role.User, new TextPart("hello"))],
      {}, { report: part => parts.push(part) }, cancelToken().token);
    assert.equal(parts.filter(p => p instanceof TextPart).map(p => p.value).join(""), "Working.");
  });

  for (const source of ["props", "remote probe"]) {
    it(`uses Strata's canonical model alias from ${source} for the picker and request profile`, async () => {
      servedModelId = "local-model";
      const identity = { modelAlias: "swift-1.5-iq3_s", modelPath: "C:\\models\\pack\\full" };
      servedProps = source === "props"
        ? { model_alias: identity.modelAlias, model_path: identity.modelPath }
        : {};
      const { provider, state } = setup({ remote: true, remoteInfo: source === "remote probe" ? identity : {} });
      state.modelCapabilities = { architecture: "gemma4", path: "/m/gemma4.gguf" };
      const [model] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
      assert.equal(model.id, "local-model");
      assert.equal(model.configurationSchema.properties.reasoningEffort.default, "xhigh");
      assert.match(model.tooltip, /Think \(Medium\)/);
      await provider.provideLanguageModelChatResponse(
        model, [msg(Role.User, new TextPart("hello"))],
        { modelConfiguration: { reasoningEffort: "Think (Medium)" } },
        { report() {} }, cancelToken().token
      );
      assert.equal(lastBody.model, "local-model");
      assert.equal(lastBody.chat_template_kwargs.reasoning_effort, "medium");
      assert.equal(lastBody.temperature, 1);
      assert.equal(lastBody.top_k, 20);
    });
  }

  it("recovers the thinking modes when a probe saw the model before its label", async () => {
    servedModelId = "unsloth-ud-iq4_xs";
    servedMetadata = undefined;
    propsStatus = 404;
    const remoteInfo = { isLlamaCpp: false, modelId: servedModelId, servedContext: 262144 };
    const { provider } = setup({ remote: true, remoteInfo });
    const announcements = [];
    provider.notifyChanged = () => announcements.push("changed");

    const [degraded] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    assert.equal(degraded.configurationSchema, undefined);
    assert.equal(degraded.maxInputTokens + degraded.maxOutputTokens, 262144);

    // The next probe reads Strata's label; Copilot must be told to re-query.
    servedMetadata = { owned_by: "strata", meta: { label: "Qwen3.8-Flash-Next (Unsloth) UD-IQ4_XS" } };
    provider.modelProbe = undefined;
    const [recovered] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    assert.equal(recovered.configurationSchema.properties.reasoningEffort.default, "xhigh");
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.deepEqual(announcements, ["changed"]);

    // A stable answer must not announce again (that would loop with Copilot's re-query).
    await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.deepEqual(announcements, ["changed"]);

    // A later probe that lists the id without its label must keep the family.
    servedMetadata = undefined;
    remoteInfo.modelName = undefined;
    provider.modelProbe = { endpoint, at: Date.now(), modelId: servedModelId };
    const [relabelled] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    assert.equal(relabelled.configurationSchema.properties.reasoningEffort.default, "xhigh");
  });

  it("keeps thinking modes when a later probe identifies no model at all", async () => {
    servedModelId = "unsloth-ud-iq4_xs";
    servedMetadata = { owned_by: "strata", meta: { label: "Qwen3.8-Flash-Next (Unsloth) UD-IQ4_XS" } };
    propsStatus = 404;
    const remoteInfo = { isLlamaCpp: false, modelId: servedModelId, servedContext: 262144 };
    const { provider } = setup({ remote: true, remoteInfo });
    const [first] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    assert.equal(first.configurationSchema.properties.reasoningEffort.default, "xhigh");

    // Strata reloading: /v1/models lists the id without a label, /props is still its web UI.
    servedMetadata = undefined;
    remoteInfo.modelName = undefined;
    provider.modelProbe = { endpoint, at: Date.now(), modelId: servedModelId };
    const [degraded] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    assert.equal(degraded.configurationSchema.properties.reasoningEffort.default, "xhigh");

    // A different, identifiable model must not inherit those modes.
    servedMetadata = { owned_by: "strata", meta: { label: "Mellum2.1-12B-Thinking" } };
    provider.modelProbe = undefined;
    const [otherModel] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    assert.equal(otherModel.configurationSchema, undefined);
  });

  it("uses the remote filename across Windows separators without matching parent folder names", () => {
    const { provider } = setup({ remote: true });
    assert.equal(provider.modeSetFor("local-model", { model_path: "C:\\models\\Qwen3.8-Flash-Next.gguf" }).familyLabel,
      "Qwen3.8 Flash-Next");
    assert.equal(provider.modeSetFor("local-model", { model_path: "C:\\Qwen3.8-Flash-Next\\pack\\full" }), undefined);
    assert.equal(provider.modeSetFor("qwen3.8-flash-next", { model_alias: "qwen3.6-35b-a3b" }).familyLabel, "Qwen3.5 / 3.6");
    assert.equal(provider.modeSetFor("local-model", {
      model_alias: "qwen3.6-35b-a3b", model_path: "C:\\models\\Qwen3.8-Flash-Next.gguf",
    }).familyLabel, "Qwen3.5 / 3.6");
  });

  for (const [toolMode, expected] of [[undefined, "auto"], [1, "auto"], [2, "required"]]) {
    it(`sends ${expected} tool choice for API tool mode ${toolMode}`, async () => {
      const { provider } = setup();
      await provider.provideLanguageModelChatResponse(
        { id: "test-model" }, [msg(Role.User, new TextPart("hello"))],
        { tools: [{ name: "read_file", description: "read", inputSchema: {} }], toolMode },
        { report() {} }, cancelToken().token
      );
      assert.equal(lastBody.tool_choice, expected);
      assert.equal(lastBody.tools[0].function.name, "read_file");
    });
  }

  it("orders tool results right after the assistant call and puts guard hints at the front", async () => {
    const { provider } = setup({ guard: true });
    const reported = [];
    await provider.provideLanguageModelChatResponse(
      { id: "test-model" },
      [
        msg(Role.User, new TextPart("Read a.ts")),
        msg(Role.Assistant, new ToolCallPart("c1", "read_file", { path: "a.ts" })),
        msg(Role.User, new ToolResultPart("c1", [new TextPart("file body")]), new TextPart("thanks")),
      ],
      { tools: [{ name: "read_file", description: "read", inputSchema: {} }] },
      { report: (p) => reported.push(p) },
      cancelToken().token
    );
    const roles = lastBody.messages.map((m) => m.role);
    assert.deepEqual(roles, ["system", "user", "assistant", "tool", "user"]);
    assert.match(lastBody.messages[0].content, /Do not repeat a tool call/);
    assert.equal(lastBody.messages[3].tool_call_id, "c1");
    assert.equal(lastBody.messages.at(-1).role, "user", "no system message after the conversation");
    assert.equal(reported.map((p) => p.value).join(""), "ok");
  });

  it("ends quietly when the user cancels mid-stream", async () => {
    reply = { events: [delta({ content: "a" }), delta({ content: "b" }), delta({ content: "c" })], delayMs: 150 };
    const { provider, perfCalls } = setup();
    const { token, cancel } = cancelToken();
    const reported = [];
    await provider.provideLanguageModelChatResponse(
      { id: "test-model" },
      [msg(Role.User, new TextPart("hi"))],
      {},
      {
        report: (p) => {
          reported.push(p);
          cancel();
        },
      },
      token
    );
    assert.equal(reported.length, 1, "nothing after cancel");
    const names = perfCalls.map(([n]) => n);
    assert.ok(names.includes("abort"));
    assert.ok(!names.includes("complete"));
  });

  it("reports an empty reply when the stream ends with nothing to show", async () => {
    reply = { events: [{ choices: [{ delta: {}, finish_reason: "stop" }] }], delayMs: 0 };
    const { provider } = setup();
    await assert.rejects(
      () =>
        provider.provideLanguageModelChatResponse(
          { id: "test-model" },
          [msg(Role.User, new TextPart("hi"))],
          {},
          { report() {} },
          cancelToken().token
        ),
      /Empty reply/
    );
  });

  it("turns a mid-stream server error into a LanguageModelError", async () => {
    reply = { events: [delta({ content: "par" }), { error: { message: "slot crashed" } }], delayMs: 0 };
    const { provider } = setup();
    await assert.rejects(
      () =>
        provider.provideLanguageModelChatResponse(
          { id: "test-model" },
          [msg(Role.User, new TextPart("hi"))],
          {},
          { report() {} },
          cancelToken().token
        ),
      (e) => e instanceof LanguageModelError && /slot crashed/.test(e.message)
    );
  });
});

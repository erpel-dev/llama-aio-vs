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
const { LlamaAioChatProvider } = require(path.join(__dirname, "..", "out-harness", "chatProvider.js"));

let server;
let endpoint = "";
let lastBody;
let servedModelId;
let servedProps;
let servedMetadata;
let propsStatus;
let servedNative;
/** How /v1/chat/completions answers: array of SSE data objects, optional per-event delay. */
let reply = { events: [], delayMs: 0 };

before(async () => {
  server = http.createServer((req, res) => {
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
    ["q2_0", "Qwen3.8-Flash-Next Q2_0", "Qwen3.8 Flash-Next", "Think (XHigh)"],
    ["unsloth-ud-iq4_xs", "Qwen3.8-Flash-Next (Unsloth) UD-IQ4_XS", "Qwen3.8 Flash-Next", "Think (XHigh)"],
    ["swift-iq3_s", "Swift 1.5 IQ3_S", "Qwen3.8 Flash-Next", "Think (XHigh)"],
    ["gyro-gyro-m", "Qwen3.8-Flash-Next Gyro-M", "Qwen3.8 Flash-Next", "Think (XHigh)"],
    ["qwen36-ud-q4_k_xl", "Qwen3.6-35B-A3B UD-Q4_K_XL", "Qwen3", "Think (Coding)"],
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
        assert.equal(lastBody.chat_template_kwargs.reasoning_effort, family === "Qwen3" ? undefined : "xhigh");
      });
    }
  }

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
      assert.equal(model.configurationSchema.properties.reasoningEffort.default, "Think (XHigh)");
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
    assert.equal(recovered.configurationSchema.properties.reasoningEffort.default, "Think (XHigh)");
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
    assert.equal(relabelled.configurationSchema.properties.reasoningEffort.default, "Think (XHigh)");
  });

  it("keeps thinking modes when a later probe identifies no model at all", async () => {
    servedModelId = "unsloth-ud-iq4_xs";
    servedMetadata = { owned_by: "strata", meta: { label: "Qwen3.8-Flash-Next (Unsloth) UD-IQ4_XS" } };
    propsStatus = 404;
    const remoteInfo = { isLlamaCpp: false, modelId: servedModelId, servedContext: 262144 };
    const { provider } = setup({ remote: true, remoteInfo });
    const [first] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    assert.equal(first.configurationSchema.properties.reasoningEffort.default, "Think (XHigh)");

    // Strata reloading: /v1/models lists the id without a label, /props is still its web UI.
    servedMetadata = undefined;
    remoteInfo.modelName = undefined;
    provider.modelProbe = { endpoint, at: Date.now(), modelId: servedModelId };
    const [degraded] = await provider.provideLanguageModelChatInformation({}, cancelToken().token);
    assert.equal(degraded.configurationSchema.properties.reasoningEffort.default, "Think (XHigh)");

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
    assert.equal(provider.modeSetFor("qwen3.8-flash-next", { model_alias: "qwen3.6-35b-a3b" }).familyLabel, "Qwen3");
    assert.equal(provider.modeSetFor("local-model", {
      model_alias: "qwen3.6-35b-a3b", model_path: "C:\\models\\Qwen3.8-Flash-Next.gguf",
    }).familyLabel, "Qwen3");
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

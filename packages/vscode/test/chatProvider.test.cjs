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
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ default_generation_settings: { n_ctx: 32768 }, total_slots: 1 }));
        return;
      }
      if (req.url === "/v1/models") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ data: [{ id: "test-model" }] }));
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

function setup({ guard = false } = {}) {
  const perfCalls = [];
  const state = {
    selectedModelPath: "/m/test.gguf",
    loadSettings: { ...core.DEFAULT_LOAD_SETTINGS },
    requestSettings: { ...core.DEFAULT_REQUEST_SETTINGS, overthinkingPenalty: false },
  };
  const store = {
    getEndpoint: () => endpoint,
    isRemote: () => false,
    getState: () => state,
    getServerTarget: () => ({ mode: "local", baseUrl: endpoint, label: "this machine" }),
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
    remote: { current: () => undefined },
  };
  const perf = new Proxy({}, { get: (_t, name) => (...args) => perfCalls.push([name, args]) });
  const provider = new LlamaAioChatProvider(store, pm, perf, path.join(__dirname, ".."));
  return { provider, perfCalls };
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
  });

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

// Optional network check after compiling core: NODE_USE_ENV_PROXY=1 node scripts/verify-copilot-sdk-bridge.cjs
// Executes pinned VS Code SDK Responses translation and renderer bridge code.
// VS Code services are stubbed; this is a bridge contract check, not a signed-in UI session.
const assert = require("node:assert/strict");
const ts = require("typescript");
const vm = require("node:vm");
const core = require("../packages/core/out/modelModes.js");
const sources = require("./model-profile-sources.json");
const revision = "9a717654ea25990c928c6fc190af37ce2e4b5047";
const root = `https://raw.githubusercontent.com/microsoft/vscode/${revision}/`;
const noop = () => {};
const services = new Proxy({
  Disposable: class {},
  ChatMessageRole: { System: 0, User: 1, Assistant: 2 },
  ChatEntitlementContextKeys: { clientByokEnabled: { key: "clientByokEnabled" } },
  SessionType: { AgentHostCopilot: "agent-host-copilot" },
  getModelContextWindowTotal: model => model.maxInputTokens + model.maxOutputTokens,
  getByokLmAgentModelId: model => `${model.vendor}/${model.id}`,
}, { get: (target, key) => key in target ? target[key] : noop });

async function load(path) {
  const response = await fetch(root + path);
  assert.equal(response.status, 200, path);
  const code = ts.transpileModule(await response.text(), { compilerOptions: {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, experimentalDecorators: true,
  } }).outputText;
  const exports = {};
  vm.runInNewContext(code, { exports, require: () => services, Buffer, console, setTimeout, clearTimeout }, { filename: path });
  return exports;
}

async function main() {
  const [translation, renderer] = await Promise.all([
    load("src/vs/platform/agentHost/node/copilot/byokResponsesTranslation.ts"),
    load("src/vs/workbench/contrib/chat/browser/agentSessions/agentHost/agentHostByokLmHandler.ts"),
  ]);
  let count = 0;
  for (const id of [...sources.map(source => source.id), "Ternary-Bonsai-2-27B", "Muse-Glimmer-30B"]) {
    const profile = core.resolveModelModes(undefined, id);
    const schema = core.buildModeConfigurationSchema(profile);
    const metadata = { isBYOK: true, vendor: "llama-aio", id: "opaque-instance", name: id,
      maxInputTokens: 245760, maxOutputTokens: 16384, configurationSchema: schema, capabilities: { vision: false } };
    let captured;
    const handler = Object.create(renderer.AgentHostByokLmHandler.prototype);
    handler._chatEntitlementService = { clientByokEnabled: true };
    handler._logService = { warn: message => { throw new Error(message); } };
    handler._languageModelsService = {
      getLanguageModelIds: () => ["llama-aio/opaque-instance"],
      lookupLanguageModel: key => key === "llama-aio/opaque-instance" ? metadata : undefined,
      isModelHidden: () => false,
      sendChatRequest: async (model, _extension, messages, options) => {
        captured = { model, messages, options };
        return { result: Promise.resolve(), stream: (async function* () {
          yield { type: "text", value: "ok" };
          yield { type: "tool_use", name: "read_file", toolCallId: "call_2", parameters: { path: "README.md" } };
        })() };
      },
    };
    const [listed] = await handler.listModels({});
    assert.deepEqual(Array.from(listed.supportedReasoningEfforts), schema.properties.reasoningEffort.enum);
    assert.equal(listed.defaultReasoningEffort, schema.properties.reasoningEffort.default);
    assert.equal(listed.maxContextWindowTokens, 262144);
    for (const effort of [...listed.supportedReasoningEfforts, undefined]) {
      const request = translation.responsesRequestToBridge("llama-aio", {
        model: "opaque-instance", ...(effort ? { reasoning: { effort } } : {}), input: [
          { type: "message", role: "user", content: "Read README.md" },
          { type: "function_call", call_id: "call_1", name: "read_file", arguments: '{"path":"README.md"}' },
          { type: "function_call_output", call_id: "call_1", output: "Example project" },
        ], tools: [{ type: "function", name: "read_file", parameters: { type: "object" } }],
      });
      const result = await handler.chat(request, {});
      assert.equal(result.error, undefined);
      assert.equal(captured.options.configuration?.reasoningEffort, effort);
      assert.equal(captured.model, "llama-aio/opaque-instance");
      assert.equal(captured.options.tools[0].name, "read_file");
      assert.equal(captured.messages.at(-1).content[0].toolCallId, "call_1");
      assert.equal(result.output[1].type, "function_call");
      assert.equal(result.output[1].argumentsJson, '{"path":"README.md"}');
      const selected = core.resolveModeParams(profile, captured.options.configuration);
      assert.equal(selected.modeName, core.resolveModeParams(profile, effort ? { reasoningEffort: effort } : undefined).modeName);
      if (effort === "none") assert.equal(selected.modeName, "No Think");
      count++;
    }
  }
  console.log(`PASS: ${count} SDK Responses/renderer bridge cases at VS Code ${revision}`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { denseCaps } from "./helpers";
import { applyModeToRequestBody, buildModeConfigurationSchema, resolveModeParams, resolveModelModes, resolveRemoteModelModes } from "../src/modelModes";

// Contract expectations taken from the author recipes linked in docs/model-profiles.md.
const cases: Array<[string, string, string, string[]]> = [
  ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B Instruct", "No Think", ["No Think"]],
  ["Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf", "Tiel-Coder 35B-A3B", "Think", ["Think", "No Think"]],
  ["MiniCPM5-2B-GGUF", "MiniCPM5 2B", "Think", ["Think", "No Think"]],
  ["Mellum2.1-12B-A2.5B-Thinking-GGUF", "Mellum 2.1 Thinking", "Think", ["Think"]],
  ["MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf", "MiMo V2.6 Distill Qwen 9B", "Think", ["Think"]],
  ["Underdog-Saluki-27B-1.0-IQ2-mix.gguf", "Underdog Saluki 27B", "Think", ["Think", "No Think"]],
  ["RecursiveSI/flint-int4-gguf", "Flint", "Think", ["Think"]],
  ["Sage-A1.5-Q4_K_M.gguf", "Sage A1.5", "Think", ["Think"]],
  ["GLM-5.3-Q4_K_M.gguf", "GLM-5.3", "Think (Max)", ["Think (Max)", "Think (High)", "Think (Low)"]],
  ["DeepSeek-V4.1-Flash-Q4_K_M.gguf", "DeepSeek V4.1 Flash", "Think (High)", ["Think (Max)", "Think (High)", "Think (Low)", "No Think"]],
  ["DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF", "DeepSeek V4.1 Flash 1017", "Think (High)", ["Think (Max)", "Think (High)", "Think (Low)", "No Think"]],
  ["XHToken/Spark-X2.5-1.7B-GGUF", "Spark X2.5", "Think", ["Think", "No Think"]],
  ["Spark-X2.5-4B-Q4_K_M.gguf", "Spark X2.5", "Think", ["Think", "No Think"]],
  ["Spark-X2.6-4B-Q4_K_M.gguf", "Spark X2.6", "Think (High)", ["Think (High)", "Think (Low)", "No Think"]],
  ["openai/gpt-oss-20b", "GPT-OSS", "Think (Medium)", ["Think (High)", "Think (Medium)", "Think (Low)"]],
  ["gpt-oss-120b-MXFP4.gguf", "GPT-OSS", "Think (Medium)", ["Think (High)", "Think (Medium)", "Think (Low)"]],
  ["GLM-5.3-Flash-Q4_K_M.gguf", "GLM-5.3-Flash", "Think (Max)", ["Think (Max)", "Think (High)", "Think (Low)"]],
  ["Ornith-1.5-9B-Q4_K_M.gguf", "Ornith 1.5 (Qwen)", "Think (Coding)", ["Think (General)", "Think (Coding)", "No Think"]],
  ["Ornith-1.5-35B-A3B", "Ornith 1.5 (Qwen)", "Think (Coding)", ["Think (General)", "Think (Coding)", "No Think"]],
  ["Ornith-1.5-122B-GGUF", "Ornith 1.5 (Qwen)", "Think (Coding)", ["Think (General)", "Think (Coding)", "No Think"]],
  ["NVIDIA-Nemotron-3-Nano-4B-BF16", "Nemotron 3 Nano 4B", "Think (Coding)", ["Think (General)", "Think (Coding)", "No Think"]],
  ["Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf", "Nemotron 3 Nano 30B", "Think (Coding)", ["Think (General)", "Think (Coding)", "No Think"]],
  ["NVIDIA-Nemotron-3.5-Bolt-E4B-4GB", "Nemotron 3.5 Bolt E4B", "Think", ["Think", "No Think"]],
  ["LiquidAI/LFM2.5-230M-GGUF", "LFM2.5 Instruct", "No Think", ["No Think"]],
  ["LFM2.5-1.2B-Instruct-Q4_0.gguf", "LFM2.5 Instruct", "No Think", ["No Think"]],
  ["LFM2.5-2.6B-Q4_0.gguf", "LFM2.5 2.6B", "Think", ["Think"]],
  ["LFM2.5-24B-A2B-GGUF", "LFM2.5 24B-A2B", "Think", ["Think"]],
  ["Ternary-Bonsai-2-27B-PQ2_0.gguf", "Ternary Bonsai 2 27B", "Think (XHigh)", ["Think (XHigh)", "Think (Medium)", "No Think"]],
  ["Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", "Qwen3-Coder Instruct", "No Think", ["No Think"]],
  ["Qwen3-Coder-480B-A35B-Instruct", "Qwen3-Coder Instruct", "No Think", ["No Think"]],
  ["Qwen3-Coder-Next-Q4_K_M.gguf", "Qwen3-Coder-Next", "No Think", ["No Think"]],
  ["Qwen3-4B-Instruct-2507", "Qwen3 Instruct 2507", "No Think", ["No Think"]],
  ["Qwen3-4B-Thinking-2507", "Qwen3 Thinking 2507", "Think", ["Think"]],
  ["Qwen3-8B", "Qwen3", "Think", ["Think", "No Think"]],
  ["empero-ai/Qwen3.8-35B-A3B-Distill-GGUF", "Qwen3.8 Distill 35B-A3B", "Think", ["Think"]],
  ["Qwen3.6-35B-A3B", "Qwen3.5 / 3.6", "Think (Coding)", ["Think (General)", "Think (Coding)", "No Think"]],
  ["Qwen3.8-27B", "Qwen3.8", "Think (XHigh)", ["Think (XHigh)", "Think (Medium)", "Think (Low)", "No Think"]],
  ["gemma-4-12B-it", "Gemma 4", "Think", ["Think", "No Think"]],
  ["Muse-Glimmer-30B", "Glimmer", "Think (High)", ["Think (XHigh)", "Think (High)", "Think (Medium)", "Think (Low)"]],
];

describe("author profile contracts and Copilot effort compatibility", () => {
  for (const [id, family, defaultMode, modes] of cases) {
    it(`advertises only supported modes for ${id} and applies both CLI and saved labels`, () => {
      const set = resolveModelModes(undefined, id);
      assert.ok(set, id);
      assert.equal(set.familyLabel, family);
      assert.equal(set.defaultMode, defaultMode);
      assert.deepEqual(Object.keys(set.modes), modes);
      const schema = buildModeConfigurationSchema(set).properties.reasoningEffort as { enum: string[]; default: string };
      assert.equal(new Set(schema.enum).size, modes.length);
      assert.ok(schema.enum.every(v => ["none", "low", "medium", "high", "xhigh", "max"].includes(v)));
      assert.equal(resolveModeParams(set, { reasoningEffort: schema.default }).modeName, defaultMode);
      for (let n = 0; n < modes.length; n++) {
        const selected = resolveModeParams(set, { reasoningEffort: schema.enum[n] });
        assert.equal(selected.modeName, modes[n]);
        assert.deepEqual(resolveModeParams(set, { reasoningEffort: modes[n] }), selected);
        const body: Record<string, unknown> = { temperature: 0, top_p: 0.1, top_k: 1, min_p: 0.9,
          repeat_penalty: 2, presence_penalty: 2, frequency_penalty: 2,
          chat_template_kwargs: { reasoning_effort: "unsupported-stale-value" } };
        applyModeToRequestBody(body, selected.params);
        assert.equal(body.frequency_penalty, 0);
        assert.notEqual(body.top_k, 1);
        assert.equal(body.reasoning_effort, undefined);
        assert.equal(body.repetition_penalty, undefined);
        assert.notEqual((body.chat_template_kwargs as Record<string, unknown>).reasoning_effort, "unsupported-stale-value");
      }
      assert.equal(resolveModeParams(set, { reasoningEffort: "invalid" }).modeName, defaultMode);
    });
  }
});

function params(id: string, mode?: string) {
  const set = resolveModelModes(undefined, id);
  assert.ok(set);
  return resolveModeParams(set, mode ? { reasoningEffort: mode } : undefined).params;
}

describe("model-specific sampler and reasoning differences", () => {
  it("maps DeepSeek's numeric budgets and keeps checkpoint-specific top-p", () => {
    for (const [mode, budget] of [["max", 100], ["high", 75], ["low", 50]] as const) {
      assert.deepEqual(params("DeepSeek-V4.1-Flash-1017", mode).chat_template_kwargs,
        { enable_thinking: true, reasoning_effort: budget });
    }
    assert.equal(params("DeepSeek-V4.1-Flash").top_p, 1);
    assert.equal(params("DeepSeek-V4.1-Flash-1017").top_p, 1);
    assert.deepEqual(params("DeepSeek-V4.1-Flash-1017", "none").chat_template_kwargs, { enable_thinking: false });
    assert.equal(resolveModelModes(undefined, "Flint")?.preserveReasoning, true);
    assert.equal(resolveModelModes(undefined, "DeepSeek-V4.1-Flash-1017")?.preserveReasoning, true);
    assert.equal(params("Underdog-Saluki-27B-1.0", "none").temperature, 0);
    assert.equal(params("Sage-A1.5").presence_penalty, 1.5);
  });
  it("keeps Spark version controls separate despite their shared architecture", () => {
    for (const mode of ["Think (High)", "Think (Low)", "No Think"]) {
      const p = params("Spark-X2.6-4B", mode);
      assert.equal(p.temperature, 1);
      assert.equal(p.top_p, 0.95);
      assert.equal(p.top_k, -1);
      assert.equal(p.min_p, 0);
      assert.equal(p.chat_template_kwargs?.enable_thinking, mode !== "No Think");
    }
    assert.equal(params("Spark-X2.6-4B", "low").chat_template_kwargs?.reasoning_effort, "low");
    assert.equal(params("Spark-X2.6-4B", "medium").chat_template_kwargs?.reasoning_effort, "high");
    assert.equal(params("Spark-X2.5-4B").chat_template_kwargs?.reasoning_effort, undefined);
    assert.equal(resolveModelModes(denseCaps({ architecture: "spark2_5", path: "/m/Spark-X2.6-4B.gguf" }))?.familyLabel, "Spark X2.6");
  });

  it("uses the actual GPT-OSS and GLM effort vocabularies", () => {
    assert.deepEqual(params("gpt-oss-20b", "low").chat_template_kwargs, { reasoning_effort: "low" });
    assert.deepEqual(params("GLM-5.3-Flash", "max").chat_template_kwargs, { reasoning_effort: "max", clear_thinking: true });
    assert.equal(params("gpt-oss-20b", "none").chat_template_kwargs?.reasoning_effort, "medium");
  });

  it("separates original Qwen3, later Qwen recipes, and Coder recipes", () => {
    assert.equal(params("Qwen3-8B").temperature, 0.6);
    assert.equal(params("Qwen3-8B", "none").presence_penalty, 0);
    assert.equal(params("Qwen3.5-9B", "high").presence_penalty, 1.5);
    assert.equal(params("Qwen3.6-35B-A3B", "medium").presence_penalty, 0);
    assert.equal(params("Qwen3-Coder-30B-A3B-Instruct").repetition_penalty, 1.05);
    assert.equal(params("Qwen3-Coder-Next").temperature, 1);
    assert.equal(params("Qwen3-Coder-Next").top_k, 40);
    assert.equal(params("Qwen3.8-35B-A3B-Distill").temperature, 0.6);
    assert.equal(params("Qwen3.8-35B-A3B-Distill").chat_template_kwargs?.reasoning_effort, undefined);
  });

  it("does not give every Nemotron or Liquid checkpoint the same sampler", () => {
    assert.equal(params("Nemotron-3-Nano-4B", "high").top_p, 0.95);
    assert.equal(params("Nemotron-3-Nano-30B-A3B", "high").top_p, 1);
    assert.equal(params("Nemotron-3.5-Bolt-E4B").top_k, 64);
    assert.equal(params("LFM2.5-2.6B").repetition_penalty, 1.1);
    assert.equal(params("LFM2.5-24B-A2B").repetition_penalty, 1.05);
    assert.equal(params("LFM2.5-1.2B-Instruct").temperature, 0.1);
  });

  it("gives Bonsai its own min-p and omits the unsupported Low mode", () => {
    assert.equal(params("Ternary-Bonsai-2-27B").min_p, 0.05);
    assert.equal(params("Ternary-Bonsai-2-27B", "none").min_p, 0);
    assert.equal(params("Ternary-Bonsai-2-27B", "low").chat_template_kwargs?.reasoning_effort, "xhigh");
  });
});

describe("profile identity boundaries", () => {
  it("keeps main models with embedded MTP heads distinct from prediction sidecars", () => {
    assert.equal(resolveModelModes(undefined, "Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF")?.familyLabel, "Qwen3.8");
    assert.equal(resolveModelModes(undefined, "Qwen3.8-27B-MTP-Q4_K_M.gguf")?.familyLabel, "Qwen3.8");
    assert.equal(resolveModelModes(undefined, "Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF")?.familyLabel, "Qwen3.8");
  });
  for (const id of ["Spark-X2.7-4B", "LFM2.5-7B", "Ornith-1.5-31B", "Ornith-2.0-9B", "GLM-4.7-Flash",
    "MiniCPM5-2B-DSpark-GGUF", "Mellum2.1-12B-A2.5B-Base", "DeepSeek-V4.1-Flash-1018", "GLM-5.4",
    "gpt-oss-20b-Base", "Spark-X2.6-4B-Base", "mtp-Qwen3.8-27B.gguf", "mmproj-Qwen3.8.gguf",
    "Qwen3.8-Flash-Next-DFlash2", "/models/Qwen3.8/unknown.gguf", "C:\\models\\Qwen3.8\\unknown.gguf"]) {
    it(`does not invent a profile for ${id}`, () => assert.equal(resolveModelModes(undefined, id), undefined));
  }
  it("ignores other-model parent directories and gives named derivatives precedence over architecture", () => {
    assert.equal(resolveModelModes(denseCaps({ architecture: "unknown", path: "/models/Qwen3.8/unknown.gguf" })), undefined);
    assert.equal(resolveModelModes(denseCaps({ architecture: "qwen35", path: "/models/Ternary-Bonsai-2-27B.gguf" }))?.familyLabel, "Ternary Bonsai 2 27B");
    assert.equal(resolveModelModes(denseCaps({ architecture: "gemma4", path: "/models/Nemotron-3.5-Bolt-E4B.gguf" }))?.familyLabel, "Nemotron 3.5 Bolt E4B");
    assert.equal(resolveModelModes(denseCaps({ architecture: "dflash", path: "/models/Qwen3.8-27B.gguf" })), undefined);
    assert.equal(resolveModelModes(denseCaps({ architecture: "gemma4_unified", path: "/models/model.gguf" }))?.familyLabel, "Gemma 4");
    assert.equal(resolveModelModes(denseCaps({ architecture: "qwen3", path: "/models/Qwen3-Coder-Unknown.gguf" })), undefined);
    assert.equal(resolveModelModes(denseCaps({ architecture: "gemma4", path: "/models/Ornith-1.5-31B.gguf" })), undefined);
    assert.equal(resolveModelModes(denseCaps({ architecture: "spark2_5", path: "/models/Spark-X2.7-4B.gguf" })), undefined);
    assert.equal(resolveModelModes(denseCaps({ architecture: "gemma4", path: "/models/gemma-4-12B-it-Q4_0-MTP.gguf" })), undefined);
    assert.equal(resolveModelModes(denseCaps({ architecture: "gemma4", path: "/models/gemma-4-12B-it-mmproj.gguf" })), undefined);
  });
});

describe("remote profile lookup shared with the sidebar", () => {
  it("uses aliases, portable filenames, labels, then IDs without local metadata", () => {
    assert.equal(resolveRemoteModelModes({ modelId: "opaque", modelName: "Spark-X2.6-4B" })?.familyLabel, "Spark X2.6");
    assert.equal(resolveRemoteModelModes({ modelId: "Spark-X2.6-4B", modelAlias: "gpt-oss-20b" })?.familyLabel, "GPT-OSS");
    assert.equal(resolveRemoteModelModes({ modelId: "Spark-X2.6-4B", modelPath: "C:\\models\\gpt-oss-20b.gguf" })?.familyLabel, "GPT-OSS");
    assert.equal(resolveRemoteModelModes({ modelId: "Spark-X2.6-4B", modelPath: "/models/opaque.gguf" })?.familyLabel, "Spark X2.6");
    assert.equal(resolveRemoteModelModes({ modelId: "opaque", modelPath: "C:\\models\\Spark-X2.6\\pack\\full" }), undefined);
    assert.equal(resolveRemoteModelModes(undefined), undefined);
  });
});

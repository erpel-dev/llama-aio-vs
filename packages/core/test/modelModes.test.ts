import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { denseCaps } from "./helpers";
import {
  applyModeToRequestBody,
  isGemma4Family,
  isGlimmerFamily,
  isQwen38Family,
  resolveModeParams,
  resolveModelModes,
} from "../src/modelModes";

describe("isQwen38Family", () => {
  it("matches Qwen3.8 filenames and ids", () => {
    assert.equal(isQwen38Family("Qwen3.8-27B-UD-Q4_K_XL.gguf"), true);
    assert.equal(isQwen38Family("unsloth/qwen3.8-27b-gguf"), true);
    assert.equal(isQwen38Family("qwen-3.8-27b"), true);
    assert.equal(isQwen38Family("qwen38"), true);
    assert.equal(isQwen38Family("qwen3_8"), true);
  });

  it("does not match Qwen3 8B or Qwen3.6", () => {
    assert.equal(isQwen38Family("Qwen3-8B-Q4_K_M.gguf"), false);
    assert.equal(isQwen38Family("qwen3-8b"), false);
    assert.equal(isQwen38Family("Qwen3.6-27B.gguf"), false);
    assert.equal(isQwen38Family("qwen3"), false);
  });
});

describe("isGlimmerFamily", () => {
  it("matches architecture and Muse-Glimmer filenames", () => {
    assert.equal(isGlimmerFamily("muse-glimmer"), true);
    assert.equal(isGlimmerFamily("Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"), true);
    assert.equal(isGlimmerFamily("muse glimmer 30b"), true);
  });

  it("does not match unrelated names", () => {
    assert.equal(isGlimmerFamily("gemma4"), false);
    assert.equal(isGlimmerFamily("qwen3"), false);
  });
});

describe("isGemma4Family", () => {
  it("matches Gemma 4 architecture and filenames", () => {
    assert.equal(isGemma4Family("gemma4"), true);
    assert.equal(isGemma4Family("gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"), true);
    assert.equal(isGemma4Family("unsloth/gemma-4-31B-it-GGUF"), true);
  });

  it("does not match Gemma 2/3 or MTP sidecars", () => {
    assert.equal(isGemma4Family("gemma3"), false);
    assert.equal(isGemma4Family("gemma-3-12b-it.gguf"), false);
    assert.equal(isGemma4Family("gemma2"), false);
    assert.equal(isGemma4Family("mtp-gemma-4-12B-it.gguf"), false);
    assert.equal(isGemma4Family("gemma-4-12B-it-Q4_0-MTP.gguf"), false);
    assert.equal(isGemma4Family("gemma4-assistant"), false);
  });
});

describe("resolveModelModes", () => {
  it("gives Qwen3.8 XHigh/Medium/Low modes from the GGUF path", () => {
    const set = resolveModelModes(
      denseCaps({
        architecture: "qwen3",
        name: "Qwen3.8 27B",
        path: "/models/Qwen3.8-27B-Q4_K_M.gguf",
      }),
      "Qwen3.8-27B"
    );
    assert.ok(set);
    assert.equal(set.familyLabel, "Qwen3.8");
    assert.equal(set.defaultMode, "Think (XHigh)");
    assert.deepEqual(Object.keys(set.modes), [
      "Think (XHigh)",
      "Think (Medium)",
      "Think (Low)",
      "No Think",
    ]);
    assert.equal(set.modes["Think (XHigh)"].chat_template_kwargs?.reasoning_effort, "xhigh");
    assert.equal(set.modes["Think (Medium)"].chat_template_kwargs?.reasoning_effort, "medium");
    assert.equal(set.modes["Think (Low)"].chat_template_kwargs?.reasoning_effort, "low");
    assert.equal(set.modes["No Think"].chat_template_kwargs?.enable_thinking, false);
  });

  it("keeps Qwen3.6 on Think General/Coding without reasoning_effort", () => {
    const set = resolveModelModes(
      denseCaps({ architecture: "qwen3", path: "/models/Qwen3.6-27B-Q4_K_M.gguf" }),
      "Qwen3.6-27B"
    );
    assert.ok(set);
    assert.equal(set.familyLabel, "Qwen3");
    assert.equal(set.defaultMode, "Think (Coding)");
    assert.equal(set.modes["Think (Coding)"].chat_template_kwargs?.reasoning_effort, undefined);
    assert.equal(set.modes["Think (Coding)"].temperature, 0.6);
  });

  it("does not treat Qwen3-8B as Qwen3.8", () => {
    const set = resolveModelModes(undefined, "Qwen3-8B-Q4_K_M.gguf");
    assert.ok(set);
    assert.equal(set.familyLabel, "Qwen3");
    assert.ok(set.modes["Think (Coding)"]);
    assert.equal(set.modes["Think (XHigh)"], undefined);
  });

  it("gives Glimmer Low/Medium/High/XHigh with no No Think", () => {
    const set = resolveModelModes(
      denseCaps({
        architecture: "muse-glimmer",
        name: "Muse Glimmer 30B",
        path: "/models/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
      }),
      "muse-glimmer-30B"
    );
    assert.ok(set);
    assert.equal(set.familyLabel, "Glimmer");
    assert.equal(set.defaultMode, "Think (High)");
    assert.deepEqual(Object.keys(set.modes), [
      "Think (XHigh)",
      "Think (High)",
      "Think (Medium)",
      "Think (Low)",
    ]);
    assert.equal(set.modes["Think (High)"].chat_template_kwargs?.reasoning_strength, "high");
    assert.equal(set.modes["Think (XHigh)"].top_k, 64);
    assert.equal(set.modes["No Think"], undefined);
  });

  it("gives Gemma 4 Think / No Think and ignores Gemma 3", () => {
    const gemma4 = resolveModelModes(
      denseCaps({
        architecture: "gemma4",
        path: "/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
      }),
      "gemma-4-12B-it"
    );
    assert.ok(gemma4);
    assert.equal(gemma4.familyLabel, "Gemma 4");
    assert.equal(gemma4.defaultMode, "Think");
    assert.equal(gemma4.modes.Think.chat_template_kwargs?.enable_thinking, true);
    assert.equal(gemma4.modes["No Think"].chat_template_kwargs?.enable_thinking, false);
    assert.equal(gemma4.modes.Think.top_k, 64);

    assert.equal(
      resolveModelModes(denseCaps({ architecture: "gemma3", path: "/models/gemma-3-12b-it.gguf" })),
      undefined
    );
    assert.equal(resolveModelModes(undefined, "mtp-gemma-4-12B-it.gguf"), undefined);
  });
});

describe("applyModeToRequestBody", () => {
  it("puts Qwen3.8 reasoning_effort on chat_template_kwargs for llama-server", () => {
    const set = resolveModelModes(undefined, "Qwen3.8-27B");
    assert.ok(set);
    const { params } = resolveModeParams(set, { reasoningEffort: "Think (Medium)" });
    const body: Record<string, unknown> = { temperature: 0.2 };
    applyModeToRequestBody(body, params);
    assert.equal(body.temperature, 1.0);
    assert.deepEqual(body.chat_template_kwargs, {
      enable_thinking: true,
      preserve_thinking: true,
      reasoning_effort: "medium",
    });
  });

  it("puts Glimmer reasoning_strength on chat_template_kwargs", () => {
    const set = resolveModelModes(undefined, "Muse-Glimmer-30B");
    assert.ok(set);
    const { params } = resolveModeParams(set, { reasoningEffort: "Think (Low)" });
    const body: Record<string, unknown> = {};
    applyModeToRequestBody(body, params);
    assert.equal(body.top_k, 64);
    assert.deepEqual(body.chat_template_kwargs, { reasoning_strength: "low" });
  });

  it("puts Gemma 4 enable_thinking on chat_template_kwargs", () => {
    const set = resolveModelModes(undefined, "gemma-4-12B-it");
    assert.ok(set);
    const { params } = resolveModeParams(set, { reasoningEffort: "No Think" });
    const body: Record<string, unknown> = {};
    applyModeToRequestBody(body, params);
    assert.equal(body.top_k, 64);
    assert.deepEqual(body.chat_template_kwargs, { enable_thinking: false });
  });
});

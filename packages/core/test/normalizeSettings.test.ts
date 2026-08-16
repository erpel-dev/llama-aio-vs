import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  DEFAULT_LOAD_SETTINGS,
  DEFAULT_REQUEST_SETTINGS,
  normalizeLoadSettings,
  normalizeRequestSettings,
} from "../src/types";

describe("normalizeLoadSettings", () => {
  it("returns the defaults for empty or missing input", () => {
    assert.deepEqual(normalizeLoadSettings(undefined), DEFAULT_LOAD_SETTINGS);
    assert.deepEqual(normalizeLoadSettings({}), DEFAULT_LOAD_SETTINGS);
  });

  it("keeps every numeric field finite whatever it is fed", () => {
    const junk = {
      contextLength: NaN,
      gpuOffload: undefined,
      cpuThreads: null,
      evalBatchSize: "abc",
      physicalBatchSize: Infinity,
      maxConcurrentPredictions: -5,
      nCpuMoe: NaN,
      contextCheckpoints: -1,
      cacheReuse: NaN,
      reasoningBudget: NaN,
      maxDraftTokens: "x",
      draftProbability: NaN,
    } as never;
    const s = normalizeLoadSettings(junk);
    for (const [key, value] of Object.entries(s)) {
      if (typeof value === "number") {
        assert.ok(Number.isFinite(value), `${key} is ${value}`);
      }
    }
  });

  it("clamps an out-of-range number to the nearest legal value", () => {
    assert.equal(normalizeLoadSettings({ physicalBatchSize: 0 }).physicalBatchSize, 32);
    assert.equal(normalizeLoadSettings({ contextLength: 0 }).contextLength, 512);
  });

  it("treats a missing value as absent rather than as zero", () => {
    // Number(null) is 0, which would silently become the minimum (-t 1) instead
    // of the configured default.
    const s = normalizeLoadSettings({
      cpuThreads: null as unknown as number,
      evalBatchSize: undefined,
      contextLength: "" as unknown as number,
    });
    assert.equal(s.cpuThreads, DEFAULT_LOAD_SETTINGS.cpuThreads);
    assert.equal(s.evalBatchSize, DEFAULT_LOAD_SETTINGS.evalBatchSize);
    assert.equal(s.contextLength, DEFAULT_LOAD_SETTINGS.contextLength);
  });

  it("never lets the physical batch exceed the logical batch", () => {
    const s = normalizeLoadSettings({ evalBatchSize: 512, physicalBatchSize: 4096 });
    assert.equal(s.physicalBatchSize, 512);
  });

  it("collapses non-finite nullable fields to auto", () => {
    const s = normalizeLoadSettings({
      ropeFreqBase: NaN,
      ropeFreqScale: undefined,
      seed: NaN as unknown as number,
    });
    assert.equal(s.ropeFreqBase, null);
    assert.equal(s.ropeFreqScale, null);
    assert.equal(s.seed, -1);
  });

  it("preserves explicit auto/random selections", () => {
    const s = normalizeLoadSettings({ ropeFreqBase: null, ropeFreqScale: null, seed: null });
    assert.equal(s.ropeFreqBase, null);
    assert.equal(s.ropeFreqScale, null);
    assert.equal(s.seed, null);
  });

  it("falls back on unknown enum values", () => {
    const s = normalizeLoadSettings({
      cacheTypeK: "q3_k" as never,
      flashAttention: "yes" as never,
      reasoningFormat: "chatml" as never,
      speculativeMode: "eagle" as never,
    });
    assert.equal(s.cacheTypeK, "q8_0");
    assert.equal(s.flashAttention, "auto");
    assert.equal(s.reasoningFormat, "deepseek-legacy");
    assert.equal(s.speculativeMode, "off");
  });

  it("keeps valid values untouched", () => {
    const wanted = {
      contextLength: 32768,
      physicalBatchSize: 1024,
      cacheTypeK: "f16" as const,
      cacheTypeV: "q4_0" as const,
      flashAttention: "on" as const,
      reasoningBudget: 2048,
      seed: 42,
      ropeFreqBase: 1000000,
    };
    const s = normalizeLoadSettings(wanted);
    for (const [k, v] of Object.entries(wanted)) {
      assert.equal(s[k as keyof typeof s], v, k);
    }
  });

  it("trims the vision projector path", () => {
    assert.equal(normalizeLoadSettings({ mmprojPath: "  /m/mmproj-F16.gguf  " }).mmprojPath, "/m/mmproj-F16.gguf");
    assert.equal(normalizeLoadSettings({ mmprojPath: undefined }).mmprojPath, "");
  });

  it("defaults vision GPU offload on and coerces the flag", () => {
    assert.equal(normalizeLoadSettings({}).mmprojOffloadToGpu, true);
    assert.equal(normalizeLoadSettings({ mmprojOffloadToGpu: false }).mmprojOffloadToGpu, false);
    assert.equal(
      normalizeLoadSettings({ mmprojOffloadToGpu: "false" as unknown as boolean }).mmprojOffloadToGpu,
      false
    );
  });

  it("treats reasoningBudget -1 as a valid unlimited marker", () => {
    assert.equal(normalizeLoadSettings({ reasoningBudget: -1 }).reasoningBudget, -1);
    assert.equal(normalizeLoadSettings({ reasoningBudget: -99 }).reasoningBudget, -1);
  });

  it("coerces string booleans instead of treating any non-empty string as true", () => {
    const s = normalizeLoadSettings({
      offloadKvCacheToGpu: "false" as unknown as boolean,
      keepModelInMemory: "true" as unknown as boolean,
      tryMmap: "0" as unknown as boolean,
      unifiedKvCache: "yes" as unknown as boolean,
    });
    assert.equal(s.offloadKvCacheToGpu, false);
    assert.equal(s.keepModelInMemory, true);
    assert.equal(s.tryMmap, false);
    assert.equal(s.unifiedKvCache, true);
  });

  it("normalizes tensor-split / split-mode / main-gpu", () => {
    assert.equal(normalizeLoadSettings({ tensorSplit: "3, 1" }).tensorSplit, "3,1");
    assert.equal(normalizeLoadSettings({ tensorSplit: "nope" }).tensorSplit, "");
    assert.equal(normalizeLoadSettings({ splitMode: "row" }).splitMode, "row");
    assert.equal(normalizeLoadSettings({ splitMode: "nope" as never }).splitMode, "layer");
    assert.equal(normalizeLoadSettings({ mainGpu: 2 }).mainGpu, 2);
    assert.equal(normalizeLoadSettings({ mainGpu: -1 }).mainGpu, 0);
    assert.equal(normalizeLoadSettings({ mainGpu: 99 }).mainGpu, 7);
  });
});

describe("normalizeRequestSettings", () => {
  it("returns defaults for junk", () => {
    assert.deepEqual(
      normalizeRequestSettings({ temperature: NaN, topP: undefined, topK: "x" } as never),
      DEFAULT_REQUEST_SETTINGS
    );
  });

  it("clamps to valid sampling ranges", () => {
    const s = normalizeRequestSettings({ temperature: 9, topP: 5, topK: -3, maxTokens: 1 });
    assert.equal(s.temperature, 2);
    assert.equal(s.topP, 1);
    assert.equal(s.topK, 0);
    assert.equal(s.maxTokens, 16);
  });
});

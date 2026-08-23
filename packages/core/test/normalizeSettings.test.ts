import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  DEFAULT_LOAD_SETTINGS,
  DEFAULT_REQUEST_SETTINGS,
  normalizeLoadSettings,
  normalizeRequestSettings,
  recommendedMaxDraftTokens,
  recommendedNgramSizes,
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
    assert.equal(normalizeLoadSettings({ splitMode: "tensor" }).splitMode, "tensor");
    assert.equal(normalizeLoadSettings({ splitMode: "nope" as never }).splitMode, "layer");
    assert.equal(normalizeLoadSettings({ mainGpu: 2 }).mainGpu, 2);
    assert.equal(normalizeLoadSettings({ mainGpu: -1 }).mainGpu, 0);
    assert.equal(normalizeLoadSettings({ mainGpu: 99 }).mainGpu, 7);
  });

  it("accepts ngram as a speculative mode and normalizes its knobs", () => {
    const s = normalizeLoadSettings({
      speculativeMode: "ngram",
      ngramVariant: "map-k",
      ngramSizeN: 24,
      ngramSizeM: 96,
      ngramMinHits: 3,
    });
    assert.equal(s.speculativeMode, "ngram");
    assert.equal(s.ngramVariant, "map-k");
    assert.equal(s.ngramSizeN, 24);
    assert.equal(s.ngramSizeM, 96);
    assert.equal(s.ngramMinHits, 3);
    // Defaults for a hand-edited config that predates n-gram support.
    assert.equal(DEFAULT_LOAD_SETTINGS.ngramVariant, "simple");
    assert.equal(DEFAULT_LOAD_SETTINGS.ngramSizeN, 12);
    assert.equal(DEFAULT_LOAD_SETTINGS.ngramSizeM, 48);
    assert.equal(DEFAULT_LOAD_SETTINGS.ngramMinHits, 1);
  });

  it("accepts stacked n-gram speculative modes", () => {
    assert.equal(normalizeLoadSettings({ speculativeMode: "ngram-mtp" }).speculativeMode, "ngram-mtp");
    assert.equal(
      normalizeLoadSettings({ speculativeMode: "ngram-dflash" }).speculativeMode,
      "ngram-dflash"
    );
  });

  it("falls back on an unknown n-gram variant and clamps knob ranges", () => {
    const s = normalizeLoadSettings({
      ngramVariant: "eagle" as never,
      ngramSizeN: 0,
      ngramSizeM: -5,
      ngramMinHits: 999,
    });
    assert.equal(s.ngramVariant, "simple");
    assert.equal(s.ngramSizeN, 2);
    assert.equal(s.ngramSizeM, 2);
    assert.equal(s.ngramMinHits, 64);
  });
});

describe("recommendedMaxDraftTokens", () => {
  it("bumps MTP leftovers to 15 for DFlash", () => {
    assert.equal(recommendedMaxDraftTokens("dflash", 2), 15);
    assert.equal(recommendedMaxDraftTokens("ngram-dflash", 4), 15);
    assert.equal(recommendedMaxDraftTokens("dflash", 12), 12);
  });

  it("replaces a DFlash leftover when switching to MTP", () => {
    assert.equal(recommendedMaxDraftTokens("mtp", 15), 2);
    assert.equal(recommendedMaxDraftTokens("ngram-mtp", 15, true), 4);
    assert.equal(recommendedMaxDraftTokens("mtp", 3), 3);
  });

  it("uses 4 for sidecar MTP instead of the stock 2", () => {
    assert.equal(recommendedMaxDraftTokens("mtp", 2, true), 4);
    assert.equal(recommendedMaxDraftTokens("mtp", 2, false), 2);
  });
});

describe("recommendedNgramSizes", () => {
  it("swaps simple 12/48 with mod 24/64 when the values are still defaults", () => {
    assert.deepEqual(recommendedNgramSizes("mod", 12, 48), { ngramSizeN: 24, ngramSizeM: 64 });
    assert.deepEqual(recommendedNgramSizes("simple", 24, 64), { ngramSizeN: 12, ngramSizeM: 48 });
  });

  it("leaves custom sizes alone", () => {
    assert.deepEqual(recommendedNgramSizes("mod", 16, 32), { ngramSizeN: 16, ngramSizeM: 32 });
    assert.deepEqual(recommendedNgramSizes("simple", 16, 32), { ngramSizeN: 16, ngramSizeM: 32 });
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

  it("defaults the new sampler knobs to llama-safe disabled values", () => {
    // llama-server's own defaults (min_p 0.05, repeat-last-n 64) don't match
    // current instruct/coder families — Llama AIO ships them off instead.
    const d = normalizeRequestSettings(undefined);
    assert.equal(d.minP, 0);
    assert.equal(d.presencePenalty, 0);
    assert.equal(d.frequencyPenalty, 0);
    assert.equal(d.repeatPenalty, 1);
  });

  it("clamps min-p and the penalties into legal ranges", () => {
    const s = normalizeRequestSettings({
      minP: 7,
      presencePenalty: -9,
      frequencyPenalty: 99,
      repeatPenalty: 42,
    });
    assert.equal(s.minP, 1);
    assert.equal(s.presencePenalty, -2);
    assert.equal(s.frequencyPenalty, 2);
    assert.equal(s.repeatPenalty, 2);
    // Junk falls back to defaults rather than becoming NaN.
    const junk = normalizeRequestSettings({
      minP: "x",
      repeatPenalty: null,
    } as never);
    assert.equal(junk.minP, 0);
    assert.equal(junk.repeatPenalty, 1);
  });
});

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { buildServerArgs, serverConfigFingerprint } from "../serverArgs";
import { DEFAULT_LOAD_SETTINGS } from "../types";
import { argValue, loadSettings } from "./helpers";

const MODEL = "/models/test.gguf";
const build = (over = {}) => buildServerArgs(MODEL, "127.0.0.1", 8742, loadSettings(over));

describe("buildServerArgs", () => {
  it("emits the core flags from defaults", () => {
    const args = build();
    assert.equal(argValue(args, "-m"), MODEL);
    assert.equal(argValue(args, "--host"), "127.0.0.1");
    assert.equal(argValue(args, "--port"), "8742");
    assert.equal(argValue(args, "--ctx-size"), String(DEFAULT_LOAD_SETTINGS.contextLength));
    assert.equal(argValue(args, "-np"), "1");
  });

  it("never emits a non-finite or empty value", () => {
    // Every flag value must be something llama-server can parse. This is the
    // regression guard for corrupted persisted state reaching the CLI.
    const args = build({
      contextLength: NaN,
      gpuOffload: undefined as unknown as number,
      cpuThreads: null as unknown as number,
      evalBatchSize: 0,
      physicalBatchSize: Number.POSITIVE_INFINITY,
      ropeFreqBase: NaN,
      seed: Number.NaN,
      maxDraftTokens: NaN,
    });
    for (const arg of args) {
      assert.ok(arg.length > 0, `empty argument in: ${args.join(" ")}`);
      assert.ok(!/^(NaN|undefined|null|Infinity|-Infinity)$/.test(arg), `bad value "${arg}"`);
    }
  });

  describe("--cache-reuse", () => {
    it("is emitted when positive", () => {
      assert.equal(argValue(build({ cacheReuse: 256 }), "--cache-reuse"), "256");
    });
    it("is omitted when zero", () => {
      assert.ok(!build({ cacheReuse: 0 }).includes("--cache-reuse"));
    });
  });

  describe("--ctx-checkpoints", () => {
    it("is omitted at the llama.cpp default so older builds still start", () => {
      assert.ok(!build({ contextCheckpoints: 32 }).includes("--ctx-checkpoints"));
    });
    it("is emitted when changed", () => {
      assert.equal(argValue(build({ contextCheckpoints: 8 }), "--ctx-checkpoints"), "8");
    });
  });

  describe("--flash-attn", () => {
    it("is omitted on auto", () => {
      assert.ok(!build({ flashAttention: "auto" }).includes("--flash-attn"));
    });
    it("is emitted when forced", () => {
      assert.equal(argValue(build({ flashAttention: "on" }), "--flash-attn"), "on");
      assert.equal(argValue(build({ flashAttention: "off" }), "--flash-attn"), "off");
    });
  });

  describe("--reasoning-budget", () => {
    it("is omitted when unlimited", () => {
      assert.ok(!build({ reasoningBudget: -1 }).includes("--reasoning-budget"));
    });
    it("is emitted for zero and positive budgets", () => {
      assert.equal(argValue(build({ reasoningBudget: 0 }), "--reasoning-budget"), "0");
      assert.equal(argValue(build({ reasoningBudget: 2048 }), "--reasoning-budget"), "2048");
    });
  });

  it("keeps deepseek-legacy as the reasoning format by default", () => {
    assert.equal(argValue(build(), "--reasoning-format"), "deepseek-legacy");
    assert.equal(
      argValue(build({ reasoningFormat: "deepseek" }), "--reasoning-format"),
      "deepseek"
    );
  });

  it("keeps -b >= -ub so llama-server accepts the batch pair", () => {
    const args = build({ evalBatchSize: 2048, physicalBatchSize: 1024 });
    assert.ok(Number(argValue(args, "-b")) >= Number(argValue(args, "-ub")));
  });

  it("passes --no-kv-unified only when unified is off", () => {
    assert.ok(build({ unifiedKvCache: true }).includes("--kv-unified"));
    assert.ok(build({ unifiedKvCache: false }).includes("--no-kv-unified"));
  });

  it("only emits MTP flags in mtp mode", () => {
    assert.ok(!build({ speculativeMode: "off" }).includes("--spec-type"));
    assert.equal(argValue(build({ speculativeMode: "mtp" }), "--spec-type"), "draft-mtp");
  });
});

describe("serverConfigFingerprint", () => {
  it("changes when a restart-relevant setting changes", () => {
    const base = serverConfigFingerprint(MODEL, loadSettings());
    const changed: Array<[string, Record<string, unknown>]> = [
      ["contextLength", { contextLength: 32768 }],
      ["cacheTypeK", { cacheTypeK: "f16" }],
      ["flashAttention", { flashAttention: "on" }],
      ["reasoningFormat", { reasoningFormat: "none" }],
      ["reasoningBudget", { reasoningBudget: 512 }],
      ["cacheReuse", { cacheReuse: 0 }],
      ["contextCheckpoints", { contextCheckpoints: 8 }],
    ];
    for (const [label, patch] of changed) {
      assert.notEqual(
        serverConfigFingerprint(MODEL, loadSettings(patch)),
        base,
        `${label} should invalidate the fingerprint`
      );
    }
  });

  it("is stable for identical settings and sensitive to the launch mode", () => {
    assert.equal(
      serverConfigFingerprint(MODEL, loadSettings()),
      serverConfigFingerprint(MODEL, loadSettings())
    );
    assert.notEqual(
      serverConfigFingerprint(MODEL, loadSettings(), "background"),
      serverConfigFingerprint(MODEL, loadSettings(), "terminal")
    );
  });
});

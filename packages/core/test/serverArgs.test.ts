import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { buildServerArgs, normalizeLoadSettingsForCpuBackend, serverConfigFingerprint } from "../src/serverArgs";
import { DEFAULT_LOAD_SETTINGS } from "../src/types";
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

  it("emits DFlash flags when mode is dflash and a draft model is set", () => {
    const draft = "/models/Qwen3-4B-DFlash.gguf";
    const args = build({
      speculativeMode: "dflash",
      draftModelPath: draft,
      maxDraftTokens: 15,
      draftGpuOffload: 99,
    });
    assert.equal(argValue(args, "--spec-type"), "draft-dflash");
    assert.equal(argValue(args, "--model-draft"), draft);
    assert.equal(argValue(args, "--spec-draft-n-max"), "15");
    assert.equal(argValue(args, "--spec-draft-ngl"), "99");
    assert.equal(argValue(args, "--cache-type-k-draft"), "f16");
    assert.equal(argValue(args, "--cache-type-v-draft"), "f16");
    assert.equal(argValue(args, "--fit"), "off");
  });

  it("defaults DFlash --spec-draft-n-max to 15 when maxDraftTokens is 0", () => {
    const args = build({
      speculativeMode: "dflash",
      draftModelPath: "/models/draft.gguf",
      maxDraftTokens: 0,
    });
    assert.equal(argValue(args, "--spec-draft-n-max"), "15");
  });

  it("does not emit DFlash without a draft model path", () => {
    const args = build({ speculativeMode: "dflash", draftModelPath: "" });
    assert.ok(!args.includes("--spec-type"));
    assert.ok(!args.includes("--model-draft"));
    assert.ok(!args.includes("--fit"));
  });

  describe("multi-GPU split flags", () => {
    it("omits split flags on defaults (single GPU)", () => {
      const args = build();
      assert.ok(!args.includes("--tensor-split"));
      assert.ok(!args.includes("--split-mode"));
      assert.ok(!args.includes("--main-gpu"));
    });

    it("emits tensor-split, split-mode, and main-gpu together", () => {
      const args = build({ tensorSplit: "3,1", splitMode: "layer", mainGpu: 0 });
      assert.equal(argValue(args, "--tensor-split"), "3,1");
      assert.equal(argValue(args, "--split-mode"), "layer");
      assert.equal(argValue(args, "--main-gpu"), "0");
    });

    it("emits split-mode row without a tensor-split", () => {
      const args = build({ splitMode: "row" });
      assert.equal(argValue(args, "--split-mode"), "row");
      assert.ok(!args.includes("--tensor-split"));
    });

    it("emits main-gpu when it is not device 0", () => {
      const args = build({ mainGpu: 1 });
      assert.equal(argValue(args, "--main-gpu"), "1");
    });

    it("does not emit split flags when GPU offload is 0", () => {
      const args = build({ gpuOffload: 0, tensorSplit: "3,1", mainGpu: 1, splitMode: "row" });
      assert.ok(!args.includes("--tensor-split"));
      assert.ok(!args.includes("--split-mode"));
      assert.ok(!args.includes("--main-gpu"));
    });

    it("split-mode none drops tensor-split and pins --device to the main GPU", () => {
      const gpus = [
        { totalBytes: 16 * 1024 ** 3, source: "test", llamaDeviceId: "Vulkan0", name: "9070" },
        { totalBytes: 16 * 1024 ** 3, source: "test", llamaDeviceId: "Vulkan1", name: "9060" },
      ];
      const args = buildServerArgs(
        MODEL,
        "127.0.0.1",
        8742,
        loadSettings({ splitMode: "none", tensorSplit: "90,10", mainGpu: 0 }),
        { gpus }
      );
      assert.ok(!args.includes("--tensor-split"));
      assert.equal(argValue(args, "--split-mode"), "none");
      assert.equal(argValue(args, "--device"), "Vulkan0");
      assert.ok(!args.includes("--main-gpu"));
    });

    it("split-mode none on GPU 1 uses that card's llama.cpp id", () => {
      const gpus = [
        { totalBytes: 16 * 1024 ** 3, source: "test", llamaDeviceId: "Vulkan0" },
        { totalBytes: 16 * 1024 ** 3, source: "test", llamaDeviceId: "Vulkan1" },
      ];
      const args = buildServerArgs(
        MODEL,
        "127.0.0.1",
        8742,
        loadSettings({ splitMode: "none", tensorSplit: "50,50", mainGpu: 1 }),
        { gpus }
      );
      assert.equal(argValue(args, "--device"), "Vulkan1");
      assert.ok(!args.includes("--tensor-split"));
    });

    it("split-mode none without device ids still omits tensor-split", () => {
      const args = build({ splitMode: "none", tensorSplit: "75,25", mainGpu: 1 });
      assert.ok(!args.includes("--tensor-split"));
      assert.equal(argValue(args, "--split-mode"), "none");
      assert.equal(argValue(args, "--main-gpu"), "1");
      assert.ok(!args.includes("--device"));
    });
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
      ["mmprojPath", { mmprojPath: "/models/mmproj-F16.gguf" }],
      ["mmprojOffloadToGpu", { mmprojOffloadToGpu: false }],
      ["tensorSplit", { tensorSplit: "3,1" }],
      ["splitMode", { splitMode: "row" }],
      ["mainGpu", { mainGpu: 1 }],
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

  it("treats DFlash maxDraftTokens 0 the same as the CLI default of 15", () => {
    const zero = serverConfigFingerprint(
      MODEL,
      loadSettings({ speculativeMode: "dflash", draftModelPath: "/d.gguf", maxDraftTokens: 0 })
    );
    const fifteen = serverConfigFingerprint(
      MODEL,
      loadSettings({ speculativeMode: "dflash", draftModelPath: "/d.gguf", maxDraftTokens: 15 })
    );
    assert.equal(zero, fifteen);
  });

  it("ignores draft-only knobs unless speculative mode uses them", () => {
    const off = serverConfigFingerprint(MODEL, loadSettings());
    assert.equal(
      serverConfigFingerprint(
        MODEL,
        loadSettings({ draftModelPath: "/other.gguf", draftGpuOffload: 12, maxDraftTokens: 8 })
      ),
      off
    );
    const mtp = serverConfigFingerprint(MODEL, loadSettings({ speculativeMode: "mtp", maxDraftTokens: 2 }));
    assert.equal(
      serverConfigFingerprint(
        MODEL,
        loadSettings({
          speculativeMode: "mtp",
          maxDraftTokens: 2,
          draftModelPath: "/other.gguf",
          draftGpuOffload: 12,
        })
      ),
      mtp
    );
  });
});

describe("normalizeLoadSettingsForCpuBackend", () => {
  it("zeros GPU offload, KV GPU offload, vision GPU offload, and n-cpu-moe", () => {
    const normalized = normalizeLoadSettingsForCpuBackend(
      loadSettings({ gpuOffload: 40, offloadKvCacheToGpu: true, mmprojOffloadToGpu: true, nCpuMoe: 12 })
    );
    assert.equal(normalized.gpuOffload, 0);
    assert.equal(normalized.offloadKvCacheToGpu, false);
    assert.equal(normalized.mmprojOffloadToGpu, false);
    assert.equal(normalized.nCpuMoe, 0);
  });

  it("makes dirty-fingerprint agree with what a CPU start would ship", () => {
    // Stored UI may still show leftover MoE / ngl knobs on a CPU backend.
    const stored = loadSettings({ gpuOffload: 40, offloadKvCacheToGpu: true, nCpuMoe: 12 });
    const launched = normalizeLoadSettingsForCpuBackend(stored);
    assert.equal(
      serverConfigFingerprint(MODEL, launched),
      serverConfigFingerprint(MODEL, normalizeLoadSettingsForCpuBackend(loadSettings({ nCpuMoe: 0 })))
    );
    assert.notEqual(
      serverConfigFingerprint(MODEL, stored),
      serverConfigFingerprint(MODEL, launched),
      "without CPU normalize, leftover nCpuMoe would falsely look dirty"
    );
  });
});

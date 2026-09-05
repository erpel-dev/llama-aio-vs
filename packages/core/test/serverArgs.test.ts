import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { buildServerArgs, normalizeLoadSettingsForCpuBackend, serverConfigFingerprint } from "../src/serverArgs";
import { DEFAULT_LOAD_SETTINGS } from "../src/types";
import { argValue, argValues, denseCaps, loadSettings, moeCaps } from "./helpers";

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

  describe("--lazy-mode", () => {
    it("is omitted on auto so older builds still start", () => {
      assert.ok(!build({ lazyMode: "auto" }).includes("--lazy-mode"));
    });
    it("is emitted when forced", () => {
      assert.equal(argValue(build({ lazyMode: "on" }), "--lazy-mode"), "on");
      assert.equal(argValue(build({ lazyMode: "off" }), "--lazy-mode"), "off");
    });
    it("keeps mmap with --n-cpu-moe when lazy is on", () => {
      const args = build({ nCpuMoe: 6, tryMmap: true, lazyMode: "on" });
      assert.equal(argValue(args, "--n-cpu-moe"), "6");
      assert.equal(argValue(args, "--load-mode"), "mmap");
      assert.equal(argValue(args, "--lazy-mode"), "on");
    });
    it("keeps mmap with --n-cpu-moe when auto lazy has a large PLE table", () => {
      const args = buildServerArgs(
        MODEL,
        "127.0.0.1",
        8742,
        loadSettings({ nCpuMoe: 24, tryMmap: true, lazyMode: "auto" }),
        { caps: moeCaps({ architecture: "qwen4exp", fileSizeBytes: 72 * 1024 ** 3, pleShare: 0.4 }) }
      );
      assert.equal(argValue(args, "--load-mode"), "mmap");
      assert.ok(!args.includes("--lazy-mode"));
    });
    it("still forces load-mode none for CPU MoE when lazy is off", () => {
      const args = build({ nCpuMoe: 6, tryMmap: true, lazyMode: "off" });
      assert.equal(argValue(args, "--load-mode"), "none");
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
    assert.equal(argValue(args, "--spec-draft-p-min"), "0.75");
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
  });

  describe("n-gram speculative decoding", () => {
    it("emits ngram-simple with lookup/draft sizes and min-hits", () => {
      const args = build({
        speculativeMode: "ngram",
        ngramVariant: "simple",
        ngramSizeN: 12,
        ngramSizeM: 48,
        ngramMinHits: 2,
      });
      assert.equal(argValue(args, "--spec-type"), "ngram-simple");
      assert.equal(argValue(args, "--spec-ngram-simple-size-n"), "12");
      assert.equal(argValue(args, "--spec-ngram-simple-size-m"), "48");
      assert.equal(argValue(args, "--spec-ngram-simple-min-hits"), "2");
    });

    it("never emits draft-model flags (no draft GGUF needed)", () => {
      const args = build({
        speculativeMode: "ngram",
        draftModelPath: "/models/leftover-draft.gguf",
      });
      assert.ok(!args.includes("--model-draft"));
      assert.ok(!args.includes("--spec-draft-ngl"));
      assert.ok(!args.includes("--cache-type-k-draft"));
    });

    it("maps each variant to its llama.cpp flags", () => {
      for (const variant of ["map-k", "map-k4v"] as const) {
        const args = build({ speculativeMode: "ngram", ngramVariant: variant });
        assert.equal(argValue(args, "--spec-type"), `ngram-${variant}`);
        assert.equal(argValue(args, `--spec-ngram-${variant}-size-n`), "12");
      }
      const mod = build({ speculativeMode: "ngram", ngramVariant: "mod" });
      assert.equal(argValue(mod, "--spec-type"), "ngram-mod");
      assert.equal(argValue(mod, "--spec-ngram-mod-n-match"), "12");
      assert.equal(argValue(mod, "--spec-ngram-mod-n-max"), "48");
      assert.ok(!mod.includes("--spec-ngram-mod-size-n"));
    });

    it("keeps the draft m-gram >= the lookup n-gram", () => {
      const args = build({
        speculativeMode: "ngram",
        ngramSizeN: 64,
        ngramSizeM: 16,
      });
      assert.equal(argValue(args, "--spec-ngram-simple-size-m"), "64");
    });

    it("is omitted in every other mode", () => {
      for (const mode of ["off", "mtp", "dflash"] as const) {
        const args = build({
          speculativeMode: mode,
          draftModelPath: "/models/draft.gguf",
        });
        assert.ok(!args.join(" ").includes("ngram"), `mode ${mode} leaked an ngram flag`);
      }
    });

    it("stacks with MTP: n-gram lookup first, then draft-mtp", () => {
      const args = build({
        speculativeMode: "ngram-mtp",
        ngramVariant: "simple",
        ngramSizeN: 12,
        maxDraftTokens: 2,
      });
      assert.deepEqual(argValues(args, "--spec-type"), ["ngram-simple", "draft-mtp"]);
      assert.equal(argValue(args, "--spec-ngram-simple-size-n"), "12");
      assert.equal(argValue(args, "--spec-draft-n-max"), "2");
      assert.ok(!args.includes("--model-draft"));
    });

    it("stacks with DFlash: n-gram lookup first, then draft-dflash", () => {
      const draft = "/models/Qwen3-4B-DFlash.gguf";
      const args = build({
        speculativeMode: "ngram-dflash",
        ngramVariant: "map-k",
        draftModelPath: draft,
        maxDraftTokens: 15,
        draftGpuOffload: 99,
      });
      assert.deepEqual(argValues(args, "--spec-type"), ["ngram-map-k", "draft-dflash"]);
      assert.equal(argValue(args, "--model-draft"), draft);
      assert.equal(argValue(args, "--cache-type-k-draft"), "f16");
    });

    it("still emits n-gram flags when stacked DFlash has no draft path yet", () => {
      const args = build({ speculativeMode: "ngram-dflash", draftModelPath: "" });
      assert.deepEqual(argValues(args, "--spec-type"), ["ngram-simple"]);
      assert.ok(!args.includes("--model-draft"));
    });
  });

  describe("server-side sampling defaults", () => {
    it("are omitted when no request sampling is provided", () => {
      const args = build();
      for (const flag of [
        "--temp",
        "--top-p",
        "--top-k",
        "--min-p",
        "--presence-penalty",
        "--frequency-penalty",
        "--repeat-penalty",
      ]) {
        assert.ok(!args.includes(flag), `${flag} should not ship without requestSampling`);
      }
    });

    it("ship the request defaults as CLI flags when provided", () => {
      const args = buildServerArgs(MODEL, "127.0.0.1", 8742, loadSettings(), {
        requestSampling: {
          temperature: 0.5,
          topP: 0.95,
          topK: 20,
          maxTokens: 8192,
          minP: 0,
          presencePenalty: 0,
          frequencyPenalty: 0,
          repeatPenalty: 1,
        },
      });
      assert.equal(argValue(args, "--temp"), "0.5");
      assert.equal(argValue(args, "--top-p"), "0.95");
      assert.equal(argValue(args, "--top-k"), "20");
      // Disabled values are shipped explicitly so raw API clients never
      // inherit llama-server's built-in min_p 0.05 / top_k 40.
      assert.equal(argValue(args, "--min-p"), "0");
      assert.equal(argValue(args, "--presence-penalty"), "0");
      assert.equal(argValue(args, "--frequency-penalty"), "0");
      assert.equal(argValue(args, "--repeat-penalty"), "1");
    });
  });

  it("always disables llama.cpp auto-fit because -ngl is user-set", () => {
    assert.equal(argValue(build(), "--fit"), "off");
    assert.equal(
      argValue(build({ nCpuMoe: 6, tensorSplit: "45,55", gpuOffload: 41 }), "--fit"),
      "off"
    );
  });

  it("pins PLE n-gram tables to CPU for qwen4exp", () => {
    const args = buildServerArgs(MODEL, "127.0.0.1", 8742, loadSettings(), {
      caps: moeCaps({ architecture: "qwen4exp", pleShare: 0.4 }),
    });
    assert.equal(argValue(args, "--override-tensor"), "per_layer_token_embd.=CPU");
  });

  it("does not emit a PLE override for ordinary models", () => {
    assert.ok(!build().includes("--override-tensor"));
    const args = buildServerArgs(MODEL, "127.0.0.1", 8742, loadSettings(), {
      caps: denseCaps(),
    });
    assert.ok(!args.includes("--override-tensor"));
  });

  it("loads without mmap when CPU MoE overrides are in use", () => {
    const args = build({ nCpuMoe: 6, tryMmap: true });
    assert.equal(argValue(args, "--n-cpu-moe"), "6");
    assert.equal(argValue(args, "--load-mode"), "none");
  });

  it("passes --n-cpu-ffn for dense CPU offload and forces load-mode none", () => {
    const args = build({ nCpuFfn: 8, tryMmap: true });
    assert.equal(argValue(args, "--n-cpu-ffn"), "8");
    assert.equal(argValue(args, "--load-mode"), "none");
  });

  it("omits --n-cpu-ffn when unset", () => {
    assert.ok(!build({ nCpuFfn: 0 }).includes("--n-cpu-ffn"));
  });

  it("keeps mmap when CPU MoE is off and tryMmap is on", () => {
    assert.equal(argValue(build({ nCpuMoe: 0, tryMmap: true }), "--load-mode"), "mmap");
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

    it("emits split-mode tensor with tensor-split and main-gpu", () => {
      const args = build({ tensorSplit: "3,1", splitMode: "tensor", mainGpu: 0 });
      assert.equal(argValue(args, "--tensor-split"), "3,1");
      assert.equal(argValue(args, "--split-mode"), "tensor");
      assert.equal(argValue(args, "--main-gpu"), "0");
    });

    it("emits bare split-mode tensor without a configured split", () => {
      const args = build({ splitMode: "tensor" });
      assert.equal(argValue(args, "--split-mode"), "tensor");
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
      ["lazyMode", { lazyMode: "on" }],
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

  it("tracks n-gram knobs only while n-gram mode is on", () => {
    const off = serverConfigFingerprint(MODEL, loadSettings());
    // Knob changes are irrelevant while off (no flags ship).
    assert.equal(
      serverConfigFingerprint(
        MODEL,
        loadSettings({ ngramVariant: "map-k", ngramSizeN: 24, ngramSizeM: 96, ngramMinHits: 3 })
      ),
      off
    );
    const ngram = serverConfigFingerprint(MODEL, loadSettings({ speculativeMode: "ngram" }));
    // Turning the mode on invalidates.
    assert.notEqual(ngram, off);
    // Every knob is restart-relevant while on.
    for (const patch of [
      { ngramVariant: "mod" as const },
      { ngramSizeN: 16 },
      { ngramSizeM: 64 },
      { ngramMinHits: 2 },
    ]) {
      assert.notEqual(
        serverConfigFingerprint(MODEL, loadSettings({ speculativeMode: "ngram", ...patch })),
        ngram,
        `ngram knob ${JSON.stringify(patch)} should invalidate the fingerprint`
      );
    }
    const stacked = serverConfigFingerprint(MODEL, loadSettings({ speculativeMode: "ngram-mtp" }));
    assert.notEqual(stacked, ngram);
    assert.notEqual(
      serverConfigFingerprint(MODEL, loadSettings({ speculativeMode: "ngram-mtp", ngramSizeN: 16 })),
      stacked
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

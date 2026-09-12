import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { after, before, describe, it } from "node:test";
import {
  capsMissDefaultSwaPattern,
  clampLoadSettingsToModel,
  defaultSwaLayout,
  heuristicPleShare,
  invalidateModelCapabilitiesCache,
  isLinearRecurrentHybrid,
  isQwen4expArchitecture,
  readModelCapabilities,
  resolveSlidingWindowPattern,
  shardFileNames,
  shouldPinPleToCpu,
  totalModelBytes,
} from "../src/ggufMetadata";
import { denseCaps, loadSettings, miniGguf, MiniGgufValue, moeCaps } from "./helpers";

describe("shardFileNames", () => {
  it("expands a split-model name into the whole set", () => {
    assert.deepEqual(shardFileNames("model-00001-of-00003.gguf"), [
      "model-00001-of-00003.gguf",
      "model-00002-of-00003.gguf",
      "model-00003-of-00003.gguf",
    ]);
  });

  it("works from any shard, not just the first", () => {
    assert.equal(shardFileNames("m-00002-of-00002.gguf")?.length, 2);
  });

  it("returns undefined for ordinary file names", () => {
    assert.equal(shardFileNames("model-Q4_K_M.gguf"), undefined);
    assert.equal(shardFileNames("model.gguf"), undefined);
    // A name that merely looks numeric must not be treated as a shard.
    assert.equal(shardFileNames("llama-3-00001.gguf"), undefined);
  });

  it("rejects an implausible shard count", () => {
    assert.equal(shardFileNames("m-00001-of-00000.gguf"), undefined);
  });
});

describe("totalModelBytes", () => {
  let dir: string;

  before(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-shards-"));
    fs.writeFileSync(path.join(dir, "solo.gguf"), Buffer.alloc(1000));
    for (let i = 1; i <= 3; i++) {
      const name = `split-0000${i}-of-00003.gguf`;
      fs.writeFileSync(path.join(dir, name), Buffer.alloc(1000));
    }
    fs.writeFileSync(path.join(dir, "partial-00001-of-00002.gguf"), Buffer.alloc(1000));
  });

  after(() => fs.rmSync(dir, { recursive: true, force: true }));

  it("returns the plain size for a single file", () => {
    const r = totalModelBytes(path.join(dir, "solo.gguf"));
    assert.equal(r.bytes, 1000);
    assert.equal(r.shardCount, 1);
  });

  it("sums every shard of a split model", () => {
    const r = totalModelBytes(path.join(dir, "split-00001-of-00003.gguf"));
    assert.equal(r.bytes, 3000, "must not report only the selected shard");
    assert.equal(r.shardCount, 3);
    assert.equal(r.shardsFound, 3);
  });

  it("gives the same total when a later shard is selected", () => {
    assert.equal(totalModelBytes(path.join(dir, "split-00003-of-00003.gguf")).bytes, 3000);
  });

  it("reports how many shards were actually found", () => {
    const r = totalModelBytes(path.join(dir, "partial-00001-of-00002.gguf"));
    assert.equal(r.shardsFound, 1);
    assert.equal(r.shardCount, 2);
    assert.equal(r.bytes, 1000);
  });

  it("does not throw for a missing file", () => {
    assert.equal(totalModelBytes(path.join(dir, "nope.gguf")).bytes, undefined);
  });
});

describe("readModelCapabilities", () => {
  it("rejects a non-GGUF file instead of parsing garbage", () => {
    const p = path.join(os.tmpdir(), `llama-aio-not-gguf-${process.pid}.gguf`);
    fs.writeFileSync(p, Buffer.from("this is not a gguf file at all"));
    try {
      assert.throws(() => readModelCapabilities(p), /GGUF/);
    } finally {
      fs.rmSync(p, { force: true });
    }
  });

  it("refuses a header claiming an absurd metadata count instead of hanging", () => {
    // Valid magic + version, then tensor/kv counts of 2^60.
    const header = Buffer.alloc(24);
    header.writeUInt32LE(0x46554747, 0);
    header.writeUInt32LE(3, 4);
    header.writeBigUInt64LE(1n << 60n, 8);
    header.writeBigUInt64LE(1n << 60n, 16);
    const p = path.join(os.tmpdir(), `llama-aio-bad-gguf-${process.pid}.gguf`);
    fs.writeFileSync(p, header);
    try {
      const started = Date.now();
      assert.throws(() => readModelCapabilities(p), /corrupt|Invalid GGUF/i);
      assert.ok(Date.now() - started < 2000, "must fail fast rather than loop");
    } finally {
      fs.rmSync(p, { force: true });
    }
  });
});

describe("clampLoadSettingsToModel", () => {
  it("caps context at the model maximum", () => {
    const s = clampLoadSettingsToModel(
      loadSettings({ contextLength: 999_999 }),
      denseCaps({ maxContextLength: 8192 })
    );
    assert.equal(s.contextLength, 8192);
  });

  it("recovers a finite context from corrupt input", () => {
    const s = clampLoadSettingsToModel(loadSettings({ contextLength: NaN }), denseCaps());
    assert.ok(Number.isFinite(s.contextLength));
  });

  it("keeps 99 as the all-layers sentinel", () => {
    assert.equal(clampLoadSettingsToModel(loadSettings({ gpuOffload: 99 }), denseCaps()).gpuOffload, 99);
  });

  it("caps an explicit layer count at the block count", () => {
    const s = clampLoadSettingsToModel(loadSettings({ gpuOffload: 60 }), denseCaps());
    assert.equal(s.gpuOffload, 48);
  });

  it("turns off MTP for models without next-n layers", () => {
    const s = clampLoadSettingsToModel(loadSettings({ speculativeMode: "mtp" }), denseCaps());
    assert.equal(s.speculativeMode, "off");
  });

  it("keeps MTP for models that support it", () => {
    const s = clampLoadSettingsToModel(
      loadSettings({ speculativeMode: "mtp" }),
      denseCaps({ nextnPredictLayers: 1 })
    );
    assert.equal(s.speculativeMode, "mtp");
  });

  it("drops MTP from ngram-mtp when the model has no next-n layers", () => {
    const s = clampLoadSettingsToModel(loadSettings({ speculativeMode: "ngram-mtp" }), denseCaps());
    assert.equal(s.speculativeMode, "ngram");
  });

  it("keeps ngram-mtp for models that support MTP", () => {
    const s = clampLoadSettingsToModel(
      loadSettings({ speculativeMode: "ngram-mtp" }),
      denseCaps({ nextnPredictLayers: 1 })
    );
    assert.equal(s.speculativeMode, "ngram-mtp");
  });

  it("zeroes --n-cpu-moe for dense models", () => {
    assert.equal(clampLoadSettingsToModel(loadSettings({ nCpuMoe: 12 }), denseCaps()).nCpuMoe, 0);
  });

  it("zeroes --n-cpu-ffn for MoE models and clamps it to the layer count for dense", () => {
    assert.equal(clampLoadSettingsToModel(loadSettings({ nCpuFfn: 12 }), moeCaps()).nCpuFfn, 0);
    const clamped = clampLoadSettingsToModel(loadSettings({ nCpuFfn: 500 }), denseCaps({ blockCount: 48 }));
    assert.equal(clamped.nCpuFfn, 48);
  });
});

describe("resolveSlidingWindowPattern", () => {
  it("expands llama.cpp scalar period 4 to SWA,SWA,SWA,dense", () => {
    const p = resolveSlidingWindowPattern(4, 8);
    assert.deepEqual(p, [true, true, true, false, true, true, true, false]);
  });

  it("tiles a 4-entry official-GGUF array across all layers", () => {
    const p = resolveSlidingWindowPattern([1, 1, 1, 0], 52);
    assert.equal(p?.length, 52);
    assert.equal(p?.filter(Boolean).length, 39);
    assert.equal(p?.[3], false);
    assert.equal(p?.[4], true);
  });

  it("treats period 0 as all SWA and period 1 as all dense", () => {
    assert.deepEqual(resolveSlidingWindowPattern(0, 3), [true, true, true]);
    assert.deepEqual(resolveSlidingWindowPattern(1, 3), [false, false, false]);
  });
});

describe("architecture SWA defaults (llama.cpp set_swa_pattern)", () => {
  const tmp: string[] = [];
  after(() => {
    for (const p of tmp) {
      fs.rmSync(p, { force: true });
    }
  });

  function capsFor(arch: string, extra: Record<string, MiniGgufValue> = {}) {
    const p = path.join(os.tmpdir(), `llama-aio-swa-${arch}-${process.pid}-${tmp.length}.gguf`);
    tmp.push(p);
    fs.writeFileSync(
      p,
      miniGguf({
        "general.architecture": { type: "string", value: arch },
        "general.name": { type: "string", value: `${arch}-test` },
        [`${arch}.block_count`]: { type: "u32", value: 12 },
        [`${arch}.context_length`]: { type: "u32", value: 131072 },
        [`${arch}.embedding_length`]: { type: "u32", value: 2048 },
        [`${arch}.attention.head_count`]: { type: "u32", value: 16 },
        [`${arch}.attention.head_count_kv`]: { type: "u32", value: 4 },
        ...extra,
      })
    );
    return readModelCapabilities(p);
  }

  it("lists the families llama.cpp hard-codes", () => {
    assert.deepEqual(defaultSwaLayout("gemma2"), { period: 2, window: 4096 });
    assert.deepEqual(defaultSwaLayout("Gemma3"), { period: 6, window: 1024 });
    assert.deepEqual(defaultSwaLayout("gpt-oss"), { period: 2, window: 128 });
    assert.deepEqual(defaultSwaLayout("llama4"), { period: 4, window: 8192 });
    assert.deepEqual(defaultSwaLayout("muse-glimmer"), { period: 4 });
    assert.equal(defaultSwaLayout("qwen3"), undefined);
    assert.equal(defaultSwaLayout(undefined), undefined);
  });

  it("gemma3: 5 of 6 layers slide over the GGUF window when the pattern key is absent", () => {
    const caps = capsFor("gemma3", {
      "gemma3.attention.sliding_window": { type: "u32", value: 1024 },
    });
    assert.equal(caps.slidingWindow, 1024);
    assert.equal(caps.slidingWindowPattern?.length, 12);
    assert.equal(caps.slidingWindowPattern?.filter(Boolean).length, 10);
    assert.equal(caps.slidingWindowPattern?.[5], false);
    assert.equal(caps.slidingWindowPattern?.[11], false);
  });

  it("gemma2 / gpt-oss: alternate SWA and full layers", () => {
    const g2 = capsFor("gemma2");
    assert.equal(g2.slidingWindow, 4096, "falls back to llama.cpp's default window");
    assert.deepEqual(g2.slidingWindowPattern?.slice(0, 4), [true, false, true, false]);
    const oss = capsFor("gpt-oss", {
      "gpt-oss.attention.sliding_window": { type: "u32", value: 128 },
    });
    assert.deepEqual(oss.slidingWindowPattern?.slice(0, 4), [true, false, true, false]);
  });

  it("llama4: chunked attention on 3 of 4 layers with the hard-coded 8192 window", () => {
    const caps = capsFor("llama4");
    assert.equal(caps.slidingWindow, 8192);
    assert.deepEqual(caps.slidingWindowPattern?.slice(0, 4), [true, true, true, false]);
  });

  it("an explicit sliding_window_pattern in the GGUF wins over the table", () => {
    const caps = capsFor("gemma3", {
      "gemma3.attention.sliding_window": { type: "u32", value: 512 },
      "gemma3.attention.sliding_window_pattern": {
        type: "bool[]",
        value: [true, false],
      },
    });
    assert.equal(caps.slidingWindow, 512);
    assert.deepEqual(caps.slidingWindowPattern?.slice(0, 4), [true, false, true, false]);
  });

  it("leaves architectures without a table entry alone", () => {
    const caps = capsFor("qwen3");
    assert.equal(caps.slidingWindow, undefined);
    assert.equal(caps.slidingWindowPattern, undefined);
    assert.equal(capsMissDefaultSwaPattern(caps), false);
  });

  it("flags stale capabilities that predate the table", () => {
    assert.equal(
      capsMissDefaultSwaPattern({ architecture: "gemma3", blockCount: 34 }),
      true
    );
    assert.equal(
      capsMissDefaultSwaPattern({
        architecture: "gemma3",
        blockCount: 34,
        slidingWindowPattern: Array.from({ length: 34 }, () => true),
      }),
      false
    );
  });
});

describe("qwen4exp / PLE helpers", () => {
  it("recognizes llama.cpp and HF architecture spellings", () => {
    assert.equal(isQwen4expArchitecture("qwen4exp"), true);
    assert.equal(isQwen4expArchitecture("qwen4_exp"), true);
    assert.equal(isQwen4expArchitecture("qwen4-exp"), true);
    assert.equal(isQwen4expArchitecture("qwen35"), false);
    assert.equal(isQwen4expArchitecture("qwen3"), false);
  });

  it("heuristically prices the PLE table only for qwen4exp", () => {
    assert.equal(heuristicPleShare("qwen4exp"), 0.4);
    assert.equal(heuristicPleShare("qwen3"), 0);
  });

  it("pins PLE to CPU when the share is known or the arch is qwen4exp", () => {
    assert.equal(shouldPinPleToCpu({ architecture: "qwen4exp" }), true);
    assert.equal(shouldPinPleToCpu({ architecture: "qwen3", pleShare: 0.3 }), true);
    assert.equal(shouldPinPleToCpu({ architecture: "qwen3" }), false);
    assert.equal(shouldPinPleToCpu(undefined), false);
  });

  it("gates the RCO graph discount on qwen35 only, never qwen4exp", () => {
    assert.equal(isLinearRecurrentHybrid({ architecture: "qwen35" }), true);
    assert.equal(isLinearRecurrentHybrid({ architecture: "qwen35moe" }), true);
    assert.equal(isLinearRecurrentHybrid({ architecture: "qwen3.5" }), true);
    assert.equal(isLinearRecurrentHybrid({ architecture: "QWEN35" }), true);
    assert.equal(isLinearRecurrentHybrid({ architecture: "qwen4exp" }), false);
    assert.equal(isLinearRecurrentHybrid({ architecture: "qwen4_exp" }), false);
    assert.equal(isLinearRecurrentHybrid({ architecture: "qwen3" }), false);
    assert.equal(isLinearRecurrentHybrid(undefined), false);
  });
});

describe("readModelCapabilities cache", () => {
  const tmp: string[] = [];
  after(() => {
    invalidateModelCapabilitiesCache();
    for (const p of tmp) {
      fs.rmSync(p, { force: true });
    }
  });

  function writeCaps(arch: string, name: string, dest: string) {
    fs.writeFileSync(
      dest,
      miniGguf({
        "general.architecture": { type: "string", value: arch },
        "general.name": { type: "string", value: name },
        [`${arch}.block_count`]: { type: "u32", value: 8 },
        [`${arch}.context_length`]: { type: "u32", value: 8192 },
        [`${arch}.embedding_length`]: { type: "u32", value: 1024 },
        [`${arch}.attention.head_count`]: { type: "u32", value: 8 },
        [`${arch}.attention.head_count_kv`]: { type: "u32", value: 2 },
      })
    );
  }

  it("returns a clone so callers cannot poison the memo", () => {
    const p = path.join(os.tmpdir(), `llama-aio-caps-clone-${process.pid}.gguf`);
    tmp.push(p);
    writeCaps("qwen3", "orig", p);
    invalidateModelCapabilitiesCache(p);
    const first = readModelCapabilities(p);
    first.name = "mutated";
    const second = readModelCapabilities(p);
    assert.equal(second.name, "orig");
    assert.notEqual(first, second);
  });

  it("re-reads after invalidation even when mtime and size stay the same", () => {
    const p = path.join(os.tmpdir(), `llama-aio-caps-inv-${process.pid}.gguf`);
    tmp.push(p);
    writeCaps("qwen3", "alpha", p);
    const stamp = Math.floor(Date.now() / 1000) - 90;
    fs.utimesSync(p, stamp, stamp);
    invalidateModelCapabilitiesCache(p);
    assert.equal(readModelCapabilities(p).name, "alpha");

    writeCaps("qwen3", "bravo", p);
    fs.utimesSync(p, stamp, stamp);
    assert.equal(readModelCapabilities(p).name, "alpha", "same mtime/size served from cache");

    invalidateModelCapabilitiesCache(p);
    assert.equal(readModelCapabilities(p).name, "bravo");
  });
});

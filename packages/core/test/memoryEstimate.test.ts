import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { computeOverheadBytes, estimateKvBytes, estimateMemory } from "../src/memoryEstimate";
import { denseCaps, GiB, loadSettings, moeCaps } from "./helpers";

const gpu = (totalGiB: number) => ({
  totalBytes: totalGiB * GiB,
  usedBytes: 0,
  name: "test",
  source: "test",
});

describe("estimateKvBytes", () => {
  const caps = denseCaps();

  it("scales linearly with context", () => {
    const at8k = estimateKvBytes(caps, 8192, "q8_0", "q8_0");
    const at16k = estimateKvBytes(caps, 16384, "q8_0", "q8_0");
    assert.equal(at16k, at8k * 2);
  });

  it("halves the K half when K drops from f16 to q8_0", () => {
    const f16 = estimateKvBytes(caps, 8192, "f16", "f16");
    const k8 = estimateKvBytes(caps, 8192, "q8_0", "f16");
    assert.equal(k8, f16 * 0.75); // K is half of a symmetric cache
  });

  it("prices q4_0 at a quarter of f16", () => {
    const f16 = estimateKvBytes(caps, 8192, "f16", "f16");
    const q4 = estimateKvBytes(caps, 8192, "q4_0", "q4_0");
    assert.equal(q4, f16 / 4);
  });

  it("caps sliding-window layers at the window size", () => {
    const swa = denseCaps({
      slidingWindow: 4096,
      slidingWindowPattern: Array.from({ length: 48 }, (_, i) => i % 2 === 0),
    });
    const full = estimateKvBytes(denseCaps(), 65536, "q8_0", "q8_0");
    assert.ok(estimateKvBytes(swa, 65536, "q8_0", "q8_0") < full);
  });

  it("skips recurrent layers in hybrid models", () => {
    const hybrid = denseCaps({
      recurrentLayers: Array.from({ length: 48 }, (_, i) => i % 4 !== 3),
    });
    const full = estimateKvBytes(denseCaps(), 32768, "q8_0", "q8_0");
    // Only every 4th layer keeps a context-scaled cache.
    assert.equal(estimateKvBytes(hybrid, 32768, "q8_0", "q8_0"), full / 4);
  });
});

describe("computeOverheadBytes", () => {
  it("grows with the physical batch, not just the logical batch", () => {
    const small = computeOverheadBytes(5120, 512, 2048);
    const bigUbatch = computeOverheadBytes(5120, 1024, 2048);
    const bigBatch = computeOverheadBytes(5120, 512, 4096);
    assert.ok(bigUbatch > small, "-ub should raise the compute buffer");
    assert.ok(bigUbatch - small > bigBatch - small, "-ub should dominate -b");
  });

  it("stays finite for junk input", () => {
    for (const v of [NaN, 0, -1, Infinity]) {
      assert.ok(Number.isFinite(computeOverheadBytes(v, v, v)));
    }
  });
});

describe("estimateMemory", () => {
  it("puts everything in RAM on the CPU backend", () => {
    const est = estimateMemory(denseCaps(), loadSettings(), undefined, { cpuOnly: true });
    assert.ok(est);
    assert.equal(est.gpuWeightsBytes, 0);
    assert.ok(est.totalCpuBytes > 18 * GiB);
  });

  it("flags spill when the estimate exceeds VRAM", () => {
    const tight = estimateMemory(denseCaps(), loadSettings({ contextLength: 65536 }), gpu(12));
    assert.ok(tight?.willSpill);
    const roomy = estimateMemory(denseCaps(), loadSettings({ contextLength: 8192 }), gpu(48));
    assert.equal(roomy?.willSpill, false);
  });

  it("splits weights when only some layers are offloaded", () => {
    const est = estimateMemory(denseCaps(), loadSettings({ gpuOffload: 24 }), gpu(24));
    assert.ok(est);
    assert.equal(est.layersOnGpu, 24);
    assert.ok(Math.abs(est.gpuWeightsBytes - est.cpuWeightsBytes) < 1024);
  });

  it("treats gpuOffload 99 as all layers", () => {
    const est = estimateMemory(denseCaps(), loadSettings({ gpuOffload: 99 }), gpu(48));
    assert.equal(est?.layersOnGpu, 48);
    assert.equal(est?.cpuWeightsBytes, 0);
  });

  it("moves KV to RAM when GPU KV offload is disabled", () => {
    const on = estimateMemory(denseCaps(), loadSettings(), gpu(48));
    const off = estimateMemory(denseCaps(), loadSettings({ offloadKvCacheToGpu: false }), gpu(48));
    assert.ok(on!.totalGpuBytes > off!.totalGpuBytes);
    assert.ok(off!.totalCpuBytes > on!.totalCpuBytes);
  });

  it("credits --n-cpu-moe with moving expert weights off the GPU", () => {
    const none = estimateMemory(moeCaps(), loadSettings({ nCpuMoe: 0 }), gpu(48));
    const some = estimateMemory(moeCaps(), loadSettings({ nCpuMoe: 24 }), gpu(48));
    assert.ok(some!.gpuWeightsBytes < none!.gpuWeightsBytes);
  });

  it("returns undefined for unreadable models", () => {
    assert.equal(estimateMemory(undefined, loadSettings(), gpu(24)), undefined);
    assert.equal(
      estimateMemory(denseCaps({ fileSizeBytes: 0 }), loadSettings(), gpu(24)),
      undefined
    );
  });

  it("adds DFlash draft weights and f16 KV when mode is dflash", () => {
    const draft = denseCaps({
      name: "dflash-draft",
      architecture: "dflash",
      fileSizeBytes: GiB,
      blockCount: 5,
      slidingWindow: 2048,
      slidingWindowPattern: [true, true, true, true, false],
    });
    const base = estimateMemory(denseCaps(), loadSettings({ speculativeMode: "off" }), gpu(48));
    const withDraft = estimateMemory(
      denseCaps(),
      loadSettings({
        speculativeMode: "dflash",
        draftModelPath: "/models/draft.gguf",
        draftGpuOffload: 99,
        maxDraftTokens: 15,
      }),
      gpu(48),
      { draftCaps: draft }
    );
    assert.ok(base && withDraft);
    assert.ok(withDraft.totalGpuBytes > base.totalGpuBytes);
    assert.equal(withDraft.draftFileSizeBytes, GiB);
    assert.ok((withDraft.draftKvBytes || 0) > 0);
    assert.ok(
      withDraft.charts.vram.segments.some((s) => s.key === "draft" && s.bytes > 0),
      "VRAM chart should include a draft segment"
    );
  });

  it("ignores draft caps unless speculativeMode is dflash", () => {
    const draft = denseCaps({ fileSizeBytes: GiB, blockCount: 5 });
    const est = estimateMemory(
      denseCaps(),
      loadSettings({ speculativeMode: "off", draftModelPath: "/models/draft.gguf" }),
      gpu(48),
      { draftCaps: draft }
    );
    assert.equal(est?.draftFileSizeBytes, undefined);
    assert.ok(!(est?.charts.vram.segments.some((s) => s.key === "draft" && s.bytes > 0)));
  });

  it("adds MTP head + KV when mode is mtp and nextn layers are present", () => {
    const caps = denseCaps({ nextnPredictLayers: 1 });
    const base = estimateMemory(caps, loadSettings({ speculativeMode: "off" }), gpu(48));
    const withMtp = estimateMemory(
      caps,
      loadSettings({ speculativeMode: "mtp", maxDraftTokens: 3 }),
      gpu(48)
    );
    assert.ok(base && withMtp);
    assert.ok(withMtp.totalGpuBytes > base.totalGpuBytes);
    assert.equal(withMtp.mtpLayers, 1);
    assert.ok((withMtp.mtpWeightsBytes || 0) > 0);
    assert.ok((withMtp.mtpKvBytes || 0) > 0);
    assert.ok(
      withMtp.charts.vram.segments.some((s) => s.key === "draft" && s.bytes > 0),
      "VRAM chart should include an MTP segment"
    );
  });

  it("omits MTP overhead when nextn_predict_layers is missing", () => {
    const est = estimateMemory(
      denseCaps({ nextnPredictLayers: 0 }),
      loadSettings({ speculativeMode: "mtp" }),
      gpu(48)
    );
    assert.equal(est?.mtpLayers, undefined);
    assert.ok(!(est?.charts.vram.segments.some((s) => s.key === "draft" && s.bytes > 0)));
  });
});

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

  it("tiles a short SWA pattern across all layers (Muse Glimmer official GGUF)", () => {
    const glimmer = denseCaps({
      blockCount: 52,
      attentionHeadCount: 32,
      attentionHeadCountKv: 2,
      keyLength: 128,
      valueLength: 128,
      embeddingLength: 6656,
      slidingWindow: 2048,
      slidingWindowPattern: [true, true, true, false],
    });
    const atWindow = estimateKvBytes(glimmer, 2048, "q8_0", "q8_0");
    const atLong = estimateKvBytes(glimmer, 131072, "q8_0", "q8_0");
    const naive = estimateKvBytes(
      denseCaps({
        blockCount: 52,
        attentionHeadCount: 32,
        attentionHeadCountKv: 2,
        keyLength: 128,
        valueLength: 128,
        embeddingLength: 6656,
      }),
      131072,
      "q8_0",
      "q8_0"
    );
    assert.ok(atLong > atWindow, "global layers still grow with context");
    assert.ok(atLong < naive / 2, "39/52 layers must stay capped at 2048, not scale with n_ctx");
    // 13 global layers * (131072-2048) extra tokens — not 52 layers.
    assert.ok(atLong / atWindow < 20, "must not grow ~64× like a dense 128k cache");
  });

  it("does not invent smaller SWA head dims when the GGUF omits key_length_swa", () => {
    const sameDim = denseCaps({
      slidingWindow: 2048,
      slidingWindowPattern: Array.from({ length: 48 }, () => true),
      keyLength: 128,
      valueLength: 128,
    });
    const atWindow = estimateKvBytes(sameDim, 2048, "q8_0", "q8_0");
    const denseAtWindow = estimateKvBytes(
      denseCaps({ keyLength: 128, valueLength: 128 }),
      2048,
      "q8_0",
      "q8_0"
    );
    assert.equal(atWindow, denseAtWindow);
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

  it("splits VRAM across two GPUs and exposes a second chart", () => {
    const g0 = { totalBytes: 16 * GiB, usedBytes: 0, name: "RX 9070 XT", source: "test" };
    const g1 = { totalBytes: 16 * GiB, usedBytes: 0, name: "RX 9060 XT", source: "test" };
    const est = estimateMemory(
      denseCaps(),
      loadSettings({ tensorSplit: "3,1", mainGpu: 0 }),
      g0,
      { gpus: [g0, g1] }
    );
    assert.ok(est);
    assert.ok(est.charts.vram2);
    const w0 = est.charts.vram.segments.find((s) => s.key === "weights")!.bytes;
    const w1 = est.charts.vram2!.segments.find((s) => s.key === "weights")!.bytes;
    assert.ok(Math.abs(w0 / (w0 + w1) - 0.75) < 0.001);
    assert.ok(Math.abs(w1 / (w0 + w1) - 0.25) < 0.001);
    const oh0 = est.charts.vram.segments.find((s) => s.key === "overhead")!.bytes;
    const oh1 = est.charts.vram2!.segments.find((s) => s.key === "overhead")!.bytes;
    assert.ok(oh0 > 0);
    assert.equal(oh1, 0);
    assert.ok(est.charts.vram.capacityBytes === 16 * GiB);
    assert.ok(est.charts.vram2!.capacityBytes === 16 * GiB);
  });

  it("puts the main GPU chart first when it is not device 0", () => {
    const g0 = { totalBytes: 16 * GiB, usedBytes: 0, name: "RX 9060 XT", source: "test" };
    const g1 = { totalBytes: 16 * GiB, usedBytes: 0, name: "RX 9070 XT", source: "test" };
    const est = estimateMemory(
      denseCaps(),
      loadSettings({ tensorSplit: "25,75", mainGpu: 1 }),
      g0,
      { gpus: [g0, g1] }
    );
    assert.ok(est?.charts.vram2);
    assert.match(est.charts.vram.title, /9070/);
    assert.match(est.charts.vram2!.title, /9060/);
    const wMain = est.charts.vram.segments.find((s) => s.key === "weights")!.bytes;
    const wOther = est.charts.vram2!.segments.find((s) => s.key === "weights")!.bytes;
    assert.ok(Math.abs(wMain / (wMain + wOther) - 0.75) < 0.001);
    assert.ok(est.charts.vram.segments.find((s) => s.key === "overhead")!.bytes > 0);
    assert.equal(est.charts.vram2!.segments.find((s) => s.key === "overhead")!.bytes, 0);
  });

  it("flags spill when either GPU is over capacity", () => {
    const g0 = gpu(8);
    const g1 = gpu(8);
    const est = estimateMemory(
      denseCaps({ fileSizeBytes: 18 * GiB }),
      loadSettings({ tensorSplit: "3,1", contextLength: 65536 }),
      g0,
      { gpus: [g0, g1] }
    );
    assert.ok(est?.willSpill);
    assert.ok(est.charts.vram2);
  });

  it("does not add a second chart for a single GPU", () => {
    const est = estimateMemory(denseCaps(), loadSettings(), gpu(16));
    assert.equal(est?.charts.vram2, undefined);
  });

  it("split-mode none parks all GPU weights on the main GPU", () => {
    const g0 = { totalBytes: 16 * GiB, usedBytes: 0, name: "RX 9070 XT", source: "test" };
    const g1 = { totalBytes: 16 * GiB, usedBytes: 0, name: "RX 9060 XT", source: "test" };
    const est = estimateMemory(
      denseCaps(),
      loadSettings({ tensorSplit: "90,10", splitMode: "none", mainGpu: 0 }),
      g0,
      { gpus: [g0, g1] }
    );
    assert.ok(est);
    const w0 = est.charts.vram.segments.find((s) => s.key === "weights")!.bytes;
    const w1 = est.charts.vram2?.segments.find((s) => s.key === "weights")?.bytes ?? 0;
    assert.ok(w0 > 0);
    assert.equal(w1, 0);
    assert.ok(est.lines.some((l) => /No GPU split/i.test(l)));
    assert.ok(!est.warnings.some((w) => /split by VRAM/i.test(w)));
  });
});

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  computeOverheadBytes,
  estimateKvBytes,
  estimateMemory,
  inferComputeBackend,
  kvSlotMultiplier,
  peerGpuOverheadBytes,
} from "../src/memoryEstimate";
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

  it("prices q4_0 at 9/32 of f16 (18 B per 32 elems)", () => {
    const f16 = estimateKvBytes(caps, 8192, "f16", "f16");
    const q4 = estimateKvBytes(caps, 8192, "q4_0", "q4_0");
    assert.equal(q4, f16 * (0.5625 / 2));
  });

  it("sizes the full llama.cpp cache-type set correctly", () => {
    const perToken = estimateKvBytes(caps, 1, "f32", "f32");
    for (const [type, ratio] of [
      ["f32", 1],
      ["f16", 0.5],
      ["bf16", 0.5],
      ["q8_0", 0.25],
      ["q5_1", 0.1875],
      ["q5_0", 0.171875],
      ["q4_1", 0.15625],
      ["q4_0", 0.140625],
      ["iq4_nl", 0.140625],
    ] as const) {
      assert.equal(
        estimateKvBytes(caps, 1, type, type),
        perToken * ratio,
        `unexpected size ratio for ${type}`,
      );
    }
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

  it("derives the same hybrid skip from fullAttentionInterval (qwen4exp GDN)", () => {
    const hybrid = denseCaps({ architecture: "qwen4exp", fullAttentionInterval: 4 });
    const full = estimateKvBytes(denseCaps(), 32768, "q8_0", "q8_0");
    assert.equal(estimateKvBytes(hybrid, 32768, "q8_0", "q8_0"), full / 4);
  });

  it("skips n_kv=0 layers instead of billing them as Q-heads (Ling / bailingmoe3 KDA)", () => {
    const kvHeads = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1];
    const ling = denseCaps({
      blockCount: 24,
      embeddingLength: 1536,
      attentionHeadCount: 16,
      attentionHeadCountKvPerLayer: kvHeads,
      keyLength: 576,
      valueLength: 128,
      fileSizeBytes: 5340611552,
    });
    const kv = estimateKvBytes(ling, 65536, "f16", "f16");
    const mlaOnly = 6 * (576 + 128) * 2 * 65536;
    assert.equal(kv, mlaOnly);
    // Old webview bug: n_kv=0 was falsy so those 18 KDA layers used 16 Q-heads.
    const billedAsQHeads = (18 * 16 + 6) * (576 + 128) * 2 * 65536;
    assert.ok(kv < billedAsQHeads / 20);

    const at34k = estimateMemory(
      ling,
      loadSettings({
        contextLength: 34091,
        gpuOffload: 99,
        cacheTypeK: "f16",
        cacheTypeV: "f16",
        physicalBatchSize: 1024,
        splitMode: "none",
      }),
      gpu(16)
    );
    assert.ok(at34k);
    assert.ok(at34k.kvBytes < 0.5 * GiB, `MLA KV at 34k should be ~0.27 GiB, got ${at34k.kvBytes}`);
    assert.ok(at34k.totalGpuBytes < 9 * GiB, `total at 34k should sit near the 8.3 GiB measurement, got ${at34k.totalGpuBytes}`);
    assert.ok(at34k.totalGpuBytes > 5 * GiB);
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

  it("grows with context and is larger on Vulkan than CUDA", () => {
    const at8k = computeOverheadBytes(2560, 1024, 2048, {
      contextLength: 8192,
      backend: "vulkan",
      flashAttention: "auto",
    });
    const at64k = computeOverheadBytes(2560, 1024, 2048, {
      contextLength: 65536,
      backend: "vulkan",
      flashAttention: "auto",
    });
    const cuda64k = computeOverheadBytes(2560, 1024, 2048, {
      contextLength: 65536,
      backend: "cuda",
      flashAttention: "auto",
    });
    assert.ok(at64k > at8k, "64k graph should exceed an 8k graph");
    assert.ok(at64k > cuda64k, "Vulkan heaps are fatter than CUDA");
    assert.ok(at64k > 2 * GiB, "64k Vulkan compute was the ~3 GiB hole on the 9070");
  });

  it("sizes a peer-GPU heap below the main-GPU graph", () => {
    const main = computeOverheadBytes(2560, 1024, 2048, {
      contextLength: 65536,
      backend: "vulkan",
    });
    const peer = peerGpuOverheadBytes(2560, 1024, "vulkan");
    assert.ok(peer > 2 * GiB, "each Vulkan device reserves ~2.5 GiB of heap slop");
    assert.ok(peer < main);
  });

  it("reads Vulkan/CUDA from llama.cpp device ids", () => {
    assert.equal(inferComputeBackend(true, []), "cpu");
    assert.equal(
      inferComputeBackend(false, [{ llamaDeviceId: "Vulkan0" }, { llamaDeviceId: "Vulkan1" }]),
      "vulkan"
    );
    assert.equal(inferComputeBackend(false, [{ llamaDeviceId: "CUDA0" }]), "cuda");
    assert.equal(inferComputeBackend(false, []), "unknown");
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

  it("keeps the PLE n-gram table in RAM even at full GPU offload", () => {
    const file = 90 * GiB;
    const caps = moeCaps({
      architecture: "qwen4exp",
      fileSizeBytes: file,
      pleShare: 0.4,
      fullAttentionInterval: 4,
    });
    const est = estimateMemory(
      caps,
      loadSettings({ gpuOffload: 99, nCpuMoe: 0, lazyMode: "off" }),
      gpu(80)
    );
    assert.ok(est);
    assert.equal(est.cpuWeightsBytes, file * 0.4);
    assert.equal(est.gpuWeightsBytes, file * 0.6);
    assert.ok(est.warnings.some((w) => /stays in system RAM/i.test(w)));
  });

  it("omits a lazily-read PLE table from the RAM bar", () => {
    const file = 90 * GiB;
    const caps = moeCaps({
      architecture: "qwen4exp",
      fileSizeBytes: file,
      pleShare: 0.4,
      fullAttentionInterval: 4,
    });
    const est = estimateMemory(
      caps,
      loadSettings({ gpuOffload: 99, nCpuMoe: 0, lazyMode: "on", tryMmap: true }),
      gpu(80)
    );
    assert.ok(est);
    assert.equal(est.cpuWeightsBytes, 0);
    assert.equal(est.gpuWeightsBytes, file * 0.6);
    assert.ok(est.warnings.some((w) => /read from disk/i.test(w)));
  });

  it("does not move PLE bytes onto the GPU for dense models without a table", () => {
    const est = estimateMemory(denseCaps(), loadSettings({ gpuOffload: 99 }), gpu(48));
    assert.equal(est?.cpuWeightsBytes, 0);
  });

  it("credits --n-cpu-ffn with moving dense FFN weights off the GPU", () => {
    const caps = denseCaps({ ffnLength: 13824, embeddingLength: 5120 });
    const none = estimateMemory(caps, loadSettings({ nCpuFfn: 0 }), gpu(48));
    const some = estimateMemory(caps, loadSettings({ nCpuFfn: 24 }), gpu(48));
    assert.ok(some!.gpuWeightsBytes < none!.gpuWeightsBytes);
    // Heuristic share: 3·ffn / (3·ffn + 4·embed) ≈ 0.619 for 13824/5120.
    const expectedDeduct = caps.fileSizeBytes! * (24 / 48) * (3 * 13824 / (3 * 13824 + 4 * 5120));
    assert.ok(
      Math.abs((none!.gpuWeightsBytes - some!.gpuWeightsBytes) - expectedDeduct) < 1,
      `expected ~${expectedDeduct} moved, got ${none!.gpuWeightsBytes - some!.gpuWeightsBytes}`,
    );
  });

  it("ignores --n-cpu-ffn for MoE models", () => {
    const none = estimateMemory(moeCaps(), loadSettings({ nCpuFfn: 0 }), gpu(48));
    const some = estimateMemory(moeCaps(), loadSettings({ nCpuFfn: 24 }), gpu(48));
    assert.equal(some!.gpuWeightsBytes, none!.gpuWeightsBytes);
  });

  it("ignores --n-cpu-ffn on the CPU backend", () => {
    const caps = denseCaps({ ffnLength: 13824 });
    const none = estimateMemory(caps, loadSettings({ nCpuFfn: 0 }), gpu(48), { cpuOnly: true });
    const some = estimateMemory(caps, loadSettings({ nCpuFfn: 24 }), gpu(48), { cpuOnly: true });
    assert.equal(some!.gpuWeightsBytes, none!.gpuWeightsBytes);
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

    const stacked = estimateMemory(
      denseCaps(),
      loadSettings({
        speculativeMode: "ngram-dflash",
        draftModelPath: "/models/draft.gguf",
        draftGpuOffload: 99,
        maxDraftTokens: 15,
      }),
      gpu(48),
      { draftCaps: draft }
    );
    assert.ok(stacked);
    assert.equal(stacked.draftFileSizeBytes, withDraft.draftFileSizeBytes);
    assert.equal(stacked.totalGpuBytes, withDraft.totalGpuBytes);
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
    const extra = withMtp.totalGpuBytes - base.totalGpuBytes;
    assert.ok(
      extra < (withMtp.mtpWeightsBytes || 0),
      `baked-in MTP heads must not be added on top of GGUF size (extra ${extra} vs weights ${withMtp.mtpWeightsBytes})`
    );
    assert.ok(Math.abs(extra - (withMtp.mtpKvBytes || 0)) < 1, "GPU extra should be MTP KV only");
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
    assert.ok(oh1 > 0, "layer-split peers have their own device heap");
    assert.ok(oh0 > oh1, "main GPU keeps the graph / driver tax");
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
    assert.ok(est.charts.vram2!.segments.find((s) => s.key === "overhead")!.bytes > 0);
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
    assert.equal(est.charts.vram2?.segments.find((s) => s.key === "overhead")?.bytes ?? 0, 0);
    assert.ok(est.lines.some((l) => /No GPU split/i.test(l)));
    assert.ok(!est.warnings.some((w) => /split by VRAM/i.test(w)));
  });

  it("does not flag spill on GPU 0 when a 0-byte second card would have dumped the whole model there", () => {
    const g0 = {
      totalBytes: 16 * GiB,
      usedBytes: 0,
      name: "RX 9070 XT",
      source: "test",
      llamaDeviceId: "Vulkan0",
    };
    const g1 = {
      totalBytes: 0,
      usedBytes: 0,
      name: "RX 9060 XT",
      source: "test",
      llamaDeviceId: "Vulkan1",
    };
    const one = estimateMemory(denseCaps(), loadSettings({ tensorSplit: "", contextLength: 32768 }), g0);
    const two = estimateMemory(
      denseCaps(),
      loadSettings({ tensorSplit: "", contextLength: 32768 }),
      g0,
      { gpus: [g0, g1] }
    );
    assert.ok(one?.willSpill, "the same load on one 16 GiB card should spill");
    assert.ok(two?.charts.vram2);
    assert.ok(
      two.charts.vram.totalBytes < (one?.totalGpuBytes || 0) * 0.7,
      "a 0-byte second card must not dump 100% of weights onto GPU 0"
    );
  });

  it("splits a ~26 GiB estimate across two 16 GiB cards without spilling", () => {
    const g0 = { totalBytes: 16 * GiB, usedBytes: 0, name: "RX 9070 XT", source: "test" };
    const g1 = { totalBytes: 16 * GiB, usedBytes: 0, name: "RX 9060 XT", source: "test" };
    const est = estimateMemory(
      denseCaps(),
      loadSettings({ tensorSplit: "50,50", contextLength: 32768 }),
      g0,
      { gpus: [g0, g1] }
    );
    assert.ok(est);
    assert.equal(est.willSpill, false);
    assert.ok(est.charts.vram.totalBytes <= 16 * GiB - 2 * GiB);
    assert.ok((est.charts.vram2?.totalBytes || 0) <= 16 * GiB - 2 * GiB);
  });

  it("prices Vulkan dual-GPU compute near RADV occupancy for Flash-Next", () => {
    const MiB = 1024 ** 2;
    const caps = moeCaps({
      architecture: "qwen4exp",
      fileSizeBytes: 81961823936,
      embeddingLength: 2560,
      attentionHeadCount: 24,
      attentionHeadCountKv: 2,
      keyLength: 256,
      valueLength: 256,
      fullAttentionInterval: 4,
      pleShare: 0.4,
      moeExpertShare: 0.9,
      expertCount: 512,
    });
    const g0 = {
      totalBytes: 16304 * MiB,
      usedBytes: 0,
      name: "RX 9070",
      source: "test",
      llamaDeviceId: "Vulkan0",
    };
    const g1 = {
      totalBytes: 16304 * MiB,
      usedBytes: 0,
      name: "RX 9060 XT",
      source: "test",
      llamaDeviceId: "Vulkan1",
    };
    const est = estimateMemory(
      caps,
      loadSettings({
        contextLength: 65536,
        gpuOffload: 48,
        nCpuMoe: 16,
        tensorSplit: "45,55",
        cacheTypeK: "q8_0",
        cacheTypeV: "q5_1",
        evalBatchSize: 2048,
        physicalBatchSize: 1024,
        mainGpu: 0,
        lazyMode: "auto",
        tryMmap: true,
      }),
      g0,
      { gpus: [g0, g1] }
    );
    assert.ok(est?.charts.vram2);
    // --n-cpu-moe 16 + layer split 45,55: first layers (cheap) on the 9070,
    // remaining full experts on the 9060. Live: 9070 ~12 GiB, 9060 15.8 + GTT.
    const w0 = est.charts.vram.segments.find((s) => s.key === "weights")!.bytes;
    const w1 = est.charts.vram2.segments.find((s) => s.key === "weights")!.bytes;
    assert.ok(w1 > w0 * 1.8, `9060 should hold the leftover experts (${w0} vs ${w1})`);
    assert.ok(
      est.charts.vram2.totalBytes > 15.9 * GiB,
      `9060 demand ${est.charts.vram2.totalBytes} should exceed 16 GiB (GTT spill)`
    );
    assert.ok(est.willSpill);
    assert.ok(est.warnings.some((w) => /later cards hold more expert weight/i.test(w)));
    const oh0 = est.charts.vram.segments.find((s) => s.key === "overhead")!.bytes;
    const oh1 = est.charts.vram2.segments.find((s) => s.key === "overhead")!.bytes;
    assert.ok(oh0 > 4 * GiB, "main-GPU graph + Vulkan heap reserve");
    assert.ok(oh1 > 2 * GiB && oh1 < oh0);
  });

  it("adds Vulkan heap reserve so a 62/38 Flash-Next split looks full on both 16 GB cards", () => {
    const MiB = 1024 ** 2;
    const caps = moeCaps({
      architecture: "qwen4exp",
      fileSizeBytes: 81961823936,
      embeddingLength: 2560,
      attentionHeadCount: 24,
      attentionHeadCountKv: 2,
      keyLength: 256,
      valueLength: 256,
      fullAttentionInterval: 4,
      pleShare: 0.4,
      moeExpertShare: 0.9,
      expertCount: 512,
    });
    const g0 = {
      totalBytes: 16304 * MiB,
      usedBytes: 0,
      name: "RX 9070",
      source: "test",
      llamaDeviceId: "Vulkan0",
    };
    const g1 = {
      totalBytes: 16304 * MiB,
      usedBytes: 0,
      name: "RX 9060 XT",
      source: "test",
      llamaDeviceId: "Vulkan1",
    };
    const est = estimateMemory(
      caps,
      loadSettings({
        contextLength: 65536,
        gpuOffload: 48,
        nCpuMoe: 16,
        tensorSplit: "62,38",
        cacheTypeK: "q8_0",
        cacheTypeV: "q5_1",
        evalBatchSize: 2048,
        physicalBatchSize: 1024,
        mainGpu: 0,
        lazyMode: "auto",
        tryMmap: true,
      }),
      g0,
      { gpus: [g0, g1] }
    );
    assert.ok(est?.charts.vram2);
    // Live sysfs/LACT: both cards 15.4–15.9 GiB. Old bars were 13.2 / 13.8.
    assert.ok(
      est.charts.vram.totalBytes > 14.8 * GiB,
      `9070 bar ${est.charts.vram.totalBytes} should sit near the 15.4 GiB measurement`
    );
    assert.ok(
      est.charts.vram2.totalBytes > 14.8 * GiB,
      `9060 bar ${est.charts.vram2.totalBytes} should sit near the 15.9 GiB measurement`
    );
    assert.ok(est.willSpill);
  });

  it("multiplies KV by parallel slots when the cache is not unified", () => {
    const one = estimateMemory(
      denseCaps(),
      loadSettings({ unifiedKvCache: false, maxConcurrentPredictions: 1, contextLength: 8192 }),
      gpu(48)
    );
    const four = estimateMemory(
      denseCaps(),
      loadSettings({ unifiedKvCache: false, maxConcurrentPredictions: 4, contextLength: 8192 }),
      gpu(48)
    );
    const unified = estimateMemory(
      denseCaps(),
      loadSettings({ unifiedKvCache: true, maxConcurrentPredictions: 4, contextLength: 8192 }),
      gpu(48)
    );
    assert.ok(one && four && unified);
    assert.equal(kvSlotMultiplier({ unifiedKvCache: false, maxConcurrentPredictions: 4 }), 4);
    assert.equal(four.kvBytes, one.kvBytes * 4);
    assert.equal(unified.kvBytes, one.kvBytes);
  });
});

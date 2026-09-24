import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  computeOverheadBytes,
  deviceWouldSpill,
  VRAM_HEADROOM_BYTES,
  estimateKvBytes,
  estimateMemory,
  estimateRecurrentStateBytesPerLayer,
  hostFallbackWarning,
  inferComputeBackend,
  integratedGpuNotes,
  kvSlotMultiplier,
  peerGpuOverheadBytes,
  vulkanDeviceReservedBytes,
  vulkanDriverBytes,
} from "../src/memoryEstimate";
import { isIntegratedGpu } from "../src/gpuInfo";
import { ModelCapabilities } from "../src/ggufMetadata";
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

describe("estimateRecurrentStateBytesPerLayer", () => {
  /** qwen35-style hybrid: every 4th layer is full attention (layers 3, 7, …). */
  const hybrid = (overrides: Partial<ModelCapabilities> = {}) =>
    denseCaps({
      architecture: "qwen35",
      blockCount: 8,
      embeddingLength: 1024,
      fullAttentionInterval: 4,
      ...overrides,
    });
  const ssm = { ssmStateSize: 128, ssmInnerSize: 1024, ssmConvKernel: 4, ssmGroupCount: 1 };
  // (d_inner·d_state + (d_conv−1)·(d_inner + 2·n_group·d_state)) · f32
  const perLayer = (1024 * 128 + 3 * (1024 + 2 * 1 * 128)) * 4;

  it("bills a fixed f32 state on recurrent layers only", () => {
    const bytes = estimateRecurrentStateBytesPerLayer(hybrid(ssm));
    assert.equal(bytes.length, 8);
    assert.equal(bytes.filter((b) => b > 0).length, 6, "2 of 8 layers are full attention");
    assert.equal(bytes[0], perLayer);
    assert.equal(bytes[3], 0);
    assert.equal(bytes[7], 0);
  });

  it("stays 0 for dense models and hybrids without ssm geometry", () => {
    const sum = (v: number[]) => v.reduce((a, b) => a + b, 0);
    assert.equal(sum(estimateRecurrentStateBytesPerLayer(denseCaps())), 0);
    assert.equal(sum(estimateRecurrentStateBytesPerLayer(hybrid())), 0);
  });

  it("ignores absurd ssm geometry instead of inventing GiBs", () => {
    const sum = (v: number[]) => v.reduce((a, b) => a + b, 0);
    assert.equal(sum(estimateRecurrentStateBytesPerLayer(hybrid({ ...ssm, ssmStateSize: 10_000_000 }))), 0);
    assert.equal(sum(estimateRecurrentStateBytesPerLayer(hybrid({ ...ssm, ssmInnerSize: 10_000_000 }))), 0);
  });

  it("folds state into the KV bucket and scales it with slots", () => {
    const caps = hybrid(ssm);
    const kvOnly = estimateKvBytes(caps, 8192, "q8_0", "q8_0");
    const kvWarmOnly = estimateKvBytes(caps, 2048, "q8_0", "q8_0");
    const state = perLayer * 6;
    const one = estimateMemory(
      caps,
      loadSettings({ contextLength: 8192, unifiedKvCache: false, maxConcurrentPredictions: 1 }),
      gpu(48)
    );
    const four = estimateMemory(
      caps,
      loadSettings({ contextLength: 8192, unifiedKvCache: false, maxConcurrentPredictions: 4 }),
      gpu(48)
    );
    assert.ok(one && four);
    // Six recurrent layers × fixed state, billed once (state is context-free).
    assert.equal(one.recurrentStateBytes, state);
    assert.equal(one.kvBytes, kvOnly + state);
    // Unlike the KV, state neither grows with ctx nor shrinks when warm.
    assert.equal(one.kvBytesWarm, kvWarmOnly + state, "warm KV still carries the full state");
    assert.equal(four.kvBytes, (kvOnly + state) * 4, "state is per sequence slot");
    assert.equal(four.recurrentStateBytes, state * 4);
  });

  it("leaves dense estimates byte-identical", () => {
    const withSsmKeys = denseCaps({ ...ssm });
    const plain = denseCaps();
    const settings = loadSettings({ contextLength: 8192 });
    assert.equal(
      estimateMemory(withSsmKeys, settings, gpu(48))?.kvBytes,
      estimateMemory(plain, settings, gpu(48))?.kvBytes
    );
  });
});

describe("hostFallbackWarning", () => {
  const MiB = 1024 ** 2;
  /** The reported field case: 9060 XT idle in VRAM, ~14 GiB of GTT. */
  const idleInGtt = {
    totalBytes: 15.9 * GiB,
    usedBytes: 0.1 * GiB,
    gttUsedBytes: 14.2 * GiB,
    gttTotalBytes: 16 * GiB,
    visVramTotalBytes: 256 * MiB,
    name: "Radeon RX 9060 XT",
    source: "test",
    llamaDeviceId: "Vulkan1",
  };
  const healthy = {
    totalBytes: 15.9 * GiB,
    usedBytes: 12.5 * GiB,
    gttUsedBytes: 0.6 * GiB,
    gttTotalBytes: 16 * GiB,
    visVramTotalBytes: 16 * GiB,
    name: "Radeon RX 9070 XT",
    source: "test",
    llamaDeviceId: "Vulkan0",
  };

  it("flags a card holding the model in GTT instead of VRAM", () => {
    const msg = hostFallbackWarning(idleInGtt, 1);
    assert.ok(msg, "expected a warning");
    assert.ok(msg!.includes("Vulkan1"), msg!);
    assert.ok(/in GTT/.test(msg!), msg!);
    assert.ok(/14.2 GiB/.test(msg!), msg!);
    assert.ok(/ReBAR/.test(msg!), msg!);
    assert.equal(/PCIe|split-mode none/.test(msg!), false, msg!);
  });

  it("stays quiet for a card that is genuinely using its VRAM", () => {
    assert.equal(hostFallbackWarning(healthy, 0), undefined);
  });

  it("needs both a real GTT figure and a small BAR window to speak up", () => {
    // Large GTT but a full BAR window: RADV is behaving, not falling back.
    assert.equal(hostFallbackWarning({ ...healthy, gttUsedBytes: 14 * GiB }, 0), undefined);
    // Small BAR window but nothing resident yet.
    assert.equal(hostFallbackWarning({ ...healthy, gttUsedBytes: 64 * MiB }, 0), undefined);
    assert.equal(hostFallbackWarning(undefined, 0), undefined);
  });

  it("warns without the BAR fields when GTT clearly dominates (non-sysfs sources)", () => {
    const msg = hostFallbackWarning(
      { totalBytes: 16 * GiB, usedBytes: 0, gttUsedBytes: 9 * GiB, source: "test" },
      1
    );
    assert.ok(msg && /in GTT/.test(msg), msg ?? "expected a warning");
  });

  it("stays quiet for an APU whose GTT is the normal path", () => {
    const apu = {
      totalBytes: 512 * MiB,
      usedBytes: 431 * MiB,
      gttUsedBytes: 1.24 * GiB,
      gttTotalBytes: 15.3 * GiB,
      visVramTotalBytes: 512 * MiB,
      name: "Cezanne [Radeon Vega Series / Radeon Vega Mobile Series]",
      source: "test",
      llamaDeviceId: "Vulkan0",
    };
    assert.equal(isIntegratedGpu(apu), true);
    assert.equal(hostFallbackWarning(apu, 0), undefined);
  });

  it("is surfaced by estimateMemory when the collector reports GTT", () => {
    const g0 = { ...healthy, usedBytes: 0 } as never;
    const g1 = { ...idleInGtt } as never;
    const est = estimateMemory(denseCaps(), loadSettings({ contextLength: 8192 }), g0, {
      gpus: [g0, g1],
    });
    assert.ok(est);
    assert.ok(
      est.warnings.some((w) => /in GTT/.test(w)),
      `warnings: ${est.warnings.join(" | ")}`
    );
    // A runtime observation must not feed the predictive spill search.
    assert.ok(est.lines.some((l) => /GTT now/.test(l)));
  });
});

describe("iGPU / APU guidance", () => {
  const MiB = 1024 ** 2;
  const apu = {
    totalBytes: 512 * MiB,
    usedBytes: 431 * MiB,
    gttUsedBytes: 1.24 * GiB,
    gttTotalBytes: 15.3 * GiB,
    visVramTotalBytes: 512 * MiB,
    name: "Onboard IGD",
    source: "test",
    llamaDeviceId: "Vulkan0",
  };

  it("does not treat a 512 MiB carve-out as already spilling", () => {
    assert.equal(deviceWouldSpill(0, 512 * MiB), false);
    assert.equal(deviceWouldSpill(512 * MiB + 1, 512 * MiB), true);
    assert.equal(deviceWouldSpill(11 * GiB, 12 * GiB), true);
    assert.equal(deviceWouldSpill(9 * GiB, 12 * GiB), false);
    assert.equal(VRAM_HEADROOM_BYTES, 1.5 * GiB);
    assert.equal(deviceWouldSpill(12 * GiB - VRAM_HEADROOM_BYTES, 12 * GiB), false);
    assert.equal(deviceWouldSpill(12 * GiB - VRAM_HEADROOM_BYTES + 1, 12 * GiB), true);
  });

  it("classifies APU names and UMA+GTT, but not a discrete RX card", () => {
    assert.equal(isIntegratedGpu(apu), true);
    assert.equal(
      isIntegratedGpu({
        totalBytes: 512 * MiB,
        gttTotalBytes: 15 * GiB,
        name: "amdgpu",
      }),
      true
    );
    assert.equal(
      isIntegratedGpu({
        totalBytes: 16 * GiB,
        gttTotalBytes: 16 * GiB,
        name: "Radeon RX 9070 XT",
      }),
      false
    );
  });

  it("replaces Tight/ReBAR banners when offload is 0", () => {
    const notes = integratedGpuNotes({
      gpu: apu,
      onGpu: 0,
      nLayers: 30,
      usedBytes: 0,
      gpuWeightsBytes: 0,
    });
    assert.equal(notes.willSpill, false);
    assert.match(notes.warnings[0] || "", /CPU only/);
    assert.match(notes.warnings[1] || "", /offload to 30/);
    const est = estimateMemory(
      denseCaps({ fileSizeBytes: 2.7 * GiB, blockCount: 30 }),
      loadSettings({ gpuOffload: 0, contextLength: 16384 }),
      apu,
      { gpus: [apu] }
    );
    assert.ok(est);
    assert.equal(est.willSpill, false);
    assert.ok(est.warnings.some((w) => /CPU only/.test(w)), est.warnings.join(" | "));
    assert.ok(est.warnings.some((w) => /GTT/.test(w)), est.warnings.join(" | "));
    assert.ok(!est.warnings.some((w) => /Tight on|PCIe|ReBAR|2.00 GiB/.test(w)), est.warnings.join(" | "));
    assert.ok(est.charts.vram.capacityBytes && est.charts.vram.capacityBytes > 10 * GiB);
  });

  it("uses GTT as the budget when layers are offloaded", () => {
    const fits = estimateMemory(
      denseCaps({ fileSizeBytes: 2.7 * GiB, blockCount: 30 }),
      loadSettings({ gpuOffload: 30, contextLength: 8192 }),
      apu,
      { gpus: [apu] }
    );
    assert.ok(fits);
    assert.equal(fits.willSpill, false);
    assert.ok(fits.warnings.some((w) => /iGPU via GTT/.test(w)), fits.warnings.join(" | "));

    const tinyGtt = { ...apu, gttTotalBytes: 512 * MiB };
    const over = estimateMemory(
      denseCaps({ fileSizeBytes: 2.7 * GiB, blockCount: 30 }),
      loadSettings({ gpuOffload: 30, contextLength: 8192 }),
      tinyGtt,
      { gpus: [tinyGtt] }
    );
    assert.ok(over?.willSpill);
    assert.ok(over!.warnings.some((w) => /Too big for GTT/.test(w)), over!.warnings.join(" | "));
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
    // Dense 2560-hidden at 64k on one Vulkan device: 0.375 GiB driver + 0.25 GiB
    // reserve + ~0.24 GiB activations + ~2.0 GiB graph ≈ 2.9 GiB.
    assert.ok(at64k > 2.5 * GiB, "64k Vulkan compute should stay above 2.5 GiB");
    const dual64k = computeOverheadBytes(2560, 1024, 2048, {
      contextLength: 65536,
      backend: "vulkan",
      flashAttention: "auto",
      vulkanDeviceCount: 2,
    });
    assert.ok(dual64k - at64k > 2 * GiB, "dual-GPU RADV heap is the extra ~2.1 GiB");
  });

  it("sizes a peer-GPU heap below the main-GPU graph", () => {
    const main = computeOverheadBytes(2560, 1024, 2048, {
      contextLength: 65536,
      backend: "vulkan",
      vulkanDeviceCount: 2,
    });
    const peer = peerGpuOverheadBytes(2560, 1024, "vulkan");
    assert.ok(peer > 2 * GiB, "peer GPUs keep the 2.5 GiB dual-device RADV heap");
    assert.ok(peer < main);
    const peerRco = peerGpuOverheadBytes(5120, 256, "vulkan", true);
    assert.ok(peerRco < 1 * GiB, "RCO peers skip the 2.5 GiB Flash-Next heap");
    assert.ok(peerRco > 0.7 * GiB, "RCO peers still bill driver + 256 MiB slop");
  });

  it("uses 384 MiB driver + 256 MiB reserve on one Vulkan device", () => {
    assert.equal(vulkanDeviceReservedBytes(0), 0);
    assert.equal(vulkanDeviceReservedBytes(1), 256 * 1024 ** 2);
    assert.equal(vulkanDeviceReservedBytes(2), 2.5 * GiB);
    assert.equal(vulkanDeviceReservedBytes(2, true), 256 * 1024 ** 2);
    assert.equal(vulkanDriverBytes(0), 0);
    assert.equal(vulkanDriverBytes(1), 384 * 1024 ** 2);
    assert.equal(vulkanDriverBytes(2), 768 * 1024 ** 2);
    const single = computeOverheadBytes(5120, 256, 512, {
      contextLength: 42496,
      backend: "vulkan",
      fullAttentionFraction: 16 / 65,
      discountRecurrentGraph: true,
      vulkanDeviceCount: 1,
    });
    const dual = computeOverheadBytes(5120, 256, 512, {
      contextLength: 42496,
      backend: "vulkan",
      fullAttentionFraction: 16 / 65,
      discountRecurrentGraph: true,
      vulkanDeviceCount: 2,
    });
    assert.equal(dual - single, vulkanDriverBytes(2) - vulkanDriverBytes(1));
    assert.ok(single < 1.6 * GiB, "27B single-card overhead should sit near 1.2 GiB, not 2.5");
    const cuda = computeOverheadBytes(5120, 256, 512, {
      contextLength: 42496,
      backend: "cuda",
      fullAttentionFraction: 16 / 65,
      discountRecurrentGraph: true,
      vulkanDeviceCount: 2,
    });
    assert.ok(cuda < single, "CUDA has no extra RADV heap");
  });

  it("discounts the graph workspace for linear-recurrent hybrids only", () => {
    const dense = computeOverheadBytes(5120, 256, 512, {
      contextLength: 131072,
      backend: "vulkan",
    });
    const hybrid = computeOverheadBytes(5120, 256, 512, {
      contextLength: 131072,
      backend: "vulkan",
      fullAttentionFraction: 16 / 65,
      discountRecurrentGraph: true,
    });
    assert.ok(hybrid < dense, "recurrent layers must not bill attention graph workspace");
    assert.ok(dense - hybrid > 0.5 * GiB, "hybrid discount should be material at 131k");
    const def = computeOverheadBytes(5120, 256, 512, {
      contextLength: 131072,
      backend: "vulkan",
    });
    assert.equal(def, dense);
    const ungated = computeOverheadBytes(5120, 256, 512, {
      contextLength: 131072,
      backend: "vulkan",
      fullAttentionFraction: 16 / 65,
    });
    assert.equal(ungated, dense, "Flash-Next-style SWA interleaves keep the dense graph");
  });

  it("wires the RCO gate through estimateMemory (qwen35 < qwen4exp)", () => {
    const base = {
      blockCount: 65,
      embeddingLength: 5120,
      attentionHeadCount: 40,
      attentionHeadCountKv: 8,
      keyLength: 128,
      valueLength: 128,
      fullAttentionInterval: 4,
      fileSizeBytes: 10 * GiB,
    };
    const settings = loadSettings({
      contextLength: 131072,
      gpuOffload: 99,
      physicalBatchSize: 256,
      evalBatchSize: 512,
      splitMode: "none",
    });
    const rco = estimateMemory(denseCaps({ architecture: "qwen35", ...base }), settings, gpu(48));
    const flash = estimateMemory(
      denseCaps({ architecture: "qwen4exp", ...base }),
      settings,
      gpu(48)
    );
    assert.ok(rco && flash);
    // Same weights, same layer/KV geometry — only the graph discount differs.
    assert.equal(rco.kvBytes, flash.kvBytes);
    assert.ok(
      rco.overheadBytes < flash.overheadBytes,
      `qwen35 overhead ${rco.overheadBytes} should be below qwen4exp ${flash.overheadBytes}`
    );
  });

  it("applies the compact heap to qwen4exp when SSM geometry is present", () => {
    const base = {
      blockCount: 48,
      embeddingLength: 2560,
      attentionHeadCount: 24,
      attentionHeadCountKv: 2,
      keyLength: 256,
      valueLength: 256,
      fullAttentionInterval: 4,
      fileSizeBytes: 10 * GiB,
    };
    const g0 = {
      totalBytes: 16 * GiB,
      usedBytes: 0,
      name: "RX 9070 XT",
      source: "test",
      llamaDeviceId: "Vulkan0",
    };
    const g1 = {
      totalBytes: 16 * GiB,
      usedBytes: 0,
      name: "RX 9060 XT",
      source: "test",
      llamaDeviceId: "Vulkan1",
    };
    const settings = loadSettings({
      contextLength: 65536,
      gpuOffload: 48,
      physicalBatchSize: 256,
      evalBatchSize: 512,
      tensorSplit: "62,38",
    });
    const swa = estimateMemory(
      denseCaps({ architecture: "qwen4exp", ...base }),
      settings,
      g0,
      { gpus: [g0, g1] }
    );
    const ssm = estimateMemory(
      denseCaps({ architecture: "qwen4exp", ssmStateSize: 128, ssmInnerSize: 6144, ...base }),
      settings,
      g0,
      { gpus: [g0, g1] }
    );
    assert.ok(swa && ssm);
    assert.ok(
      ssm.overheadBytes < swa.overheadBytes,
      `qwen4exp+SSM overhead ${ssm.overheadBytes} should drop the Flash-Next heap (${swa.overheadBytes})`
    );
    const oh0 = ssm.charts.vram.segments.find((s) => s.key === "overhead")!.bytes;
    const oh1 = ssm.charts.vram2!.segments.find((s) => s.key === "overhead")!.bytes;
    assert.ok(oh0 < 2.3 * GiB, `main overhead ${oh0} still has the 2.5 GiB Flash-Next heap`);
    assert.ok(oh1 < 1.1 * GiB, `peer overhead ${oh1} still has the 2.5 GiB Flash-Next heap`);
    assert.ok(oh0 > 0.9 * GiB, `main overhead ${oh0} dropped driver/graph`);
  });

  it("prices a single-Vulkan 27B RCO near LACT, not the dual-GPU 2.5 GiB heap", () => {
    const g0 = {
      totalBytes: 16 * GiB,
      usedBytes: 0,
      name: "RX 9070 XT",
      source: "test",
      llamaDeviceId: "Vulkan0",
    };
    const g1 = {
      totalBytes: 16 * GiB,
      usedBytes: 0,
      name: "RX 9060 XT",
      source: "test",
      llamaDeviceId: "Vulkan1",
    };
    const caps = denseCaps({
      architecture: "qwen35",
      fileSizeBytes: Math.round(9.76 * GiB),
      blockCount: 65,
      embeddingLength: 5120,
      fullAttentionInterval: 4,
    });
    const settings = loadSettings({
      contextLength: 42496,
      gpuOffload: 65,
      physicalBatchSize: 256,
      evalBatchSize: 512,
      splitMode: "none",
      cacheTypeK: "q8_0",
      cacheTypeV: "q5_1",
    });
    const est = estimateMemory(caps, settings, g0, { gpus: [g0, g1] });
    assert.ok(est);
    assert.ok(est.overheadBytes < 1.6 * GiB, `overhead ${est.overheadBytes} still has dual-GPU slop`);
    assert.ok(est.overheadBytes > 0.9 * GiB, `overhead ${est.overheadBytes} dropped the real RADV heap`);
    assert.equal(est.charts.vram2?.segments.find((s) => s.key === "overhead")?.bytes ?? 0, 0);
  });

  it("prices dual-Vulkan 27B RCO near LACT, not the 2.5 GiB Flash-Next heap", () => {
    const g0 = {
      totalBytes: 16 * GiB,
      usedBytes: 0,
      name: "RX 9070 XT",
      source: "test",
      llamaDeviceId: "Vulkan0",
    };
    const g1 = {
      totalBytes: 16 * GiB,
      usedBytes: 0,
      name: "RX 9060 XT",
      source: "test",
      llamaDeviceId: "Vulkan1",
    };
    const caps = denseCaps({
      architecture: "qwen35",
      fileSizeBytes: Math.round(16.2 * GiB),
      blockCount: 65,
      embeddingLength: 5120,
      attentionHeadCount: 40,
      attentionHeadCountKv: 8,
      keyLength: 128,
      valueLength: 128,
      fullAttentionInterval: 4,
    });
    const settings = loadSettings({
      contextLength: 61440,
      gpuOffload: 65,
      physicalBatchSize: 256,
      evalBatchSize: 512,
      tensorSplit: "50,50",
      cacheTypeK: "q8_0",
      cacheTypeV: "q5_1",
      mainGpu: 0,
    });
    const est = estimateMemory(caps, settings, g0, { gpus: [g0, g1] });
    assert.ok(est?.charts.vram2);
    const oh0 = est.charts.vram.segments.find((s) => s.key === "overhead")!.bytes;
    const oh1 = est.charts.vram2.segments.find((s) => s.key === "overhead")!.bytes;
    assert.ok(oh0 < 2.3 * GiB, `main overhead ${oh0} still has the 2.5 GiB Flash-Next heap`);
    assert.ok(oh0 > 1.4 * GiB, `main overhead ${oh0} dropped driver/graph`);
    assert.ok(oh1 < 1.1 * GiB, `peer overhead ${oh1} still has the 2.5 GiB Flash-Next heap`);
    assert.ok(oh1 > 0.7 * GiB, `peer overhead ${oh1} dropped the 256 MiB slop`);
    assert.ok(
      est.charts.vram.totalBytes < 13 * GiB,
      `9070 bar ${est.charts.vram.totalBytes} should sit near LACT 11.4 GiB, not 13.9`
    );
    assert.ok(
      est.charts.vram2.totalBytes < 13 * GiB,
      `9060 bar ${est.charts.vram2.totalBytes} should sit near live ~11 GiB, not 13.7`
    );
  });

  it("prices Ornith-class 9B RCO overhead near LACT (~1.4 GiB, not ~2.5)", () => {
    const g0 = {
      totalBytes: 16 * GiB,
      usedBytes: 0,
      name: "RX 9070 XT",
      source: "test",
      llamaDeviceId: "Vulkan0",
    };
    const g1 = {
      totalBytes: 16 * GiB,
      usedBytes: 0,
      name: "RX 9060 XT",
      source: "test",
      llamaDeviceId: "Vulkan1",
    };
    const caps = denseCaps({
      architecture: "qwen35",
      fileSizeBytes: Math.round(9.5 * GiB),
      blockCount: 32,
      embeddingLength: 4096,
      attentionHeadCount: 32,
      attentionHeadCountKv: 8,
      keyLength: 128,
      valueLength: 128,
      fullAttentionInterval: 4,
    });
    const settings = loadSettings({
      contextLength: 48128,
      gpuOffload: 99,
      physicalBatchSize: 512,
      evalBatchSize: 2048,
      splitMode: "none",
      cacheTypeK: "f16",
      cacheTypeV: "f16",
    });
    const est = estimateMemory(caps, settings, g0, { gpus: [g0, g1] });
    assert.ok(est);
    assert.ok(est.kvBytes > 1.4 * GiB && est.kvBytes < 1.55 * GiB, `KV ${est.kvBytes}`);
    assert.ok(
      est.overheadBytes < 1.55 * GiB,
      `overhead ${est.overheadBytes} is still ~2× the 12.4 vs 13.6 GiB LACT gap`
    );
    assert.ok(est.overheadBytes > 1.1 * GiB, `overhead ${est.overheadBytes} undershot compute+RADV`);
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
    assert.equal(off!.gpuKvBytes, 0);
    assert.equal(off!.cpuKvBytes, off!.kvBytes);
  });

  it("puts all KV in VRAM at full offload", () => {
    const est = estimateMemory(denseCaps(), loadSettings({ gpuOffload: 99 }), gpu(48))!;
    assert.equal(est.gpuKvBytes, est.kvBytes);
    assert.equal(est.cpuKvBytes, 0);
  });

  it("bills only the offloaded layers' KV to VRAM under partial offload", () => {
    const full = estimateMemory(denseCaps(), loadSettings({ gpuOffload: 99 }), gpu(48))!;
    const half = estimateMemory(denseCaps(), loadSettings({ gpuOffload: 24 }), gpu(48))!;
    // Uniform layers: 24/48 of the cache follows the offloaded layers.
    assert.ok(Math.abs(half.gpuKvBytes - full.kvBytes / 2) < 1024);
    assert.ok(Math.abs(half.cpuKvBytes - full.kvBytes / 2) < 1024);
    assert.equal(half.gpuKvBytes + half.cpuKvBytes, half.kvBytes);
    assert.ok(half.totalGpuBytes < full.totalGpuBytes);
    assert.ok(half.totalCpuBytes > full.totalCpuBytes);
    assert.ok(half.warnings.some((w) => /KV cache follows the layers/.test(w)));
    assert.equal(half.charts.ram.segments.find((s) => s.key === "kv")?.bytes, half.cpuKvBytes);
  });

  it("uses per-layer KV sizes when splitting: the last (offloaded) layers may be SWA", () => {
    // Layers 0-11 dense/full, 12-23 sliding-window: offloading the last 12
    // puts only the small SWA caches on the GPU.
    const caps = denseCaps({
      blockCount: 24,
      slidingWindow: 1024,
      slidingWindowPattern: Array.from({ length: 24 }, (_, i) => i >= 12),
    });
    const est = estimateMemory(
      caps,
      loadSettings({ gpuOffload: 12, contextLength: 32768 }),
      gpu(48)
    )!;
    assert.ok(est.gpuKvBytes < est.cpuKvBytes / 8, "SWA layers must be much smaller than full ones");
  });

  it("does not spill a partially-offloaded model whose KV would only overflow if all of it were in VRAM", () => {
    // 12 GiB card, 18 GiB model, 24/48 layers → 9 GiB weights on GPU. Full KV
    // at 64k ≈ 3 GiB (q8_0, 48×8×128×2). Billing all KV to the GPU would spill;
    // only half of it really lands there.
    const est = estimateMemory(
      denseCaps(),
      loadSettings({ gpuOffload: 24, contextLength: 65536 }),
      gpu(12)
    )!;
    assert.ok(est.gpuKvBytes < est.kvBytes);
    assert.ok(
      est.totalGpuBytes < 9 * GiB + est.kvBytes,
      "total VRAM must not include the CPU layers' KV"
    );
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
    // Regression: the GPU line used to read `gpuWeightsBytes || cpuWeightsBytes`
    // (both 0 by design) and rendered "MTP: ~0 B next-n head".
    const mtpLine = withMtp.lines.find((l) => l.startsWith("MTP:"));
    assert.ok(mtpLine, "expected an MTP line");
    assert.ok(!/~0 B/.test(mtpLine!), `MTP line should show the head size, got: ${mtpLine}`);
    assert.ok(mtpLine!.includes("next-n heads already in the GGUF weights"), mtpLine!);
  });

  it("sizes a sidecar MTP draft from its next-n layers, not the parent block_count", () => {
    // Real file: mtp-Qwen3.8-27B-Q4_0.gguf is 1.28 GiB and declares the parent's
    // block_count (65) + interval (4) while holding a single next-n head.
    const draft = denseCaps({
      name: "mtp-draft",
      architecture: "qwen35",
      fileSizeBytes: Math.round(1.28 * GiB),
      blockCount: 65,
      embeddingLength: 5120,
      attentionHeadCount: 40,
      attentionHeadCountKv: 4,
      keyLength: 128,
      valueLength: 128,
      fullAttentionInterval: 4,
      nextnPredictLayers: 1,
    });
    const settings = loadSettings({
      speculativeMode: "mtp",
      draftModelPath: "/models/mtp-Qwen3.8-27B-Q4_0.gguf",
      draftGpuOffload: 99,
      contextLength: 83200,
      cacheTypeK: "f16",
      cacheTypeV: "f16",
    });
    const est = estimateMemory(denseCaps(), settings, gpu(48), { draftCaps: draft });
    assert.ok(est);
    assert.equal(est.draftFileSizeBytes, draft.fileSizeBytes);
    // One next-n layer of f16 KV: 4 kv heads × 128 dims × 2 B × 2 (K+V) × 83200.
    const oneLayer = 4 * 128 * 2 * 2 * 83200;
    assert.ok(
      Math.abs((est.draftKvBytes || 0) - oneLayer) < 1024,
      `draft KV ${est.draftKvBytes} should be one next-n layer (~${oneLayer})`,
    );
    // The old geometry billed all 16 full-attn layers of the parent's mask.
    assert.ok(
      (est.draftKvBytes || 0) < oneLayer * 2,
      `draft KV ${est.draftKvBytes} still bills the parent's 16 full-attn layers`,
    );
  });

  it("keeps a DFlash draft's own layer geometry", () => {
    const draft = denseCaps({
      name: "dflash-draft",
      architecture: "dflash",
      fileSizeBytes: Math.round(1.28 * GiB),
      blockCount: 8,
      nextnPredictLayers: 1,
      fullAttentionInterval: undefined,
    });
    const est = estimateMemory(
      denseCaps(),
      loadSettings({
        speculativeMode: "dflash",
        draftModelPath: "/models/draft.gguf",
        draftGpuOffload: 99,
        contextLength: 83200,
        cacheTypeK: "f16",
        cacheTypeV: "f16",
      }),
      gpu(48),
      { draftCaps: draft }
    );
    // All 8 draft layers keep cache — DFlash is a full model, not an MTP head.
    // denseCaps geometry: 8 kv heads × 128 dims × 2 B × 2 (K+V) × 83200.
    const perLayer = 8 * 128 * 2 * 2 * 83200;
    assert.ok(
      (est?.draftKvBytes || 0) > perLayer * 7,
      `DFlash draft KV ${est?.draftKvBytes} must cover every draft layer`,
    );
  });

  it("splits the speculative draft across GPUs like --tensor-split", () => {
    const g0 = { totalBytes: 16 * GiB, usedBytes: 0, name: "RX 9070 XT", source: "t", llamaDeviceId: "Vulkan0" };
    const g1 = { totalBytes: 16 * GiB, usedBytes: 0, name: "RX 9060 XT", source: "t", llamaDeviceId: "Vulkan1" };
    const draft = denseCaps({
      name: "mtp-draft",
      architecture: "qwen35",
      fileSizeBytes: Math.round(1.28 * GiB),
      blockCount: 65,
      fullAttentionInterval: 4,
      nextnPredictLayers: 1,
    });
    const est = estimateMemory(
      denseCaps({ architecture: "qwen35", blockCount: 65, fullAttentionInterval: 4 }),
      loadSettings({
        speculativeMode: "mtp",
        draftModelPath: "/models/mtp-Qwen3.8-27B-Q4_0.gguf",
        draftGpuOffload: 99,
        contextLength: 32768,
        tensorSplit: "44,56",
        splitMode: "layer",
        mainGpu: 0,
      }),
      g0,
      { gpus: [g0, g1], draftCaps: draft }
    );
    assert.ok(est?.charts.vram2);
    const draft0 = est.charts.vram.segments.find((s) => s.key === "draft")?.bytes ?? 0;
    const draft1 = est.charts.vram2!.segments.find((s) => s.key === "draft")?.bytes ?? 0;
    assert.ok(draft0 > 0, "main GPU keeps its share of the draft");
    // Regression: the whole draft used to be parked on Main, which pushed the
    // 9070 XT bar to 118% of a 16 GB card while sysfs showed 12 GiB.
    assert.ok(draft1 > 0, "the second GPU must carry its share of the draft too");
    const draftTotal = (est.draftFileSizeBytes || 0) + (est.draftKvBytes || 0);
    assert.ok(
      Math.abs(draft0 + draft1 - draftTotal) < 2048,
      `draft bundle must be conserved: ${draft0} + ${draft1} vs ${draftTotal}`,
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
    assert.ok(est.charts.vram.totalBytes <= 16 * GiB - VRAM_HEADROOM_BYTES);
    assert.ok((est.charts.vram2?.totalBytes || 0) <= 16 * GiB - VRAM_HEADROOM_BYTES);
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

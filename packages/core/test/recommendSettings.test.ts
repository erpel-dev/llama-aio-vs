import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { mainShareFromSplit } from "../src/gpuSplit";
import { estimateMemory } from "../src/memoryEstimate";
import { PREFERRED_CONTEXT, fittingContextLength, recommendLoadSettings } from "../src/recommendSettings";
import { denseCaps, GiB, loadSettings, moeCaps } from "./helpers";

const gpu = (totalGiB: number) => ({
  totalBytes: totalGiB * GiB,
  usedBytes: 0,
  name: "test",
  source: "test",
});

describe("recommendLoadSettings", () => {
  it("caps context at the model maximum", () => {
    const r = recommendLoadSettings(loadSettings(), denseCaps({ maxContextLength: 8192 }), {
      gpu: gpu(48),
    });
    assert.equal(r.contextLength, 8192);
  });

  it("prefers the agent context when the model allows it", () => {
    const r = recommendLoadSettings(loadSettings(), denseCaps(), { gpu: gpu(48) });
    assert.equal(r.contextLength, PREFERRED_CONTEXT);
  });

  it("offloads everything when the model fits", () => {
    const r = recommendLoadSettings(loadSettings(), denseCaps(), { gpu: gpu(80) });
    assert.equal(r.gpuOffload, 99);
  });

  it("reduces layers on a small GPU instead of overcommitting", () => {
    const r = recommendLoadSettings(loadSettings(), denseCaps(), { gpu: gpu(12) });
    assert.ok(r.gpuOffload < 48, `expected partial offload, got ${r.gpuOffload}`);
    const est = estimateMemory(denseCaps(), r, gpu(12));
    assert.ok(est!.totalGpuBytes < 12 * GiB);
  });

  it("raises --n-cpu-moe rather than dropping layers for MoE models", () => {
    const r = recommendLoadSettings(loadSettings(), moeCaps(), { gpu: gpu(12) });
    assert.equal(r.gpuOffload, 99);
    assert.ok(r.nCpuMoe > 0);
  });

  it("raises --n-cpu-ffn before dropping layers for dense models when it fits", () => {
    // 24 GiB card: full model + KV (~25 GiB) does not fit; FFN-on-CPU does.
    const caps = denseCaps({ ffnLength: 13824 });
    const r = recommendLoadSettings(loadSettings(), caps, { gpu: gpu(24) });
    assert.equal(r.gpuOffload, 99, "expected full layer offload with FFN on CPU");
    assert.ok(r.nCpuFfn > 0, "expected --n-cpu-ffn > 0 when all layers stay on GPU");
    const est = estimateMemory(caps, r, gpu(24));
    assert.ok(est && !est.willSpill);
  });

  it("resets --n-cpu-ffn when recommending for an MoE model", () => {
    const r = recommendLoadSettings(loadSettings({ nCpuFfn: 12 }), moeCaps(), { gpu: gpu(80) });
    assert.equal(r.nCpuFfn, 0);
  });

  it("preserves DFlash when recommending for a non-MTP target", () => {
    const withDflash = recommendLoadSettings(
      loadSettings({
        speculativeMode: "dflash",
        draftModelPath: "/models/draft.gguf",
        maxDraftTokens: 15,
      }),
      denseCaps(),
      { gpu: gpu(48) }
    );
    assert.equal(withDflash.speculativeMode, "dflash");
    assert.equal(withDflash.draftModelPath, "/models/draft.gguf");
  });

  it("preserves DFlash even when the new GGUF reports MTP next-n layers", () => {
    const withDflash = recommendLoadSettings(
      loadSettings({
        speculativeMode: "dflash",
        draftModelPath: "/models/draft.gguf",
        maxDraftTokens: 15,
      }),
      denseCaps({ nextnPredictLayers: 1 }),
      { gpu: gpu(48) }
    );
    assert.equal(withDflash.speculativeMode, "dflash");
    assert.equal(withDflash.draftModelPath, "/models/draft.gguf");
    assert.equal(withDflash.maxDraftTokens, 15);
  });

  it("replaces a leftover DFlash n-max when recommending MTP", () => {
    const r = recommendLoadSettings(
      loadSettings({ speculativeMode: "mtp", maxDraftTokens: 15 }),
      denseCaps({ nextnPredictLayers: 1 }),
      { gpu: gpu(48) }
    );
    assert.equal(r.speculativeMode, "mtp");
    assert.equal(r.maxDraftTokens, 2);
  });

  it("replaces a leftover MTP n-max when recommending DFlash", () => {
    const r = recommendLoadSettings(
      loadSettings({
        speculativeMode: "dflash",
        draftModelPath: "/models/draft.gguf",
        maxDraftTokens: 2,
      }),
      denseCaps(),
      { gpu: gpu(48) }
    );
    assert.equal(r.speculativeMode, "dflash");
    assert.equal(r.maxDraftTokens, 15);
  });

  it("does not enable MTP from a Flash-Next mtp-* sidecar", () => {
    const r = recommendLoadSettings(
      loadSettings({
        speculativeMode: "mtp",
        draftModelPath: "/models/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf",
      }),
      moeCaps({ architecture: "qwen4exp", nextnPredictLayers: 0, pleShare: 0.4 }),
      { gpu: gpu(48) }
    );
    assert.equal(r.speculativeMode, "off");
    assert.equal(r.draftModelPath, "");
  });

  it("enables MTP only for models that report next-n layers", () => {
    const withMtp = recommendLoadSettings(loadSettings(), denseCaps({ nextnPredictLayers: 1 }), {
      gpu: gpu(48),
    });
    assert.equal(withMtp.speculativeMode, "mtp");
    const without = recommendLoadSettings(
      loadSettings({ speculativeMode: "mtp" }),
      denseCaps(),
      { gpu: gpu(48) }
    );
    assert.equal(without.speculativeMode, "off");
  });

  it("stacks n-gram onto MTP when the GGUF is MTP-capable", () => {
    const r = recommendLoadSettings(
      loadSettings({ speculativeMode: "ngram" }),
      denseCaps({ nextnPredictLayers: 1 }),
      { gpu: gpu(48) }
    );
    assert.equal(r.speculativeMode, "ngram-mtp");
  });

  it("keeps n-gram when recommending for a non-MTP target", () => {
    const r = recommendLoadSettings(loadSettings({ speculativeMode: "ngram" }), denseCaps(), {
      gpu: gpu(48),
    });
    assert.equal(r.speculativeMode, "ngram");
  });

  it("preserves N-gram + DFlash when recommending", () => {
    const r = recommendLoadSettings(
      loadSettings({
        speculativeMode: "ngram-dflash",
        draftModelPath: "/models/draft.gguf",
        maxDraftTokens: 15,
      }),
      denseCaps({ nextnPredictLayers: 1 }),
      { gpu: gpu(48) }
    );
    assert.equal(r.speculativeMode, "ngram-dflash");
    assert.equal(r.draftModelPath, "/models/draft.gguf");
  });

  describe("physical batch tuning", () => {
    it("raises -ub when the model is fully offloaded with room to spare", () => {
      const r = recommendLoadSettings(loadSettings(), denseCaps(), { gpu: gpu(80) });
      assert.equal(r.physicalBatchSize, 1024);
    });

    it("leaves -ub alone on partial offload", () => {
      const r = recommendLoadSettings(loadSettings(), denseCaps(), { gpu: gpu(12) });
      assert.equal(r.physicalBatchSize, 512);
    });

    it("leaves -ub alone on the CPU backend", () => {
      const r = recommendLoadSettings(loadSettings(), denseCaps(), { cpuOnly: true });
      assert.equal(r.physicalBatchSize, 512);
    });

    it("never lets -ub exceed -b", () => {
      const r = recommendLoadSettings(loadSettings({ evalBatchSize: 512 }), denseCaps(), {
        gpu: gpu(80),
      });
      assert.ok(r.physicalBatchSize <= r.evalBatchSize);
    });
  });

  it("leaves GPU knobs untouched on the CPU backend", () => {
    const before = loadSettings({ gpuOffload: 33, nCpuMoe: 7 });
    const r = recommendLoadSettings(before, moeCaps(), { cpuOnly: true });
    assert.equal(r.gpuOffload, 33);
    assert.equal(r.nCpuMoe, 7);
  });

  describe("dual GPU", () => {
    const two16 = [gpu(16), gpu(16)];

    it("splits a dense model across both cards instead of spilling to RAM", () => {
      // One 16 GB card: 18 GiB weights + 64k KV + compute do not leave 2 GiB
      // free even with --n-cpu-ffn. Two cards keep every layer on GPU.
      const one = recommendLoadSettings(loadSettings(), denseCaps(), { gpu: gpu(16) });
      const oneEst = estimateMemory(denseCaps(), one, two16[0]);
      assert.ok(oneEst && !oneEst.willSpill);
      assert.ok(
        one.gpuOffload < 99 || one.nCpuFfn > 0,
        `expected a RAM spill of layers or FFN on one card, got gpuOffload ${one.gpuOffload}, nCpuFfn ${one.nCpuFfn}`
      );

      const two = recommendLoadSettings(loadSettings(), denseCaps(), { gpus: two16 });
      assert.equal(two.gpuOffload, 99);
      assert.equal(two.nCpuMoe, 0);
      assert.ok(two.tensorSplit, "expected an explicit tensor split");
      assert.equal(two.splitMode, "layer");
      const est = estimateMemory(denseCaps(), two, two16[0], { gpus: two16 });
      assert.ok(est && !est.willSpill);
    });

    it("parks a small model entirely on Main instead of equalizing occupancy", () => {
      const two80 = [gpu(80), gpu(80)];
      const r = recommendLoadSettings(loadSettings({ mainGpu: 0 }), denseCaps(), { gpus: two80 });
      assert.equal(r.gpuOffload, 99);
      assert.equal(r.splitMode, "none");
      const flipped = recommendLoadSettings(loadSettings({ mainGpu: 1 }), denseCaps(), {
        gpus: two80,
      });
      assert.equal(flipped.splitMode, "none");
      assert.equal(flipped.mainGpu, 1);
    });

    it("fills the larger card first on mixed VRAM", () => {
      const mixed = [gpu(32), gpu(12)];
      const caps = denseCaps({ fileSizeBytes: 26 * GiB, blockCount: 48 });
      const r = recommendLoadSettings(loadSettings({ mainGpu: 0 }), caps, { gpus: mixed });
      assert.equal(r.gpuOffload, 99);
      assert.equal(r.splitMode, "layer");
      const vram = [32 * GiB, 12 * GiB];
      const share = mainShareFromSplit(r.tensorSplit, 0, 2, vram);
      assert.ok(share >= 0.7, `expected most weights on the 32 GiB Main, got ${share} (${r.tensorSplit})`);
      const other = mainShareFromSplit(r.tensorSplit, 1, 2, vram);
      assert.ok(other > 0, "the 12 GiB card should still take the overflow");
    });

    it("leaves the middle card idle on 3 GPUs when two cards are enough (back-to-front)", () => {
      const three = [gpu(16), gpu(16), gpu(16)];
      const r = recommendLoadSettings(loadSettings({ mainGpu: 0 }), denseCaps(), { gpus: three });
      assert.equal(r.gpuOffload, 99);
      const parts = (r.tensorSplit || "").split(",").map(Number);
      if (r.splitMode === "none") {
        return;
      }
      assert.equal(parts.length, 3);
      assert.equal(parts[1], 0, `expected GPU 1 idle, got ${r.tensorSplit}`);
      assert.ok(parts[2] > 0, `expected overflow on GPU 2, got ${r.tensorSplit}`);
    });

    it("clears a leftover tensor split when recommending for a single GPU", () => {
      const r = recommendLoadSettings(loadSettings({ tensorSplit: "50,50" }), denseCaps(), {
        gpu: gpu(80),
      });
      assert.equal(r.tensorSplit, "");
    });

    it("still drops layers when even both cards cannot hold the model", () => {
      const huge = denseCaps({ fileSizeBytes: 80 * GiB, blockCount: 80 });
      const r = recommendLoadSettings(loadSettings(), huge, { gpus: two16 });
      assert.ok(r.gpuOffload < 80, `expected partial offload, got ${r.gpuOffload}`);
      assert.ok(r.tensorSplit, "keep a split for the layers that do fit");
    });

    it("avoids --n-cpu-moe on two GPUs when one small card needed it", () => {
      const one = recommendLoadSettings(loadSettings(), moeCaps(), { gpu: gpu(12) });
      assert.ok(one.nCpuMoe > 0);
      const two = recommendLoadSettings(loadSettings(), moeCaps(), { gpus: two16 });
      assert.equal(two.gpuOffload, 99);
      assert.equal(two.nCpuMoe, 0);
      assert.ok(two.tensorSplit);
    });
  });

  describe("fittingContextLength", () => {
    it("returns the model max when VRAM is plentiful", () => {
      const r = fittingContextLength(denseCaps(), loadSettings(), { gpu: gpu(80) });
      assert.equal(r, 262144);
    });

    it("finds an aligned context between 8k and the model max on a tight GPU", () => {
      const settings = loadSettings({ cacheTypeK: "q8_0", cacheTypeV: "q4_0" });
      const r = fittingContextLength(denseCaps(), settings, { gpu: gpu(24) });
      assert.ok(r >= 8192, `expected at least 8k, got ${r}`);
      assert.equal(r % 256, 0);
      const est = estimateMemory(denseCaps(), { ...settings, contextLength: r }, gpu(24));
      assert.equal(est?.willSpill, false);
      const next = Math.min(262144, r + 256);
      if (next > r) {
        const over = estimateMemory(denseCaps(), { ...settings, contextLength: next }, gpu(24));
        // Either next still fits (then we would have picked it) or it spills.
        if (over && !over.willSpill) {
          assert.equal(r, 262144, "search should have kept going if next still fits");
        }
      }
    });

    it("falls back to 8k when even that spills", () => {
      const r = fittingContextLength(denseCaps(), loadSettings(), { gpu: gpu(8) });
      assert.equal(r, 8192);
    });
  });
});

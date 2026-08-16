import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { estimateMemory } from "../src/memoryEstimate";
import { PREFERRED_CONTEXT, recommendLoadSettings } from "../src/recommendSettings";
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
      const one = recommendLoadSettings(loadSettings(), denseCaps(), { gpu: gpu(16) });
      assert.ok(one.gpuOffload < 99, `expected partial offload on one 16 GiB GPU, got ${one.gpuOffload}`);

      const two = recommendLoadSettings(loadSettings(), denseCaps(), { gpus: two16 });
      assert.equal(two.gpuOffload, 99);
      assert.equal(two.nCpuMoe, 0);
      assert.ok(two.tensorSplit, "expected an explicit tensor split");
      assert.equal(two.splitMode, "layer");
      const est = estimateMemory(denseCaps(), two, two16[0], { gpus: two16 });
      assert.ok(est && !est.willSpill);
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
});

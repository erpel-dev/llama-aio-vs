import assert from "node:assert/strict";
import { it } from "node:test";
import { estimateMemory } from "../src/memoryEstimate";
import { buildServerArgs } from "../src/serverArgs";
import { argValue, GiB, loadSettings, moeCaps } from "./helpers";

it("only reorders per-card estimates when launch emits the same device order", () => {
  for (const mode of ["auto", "explicit", "mismatched", "missing-id", "sycl-row"] as const) {
    const sycl = mode === "sycl-row";
    const gpus = [2, 16].map((pcieGBps, i) => ({
      totalBytes: 16 * GiB, usedBytes: 0, source: "test", name: `GPU ${i}`, pcieGBps,
      llamaDeviceId: mode === "missing-id" && i === 1 ? undefined : `${sycl ? "SYCL" : "CUDA"}${i}`,
    }));
    const settings = loadSettings({ gpuOffload: 99, nCpuMoe: 24,
      splitMode: sycl ? "row" : "layer",
      tensorSplit: mode === "auto" ? "" : mode === "mismatched" ? "1,1,1" : "1,1" });
    const caps = moeCaps();
    const args = buildServerArgs("/m/model.gguf", "127.0.0.1", 8742, settings, { gpus, caps });
    const est = estimateMemory(caps, settings, gpus[0], { gpus })!;
    const baselineGpus = gpus.map(g => ({ ...g, pcieGBps: undefined }));
    const baseline = estimateMemory(caps, settings, baselineGpus[0], { gpus: baselineGpus })!;
    const totals = (e: typeof est) => e.charts.gpus!.map(g => g.totalBytes);
    if (mode === "explicit" || sycl) {
      assert.equal(argValue(args, "--device"), `${sycl ? "SYCL" : "CUDA"}1,${sycl ? "SYCL" : "CUDA"}0`);
      assert.equal(argValue(args, "--split-mode"), "layer");
      assert.notDeepEqual(totals(est), totals(baseline));
      if (sycl) {
        const layer = estimateMemory(caps, { ...settings, splitMode: "layer" }, gpus[0], { gpus })!;
        assert.deepEqual(totals(est), totals(layer));
      }
    } else {
      assert.equal(argValue(args, "--device"), undefined, mode);
      assert.deepEqual(totals(est), totals(baseline), mode);
    }
  }
});

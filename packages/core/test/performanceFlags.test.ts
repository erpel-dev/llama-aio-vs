import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { NO_SERVER_FEATURES, type LlamaServerFeatures } from "../src/llamaFeatures";
import { buildServerArgs, normalizeLoadSettingsForCpuBackend, performanceFeatureNotes,
  serverConfigFingerprint, supportedPerformanceSettings } from "../src/serverArgs";
import { estimateMemory } from "../src/memoryEstimate";
import { normalizeLoadSettings } from "../src/types";
import { argValue, denseCaps, loadSettings, moeCaps, GiB } from "./helpers";

const features: LlamaServerFeatures = { probed: true, moeCache: true, draftSampling: true, draftUbatch: true };
const caps = moeCaps();
const gpu = (name: string, totalBytes = 24 * GiB) => ({ name, totalBytes, source: "test" });
const build = (over = {}, detected: LlamaServerFeatures | undefined = features, modelCaps = caps) =>
  buildServerArgs("test.gguf", "127.0.0.1", 8742, loadSettings(over), { features: detected, caps: modelCaps });

describe("guarded performance flags", () => {
  for (const detected of [undefined, { ...NO_SERVER_FEATURES }, { ...NO_SERVER_FEATURES, probed: true }]) {
    it(`omits all new flags without advertised support (${JSON.stringify(detected)})`, () => {
      // Pass undefined directly: the helper otherwise defaults to supported.
      const args = buildServerArgs("m.gguf", "127.0.0.1", 8742, loadSettings({ nCpuMoe: 48,
        moeCacheMiB: 4096, speculativeMode: "mtp", draftSampling: "probabilistic", draftBatchSize: 64 }),
      { features: detected, caps });
      for (const flag of ["--moe-cache-mib", "--spec-draft-sampling", "--spec-draft-n-ubatch"]) {
        assert.ok(!args.includes(flag), flag);
      }
    });
  }
  it("guards each feature independently for intermediate builds", () => {
    const args = build({ nCpuMoe: 48, moeCacheMiB: 4096, speculativeMode: "mtp", draftBatchSize: 64 },
      { ...features, draftUbatch: false });
    assert.equal(argValue(args, "--moe-cache-mib"), "4096");
    assert.equal(argValue(args, "--spec-draft-sampling"), "probabilistic");
    assert.ok(!args.includes("--spec-draft-n-ubatch"));
  });
  it("keeps expert caching off and inherits draft batch by default; Auto enables probabilistic MTP", () => {
    const args = build({ speculativeMode: "mtp" });
    assert.ok(!args.includes("--moe-cache-mib"));
    assert.ok(!args.includes("--spec-draft-n-ubatch"));
    assert.equal(argValue(args, "--spec-draft-sampling"), "probabilistic");
  });
  for (const mode of ["off", "ngram"] as const) {
    it(`omits neural draft knobs in ${mode} mode`, () => {
      const args = build({ speculativeMode: mode, draftSampling: "probabilistic", draftBatchSize: 128 });
      assert.ok(!args.includes("--spec-draft-sampling"));
      assert.ok(!args.includes("--spec-draft-n-ubatch"));
    });
  }
  it("supports stacked MTP, explicit greedy and a separate draft batch", () => {
    const args = build({ speculativeMode: "ngram-mtp", draftSampling: "greedy", draftBatchSize: 96 });
    assert.equal(argValue(args, "--spec-draft-sampling"), "greedy");
    assert.equal(argValue(args, "--spec-draft-n-ubatch"), "96");
  });
  it("leaves DFlash sampling unchanged even on builds advertising the shared sampling flag", () => {
    const base = { speculativeMode: "ngram-dflash", draftModelPath: "draft.gguf", draftBatchSize: 64 };
    const args = build(base);
    assert.ok(!args.includes("--spec-draft-sampling"));
    assert.equal(argValue(args, "--spec-draft-n-ubatch"), "64");
    assert.equal(argValue(args, "--cache-type-k-draft"), "f16");
    assert.ok(!build({ ...base, draftSampling: "probabilistic" }).includes("--spec-draft-sampling"));
    const missing = build({ ...base, draftModelPath: "" });
    assert.ok(!missing.includes("--spec-draft-n-ubatch"));
  });
  for (const [name, over, model] of [
    ["dense model", { nCpuMoe: 48 }, denseCaps()],
    ["no host experts", { nCpuMoe: 0 }, caps],
    ["CPU-only", { nCpuMoe: 48, gpuOffload: 0 }, caps],
    ["CPU experts outside GPU-assigned layers", { nCpuMoe: 12, gpuOffload: 24 }, caps],
    ["unsupported tensor parallelism", { nCpuMoe: 48, splitMode: "tensor" }, caps],
  ] as const) {
    it(`omits expert caching for ${name}`, () => {
      assert.ok(!build({ ...over, moeCacheMiB: 4096 }, features, model).includes("--moe-cache-mib"));
    });
  }
  it("supports expert caching where CPU experts and GPU layers overlap", () => {
    assert.equal(argValue(build({ nCpuMoe: 30, gpuOffload: 24, moeCacheMiB: 4096 }), "--moe-cache-mib"), "4096");
  });
  it("normalizes new settings, preserves choices and disables unsupported features for estimates", () => {
    assert.equal(normalizeLoadSettings({ moeCacheMiB: -1 }).moeCacheMiB, 0);
    assert.equal(normalizeLoadSettings({ draftBatchSize: NaN }).draftBatchSize, 0);
    assert.equal(normalizeLoadSettings({ draftSampling: "unknown" as never }).draftSampling, "auto");
    const saved = loadSettings({ moeCacheMiB: 4096, draftSampling: "probabilistic", draftBatchSize: 64 });
    const effective = supportedPerformanceSettings(saved, NO_SERVER_FEATURES);
    assert.equal(effective.moeCacheMiB, 0);
    assert.equal(effective.draftSampling, "greedy");
    assert.equal(effective.draftBatchSize, 0);
    assert.equal(saved.moeCacheMiB, 4096);
    assert.equal(normalizeLoadSettingsForCpuBackend(saved).moeCacheMiB, 0);
  });
  it("fingerprints active settings and logs skipped features", () => {
    const original = loadSettings({ speculativeMode: "mtp" });
    for (const patch of [{ moeCacheMiB: 1024 }, { draftSampling: "greedy" as const }, { draftBatchSize: 64 }]) {
      assert.notEqual(serverConfigFingerprint("m", original), serverConfigFingerprint("m", { ...original, ...patch }));
    }
    const requested = loadSettings({ speculativeMode: "mtp", nCpuMoe: 48, moeCacheMiB: 4096, draftBatchSize: 64 });
    assert.equal(performanceFeatureNotes(requested, NO_SERVER_FEATURES, caps).length, 3);
    assert.deepEqual(performanceFeatureNotes(requested, features, caps), []);
  });
});

describe("GPU expert cache memory budget", () => {
  it("adds VRAM to full/warm estimates and bars without removing RAM weights", () => {
    const settings = loadSettings({ nCpuMoe: 48 });
    const original = estimateMemory(caps, settings, gpu("CUDA0"))!;
    const cached = estimateMemory(caps, { ...settings, moeCacheMiB: 4096 }, gpu("CUDA0"))!;
    assert.equal(cached.moeCacheBytes, 4 * GiB);
    assert.equal(cached.totalGpuBytes - original.totalGpuBytes, 4 * GiB);
    assert.equal(cached.totalGpuBytesWarm - original.totalGpuBytesWarm, 4 * GiB);
    assert.equal(cached.totalCpuBytes, original.totalCpuBytes);
    assert.equal(cached.charts.vram.totalBytes, cached.totalGpuBytes);
    assert.equal(cached.charts.vram.segments.reduce((n, s) => n + s.bytes, 0), cached.totalGpuBytes);
  });
  it("reserves the whole budget on the sole GPU with eligible experts", () => {
    const gpus = [gpu("CUDA0"), gpu("CUDA1")];
    const settings = loadSettings({ nCpuMoe: 24, tensorSplit: "1,1", splitMode: "layer" });
    const original = estimateMemory(caps, settings, gpus[0], { gpus })!;
    const cached = estimateMemory(caps, { ...settings, moeCacheMiB: 4096 }, gpus[0], { gpus })!;
    const delta = cached.charts.gpus!.map((chart, i) => chart.totalBytes - original.charts.gpus![i]!.totalBytes);
    assert.deepEqual(delta, [4 * GiB, 0]);
  });
  it("splits the cache budget across GPUs with eligible experts", () => {
    const gpus = [gpu("CUDA0"), gpu("CUDA1")];
    const settings = loadSettings({ nCpuMoe: 48, tensorSplit: "1,1", splitMode: "layer" });
    const original = estimateMemory(caps, settings, gpus[0], { gpus })!;
    const cached = estimateMemory(caps, { ...settings, moeCacheMiB: 4096 }, gpus[0], { gpus })!;
    assert.deepEqual(cached.charts.gpus!.map((c, i) => c.totalBytes - original.charts.gpus![i]!.totalBytes), [2 * GiB, 2 * GiB]);
  });
  it("includes the budget in spill detection and ignores it for CPU/dense/tensor modes", () => {
    const settings = loadSettings({ nCpuMoe: 48 });
    const base = estimateMemory(caps, settings, gpu("CUDA0"))!;
    const limited = gpu("CUDA0", base.totalGpuBytes + 2 * GiB);
    assert.equal(estimateMemory(caps, settings, limited)!.willSpill, false);
    assert.equal(estimateMemory(caps, { ...settings, moeCacheMiB: 4096 }, limited)!.willSpill, true);
    const cached = { ...settings, moeCacheMiB: 4096 };
    assert.equal(estimateMemory(caps, cached, limited, { cpuOnly: true })!.moeCacheBytes, undefined);
    assert.equal(estimateMemory(denseCaps(), cached, limited)!.moeCacheBytes, undefined);
    assert.equal(estimateMemory(caps, { ...cached, splitMode: "tensor" }, limited)!.moeCacheBytes, undefined);
  });
});

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { estimateMemory, VRAM_HEADROOM_BYTES } from "../src/memoryEstimate";
import { buildMemoryView, computeMemoryView, shortGpuName, suggestMemoryFixes } from "../src/memoryView";
import { diffLoadSettings } from "../src/settingsDiff";
import { friendlyModelTitle } from "../src/modelLibrary";
import { denseCaps, GiB, loadSettings, moeCaps } from "./helpers";

const gpu = (totalGiB: number, name = "AMD Radeon RX 9070/9070 XT/9070 GRE", index = 0) => ({
  totalBytes: totalGiB * GiB,
  usedBytes: 1 * GiB,
  name,
  index,
  llamaDeviceId: `Vulkan${index}`,
  source: "test",
});

describe("shortGpuName", () => {
  it("drops the vendor and the SKU list", () => {
    assert.equal(shortGpuName({ name: "AMD Radeon RX 9070/9070 XT/9070 GRE" }, 0), "RX 9070");
    assert.equal(shortGpuName({ name: "NVIDIA GeForce RTX 4070" }, 0), "RTX 4070");
    assert.equal(shortGpuName({ name: "Radeon RX 9050 / 9060 XT" }, 1), "RX 9050/9060 XT");
  });

  it("falls back to the llama.cpp id for driver-only names", () => {
    assert.equal(shortGpuName({ name: "amdgpu", llamaDeviceId: "Vulkan1" }, 1), "Vulkan1");
    assert.equal(shortGpuName(undefined, 2), "GPU 2");
  });
});

describe("buildMemoryView", () => {
  it("reports a comfortable fit on one GPU as good with free space in the headline", () => {
    const caps = denseCaps({ fileSizeBytes: 8 * GiB });
    const s = loadSettings({ contextLength: 16384 });
    const g = [gpu(24)];
    const { view } = computeMemoryView(caps, s, g);
    assert.equal(view!.level, "good");
    assert.match(view!.headline, /^Fits on RX 9070 with .* to spare\.$/);
    assert.equal(view!.devices.length, 1);
    assert.equal(view!.devices[0]!.label, "RX 9070");
    assert.equal(view!.fixes.length, 0);
  });

  it("calls it tight (not spill) when it fits but leaves less than the headroom target", () => {
    const caps = denseCaps();
    const g = [gpu(16)];
    // Search a context whose estimate lands just under capacity − headroom.
    let s = loadSettings({ contextLength: 8192 });
    for (let ctx = 8192; ctx <= 262144; ctx += 1024) {
      const est = estimateMemory(caps, loadSettings({ contextLength: ctx, gpuOffload: 30 }), g[0], { gpus: g });
      const free = g[0]!.totalBytes - est!.totalGpuBytes;
      if (free > 0 && free < VRAM_HEADROOM_BYTES) {
        s = loadSettings({ contextLength: ctx, gpuOffload: 30 });
        break;
      }
    }
    const { view, estimate } = computeMemoryView(caps, s, g, { withFixes: true });
    assert.ok(estimate!.willSpill, "core should flag the missing headroom");
    assert.equal(view!.level, "tight");
    assert.match(view!.headline, /^Fits, but only .* free on RX 9070 \(target 1\.5 GiB\)\.$/);
    assert.ok(!view!.notes.some((n) => /^Tight on /.test(n)), "headline replaces the device warning");
  });

  it("calls real overflow spill and names the overage", () => {
    const caps = denseCaps({ fileSizeBytes: 30 * GiB });
    const g = [gpu(16)];
    const { view } = computeMemoryView(caps, loadSettings({ contextLength: 32768 }), g);
    assert.equal(view!.level, "spill");
    assert.match(view!.headline, /over VRAM on RX 9070/);
  });

  it("lists a GPU that gets no layers as idle instead of drawing an empty bar", () => {
    const caps = denseCaps({ fileSizeBytes: 8 * GiB });
    const g = [gpu(16, "AMD Radeon RX 9070/9070 XT/9070 GRE", 0), gpu(16, "AMD Radeon RX 9050 / 9060 XT", 1)];
    const { view } = computeMemoryView(caps, loadSettings({ contextLength: 16384, splitMode: "none" }), g);
    assert.deepEqual(view!.devices.map((d) => d.label), ["RX 9070"]);
    assert.deepEqual(view!.idle, ["RX 9050/9060 XT"]);
    assert.match(view!.footnote, /RX 9050\/9060 XT idle/);
  });

  it("shows RAM as a bar only when it holds real weight", () => {
    const caps = denseCaps();
    const g = [gpu(16)];
    const full = computeMemoryView(caps, loadSettings({ contextLength: 8192, gpuOffload: 99 }), [gpu(48)]).view!;
    assert.ok(!full.devices.some((d) => d.kind === "ram"));
    const partial = computeMemoryView(caps, loadSettings({ contextLength: 8192, gpuOffload: 10 }), g).view!;
    assert.ok(partial.devices.some((d) => d.kind === "ram"));
  });

  it("uses one RAM bar on the CPU backend", () => {
    const { view } = computeMemoryView(denseCaps({ fileSizeBytes: 4 * GiB }), loadSettings({ contextLength: 8192 }), [], {
      cpuOnly: true,
    });
    assert.deepEqual(view!.devices.map((d) => d.kind), ["ram"]);
    assert.match(view!.headline, /system RAM/);
  });

  it("returns undefined without an estimate", () => {
    assert.equal(buildMemoryView(undefined, loadSettings(), undefined, []), undefined);
  });
});

describe("suggestMemoryFixes", () => {
  it("offers a shorter context that actually restores headroom", () => {
    const caps = denseCaps({ fileSizeBytes: 11 * GiB });
    const g = [gpu(16)];
    const s = loadSettings({ contextLength: 262144, cacheTypeK: "q8_0", cacheTypeV: "q8_0" });
    const { view } = computeMemoryView(caps, s, g, { withFixes: true });
    assert.notEqual(view!.level, "good");
    const ctx = view!.fixes.find((f) => f.id === "context");
    assert.ok(ctx, `expected a context fix, got ${JSON.stringify(view!.fixes)}`);
    assert.ok((ctx!.patch.contextLength || 0) < s.contextLength);
    const after = computeMemoryView(caps, { ...s, ...ctx!.patch }, g).view!;
    assert.equal(after.level, "good");
  });

  it("suggests the idle second GPU when split mode is none", () => {
    const caps = denseCaps({ fileSizeBytes: 13 * GiB });
    const g = [gpu(16, "AMD Radeon RX 9070", 0), gpu(16, "AMD Radeon RX 9060 XT", 1)];
    const s = loadSettings({ contextLength: 65536, splitMode: "none" });
    const fixes = suggestMemoryFixes(caps, s, g);
    const both = fixes.find((f) => f.id === "gpus");
    assert.ok(both, `expected a multi-GPU fix, got ${JSON.stringify(fixes)}`);
    assert.equal(both!.label, "Use both GPUs");
    assert.deepEqual(both!.patch, { splitMode: "layer", tensorSplit: "" });
  });

  it("says how far a partial fix is still over instead of '0 B free'", () => {
    const caps = denseCaps({ fileSizeBytes: 20 * GiB });
    const g = [gpu(16, "AMD Radeon RX 9070", 0), gpu(16, "AMD Radeon RX 9060 XT", 1)];
    const s = loadSettings({ contextLength: 262144, cacheTypeK: "f16", cacheTypeV: "f16", splitMode: "none" });
    const fixes = suggestMemoryFixes(caps, s, g);
    assert.ok(fixes.length > 0);
    for (const f of fixes) {
      assert.ok(!/^0 B free$/.test(f.detail), `${f.id}: ${f.detail}`);
      if (f.level === "spill") {
        assert.match(f.detail, /^still .* over$/);
      }
    }
    const ranks = fixes.map((f) => ({ good: 0, unknown: 1, tight: 2, spill: 3 })[f.level]);
    assert.deepEqual(ranks, [...ranks].sort((a, b) => a - b), "better outcomes first");
  });

  it("never offers a fix that makes things worse and caps the list at three", () => {
    const caps = moeCaps({ fileSizeBytes: 60 * GiB });
    const g = [gpu(16)];
    const s = loadSettings({ contextLength: 131072, cacheTypeK: "f16", cacheTypeV: "f16" });
    const fixes = suggestMemoryFixes(caps, s, g);
    assert.ok(fixes.length <= 3);
    const base = computeMemoryView(caps, s, g).view!;
    const rank = { good: 0, unknown: 1, tight: 2, spill: 3 } as const;
    for (const f of fixes) {
      assert.ok(rank[f.level] <= rank[base.level], `${f.id} made it worse`);
    }
  });
});

describe("diffLoadSettings", () => {
  it("lists only changed, labelled keys with readable values", () => {
    const a = loadSettings({ contextLength: 81920, cacheTypeV: "q8_0", gpuOffload: 99 });
    const b = { ...a, contextLength: 65536, cacheTypeV: "q4_0" as const };
    assert.deepEqual(diffLoadSettings(a, b), [
      { key: "contextLength", label: "ctx", from: "80k", to: "64k" },
      { key: "cacheTypeV", label: "KV V", from: "q8_0", to: "q4_0" },
    ]);
  });

  it("is empty without a launched snapshot (older lock files)", () => {
    assert.deepEqual(diffLoadSettings(undefined, loadSettings()), []);
  });
});

describe("friendlyModelTitle", () => {
  it("prefers the file name when general.name lacks the parameter count", () => {
    assert.equal(
      friendlyModelTitle(
        "Swift15-GSQfix-V1MIX-ISTAalloc-IQ3_XXS MTP",
        "/m/ukisai_Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
      ),
      "Swift 1.5 Qwen3.8 27B · IQ3_XXS"
    );
  });

  it("keeps a good general.name and appends the quant", () => {
    assert.equal(
      friendlyModelTitle("Qwen3 Coder 30B A3B Instruct", "/m/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"),
      "Qwen3 Coder 30B A3B Instruct · Q4_K_M"
    );
  });

  it("handles MoE sizes and falls back when no size is in the file name", () => {
    assert.equal(friendlyModelTitle("", "/m/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"), "Qwen3 Coder 30B-A3B · Q4_K_M");
    assert.equal(friendlyModelTitle("my-model", "/m/my-model-q8.gguf"), "my-model");
  });
});

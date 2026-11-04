import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { it } from "node:test";
import { estimateMemory } from "../src/memoryEstimate";
import { normalizeLoadSettings } from "../src/types";
import { denseCaps, GiB, loadSettings } from "./helpers";

it("moves projector and draft allocations to the selected GPU, or draft CPU", t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "runtime-devices-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const mmprojPath = path.join(dir, "mmproj.gguf");
  fs.writeFileSync(mmprojPath, Buffer.alloc(1024));
  const gpus = [0, 1].map(i => ({ totalBytes: 16 * GiB, usedBytes: 0, source: "test", name: `GPU ${i}`, llamaDeviceId: `Vulkan${i}` }));
  const draftCaps = denseCaps({ architecture: "dflash", blockCount: 5, fileSizeBytes: GiB });
  const settings = loadSettings({ gpuOffload: 99, splitMode: "none", mainGpu: 0, mmprojPath,
    speculativeMode: "dflash", draftModelPath: "/models/draft.gguf", draftDevice: "Vulkan1", mmprojDevice: "Vulkan1" });
  const estimate = estimateMemory(denseCaps(), settings, gpus[0], { gpus, draftCaps })!;
  const bytes = (chart: typeof estimate.charts.vram, key: string) => chart.segments.find(s => s.key === key)?.bytes ?? 0;
  assert.equal(bytes(estimate.charts.vram, "vision"), 0);
  assert.equal(bytes(estimate.charts.vram, "draft"), 0);
  assert.equal(bytes(estimate.charts.vram2!, "vision"), 1024);
  assert.ok(bytes(estimate.charts.vram2!, "draft") >= GiB);
  const cpu = estimateMemory(denseCaps(), { ...settings, draftDevice: "none" }, gpus[0], { gpus, draftCaps })!;
  assert.equal(bytes(cpu.charts.vram2!, "draft"), 0);
  assert.ok(cpu.totalCpuBytes >= estimate.totalCpuBytes + GiB);
});

it("normalizes persisted controls and clears an inherited prompt-cache override", () => {
  const settings = normalizeLoadSettings({ cacheRamMiB: 0, mmprojDevice: " Vulkan1 ", draftDevice: "none" });
  assert.equal(settings.cacheRamMiB, 0);
  assert.equal(settings.mmprojDevice, "Vulkan1");
  assert.equal(settings.draftDevice, "none");
  assert.equal(normalizeLoadSettings({ ...settings, cacheRamMiB: undefined }).cacheRamMiB, undefined);
  assert.equal(normalizeLoadSettings({ draftDevice: "Vulkan0,Vulkan1" }).draftDevice, "");
  assert.equal(normalizeLoadSettings({ draftDevice: "--port 8000" }).draftDevice, "");
});

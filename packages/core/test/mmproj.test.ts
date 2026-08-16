import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { after, before, describe, it } from "node:test";
import {
  findSiblingMmproj,
  isMmprojFileName,
  preferMmprojPath,
  resolveMmprojPath,
} from "../src/modelLibrary";
import { languageGgufFiles, preferredMmprojFile } from "../src/huggingFace";
import { buildServerArgs } from "../src/serverArgs";
import { estimateMemory } from "../src/memoryEstimate";
import { denseCaps, GiB, loadSettings, argValue } from "./helpers";

describe("isMmprojFileName", () => {
  it("matches common projector names", () => {
    assert.equal(isMmprojFileName("mmproj-F16.gguf"), true);
    assert.equal(isMmprojFileName("Qwen3.8-27B-mmproj-BF16.gguf"), true);
    assert.equal(isMmprojFileName("Qwen3.8-27B-UD-Q3_K_XL.gguf"), false);
  });
});

describe("preferMmprojPath", () => {
  it("prefers F16 over BF16 and Q8", () => {
    assert.equal(
      preferMmprojPath(["mmproj-Q8_0.gguf", "mmproj-BF16.gguf", "mmproj-F16.gguf"]),
      "mmproj-F16.gguf"
    );
  });

  it("returns undefined for an empty list", () => {
    assert.equal(preferMmprojPath([]), undefined);
  });
});

describe("findSiblingMmproj / resolveMmprojPath", () => {
  let dir: string;

  before(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-mmproj-"));
    fs.writeFileSync(path.join(dir, "model-Q4_K_M.gguf"), Buffer.alloc(64));
    fs.writeFileSync(path.join(dir, "mmproj-BF16.gguf"), Buffer.alloc(2 * 1024 * 1024));
    fs.writeFileSync(path.join(dir, "mmproj-F16.gguf"), Buffer.alloc(2 * 1024 * 1024));
  });

  after(() => fs.rmSync(dir, { recursive: true, force: true }));

  it("picks the F16 sibling next to the language GGUF", () => {
    const found = findSiblingMmproj(path.join(dir, "model-Q4_K_M.gguf"));
    assert.equal(found && path.basename(found), "mmproj-F16.gguf");
  });

  it("attaches a sibling when switching models", () => {
    const next = resolveMmprojPath(path.join(dir, "model-Q4_K_M.gguf"), "/elsewhere/mmproj.gguf", true);
    assert.equal(next && path.basename(next), "mmproj-F16.gguf");
  });

  it("keeps a manual pick when the model path did not change", () => {
    const kept = path.join(dir, "mmproj-BF16.gguf");
    assert.equal(
      resolveMmprojPath(path.join(dir, "model-Q4_K_M.gguf"), kept, false),
      kept
    );
  });

  it("does not auto-attach when the same model is reapplied with vision cleared", () => {
    assert.equal(resolveMmprojPath(path.join(dir, "model-Q4_K_M.gguf"), "", false), "");
  });
});

describe("HF mmproj helpers", () => {
  const files = [
    { path: "Qwen3.8-27B-Q4_K_M.gguf", size: 17e9, url: "" },
    { path: "mmproj-BF16.gguf", size: 9e8, url: "" },
    { path: "mmproj-F16.gguf", size: 9e8, url: "" },
  ];

  it("hides projectors from the language-file picker", () => {
    assert.deepEqual(
      languageGgufFiles(files).map((f) => f.path),
      ["Qwen3.8-27B-Q4_K_M.gguf"]
    );
  });

  it("selects the preferred projector from a repo listing", () => {
    assert.equal(preferredMmprojFile(files)?.path, "mmproj-F16.gguf");
  });
});

describe("buildServerArgs --mmproj", () => {
  it("omits the flag when no projector is set", () => {
    const args = buildServerArgs("/m.gguf", "127.0.0.1", 8742, loadSettings());
    assert.ok(!args.includes("--mmproj"));
  });

  it("passes --mmproj when a path is set", () => {
    const proj = "/models/mmproj-F16.gguf";
    const args = buildServerArgs("/m.gguf", "127.0.0.1", 8742, loadSettings({ mmprojPath: proj }));
    assert.equal(argValue(args, "--mmproj"), proj);
    assert.ok(!args.includes("--no-mmproj-offload"));
  });

  it("passes --no-mmproj-offload when vision GPU offload is off", () => {
    const proj = "/models/mmproj-F16.gguf";
    const args = buildServerArgs(
      "/m.gguf",
      "127.0.0.1",
      8742,
      loadSettings({ mmprojPath: proj, mmprojOffloadToGpu: false })
    );
    assert.equal(argValue(args, "--mmproj"), proj);
    assert.ok(args.includes("--no-mmproj-offload"));
  });

  it("does not emit --no-mmproj-offload without a projector", () => {
    const args = buildServerArgs(
      "/m.gguf",
      "127.0.0.1",
      8742,
      loadSettings({ mmprojOffloadToGpu: false })
    );
    assert.ok(!args.includes("--mmproj"));
    assert.ok(!args.includes("--no-mmproj-offload"));
  });
});

describe("estimateMemory mmproj", () => {
  let proj: string;

  before(() => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-mmproj-mem-"));
    proj = path.join(dir, "mmproj-F16.gguf");
    fs.writeFileSync(proj, Buffer.alloc(64 * 1024 * 1024));
  });

  after(() => fs.rmSync(path.dirname(proj), { recursive: true, force: true }));

  it("adds projector bytes to VRAM when layers are on GPU", () => {
    const gpu = { totalBytes: 48 * GiB, usedBytes: 0, name: "test", source: "test" };
    const without = estimateMemory(denseCaps(), loadSettings({ mmprojPath: "" }), gpu);
    const withProj = estimateMemory(denseCaps(), loadSettings({ mmprojPath: proj }), gpu);
    assert.ok(without && withProj);
    assert.equal(withProj.mmprojFileSizeBytes, 64 * 1024 * 1024);
    assert.equal(withProj.totalGpuBytes - without.totalGpuBytes, 64 * 1024 * 1024);
    assert.equal(withProj.gpuWeightsBytes, without.gpuWeightsBytes);
    assert.equal(
      withProj.charts.vram.segments.find((s) => s.key === "vision")?.bytes,
      64 * 1024 * 1024
    );
    assert.equal(without.charts.vram.segments.find((s) => s.key === "vision")?.bytes, 0);
  });

  it("parks the projector on the main GPU and does not tensor-split it", () => {
    const g0 = { totalBytes: 16 * GiB, usedBytes: 0, name: "RX 9060 XT", source: "test" };
    const g1 = { totalBytes: 16 * GiB, usedBytes: 0, name: "RX 9070 XT", source: "test" };
    const est = estimateMemory(
      denseCaps(),
      loadSettings({ mmprojPath: proj, tensorSplit: "50,50", mainGpu: 1 }),
      g0,
      { gpus: [g0, g1] }
    );
    assert.ok(est?.charts.vram2);
    const visMain = est.charts.vram.segments.find((s) => s.key === "vision")!.bytes;
    const visOther = est.charts.vram2.segments.find((s) => s.key === "vision")!.bytes;
    assert.equal(visMain, 64 * 1024 * 1024);
    assert.equal(visOther, 0);
    assert.match(est.charts.vram.title, /9070/);
    assert.ok(est.warnings.some((w) => /Vision projector included/.test(w) && /9070/.test(w)));
  });

  it("puts the projector in system RAM when GPU offload is off", () => {
    const gpu = { totalBytes: 48 * GiB, usedBytes: 0, name: "test", source: "test" };
    const onGpu = estimateMemory(denseCaps(), loadSettings({ mmprojPath: proj }), gpu);
    const onCpu = estimateMemory(
      denseCaps(),
      loadSettings({ mmprojPath: proj, mmprojOffloadToGpu: false }),
      gpu
    );
    assert.ok(onGpu && onCpu);
    assert.equal(onGpu.charts.vram.segments.find((s) => s.key === "vision")?.bytes, 64 * 1024 * 1024);
    assert.equal(onCpu.charts.vram.segments.find((s) => s.key === "vision")?.bytes, 0);
    assert.equal(onCpu.charts.ram.segments.find((s) => s.key === "vision")?.bytes, 64 * 1024 * 1024);
    assert.equal(onCpu.totalGpuBytes, onGpu.totalGpuBytes - 64 * 1024 * 1024);
    assert.ok(onCpu.warnings.some((w) => /no-mmproj-offload/.test(w)));
  });
});

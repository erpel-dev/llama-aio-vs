import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { formatBytes } from "../src/memoryEstimate";
import {
  classifyModelFit,
  describeModelKind,
  formatContext,
  formatPickerDetail,
  modelLikelySupportsTools,
  quantFromGgufName,
} from "../src/modelPickerInfo";
import { collapseLocalModelEntries, displayGgufTitle, localShardGroupKey } from "../src/modelLibrary";
import { denseCaps, moeCaps } from "./helpers";
import type { MemoryEstimate } from "../src/memoryEstimate";

function est(overrides: Partial<MemoryEstimate>): MemoryEstimate {
  return {
    fileSizeBytes: 10,
    layersTotal: 48,
    layersOnGpu: 48,
    gpuWeightsBytes: 8,
    cpuWeightsBytes: 0,
    kvBytes: 1,
    kvBytesWarm: 1,
    gpuKvBytes: 1,
    cpuKvBytes: 0,
    kvOnGpu: true,
    overheadBytes: 1,
    gpuOverheadBytes: 1,
    peerGpuOverheadBytes: 0,
    cpuOverheadBytes: 0,
    totalGpuBytes: 10,
    totalCpuBytes: 2,
    totalGpuBytesWarm: 9,
    totalCpuBytesWarm: 2,
    gpuTotalBytes: 24 * 1024 ** 3,
    systemRamTotalBytes: 64 * 1024 ** 3,
    charts: {
      vram: { title: "VRAM", segments: [], totalBytes: 0 },
      ram: { title: "RAM", segments: [], totalBytes: 0 },
    },
    willSpill: false,
    warnings: [],
    lines: [],
    summary: "",
    ...overrides,
  };
}

describe("displayGgufTitle / quant", () => {
  it("strips the shard suffix and keeps the quant", () => {
    assert.equal(
      displayGgufTitle("/m/Qwen3-235B-A22B-Q3_K_S-00001-of-00005.gguf"),
      "Qwen3-235B-A22B-Q3_K_S"
    );
    assert.equal(quantFromGgufName("/m/Qwen3-32B-Q5_K_M.gguf"), "Q5_K_M");
    assert.equal(quantFromGgufName("gpt-oss-20b-MXFP4.gguf"), "MXFP4");
  });
});

describe("collapseLocalModelEntries (B-38)", () => {
  it("merges split shards into one row pointing at the first part", () => {
    const dir = "/models/qwen";
    const entries = [1, 2, 3].map((i) => ({
      path: `${dir}/Foo-Q4_K_M-0000${i}-of-00003.gguf`,
      source: "Llama AIO",
      sizeBytes: i * 100,
    }));
    entries.push({ path: `${dir}/plain-Q4_K_M.gguf`, source: "Llama AIO", sizeBytes: 50 });
    const out = collapseLocalModelEntries(entries);
    const split = out.find((e) => e.shardCount === 3);
    assert.ok(split);
    assert.equal(split.path, `${dir}/Foo-Q4_K_M-00001-of-00003.gguf`);
    assert.equal(split.sizeBytes, 600);
    assert.equal(out.filter((e) => e.path.endsWith("plain-Q4_K_M.gguf")).length, 1);
    assert.equal(localShardGroupKey(`${dir}/Foo-Q4_K_M-00002-of-00003.gguf`), `${dir}/foo-q4_k_m`);
  });
});

describe("classifyModelFit", () => {
  const GiB = 1024 ** 3;

  it("marks a comfortable estimate as fits", () => {
    assert.equal(classifyModelFit(est({ willSpill: false, totalGpuBytes: 8 * GiB })).fit, "fits");
  });

  it("marks a VRAM spill that still fits in RAM as tight", () => {
    const r = classifyModelFit(
      est({
        willSpill: true,
        layersOnGpu: 30,
        layersTotal: 48,
        totalGpuBytes: 20 * GiB,
        gpuTotalBytes: 12 * GiB,
        totalCpuBytes: 8 * GiB,
        systemRamTotalBytes: 64 * GiB,
      })
    );
    assert.equal(r.fit, "tight");
    assert.match(r.detail || "", /18 layers on CPU/);
  });

  it("marks overflow past RAM+VRAM as won't fit", () => {
    const r = classifyModelFit(
      est({
        willSpill: true,
        totalGpuBytes: 80 * GiB,
        gpuTotalBytes: 12 * GiB,
        totalCpuBytes: 90 * GiB,
        systemRamTotalBytes: 32 * GiB,
      })
    );
    assert.equal(r.fit, "wont-fit");
    assert.ok(r.detail);
  });
});

describe("picker copy", () => {
  it("describes MoE vs dense and formats context", () => {
    assert.equal(describeModelKind(moeCaps({ expertCount: 128, expertUsedCount: 8 })), "MoE 128×8");
    assert.equal(describeModelKind(denseCaps()), "dense");
    assert.equal(formatContext(262144), "256k");
    assert.equal(formatContext(32768), "32k");
  });

  it("guesses tool support from the file name", () => {
    assert.equal(modelLikelySupportsTools("Qwen3-Coder-30B-Instruct-Q4_K_M.gguf"), true);
    assert.equal(modelLikelySupportsTools("nomic-embed-text-v1.5.gguf"), false);
  });

  it("joins the mockup-style detail line", () => {
    const line = formatPickerDetail({
      title: "Qwen3",
      folder: "~/models/qwen",
      kind: "MoE 128×3",
      contextLabel: "ctx 262k",
      tools: true,
      vision: false,
      shardCount: 1,
      fit: "fits",
    });
    assert.equal(line, "~/models/qwen · MoE 128×3 · ctx 262k · tools ✓ · vision ✗");
    assert.ok(formatBytes(17.3 * 1024 ** 3).includes("GiB"));
  });
});

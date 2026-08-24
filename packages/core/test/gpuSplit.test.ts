import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  alignTensorSplitToMainGpu,
  gpuDisplayOrder,
  isLegacyGpu0FirstSplit,
  mainShareForUi,
  mainShareFromSplit,
  normalizeGpuSplitMode,
  normalizeTensorSplit,
  parseTensorSplit,
  capacityAwareTensorSplit,
  retargetTensorSplitMainShare,
  tensorSplitFromFractions,
  tensorSplitForMainShare,
  tensorSplitShares,
  tensorSplitSharesEqual,
  effectiveTensorSplitShares,
} from "../src/gpuSplit";

describe("parseTensorSplit", () => {
  it("accepts comma-separated positive numbers", () => {
    assert.deepEqual(parseTensorSplit("3,1"), [3, 1]);
    assert.deepEqual(parseTensorSplit("0.75, 0.25"), [0.75, 0.25]);
    assert.deepEqual(parseTensorSplit("3;1"), [3, 1]);
    assert.deepEqual(parseTensorSplit("0,1"), [0, 1]);
    assert.deepEqual(parseTensorSplit("70,0,30"), [70, 0, 30]);
  });

  it("rejects empty, junk, and a single value", () => {
    assert.deepEqual(parseTensorSplit(""), []);
    assert.deepEqual(parseTensorSplit("  "), []);
    assert.deepEqual(parseTensorSplit("nope"), []);
    assert.deepEqual(parseTensorSplit("3"), []);
    assert.deepEqual(parseTensorSplit(undefined), []);
  });
});

describe("normalizeTensorSplit", () => {
  it("canonicalizes or clears", () => {
    assert.equal(normalizeTensorSplit("3, 1"), "3,1");
    assert.equal(normalizeTensorSplit("0.75,0.25"), "0.75,0.25");
    assert.equal(normalizeTensorSplit("1"), "");
    assert.equal(normalizeTensorSplit("abc"), "");
  });
});

describe("normalizeGpuSplitMode", () => {
  it("keeps layer/row/tensor/none and falls back otherwise", () => {
    assert.equal(normalizeGpuSplitMode("row"), "row");
    assert.equal(normalizeGpuSplitMode("none"), "none");
    assert.equal(normalizeGpuSplitMode("layer"), "layer");
    assert.equal(normalizeGpuSplitMode("tensor"), "tensor");
    assert.equal(normalizeGpuSplitMode("nope"), "layer");
  });
});

describe("tensorSplitShares", () => {
  it("uses explicit split when set", () => {
    assert.deepEqual(tensorSplitShares("3,1", 2, [16, 16]), [0.75, 0.25]);
  });

  it("falls back to VRAM-proportional shares", () => {
    assert.deepEqual(tensorSplitShares("", 2, [16, 8]), [2 / 3, 1 / 3]);
  });

  it("is a single 1 for one GPU", () => {
    assert.deepEqual(tensorSplitShares("3,1", 1, [16]), [1]);
  });

  it("does not dump 100% onto one card when the other reports 0 VRAM", () => {
    assert.deepEqual(tensorSplitShares("", 2, [16, 0]), [0.5, 0.5]);
  });
});

describe("effectiveTensorSplitShares", () => {
  it("ignores tensor-split when split-mode is none", () => {
    assert.deepEqual(effectiveTensorSplitShares("90,10", "none", 0, 2, [16, 16]), [1, 0]);
    assert.deepEqual(effectiveTensorSplitShares("90,10", "none", 1, 2, [16, 16]), [0, 1]);
  });

  it("keeps a layer split when mode is layer", () => {
    assert.deepEqual(effectiveTensorSplitShares("3,1", "layer", 0, 2, [16, 16]), [0.75, 0.25]);
  });
});

describe("tensorSplitForMainShare", () => {
  it("puts the slider share on GPU 0", () => {
    assert.equal(tensorSplitForMainShare(0.75, 0, 2), "75,25");
  });

  it("puts the slider share on GPU 1 (device-order string, not main-first)", () => {
    assert.equal(tensorSplitForMainShare(0.75, 1, 2), "25,75");
  });

  it("splits the remainder evenly across extra GPUs", () => {
    assert.equal(tensorSplitForMainShare(0.75, 1, 3), "13,75,12");
  });

  it("is empty for a single GPU", () => {
    assert.equal(tensorSplitForMainShare(0.75, 0, 1), "");
  });
});

describe("mainShareFromSplit / mainShareForUi", () => {
  it("reads the main GPU slice in device order", () => {
    assert.equal(mainShareFromSplit("3,1", 0, 2, [16, 16]), 0.75);
    assert.equal(mainShareFromSplit("3,1", 1, 2, [16, 16]), 0.25);
    assert.equal(mainShareFromSplit("25,75", 1, 2, [16, 16]), 0.75);
  });

  it("gives Main the larger existing share so 3,1 + main GPU 1 becomes 75%", () => {
    assert.equal(mainShareForUi("3,1", 1, 2, [16, 16]), 0.75);
    assert.equal(mainShareForUi("3,1", 0, 2, [16, 16]), 0.75);
    assert.equal(mainShareForUi("", 1, 2, [16, 16]), 0.5);
  });
});

describe("alignTensorSplitToMainGpu", () => {
  it("rewrites 3,1 onto GPU 1 when that card is Main", () => {
    assert.equal(alignTensorSplitToMainGpu("3,1", 1, 2, [16, 16]), "25,75");
    assert.equal(alignTensorSplitToMainGpu("3,1", 0, 2, [16, 16]), "75,25");
  });

  it("leaves auto empty", () => {
    assert.equal(alignTensorSplitToMainGpu("", 1, 2, [16, 16]), "");
  });
});

describe("tensorSplitSharesEqual", () => {
  it("treats 3,1 and 75,25 as the same split", () => {
    assert.equal(tensorSplitSharesEqual("3,1", "75,25", 2, [16, 16]), true);
    assert.equal(tensorSplitSharesEqual("3,1", "25,75", 2, [16, 16]), false);
  });
});

describe("isLegacyGpu0FirstSplit", () => {
  it("recognizes the old GPU0-first presets", () => {
    assert.equal(isLegacyGpu0FirstSplit("3,1"), true);
    assert.equal(isLegacyGpu0FirstSplit("2,1"), true);
    assert.equal(isLegacyGpu0FirstSplit("75,25"), false);
    assert.equal(isLegacyGpu0FirstSplit("25,75"), false);
    assert.equal(isLegacyGpu0FirstSplit(""), false);
  });
});

describe("gpuDisplayOrder", () => {
  it("puts the main GPU first", () => {
    assert.deepEqual(gpuDisplayOrder(2, 0), [0, 1]);
    assert.deepEqual(gpuDisplayOrder(2, 1), [1, 0]);
    assert.deepEqual(gpuDisplayOrder(3, 2), [2, 0, 1]);
  });
});

describe("capacityAwareTensorSplit", () => {
  const GiB = 1024 ** 3;

  it("fills Main first, then remaining cards back-to-front", () => {
    const ts = capacityAwareTensorSplit(
      [16 * GiB, 16 * GiB, 16 * GiB],
      0,
      1 * GiB,
      20 * GiB,
      2 * GiB
    );
    const parts = parseTensorSplit(ts);
    assert.equal(parts.length, 3);
    assert.equal(parts[1], 0);
    assert.ok(parts[0]! > parts[2]!);
    assert.ok(parts[2]! > 0);
  });

  it("keeps zeros when retargeting Main’s share", () => {
    assert.equal(retargetTensorSplitMainShare("70,0,30", 0, 0.65), "65,0,35");
  });

  it("emits percents that sum to 100", () => {
    const ts = tensorSplitFromFractions([13, 0, 7]);
    const parts = parseTensorSplit(ts);
    assert.equal(parts.reduce((a, b) => a + b, 0), 100);
    assert.equal(parts[1], 0);
  });
});

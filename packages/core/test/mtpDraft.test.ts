import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { after, before, describe, it } from "node:test";
import {
  classifyGgufFile,
  findSiblingMtpDraft,
  isBakedInMtpNearCopy,
  languageRejectsSidecarMtp,
  isConfidentMtpSidecarPath,
  isImatrixFileName,
  isMtpDraftFileName,
  isMtpSidecarFile,
  matchMtpDraftToLanguage,
  preferMtpDraftPath,
  resolveMtpDraftPath,
} from "../src/modelLibrary";
import {
  companionDownloadHint,
  describeLanguageGgufFile,
  languageGgufFiles,
  preferredMtpDraftFile,
} from "../src/huggingFace";
import { clampLoadSettingsToModel, isMtpDraftArchitecture } from "../src/ggufMetadata";
import { buildServerArgs, serverConfigFingerprint } from "../src/serverArgs";
import { estimateMemory } from "../src/memoryEstimate";
import { recommendLoadSettings } from "../src/recommendSettings";
import { denseCaps, GiB, loadSettings, argValue, moeCaps } from "./helpers";

describe("isMtpDraftFileName", () => {
  it("matches Unsloth sidecar names", () => {
    assert.equal(isMtpDraftFileName("mtp-gemma-4-12B-it.gguf"), true);
    assert.equal(isMtpDraftFileName("MTP/gemma-4-12B-it-Q4_0-MTP.gguf"), true);
    assert.equal(isMtpDraftFileName("gemma-4-12B-it-Q8_0-MTP.gguf"), true);
    assert.equal(isMtpDraftFileName("gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"), false);
    assert.equal(isMtpDraftFileName("mmproj-F16.gguf"), false);
    assert.equal(isMtpDraftFileName("imatrix-qwen3.8-27b.gguf"), false);
  });

  it("treats prefix / MTP-dir as confident sidecars, not a bare -mtp suffix", () => {
    assert.equal(isConfidentMtpSidecarPath("mtp-gemma-4-12B-it.gguf"), true);
    assert.equal(isConfidentMtpSidecarPath("MTP/gemma-4-12B-it-Q4_0-MTP.gguf"), true);
    assert.equal(isConfidentMtpSidecarPath("Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf"), false);
  });
});

describe("imatrix file names", () => {
  it("matches importance-matrix dumps", () => {
    assert.equal(isImatrixFileName("imatrix-qwen3.8-27b.gguf"), true);
    assert.equal(isImatrixFileName("imatrix.gguf"), true);
    assert.equal(isImatrixFileName("Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf"), false);
  });
});

describe("isMtpDraftArchitecture", () => {
  it("matches gemma4-assistant", () => {
    assert.equal(isMtpDraftArchitecture("gemma4-assistant"), true);
    assert.equal(isMtpDraftArchitecture("gemma4_assistant"), true);
    assert.equal(isMtpDraftArchitecture("gemma4"), false);
    assert.equal(isMtpDraftArchitecture("dflash"), false);
  });
});

describe("preferMtpDraftPath", () => {
  it("prefers the repo-root mtp-*.gguf over MTP/ folder quants", () => {
    assert.equal(
      preferMtpDraftPath([
        "MTP/gemma-4-12B-it-BF16-MTP.gguf",
        "MTP/gemma-4-12B-it-Q4_0-MTP.gguf",
        "mtp-gemma-4-12B-it.gguf",
      ]),
      "mtp-gemma-4-12B-it.gguf"
    );
  });

  it("prefers Q4 over Q8 when only folder files exist", () => {
    assert.equal(
      preferMtpDraftPath([
        "MTP/gemma-4-12B-it-Q8_0-MTP.gguf",
        "MTP/gemma-4-12B-it-Q4_0-MTP.gguf",
      ]),
      "MTP/gemma-4-12B-it-Q4_0-MTP.gguf"
    );
  });
});

describe("findSiblingMtpDraft / resolveMtpDraftPath", () => {
  let dir: string;

  before(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-mtp-"));
    fs.writeFileSync(path.join(dir, "gemma-4-12B-it-Q4_K_M.gguf"), Buffer.alloc(64));
    fs.writeFileSync(path.join(dir, "mtp-gemma-4-12B-it.gguf"), Buffer.alloc(2 * 1024 * 1024));
    fs.mkdirSync(path.join(dir, "MTP"));
    fs.writeFileSync(
      path.join(dir, "MTP", "gemma-4-12B-it-Q4_0-MTP.gguf"),
      Buffer.alloc(2 * 1024 * 1024)
    );
  });

  after(() => fs.rmSync(dir, { recursive: true, force: true }));

  it("picks the root mtp- sibling next to the language GGUF", () => {
    const found = findSiblingMtpDraft(path.join(dir, "gemma-4-12B-it-Q4_K_M.gguf"));
    assert.equal(found && path.basename(found), "mtp-gemma-4-12B-it.gguf");
  });

  it("attaches a sibling when switching models", () => {
    const next = resolveMtpDraftPath(
      path.join(dir, "gemma-4-12B-it-Q4_K_M.gguf"),
      "/elsewhere/draft.gguf",
      true,
      "dflash"
    );
    assert.equal(next && path.basename(next), "mtp-gemma-4-12B-it.gguf");
  });

  it("keeps a manual pick when the model path did not change", () => {
    const kept = path.join(dir, "MTP", "gemma-4-12B-it-Q4_0-MTP.gguf");
    assert.equal(
      resolveMtpDraftPath(path.join(dir, "gemma-4-12B-it-Q4_K_M.gguf"), kept, false, "mtp"),
      kept
    );
  });

  it("does not auto-attach when the same model is reapplied with the draft cleared", () => {
    assert.equal(
      resolveMtpDraftPath(path.join(dir, "gemma-4-12B-it-Q4_K_M.gguf"), "", false, "off"),
      ""
    );
  });

  it("keeps a DFlash draft when switching to a model with no sidecar", () => {
    const other = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-mtp-empty-"));
    try {
      const model = path.join(other, "plain.gguf");
      const dflashPath = path.join(other, "keep-dflash.gguf");
      fs.writeFileSync(model, Buffer.alloc(64));
      fs.writeFileSync(dflashPath, Buffer.alloc(2 * 1024 * 1024));
      assert.equal(resolveMtpDraftPath(model, dflashPath, true, "dflash"), dflashPath);
    } finally {
      fs.rmSync(other, { recursive: true, force: true });
    }
  });

  it("does not attach Unsloth Flash-Next mtp-* as --model-draft", () => {
    const flash = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-flash-mtp-"));
    try {
      const main = path.join(flash, "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf");
      const sidecar = path.join(flash, "mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf");
      fs.writeFileSync(main, Buffer.alloc(64));
      fs.writeFileSync(sidecar, Buffer.alloc(3 * 1024 * 1024));
      assert.equal(languageRejectsSidecarMtp(main), true);
      assert.equal(findSiblingMtpDraft(main), undefined);
      assert.equal(resolveMtpDraftPath(main, sidecar, true, "mtp"), "");
      const clamped = clampLoadSettingsToModel(
        loadSettings({ speculativeMode: "mtp", draftModelPath: sidecar }),
        moeCaps({ architecture: "qwen4exp", nextnPredictLayers: 0 })
      );
      assert.equal(clamped.draftModelPath, "");
      assert.equal(clamped.speculativeMode, "off");
    } finally {
      fs.rmSync(flash, { recursive: true, force: true });
    }
  });

  it("does not attach a baked-in Foo-mtp near-copy as a sidecar", () => {
    const ista = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-ista-mtp-"));
    try {
      const main = path.join(ista, "Qwen-IQ2_S.gguf");
      const baked = path.join(ista, "Qwen-IQ2_S-mtp.gguf");
      fs.writeFileSync(main, Buffer.alloc(64));
      fs.writeFileSync(baked, Buffer.alloc(64));
      assert.equal(findSiblingMtpDraft(main), undefined);
      assert.equal(isBakedInMtpNearCopy(main, baked), true);
      assert.equal(resolveMtpDraftPath(main, "", true, "off"), "");
      assert.equal(resolveMtpDraftPath(main, baked, false, "mtp"), "");
      assert.equal(resolveMtpDraftPath(main, baked, true, "mtp"), "");
      const clamped = clampLoadSettingsToModel(
        loadSettings({ speculativeMode: "mtp", draftModelPath: baked }),
        denseCaps()
      );
      assert.equal(clamped.draftModelPath, "");
      assert.equal(clamped.speculativeMode, "off");
    } finally {
      fs.rmSync(ista, { recursive: true, force: true });
    }
  });
});

describe("HF MTP helpers", () => {
  const files = [
    { path: "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf", size: 8e9, url: "" },
    { path: "mtp-gemma-4-12B-it.gguf", size: 8e8, url: "" },
    { path: "MTP/gemma-4-12B-it-Q4_0-MTP.gguf", size: 9e8, url: "" },
    { path: "mmproj-F16.gguf", size: 9e8, url: "" },
  ];

  it("hides MTP drafters and projectors from the language-file picker", () => {
    assert.deepEqual(
      languageGgufFiles(files).map((f) => f.path),
      ["gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"]
    );
  });

  it("selects the preferred root mtp- file from a repo listing", () => {
    assert.equal(preferredMtpDraftFile(files)?.path, "mtp-gemma-4-12B-it.gguf");
  });

  it("mentions only confirmed companions in the download hint", () => {
    assert.match(companionDownloadHint(files), /mmproj-F16\.gguf/);
    assert.match(companionDownloadHint(files), /matching MTP sidecar|mtp-gemma-4-12B-it\.gguf/);
  });
});

const ISTA_FILES = [
  { path: "Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf", size: 9259510912, url: "" },
  { path: "Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf", size: 9607981120, url: "" },
  { path: "Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf", size: 8422841472, url: "" },
  { path: "Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp.gguf", size: 8771311680, url: "" },
  { path: "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf", size: 11771546784, url: "" },
  { path: "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf", size: 12120016960, url: "" },
  { path: "Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf", size: 10094357632, url: "" },
  { path: "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf", size: 10442827840, url: "" },
  { path: "mmproj-Qwen3.8-27B-BF16.gguf", size: 931146528, url: "" },
  { path: "imatrix-qwen3.8-27b.gguf", size: 13642624, url: "" },
];

describe("ISTA baked-in -mtp builds", () => {
  it("classifies Foo-mtp next to a near-copy Foo as baked-in, not a sidecar", () => {
    assert.equal(
      classifyGgufFile(
        { path: "Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf", size: 9607981120 },
        ISTA_FILES
      ),
      "mtp-baked"
    );
    assert.equal(
      classifyGgufFile(
        { path: "Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf", size: 9259510912 },
        ISTA_FILES
      ),
      "language"
    );
    assert.equal(isMtpSidecarFile({ path: "Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf", size: 9607981120 }, ISTA_FILES), false);
  });

  it("keeps baked-in -mtp files in the language picker and hides imatrix / mmproj", () => {
    assert.deepEqual(
      languageGgufFiles(ISTA_FILES).map((f) => f.path).sort(),
      [
        "Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf",
        "Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf",
        "Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf",
        "Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp.gguf",
        "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
        "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
        "Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf",
        "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf",
      ].sort()
    );
  });

  it("does not auto-fetch a baked-in -mtp file as a sidecar", () => {
    assert.equal(preferredMtpDraftFile(ISTA_FILES), undefined);
    assert.equal(preferredMtpDraftFile(ISTA_FILES, "Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf"), undefined);
    assert.equal(companionDownloadHint(ISTA_FILES), "mmproj-Qwen3.8-27B-BF16.gguf");
  });

  it("labels baked-in variants in the picker description", () => {
    const baked = ISTA_FILES.find((f) => f.path.endsWith("IQ2_S-mtp.gguf"))!;
    const plain = ISTA_FILES.find((f) => f.path.endsWith("IQ2_S.gguf"))!;
    assert.match(describeLanguageGgufFile(baked, ISTA_FILES), /MTP head included/);
    assert.doesNotMatch(describeLanguageGgufFile(plain, ISTA_FILES), /MTP head included/);
  });
});

describe("per-quant sidecar matching", () => {
  const files = [
    { path: "model-IQ2_S.gguf", size: 8e9, url: "" },
    { path: "model-IQ2_XS.gguf", size: 7e9, url: "" },
    { path: "model-IQ2_S-mtp.gguf", size: 8e8, url: "" },
    { path: "model-IQ2_XS-mtp.gguf", size: 7e8, url: "" },
  ];

  it("pairs a small Foo-mtp with Foo as a sidecar and matches the chosen quant", () => {
    assert.equal(preferredMtpDraftFile(files, "model-IQ2_XS.gguf")?.path, "model-IQ2_XS-mtp.gguf");
    assert.equal(preferredMtpDraftFile(files, "model-IQ2_S.gguf")?.path, "model-IQ2_S-mtp.gguf");
    assert.equal(matchMtpDraftToLanguage("model-IQ2_XS.gguf", files.slice(2).map((f) => f.path)), "model-IQ2_XS-mtp.gguf");
  });

  it("hides those small -mtp files from the language picker", () => {
    assert.deepEqual(
      languageGgufFiles(files).map((f) => f.path),
      ["model-IQ2_S.gguf", "model-IQ2_XS.gguf"]
    );
  });
});

describe("sidecar MTP load path", () => {
  const draft = "/models/mtp-gemma-4-12B-it.gguf";

  it("keeps MTP when a sidecar draft is set even without next-n layers", () => {
    const s = clampLoadSettingsToModel(
      loadSettings({ speculativeMode: "mtp", draftModelPath: draft }),
      denseCaps()
    );
    assert.equal(s.speculativeMode, "mtp");
  });

  it("turns off MTP when the draft path is a baked-in -mtp near-copy name", () => {
    const s = clampLoadSettingsToModel(
      loadSettings({
        speculativeMode: "mtp",
        draftModelPath: "/models/Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf",
      }),
      denseCaps()
    );
    assert.equal(s.speculativeMode, "off");
  });

  it("does not treat a leftover baked-in -mtp path as a sidecar in the estimate", () => {
    const leftover = "/models/Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf";
    const draftCaps = denseCaps({
      name: "qwen-mtp-build",
      fileSizeBytes: 9 * GiB,
      blockCount: 65,
      nextnPredictLayers: 1,
    });
    const est = estimateMemory(
      denseCaps({ blockCount: 64, fileSizeBytes: 8.6 * GiB }),
      loadSettings({
        speculativeMode: "mtp",
        draftModelPath: leftover,
        draftGpuOffload: 99,
        gpuOffload: 0,
      }),
      { totalBytes: 16 * GiB, usedBytes: 0, name: "test", source: "test" },
      { draftCaps }
    );
    assert.ok(est);
    assert.equal(est.draftFileSizeBytes, undefined);
    assert.ok(!est.warnings.some((w) => /MTP sidecar included/i.test(w)));
  });

  it("still turns off MTP without next-n layers or a sidecar", () => {
    const s = clampLoadSettingsToModel(
      loadSettings({ speculativeMode: "mtp", draftModelPath: "/models/other.gguf" }),
      denseCaps()
    );
    assert.equal(s.speculativeMode, "off");
  });

  it("keeps ngram-mtp when a sidecar draft is set even without next-n layers", () => {
    const s = clampLoadSettingsToModel(
      loadSettings({ speculativeMode: "ngram-mtp", draftModelPath: draft }),
      denseCaps()
    );
    assert.equal(s.speculativeMode, "ngram-mtp");
  });

  it("falls back to n-gram from ngram-mtp without next-n layers or a sidecar", () => {
    const s = clampLoadSettingsToModel(
      loadSettings({ speculativeMode: "ngram-mtp", draftModelPath: "/models/other.gguf" }),
      denseCaps()
    );
    assert.equal(s.speculativeMode, "ngram");
  });

  it("enables MTP when recommending with a sidecar draft attached", () => {
    const r = recommendLoadSettings(
      loadSettings({ draftModelPath: draft }),
      denseCaps(),
      { gpu: { totalBytes: 48 * GiB, usedBytes: 0, name: "test", source: "test" } }
    );
    assert.equal(r.speculativeMode, "mtp");
    assert.equal(r.draftModelPath, draft);
    assert.equal(r.maxDraftTokens, 4);
    const fromZero = recommendLoadSettings(
      loadSettings({ draftModelPath: draft, maxDraftTokens: 0 }),
      denseCaps(),
      { gpu: { totalBytes: 48 * GiB, usedBytes: 0, name: "test", source: "test" } }
    );
    assert.equal(fromZero.maxDraftTokens, 4);
  });

  it("emits --model-draft with --spec-type draft-mtp for a sidecar", () => {
    const args = buildServerArgs(
      "/models/gemma.gguf",
      "127.0.0.1",
      8742,
      loadSettings({
        speculativeMode: "mtp",
        draftModelPath: draft,
        maxDraftTokens: 4,
        draftGpuOffload: 99,
      })
    );
    assert.equal(argValue(args, "--spec-type"), "draft-mtp");
    assert.equal(argValue(args, "--model-draft"), draft);
    assert.equal(argValue(args, "--spec-draft-n-max"), "4");
    assert.equal(argValue(args, "--spec-draft-ngl"), "99");
    assert.equal(argValue(args, "--fit"), "off");
    assert.ok(!args.includes("--cache-type-k-draft"));
  });

  it("does not pass --model-draft for baked-in MTP", () => {
    const args = buildServerArgs(
      "/models/qwen.gguf",
      "127.0.0.1",
      8742,
      loadSettings({ speculativeMode: "mtp", maxDraftTokens: 2 })
    );
    assert.equal(argValue(args, "--spec-type"), "draft-mtp");
    assert.ok(!args.includes("--model-draft"));
  });

  it("includes the sidecar path in the server fingerprint", () => {
    const MODEL = "/models/gemma.gguf";
    const base = serverConfigFingerprint(MODEL, loadSettings({ speculativeMode: "mtp" }));
    const withSidecar = serverConfigFingerprint(
      MODEL,
      loadSettings({ speculativeMode: "mtp", draftModelPath: draft, draftGpuOffload: 99 })
    );
    assert.notEqual(base, withSidecar);
  });

  it("adds sidecar draft weights to the memory estimate", () => {
    const draftCaps = denseCaps({
      name: "mtp-draft",
      architecture: "gemma4-assistant",
      fileSizeBytes: GiB,
      blockCount: 8,
    });
    const base = estimateMemory(denseCaps(), loadSettings({ speculativeMode: "off" }), {
      totalBytes: 48 * GiB,
      usedBytes: 0,
      name: "test",
      source: "test",
    });
    const withMtp = estimateMemory(
      denseCaps(),
      loadSettings({
        speculativeMode: "mtp",
        draftModelPath: draft,
        draftGpuOffload: 99,
        maxDraftTokens: 4,
      }),
      { totalBytes: 48 * GiB, usedBytes: 0, name: "test", source: "test" },
      { draftCaps }
    );
    assert.ok(base && withMtp);
    assert.ok(withMtp.totalGpuBytes > base.totalGpuBytes);
    assert.equal(withMtp.draftFileSizeBytes, GiB);
    assert.equal(withMtp.mtpLayers, undefined);
    assert.ok(
      withMtp.charts.vram.segments.some((s) => s.key === "draft" && s.bytes > 0),
      "VRAM chart should include a sidecar MTP segment"
    );
  });
});

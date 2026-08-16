import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { after, before, describe, it } from "node:test";
import {
  findSiblingMtpDraft,
  isMtpDraftFileName,
  preferMtpDraftPath,
  resolveMtpDraftPath,
} from "../src/modelLibrary";
import { languageGgufFiles, preferredMtpDraftFile } from "../src/huggingFace";
import { clampLoadSettingsToModel, isMtpDraftArchitecture } from "../src/ggufMetadata";
import { buildServerArgs, serverConfigFingerprint } from "../src/serverArgs";
import { estimateMemory } from "../src/memoryEstimate";
import { recommendLoadSettings } from "../src/recommendSettings";
import { denseCaps, GiB, loadSettings, argValue } from "./helpers";

describe("isMtpDraftFileName", () => {
  it("matches Unsloth sidecar names", () => {
    assert.equal(isMtpDraftFileName("mtp-gemma-4-12B-it.gguf"), true);
    assert.equal(isMtpDraftFileName("MTP/gemma-4-12B-it-Q4_0-MTP.gguf"), true);
    assert.equal(isMtpDraftFileName("gemma-4-12B-it-Q8_0-MTP.gguf"), true);
    assert.equal(isMtpDraftFileName("gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"), false);
    assert.equal(isMtpDraftFileName("mmproj-F16.gguf"), false);
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

  it("still turns off MTP without next-n layers or a sidecar", () => {
    const s = clampLoadSettingsToModel(
      loadSettings({ speculativeMode: "mtp", draftModelPath: "/models/other.gguf" }),
      denseCaps()
    );
    assert.equal(s.speculativeMode, "off");
  });

  it("enables MTP when recommending with a sidecar draft attached", () => {
    const r = recommendLoadSettings(
      loadSettings({ draftModelPath: draft }),
      denseCaps(),
      { gpu: { totalBytes: 48 * GiB, usedBytes: 0, name: "test", source: "test" } }
    );
    assert.equal(r.speculativeMode, "mtp");
    assert.equal(r.draftModelPath, draft);
    assert.equal(r.maxDraftTokens, 2);
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

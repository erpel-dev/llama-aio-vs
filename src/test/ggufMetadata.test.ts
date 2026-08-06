import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { after, before, describe, it } from "node:test";
import {
  clampLoadSettingsToModel,
  readModelCapabilities,
  shardFileNames,
  totalModelBytes,
} from "../ggufMetadata";
import { denseCaps, loadSettings } from "./helpers";

describe("shardFileNames", () => {
  it("expands a split-model name into the whole set", () => {
    assert.deepEqual(shardFileNames("model-00001-of-00003.gguf"), [
      "model-00001-of-00003.gguf",
      "model-00002-of-00003.gguf",
      "model-00003-of-00003.gguf",
    ]);
  });

  it("works from any shard, not just the first", () => {
    assert.equal(shardFileNames("m-00002-of-00002.gguf")?.length, 2);
  });

  it("returns undefined for ordinary file names", () => {
    assert.equal(shardFileNames("model-Q4_K_M.gguf"), undefined);
    assert.equal(shardFileNames("model.gguf"), undefined);
    // A name that merely looks numeric must not be treated as a shard.
    assert.equal(shardFileNames("llama-3-00001.gguf"), undefined);
  });

  it("rejects an implausible shard count", () => {
    assert.equal(shardFileNames("m-00001-of-00000.gguf"), undefined);
  });
});

describe("totalModelBytes", () => {
  let dir: string;

  before(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-shards-"));
    fs.writeFileSync(path.join(dir, "solo.gguf"), Buffer.alloc(1000));
    for (let i = 1; i <= 3; i++) {
      const name = `split-0000${i}-of-00003.gguf`;
      fs.writeFileSync(path.join(dir, name), Buffer.alloc(1000));
    }
    fs.writeFileSync(path.join(dir, "partial-00001-of-00002.gguf"), Buffer.alloc(1000));
  });

  after(() => fs.rmSync(dir, { recursive: true, force: true }));

  it("returns the plain size for a single file", () => {
    const r = totalModelBytes(path.join(dir, "solo.gguf"));
    assert.equal(r.bytes, 1000);
    assert.equal(r.shardCount, 1);
  });

  it("sums every shard of a split model", () => {
    const r = totalModelBytes(path.join(dir, "split-00001-of-00003.gguf"));
    assert.equal(r.bytes, 3000, "must not report only the selected shard");
    assert.equal(r.shardCount, 3);
    assert.equal(r.shardsFound, 3);
  });

  it("gives the same total when a later shard is selected", () => {
    assert.equal(totalModelBytes(path.join(dir, "split-00003-of-00003.gguf")).bytes, 3000);
  });

  it("reports how many shards were actually found", () => {
    const r = totalModelBytes(path.join(dir, "partial-00001-of-00002.gguf"));
    assert.equal(r.shardsFound, 1);
    assert.equal(r.shardCount, 2);
    assert.equal(r.bytes, 1000);
  });

  it("does not throw for a missing file", () => {
    assert.equal(totalModelBytes(path.join(dir, "nope.gguf")).bytes, undefined);
  });
});

describe("readModelCapabilities", () => {
  it("rejects a non-GGUF file instead of parsing garbage", () => {
    const p = path.join(os.tmpdir(), `llama-aio-not-gguf-${process.pid}.gguf`);
    fs.writeFileSync(p, Buffer.from("this is not a gguf file at all"));
    try {
      assert.throws(() => readModelCapabilities(p), /GGUF/);
    } finally {
      fs.rmSync(p, { force: true });
    }
  });

  it("refuses a header claiming an absurd metadata count instead of hanging", () => {
    // Valid magic + version, then tensor/kv counts of 2^60.
    const header = Buffer.alloc(24);
    header.writeUInt32LE(0x46554747, 0);
    header.writeUInt32LE(3, 4);
    header.writeBigUInt64LE(1n << 60n, 8);
    header.writeBigUInt64LE(1n << 60n, 16);
    const p = path.join(os.tmpdir(), `llama-aio-bad-gguf-${process.pid}.gguf`);
    fs.writeFileSync(p, header);
    try {
      const started = Date.now();
      assert.throws(() => readModelCapabilities(p), /corrupt|Invalid GGUF/i);
      assert.ok(Date.now() - started < 2000, "must fail fast rather than loop");
    } finally {
      fs.rmSync(p, { force: true });
    }
  });
});

describe("clampLoadSettingsToModel", () => {
  it("caps context at the model maximum", () => {
    const s = clampLoadSettingsToModel(
      loadSettings({ contextLength: 999_999 }),
      denseCaps({ maxContextLength: 8192 })
    );
    assert.equal(s.contextLength, 8192);
  });

  it("recovers a finite context from corrupt input", () => {
    const s = clampLoadSettingsToModel(loadSettings({ contextLength: NaN }), denseCaps());
    assert.ok(Number.isFinite(s.contextLength));
  });

  it("keeps 99 as the all-layers sentinel", () => {
    assert.equal(clampLoadSettingsToModel(loadSettings({ gpuOffload: 99 }), denseCaps()).gpuOffload, 99);
  });

  it("caps an explicit layer count at the block count", () => {
    const s = clampLoadSettingsToModel(loadSettings({ gpuOffload: 60 }), denseCaps());
    assert.equal(s.gpuOffload, 48);
  });

  it("turns off MTP for models without next-n layers", () => {
    const s = clampLoadSettingsToModel(loadSettings({ speculativeMode: "mtp" }), denseCaps());
    assert.equal(s.speculativeMode, "off");
  });

  it("keeps MTP for models that support it", () => {
    const s = clampLoadSettingsToModel(
      loadSettings({ speculativeMode: "mtp" }),
      denseCaps({ nextnPredictLayers: 1 })
    );
    assert.equal(s.speculativeMode, "mtp");
  });

  it("zeroes --n-cpu-moe for dense models", () => {
    assert.equal(clampLoadSettingsToModel(loadSettings({ nCpuMoe: 12 }), denseCaps()).nCpuMoe, 0);
  });
});

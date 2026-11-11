import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, it } from "node:test";
import { ConfigFile } from "../src/config";
import { SettingsStore } from "../src/settings";
import { isPleOnlyGguf, readEffectiveModelCapabilities, readModelCapabilities, invalidateModelCapabilitiesCache } from "../src/ggufMetadata";
import { parseLlamaServerFeatures } from "../src/llamaFeatures";
import { classifyGgufFile, collapseLocalModelEntries, removalPathsForQuant } from "../src/modelLibrary";
import { estimateMemory } from "../src/memoryEstimate";
import { buildServerArgs, serverConfigFingerprint } from "../src/serverArgs";
import { diffLoadSettings } from "../src/settingsDiff";
import { ProcessManager } from "../src/processManager";
import { normalizeLoadSettings } from "../src/types";
import { loadSettings, miniGgufWithTensors, type MiniGgufValue } from "./helpers";

const tensor = "per_layer_token_embd.weight";
const meta: Record<string, MiniGgufValue> = {
  "general.architecture": { type: "string", value: "gemma4" },
  "gemma4.block_count": { type: "u32", value: 4 },
  "gemma4.context_length": { type: "u32", value: 8192 },
  "gemma4.embedding_length_per_layer_input": { type: "u32", value: 2 },
};
const embedding = { name: "token_embd.weight", bytes: 1024, dimensions: [16, 16] };
const original = { name: tensor, bytes: 4096, dimensions: [8, 16] };

function fixture(t: { after: (fn: () => void) => void }) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-external-ple-"));
  t.after(() => { invalidateModelCapabilitiesCache(); fs.rmSync(dir, { recursive: true, force: true }); });
  const write = (name: string, kv = meta, tensors = [embedding, original]) => {
    const file = path.join(dir, name);
    fs.writeFileSync(file, miniGgufWithTensors(kv, tensors));
    return file;
  };
  const model = write("model.gguf");
  // The external table deliberately has no tokenizer or architecture metadata.
  const ple = write("arbitrary-name.gguf", {}, [{ ...original, bytes: 512 }]);
  return { dir, write, model, ple };
}

describe("external PLE tables", () => {
  it("rejects partially truncated tables and unsafe tensor element counts", t => {
    const { model, ple } = fixture(t);
    fs.writeFileSync(ple, miniGgufWithTensors({}, [{ ...original, bytes: 511 }]));
    assert.throws(() => readEffectiveModelCapabilities(model, ple), /truncated/);
    fs.writeFileSync(ple, miniGgufWithTensors({}, [{ ...original, dimensions: [2 ** 40, 2 ** 30], bytes: 1024 }]));
    assert.throws(() => readEffectiveModelCapabilities(model, ple), /truncated/);
  });

  it("accepts aligned data padding but rejects invalid tensor offsets", t => {
    const { model, ple } = fixture(t);
    fs.writeFileSync(ple, miniGgufWithTensors({}, [{ ...original, bytes: 512, offset: 64 }]));
    assert.equal(readEffectiveModelCapabilities(model, ple).pleOverrideBytes, 512);
    fs.writeFileSync(ple, miniGgufWithTensors({}, [{ ...original, bytes: 512, offset: 1 }]));
    assert.throws(() => readEffectiveModelCapabilities(model, ple), /invalid tensor data/);
  });

  it("does not enumerate sibling files for an external numbered filename", t => {
    const { model, write } = fixture(t);
    const ple = write("external-00001-of-00100.gguf", {}, [{ ...original, bytes: 512 }]);
    const stat = t.mock.method(require("node:fs") as typeof fs, "statSync");
    readEffectiveModelCapabilities(model, ple);
    assert.equal(stat.mock.calls.some(c => String(c.arguments[0]).includes("external-00002")), false);
  });

  it("checks quantized row blocks and payload sizes", t => {
    const { write } = fixture(t);
    const kv = { ...meta, "gemma4.embedding_length_per_layer_input": { type: "u32" as const, value: 8 } };
    const model = write("wide.gguf", kv, [embedding, { ...original, dimensions: [32, 16] }]);
    const ple = write("quant.gguf", {}, [{ ...original, dimensions: [32, 16], bytes: 544 }]);
    fs.writeFileSync(ple, miniGgufWithTensors({}, [{ ...original, dimensions: [32, 16], bytes: 544, type: 8 }]));
    assert.equal(readEffectiveModelCapabilities(model, ple).pleOverrideBytes, 544);
    fs.truncateSync(ple, fs.statSync(ple).size - 1);
    assert.throws(() => readEffectiveModelCapabilities(model, ple), /truncated/);
  });

  it("accepts exact common GGML payload sizes and rejects a one-byte truncation", t => {
    const { write } = fixture(t);
    const model = write("formats.gguf", { ...meta,
      "gemma4.embedding_length_per_layer_input": { type: "u32", value: 64 } },
      [embedding, { ...original, dimensions: [256, 16], bytes: 16384 }]);
    // GGML b12233 row layouts: ggml.c / ggml-common.h static assertions.
    const formats = [[0,1,4], [1,1,2], [2,32,18], [3,32,20], [6,32,22],
      [7,32,24], [8,32,34], [9,32,36], [10,256,84], [11,256,110],
      [12,256,144], [13,256,176], [14,256,210], [15,256,292], [30,1,2]];
    for (const [type, block, size] of formats) {
      const bytes = 256 / block * size * 16;
      const file = write(`format-${type}.gguf`, {}, []);
      fs.writeFileSync(file, miniGgufWithTensors({}, [{ name: tensor, dimensions: [256, 16], bytes, type }]));
      assert.equal(readEffectiveModelCapabilities(model, file).pleOverrideBytes, bytes, `GGML type ${type}`);
      fs.truncateSync(file, fs.statSync(file).size - 1);
      assert.throws(() => readEffectiveModelCapabilities(model, file), /truncated/, `GGML type ${type}`);
    }
  });

  it("keeps repeated ordinary model classification to a fixed header read", t => {
    const { model } = fixture(t);
    invalidateModelCapabilitiesCache();
    const read = t.mock.method(require("node:fs") as typeof fs, "readSync");
    for (let i = 0; i < 100; i++) assert.equal(isPleOnlyGguf(model), false);
    assert.equal(read.mock.callCount(), 1);
    assert.equal((read.mock.calls[0].arguments as unknown[])[3], 24);
  });

  it("does not rescan tensor headers on repeated effective-capability reads", t => {
    const { model, ple } = fixture(t);
    readEffectiveModelCapabilities(model, ple);
    const read = t.mock.method(require("node:fs") as typeof fs, "readSync");
    for (let i = 0; i < 100; i++) readEffectiveModelCapabilities(model, ple);
    assert.equal(read.mock.callCount(), 0);
  });

  it("omits fit estimates when a required PLE attachment is missing", t => {
    const { write } = fixture(t);
    const model = write("missing-ple.gguf", meta, [embedding]);
    assert.equal(estimateMemory(readModelCapabilities(model), loadSettings()), undefined);
  });
  it("replaces embedded bytes once, keeps base caps unchanged and invalidates on replacement", t => {
    const { model, ple } = fixture(t);
    const base = readModelCapabilities(model);
    const caps = readEffectiveModelCapabilities(model, ple);
    assert.equal(caps.fileSizeBytes, base.fileSizeBytes! - original.bytes + 512);
    assert.equal(caps.pleOverrideBytes, 512);
    assert.equal(caps.fileSizeBytes! * caps.pleShare!, 512);
    assert.deepEqual(readModelCapabilities(model), base);
    caps.pleShare = 0;
    assert.notEqual(readEffectiveModelCapabilities(model, ple).pleShare, 0, "cached caps are not mutable by callers");
    const stamp = serverConfigFingerprint(model, loadSettings({ plePath: ple }));
    fs.writeFileSync(ple, miniGgufWithTensors({}, [{ ...original, bytes: 1024 }]));
    assert.equal(readEffectiveModelCapabilities(model, ple).pleOverrideBytes, 1024);
    assert.notEqual(serverConfigFingerprint(model, loadSettings({ plePath: ple })), stamp);
    assert.deepEqual(readEffectiveModelCapabilities(model), base);
  });

  it("supports PLE-less backbones and checks Gemma table geometry from metadata", t => {
    const { write, ple } = fixture(t);
    const model = write("no-ple.gguf", meta, [embedding]);
    assert.equal(readModelCapabilities(model).pleMissing, true);
    assert.throws(() => buildServerArgs(model, "127.0.0.1", 8742, loadSettings(), { caps: readModelCapabilities(model) }), /no PLE table/);
    assert.equal(readEffectiveModelCapabilities(model, ple).pleOverrideBytes, 512);
    fs.writeFileSync(ple, miniGgufWithTensors({}, [{ ...original, dimensions: [4, 128] }]));
    assert.throws(() => readEffectiveModelCapabilities(model, ple), /dimensions do not match/);
  });

  it("rejects invalid tables and unsupported binaries before reusing or stopping a running server", async t => {
    const { model, ple } = fixture(t);
    t.mock.method(require("../src/installSwap"), "activeInstallLock", () => undefined);
    const state = { selectedModelPath: model, loadSettings: loadSettings({ plePath: ple }) };
    const start = (ProcessManager.prototype as unknown as {
      startInner(model: string | undefined, report: (s: string) => void): Promise<unknown>;
    }).startInner;
    const manager = {
      store: { getState: () => state },
      getServerFeatures: async () => parseLlamaServerFeatures("--ctx-size N"),
      isHttpReady: async () => assert.fail("must validate before touching an existing server"),
    };
    await assert.rejects(start.call(manager, undefined, () => undefined), /b12233/);
    fs.unlinkSync(ple);
    await assert.rejects(start.call(manager, undefined, () => undefined), /ENOENT/);
  });

  it("preserves the running server when Reload has an invalid PLE attachment", async t => {
    const { model, ple } = fixture(t);
    fs.unlinkSync(ple);
    t.mock.method(require("../src/launchLock"), "acquireLaunchLock", () => ({ release() {} }));
    const manager = {
      store: { getState: () => ({ selectedModelPath: model, loadSettings: loadSettings({ plePath: ple }) }) },
      assertLocal() {}, takeLaunch() { return {}; }, releaseLaunch() {}, updateBootMessage() {},
      stopServer: async () => assert.fail("invalid Reload must preserve the running server"),
    };
    await assert.rejects(ProcessManager.prototype.reload.call(manager as unknown as ProcessManager), /ENOENT/);
  });

  it("validates hashed PLE row ranges and permits differently padded tables", t => {
    const { write } = fixture(t);
    const kv: Record<string, MiniGgufValue> = {
      "general.architecture": { type: "string", value: "qwen4exp" },
      "qwen4exp.block_count": { type: "u32", value: 4 },
      "qwen4exp.embedding_length_per_layer_input": { type: "u32", value: 8 },
      "qwen4exp.ple.head_offsets": { type: "u32[]", value: [0, 32] },
      "qwen4exp.ple.head_vocab_sizes": { type: "u32[]", value: [32, 32] },
    };
    const model = write("hashed-no-ple.gguf", kv, [embedding]);
    const ple = write("padded.gguf", {}, [{ ...original, dimensions: [8, 96] }]);
    assert.equal(readEffectiveModelCapabilities(model, ple).pleOverrideBytes, 4096);
    const small = write("too-small.gguf", {}, [{ ...original, dimensions: [8, 63] }]);
    assert.throws(() => readEffectiveModelCapabilities(model, small), /PLE head width and row ranges/);
    const corrupt = write("bad-ranges.gguf", { ...kv,
      "qwen4exp.ple.head_vocab_sizes": { type: "u32[]", value: [0] } }, [embedding]);
    assert.throws(() => readEffectiveModelCapabilities(corrupt, ple), /Invalid PLE head ranges/);
  });

  it("allows one dedicated PLE shard to be present or absent, but rejects missing backbone weights", t => {
    const { write, ple } = fixture(t);
    const kv = { ...meta, "split.tensors.count": { type: "u32" as const, value: 2 } };
    const model = write("split-00001-of-00002.gguf", kv, [embedding]);
    const shard = write("split-00002-of-00002.gguf", {}, [original]);
    const caps = readEffectiveModelCapabilities(model, ple);
    assert.equal(caps.fileSizeBytes, fs.statSync(model).size + 512);
    fs.unlinkSync(shard);
    assert.deepEqual(readEffectiveModelCapabilities(model, ple), caps);
    write("split-00001-of-00002.gguf", { ...kv, "split.tensors.count": { type: "u32", value: 3 } }, [embedding]);
    assert.throws(() => readEffectiveModelCapabilities(model, ple), /Incomplete backbone/);
  });

  it("rejects language files, wrong shapes/architectures, corrupt files and an already-discovered table", t => {
    const { model, ple, write } = fixture(t);
    assert.equal(isPleOnlyGguf(ple), true);
    assert.equal(isPleOnlyGguf(model), false);
    assert.throws(() => readEffectiveModelCapabilities(model, model), /PLE-only/);
    const wrong = write("wrong.gguf", {}, [{ ...original, dimensions: [16, 32] }]);
    assert.throws(() => readEffectiveModelCapabilities(model, wrong), /dimensions do not match/);
    const arch = write("arch.gguf", { "general.architecture": { type: "string", value: "qwen4exp" } }, [original]);
    assert.throws(() => readEffectiveModelCapabilities(model, arch), /architecture/);
    const broken = write("truncated.gguf", {}, [{ ...original, bytes: 0 }]);
    assert.throws(() => readEffectiveModelCapabilities(model, broken), /tensor data/);
    fs.writeFileSync(broken, "GGUF");
    assert.throws(() => readEffectiveModelCapabilities(model, broken));
    const backbone = write("numbered-00001-of-00002.gguf", meta, [embedding]);
    const shard = write("numbered-00002-of-00002.gguf", {}, [original]);
    assert.throws(() => readEffectiveModelCapabilities(backbone, shard), /already a model shard/);
  });

  it("recalculates expert shares without changing the absolute expert weight", t => {
    const { write, ple } = fixture(t);
    const model = write("moe.gguf", { ...meta, "gemma4.expert_count": { type: "u32", value: 8 } },
      [embedding, { name: "blk.0.ffn_down_exps.weight", bytes: 8192, dimensions: [16, 16] }, original]);
    const caps = readEffectiveModelCapabilities(model, ple);
    assert.equal(caps.moeExpertShare! * caps.fileSizeBytes!, 8192);
    const all = estimateMemory(caps, loadSettings({ gpuOffload: 99, lazyMode: "off" }));
    const half = estimateMemory(caps, loadSettings({ gpuOffload: 2, lazyMode: "off" }));
    assert.ok(all && half);
    assert.equal(all.gpuWeightsBytes, caps.fileSizeBytes! - 512);
    assert.equal(half.gpuWeightsBytes, all.gpuWeightsBytes / 2);
    assert.equal(all.cpuWeightsBytes, 512);
    const lazy = estimateMemory(caps, loadSettings({ gpuOffload: 99, lazyMode: "on", tryMmap: true }));
    assert.equal(lazy?.cpuWeightsBytes, 0);
  });

  it("persists the attachment, clears it on model changes, and leaves Clear usable after deletion", async t => {
    const { dir, model, ple, write } = fixture(t);
    const config = new ConfigFile(path.join(dir, "config.json"));
    t.after(() => config.dispose());
    const store = new SettingsStore(config);
    await store.applySelectedModel(model, { recommendDefaults: false });
    await store.updateLoadSettings({ plePath: ` ${ple} ` });
    assert.equal(store.getState().loadSettings.plePath, ple);
    assert.equal(store.getState().modelCapabilities?.pleOverrideBytes, 512);
    assert.equal(config.getState().modelCapabilities?.pleOverridePath, undefined, "base caps remain persisted");
    const other = new ConfigFile(config.path);
    t.after(() => other.dispose());
    assert.equal(new SettingsStore(other).getState().modelCapabilities?.pleOverrideBytes, 512);
    await assert.rejects(store.updateLoadSettings({ plePath: model }), /PLE-only/);
    assert.equal(store.getState().loadSettings.plePath, ple, "invalid pick does not change the attachment");
    fs.unlinkSync(ple);
    assert.ok(store.getState().modelCapabilities?.pleOverrideError);
    assert.equal(store.getState().modelCapabilities?.fileSizeBytes, undefined);
    await store.updateLoadSettings({ plePath: "" });
    assert.equal(store.getState().modelCapabilities?.fileSizeBytes, fs.statSync(model).size);
    write("arbitrary-name.gguf", {}, [{ ...original, bytes: 512 }]);
    await store.updateLoadSettings({ plePath: ple });
    await store.applySelectedModel(write("other.gguf"), { recommendDefaults: false });
    assert.equal(store.getState().loadSettings.plePath, "");
    await assert.rejects(store.applySelectedModel(ple), /PLE-only table/);
  });

  it("uses an advertised flag, requires support, and reports reload differences", t => {
    const { model, ple } = fixture(t);
    assert.equal(normalizeLoadSettings({}).plePath, "");
    const settings = loadSettings({ plePath: ple });
    for (const flag of ["--ple", "--shard"] as const) {
      const features = parseLlamaServerFeatures(`${flag} PATH`);
      const args = buildServerArgs(model, "127.0.0.1", 8742, settings, { features });
      assert.equal(args[args.indexOf(flag) + 1], ple);
    }
    assert.equal(parseLlamaServerFeatures("Use --shard on recent builds\n--shard-extra PATH").externalPle, undefined);
    assert.throws(() => buildServerArgs(model, "127.0.0.1", 8742, settings), /b12233/);
    assert.throws(() => buildServerArgs(model, "127.0.0.1", 8742, loadSettings({ plePath: "a,b.gguf" }),
      { features: parseLlamaServerFeatures("--shard PATH") }), /comma/);
    assert.notEqual(serverConfigFingerprint(model, settings), serverConfigFingerprint(model, loadSettings()));
    assert.deepEqual(diffLoadSettings(loadSettings(), settings).map(c => [c.key, c.to]), [["plePath", path.basename(ple)]]);
    const aliases = parseLlamaServerFeatures("--shard, --ple PATH");
    assert.equal(aliases.externalPle, "--shard");
  });

  it("classifies PLE files by contents and protects shared tables during companion removal", t => {
    const { dir, model, ple, write } = fixture(t);
    assert.equal(classifyGgufFile({ path: ple, size: fs.statSync(ple).size }), "ple");
    const plan = removalPathsForQuant(model, dir, [ple]);
    assert.deepEqual(plan.files, [model]);
    assert.deepEqual(plan.rejected, [ple]);
    const first = write("sharded-00001-of-00002.gguf", meta, [embedding]);
    const second = write("sharded-00002-of-00002.gguf", {}, [original]);
    const entries = [first, second].map(p => ({ path: p, source: "test", sizeBytes: fs.statSync(p).size }));
    assert.equal(collapseLocalModelEntries(entries)[0]?.sizeBytes, entries[0].sizeBytes + entries[1].sizeBytes);
    assert.deepEqual(collapseLocalModelEntries([entries[1]]), []);
  });
});

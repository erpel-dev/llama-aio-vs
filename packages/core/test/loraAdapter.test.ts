import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { after, describe, it } from "node:test";
import { argValue, denseCaps, loadSettings, miniGguf, miniGgufWithTensors } from "./helpers";
import { isLoraAdapterGguf, readLoraAdapterInfo, readModelCapabilities } from "../src/ggufMetadata";
import { checkLoraAdapter, describeLoraLoadError, modelLineage, planLoraScales } from "../src/loraAdapter";
import { parseLlamaServerFeatures } from "../src/llamaFeatures";
import { buildServerArgs, serverConfigFingerprint } from "../src/serverArgs";
import { diffLoadSettings } from "../src/settingsDiff";
import { classifyGgufFile } from "../src/modelLibrary";
import { normalizeLoadSettings } from "../src/types";

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-lora-"));
after(() => fs.rmSync(dir, { recursive: true, force: true }));
const write = (name: string, bytes: Buffer) => {
  const file = path.join(dir, name);
  fs.writeFileSync(file, bytes);
  return file;
};
const str = (value: string) => ({ type: "string" as const, value });
const adapter = (name: string, extra: Parameters<typeof miniGguf>[0] = {}) => write(name, miniGgufWithTensors({
  "general.architecture": str("qwen35"),
  "general.type": str("adapter"),
  "adapter.type": str("lora"),
  "adapter.lora.alpha": { type: "u32", value: 8 },
  ...extra,
}, [
  { name: "blk.0.ffn_up.weight.lora_a", bytes: 64, dimensions: [16, 4] },
  { name: "blk.0.ffn_up.weight.lora_b", bytes: 64, dimensions: [4, 16] },
]));
const model = denseCaps({ path: "/models/Qwen3.5-2B-Q4_K_M.gguf", name: "Qwen3.5-2B", architecture: "qwen35" });
const features = parseLlamaServerFeatures("--lora FNAME   path to LoRA adapter");

describe("LoRA adapter files", () => {
  it("tells adapter files from models using only the start of the file", () => {
    const lora = adapter("a-lora.gguf");
    const plain = write("model.gguf", miniGguf({ "general.architecture": str("qwen35"), "general.type": str("model") }));
    assert.equal(isLoraAdapterGguf(lora), true);
    assert.equal(isLoraAdapterGguf(plain), false);
    assert.equal(isLoraAdapterGguf(path.join(dir, "missing.gguf")), false);
    assert.equal(isLoraAdapterGguf(write("text.gguf", Buffer.from("general.type adapter, but not a GGUF"))), false);
    assert.equal(classifyGgufFile({ path: lora, size: fs.statSync(lora).size }), "lora");
  });

  it("reads adapter metadata and rejects other files", () => {
    const info = readLoraAdapterInfo(adapter("info-lora.gguf", { "general.base_model.0.name": str("Qwen3.5 2B") }));
    assert.equal(info.architecture, "qwen35");
    assert.equal(info.baseModelName, "Qwen3.5 2B");
    assert.equal(info.alpha, 8);
    assert.equal(info.activated, false);
    assert.equal(info.tensorCount, 2);
    const activated = adapter("alora.gguf", { "adapter.alora.invocation_tokens": { type: "u32[]", value: [5, 6] } });
    assert.equal(readLoraAdapterInfo(activated).activated, true);
    const plain = write("plain.gguf", miniGguf({ "general.architecture": str("qwen35") }));
    assert.throws(() => readLoraAdapterInfo(plain), /not a LoRA adapter/);
    const control = write("cv.gguf", miniGguf({ "general.type": str("adapter"), "adapter.type": str("control_vector") }));
    assert.throws(() => readLoraAdapterInfo(control), /Only LoRA adapters/);
  });

  it("grades compatibility from what the files say", () => {
    assert.equal(checkLoraAdapter(adapter("ok.gguf", { "general.base_model.0.name": str("Qwen3.5 2B") }), model).fit, "match");
    assert.equal(checkLoraAdapter(adapter("nobase.gguf"), model).fit, "unknown-base");
    const other = checkLoraAdapter(adapter("other.gguf", { "general.base_model.0.name": str("Qwen3.5 9B") }), model);
    assert.equal(other.fit, "base-mismatch");
    assert.match(other.message, /Qwen3\.5 9B/);
    const wrong = checkLoraAdapter(adapter("wrong.gguf", { "general.architecture": str("gemma3") }), model);
    assert.equal(wrong.fit, "wrong-architecture");
    assert.match(wrong.message, /gemma3.*qwen35/);
    assert.equal(checkLoraAdapter(path.join(dir, "missing.gguf"), model).fit, "invalid");
  });

  it("reports built-in adapters and lineage on model files, and nothing for a plain model", () => {
    const withAdapters = write("switch.gguf", miniGguf({
      "general.architecture": str("granite-switch"),
      "granite-switch.context_length": { type: "u32", value: 4096 },
      "granite-switch.block_count": { type: "u32", value: 2 },
      "granite-switch.adapters.count": { type: "u32", value: 12 },
      "granite-switch.adapters.lora_rank": { type: "u32", value: 32 },
      "general.finetune": str("Storywriter"),
      "general.base_model.0.name": str("Granite 8B"),
    }));
    const caps = readModelCapabilities(withAdapters);
    assert.deepEqual(caps.builtInAdapters, { count: 12, rank: 32 });
    assert.equal(caps.finetune, "Storywriter");
    assert.equal(caps.baseModelName, "Granite 8B");
    const plain = readModelCapabilities(write("plain-model.gguf", miniGguf({
      "general.architecture": str("qwen35"),
      "qwen35.context_length": { type: "u32", value: 4096 },
      "qwen35.block_count": { type: "u32", value: 2 },
    })));
    assert.equal(plain.builtInAdapters, undefined);
    assert.equal(plain.finetune, undefined);
  });
});

describe("model lineage", () => {
  it("is shown only when the file names a different base model", () => {
    // Quantized uploads name the original as their base: that is not a fine-tune.
    assert.equal(modelLineage(denseCaps({ name: "Qwen3.5-2B", baseModelName: "Qwen3.5 2B" })), undefined);
    assert.equal(modelLineage(denseCaps({ name: "Gemma 4 E4B It", baseModelName: "Gemma 4 E4B", finetune: "it" })), undefined);
    assert.equal(modelLineage(denseCaps({ name: "Qwen3.5-2B" })), undefined);
    assert.deepEqual(modelLineage(denseCaps({ name: "Storywriter 2B", baseModelName: "Qwen3.5 2B", finetune: "Storywriter" })),
      { baseModelName: "Qwen3.5 2B", finetune: "Storywriter" });
    assert.deepEqual(modelLineage(denseCaps({ name: "Qwen3.5-2B-Storywriter", baseModelName: "Qwen3.5 2B", finetune: "Storywriter" })),
      { baseModelName: "Qwen3.5 2B", finetune: "Storywriter" });
  });
});

describe("LoRA server settings", () => {
  const two = [
    { path: "/l/a.gguf", scale: 0.8, enabled: true },
    { path: "C:\\l\\b.gguf", scale: 1, enabled: false },
  ];

  it("normalizes the adapter list", () => {
    const s = normalizeLoadSettings({ loraAdapters: [
      { path: " /l/a.gguf ", scale: 9, enabled: true }, { path: "/l/a.gguf", scale: 1, enabled: true },
      { path: "", scale: 1, enabled: true }, { path: "/l/b.gguf", scale: "x", enabled: undefined },
    ] as never });
    assert.deepEqual(s.loraAdapters, [
      { path: "/l/a.gguf", scale: 2, enabled: true }, { path: "/l/b.gguf", scale: 1, enabled: true },
    ]);
    assert.deepEqual(normalizeLoadSettings({}).loraAdapters, []);
  });

  it("passes one comma-separated --lora and keeps Windows drive paths intact", () => {
    const args = buildServerArgs("/m/model.gguf", "127.0.0.1", 8742, loadSettings({ loraAdapters: two }), { features });
    assert.equal(argValue(args, "--lora"), "/l/a.gguf,C:\\l\\b.gguf");
    assert.equal(args.includes("--lora-scaled"), false);
    assert.equal(buildServerArgs("/m/model.gguf", "127.0.0.1", 8742, loadSettings(), { features }).includes("--lora"), false);
  });

  it("refuses builds without --lora and paths with commas", () => {
    assert.throws(() => buildServerArgs("/m/model.gguf", "127.0.0.1", 8742, loadSettings({ loraAdapters: two }),
      { features: parseLlamaServerFeatures("--ctx-size N") }), /Update llama\.cpp/);
    assert.throws(() => buildServerArgs("/m/model.gguf", "127.0.0.1", 8742,
      loadSettings({ loraAdapters: [{ path: "/l/a,b.gguf", scale: 1, enabled: true }] }), { features }), /comma/);
  });

  it("needs a reload only when the file list changes", () => {
    const base = loadSettings({ loraAdapters: two });
    const retuned = loadSettings({ loraAdapters: [{ ...two[0], scale: 0.2, enabled: false }, two[1]] });
    assert.deepEqual(diffLoadSettings(base, retuned), []);
    assert.equal(serverConfigFingerprint("/m/model.gguf", base, "background"), serverConfigFingerprint("/m/model.gguf", retuned, "background"));
    const removed = loadSettings({ loraAdapters: [two[0]] });
    assert.deepEqual(diffLoadSettings(base, removed), [
      { key: "loraAdapters", label: "LoRA adapters", from: "a.gguf, b.gguf", to: "a.gguf" },
    ]);
    assert.notEqual(serverConfigFingerprint("/m/model.gguf", base, "background"), serverConfigFingerprint("/m/model.gguf", removed, "background"));
    assert.equal(diffLoadSettings(loadSettings(), base)[0].from, "none");
  });

  it("maps strengths to the server's adapter ids, with switched-off adapters at 0", () => {
    const wanted = [{ path: "/l/a.gguf", scale: 0.8, enabled: true }, { path: "/l/b.gguf", scale: 1, enabled: false }];
    const loaded = ["/l/a.gguf", "/l/b.gguf"];
    assert.deepEqual(planLoraScales(loaded, wanted, [{ id: 3, path: "/l/b.gguf" }, { id: 7, path: "/l/a.gguf" }]),
      [{ id: 7, scale: 0.8 }, { id: 3, scale: 0 }]);
    assert.deepEqual(planLoraScales(loaded, wanted, [{ id: 0 }, { id: 1 }]), [{ id: 0, scale: 0.8 }, { id: 1, scale: 0 }]);
    assert.throws(() => planLoraScales(loaded, wanted, [{ id: 0, path: "/l/a.gguf" }]), /did not load b\.gguf/);
  });

  it("keeps loaded adapters live while file changes wait for a reload", () => {
    const server = [{ id: 0, path: "/l/a.gguf", scale: 1 }, { id: 1, path: "/l/b.gguf", scale: 0.6 }];
    // A loaded, B added but not loaded yet: switching A off still reaches the server.
    assert.deepEqual(planLoraScales(["/l/a.gguf"], [
      { path: "/l/a.gguf", scale: 1, enabled: false }, { path: "/l/new.gguf", scale: 1, enabled: true },
    ], [server[0]]), [{ id: 0, scale: 0 }]);
    // B removed in the sidebar but still loaded: A changes, B keeps what the server has until Reload.
    assert.deepEqual(planLoraScales(["/l/a.gguf", "/l/b.gguf"], [{ path: "/l/a.gguf", scale: 0.3, enabled: true }], server),
      [{ id: 0, scale: 0.3 }, { id: 1, scale: 0.6 }]);
    assert.deepEqual(planLoraScales([], [{ path: "/l/a.gguf", scale: 1, enabled: true }], []), []);
  });

  it("explains llama-server's adapter load errors", () => {
    assert.match(describeLoraLoadError("failed to apply lora adapter: model arch and LoRA arch mismatch")!, /different model family/);
    assert.match(describeLoraLoadError("tensor 'blk.0.ffn_up.weight' has incorrect shape (hint: maybe wrong base model?)")!, /different base model/);
    assert.match(describeLoraLoadError("common_init_: failed to load lora adapter '/l/a.gguf'")!, /could not load/);
    assert.equal(describeLoraLoadError("error: out of memory"), undefined);
  });
});

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, it } from "node:test";
import { findLoraCandidates, findPleCandidates } from "../src/attachmentCandidates";
import { invalidateModelCapabilitiesCache, readModelCapabilities } from "../src/ggufMetadata";
import { invalidateModelLibraryCache } from "../src/modelLibrary";
import { miniGguf, miniGgufWithTensors, type MiniGgufValue } from "./helpers";

const str = (value: string) => ({ type: "string" as const, value });

function library(t: { after: (fn: () => void) => void }) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-candidates-"));
  const models = path.join(root, "models");
  const outside = path.join(root, "outside");
  fs.mkdirSync(path.join(models, "sub"), { recursive: true });
  fs.mkdirSync(outside);
  invalidateModelLibraryCache();
  t.after(() => {
    invalidateModelLibraryCache();
    invalidateModelCapabilitiesCache();
    fs.rmSync(root, { recursive: true, force: true });
  });
  const write = (file: string, bytes: Buffer) => { fs.writeFileSync(file, bytes); return file; };
  const config = { get: (key: string) => (key === "modelsDir" ? models : undefined) } as never;
  // Other scan roots (LM Studio, the Hugging Face cache…) depend on the machine: look at this library only.
  const mine = <T extends { path: string }>(list: T[]) => list.filter(c => c.path.startsWith(root));
  return { root, models, outside, write, config, mine };
}

describe("LoRA adapters found in the library", () => {
  const lora = (kv: Record<string, MiniGgufValue> = {}) => miniGgufWithTensors({
    "general.architecture": str("qwen35"), "general.type": str("adapter"), "adapter.type": str("lora"), ...kv,
  }, [{ name: "blk.0.ffn_up.weight.lora_a", bytes: 64, dimensions: [16, 4] }]);
  const modelKv = {
    "general.architecture": str("qwen35"), "general.name": str("Qwen3.5-2B"),
    "qwen35.context_length": { type: "u32" as const, value: 4096 }, "qwen35.block_count": { type: "u32" as const, value: 2 },
  };

  it("lists adapters best fit first, marks unusable and already added ones, and ignores other GGUFs", t => {
    const { models, write, config, mine } = library(t);
    const model = write(path.join(models, "Qwen3.5-2B-Q4_K_M.gguf"), miniGguf(modelKv));
    write(path.join(models, "z-match-lora.gguf"), lora({ "general.base_model.0.name": str("Qwen3.5 2B") }));
    write(path.join(models, "sub", "a-nobase-lora.gguf"), lora());
    write(path.join(models, "b-other-base-lora.gguf"), lora({ "general.base_model.0.name": str("Qwen3.5 9B") }));
    write(path.join(models, "c-gemma-lora.gguf"), lora({ "general.architecture": str("gemma3") }));
    write(path.join(models, "plain-model.gguf"), miniGguf(modelKv));
    fs.writeFileSync(path.join(models, "notes-lora.txt"), "not a gguf");

    const found = mine(findLoraCandidates(config, model, readModelCapabilities(model),
      [path.join(models, "sub", "a-nobase-lora.gguf")]));
    assert.deepEqual(found.map(c => [path.basename(c.path), c.usable, c.verdict, c.attached]), [
      ["z-match-lora.gguf", true, "Matches this model", false],
      ["a-nobase-lora.gguf", true, "Same model family, base model not named", true],
      ["b-other-base-lora.gguf", true, "Same model family, different base model", false],
      ["c-gemma-lora.gguf", false, "Different model family", false],
    ]);
    assert.equal(found[0].note, "");
    assert.match(found[2].note, /Qwen3\.5 9B/);
    assert.match(found[3].note, /gemma3/);
    assert.equal(found[0].source, "Llama AIO");
    assert.ok(found[0].sizeBytes > 0);
  });

  it("also looks next to a model opened from outside the library", t => {
    const { outside, write, config, mine } = library(t);
    const model = write(path.join(outside, "Qwen3.5-2B-Q4_K_M.gguf"), miniGguf(modelKv));
    write(path.join(outside, "beside-lora.gguf"), lora());
    const found = mine(findLoraCandidates(config, model, readModelCapabilities(model)));
    assert.deepEqual(found.map(c => [path.basename(c.path), c.source]), [["beside-lora.gguf", "Next to the model"]]);
  });

  it("returns nothing when no adapter exists", t => {
    const { models, write, config, mine } = library(t);
    const model = write(path.join(models, "Qwen3.5-2B-Q4_K_M.gguf"), miniGguf(modelKv));
    assert.deepEqual(mine(findLoraCandidates(config, model, readModelCapabilities(model))), []);
  });
});

describe("PLE tables found in the library", () => {
  const tensor = "per_layer_token_embd.weight";
  const meta: Record<string, MiniGgufValue> = {
    "general.architecture": str("gemma4"),
    "gemma4.block_count": { type: "u32", value: 4 },
    "gemma4.context_length": { type: "u32", value: 8192 },
    "gemma4.embedding_length_per_layer_input": { type: "u32", value: 2 },
  };
  const embedding = { name: "token_embd.weight", bytes: 1024, dimensions: [16, 16] };
  const table = { name: tensor, bytes: 4096, dimensions: [8, 16] };

  it("lists PLE-only files with fitting ones first and says why the others do not fit", t => {
    const { models, write, config, mine } = library(t);
    const model = write(path.join(models, "model.gguf"), miniGgufWithTensors(meta, [embedding, table]));
    const good = write(path.join(models, "sub", "ple-q8.gguf"), miniGgufWithTensors({}, [{ ...table, bytes: 512 }]));
    write(path.join(models, "a-wrong-shape-ple.gguf"), miniGgufWithTensors({}, [{ ...table, dimensions: [8, 32], bytes: 512 }]));

    const found = mine(findPleCandidates(config, model, good));
    assert.deepEqual(found.map(c => [path.basename(c.path), c.usable, c.attached]), [
      ["ple-q8.gguf", true, true],
      ["a-wrong-shape-ple.gguf", false, false],
    ]);
    assert.equal(found[0].verdict, "Dimensions match this model");
    assert.equal(found[1].verdict, "Does not fit this model");
    assert.ok(found[1].note.length > 0, "the reason is kept for the picker");
    assert.equal(found.some(c => c.path === model), false, "the model itself is not a PLE table");
  });
});

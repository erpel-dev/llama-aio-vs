import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { after, describe, it } from "node:test";
import { ConfigFile, normalizeConfig } from "../src/config";
import { chatTemplateFileProblem } from "../src/serverArgs";
import { SettingsStore } from "../src/settings";
import { DEFAULT_LOAD_SETTINGS, normalizeLoadSettings } from "../src/types";
import { miniGguf } from "./helpers";

const tmp: string[] = [];

after(() => {
  for (const p of tmp) {
    fs.rmSync(p, { recursive: true, force: true });
  }
});

function tempDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-chat-tpl-"));
  tmp.push(dir);
  return dir;
}

function writeModel(dir: string, name: string): string {
  const file = path.join(dir, name);
  fs.writeFileSync(
    file,
    miniGguf({
      "general.architecture": { type: "string", value: "qwen3" },
      "general.name": { type: "string", value: name },
      "qwen3.block_count": { type: "u32", value: 8 },
      "qwen3.context_length": { type: "u32", value: 8192 },
      "qwen3.embedding_length": { type: "u32", value: 512 },
      "qwen3.attention.head_count": { type: "u32", value: 8 },
      "qwen3.attention.head_count_kv": { type: "u32", value: 2 },
    })
  );
  return file;
}

describe("chat template config", () => {
  it("drops default and unknown entries from the per-model map", () => {
    const kept = path.resolve("/models/kept.gguf");
    const cfg = normalizeConfig({
      app: {
        chatTemplates: {
          "  /models/kept.gguf  ": { source: "builtin", name: "chatml", file: "", jinja: false },
          "/models/empty.gguf": { source: "model", name: "", file: "", jinja: true },
          "/models/bogus.gguf": { source: "builtin", name: "nope", file: "", jinja: true },
        },
      },
    } as never);
    assert.deepEqual(cfg.app.chatTemplates, {
      [kept]: { source: "builtin", name: "chatml", file: "", jinja: true },
    });
  });
});

describe("chatTemplateFileProblem", () => {
  it("accepts a small .jinja file and refuses a missing, huge, or non-template path", () => {
    const dir = tempDir();
    const ok = path.join(dir, "ok.jinja");
    fs.writeFileSync(ok, "{% for m in messages %}{{ m.content }}{% endfor %}\n");
    const huge = path.join(dir, "huge.jinja");
    fs.writeFileSync(huge, Buffer.alloc(256 * 1024 + 1));
    const base = normalizeLoadSettings(undefined);

    assert.equal(
      chatTemplateFileProblem({ ...base, chatTemplateSource: "file", chatTemplateFile: ok }),
      undefined
    );
    assert.match(
      chatTemplateFileProblem({
        ...base,
        chatTemplateSource: "file",
        chatTemplateFile: path.join(dir, "missing.jinja"),
      }) || "",
      /missing/
    );
    assert.match(
      chatTemplateFileProblem({ ...base, chatTemplateSource: "file", chatTemplateFile: huge }) || "",
      /too large/
    );
    assert.match(
      chatTemplateFileProblem({
        ...base,
        chatTemplateSource: "file",
        chatTemplateFile: path.join(dir, "model.gguf"),
      }) || "",
      /\.jinja or \.txt/
    );
    assert.equal(chatTemplateFileProblem(base), undefined);
  });
});

describe("per-model chat template", () => {
  it("restores each model's template when the selection changes", async () => {
    const dir = tempDir();
    const modelA = writeModel(dir, "a.gguf");
    const modelB = writeModel(dir, "b.gguf");
    const store = new SettingsStore(new ConfigFile(path.join(dir, "config.json")));

    await store.applySelectedModel(modelA, { recommendDefaults: false });
    await store.updateLoadSettings({ chatTemplateSource: "builtin", chatTemplateName: "chatml" });
    assert.equal(store.getState().loadSettings.chatTemplateName, "chatml");

    await store.applySelectedModel(modelB, { recommendDefaults: false });
    assert.equal(store.getState().loadSettings.chatTemplateSource, "model");
    assert.equal(store.getState().loadSettings.chatTemplateName, "");

    const templateFile = path.join(dir, "custom.jinja");
    fs.writeFileSync(templateFile, "{{ messages }}\n");
    await store.updateLoadSettings({
      chatTemplateSource: "file",
      chatTemplateFile: templateFile,
    });

    await store.applySelectedModel(modelA, { recommendDefaults: false });
    assert.equal(store.getState().loadSettings.chatTemplateSource, "builtin");
    assert.equal(store.getState().loadSettings.chatTemplateName, "chatml");
    assert.equal(store.getState().loadSettings.chatTemplateFile, "");

    await store.applySelectedModel(modelB, { recommendDefaults: false });
    assert.equal(store.getState().loadSettings.chatTemplateSource, "file");
    assert.equal(store.getState().loadSettings.chatTemplateFile, templateFile);

    await store.updateLoadSettings({
      chatTemplateSource: DEFAULT_LOAD_SETTINGS.chatTemplateSource,
      chatTemplateName: DEFAULT_LOAD_SETTINGS.chatTemplateName,
      chatTemplateFile: DEFAULT_LOAD_SETTINGS.chatTemplateFile,
      chatTemplateJinja: DEFAULT_LOAD_SETTINGS.chatTemplateJinja,
    });
    const map = store.getConfig().get<Record<string, { name?: string }>>("chatTemplates") || {};
    assert.equal(map[path.resolve(modelB)], undefined);
    assert.equal(map[path.resolve(modelA)]?.name, "chatml");

    await store.applySelectedModel(modelA, { recommendDefaults: false });
    assert.equal(store.getState().loadSettings.chatTemplateName, "chatml");
    await store.applySelectedModel(modelB, { recommendDefaults: false });
    assert.equal(store.getState().loadSettings.chatTemplateSource, "model");
    assert.equal(
      (store.getConfig().get<Record<string, { name?: string }>>("chatTemplates") || {})[path.resolve(modelA)]?.name,
      "chatml"
    );
  });
});

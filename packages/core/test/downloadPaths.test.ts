import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as http from "node:http";
import * as os from "node:os";
import * as path from "node:path";
import { describe, it } from "node:test";
import { HuggingFaceClient } from "../src/huggingFace";
import type { SettingsStore } from "../src/settings";

describe("Hugging Face local destinations", () => {
  it("preserves distinct nested paths and their downloaded bytes", async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-hf-paths-"));
    const server = http.createServer((req, res) => res.end(req.url!.includes("/Q4/") ? "Q444" : "Q888"));
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const old = process.env.HF_ENDPOINT;
    process.env.HF_ENDPOINT = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
    t.after(async () => {
      if (old === undefined) delete process.env.HF_ENDPOINT; else process.env.HF_ENDPOINT = old;
      await new Promise<void>((resolve) => server.close(() => resolve()));
      fs.rmSync(dir, { recursive: true, force: true });
    });
    const hf = new HuggingFaceClient({ getConfig: () => ({ get: () => dir }) } as unknown as SettingsStore);
    const files = ["Q4/model.gguf", "Q8/model.gguf"].map((p) => ({ path: p, size: 4, url: "" }));
    const first = await hf.downloadModelFile("org/repo", files[0]!.path, undefined, files);
    const second = await hf.downloadModelFile("org/repo", files[1]!.path, undefined, files);
    assert.notEqual(first, second);
    assert.equal(fs.readFileSync(first, "utf8"), "Q444");
    assert.equal(fs.readFileSync(second, "utf8"), "Q888");
    assert.equal(hf.localDestFor("org/repo", "model.gguf"), path.join(dir, "org__repo", "model.gguf"), "root paths stay compatible");
    assert.equal(hf.localDestFor("org/repo", "MTP/model.gguf"), path.join(dir, "org__repo", "MTP", "model.gguf"));
  });

  it("rejects traversal, absolute paths and platform-specific separators", () => {
    const hf = new HuggingFaceClient({ getConfig: () => ({ get: () => "/models" }) } as unknown as SettingsStore);
    for (const file of ["../model.gguf", "sub/../../model.gguf", "/model.gguf", "C:/model.gguf", "sub\\model.gguf", "sub//model.gguf", "model\0.gguf"]) {
      assert.throws(() => hf.localDestFor("org/repo", file), /Invalid/);
    }
    for (const repo of ["../repo", "/org/repo", "org/../repo", "C:\\repo", "repo"]) {
      assert.throws(() => hf.localDestFor(repo, "model.gguf"));
    }
  });
});

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, it } from "node:test";
import { ConfigFile } from "../src/config";
import { SettingsStore } from "../src/settings";
import { resetHardwareDetectionCache } from "../src/llamaInstaller";
import type { GpuMemoryInfo } from "../src/gpuInfo";
import { miniGguf } from "./helpers";

describe("asynchronous model selection", () => {
  it("keeps auto backend detection asynchronous with a cold hardware cache", async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-select-auto-"));
    t.after(() => { resetHardwareDetectionCache(); fs.rmSync(dir, { recursive: true, force: true }); });
    const config = new ConfigFile(path.join(dir, "config.json"));
    await config.setPreferences({ backend: "auto", installDir: path.join(dir, "install") });
    const model = path.join(dir, "model.gguf");
    fs.writeFileSync(model, miniGguf({}));
    resetHardwareDetectionCache();
    const lookup = require("../src/paths").whichOnPath;
    t.mock.method(require("../src/paths"), "whichOnPath", (name: string) => name === "nvidia-smi" ? path.join(dir, name) : lookup(name));
    t.mock.method(require("child_process"), "execFileSync", () => assert.fail("synchronous auto-backend probe"));
    let yielded = false;
    t.mock.method(require("../src/execRunner"), "execText", async () => {
      await new Promise((resolve) => setImmediate(resolve));
      yielded = true;
      return { ok: true, status: 0, stdout: "", stderr: "" };
    });
    t.mock.method(require("../src/gpuInfo"), "detectGpusAsync", async () => []);
    const result = new SettingsStore(config).applySelectedModel(model, { cpuOnly: false });
    await result;
    if (process.platform !== "darwin" && !process.env.CUDA_PATH && !process.env.CUDA_HOME && !fs.existsSync("/dev/nvidia0")) {
      assert.equal(yielded, true);
    }
  });

  for (const cpuOnly of [false, true]) for (const foundGpu of [false, true]) {
    it(`uses asynchronous GPU discovery without synchronous tools (CPU=${cpuOnly}, GPU=${foundGpu})`, async (t) => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-select-async-"));
      t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
      const config = new ConfigFile(path.join(dir, "config.json"));
      await config.setPreferences({ backend: "path", installDir: path.join(dir, "install") });
      const store = new SettingsStore(config);
      const model = path.join(dir, "model.gguf");
      fs.writeFileSync(model, miniGguf({ "general.architecture": { type: "string", value: "llama" } }));
      t.mock.method(require("child_process"), "execFileSync", () => assert.fail("synchronous GPU tool"));
      t.mock.method(require("child_process"), "spawnSync", () => assert.fail("synchronous list-devices"));
      let heartbeat = false;
      const probe = t.mock.method(require("../src/gpuInfo"), "detectGpusAsync", async () => {
        await new Promise((resolve) => setImmediate(resolve));
        heartbeat = true;
        return foundGpu ? [{ totalBytes: 8 * 1024 ** 3, usedBytes: 0, name: "test GPU", index: 0 }] : [];
      });
      const selecting = store.applySelectedModel(model, { cpuOnly });
      assert.equal(heartbeat, false);
      const selected = await selecting;
      assert.equal(probe.mock.callCount(), cpuOnly ? 0 : 1);
      assert.equal(heartbeat, !cpuOnly);
      assert.equal(selected.selectedModelPath, model);
    });
  }

  it("keeps a later selection when an earlier GPU probe finishes last", async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-select-order-"));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const config = new ConfigFile(path.join(dir, "config.json"));
    await config.setPreferences({ backend: "path", installDir: path.join(dir, "install") });
    const store = new SettingsStore(config);
    const models = ["first.gguf", "last.gguf"].map((name) => path.join(dir, name));
    for (const model of models) fs.writeFileSync(model, miniGguf({}));
    const release: Array<(gpus: GpuMemoryInfo[]) => void> = [];
    t.mock.method(require("../src/gpuInfo"), "detectGpusAsync", () => new Promise<GpuMemoryInfo[]>((resolve) => release.push(resolve)));
    const first = store.applySelectedModel(models[0]!, { cpuOnly: false });
    const last = store.applySelectedModel(models[1]!, { cpuOnly: false });
    while (release.length < 2) await new Promise((resolve) => setImmediate(resolve));
    release[1]!([]); await last;
    release[0]!([]); await first;
    assert.equal(store.getState().selectedModelPath, models[1]);
  });
});

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as net from "node:net";
import * as path from "node:path";
import { it } from "node:test";
import { ConfigFile } from "../src/config";
import { ProcessManager } from "../src/processManager";
import { SettingsStore } from "../src/settings";
import { argValue, loadSettings, moeCaps } from "./helpers";

const os = require("node:os") as typeof import("node:os");

async function freePort(): Promise<number> {
  const server = net.createServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as net.AddressInfo).port;
  await new Promise<void>((resolve) => server.close(() => resolve()));
  return port;
}

for (const modern of [false, true]) {
  it(`launch and Copy command share optional feature guards (${modern ? "modern" : "old"} binary)`, async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-performance-launch-"));
    // Redirect only this test process's data paths; never touch the real user configuration.
    t.mock.method(os, "homedir", () => dir);
    const model = path.join(dir, "model.gguf");
    fs.writeFileSync(model, "test model");
    const binary = path.join(dir, "llama-server.cjs");
    const argvFile = path.join(dir, "argv.json");
    const help = modern ? `--moe-cache-mib N\n--spec-draft-sampling {greedy,probabilistic}\n--spec-draft-n-ubatch N` : "--ctx-size N";
    fs.writeFileSync(binary, `
const fs = require('fs');
const args = process.argv.slice(2);
if (args.includes('--help')) { console.log(${JSON.stringify(help)}); process.exit(0); }
if (args.includes('--list-devices')) { console.log('CUDA0: Test GPU (24576 MiB, 22000 MiB free)'); process.exit(0); }
fs.writeFileSync(${JSON.stringify(argvFile)}, JSON.stringify(args));
require('http').createServer((_req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify({ status: 'ok', default_generation_settings: { n_ctx: 8192 } }));
}).listen(Number(args[args.indexOf('--port') + 1]), '127.0.0.1');
`);
    const config = new ConfigFile(path.join(dir, "config.json"));
    const store = new SettingsStore(config);
    await config.setPreferences({ backend: "path", launchMode: "background", port: await freePort() });
    await store.setState({ selectedModelPath: model, modelCapabilities: moeCaps({ path: model, nextnPredictLayers: 1 }),
      loadSettings: loadSettings({ nCpuMoe: 48, moeCacheMiB: 4096, speculativeMode: "mtp", draftBatchSize: 64 }) });
    const manager = new ProcessManager(store);
    manager.resolveLaunch = () => ({ binary, command: process.execPath, prefixArgs: [binary], env: { ...process.env }, method: "path" });
    manager.isCpuBackend = () => false;
    t.after(async () => {
      await manager.stop(true);
      config.dispose();
      fs.rmSync(dir, { recursive: true, force: true });
    });
    const copied = manager.describeCommandLine();
    const status = await manager.start();
    assert.equal(status.running, true);
    const args = JSON.parse(fs.readFileSync(argvFile, "utf8")) as string[];
    for (const [flag, value] of [["--moe-cache-mib", "4096"], ["--spec-draft-sampling", "probabilistic"], ["--spec-draft-n-ubatch", "64"]]) {
      assert.equal(argValue(args, flag), modern ? value : undefined);
      assert.equal(copied.includes(flag), modern);
    }
    assert.equal(store.getState().loadSettings.moeCacheMiB, 4096, "unsupported settings remain saved");
    assert.equal(manager.getStatus().configDirty, false);
  });
}

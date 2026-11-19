import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as net from "node:net";
import * as path from "node:path";
import { it } from "node:test";
import { ConfigFile } from "../src/config";
import { ProcessManager } from "../src/processManager";
import { getLockPath } from "../src/paths";
import { serverConfigFingerprint } from "../src/serverArgs";
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
    const help = modern ? `--moe-cache-mib N\n--moe-cache-cpu-share SHARE\n--cache-type-k f16, q8_0, f8_e4m3\n--spec-draft-sampling {greedy,probabilistic}\n--spec-draft-n-ubatch N` : "--ctx-size N";
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
    await store.updateLoadSettings({ cacheTypeV: "f8_e4m3" });
    assert.equal(manager.getStatus().configDirty, false, "unsupported FP8 on V falls back to the running q8_0");
    await store.updateLoadSettings({ cacheTypeK: "f8_e4m3" });
    assert.equal(manager.getStatus().configDirty, modern, "FP8 K is reload-relevant only when supported");
    await store.updateLoadSettings({ cacheTypeK: "q8_0", moeCacheCpuPercent: 35 });
    assert.equal(manager.getStatus().configDirty, modern, "only an active CPU share changes the launch");
    assert.equal(store.getState().loadSettings.moeCacheCpuPercent, 35);

    // An already running server may have a lock from before effective cache
    // values were fingerprinted. Its argv is the source of the applied values.
    const lockPath = getLockPath();
    const lock = JSON.parse(fs.readFileSync(lockPath, "utf8"));
    delete lock.serverFeatures;
    lock.loadSettings.cacheTypeV = "f8_e4m3";
    lock.configFingerprint = serverConfigFingerprint(model, lock.loadSettings, lock.launchMode);
    fs.writeFileSync(lockPath, JSON.stringify(lock));
    await store.updateLoadSettings({ moeCacheCpuPercent: 0 });
    assert.equal(manager.getStatus().configDirty, false);
    assert.equal(manager.getLaunchedConfig()?.cacheSettings?.cacheTypeV, "q8_0");
  });
}

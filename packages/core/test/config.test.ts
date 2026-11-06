import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, it } from "node:test";
import { ConfigFile } from "../src/config";
import { SettingsStore } from "../src/settings";

describe("ConfigFile", () => {
  it("serializes real processes across the entire read/merge/write", { timeout: 8000 }, async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-config-writers-"));
    const file = path.join(dir, "config.json");
    const release = path.join(dir, "release");
    const config = new ConfigFile(file);
    await config.ensureExists();
    const script = `
      const fs = require('fs');
      const { ConfigFile } = require(process.argv[1]);
      const config = new ConfigFile(process.argv[2]);
      const role = process.argv[3];
      process.send('started');
      config.updatePreferences(current => {
        fs.writeFileSync(process.argv[4] + '.' + role, 'entered');
        if (role === 'first') {
          process.send('holding');
          const until = Date.now() + 4000;
          while (!fs.existsSync(process.argv[4])) {
            if (Date.now() > until) throw new Error('release timed out');
            Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 5);
          }
          return { port: 9001 };
        }
        if (current.port !== 9001) throw new Error('read an obsolete snapshot');
        return { host: '0.0.0.0' };
      }).then(() => { config.dispose(); process.disconnect(); }, err => { console.error(err); process.exit(1); });
    `;
    const launch = (role: string) => {
      const child = spawn(process.execPath, ["-e", script, require.resolve("../src/config"), file, role, release], {
        stdio: ["ignore", "ignore", "pipe", "ipc"],
      });
      let errors = "";
      child.stderr?.on("data", (part) => { errors += part; });
      const messages: string[] = [];
      child.on("message", (msg) => messages.push(String(msg)));
      const done = new Promise<void>((resolve, reject) => {
        child.on("error", reject);
        child.on("exit", (code) => code === 0 ? resolve() : reject(new Error(errors || `writer exited ${code}`)));
      });
      done.catch(() => undefined);
      t.after(() => child.kill());
      return { messages, done };
    };
    t.after(() => { config.dispose(); fs.rmSync(dir, { recursive: true, force: true }); });
    const waitFor = async (check: () => boolean) => {
      const until = Date.now() + 3000;
      while (!check()) {
        assert.ok(Date.now() < until, "worker did not reach its barrier");
        await new Promise((resolve) => setTimeout(resolve, 5));
      }
    };
    const first = launch("first");
    await waitFor(() => first.messages.includes("holding"));
    const second = launch("second");
    await waitFor(() => second.messages.includes("started"));
    await new Promise((resolve) => setTimeout(resolve, 50));
    assert.equal(fs.existsSync(release + ".second"), false, "second writer waits outside the transaction");
    fs.writeFileSync(release, "continue");
    await Promise.all([first.done, second.done]);
    const saved = JSON.parse(fs.readFileSync(file, "utf8"));
    assert.equal(saved.app.port, 9001);
    assert.equal(saved.app.host, "0.0.0.0");
    assert.equal(fs.existsSync(file + ".lock"), false);
  });

  it("merges concurrent load/request/server patches using their latest snapshot", async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-config-patches-"));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const file = path.join(dir, "config.json");
    const a = new SettingsStore(new ConfigFile(file));
    const b = new SettingsStore(new ConfigFile(file));
    await Promise.all([
      a.updateLoadSettings({ contextLength: 8192 }), b.updateLoadSettings({ gpuOffload: 7 }),
      a.updateRequestSettings({ temperature: 0.2 }), b.updateRequestSettings({ topP: 0.7 }),
      a.setRemoteServer("https://example.invalid", "test"), b.confirmPlainHttp("http://example.invalid"),
    ]);
    const saved = new ConfigFile(file).get();
    assert.equal(saved.state.loadSettings.contextLength, 8192);
    assert.equal(saved.state.loadSettings.gpuOffload, 7);
    assert.equal(saved.state.requestSettings.temperature, 0.2);
    assert.equal(saved.state.requestSettings.topP, 0.7);
    assert.equal(saved.app.server.remoteUrl, "https://example.invalid");
    assert.deepEqual(saved.app.server.plainHttpConfirmed, ["http://example.invalid"]);
  });

  it("re-points a saved main GPU and split from the pre-fix GPU order once", async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-config-gpuorder-"));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const file = path.join(dir, "config.json");
    const store = new SettingsStore(new ConfigFile(file));
    await store.updateLoadSettings({ mainGpu: 0, tensorSplit: "90,10", splitMode: "none" });
    const gpus = [
      { totalBytes: 16e9, source: "sysfs:card0", llamaDeviceId: "Vulkan0", legacyIndex: 1 },
      { totalBytes: 16e9, source: "sysfs:card1", llamaDeviceId: "Vulkan1", legacyIndex: 0 },
    ];
    assert.equal(await store.migrateLegacyGpuOrder(gpus), true);
    let load = new SettingsStore(new ConfigFile(file)).getState().loadSettings;
    assert.equal(load.mainGpu, 1);
    assert.equal(load.tensorSplit, "10,90");
    // The user's later choice in the new order is never remapped again.
    await store.updateLoadSettings({ mainGpu: 0 });
    assert.equal(await new SettingsStore(new ConfigFile(file)).migrateLegacyGpuOrder(gpus), false);
    load = new SettingsStore(new ConfigFile(file)).getState().loadSettings;
    assert.equal(load.mainGpu, 0);
    assert.equal(new ConfigFile(file).get().state.gpuOrderRevision, 1);
  });

  it("does not swap the cards back when an older version drops the migration flag", async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-config-gpuorder-old-"));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const file = path.join(dir, "config.json");
    const store = new SettingsStore(new ConfigFile(file));
    await store.updateLoadSettings({ mainGpu: 0, tensorSplit: "90,10", splitMode: "none" });
    const gpus = [
      { totalBytes: 16e9, source: "sysfs:card1", llamaDeviceId: "Vulkan0", legacyIndex: 1 },
      { totalBytes: 16e9, source: "sysfs:card0", llamaDeviceId: "Vulkan1", legacyIndex: 0 },
    ];
    assert.equal(await store.migrateLegacyGpuOrder(gpus), true);
    assert.ok(fs.existsSync(path.join(dir, "gpu-order-v1")));
    // Llama AIO 0.3.9 rewrites config.json with the fields it knows (seen live).
    const raw = JSON.parse(fs.readFileSync(file, "utf8"));
    delete raw.state.gpuOrderRevision;
    fs.writeFileSync(file, JSON.stringify(raw));
    const fresh = new SettingsStore(new ConfigFile(file));
    assert.equal(fresh.getState().loadSettings.mainGpu, 1);
    assert.equal(await fresh.migrateLegacyGpuOrder(gpus), false, "a second swap would undo the first");
    const load = new SettingsStore(new ConfigFile(file)).getState().loadSettings;
    assert.equal(load.mainGpu, 1);
    assert.equal(load.tensorSplit, "10,90");
    assert.equal(new ConfigFile(file).get().state.gpuOrderRevision, 1, "flag restored from the marker");
  });

  it("starts a fresh two-GPU config on the stronger card, once", async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-config-maingpu-"));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const file = path.join(dir, "config.json");
    const store = new SettingsStore(new ConfigFile(file));
    const gpus = [
      { totalBytes: 16e9, name: "AMD Radeon RX 9060 XT", source: "sysfs:card0", llamaDeviceId: "Vulkan0" },
      { totalBytes: 16e9, name: "AMD Radeon RX 9070 XT", source: "sysfs:card1", llamaDeviceId: "Vulkan1" },
    ];
    // No llama.cpp ids yet (runtime not installed): the order is not final, wait.
    assert.equal(await store.migrateLegacyGpuOrder(gpus.map((g) => ({ ...g, llamaDeviceId: undefined }))), false);
    assert.equal(new ConfigFile(file).get().state.gpuOrderRevision, undefined);
    assert.equal(await store.migrateLegacyGpuOrder(gpus), true);
    assert.equal(store.getState().loadSettings.mainGpu, 1);
    await store.updateLoadSettings({ mainGpu: 0 });
    assert.equal(await store.migrateLegacyGpuOrder(gpus), false, "a later choice is the user's");
    assert.equal(store.getState().loadSettings.mainGpu, 0);
  });

  it("keeps a saved split and main GPU when the order did not change", async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-config-keepsplit-"));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const file = path.join(dir, "config.json");
    const store = new SettingsStore(new ConfigFile(file));
    await store.updateLoadSettings({ mainGpu: 0, tensorSplit: "60,40" });
    const gpus = [
      { totalBytes: 16e9, name: "AMD Radeon RX 9060 XT", source: "sysfs:card0", llamaDeviceId: "Vulkan0" },
      { totalBytes: 16e9, name: "AMD Radeon RX 9070 XT", source: "sysfs:card1", llamaDeviceId: "Vulkan1" },
    ];
    assert.equal(await store.migrateLegacyGpuOrder(gpus), false);
    assert.equal(store.getState().loadSettings.mainGpu, 0);
    assert.equal(store.getState().loadSettings.tensorSplit, "60,40");
  });

  it("recovers an abandoned lock and releases a lock after a failed write", async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-config-lock-"));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const file = path.join(dir, "config.json");
    fs.mkdirSync(file + ".lock");
    const stale = new Date(Date.now() - 30_000);
    fs.utimesSync(file + ".lock", stale, stale);
    const config = new ConfigFile(file);
    await config.setPreferences({ port: 9002 });
    const rename = t.mock.method(require("fs"), "renameSync", () => { throw new Error("write failed"); });
    await assert.rejects(config.setPreferences({ port: 9003 }), /write failed/);
    assert.equal(config.getPreferences().port, 9002, "failed writes do not update the cache");
    assert.equal(fs.existsSync(file + ".lock"), false);
    rename.mock.restore();
    await config.setPreferences({ port: 9004 });
    assert.equal(config.getPreferences().port, 9004);
  });

  it("announces a peer's edit that a local write absorbed, and keeps it", async () => {
    const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-config-")), "config.json");
    const config = new ConfigFile(file);
    await config.setPreferences({ port: 8742 } as never);

    // The TUI writes a new port; this window writes something else before its watcher fires.
    const peer = JSON.parse(fs.readFileSync(file, "utf8"));
    peer.app.port = 9000;
    fs.writeFileSync(file, JSON.stringify(peer, null, 2));

    const external: number[] = [];
    config.onDidChangeExternally((c) => external.push(c.app.port as number));
    await config.setPreferences({ launchMode: "background" } as never);

    assert.deepEqual(external, [9000], "the absorbed peer edit is announced once");
    assert.equal(config.getPreferences().port, 9000, "peer edit kept");
    assert.equal(config.getPreferences().launchMode, "background", "local edit applied");

    // A plain local write with nothing new on disk is not external.
    await config.setPreferences({ launchMode: "externalTerminal" } as never);
    assert.equal(external.length, 1);
    config.dispose();
  });
});

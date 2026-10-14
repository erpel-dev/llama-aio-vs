import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as net from "node:net";
import * as os from "node:os";
import * as path from "node:path";
import { after, before, describe, it } from "node:test";

// Each test file runs in its own process: point the shared data root
// (~/.llama-aio-vs) at a temp home before core reads it.
const home = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-home-"));
const bin = path.join(home, "bin");
process.env.HOME = home;
process.env.USERPROFILE = home;

import { ConfigFile } from "../src/config";
import { isPidAlive } from "../src/processIdentity";
import { LaunchCancelledError, ProcessManager, readLogTail } from "../src/processManager";
import { SettingsStore } from "../src/settings";

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const port = (srv.address() as net.AddressInfo).port;
      srv.close(() => resolve(port));
    });
  });
}

describe("ProcessManager stop during start", { skip: process.platform === "win32" }, () => {
  let pm: ProcessManager;
  let store: SettingsStore;
  const pidFile = path.join(home, "fake-server.pid");

  before(async () => {
    // A "llama-server" that never answers HTTP: start keeps waiting until stopped.
    fs.mkdirSync(bin, { recursive: true });
    const fake = path.join(bin, "llama-server");
    fs.writeFileSync(
      fake,
      [
        "#!/bin/sh",
        'case "$1" in --version|--list-devices) echo "version: 1 (fake)"; exit 0;; esac',
        `echo $$ > "${pidFile}"`,
        'echo "loading model"',
        "exec sleep 60",
        "",
      ].join("\n"),
      { mode: 0o755 }
    );
    process.env.PATH = `${bin}${path.delimiter}${process.env.PATH || ""}`;
    const model = path.join(home, "model.gguf");
    fs.writeFileSync(model, "not really a gguf");

    const config = new ConfigFile();
    await config.setPreferences({ backend: "path", launchMode: "background", port: await freePort() } as never);
    store = new SettingsStore(config);
    await store.setState({ selectedModelPath: model });
    pm = new ProcessManager(store);
  });

  after(() => {
    fs.rmSync(home, { recursive: true, force: true });
  });

  it("ends the start with LaunchCancelledError and kills the child", async () => {
    const starting = pm.start();
    const outcome = starting.then(
      () => "resolved",
      (e: unknown) => e
    );
    // Wait until the fake server is running, then press Stop.
    for (let i = 0; i < 100 && !fs.existsSync(pidFile); i++) {
      await new Promise((r) => setTimeout(r, 50));
    }
    assert.ok(fs.existsSync(pidFile), "fake llama-server was spawned");
    const pid = Number(fs.readFileSync(pidFile, "utf8").trim());
    await pm.stop(true);
    const result = await outcome;
    assert.ok(result instanceof LaunchCancelledError, `expected cancel, got ${String(result)}`);
    assert.equal(isPidAlive(pid), false);
    assert.equal(pm.getStatus().starting, false, "launch lock released");
  });

  it("a later start is not affected by the earlier cancel", async () => {
    fs.rmSync(pidFile, { force: true });
    const starting = pm.start().catch((e: unknown) => e);
    for (let i = 0; i < 100 && !fs.existsSync(pidFile); i++) {
      await new Promise((r) => setTimeout(r, 50));
    }
    // Still waiting for HTTP, not cancelled by the previous stop.
    assert.equal(pm.getStatus().starting, true);
    await pm.stop(true);
    assert.ok((await starting) instanceof LaunchCancelledError);
  });
});

describe("readLogTail", () => {
  it("returns only the last bytes, starting at a full line", () => {
    const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-log-")), "server.log");
    const lines = Array.from({ length: 5000 }, (_, i) => `line ${i} ${"x".repeat(40)}`);
    fs.writeFileSync(file, lines.join("\n") + "\n");
    const tail = readLogTail(file, 1024);
    assert.ok(tail.length <= 1024);
    assert.match(tail, /^line \d+ x+\n/, "starts at a line boundary");
    assert.ok(tail.trimEnd().endsWith(`line 4999 ${"x".repeat(40)}`));
    assert.equal(readLogTail(path.join(path.dirname(file), "missing.log")), "");
  });
});

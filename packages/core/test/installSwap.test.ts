import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, it } from "node:test";
import {
  activeInstallLock,
  createBusyFsError,
  formatBusyInstallError,
  isBusyFsError,
  replaceInstallDirInPlace,
  swapInstallDir,
  withInstallLock,
} from "../src/installSwap";
import { isPidAlive } from "../src/processIdentity";

function writeFile(dir: string, name: string, body: string): void {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, name), body);
}

function readFile(dir: string, name: string): string {
  return fs.readFileSync(path.join(dir, name), "utf8");
}

describe("isBusyFsError", () => {
  it("recognizes the Windows rename codes", () => {
    assert.equal(isBusyFsError(createBusyFsError("busy", "EBUSY")), true);
    assert.equal(isBusyFsError(createBusyFsError("perm", "EPERM")), true);
    assert.equal(isBusyFsError(createBusyFsError("access", "EACCES")), true);
    assert.equal(isBusyFsError(Object.assign(new Error("nope"), { code: "ENOENT" })), false);
    assert.equal(isBusyFsError("string"), false);
  });
});

describe("formatBusyInstallError", () => {
  it("tells the user to stop llama-server and extra windows", () => {
    const msg = formatBusyInstallError(
      String.raw`C:\Users\les1si\.llama-aio-vs\llama.cpp\vulkan\bin`,
      createBusyFsError("EBUSY: resource busy or locked, rename 'bin' -> 'bin.old'")
    ).message;
    assert.match(msg, /still has it locked/);
    assert.match(msg, /Stop llama-server/);
    assert.match(msg, /extra VS Code/);
    assert.match(msg, /EBUSY/);
  });
});

describe("swapInstallDir", () => {
  it("replaces the live tree with the staged one", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-swap-"));
    try {
      const binDir = path.join(root, "bin");
      const stageDir = path.join(root, "bin.new-1");
      writeFile(binDir, "llama-server", "old");
      writeFile(binDir, "keep-me-not", "gone");
      writeFile(stageDir, "llama-server", "new");
      writeFile(stageDir, "ggml.dll", "dll");
      await swapInstallDir(stageDir, binDir, { retries: 0 });
      assert.equal(readFile(binDir, "llama-server"), "new");
      assert.equal(readFile(binDir, "ggml.dll"), "dll");
      assert.equal(fs.existsSync(path.join(binDir, "keep-me-not")), false);
      assert.equal(fs.existsSync(stageDir), false);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it("falls back to per-file replace after a busy directory swap", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-swap-"));
    try {
      const binDir = path.join(root, "bin");
      const stageDir = path.join(root, "bin.new-1");
      writeFile(binDir, "llama-server", "old");
      writeFile(stageDir, "llama-server", "new");
      let calls = 0;
      const renameSync = (from: string, to: string) => {
        calls += 1;
        if (calls === 2) {
          throw createBusyFsError(`EBUSY: rename '${from}' -> '${to}'`);
        }
        fs.renameSync(from, to);
      };
      await swapInstallDir(stageDir, binDir, { renameSync, retries: 0, retryDelayMs: 1 });
      // Fallback should still land the new files because file-level rename works.
      assert.equal(readFile(binDir, "llama-server"), "new");
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it("falls back to per-file replace when directory rename stays busy", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-swap-"));
    try {
      const binDir = path.join(root, "bin");
      const stageDir = path.join(root, "bin.new-1");
      writeFile(binDir, "llama-server", "old");
      writeFile(binDir, "ggml.dll", "old-dll");
      writeFile(stageDir, "llama-server", "new");
      writeFile(stageDir, "ggml.dll", "new-dll");
      const renameSync = (from: string, to: string) => {
        if (fs.statSync(from).isDirectory()) {
          throw createBusyFsError(`EBUSY: rename '${from}' -> '${to}'`);
        }
        fs.renameSync(from, to);
      };
      await swapInstallDir(stageDir, binDir, { renameSync, retries: 1, retryDelayMs: 1 });
      assert.equal(readFile(binDir, "llama-server"), "new");
      assert.equal(readFile(binDir, "ggml.dll"), "new-dll");
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});

describe("replaceInstallDirInPlace", () => {
  it("renames a busy target aside, then moves the new file in", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-inplace-"));
    try {
      const binDir = path.join(root, "bin");
      const stageDir = path.join(root, "bin.new-1");
      writeFile(binDir, "llama-server", "old");
      writeFile(stageDir, "llama-server", "new");
      const renameSync = (from: string, to: string) => {
        fs.renameSync(from, to);
      };
      replaceInstallDirInPlace(stageDir, binDir, { renameSync });
      assert.equal(readFile(binDir, "llama-server"), "new");
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});

describe("withInstallLock", () => {
  it("is reentrant for the same process", async () => {
    const lockPath = path.join(
      fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-lock-")),
      "install.lock.json"
    );
    try {
      const order: string[] = [];
      await withInstallLock(
        async () => {
          order.push("outer");
          await withInstallLock(async () => {
            order.push("inner");
            assert.ok(activeInstallLock(lockPath));
          }, { lockPath });
          order.push("after");
        },
        { lockPath }
      );
      assert.deepEqual(order, ["outer", "inner", "after"]);
      assert.equal(activeInstallLock(lockPath), undefined);
      assert.equal(fs.existsSync(lockPath), false);
    } finally {
      fs.rmSync(path.dirname(lockPath), { recursive: true, force: true });
    }
  });

  it("times out while another process holds the lock", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-lock-"));
    const lockPath = path.join(dir, "install.lock.json");
    const child = spawn("sleep", ["30"], { stdio: "ignore" });
    try {
      const pid = child.pid;
      assert.ok(pid);
      for (let i = 0; i < 20 && !isPidAlive(pid); i++) {
        await new Promise((r) => setTimeout(r, 10));
      }
      fs.writeFileSync(
        lockPath,
        JSON.stringify({ pid, startedAt: new Date().toISOString() })
      );
      await assert.rejects(
        () =>
          withInstallLock(async () => "nope", {
            lockPath,
            waitMs: 40,
            sleep: async () => undefined,
            now: (() => {
              let t = 0;
              return () => {
                t += 100;
                return t;
              };
            })(),
          }),
        /Another Llama AIO window is installing/
      );
    } finally {
      child.kill("SIGKILL");
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

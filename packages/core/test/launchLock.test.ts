import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, it } from "node:test";
import {
  acquireLaunchLock,
  activeLaunchLock,
  LAUNCH_LOCK_STALE_MS,
  LaunchLockHeldError,
  readLaunchLock,
} from "../src/launchLock";

function tmpLock(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-launch-"));
  return path.join(dir, "runtime", "launch.lock.json");
}

/** A pid that is guaranteed not to be running (or at least not ours). */
function deadPid(): number {
  return 2 ** 22 - 7;
}

describe("acquireLaunchLock", () => {
  it("creates the lock file and removes it on release", () => {
    const lockPath = tmpLock();
    const handle = acquireLaunchLock("start", { lockPath });
    const lock = readLaunchLock(lockPath);
    assert.equal(lock?.pid, process.pid);
    assert.equal(lock?.kind, "start");
    assert.ok(activeLaunchLock(lockPath));
    handle.release();
    assert.equal(fs.existsSync(lockPath), false);
    assert.equal(activeLaunchLock(lockPath), undefined);
  });

  it("refuses while a live foreign process holds it", () => {
    const lockPath = tmpLock();
    fs.mkdirSync(path.dirname(lockPath), { recursive: true });
    // Our parent is alive for the duration of the test and is not us.
    const holderPid = process.ppid;
    fs.writeFileSync(
      lockPath,
      JSON.stringify({ pid: holderPid, startedAt: new Date().toISOString(), kind: "reload" })
    );
    assert.throws(
      () => acquireLaunchLock("start", { lockPath }),
      (err: unknown) =>
        err instanceof LaunchLockHeldError &&
        err.holder?.pid === holderPid &&
        /another Llama AIO window/.test(err.message)
    );
    assert.equal(readLaunchLock(lockPath)?.pid, holderPid, "must not steal a live lock");
  });

  it("steals a lock left by a dead process", () => {
    const lockPath = tmpLock();
    fs.mkdirSync(path.dirname(lockPath), { recursive: true });
    fs.writeFileSync(
      lockPath,
      JSON.stringify({ pid: deadPid(), startedAt: new Date().toISOString(), kind: "start" })
    );
    assert.equal(activeLaunchLock(lockPath), undefined);
    const handle = acquireLaunchLock("start", { lockPath });
    assert.equal(readLaunchLock(lockPath)?.pid, process.pid);
    handle.release();
  });

  it("steals a lock that is older than the stale window even if the pid lives", () => {
    const lockPath = tmpLock();
    fs.mkdirSync(path.dirname(lockPath), { recursive: true });
    const old = new Date(Date.now() - LAUNCH_LOCK_STALE_MS - 1000).toISOString();
    fs.writeFileSync(
      lockPath,
      JSON.stringify({ pid: process.ppid, startedAt: old, kind: "start" })
    );
    const handle = acquireLaunchLock("reload", { lockPath });
    assert.equal(readLaunchLock(lockPath)?.pid, process.pid);
    handle.release();
  });

  it("replaces a leftover lock from this very pid", () => {
    const lockPath = tmpLock();
    fs.mkdirSync(path.dirname(lockPath), { recursive: true });
    fs.writeFileSync(
      lockPath,
      JSON.stringify({ pid: process.pid, startedAt: new Date().toISOString(), kind: "start" })
    );
    const handle = acquireLaunchLock("start", { lockPath });
    assert.equal(readLaunchLock(lockPath)?.kind, "start");
    handle.release();
    assert.equal(fs.existsSync(lockPath), false);
  });

  it("release does not remove a lock a later launcher took over", () => {
    const lockPath = tmpLock();
    const handle = acquireLaunchLock("start", { lockPath });
    // Simulate another window stealing the (e.g. stale) lock.
    fs.writeFileSync(
      lockPath,
      JSON.stringify({ pid: process.ppid, startedAt: new Date().toISOString(), kind: "reload" })
    );
    handle.release();
    assert.equal(readLaunchLock(lockPath)?.pid, process.ppid);
  });

  it("treats unreadable lock files as absent", () => {
    const lockPath = tmpLock();
    fs.mkdirSync(path.dirname(lockPath), { recursive: true });
    fs.writeFileSync(lockPath, "{not json");
    assert.equal(readLaunchLock(lockPath), undefined);
    const handle = acquireLaunchLock("start", { lockPath });
    assert.equal(readLaunchLock(lockPath)?.pid, process.pid);
    handle.release();
  });
});

/**
 * Cross-process lock around llama-server start/reload.
 *
 * `ProcessManager.claimLaunch` only serialises launches inside one extension
 * host. Two VS Code windows (or VS Code + the TUI) could both pass the
 * "not running" check, spawn two servers on the same port, truncate each
 * other's log and later kill the healthy one. This file lock makes the
 * check-and-spawn window exclusive across processes.
 */

import * as fs from "fs";
import * as path from "path";
import { getLockDir } from "./paths";
import { isPidAlive } from "./processIdentity";

/** A launch that is still marked in progress after this long is treated as abandoned. */
export const LAUNCH_LOCK_STALE_MS = 10 * 60 * 1000;

export interface LaunchLockFile {
  pid: number;
  startedAt: string;
  kind: "start" | "reload";
}

export interface LaunchLockHandle {
  readonly path: string;
  release(): void;
}

export class LaunchLockHeldError extends Error {
  constructor(readonly holder: LaunchLockFile | undefined, lockPath: string) {
    super(
      "A server start/reload is already in progress in another Llama AIO window" +
        (holder?.pid ? ` (pid ${holder.pid})` : "") +
        `. Wait for it to finish, or delete ${lockPath} if that window is gone.`
    );
    this.name = "LaunchLockHeldError";
  }
}

export function getLaunchLockPath(): string {
  return path.join(getLockDir(), "launch.lock.json");
}

export function readLaunchLock(lockPath = getLaunchLockPath()): LaunchLockFile | undefined {
  try {
    const parsed = JSON.parse(fs.readFileSync(lockPath, "utf8")) as LaunchLockFile;
    if (!parsed || !Number.isFinite(parsed.pid)) {
      return undefined;
    }
    return parsed;
  } catch {
    return undefined;
  }
}

function isStale(lock: LaunchLockFile, now: number): boolean {
  if (lock.pid !== process.pid && !isPidAlive(lock.pid)) {
    return true;
  }
  const started = Date.parse(lock.startedAt);
  return Number.isFinite(started) && now - started > LAUNCH_LOCK_STALE_MS;
}

/**
 * Launch lock currently held by a *live* process (this one included), or
 * undefined. Used by status readers to avoid treating a provisional server
 * lock as stale while another window is still booting the server.
 */
export function activeLaunchLock(
  lockPath = getLaunchLockPath(),
  now: () => number = Date.now
): LaunchLockFile | undefined {
  const lock = readLaunchLock(lockPath);
  if (!lock || isStale(lock, now())) {
    return undefined;
  }
  return lock;
}

/**
 * Acquire the launch lock or throw {@link LaunchLockHeldError}. Locks left by
 * dead processes (or older than {@link LAUNCH_LOCK_STALE_MS}) are stolen.
 */
export function acquireLaunchLock(
  kind: "start" | "reload",
  options?: { lockPath?: string; now?: () => number }
): LaunchLockHandle {
  const lockPath = options?.lockPath || getLaunchLockPath();
  const now = options?.now ?? Date.now;
  fs.mkdirSync(path.dirname(lockPath), { recursive: true });
  const payload: LaunchLockFile = {
    pid: process.pid,
    startedAt: new Date(now()).toISOString(),
    kind,
  };

  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      fs.writeFileSync(lockPath, `${JSON.stringify(payload, null, 2)}\n`, {
        encoding: "utf8",
        flag: "wx",
      });
      return {
        path: lockPath,
        release: () => {
          const current = readLaunchLock(lockPath);
          // Only remove our own lock — never one a later launcher took over.
          if (!current || (current.pid === process.pid && current.startedAt === payload.startedAt)) {
            try {
              fs.unlinkSync(lockPath);
            } catch {
              // ignore
            }
          }
        },
      };
    } catch (err) {
      if ((err as NodeJS.ErrnoException)?.code !== "EEXIST") {
        throw err;
      }
      const holder = readLaunchLock(lockPath);
      if (holder && !isStale(holder, now()) && holder.pid !== process.pid) {
        throw new LaunchLockHeldError(holder, lockPath);
      }
      // Stale, unreadable, or left behind by this very pid (ProcessManager
      // already serialises launches in-process): replace it.
      try {
        fs.unlinkSync(lockPath);
      } catch {
        // ignore — retry will surface EEXIST again if someone else won
      }
    }
  }
  throw new LaunchLockHeldError(readLaunchLock(lockPath), lockPath);
}

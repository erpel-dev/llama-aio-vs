/**
 * Atomic-ish replacement of a llama.cpp `bin` tree.
 *
 * Windows cannot rename a directory while any file inside it is mapped
 * (running llama-server, a `--version` probe from another VS Code window,
 * Defender). Unix can. We try a directory rename first, retry on busy
 * errors, then fall back to renaming individual files — that *does* work
 * on Windows even for a running .exe.
 */

import * as fs from "fs";
import * as path from "path";
import { ensureDirs, getInstallLockPath } from "./paths";
import { isPidAlive } from "./processIdentity";

export const INSTALL_LOCK_STALE_MS = 2 * 60 * 60 * 1000;
export const INSTALL_LOCK_WAIT_MS = 30 * 60 * 1000;

export interface InstallLock {
  pid: number;
  startedAt: string;
  binDir?: string;
}

export interface ProgressSink {
  report(value: { message?: string; increment?: number }): void;
}

export type RenameFn = (from: string, to: string) => void;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function isBusyFsError(err: unknown): boolean {
  const code = (err as NodeJS.ErrnoException)?.code;
  return code === "EBUSY" || code === "EPERM" || code === "EACCES";
}

export function formatBusyInstallError(binDir: string, err: unknown): Error {
  const detail = err instanceof Error ? err.message : String(err);
  return new Error(
    `Could not replace ${binDir} because another process still has it locked.\n` +
      `Stop llama-server, close extra VS Code / Llama AIO windows, then retry.\n` +
      `(${detail})`
  );
}

function busyError(code: string, message: string): NodeJS.ErrnoException {
  const err = new Error(message) as NodeJS.ErrnoException;
  err.code = code;
  return err;
}

export function readInstallLock(lockPath = getInstallLockPath()): InstallLock | undefined {
  try {
    if (!fs.existsSync(lockPath)) {
      return undefined;
    }
    const parsed = JSON.parse(fs.readFileSync(lockPath, "utf8")) as InstallLock;
    if (!parsed || !Number.isFinite(parsed.pid)) {
      return undefined;
    }
    return parsed;
  } catch {
    return undefined;
  }
}

/** Live lock held by this or another process. Dead / ancient files are ignored. */
export function activeInstallLock(lockPath = getInstallLockPath()): InstallLock | undefined {
  const lock = readInstallLock(lockPath);
  if (!lock) {
    return undefined;
  }
  if (lock.pid === process.pid) {
    return lock;
  }
  if (!isPidAlive(lock.pid)) {
    return undefined;
  }
  const started = Date.parse(lock.startedAt);
  if (Number.isFinite(started) && Date.now() - started > INSTALL_LOCK_STALE_MS) {
    return undefined;
  }
  return lock;
}

function writeInstallLock(lockPath: string, lock: InstallLock): void {
  ensureDirs(path.dirname(lockPath));
  fs.writeFileSync(lockPath, `${JSON.stringify(lock, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
}

function removeInstallLock(lockPath: string): void {
  try {
    fs.unlinkSync(lockPath);
  } catch {
    // ignore
  }
}

function stealStaleLock(lockPath: string): void {
  const lock = readInstallLock(lockPath);
  if (!lock) {
    removeInstallLock(lockPath);
    return;
  }
  const started = Date.parse(lock.startedAt);
  const staleAge = Number.isFinite(started) && Date.now() - started > INSTALL_LOCK_STALE_MS;
  if (lock.pid === process.pid || !isPidAlive(lock.pid) || staleAge) {
    removeInstallLock(lockPath);
  }
}

/** Depth so installByTag → installFromArchive does not deadlock on the same pid. */
let lockDepth = 0;

/**
 * Exclusive cross-window lock around llama.cpp install/replace.
 * Other windows skip `--version` probes and refuse to start llama-server
 * while this file exists.
 */
export async function withInstallLock<T>(
  fn: () => Promise<T>,
  options?: {
    binDir?: string;
    progress?: ProgressSink;
    waitMs?: number;
    lockPath?: string;
    now?: () => number;
    sleep?: (ms: number) => Promise<void>;
  }
): Promise<T> {
  if (lockDepth > 0) {
    lockDepth += 1;
    try {
      return await fn();
    } finally {
      lockDepth -= 1;
    }
  }

  const lockPath = options?.lockPath || getInstallLockPath();
  const waitMs = options?.waitMs ?? INSTALL_LOCK_WAIT_MS;
  const now = options?.now ?? Date.now;
  const wait = options?.sleep ?? sleep;
  const deadline = now() + waitMs;
  let acquired = false;

  while (!acquired) {
    stealStaleLock(lockPath);
    try {
      writeInstallLock(lockPath, {
        pid: process.pid,
        startedAt: new Date(now()).toISOString(),
        binDir: options?.binDir,
      });
      acquired = true;
    } catch (err) {
      const code = (err as NodeJS.ErrnoException)?.code;
      if (code !== "EEXIST") {
        throw err;
      }
      if (now() >= deadline) {
        const holder = readInstallLock(lockPath);
        throw new Error(
          `Another Llama AIO window is installing llama.cpp` +
            (holder?.pid ? ` (pid ${holder.pid})` : "") +
            `. Wait for it to finish, or delete ${lockPath} if that window is gone.`
        );
      }
      const holder = activeInstallLock(lockPath);
      options?.progress?.report({
        message: holder?.pid
          ? `Waiting for install in another window (pid ${holder.pid})…`
          : "Waiting for another Llama AIO install to finish…",
      });
      await wait(250);
    }
  }

  lockDepth = 1;
  try {
    return await fn();
  } finally {
    lockDepth = 0;
    removeInstallLock(lockPath);
  }
}

function renameSwap(
  stageDir: string,
  binDir: string,
  rename: RenameFn,
  now: () => number
): void {
  const backupDir = `${binDir}.old-${now()}`;
  const hadPrevious = fs.existsSync(binDir);
  if (hadPrevious) {
    rename(binDir, backupDir);
  }
  try {
    rename(stageDir, binDir);
  } catch (err) {
    if (hadPrevious && fs.existsSync(backupDir) && !fs.existsSync(binDir)) {
      try {
        rename(backupDir, binDir);
      } catch {
        // Keep the backup; the caller may retry or fall back to per-file replace.
      }
    }
    throw err;
  }
  if (hadPrevious) {
    try {
      fs.rmSync(backupDir, { recursive: true, force: true });
    } catch {
      // Mapped leftover backup is harmless; cleaned up on the next successful swap.
    }
  }
}

function listFiles(dir: string): string[] {
  try {
    return fs.readdirSync(dir).filter((name) => {
      try {
        return fs.statSync(path.join(dir, name)).isFile();
      } catch {
        return false;
      }
    });
  } catch {
    return [];
  }
}

/**
 * Move each staged file into `binDir`, renaming locked targets aside.
 * Windows allows renaming a mapped .exe/.dll even when overwrite/unlink fails.
 */
export function replaceInstallDirInPlace(
  stageDir: string,
  binDir: string,
  options?: { renameSync?: RenameFn; now?: () => number }
): void {
  const rename = options?.renameSync ?? fs.renameSync;
  const now = options?.now ?? Date.now;
  ensureDirs(binDir);
  const leftovers: string[] = [];

  for (const name of listFiles(stageDir)) {
    const src = path.join(stageDir, name);
    const dest = path.join(binDir, name);
    if (fs.existsSync(dest)) {
      try {
        fs.unlinkSync(dest);
      } catch (err) {
        if (!isBusyFsError(err)) {
          throw err;
        }
        const backup = `${dest}.old-${now()}`;
        rename(dest, backup);
        leftovers.push(backup);
      }
    }
    try {
      rename(src, dest);
    } catch (err) {
      if (isBusyFsError(err) || (err as NodeJS.ErrnoException)?.code === "EXDEV") {
        fs.copyFileSync(src, dest);
        try {
          fs.unlinkSync(src);
        } catch {
          // ignore
        }
      } else {
        throw err;
      }
    }
  }

  try {
    fs.rmSync(stageDir, { recursive: true, force: true });
  } catch {
    // ignore
  }
  for (const leftover of leftovers) {
    try {
      fs.unlinkSync(leftover);
    } catch {
      // Still mapped; Windows will free it after the last handle closes.
    }
  }
}

function cleanupSiblingTrees(binDir: string): void {
  const parent = path.dirname(binDir);
  const base = path.basename(binDir);
  let names: string[];
  try {
    names = fs.readdirSync(parent);
  } catch {
    return;
  }
  for (const name of names) {
    if (!name.startsWith(`${base}.old-`) && !name.startsWith(`${base}.new-`)) {
      continue;
    }
    try {
      fs.rmSync(path.join(parent, name), { recursive: true, force: true });
    } catch {
      // ignore
    }
  }
}

/**
 * Put a fully staged `bin.new-*` tree at `binDir`.
 * Directory rename first (fast, atomic); per-file replace if Windows says EBUSY.
 */
export async function swapInstallDir(
  stageDir: string,
  binDir: string,
  options?: {
    renameSync?: RenameFn;
    retries?: number;
    retryDelayMs?: number;
    now?: () => number;
    sleep?: (ms: number) => Promise<void>;
    progress?: ProgressSink;
  }
): Promise<void> {
  const rename = options?.renameSync ?? fs.renameSync;
  const retries = options?.retries ?? 8;
  const delay = options?.retryDelayMs ?? 150;
  const now = options?.now ?? Date.now;
  const wait = options?.sleep ?? sleep;

  if (!fs.existsSync(stageDir)) {
    throw new Error(`Staged install missing: ${stageDir}`);
  }

  let lastErr: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      renameSwap(stageDir, binDir, rename, now);
      cleanupSiblingTrees(binDir);
      return;
    } catch (err) {
      lastErr = err;
      if (!isBusyFsError(err) || attempt === retries) {
        break;
      }
      options?.progress?.report({
        message: `Waiting for ${path.basename(binDir)} to be unlocked (${attempt + 1}/${retries})…`,
      });
      await wait(delay * 2 ** Math.min(attempt, 4));
    }
  }

  if (!isBusyFsError(lastErr)) {
    throw lastErr;
  }

  options?.progress?.report({
    message: `Replacing files in ${path.basename(path.dirname(binDir))}/${path.basename(binDir)} while some are still in use…`,
  });
  try {
    replaceInstallDirInPlace(stageDir, binDir, { renameSync: rename, now });
    cleanupSiblingTrees(binDir);
  } catch (err) {
    throw isBusyFsError(err) ? formatBusyInstallError(binDir, err) : err;
  }
}

/** Used by tests that need a synthetic busy error. */
export function createBusyFsError(message: string, code = "EBUSY"): NodeJS.ErrnoException {
  return busyError(code, message);
}

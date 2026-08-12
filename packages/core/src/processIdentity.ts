/**
 * Deciding whether a process is *our* llama-server.
 *
 * A port number is not proof of ownership — killing or adopting whatever
 * answers on the configured port can hit an unrelated dev server. Kept free of
 * `vscode` imports so it can be unit tested outside the extension host.
 */

import { execFileSync } from "child_process";
import * as fs from "fs";
import * as path from "path";

/** Command line of a running process, lowercased, or undefined when unknown. */
export function processCommandLine(pid: number): string | undefined {
  try {
    if (process.platform === "linux") {
      return fs.readFileSync(`/proc/${pid}/cmdline`, "utf8").replace(/\0/g, " ").toLowerCase();
    }
    if (process.platform === "darwin") {
      return execFileSync("ps", ["-p", String(pid), "-o", "command="], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
        timeout: 5000,
      }).toLowerCase();
    }
    if (process.platform === "win32") {
      return execFileSync("tasklist", ["/FI", `PID eq ${pid}`, "/FO", "CSV", "/NH"], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
        windowsHide: true,
        timeout: 8000,
      }).toLowerCase();
    }
  } catch {
    // Process gone, or no permission to inspect it.
  }
  return undefined;
}

/**
 * Whether a PID looks like a llama-server we may stop or adopt.
 * Unknown (unreadable) processes are treated as foreign and left alone.
 */
export function isLlamaServerProcess(pid: number): boolean {
  return looksLikeLlamaServer(processCommandLine(pid));
}

/** Split out for testing: the command-line check itself. */
export function looksLikeLlamaServer(commandLine: string | undefined): boolean {
  return !!commandLine && commandLine.toLowerCase().includes("llama-server");
}

/** Same file on disk, tolerating separator and case differences per platform. */
export function sameModelFile(a: string | undefined, b: string | undefined): boolean {
  if (!a || !b) {
    return false;
  }
  const norm = (p: string) => {
    const resolved = path.resolve(p);
    return process.platform === "win32" ? resolved.toLowerCase() : resolved;
  };
  return norm(a) === norm(b);
}

/** Parse PID strings from tool output, dropping anything unusable. */
export function uniquePids(raw: Array<string | undefined>): number[] {
  const pids = new Set<number>();
  for (const value of raw) {
    const pid = Number(value);
    if (Number.isFinite(pid) && pid > 0) {
      pids.add(pid);
    }
  }
  return [...pids];
}

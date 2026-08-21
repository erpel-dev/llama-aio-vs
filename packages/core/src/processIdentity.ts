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

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** True when `pid` still exists (including as a zombie until reaped). */
export function isPidAlive(pid: number): boolean {
  if (!Number.isFinite(pid) || pid <= 0) {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

/** Same folder on disk, tolerating separator and case differences per platform. */
export function normalizePathForCompare(p: string): string {
  const resolved = path.resolve(p);
  return process.platform === "win32" ? resolved.toLowerCase() : resolved;
}

/** True when `exe` is `dir` itself or a file/subdir inside it. */
export function executableIsUnderDir(exe: string | undefined, dir: string): boolean {
  if (!exe || !dir) {
    return false;
  }
  const e = normalizePathForCompare(exe);
  const d = normalizePathForCompare(dir);
  return e === d || e.startsWith(d.endsWith(path.sep) ? d : d + path.sep);
}

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
  return normalizePathForCompare(a) === normalizePathForCompare(b);
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

/** Absolute executable path of a running process, when the OS will tell us. */
export function processExecutablePath(pid: number): string | undefined {
  try {
    if (process.platform === "linux") {
      return fs.readlinkSync(`/proc/${pid}/exe`);
    }
    if (process.platform === "darwin") {
      const out = execFileSync("ps", ["-p", String(pid), "-o", "command="], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
        timeout: 5000,
      }).trim();
      const token = out.split(/\s+/).find((part) => part.includes("/") || part.startsWith("."));
      return token || undefined;
    }
    if (process.platform === "win32") {
      return windowsExecutablePath(pid);
    }
  } catch {
    // Process gone, or no permission to inspect it.
  }
  return undefined;
}

function windowsExecutablePath(pid: number): string | undefined {
  try {
    const out = execFileSync(
      "wmic",
      ["process", "where", `ProcessId=${pid}`, "get", "ExecutablePath", "/VALUE"],
      {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
        windowsHide: true,
        timeout: 8000,
      }
    );
    const line = out.split(/\r?\n/).find((l) => /^ExecutablePath=/i.test(l));
    const value = line?.slice(line.indexOf("=") + 1).trim();
    if (value) {
      return value;
    }
  } catch {
    // wmic is missing on some Windows 11 installs.
  }
  try {
    const out = execFileSync(
      "powershell.exe",
      [
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        `(Get-CimInstance Win32_Process -Filter "ProcessId=${pid}").ExecutablePath`,
      ],
      {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
        windowsHide: true,
        timeout: 8000,
      }
    );
    const value = out.trim();
    return value || undefined;
  } catch {
    return undefined;
  }
}

function listCandidateLlamaPids(): number[] {
  try {
    if (process.platform === "linux") {
      const pids: number[] = [];
      for (const name of fs.readdirSync("/proc")) {
        if (!/^\d+$/.test(name)) {
          continue;
        }
        const pid = Number(name);
        try {
          const exe = fs.readlinkSync(`/proc/${pid}/exe`);
          if (/llama-server/i.test(exe)) {
            pids.push(pid);
            continue;
          }
        } catch {
          // ignore
        }
        try {
          const cmd = fs.readFileSync(`/proc/${pid}/cmdline`, "utf8");
          if (looksLikeLlamaServer(cmd)) {
            pids.push(pid);
          }
        } catch {
          // ignore
        }
      }
      return pids;
    }
    if (process.platform === "win32") {
      const out = execFileSync("tasklist", ["/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
        windowsHide: true,
        timeout: 8000,
      });
      const pids: number[] = [];
      for (const line of out.split(/\r?\n/)) {
        const cols = line.split(",");
        if (cols.length < 2) {
          continue;
        }
        const pid = Number(cols[1].replace(/"/g, "").trim());
        if (Number.isFinite(pid) && pid > 0) {
          pids.push(pid);
        }
      }
      return pids;
    }
    const out = execFileSync("pgrep", ["-f", "llama-server"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 8000,
    });
    return uniquePids(out.split(/\s+/));
  } catch {
    return [];
  }
}

/**
 * llama-server processes whose executable (or command line) lives under `dir`.
 * Unknown paths are left alone — same rule as {@link isLlamaServerProcess}.
 */
export function findLlamaServerPidsUsingDir(dir: string): number[] {
  if (!dir) {
    return [];
  }
  const want = normalizePathForCompare(dir);
  const found = new Set<number>();
  for (const pid of listCandidateLlamaPids()) {
    if (pid === process.pid || !isPidAlive(pid)) {
      continue;
    }
    const exe = processExecutablePath(pid);
    if (executableIsUnderDir(exe, dir)) {
      found.add(pid);
      continue;
    }
    const cmd = processCommandLine(pid);
    if (cmd && looksLikeLlamaServer(cmd) && cmd.toLowerCase().includes(want.toLowerCase())) {
      found.add(pid);
    }
  }
  return [...found];
}

/** SIGKILL / taskkill a pid and its children; waits briefly for the pid to exit. */
export async function stopProcessTree(pid: number, force = true): Promise<void> {
  if (!isPidAlive(pid)) {
    return;
  }
  try {
    if (process.platform === "win32") {
      try {
        execFileSync("taskkill", ["/PID", String(pid), "/T", "/F"], {
          stdio: "ignore",
          windowsHide: true,
          timeout: 15_000,
        });
      } catch {
        // Process may already be gone.
      }
      for (let i = 0; i < 25; i++) {
        if (!isPidAlive(pid)) {
          break;
        }
        await sleep(100);
      }
      return;
    }
    try {
      process.kill(pid, force ? "SIGKILL" : "SIGTERM");
    } catch {
      return;
    }
    await sleep(force ? 200 : 500);
    if (isPidAlive(pid)) {
      try {
        process.kill(pid, "SIGKILL");
      } catch {
        // ignore
      }
    }
  } catch {
    // ignore
  }
}

/** Stop every llama-server whose binary lives in `dir`. Returns the pids we tried. */
export async function stopLlamaServersUsingDir(dir: string): Promise<number[]> {
  const pids = findLlamaServerPidsUsingDir(dir);
  for (const pid of pids) {
    await stopProcessTree(pid, true);
  }
  return pids;
}

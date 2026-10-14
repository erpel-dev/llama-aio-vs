/**
 * Probe logic that shells out (GPU tools, port scans, process lookups) is
 * written once as a generator that yields {@link ExecRequest}s. The TUI and
 * one-off callers drive it synchronously; the VS Code extension host drives
 * the same steps asynchronously so `nvidia-smi` / `netstat` never freeze
 * every other extension in the window.
 */
import { execFile, execFileSync, spawn, spawnSync } from "child_process";

export interface ExecRequest {
  command: string;
  args: string[];
  timeoutMs: number;
  env?: NodeJS.ProcessEnv;
  cwd?: string;
  /** Return raw stdout bytes in {@link ExecResult.buffer} (registry dumps in UTF-16). */
  binary?: boolean;
  /**
   * Collect stdout and stderr even when the exit code is non-zero
   * (`llama-server --list-devices` prints to stderr and may exit 1).
   */
  captureStderr?: boolean;
}

export interface ExecResult {
  /** Exit code 0 and the command could be started. */
  ok: boolean;
  /** Exit code, or null when killed / not started. */
  status: number | null;
  stdout: string;
  stderr: string;
  buffer?: Buffer;
}

/** A probe: yields commands to run, receives their results, returns a value. */
export type ExecSteps<T> = Generator<ExecRequest, T, ExecResult>;

const MAX_BUFFER = 16 * 1024 * 1024;

function failed(err: unknown): ExecResult {
  const e = err as { status?: number | null; stdout?: string | Buffer; stderr?: string | Buffer };
  return {
    ok: false,
    status: typeof e?.status === "number" ? e.status : null,
    stdout: e?.stdout ? String(e.stdout) : "",
    stderr: e?.stderr ? String(e.stderr) : "",
  };
}

export function execRequestSync(req: ExecRequest): ExecResult {
  if (req.captureStderr) {
    const r = spawnSync(req.command, req.args, {
      encoding: "utf8",
      timeout: req.timeoutMs,
      windowsHide: true,
      cwd: req.cwd,
      env: req.env,
      maxBuffer: MAX_BUFFER,
    });
    return { ok: !r.error && r.status === 0, status: r.status, stdout: r.stdout || "", stderr: r.stderr || "" };
  }
  try {
    if (req.binary) {
      const buffer = execFileSync(req.command, req.args, {
        stdio: ["ignore", "pipe", "ignore"],
        timeout: req.timeoutMs,
        windowsHide: true,
        cwd: req.cwd,
        env: req.env,
        maxBuffer: MAX_BUFFER,
      });
      return { ok: true, status: 0, stdout: "", stderr: "", buffer };
    }
    const stdout = execFileSync(req.command, req.args, {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: req.timeoutMs,
      windowsHide: true,
      cwd: req.cwd,
      env: req.env,
      maxBuffer: MAX_BUFFER,
    });
    return { ok: true, status: 0, stdout, stderr: "" };
  } catch (err) {
    return failed(err);
  }
}

export function execRequestAsync(req: ExecRequest): Promise<ExecResult> {
  if (req.captureStderr) {
    return new Promise((resolve) => {
      let stdout = "";
      let stderr = "";
      let child;
      try {
        child = spawn(req.command, req.args, {
          windowsHide: true,
          cwd: req.cwd,
          env: req.env,
          stdio: ["ignore", "pipe", "pipe"],
        });
      } catch (err) {
        resolve(failed(err));
        return;
      }
      const timer = setTimeout(() => child.kill(), req.timeoutMs);
      child.stdout?.setEncoding("utf8").on("data", (d: string) => {
        stdout += d;
      });
      child.stderr?.setEncoding("utf8").on("data", (d: string) => {
        stderr += d;
      });
      child.on("error", () => {
        clearTimeout(timer);
        resolve({ ok: false, status: null, stdout, stderr });
      });
      child.on("close", (status) => {
        clearTimeout(timer);
        resolve({ ok: status === 0, status, stdout, stderr });
      });
    });
  }
  return new Promise((resolve) => {
    try {
      execFile(
        req.command,
        req.args,
        {
          encoding: req.binary ? "buffer" : "utf8",
          timeout: req.timeoutMs,
          windowsHide: true,
          cwd: req.cwd,
          env: req.env,
          maxBuffer: MAX_BUFFER,
        },
        (err, stdout, stderr) => {
          if (err) {
            const code = (err as { code?: unknown }).code;
            resolve({
              ok: false,
              status: typeof code === "number" ? code : null,
              stdout: req.binary ? "" : String(stdout || ""),
              stderr: String(stderr || ""),
            });
            return;
          }
          resolve(
            req.binary
              ? { ok: true, status: 0, stdout: "", stderr: "", buffer: stdout as unknown as Buffer }
              : { ok: true, status: 0, stdout: String(stdout), stderr: String(stderr || "") }
          );
        }
      );
    } catch (err) {
      resolve(failed(err));
    }
  });
}

export function runStepsSync<T>(steps: ExecSteps<T>): T {
  let next = steps.next();
  while (!next.done) {
    next = steps.next(execRequestSync(next.value));
  }
  return next.value;
}

export async function runStepsAsync<T>(steps: ExecSteps<T>): Promise<T> {
  let next = steps.next();
  while (!next.done) {
    next = steps.next(await execRequestAsync(next.value));
  }
  return next.value;
}

/** Run one command through the async driver. */
export function execText(command: string, args: string[], timeoutMs: number): Promise<ExecResult> {
  return execRequestAsync({ command, args, timeoutMs });
}

/**
 * NixOS cannot run official llama.cpp Ubuntu binaries as-is: they hardcode the
 * FHS dynamic linker `/lib64/ld-linux-x86-64.so.2` (or the aarch64 equivalent),
 * which does not exist on a stock NixOS system.
 *
 * Workarounds we support:
 * - programs.nix-ld (makes the FHS linker path work system-wide)
 * - wrapping with `steam-run` (FHS user env) when available on PATH
 * - falling back to a nixpkgs `llama-server` already on PATH
 */
import * as fs from "fs";
import * as path from "path";
import { spawnSync } from "child_process";
import { findLlamaServerOnPath, whichOnPath, withBinaryDirEnv } from "./paths";

export function isNixOS(): boolean {
  if (process.platform !== "linux") {
    return false;
  }
  try {
    if (fs.existsSync("/etc/NIXOS")) {
      return true;
    }
  } catch {
    // ignore
  }
  try {
    if (fs.existsSync("/run/current-system/nixos-version")) {
      return true;
    }
  } catch {
    // ignore
  }
  return false;
}

/** Expected FHS program interpreter for official llama.cpp Linux builds. */
export function expectedFhsDynamicLinker(): string {
  if (process.arch === "arm64") {
    return "/lib/ld-linux-aarch64.so.1";
  }
  return "/lib64/ld-linux-x86-64.so.2";
}

export function hasUsableFhsDynamicLinker(): boolean {
  try {
    return fs.existsSync(expectedFhsDynamicLinker());
  } catch {
    return false;
  }
}

export function findSteamRun(): string | undefined {
  return whichOnPath("steam-run");
}

export interface BinaryProbeResult {
  ok: boolean;
  /** First line / short detail from --version or the error. */
  detail: string;
  /** Heuristic: failure looks like a missing dynamic linker / FHS layout. */
  missingLinkerLikely: boolean;
}

/**
 * Try `llama-server --version` the same way we launch (cwd + LD_LIBRARY_PATH).
 * Optional wrapper: e.g. steam-run → `steam-run <binary> --version`.
 */
export function probeLlamaServerRunnable(
  binary: string,
  options?: {
    env?: NodeJS.ProcessEnv;
    wrapper?: string;
    timeoutMs?: number;
  }
): BinaryProbeResult {
  const env = options?.env || withBinaryDirEnv(binary);
  const timeout = options?.timeoutMs ?? 8000;
  const wrapper = options?.wrapper;
  const command = wrapper || binary;
  const args = wrapper ? [binary, "--version"] : ["--version"];
  const result = spawnSync(command, args, {
    encoding: "utf8",
    timeout,
    windowsHide: true,
    cwd: path.dirname(binary),
    env,
  });

  const stdout = (result.stdout || "").toString().trim();
  const stderr = (result.stderr || "").toString().trim();
  const combined = [stdout, stderr].filter(Boolean).join("\n").trim();

  if (result.error) {
    const err = result.error as NodeJS.ErrnoException;
    const detail = [`code=${err.code || "error"}`, err.message, combined]
      .filter(Boolean)
      .join(" | ")
      .slice(0, 500);
    return {
      ok: false,
      detail,
      missingLinkerLikely: looksLikeMissingDynamicLinker(binary, err, detail),
    };
  }

  // llama-server often prints the version banner to stderr; treat a version
  // line as success even when the process exits non-zero.
  if (combined && /version\s*:/i.test(combined)) {
    return {
      ok: true,
      detail: combined.split("\n")[0]?.trim() || combined,
      missingLinkerLikely: false,
    };
  }

  if (result.status === 0) {
    return {
      ok: true,
      detail: (combined.split("\n")[0]?.trim() || combined || "ok").slice(0, 200),
      missingLinkerLikely: false,
    };
  }

  const detail =
    [
      typeof result.status === "number" ? `status=${result.status}` : "",
      result.signal ? `signal=${result.signal}` : "",
      combined,
    ]
      .filter(Boolean)
      .join(" | ")
      .slice(0, 500) || "unknown error";
  const fakeErr = Object.assign(new Error(detail), {
    code: result.status === null ? "SPAWN_FAILED" : undefined,
    status: result.status ?? undefined,
    stdout,
    stderr,
  });
  return {
    ok: false,
    detail,
    missingLinkerLikely: looksLikeMissingDynamicLinker(binary, fakeErr, detail),
  };
}

export function looksLikeMissingDynamicLinker(
  binary: string,
  err: NodeJS.ErrnoException & { status?: number; stderr?: string | Buffer; stdout?: string | Buffer },
  detail = ""
): boolean {
  const blob = `${detail}\n${err.message || ""}\n${err.stderr || ""}\n${err.stdout || ""}`.toLowerCase();
  // Kernel reports ENOENT when the ELF interpreter is missing, even if `binary` exists.
  if (err.code === "ENOENT" && fs.existsSync(binary)) {
    return true;
  }
  if (/no such file or directory/.test(blob) && fs.existsSync(binary)) {
    return true;
  }
  if (/error while loading shared libraries/.test(blob)) {
    return true;
  }
  if (/cannot open shared object file/.test(blob)) {
    return true;
  }
  return false;
}

export function nixOsIncompatibilityHint(): string {
  const linker = expectedFhsDynamicLinker();
  return (
    "Official llama.cpp Linux builds are Ubuntu binaries and need the FHS dynamic linker " +
    `(${linker}), which stock NixOS does not provide.\n\n` +
    "Fix options (pick one):\n" +
    "1. Enable nix-ld in configuration.nix:\n" +
    "     programs.nix-ld.enable = true;\n" +
    "   then rebuild, and restart VS Code.\n" +
    "2. Install steam-run (pkgs.steam-run) so Llama AIO can wrap the binary in an FHS env.\n" +
    "3. Install llama.cpp from nixpkgs (`llama-cpp`) so `llama-server` is on PATH —\n" +
    "   Llama AIO will use that when the downloaded binary cannot run.\n\n" +
    "Vulkan GPU on NixOS may also need nixGL / correct ICD packages."
  );
}

export type LaunchMethod = "direct" | "steam-run" | "path";

export interface LaunchPlan {
  /** Path recorded as the llama-server binary (lock / UI). */
  binary: string;
  /** Executable passed to spawn / terminal. */
  command: string;
  /**
   * Args that must precede the llama-server argv.
   * For steam-run this is `[binary]`; the caller appends server args.
   */
  prefixArgs: string[];
  env: NodeJS.ProcessEnv;
  method: LaunchMethod;
  /** Optional note for logs / UI. */
  note?: string;
}

/**
 * Decide how to launch an installed (or PATH) llama-server on this host.
 * Throws with a NixOS-oriented hint when nothing runnable is found.
 */
export function resolveLaunchPlan(installedBinary: string): LaunchPlan {
  const installedExists = !!installedBinary && fs.existsSync(installedBinary);
  const baseEnv = installedExists ? withBinaryDirEnv(installedBinary) : { ...process.env };

  if (installedExists) {
    const direct = probeLlamaServerRunnable(installedBinary, { env: baseEnv });
    if (direct.ok) {
      return {
        binary: installedBinary,
        command: installedBinary,
        prefixArgs: [],
        env: baseEnv,
        method: "direct",
      };
    }

    const steamRun = findSteamRun();
    if (steamRun) {
      const wrapped = probeLlamaServerRunnable(installedBinary, {
        env: baseEnv,
        wrapper: steamRun,
      });
      if (wrapped.ok) {
        return {
          binary: installedBinary,
          command: steamRun,
          prefixArgs: [installedBinary],
          env: baseEnv,
          method: "steam-run",
          note: `Wrapping via steam-run (${wrapped.detail})`,
        };
      }
    }

    // Prefer a system binary when the downloaded Ubuntu build cannot exec.
    const onPath = findLlamaServerOnPath();
    if (onPath && path.resolve(onPath) !== path.resolve(installedBinary)) {
      const pathProbe = probeLlamaServerRunnable(onPath, { env: withBinaryDirEnv(onPath) });
      if (pathProbe.ok) {
        return {
          binary: onPath,
          command: onPath,
          prefixArgs: [],
          env: withBinaryDirEnv(onPath),
          method: "path",
          note: `Downloaded binary is not runnable here; using PATH llama-server (${pathProbe.detail})`,
        };
      }
    }

    if (direct.missingLinkerLikely || isNixOS() || !hasUsableFhsDynamicLinker()) {
      throw new Error(
        `Cannot execute installed llama-server at ${installedBinary}.\n` +
          `Probe: ${direct.detail}\n\n` +
          nixOsIncompatibilityHint()
      );
    }

    throw new Error(
      `Cannot execute installed llama-server at ${installedBinary}.\nProbe: ${direct.detail}`
    );
  }

  const onPath = findLlamaServerOnPath();
  if (onPath) {
    const pathProbe = probeLlamaServerRunnable(onPath, { env: withBinaryDirEnv(onPath) });
    if (pathProbe.ok) {
      return {
        binary: onPath,
        command: onPath,
        prefixArgs: [],
        env: withBinaryDirEnv(onPath),
        method: "path",
      };
    }
  }

  throw new Error(
    `llama-server binary not found at ${installedBinary || "(unknown)"}. ` +
      `Run "Llama AIO: Install / Upgrade llama.cpp" first.` +
      (isNixOS() ? `\n\n${nixOsIncompatibilityHint()}` : "")
  );
}

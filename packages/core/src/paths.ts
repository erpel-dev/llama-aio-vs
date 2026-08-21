import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import type { ConfigAccessor } from "./config";

export type InstallBackendId = "vulkan" | "cuda" | "cpu";

/**
 * Shared data directory. Everything the extension and the TUI have in common
 * — config, llama.cpp builds, models, the server lock — hangs off this.
 */
export function getDataRoot(): string {
  return path.join(os.homedir(), ".llama-aio-vs");
}

/** The shared configuration file, written by whichever frontend is in front. */
export function getConfigPath(): string {
  return path.join(getDataRoot(), "config.json");
}

/** Root folder for llama.cpp installs (contains per-backend subdirs). */
export function getInstallDir(config: ConfigAccessor): string {
  const override = (config.get<string>("installDir") || "").trim();
  return override || path.join(getDataRoot(), "llama.cpp");
}

/** Per-backend install directory: `…/llama.cpp/vulkan`, `…/cpu`, etc. */
export function getBackendInstallDir(
  config: ConfigAccessor,
  backend: InstallBackendId
): string {
  return path.join(getInstallDir(config), backend);
}

export function getModelsDir(config: ConfigAccessor): string {
  const override = (config.get<string>("modelsDir") || "").trim();
  return override || path.join(getDataRoot(), "models");
}

/**
 * Prompt-replacement defaults bundled with the package.
 * Looks next to the running JS (packaged TUI / extension bundle) and one
 * directory up (compiled `packages/core/out` layout).
 */
export function getDefaultPromptReplacementsPath(): string {
  const name = path.join("prompt-replacements", "default-prompt-replacements.json");
  const candidates = [path.join(__dirname, name), path.join(__dirname, "..", name)];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return candidates[0];
}

export function getLockDir(): string {
  return path.join(getDataRoot(), "runtime");
}

export function getLockPath(): string {
  return path.join(getLockDir(), "server.lock.json");
}

/** Cross-window lock while llama.cpp binaries are being replaced. */
export function getInstallLockPath(): string {
  return path.join(getLockDir(), "install.lock.json");
}

export function getLogPath(): string {
  return path.join(getLockDir(), "llama-server.log");
}

export function ensureDirs(...dirs: string[]): void {
  for (const dir of dirs) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

export function getLlamaServerBinary(installDir: string): string {
  const binName = process.platform === "win32" ? "llama-server.exe" : "llama-server";
  const candidates = [
    path.join(installDir, "bin", binName),
    path.join(installDir, binName),
    path.join(installDir, "build", "bin", binName),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) {
      return c;
    }
  }
  return candidates[0];
}

export function whichOnPath(command: string): string | undefined {
  const pathEnv = process.env.PATH || "";
  const sep = process.platform === "win32" ? ";" : ":";
  const exts =
    process.platform === "win32"
      ? (process.env.PATHEXT || ".EXE;.CMD;.BAT").split(";").filter(Boolean)
      : [""];

  for (const dir of pathEnv.split(sep)) {
    if (!dir) {
      continue;
    }
    for (const ext of exts) {
      const candidateName =
        process.platform === "win32" && !path.extname(command) ? command + ext : command;
      const full = path.join(dir, candidateName);
      try {
        if (fs.existsSync(full) && fs.statSync(full).isFile()) {
          return full;
        }
      } catch {
        // ignore
      }
    }
  }
  return undefined;
}

export function findLlamaServerOnPath(): string | undefined {
  return whichOnPath(process.platform === "win32" ? "llama-server.exe" : "llama-server");
}

/**
 * Env for running a llama.cpp binary so sibling DLLs / shared libs resolve.
 * Windows needs the exe directory on PATH; Linux/macOS use *LD*_LIBRARY_PATH.
 */
export function withBinaryDirEnv(
  binary: string,
  base: NodeJS.ProcessEnv = process.env
): NodeJS.ProcessEnv {
  const binDir = path.dirname(binary);
  const env: NodeJS.ProcessEnv = { ...base };
  if (process.platform === "win32") {
    const sep = ";";
    env.PATH = `${binDir}${sep}${base.PATH || ""}`;
  } else if (process.platform === "linux") {
    env.LD_LIBRARY_PATH = `${binDir}${base.LD_LIBRARY_PATH ? `:${base.LD_LIBRARY_PATH}` : ""}`;
  } else if (process.platform === "darwin") {
    env.DYLD_LIBRARY_PATH = `${binDir}${base.DYLD_LIBRARY_PATH ? `:${base.DYLD_LIBRARY_PATH}` : ""}`;
  }
  return env;
}

import * as fs from "fs";
import * as path from "path";

/**
 * Intel oneAPI library directories relative to the toolkit root
 * (`/opt/intel/oneapi` or `$ONEAPI_ROOT`). The Ubuntu SYCL archive links
 * these and does not ship them. Windows zips vendor the DLLs beside the exe.
 */
const ONEAPI_LIB_DIRS = [
  "compiler/latest/lib",
  "mkl/latest/lib",
  "mkl/latest/lib/intel64",
  "tbb/latest/lib",
  "tbb/latest/lib/intel64/gcc4.8",
  "dnnl/latest/lib",
  "umf/latest/lib",
];

export interface SyclSupport {
  /** Official llama.cpp archive exists for this OS/arch, and Linux can see libsycl. */
  available: boolean;
  reason?: string;
}

function dirHasLibsycl(dir: string): boolean {
  return (
    fs.existsSync(path.join(dir, "libsycl.so")) ||
    fs.existsSync(path.join(dir, "libsycl.so.8"))
  );
}

/** Toolkit roots to search. `ONEAPI_ROOT` wins over the default Linux prefix. */
export function oneApiRoots(env: NodeJS.ProcessEnv = process.env): string[] {
  const roots: string[] = [];
  const fromEnv = (env.ONEAPI_ROOT || "").trim();
  if (fromEnv) {
    roots.push(fromEnv);
  }
  if (process.platform !== "win32") {
    const def = "/opt/intel/oneapi";
    if (!roots.includes(def)) {
      roots.push(def);
    }
  }
  return roots;
}

/**
 * Existing oneAPI lib directories under `root`. Includes a versioned
 * `compiler/<ver>/lib` when `compiler/latest` is absent. Empty when the
 * root is missing.
 */
export function discoverOneApiLibraryDirs(root: string): string[] {
  if (!root || !fs.existsSync(root)) {
    return [];
  }
  const found: string[] = [];
  const push = (dir: string) => {
    if (!dir || found.includes(dir) || !fs.existsSync(dir)) {
      return;
    }
    found.push(dir);
  };

  const compiler = path.join(root, "compiler");
  if (fs.existsSync(compiler)) {
    let versions: string[] = [];
    try {
      versions = fs.readdirSync(compiler);
    } catch {
      versions = [];
    }
    // `latest` first so the active toolkit is ahead of older side-by-side installs.
    versions.sort((a, b) => (a === "latest" ? -1 : b === "latest" ? 1 : a < b ? 1 : -1));
    for (const ver of versions) {
      const lib = path.join(compiler, ver, "lib");
      if (dirHasLibsycl(lib)) {
        push(lib);
      }
    }
  }

  for (const rel of ONEAPI_LIB_DIRS) {
    push(path.join(root, rel));
  }
  return found;
}

export function oneApiRuntimePresent(env: NodeJS.ProcessEnv = process.env): boolean {
  return oneApiRoots(env).some((root) =>
    discoverOneApiLibraryDirs(root).some((dir) => dirHasLibsycl(dir))
  );
}

export function syclRuntimeMissingReason(): string {
  return (
    "The Linux SYCL build does not include the Intel runtime. Install the Intel oneAPI Toolkit " +
    "so libsycl.so is under /opt/intel/oneapi (or set ONEAPI_ROOT), add your user to the render and video groups, and log in again."
  );
}

/** Whether this machine can download and run an official SYCL llama-server. */
export function syclSupport(env: NodeJS.ProcessEnv = process.env): SyclSupport {
  if (process.arch !== "x64" || (process.platform !== "win32" && process.platform !== "linux")) {
    return {
      available: false,
      reason: "No SYCL release packages for this OS/arch. Official builds are Windows x64 and Linux x64.",
    };
  }
  if (process.platform === "linux" && !oneApiRuntimePresent(env)) {
    return { available: false, reason: syclRuntimeMissingReason() };
  }
  return { available: true };
}

/**
 * Env for a SYCL llama-server. Sets `ZES_ENABLE_SYSMAN` so Level Zero can
 * report free memory, and on Linux prepends oneAPI lib dirs to
 * `LD_LIBRARY_PATH`. Does not override a sysman value the user already set.
 */
export function withSyclRuntimeEnv(base: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...base };
  if (!env.ZES_ENABLE_SYSMAN) {
    env.ZES_ENABLE_SYSMAN = "1";
  }
  if (process.platform === "win32") {
    return env;
  }
  const dirs: string[] = [];
  for (const root of oneApiRoots(env)) {
    for (const dir of discoverOneApiLibraryDirs(root)) {
      if (!dirs.includes(dir)) {
        dirs.push(dir);
      }
    }
  }
  if (!dirs.length) {
    return env;
  }
  const prefix = dirs.join(path.delimiter);
  env.LD_LIBRARY_PATH = env.LD_LIBRARY_PATH ? `${prefix}${path.delimiter}${env.LD_LIBRARY_PATH}` : prefix;
  return env;
}

export function formatSyclLaunchFailure(binary: string, probe: string): string {
  const lines = [`Cannot execute the SYCL llama-server at ${binary}.`, `Probe: ${probe}`];
  if (process.platform === "linux" || /libsycl|libze_loader|cannot open shared object/i.test(probe)) {
    lines.push("", syclRuntimeMissingReason());
  } else {
    lines.push("", "Install the Intel GPU driver, then try again.");
  }
  return lines.join("\n");
}

import * as fs from "fs";
import * as path from "path";

/**
 * AMD ROCm library locations under `$ROCM_PATH` or `/opt/rocm`.
 * The Ubuntu llama.cpp archive links hipBLAS and does not ship it.
 * Newer TheRock packages keep libraries in `core-<version>/lib`.
 */
const HIPBLAS_NAMES = ["libhipblas.so", "libhipblas.so.2", "libhipblas.so.3"];

export interface RocmSupport {
  /** Official llama.cpp archive exists for this OS/arch, and Linux can see libhipblas. */
  available: boolean;
  reason?: string;
}

function dirHasHipblas(dir: string): boolean {
  return HIPBLAS_NAMES.some((name) => fs.existsSync(path.join(dir, name)));
}

/** ROCm roots. `ROCM_PATH` wins over `/opt/rocm`. */
export function rocmRoots(env: NodeJS.ProcessEnv = process.env): string[] {
  const roots: string[] = [];
  const fromEnv = (env.ROCM_PATH || "").trim();
  if (fromEnv) {
    roots.push(fromEnv);
  }
  if (process.platform !== "win32") {
    const def = "/opt/rocm";
    if (!roots.includes(def)) {
      roots.push(def);
    }
  }
  return roots;
}

/** Existing library directories under one ROCm root. Empty when the root is missing. */
export function discoverRocmLibraryDirs(root: string): string[] {
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
  push(path.join(root, "lib"));
  push(path.join(root, "lib64"));
  let entries: string[] = [];
  try {
    entries = fs.readdirSync(root);
  } catch {
    entries = [];
  }
  for (const name of entries) {
    if (name === "lib" || name === "lib64" || name.startsWith("core-")) {
      push(path.join(root, name, "lib"));
    }
  }
  return found;
}

export function rocmRuntimePresent(env: NodeJS.ProcessEnv = process.env): boolean {
  return rocmRoots(env).some((root) => discoverRocmLibraryDirs(root).some((dir) => dirHasHipblas(dir)));
}

export function rocmRuntimeMissingReason(): string {
  return (
    "The Linux ROCm build does not include the AMD runtime. Install a ROCm version that matches this nightly " +
    "(libhipblas.so under /opt/rocm or ROCM_PATH), add your user to the render and video groups, and log in again."
  );
}

/** Whether this machine can download and run an official ROCm llama-server. */
export function rocmSupport(env: NodeJS.ProcessEnv = process.env): RocmSupport {
  if (process.arch !== "x64" || (process.platform !== "win32" && process.platform !== "linux")) {
    return {
      available: false,
      reason: "No ROCm release packages for this OS/arch. Official builds are Windows x64 and Linux x64.",
    };
  }
  if (process.platform === "linux" && !rocmRuntimePresent(env)) {
    return { available: false, reason: rocmRuntimeMissingReason() };
  }
  return { available: true };
}

/**
 * Env for a ROCm llama-server. Prepends hipBLAS library dirs and sets
 * `ROCM_PATH` when the user has not set it. Does not set
 * `HSA_OVERRIDE_GFX_VERSION`; that override is for cards the build does not list.
 */
export function withRocmRuntimeEnv(base: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...base };
  if (process.platform === "win32") {
    return env;
  }
  const dirs: string[] = [];
  let chosenRoot: string | undefined;
  for (const root of rocmRoots(env)) {
    const hits = discoverRocmLibraryDirs(root).filter((dir) => dirHasHipblas(dir));
    if (!hits.length) {
      continue;
    }
    chosenRoot = chosenRoot || root;
    for (const dir of hits) {
      if (!dirs.includes(dir)) {
        dirs.push(dir);
      }
    }
  }
  if (!dirs.length) {
    return env;
  }
  if (!env.ROCM_PATH && chosenRoot) {
    env.ROCM_PATH = chosenRoot;
  }
  const prefix = dirs.join(path.delimiter);
  env.LD_LIBRARY_PATH = env.LD_LIBRARY_PATH ? `${prefix}${path.delimiter}${env.LD_LIBRARY_PATH}` : prefix;
  return env;
}

export function formatRocmLaunchFailure(binary: string, probe: string): string {
  const lines = [`Cannot execute the ROCm llama-server at ${binary}.`, `Probe: ${probe}`];
  if (process.platform === "linux" || /libhipblas|libamdhip64|librocblas/i.test(probe)) {
    lines.push("", rocmRuntimeMissingReason());
  } else {
    lines.push("", "Install the AMD ROCm or HIP runtime that matches this build, then try again.");
  }
  return lines.join("\n");
}

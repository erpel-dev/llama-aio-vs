import * as fs from "fs";
import * as path from "path";
import { runStepsAsync, runStepsSync, type ExecRequest, type ExecSteps } from "./execRunner";
import { parseLlamaListDevices } from "./gpuInfo";
import type { LaunchPlan } from "./nixCompat";

/** Optional flags added in October 2026. Probe the executable, including forks/backports. */
export interface LlamaServerFeatures {
  probed: boolean;
  moeCache: boolean;
  draftSampling: boolean;
  draftUbatch: boolean;
}

export const NO_SERVER_FEATURES: Readonly<LlamaServerFeatures> = Object.freeze({
  probed: false, moeCache: false, draftSampling: false, draftUbatch: false,
});

export function parseLlamaServerFeatures(help: string): LlamaServerFeatures {
  const plain = help.replace(/\x1b\[[0-9;]*m/g, "");
  const definition = (flag: string): string =>
    plain.match(new RegExp(`^\\s*(?:-[\\w-]+,\\s*)*${flag}(?=[\\s,=]|$)[^\\n]*`, "m"))?.[0] || "";
  return {
    probed: true,
    moeCache: !!definition("--moe-cache-mib"),
    draftSampling: /\bprobabilistic\b/.test(definition("--spec-draft-sampling")),
    draftUbatch: !!definition("--spec-draft-n-ubatch"),
  };
}

type ProbeLaunch = Pick<LaunchPlan, "binary" | "command" | "prefixArgs" | "env">;
const cache = new Map<string, { features: LlamaServerFeatures; expires: number }>();
const pending = new Map<string, Promise<LlamaServerFeatures>>();

function cacheKey(launch: ProbeLaunch): string | undefined {
  try {
    const st = fs.statSync(launch.binary);
    // ctime/ino also catch an atomic replacement with preserved mtime and size.
    return JSON.stringify([launch.binary, st.size, st.mtimeMs, st.ctimeMs, st.ino,
      launch.command, launch.prefixArgs, launch.env.LD_LIBRARY_PATH, launch.env.PATH]);
  } catch {
    return undefined;
  }
}

function request(launch: ProbeLaunch, flag: string): ExecRequest {
  return { command: launch.command, args: [...launch.prefixArgs, flag],
    env: launch.env, cwd: path.dirname(launch.binary), timeoutMs: 5000 };
}

function* probeSteps(launch: ProbeLaunch): ExecSteps<LlamaServerFeatures> {
  const result = yield request(launch, "--help");
  // Help must complete successfully; partial output from a killed/failed probe is unsafe.
  if (!result.ok) return { ...NO_SERVER_FEATURES };
  const features = parseLlamaServerFeatures(result.stdout + "\n" + result.stderr);
  if (features.moeCache) {
    // CPU builds advertise the shared flag too, but using it throws without
    // a GPU backend. Probe the binary itself, not just physical GPU hardware.
    const devices = yield { ...request(launch, "--list-devices"), captureStderr: true };
    features.moeCache = devices.status !== null &&
      parseLlamaListDevices(devices.stdout + "\n" + devices.stderr).length > 0;
  }
  return features;
}

function finish(key: string, features: LlamaServerFeatures): LlamaServerFeatures {
  if (cache.size >= 32) cache.delete(cache.keys().next().value!);
  cache.set(key, { features, expires: features.probed ? Date.now() + 60_000 : Date.now() + 5000 });
  return features;
}

/** Cached by executable identity. The probe follows the exact runtime env / steam-run wrapper. */
export async function probeLlamaServerFeatures(launch: ProbeLaunch): Promise<LlamaServerFeatures> {
  const key = cacheKey(launch);
  if (!key) return { ...NO_SERVER_FEATURES };
  const hit = cache.get(key);
  if (hit && hit.expires > Date.now()) return { ...hit.features };
  let work = pending.get(key);
  if (!work) {
    work = runStepsAsync(probeSteps(launch)).then((features) => finish(key, features))
      .finally(() => pending.delete(key));
    pending.set(key, work);
  }
  return { ...await work };
}

/** For the synchronous TUI and Copy command. Launch uses the non-blocking counterpart. */
export function probeLlamaServerFeaturesSync(launch: ProbeLaunch): LlamaServerFeatures {
  const key = cacheKey(launch);
  if (!key) return { ...NO_SERVER_FEATURES };
  const hit = cache.get(key);
  return { ...(hit && hit.expires > Date.now() ? hit.features : finish(key, runStepsSync(probeSteps(launch)))) };
}

/**
 * What a remote llama-server (or other OpenAI-compatible server) tells us, and
 * how its connection state is derived. Everything here comes from HTTP; nothing
 * is filled in from this machine's GPUs or model files.
 */
import { Emitter, Event } from "./events";
import {
  ConnectionError,
  type ConnectionErrorKind,
  joinServerUrl,
  readBody,
  serverJson,
  serverRequest,
} from "./serverHttp";

export type RemoteState =
  | "connecting"
  | "loading"
  | "ready"
  | "busy"
  | "auth"
  | "limited"
  | "unreachable";

export interface RemoteProbeRow {
  path: "/health" | "/v1/models" | "/props" | "/slots" | "/metrics" | "/api/native";
  /** true = answered OK, false = failed, null = skipped or endpoint off. */
  ok: boolean | null;
  status?: number;
  detail: string;
}

/** The four `/metrics` gauges the sidebar shows. Other series are ignored. */
export interface LlamaMetrics {
  /** `llamacpp:requests_processing` — requests currently in a slot. */
  requestsProcessing?: number;
  /** `llamacpp:requests_deferred` — requests waiting because every slot is taken. */
  requestsDeferred?: number;
  /** `llamacpp:predicted_tokens_seconds` — server-wide generation throughput, tokens/s. */
  serverGenTokPerSec?: number;
  /** `llamacpp:prompt_tokens_seconds` — server-wide prompt throughput, tokens/s. */
  serverPromptTokPerSec?: number;
}

export interface RemoteServerInfo {
  url: string;
  state: RemoteState;
  checkedAt: number;
  rttMs?: number;
  error?: { kind: ConnectionErrorKind; message: string };
  /** `/props` answered: this is llama.cpp with its extras (`/tokenize`, slots). */
  isLlamaCpp: boolean;
  modelId?: string;
  /** Canonical served identity, distinct from a user-facing request alias. */
  modelAlias?: string;
  /** Strata Desktop's human-readable family name for opaque IDs such as q2_0. */
  modelName?: string;
  modelSizeBytes?: number;
  modelParams?: number;
  trainedContext?: number;
  /** Served context from the model list or the matching Strata Desktop engine. */
  servedContext?: number;
  slotContext?: number;
  totalSlots?: number;
  /**
   * Slots with `is_processing` from `GET /slots`. When that endpoint is off,
   * filled from `requests_processing` and `/props` `total_slots` instead.
   * Prompt text from `/slots` is never stored.
   */
  slotsBusy?: number;
  slotsTotal?: number;
  /** From `GET /metrics`. Absent when the server was not started with `--metrics`. */
  requestsProcessing?: number;
  requestsDeferred?: number;
  /** Server-wide, all clients. Not this window's Performance-card rate. */
  serverGenTokPerSec?: number;
  serverPromptTokPerSec?: number;
  /**
   * vLLM counters from the last `/metrics` scrape. The next scrape turns the
   * difference into tok/s. Not shown on their own.
   */
  vllmGenerationTokensTotal?: number;
  vllmPromptTokensTotal?: number;
  vllmMetricsAt?: number;
  modelPath?: string;
  buildInfo?: string;
  vision?: boolean;
  audio?: boolean;
  probes: RemoteProbeRow[];
}

interface ModelsResponse {
  data?: Array<{
    id?: string;
    owned_by?: string;
    meta?: { size?: number; n_params?: number; n_ctx_train?: number; label?: string; n_ctx?: number; context?: number; instanceId?: string };
    /** vLLM's served context length. */
    max_model_len?: number;
  }>;
}

interface PropsResponse {
  default_generation_settings?: { n_ctx?: number };
  total_slots?: number;
  model_path?: string;
  model_alias?: string;
  build_info?: string;
  modalities?: { vision?: boolean; audio?: boolean };
}

/** Remote timeouts are looser than local ones: Wi-Fi and VPN round-trips are slow. */
export const REMOTE_HEALTH_TIMEOUT_MS = 4000;
const REMOTE_PROPS_TIMEOUT_MS = 5000;

function positive(n: unknown): number | undefined {
  return typeof n === "number" && Number.isFinite(n) && n > 0 ? n : undefined;
}

/** The loaded engine's limit, never a catalog/training limit or a saved config. */
async function attachStrataContext(info: RemoteServerInfo, instanceId: string | undefined, apiKey?: string): Promise<void> {
  try {
    const native = await serverJson<{ instances?: Array<{
      id?: string; apiModelId?: string; status?: string; vision?: boolean;
      metrics?: { engine?: { max_context?: number; images?: boolean } };
    }> }>(joinServerUrl(info.url, "/api/native"), {
      apiKey,
      timeoutMs: 2000,
      signal: AbortSignal.timeout(2000),
    });
    const matches = Array.isArray(native?.instances) ? native.instances.filter(i =>
      i?.apiModelId === info.modelId && (!instanceId || i.id === instanceId) && i.status === "ready") : [];
    const engine = matches.length === 1 ? matches[0].metrics?.engine : undefined;
    const context = matches.length === 1 ? positive(engine?.max_context) : undefined;
    // Strata answers /props with its web UI, so `/api/native` is the only place
    // that says whether the loaded engine has a picture reader (mmproj).
    const vision = matches.length === 1
      ? typeof matches[0].vision === "boolean" ? matches[0].vision
        : typeof engine?.images === "boolean" ? engine.images : undefined
      : undefined;
    if (vision !== undefined) {
      info.vision = vision;
    }
    const bits = [
      context !== undefined && Number.isSafeInteger(context) ? `served context ${context}` : "no matching engine context reported",
      vision === true ? "vision" : "",
    ].filter(Boolean);
    if (context !== undefined && Number.isSafeInteger(context)) {
      info.servedContext = context;
      info.probes.push({ path: "/api/native", ok: true, status: 200, detail: bits.join(" · ") });
    } else {
      info.probes.push({ path: "/api/native", ok: null, status: 200, detail: bits.join(" · ") });
    }
  } catch {
    // Optional metadata must not block chat or expose runtime response bodies.
    info.probes.push({ path: "/api/native", ok: null, detail: "engine context unavailable" });
  }
}

/** Finite and non-negative. Zero is a real gauge reading (idle). */
function gauge(n: number): number | undefined {
  return Number.isFinite(n) && n >= 0 ? n : undefined;
}

const METRIC_FIELDS: Record<string, keyof LlamaMetrics> = {
  "llamacpp:requests_processing": "requestsProcessing",
  "llamacpp:requests_deferred": "requestsDeferred",
  "llamacpp:predicted_tokens_seconds": "serverGenTokPerSec",
  "llamacpp:prompt_tokens_seconds": "serverPromptTokPerSec",
};

/**
 * Count slots whose `is_processing` is true. Returns undefined when the body
 * is not a slot array (disabled endpoint, HTML, error object). The `prompt`
 * field is not copied.
 */
export function parseSlots(body: string): { busy: number; total: number } | undefined {
  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    return undefined;
  }
  if (!Array.isArray(parsed) || parsed.length === 0) {
    return undefined;
  }
  let busy = 0;
  for (const slot of parsed) {
    if (slot && typeof slot === "object" && (slot as { is_processing?: unknown }).is_processing === true) {
      busy += 1;
    }
  }
  return { busy, total: parsed.length };
}

/** Pull the four sidebar gauges out of Prometheus text. Other series are dropped. */
export function parseLlamaMetrics(text: string): LlamaMetrics | undefined {
  const out: LlamaMetrics = {};
  let found = false;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const match = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(\S+)/.exec(line);
    if (!match) {
      continue;
    }
    const field = METRIC_FIELDS[match[1]];
    if (!field) {
      continue;
    }
    const value = gauge(Number(match[2]));
    if (value === undefined) {
      continue;
    }
    out[field] = value;
    found = true;
  }
  return found ? out : undefined;
}

function finite(n: unknown): number | undefined {
  return typeof n === "number" && Number.isFinite(n) ? n : undefined;
}

/** Tokens per second from a token count and a duration in milliseconds. */
function tokensPerSecond(tokens: number | undefined, durationMs: number | undefined): number | undefined {
  if (tokens === undefined || durationMs === undefined || durationMs <= 0) {
    return undefined;
  }
  return gauge(tokens / (durationMs / 1000));
}

/**
 * Strata's `GET /metrics` is JSON, not Prometheus. Live rates are null while
 * the server is idle, so generation and prompt throughput fall back to the
 * same session averages Strata publishes as `llamacpp:predicted_tokens_seconds`
 * and `llamacpp:prompt_tokens_seconds`. Request history is not kept.
 */
export function parseStrataMetrics(text: string): LlamaMetrics | undefined {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return undefined;
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return undefined;
  }
  const root = parsed as { live?: unknown; totals?: unknown };
  const live = root.live && typeof root.live === "object" && !Array.isArray(root.live)
    ? (root.live as Record<string, unknown>)
    : undefined;
  const totals = root.totals && typeof root.totals === "object" && !Array.isArray(root.totals)
    ? (root.totals as Record<string, unknown>)
    : undefined;
  if (!live && !totals) {
    return undefined;
  }
  const out: LlamaMetrics = {};
  const queued = finite(live?.queued);
  if (queued !== undefined && queued >= 0) {
    out.requestsDeferred = queued;
  }
  const state = typeof live?.state === "string" ? live.state : "";
  if (state === "generating" || state === "reading") {
    const running = finite(live?.running);
    out.requestsProcessing = running !== undefined && running >= 0 ? running : 1;
  } else if (state === "idle" || state === "unloaded") {
    out.requestsProcessing = 0;
  }
  const genNow = positive(finite(live?.tok_s)) ?? positive(finite(live?.tok_s_mean));
  const gen = genNow ?? tokensPerSecond(finite(totals?.output_tokens), finite(totals?.decode_ms));
  if (gen !== undefined) {
    out.serverGenTokPerSec = gen;
  }
  const promptNow = positive(finite(live?.prefill_tok_s_mean));
  const promptTokens = finite(totals?.prompt_tokens);
  const reused = finite(totals?.reused) ?? 0;
  const promptRead = promptTokens !== undefined ? Math.max(0, promptTokens - reused) : undefined;
  const prompt = promptNow ?? tokensPerSecond(promptRead, finite(totals?.prompt_ms));
  if (prompt !== undefined) {
    out.serverPromptTokPerSec = prompt;
  }
  return out.requestsDeferred !== undefined || out.requestsProcessing !== undefined || gen !== undefined || prompt !== undefined
    ? out
    : undefined;
}

/** Gauges a current vLLM `/metrics` scrape can show immediately. Throughput gauges are the removed pre-v1 names. */
const VLLM_GAUGES: Record<string, keyof LlamaMetrics> = {
  "vllm:num_requests_running": "requestsProcessing",
  "vllm:num_requests_waiting": "requestsDeferred",
  "vllm:avg_generation_throughput_toks_per_s": "serverGenTokPerSec",
  "vllm:avg_prompt_throughput_toks_per_s": "serverPromptTokPerSec",
};

const VLLM_COUNTERS = {
  "vllm:generation_tokens_total": "generationTokensTotal",
  "vllm:prompt_tokens_total": "promptTokensTotal",
} as const;

export interface VllmMetrics extends LlamaMetrics {
  generationTokensTotal?: number;
  promptTokensTotal?: number;
}

/**
 * vLLM's `/metrics` is Prometheus text with a `vllm:` prefix and a `model_name`
 * label. Samples of the same series are summed. Token totals are counters, not
 * tok/s; {@link applyVllmThroughput} turns two scrapes into a rate.
 */
export function parseVllmMetrics(text: string): VllmMetrics | undefined {
  const sums = new Map<string, number>();
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const match = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(\S+)/.exec(line);
    if (!match || (!VLLM_GAUGES[match[1]] && !VLLM_COUNTERS[match[1] as keyof typeof VLLM_COUNTERS])) {
      continue;
    }
    const value = gauge(Number(match[2]));
    if (value === undefined) {
      continue;
    }
    sums.set(match[1], (sums.get(match[1]) ?? 0) + value);
  }
  if (sums.size === 0) {
    return undefined;
  }
  const out: VllmMetrics = {};
  for (const [name, field] of Object.entries(VLLM_GAUGES)) {
    const value = sums.get(name);
    if (value !== undefined) {
      out[field] = value;
    }
  }
  for (const [name, field] of Object.entries(VLLM_COUNTERS)) {
    const value = sums.get(name);
    if (value !== undefined) {
      out[field] = value;
    }
  }
  return out;
}

/** Difference of two vLLM counter scrapes, in tokens/s. A restart (counter went backwards) is ignored. */
export function applyVllmThroughput(prev: RemoteServerInfo | undefined, next: RemoteServerInfo): void {
  if (!prev || prev.url !== next.url || typeof prev.vllmMetricsAt !== "number" || typeof next.vllmMetricsAt !== "number") {
    return;
  }
  const dt = (next.vllmMetricsAt - prev.vllmMetricsAt) / 1000;
  if (!(dt >= 1)) {
    return;
  }
  const gen = counterRate(prev.vllmGenerationTokensTotal, next.vllmGenerationTokensTotal, dt);
  const prompt = counterRate(prev.vllmPromptTokensTotal, next.vllmPromptTokensTotal, dt);
  if (next.serverGenTokPerSec === undefined && gen !== undefined) {
    next.serverGenTokPerSec = gen;
  }
  if (next.serverPromptTokPerSec === undefined && prompt !== undefined) {
    next.serverPromptTokPerSec = prompt;
  }
}

function counterRate(before: number | undefined, after: number | undefined, dt: number): number | undefined {
  if (before === undefined || after === undefined) {
    return undefined;
  }
  const delta = after - before;
  if (delta < 0) {
    return undefined;
  }
  return delta / dt;
}

/** Every slot is taken, or the server is already queueing requests. */
export function remoteLoadBusy(info: Pick<RemoteServerInfo, "slotsBusy" | "slotsTotal" | "requestsDeferred">): boolean {
  const full = typeof info.slotsTotal === "number" && info.slotsTotal > 0 && info.slotsBusy === info.slotsTotal;
  return full || (typeof info.requestsDeferred === "number" && info.requestsDeferred > 0);
}

async function readLoad(
  baseUrl: string,
  path: "/slots" | "/metrics",
  apiKey: string | undefined
): Promise<{ status: number; body: string } | { error: ConnectionError }> {
  try {
    const res = await serverRequest(joinServerUrl(baseUrl, path), {
      apiKey,
      timeoutMs: REMOTE_PROPS_TIMEOUT_MS,
    });
    const body = await readBody(res, 1024 * 1024);
    return { status: res.statusCode || 0, body };
  } catch (err) {
    return { error: err instanceof ConnectionError ? err : new ConnectionError("network", String(err)) };
  }
}

/** `/slots` then `/metrics`. Either being off leaves the connection usable. Returns true on 401/403. */
async function attachServerLoad(info: RemoteServerInfo, baseUrl: string, apiKey: string | undefined): Promise<boolean> {
  let auth = false;
  const slots = await readLoad(baseUrl, "/slots", apiKey);
  if ("error" in slots) {
    info.probes.push({ path: "/slots", ok: false, status: slots.error.status, detail: slots.error.message });
    auth = slots.error.kind === "auth";
  } else if (slots.status === 401 || slots.status === 403) {
    info.probes.push({ path: "/slots", ok: false, status: slots.status, detail: "rejected (API key)" });
    auth = true;
  } else if (slots.status === 404 || slots.status === 501) {
    info.probes.push({ path: "/slots", ok: null, status: slots.status, detail: "off (start with --slots)" });
  } else if (slots.status < 200 || slots.status >= 300) {
    info.probes.push({ path: "/slots", ok: false, status: slots.status, detail: `HTTP ${slots.status}` });
  } else {
    const parsed = parseSlots(slots.body);
    if (parsed) {
      info.slotsBusy = parsed.busy;
      info.slotsTotal = parsed.total;
      info.probes.push({ path: "/slots", ok: true, status: slots.status, detail: `${parsed.busy}/${parsed.total} busy` });
    } else {
      info.probes.push({ path: "/slots", ok: null, status: slots.status, detail: "unexpected body" });
    }
  }

  const metrics = await readLoad(baseUrl, "/metrics", apiKey);
  if ("error" in metrics) {
    info.probes.push({ path: "/metrics", ok: false, status: metrics.error.status, detail: metrics.error.message });
    auth = auth || metrics.error.kind === "auth";
  } else if (metrics.status === 401 || metrics.status === 403) {
    info.probes.push({ path: "/metrics", ok: false, status: metrics.status, detail: "rejected (API key)" });
    auth = true;
  } else if (metrics.status === 404 || metrics.status === 501) {
    info.probes.push({ path: "/metrics", ok: null, status: metrics.status, detail: "off (start with --metrics)" });
  } else if (metrics.status < 200 || metrics.status >= 300) {
    info.probes.push({ path: "/metrics", ok: false, status: metrics.status, detail: `HTTP ${metrics.status}` });
  } else {
    const parsed = parseLlamaMetrics(metrics.body) ?? parseStrataMetrics(metrics.body) ?? parseVllmMetrics(metrics.body);
    if (parsed) {
      info.requestsProcessing = parsed.requestsProcessing;
      info.requestsDeferred = parsed.requestsDeferred;
      info.serverGenTokPerSec = parsed.serverGenTokPerSec;
      info.serverPromptTokPerSec = parsed.serverPromptTokPerSec;
      if ("generationTokensTotal" in parsed || "promptTokensTotal" in parsed) {
        const vllm = parsed as VllmMetrics;
        info.vllmGenerationTokensTotal = vllm.generationTokensTotal;
        info.vllmPromptTokensTotal = vllm.promptTokensTotal;
        info.vllmMetricsAt = Date.now();
      }
      const bits = [
        parsed.requestsProcessing !== undefined ? `${parsed.requestsProcessing} processing` : "",
        parsed.requestsDeferred ? `${parsed.requestsDeferred} queued` : "",
      ].filter(Boolean);
      info.probes.push({ path: "/metrics", ok: true, status: metrics.status, detail: bits.join(" · ") || "gauges" });
    } else {
      info.probes.push({ path: "/metrics", ok: null, status: metrics.status, detail: "no llamacpp gauges" });
    }
  }

  if (info.slotsBusy === undefined && info.requestsProcessing !== undefined && info.totalSlots) {
    info.slotsBusy = Math.min(info.requestsProcessing, info.totalSlots);
    info.slotsTotal = info.totalSlots;
  }
  return auth;
}

/** Result of `GET /health`, with round-trip time. */
export async function probeHealth(
  baseUrl: string,
  opts: { apiKey?: string; timeoutMs?: number } = {}
): Promise<{ status: number; rttMs: number; body: string }> {
  const t0 = Date.now();
  const res = await serverRequest(joinServerUrl(baseUrl, "/health"), {
    apiKey: opts.apiKey,
    timeoutMs: opts.timeoutMs ?? REMOTE_HEALTH_TIMEOUT_MS,
  });
  const body = await readBody(res, 64 * 1024);
  return { status: res.statusCode || 0, rttMs: Date.now() - t0, body };
}

/**
 * Full probe: `/health` → `/v1/models` → `/props`, then `/slots` and `/metrics`
 * when this is llama.cpp. A missing load endpoint (404/501) does not fail the
 * connection. `/health` and `/v1/models` stay public under `llama-server --api-key`,
 * so a 401 on `/props` pinpoints a missing or wrong key while the model can still be named.
 */
export async function probeRemoteServer(
  baseUrl: string,
  opts: { apiKey?: string } = {}
): Promise<RemoteServerInfo> {
  const info: RemoteServerInfo = {
    url: baseUrl,
    state: "connecting",
    checkedAt: Date.now(),
    isLlamaCpp: false,
    probes: [],
  };

  let healthStatus: number;
  try {
    const health = await probeHealth(baseUrl, { apiKey: opts.apiKey });
    healthStatus = health.status;
    info.rttMs = health.rttMs;
    if (health.status >= 200 && health.status < 300) {
      info.probes.push({ path: "/health", ok: true, status: health.status, detail: `reachable · ${health.rttMs} ms` });
    } else if (health.status === 503) {
      info.probes.push({ path: "/health", ok: false, status: 503, detail: "loading model" });
    } else if (health.status === 401 || health.status === 403) {
      info.probes.push({ path: "/health", ok: false, status: health.status, detail: "rejected (API key)" });
    } else if (health.status === 404) {
      info.probes.push({ path: "/health", ok: null, status: 404, detail: `no /health · ${health.rttMs} ms` });
    } else {
      info.probes.push({ path: "/health", ok: false, status: health.status, detail: `HTTP ${health.status}` });
    }
  } catch (err) {
    const e = err instanceof ConnectionError ? err : new ConnectionError("network", String(err));
    info.state = "unreachable";
    info.error = { kind: e.kind, message: e.message };
    info.probes.push({ path: "/health", ok: false, detail: e.message });
    info.probes.push({ path: "/v1/models", ok: null, detail: "skipped" });
    info.probes.push({ path: "/props", ok: null, detail: "skipped" });
    return info;
  }

  if (healthStatus === 503) {
    info.state = "loading";
    info.error = { kind: "loading", message: "The server is still loading the model." };
    info.probes.push({ path: "/v1/models", ok: null, detail: "waiting for the model" });
    info.probes.push({ path: "/props", ok: null, detail: "waiting for the model" });
    return info;
  }
  if (healthStatus >= 500) {
    info.state = "unreachable";
    info.error = { kind: "http", message: `The server answered HTTP ${healthStatus} on /health.` };
    return info;
  }

  let authFailed = healthStatus === 401 || healthStatus === 403;
  let strataModel = false;
  let strataInstanceId: string | undefined;

  try {
    const models = await serverJson<ModelsResponse>(joinServerUrl(baseUrl, "/v1/models"), {
      apiKey: opts.apiKey,
      timeoutMs: REMOTE_PROPS_TIMEOUT_MS,
    });
    const first = models.data?.[0];
    strataModel = first?.owned_by === "strata";
    strataInstanceId = typeof first?.meta?.instanceId === "string" ? first.meta.instanceId : undefined;
    info.modelId = typeof first?.id === "string" && first.id ? first.id : undefined;
    info.modelName = first?.owned_by === "strata" && typeof first.meta?.label === "string"
      ? first.meta.label : undefined;
    info.modelSizeBytes = positive(first?.meta?.size);
    info.modelParams = positive(first?.meta?.n_params);
    info.trainedContext = positive(first?.meta?.n_ctx_train);
    info.servedContext = positive(first?.max_model_len) ?? positive(first?.meta?.n_ctx)
      ?? (strataModel ? positive(first?.meta?.context) : undefined);
    info.probes.push({
      path: "/v1/models",
      ok: !!info.modelId,
      status: 200,
      detail: info.modelId || "no model listed",
    });
  } catch (err) {
    const e = err instanceof ConnectionError ? err : new ConnectionError("network", String(err));
    if (e.kind === "auth") {
      authFailed = true;
    }
    info.probes.push({ path: "/v1/models", ok: false, status: e.status, detail: e.kind === "auth" ? "rejected (API key)" : e.message });
  }

  try {
    const props = await serverJson<PropsResponse>(joinServerUrl(baseUrl, "/props"), {
      apiKey: opts.apiKey,
      timeoutMs: REMOTE_PROPS_TIMEOUT_MS,
    });
    info.isLlamaCpp = true;
    info.slotContext = positive(props.default_generation_settings?.n_ctx);
    info.totalSlots = positive(props.total_slots);
    info.modelPath = typeof props.model_path === "string" && props.model_path ? props.model_path : undefined;
    info.modelAlias = typeof props.model_alias === "string" && props.model_alias ? props.model_alias : undefined;
    info.buildInfo = typeof props.build_info === "string" && props.build_info ? props.build_info : undefined;
    info.vision = props.modalities?.vision === true;
    info.audio = props.modalities?.audio === true;
    const bits = [
      info.buildInfo ? `llama.cpp ${shortBuild(info.buildInfo)}` : "llama.cpp",
      info.totalSlots ? `${info.totalSlots} slot${info.totalSlots === 1 ? "" : "s"}` : "",
      info.slotContext ? `${Math.round(info.slotContext / 1024)}k each` : "",
      info.vision ? "vision" : "",
    ].filter(Boolean);
    info.probes.push({ path: "/props", ok: true, status: 200, detail: bits.join(" · ") });
  } catch (err) {
    const e = err instanceof ConnectionError ? err : new ConnectionError("network", String(err));
    if (e.kind === "auth") {
      authFailed = true;
      info.probes.push({ path: "/props", ok: false, status: e.status, detail: "rejected (API key)" });
    } else {
      info.probes.push({
        path: "/props",
        ok: null,
        status: e.status,
        detail: e.status === 404 ? "not llama.cpp (no /props)" : e.message,
      });
    }
  }

  if (!authFailed && strataModel && info.modelId) {
    const needsContext = !info.slotContext && !info.servedContext;
    const needsVision = info.vision === undefined;
    if (needsContext || needsVision) {
      await attachStrataContext(info, strataInstanceId, opts.apiKey);
    }
  }

  if (!authFailed && (info.isLlamaCpp || info.modelId || (healthStatus >= 200 && healthStatus < 300))) {
    authFailed = (await attachServerLoad(info, baseUrl, opts.apiKey)) || authFailed;
  }

  if (authFailed) {
    info.state = "auth";
    info.error = { kind: "auth", message: "The server rejected the request (401). It needs an API key." };
  } else if (info.isLlamaCpp) {
    info.state = remoteLoadBusy(info) ? "busy" : "ready";
  } else if (info.modelId || healthStatus < 300) {
    info.state = "limited";
  } else {
    info.state = "unreachable";
    info.error = { kind: "http", message: "The server answered, but not like an OpenAI-compatible server." };
  }
  return info;
}

/** `b10517-abc1234` → `b10517`. */
export function shortBuild(buildInfo: string | undefined): string {
  if (!buildInfo) {
    return "";
  }
  const m = /^(b\d+)/.exec(buildInfo.trim());
  return m ? m[1] : buildInfo.trim().slice(0, 24);
}

/** Context for servers without `/props`: what `/v1/models` reports, else a safe default. */
export const LIMITED_SERVER_DEFAULT_CONTEXT = 32768;

/** Per-request context the remote allows. llama.cpp: slot size; others: served or default. */
export function remoteSlotContext(info: RemoteServerInfo | undefined): number {
  return info?.slotContext || info?.servedContext || LIMITED_SERVER_DEFAULT_CONTEXT;
}

/** Chat can be sent in these states. Busy still accepts requests; the server queues them. */
export function remoteStateUsable(state: RemoteState | undefined): boolean {
  return state === "ready" || state === "busy" || state === "limited";
}

/** Short pill label for the sidebar and status bar. */
export function remoteStateLabel(state: RemoteState | undefined): string {
  switch (state) {
    case "ready":
      return "Connected";
    case "busy":
      return "Connected · busy";
    case "limited":
      return "Connected · limited";
    case "loading":
      return "Loading model";
    case "auth":
      return "API key required";
    case "unreachable":
      return "Not reachable";
    default:
      return "Connecting…";
  }
}

const BACKOFF_MS = [5_000, 10_000, 30_000, 60_000];
/** Full probe (`/props`, `/slots`, `/metrics`) interval while health keeps answering. */
const FULL_PROBE_MS = 15_000;

/**
 * Keeps the last probe of the configured remote server. `/health` is cheap and
 * runs every call; the full probe runs on change or every 15 s. Failures back
 * off 5 / 10 / 30 / 60 s until {@link refresh} is forced.
 */
export class RemoteServerMonitor {
  private info: RemoteServerInfo | undefined;
  private inflight: Promise<RemoteServerInfo> | undefined;
  private failures = 0;
  private nextTryAt = 0;
  private fullAt = 0;
  private readonly _onDidChange = new Emitter<RemoteServerInfo>();
  /** Fires when state, model, slot context, or server load changed (not on every RTT update). */
  readonly onDidChange: Event<RemoteServerInfo> = this._onDidChange.event;

  constructor(private readonly target: () => string | undefined) {}

  /** Last probe for the current URL, if any. */
  current(): RemoteServerInfo | undefined {
    const url = this.target();
    return url && this.info?.url === url ? this.info : undefined;
  }

  /** When the next automatic retry is due (unreachable only). */
  nextRetryAt(): number | undefined {
    return this.info?.state === "unreachable" ? this.nextTryAt : undefined;
  }

  reset(): void {
    this.info = undefined;
    this.failures = 0;
    this.nextTryAt = 0;
    this.fullAt = 0;
  }

  async refresh(force = false): Promise<RemoteServerInfo | undefined> {
    const url = this.target();
    if (!url) {
      return undefined;
    }
    if (this.info && this.info.url !== url) {
      this.reset();
    }
    if (this.inflight) {
      return this.inflight;
    }
    const now = Date.now();
    if (!force && this.info && now < this.nextTryAt) {
      return this.info;
    }
    this.inflight = this.run(url, force).finally(() => {
      this.inflight = undefined;
    });
    return this.inflight;
  }

  private async run(url: string, force: boolean): Promise<RemoteServerInfo> {
    const prev = this.info;
    let next: RemoteServerInfo;
    const settled = prev?.state === "ready" || prev?.state === "busy" || prev?.state === "limited";
    const needFull = force || !prev || !settled || Date.now() - this.fullAt >= FULL_PROBE_MS;
    let probed = false;
    if (needFull) {
      next = await probeRemoteServer(url);
      this.fullAt = Date.now();
      probed = true;
    } else {
      next = await this.healthOnly(url, prev);
      if (next.state !== prev.state) {
        next = await probeRemoteServer(url);
        this.fullAt = Date.now();
        probed = true;
      }
    }
    if (probed) {
      applyVllmThroughput(prev, next);
    }
    if (next.state === "unreachable") {
      this.nextTryAt = Date.now() + BACKOFF_MS[Math.min(this.failures, BACKOFF_MS.length - 1)];
      this.failures += 1;
    } else {
      this.failures = 0;
      this.nextTryAt = 0;
    }
    if (this.target() !== url) {
      return next;
    }
    this.info = next;
    if (
      !prev ||
      prev.state !== next.state ||
      prev.modelId !== next.modelId ||
      prev.slotContext !== next.slotContext ||
      prev.servedContext !== next.servedContext ||
      prev.vision !== next.vision ||
      prev.slotsBusy !== next.slotsBusy ||
      prev.slotsTotal !== next.slotsTotal ||
      prev.requestsDeferred !== next.requestsDeferred ||
      prev.requestsProcessing !== next.requestsProcessing ||
      prev.serverGenTokPerSec !== next.serverGenTokPerSec ||
      prev.serverPromptTokPerSec !== next.serverPromptTokPerSec
    ) {
      this._onDidChange.fire(next);
    }
    return next;
  }

  private async healthOnly(url: string, prev: RemoteServerInfo): Promise<RemoteServerInfo> {
    try {
      const h = await probeHealth(url);
      if (h.status >= 200 && h.status < 300) {
        return { ...prev, checkedAt: Date.now(), rttMs: h.rttMs };
      }
      return { ...prev, state: h.status === 503 ? "loading" : "unreachable", checkedAt: Date.now() };
    } catch (err) {
      const e = err instanceof ConnectionError ? err : new ConnectionError("network", String(err));
      return { ...prev, state: "unreachable", checkedAt: Date.now(), error: { kind: e.kind, message: e.message } };
    }
  }

  dispose(): void {
    this._onDidChange.dispose();
  }
}

/** Plain-text summary for "Copy connection diagnostics". The API key is never included. */
export function formatRemoteDiagnostics(info: RemoteServerInfo | undefined, extra: { url: string; label?: string; hasApiKey: boolean }): string {
  const lines = [
    "Llama AIO remote connection",
    `URL: ${extra.url}`,
    extra.label ? `Label: ${extra.label}` : "",
    `API key: ${extra.hasApiKey ? "set (redacted)" : "none"}`,
    `State: ${info ? info.state : "not probed yet"}`,
    info?.checkedAt ? `Checked: ${new Date(info.checkedAt).toISOString()}` : "",
    info?.rttMs !== undefined ? `Round trip: ${info.rttMs} ms` : "",
    info?.error ? `Error (${info.error.kind}): ${info.error.message}` : "",
    info?.modelId ? `Model: ${info.modelId}` : "",
    info?.buildInfo ? `Build: ${info.buildInfo}` : "",
    info?.slotContext ? `Slot context: ${info.slotContext}` : "",
    info?.servedContext ? `Served context: ${info.servedContext}` : "",
    info?.totalSlots ? `Slots: ${info.totalSlots}` : "",
    info?.slotsTotal ? `Slots busy: ${info.slotsBusy ?? 0}/${info.slotsTotal}` : "",
    typeof info?.vision === "boolean" ? `Vision: ${info.vision ? "yes" : "no"}` : "",
    typeof info?.requestsDeferred === "number" ? `Queued: ${info.requestsDeferred}` : "",
    info?.serverGenTokPerSec ? `Server gen: ${info.serverGenTokPerSec} tok/s` : "",
    info?.serverPromptTokPerSec ? `Server prompt: ${info.serverPromptTokPerSec} tok/s` : "",
    info ? `llama.cpp: ${info.isLlamaCpp ? "yes" : "no /props"}` : "",
    "",
    "Probes:",
    ...(info?.probes || []).map(
      (p) => `  ${p.ok === true ? "ok  " : p.ok === false ? "FAIL" : "--  "} ${p.path}${p.status ? ` (${p.status})` : ""} ${p.detail}`
    ),
    `Llama AIO time: ${new Date().toISOString()}`,
  ];
  return lines.filter((l, i, all) => l !== "" || (i > 0 && all[i - 1] !== "")).join("\n");
}

/**
 * llama-server processes already listening on this machine, found so the
 * sidebar can offer Connect. Connecting uses remote mode. Nothing here writes
 * the launch lock or adopts the process — adoption is what lets Stop kill it.
 */
import { runStepsAsync, runStepsSync, type ExecSteps } from "./execRunner";
import { processCommandLine, processCommandLineAsync } from "./processIdentity";
import { isLoopbackHost, isWildcardHost, readBody, serverJson, serverRequest } from "./serverHttp";

export interface ListeningSocket {
  /** Bind address without brackets. May be `*`, `0.0.0.0`, or `::`. */
  bind: string;
  port: number;
  pid: number;
}

export interface DiscoveredLlamaServer {
  /** Loopback URL to pass to the existing remote connect path. */
  url: string;
  port: number;
  /** Unknown when discovered through Strata's public health endpoint. */
  pid?: number;
  /** Sidebar wording. Only llama-server may be adopted or stopped; the others are separate apps. */
  name: "llama-server" | "llamafile" | "Strata" | "BigMoeOnEdge" | "Project Maya" | "RustingStrata"
    | "Ollama" | "LM Studio" | "vLLM" | "SGLang" | "TabbyAPI" | "KoboldCpp";
}

/**
 * How a Strata process relates to the API a user can connect to.
 *
 * Strata Desktop starts several processes from one install: the API host, the
 * window, a harness, and one engine per loaded model. Only the host (or a
 * standalone engine run on its own) serves the chat API, so the other ports
 * must not be offered as separate servers.
 */
export type StrataRole = "host" | "engine" | "internal";

export function strataListenerRole(commandLine: string | undefined): StrataRole | undefined {
  const cmd = (commandLine || "").toLowerCase();
  if (!cmd) {
    return undefined;
  }
  // Electron's window/render helper and the web harness never serve the API.
  if (cmd.includes("--window-child") || /harness[/\\][^\s]*bin\.js/.test(cmd)) {
    return cmd.includes("strata") ? "internal" : undefined;
  }
  if (/host[/\\]server[/\\]index\.(mjs|js)/.test(cmd)) {
    return "host";
  }
  // The port is held by `server.py --engine strata`, or by the `strata --serve` binary.
  if (cmd.includes("--engine strata") || /(^|[^\w])strata([^\w]|$)/.test(cmd)) {
    return "engine";
  }
  return undefined;
}

/**
 * What a listening process is, for the suggestion line only.
 * This is not {@link looksLikeLlamaServer}: that one decides what Stop may kill.
 */
export function localListenerName(commandLine: string | undefined): DiscoveredLlamaServer["name"] | undefined {
  const cmd = (commandLine || "").toLowerCase();
  if (!cmd) {
    return undefined;
  }
  if (cmd.includes("llama-server")) {
    return "llama-server";
  }
  // llamafile bundles llama.cpp's server in one executable (`x.llamafile`, or `llamafile -m x.gguf`).
  if (/(^|[^\w-])[\w.-]*\.?llamafile(\.exe)?([^\w-]|$)/.test(cmd)) {
    return "llamafile";
  }
  // bmoe-desktop runs bmoe-server as a child process; only the server listens for the API.
  if (/(^|[^\w-])bmoe-server(\.exe)?([^\w-]|$)/.test(cmd)) {
    return "BigMoeOnEdge";
  }
  // The Cargo binary is `rustingstrata`. Checked before Strata, whose own pattern needs "strata" as a whole word.
  if (/(^|[^\w-])rustingstrata(\.exe)?([^\w-]|$)/.test(cmd)) {
    return "RustingStrata";
  }
  const role = strataListenerRole(commandLine);
  return role === "internal" ? undefined : role ? "Strata" : undefined;
}

/** How many suggestions the sidebar shows. */
export const MAX_DISCOVERED_SERVERS = 3;

/** Desktop host, standalone server, launcher, and source-run host defaults. */
export const STRATA_DEFAULT_PORTS = [47831, 8095, 8080, 8791] as const;

/** BigMoeOnEdge's bmoe-server default (`server/server.h`, kDefaultPort). */
export const BIGMOEONEDGE_DEFAULT_PORTS = [8765] as const;

/**
 * Identify BigMoeOnEdge's bmoe-server without process access. Its /api/info
 * reports planner and expert-overlap flags no other local server has. When it
 * was started with an API key, /api/info answers 401 with its own wording.
 */
export async function probeLocalBigMoeOnEdgeServer(url: string): Promise<boolean> {
  try {
    if (!isLoopbackHost(new URL(url).hostname)) {
      return false;
    }
    // Never send the managed server's API key to another program.
    const res = await serverRequest(`${url}/api/info`, { apiKey: "", timeoutMs: 750, signal: AbortSignal.timeout(750) });
    const body = await readBody(res, 64 * 1024);
    if (res.statusCode === 401) {
      return /API key is required: send it as Authorization: Bearer <key> \(or X-Api-Key: <key>\)/i.test(body);
    }
    const info = JSON.parse(body);
    return res.statusCode === 200 && !!info && typeof info === "object" && typeof info.version === "string"
      && typeof info.planner_available === "boolean" && typeof info.overlap_available === "boolean"
      && typeof info.state === "string";
  } catch {
    return false;
  }
}

/** Project Maya's launcher (maya.py) serves its API on 8080; serve/server.py alone defaults to 8095. */
export const PROJECT_MAYA_DEFAULT_PORTS = [8080, 8095] as const;

/**
 * Identify Project Maya. Its public /health carries a `model_label` and the
 * refusal-ablation state (`ablate`) next to its version; it does not name a
 * `service`, which is how it differs from Strata and PolyStrata.
 */
export async function probeLocalProjectMayaServer(url: string): Promise<boolean> {
  try {
    if (!isLoopbackHost(new URL(url).hostname)) return false;
    const health = await serverJson<Record<string, unknown>>(`${url}/health`,
      { apiKey: "", timeoutMs: 750, signal: AbortSignal.timeout(750) });
    return !!health && typeof health.version === "string" && "model_label" in health && "ablate" in health
      && "history_thinking" in health && !("service" in health);
  } catch {
    return false;
  }
}

/** RustingStrata's default listen address (src/main.rs: 127.0.0.1:8080). */
export const RUSTINGSTRATA_DEFAULT_PORTS = [8080] as const;

/**
 * Identify RustingStrata. Its /health needs no key: with a model loaded it
 * reports exactly status, waiting and slots; without one it answers 503 with
 * "no model" or "loading".
 */
export async function probeLocalRustingStrataServer(url: string): Promise<boolean> {
  try {
    if (!isLoopbackHost(new URL(url).hostname)) return false;
    const res = await serverRequest(`${url}/health`, { apiKey: "", timeoutMs: 750, signal: AbortSignal.timeout(750) });
    const health = JSON.parse(await readBody(res, 16 * 1024));
    if (!health || typeof health !== "object" || Array.isArray(health)) return false;
    const keys = Object.keys(health).sort().join(",");
    if (res.statusCode === 200) {
      return keys === "slots,status,waiting" && health.status === "ok"
        && Number.isInteger(health.waiting) && Number.isInteger(health.slots);
    }
    return res.statusCode === 503 && keys === "status" && (health.status === "no model" || health.status === "loading");
  } catch {
    return false;
  }
}

const PROBE = { apiKey: "", timeoutMs: 750 } as const;
const probeOptions = () => ({ ...PROBE, signal: AbortSignal.timeout(750) });

/** GET `path` on a loopback URL without any key. Undefined for another host or no answer. */
async function probeGet(url: string, path: string): Promise<{ status: number; body: string } | undefined> {
  try {
    if (!isLoopbackHost(new URL(url).hostname)) return undefined;
    const res = await serverRequest(`${url}${path}`, probeOptions());
    return { status: res.statusCode || 0, body: await readBody(res, 256 * 1024) };
  } catch {
    return undefined;
  }
}

function parseJson(body: string | undefined): any {
  try { return JSON.parse(body || ""); } catch { return undefined; }
}

/** Ollama (server/routes.go): `GET /` answers the plain text "Ollama is running"; default port 11434. */
export async function probeLocalOllamaServer(url: string): Promise<boolean> {
  const root = await probeGet(url, "/");
  if (root?.status !== 200 || root.body.trim() !== "Ollama is running") return false;
  const version = parseJson((await probeGet(url, "/api/version"))?.body);
  return typeof version?.version === "string";
}

/**
 * LM Studio's native REST API lists models with a publisher and LM Studio's own
 * fields: /api/v1/models (0.4.0 and newer) or /api/v0/models (0.3.6 and newer).
 * Its docs use port 1234. A server with authentication switched on is not identified.
 */
export async function probeLocalLmStudioServer(url: string): Promise<boolean> {
  const v1 = parseJson((await probeGet(url, "/api/v1/models"))?.body);
  if (Array.isArray(v1?.models)) {
    return v1.models.length === 0 || v1.models.some((m: any) =>
      typeof m?.publisher === "string" && typeof m?.key === "string" && Array.isArray(m?.loaded_instances));
  }
  const v0 = parseJson((await probeGet(url, "/api/v0/models"))?.body);
  return v0?.object === "list" && Array.isArray(v0.data) && v0.data.some((m: any) =>
    typeof m?.publisher === "string" && typeof m?.compatibility_type === "string" && typeof m?.state === "string");
}

/** True when GET /v1/models lists at least one model owned by `owner` (vLLM, SGLang, TabbyAPI set their own name). */
async function modelsOwnedBy(url: string, owner: string): Promise<boolean> {
  const list = parseJson((await probeGet(url, "/v1/models"))?.body);
  return Array.isArray(list?.data) && list.data.some((m: any) => m?.owned_by === owner);
}

/** vLLM (entrypoints/serve/engine/protocol.py): model cards are owned_by "vllm". */
export const probeLocalVllmServer = (url: string) => modelsOwnedBy(url, "vllm");
/** SGLang (srt/entrypoints/openai/protocol.py): model cards are owned_by "sglang". */
export const probeLocalSglangServer = (url: string) => modelsOwnedBy(url, "sglang");
/** TabbyAPI (endpoints/core/types/model.py): model cards are owned_by "tabbyAPI"; default port 5000. */
export const probeLocalTabbyApiServer = (url: string) => modelsOwnedBy(url, "tabbyAPI");

/** KoboldCpp (koboldcpp.py): /api/extra/version answers result "KoboldCpp"; default port 5001. */
export async function probeLocalKoboldCppServer(url: string): Promise<boolean> {
  const info = parseJson((await probeGet(url, "/api/extra/version"))?.body);
  return info?.result === "KoboldCpp" && typeof info.version === "string";
}

/** Identify Strata without process access (notably python.exe on Windows). */
export async function probeLocalStrataServer(url: string): Promise<boolean> {
  try {
    if (!isLoopbackHost(new URL(url).hostname)) {
      return false;
    }
    const options = {
      // Health endpoints are public. Never send the managed server's API key.
      apiKey: "",
      timeoutMs: 750,
      signal: AbortSignal.timeout(750),
    };
    try {
      const health = await serverJson<{ service?: string }>(`${url}/health`, options);
      if (health?.service === "strata") {
        return true;
      }
    } catch {
      // The Desktop host serves the UI here; its health route is /api/health.
    }
    const res = await serverRequest(`${url}/api/health`, options);
    const body = await readBody(res, 64 * 1024);
    const health = JSON.parse(body);
    const headers = String(res.headers["access-control-allow-headers"] || "").toLowerCase().split(/\s*,\s*/);
    return res.statusCode === 200 && headers.includes("x-strata-client")
      && health?.ok === true && typeof health.backend?.kind === "string"
      && !!health.engine && typeof health.engine === "object" && !Array.isArray(health.engine);
  } catch {
    return false;
  }
}

/** True when a socket bound here accepts a connection to loopback. */
export function isLocalhostReachableBind(bind: string): boolean {
  const h = bind.trim().replace(/^\[|\]$/g, "").replace(/%.*$/, "").toLowerCase();
  if (!h || h === "*") {
    return true;
  }
  return isLoopbackHost(h) || isWildcardHost(h);
}

/** URL the sidebar connects to. Wildcard binds are reached at loopback. */
export function connectUrlForListener(bind: string, port: number): string | undefined {
  if (!Number.isInteger(port) || port <= 0 || port > 65535 || !isLocalhostReachableBind(bind)) {
    return undefined;
  }
  const h = bind.trim().replace(/^\[|\]$/g, "").replace(/%.*$/, "").toLowerCase();
  if (!h || h === "*" || h === "0.0.0.0" || h === "::") {
    return `http://127.0.0.1:${port}`;
  }
  if (h === "::1") {
    return `http://[::1]:${port}`;
  }
  return h.includes(":") ? `http://[${h}]:${port}` : `http://${h}:${port}`;
}

function parseHostPort(token: string): { bind: string; port: number } | undefined {
  const match = /^(?:\[([^\]]+)\]|([^:\s]+)):(\d+)$/.exec(token.trim());
  if (!match) {
    return undefined;
  }
  const port = Number(match[3]);
  if (!Number.isInteger(port)) {
    return undefined;
  }
  return { bind: (match[1] || match[2] || "").trim(), port };
}

/** `ss -ltnp` text. Lines without a pid are dropped. */
export function parseSsListeners(text: string): ListeningSocket[] {
  const out: ListeningSocket[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (!/^\s*LISTEN\b/i.test(line)) {
      continue;
    }
    const pidMatch = /pid=(\d+)/.exec(line);
    if (!pidMatch) {
      continue;
    }
    const local = parseHostPort(line.trim().split(/\s+/)[3] || "");
    if (!local) {
      continue;
    }
    out.push({ bind: local.bind, port: local.port, pid: Number(pidMatch[1]) });
  }
  return out;
}

/** `lsof -nP -iTCP -sTCP:LISTEN` text. The command column is truncated, so the pid is what matters. */
export function parseLsofListeners(text: string): ListeningSocket[] {
  const out: ListeningSocket[] = [];
  for (const line of text.split(/\r?\n/)) {
    const addr = /TCP\s+(\S+)\s+\(LISTEN\)/.exec(line);
    const pid = Number(line.trim().split(/\s+/)[1]);
    if (!addr || !Number.isInteger(pid) || pid <= 0) {
      continue;
    }
    const local = parseHostPort(addr[1]);
    if (!local) {
      continue;
    }
    out.push({ bind: local.bind, port: local.port, pid });
  }
  return out;
}

/** Windows `netstat -ano -p tcp` text. */
export function parseNetstatListeners(text: string): ListeningSocket[] {
  const out: ListeningSocket[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (!/LISTENING/i.test(line)) {
      continue;
    }
    const parts = line.trim().split(/\s+/);
    if (parts.length < 5) {
      continue;
    }
    const local = parseHostPort(parts[1] || "");
    const pid = Number(parts[parts.length - 1]);
    if (!local || !Number.isInteger(pid) || pid <= 0) {
      continue;
    }
    out.push({ bind: local.bind, port: local.port, pid });
  }
  return out;
}

function preferIpv4(current: DiscoveredLlamaServer, next: DiscoveredLlamaServer): DiscoveredLlamaServer {
  const currentV4 = current.url.startsWith("http://127.");
  const nextV4 = next.url.startsWith("http://127.");
  return !currentV4 && nextV4 ? next : current;
}

/**
 * One API per Strata install. The desktop host serves it; its window and harness
 * never do, and its per-model engines only serve the desktop app, which reaches
 * them through the host. So internal components drop out, and an engine yields
 * to a host discovered on the same machine.
 */
function dropStrataInternals(
  candidates: readonly { server: DiscoveredLlamaServer; role: StrataRole }[]
): DiscoveredLlamaServer[] {
  const usable = candidates.filter((c) => c.role !== "internal");
  const hosts = usable.filter((c) => c.role === "host");
  return (hosts.length ? hosts : usable).map((c) => c.server);
}

/**
 * Keep loopback-reachable sockets whose command line is llama-server.
 * One URL per pid and port. IPv4 wins over IPv6 for the same process.
 */
export function discoverLoopbackLlamaServers(opts: {
  sockets: ListeningSocket[] | undefined;
  commandLine: (pid: number) => string | undefined;
  skipPids?: readonly number[];
}): DiscoveredLlamaServer[] {
  if (!opts.sockets) {
    return [];
  }
  const skip = new Set(opts.skipPids ?? []);
  const byProcess = new Map<string, DiscoveredLlamaServer>();
  const strata: { server: DiscoveredLlamaServer; role: StrataRole }[] = [];
  for (const sock of opts.sockets) {
    if (skip.has(sock.pid)) {
      continue;
    }
    const url = connectUrlForListener(sock.bind, sock.port);
    if (!url) {
      continue;
    }
    const key = `${sock.pid}:${sock.port}`;
    const prev = byProcess.get(key);
    if (prev) {
      byProcess.set(key, preferIpv4(prev, { ...prev, url }));
      continue;
    }
    const commandLine = opts.commandLine(sock.pid);
    const name = localListenerName(commandLine);
    if (!name) {
      continue;
    }
    const server = { url, port: sock.port, pid: sock.pid, name };
    if (name === "Strata") {
      strata.push({ server, role: strataListenerRole(commandLine) ?? "engine" });
      continue;
    }
    byProcess.set(key, server);
  }
  for (const server of dropStrataInternals(strata)) {
    byProcess.set(`${server.pid}:${server.port}`, server);
  }
  return [...byProcess.values()]
    .sort((a, b) => a.port - b.port || (a.pid ?? 0) - (b.pid ?? 0))
    .slice(0, MAX_DISCOVERED_SERVERS);
}

function* listListeningSockets(): ExecSteps<ListeningSocket[] | undefined> {
  if (process.platform === "win32") {
    const res = yield { command: "netstat", args: ["-ano", "-p", "tcp"], timeoutMs: 8000 };
    return res.ok ? parseNetstatListeners(res.stdout) : undefined;
  }
  if (process.platform === "darwin") {
    const res = yield { command: "lsof", args: ["-nP", "-iTCP", "-sTCP:LISTEN"], timeoutMs: 8000 };
    return res.ok ? parseLsofListeners(res.stdout) : undefined;
  }
  const res = yield { command: "ss", args: ["-ltnp"], timeoutMs: 4000 };
  return res.ok ? parseSsListeners(res.stdout) : undefined;
}

/**
 * llama-server processes on this machine that accept a loopback connection.
 * `skipPids` is the server this extension already owns, so it is not offered twice.
 * Tooling failure returns an empty list.
 */
export function discoverLoopbackLlamaServersOnThisMachine(skipPids: readonly number[] = []): DiscoveredLlamaServer[] {
  return discoverLoopbackLlamaServers({
    sockets: runStepsSync(listListeningSockets()),
    commandLine: processCommandLine,
    skipPids,
  });
}

/**
 * Servers recognised by their own public replies on their default ports. Each
 * probe checks a reply only that program gives, so another server on a shared
 * port such as 8080 is not mistaken for it.
 */
const OTHER_HTTP_PROBES: ReadonlyArray<{
  name: DiscoveredLlamaServer["name"];
  ports: readonly number[];
  probe: (url: string) => Promise<boolean>;
}> = [
  { name: "BigMoeOnEdge", ports: BIGMOEONEDGE_DEFAULT_PORTS, probe: probeLocalBigMoeOnEdgeServer },
  { name: "Project Maya", ports: PROJECT_MAYA_DEFAULT_PORTS, probe: probeLocalProjectMayaServer },
  { name: "RustingStrata", ports: RUSTINGSTRATA_DEFAULT_PORTS, probe: probeLocalRustingStrataServer },
  { name: "Ollama", ports: [11434], probe: probeLocalOllamaServer },
  { name: "LM Studio", ports: [1234], probe: probeLocalLmStudioServer },
  { name: "vLLM", ports: [8000], probe: probeLocalVllmServer },
  { name: "SGLang", ports: [30000], probe: probeLocalSglangServer },
  { name: "TabbyAPI", ports: [5000], probe: probeLocalTabbyApiServer },
  { name: "KoboldCpp", ports: [5001], probe: probeLocalKoboldCppServer },
];

/** Process discovery plus public health probes, without blocking the event loop. */
export async function discoverLoopbackLlamaServersOnThisMachineAsync(
  skipPids: readonly number[] = []
): Promise<DiscoveredLlamaServer[]> {
  const [sockets, strata, others] = await Promise.all([
    runStepsAsync(listListeningSockets()),
    Promise.all(STRATA_DEFAULT_PORTS.map(async (port) => {
      const url = `http://127.0.0.1:${port}`;
      return await probeLocalStrataServer(url) ? { url, port, name: "Strata" as const } : undefined;
    })),
    Promise.all(OTHER_HTTP_PROBES.flatMap(({ name, ports, probe }) => ports.map(async (port) => {
      const url = `http://127.0.0.1:${port}`;
      return await probe(url) ? { url, port, name } : undefined;
    }))),
  ]);
  const commandLines = new Map<number, string | undefined>();
  for (const pid of new Set((sockets || []).map((sock) => sock.pid))) {
    if (!skipPids.includes(pid)) {
      commandLines.set(pid, await processCommandLineAsync(pid));
    }
  }
  const found = discoverLoopbackLlamaServers({
    sockets,
    commandLine: (pid) => commandLines.get(pid),
    skipPids,
  });
  for (const server of strata) {
    if (server && !found.some((s) => s.name === "Strata" && s.port === server.port)) {
      // The HTTP probe cannot tell the host from the desktop app's own window,
      // harness, or engine ports, so a known non-host listener stays hidden.
      const listeners = (sockets ?? []).filter((s) => s.port === server.port);
      if (listeners.some((s) => skipPids.includes(s.pid))) {
        continue;
      }
      const role = listeners.length
        ? strataListenerRole(commandLines.get(listeners[0].pid)) ?? "engine"
        : undefined;
      if (role === "internal" || (role === "engine" && found.some((s) => s.name === "Strata"))) {
        continue;
      }
      found.push(server);
    }
  }
  // HTTP probes find these servers where process names are hidden (another user, a sandbox).
  // A port is listed once: what the process scan or Strata's probe found there wins.
  for (const server of others) {
    if (!server || found.some((s) => s.port === server.port)) continue;
    if ((sockets ?? []).some((s) => s.port === server.port && skipPids.includes(s.pid))) continue;
    found.push(server);
  }
  return found.slice(0, MAX_DISCOVERED_SERVERS);
}

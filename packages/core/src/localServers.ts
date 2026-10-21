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
  /** Sidebar wording. Strata is not llama-server and must not be adopted or stopped. */
  name: "llama-server" | "Strata";
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
  const role = strataListenerRole(commandLine);
  return role === "internal" ? undefined : role ? "Strata" : undefined;
}

/** How many suggestions the sidebar shows. */
export const MAX_DISCOVERED_SERVERS = 3;

/** Desktop host, standalone server, launcher, and source-run host defaults. */
export const STRATA_DEFAULT_PORTS = [47831, 8095, 8080, 8791] as const;

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

/** Process discovery plus public Strata health probes, without blocking the event loop. */
export async function discoverLoopbackLlamaServersOnThisMachineAsync(
  skipPids: readonly number[] = []
): Promise<DiscoveredLlamaServer[]> {
  const [sockets, strata] = await Promise.all([
    runStepsAsync(listListeningSockets()),
    Promise.all(STRATA_DEFAULT_PORTS.map(async (port) => {
      const url = `http://127.0.0.1:${port}`;
      return await probeLocalStrataServer(url) ? { url, port, name: "Strata" as const } : undefined;
    })),
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
  return found.slice(0, MAX_DISCOVERED_SERVERS);
}

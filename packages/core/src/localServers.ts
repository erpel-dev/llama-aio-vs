/**
 * llama-server processes already listening on this machine, found so the
 * sidebar can offer Connect. Connecting uses remote mode. Nothing here writes
 * the launch lock or adopts the process — adoption is what lets Stop kill it.
 */
import { execFileSync } from "child_process";
import { processCommandLine } from "./processIdentity";
import { isLoopbackHost, isWildcardHost } from "./serverHttp";

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
  pid: number;
  /** Sidebar wording. Strata is not llama-server and must not be adopted or stopped. */
  name: "llama-server" | "Strata";
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
  // The port is held by `server.py --engine strata`, or by the `strata --serve` binary.
  if (cmd.includes("--engine strata") || /(^|[^\w])strata([^\w]|$)/.test(cmd)) {
    return "Strata";
  }
  return undefined;
}

/** How many suggestions the sidebar shows. */
export const MAX_DISCOVERED_SERVERS = 3;

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
    const name = localListenerName(opts.commandLine(sock.pid));
    if (!name) {
      continue;
    }
    byProcess.set(key, { url, port: sock.port, pid: sock.pid, name });
  }
  return [...byProcess.values()]
    .sort((a, b) => a.port - b.port || a.pid - b.pid)
    .slice(0, MAX_DISCOVERED_SERVERS);
}

function listListeningSockets(): ListeningSocket[] | undefined {
  try {
    if (process.platform === "win32") {
      const out = execFileSync("netstat", ["-ano", "-p", "tcp"], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
        windowsHide: true,
        timeout: 8000,
      });
      return parseNetstatListeners(out);
    }
    if (process.platform === "darwin") {
      const out = execFileSync("lsof", ["-nP", "-iTCP", "-sTCP:LISTEN"], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
        timeout: 8000,
      });
      return parseLsofListeners(out);
    }
    const out = execFileSync("ss", ["-ltnp"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 4000,
    });
    return parseSsListeners(out);
  } catch {
    return undefined;
  }
}

/**
 * llama-server processes on this machine that accept a loopback connection.
 * `skipPids` is the server this extension already owns, so it is not offered twice.
 * Tooling failure returns an empty list.
 */
export function discoverLoopbackLlamaServersOnThisMachine(skipPids: readonly number[] = []): DiscoveredLlamaServer[] {
  return discoverLoopbackLlamaServers({
    sockets: listListeningSockets(),
    commandLine: processCommandLine,
    skipPids,
  });
}

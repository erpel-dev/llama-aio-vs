/**
 * One transport for every request to a llama-server, local or remote:
 * http or https by URL, path prefixes kept (reverse proxies at `/llama/`),
 * `Authorization: Bearer` from the key registered for the URL's origin, and
 * failures reported as a typed {@link ConnectionError}.
 */
import * as http from "http";
import * as https from "https";
import * as net from "net";

export type ConnectionErrorKind =
  | "refused"
  | "timeout"
  | "dns"
  | "tls"
  | "auth"
  | "loading"
  | "http"
  | "network"
  | "cancelled";

export class ConnectionError extends Error {
  constructor(
    readonly kind: ConnectionErrorKind,
    message: string,
    readonly status?: number,
    /** Response body, for HTTP errors. */
    readonly body?: string,
    readonly code?: string
  ) {
    super(message);
    this.name = "ConnectionError";
  }
}

type ApiKeyProvider = (origin: string) => string | undefined;
let apiKeyProvider: ApiKeyProvider = () => undefined;

/**
 * Register where API keys come from (VS Code SecretStorage cache). Keys are
 * looked up per origin so a key is never sent to a different server.
 */
export function setServerApiKeyProvider(provider: ApiKeyProvider | undefined): void {
  apiKeyProvider = provider ?? (() => undefined);
}

export function apiKeyForUrl(url: string): string | undefined {
  try {
    return apiKeyProvider(new URL(url).origin) || undefined;
  } catch {
    return undefined;
  }
}

/** `base` + `path`, keeping any path prefix on `base` (`https://h/llama` + `/props`). */
export function joinServerUrl(base: string, path: string): string {
  const u = new URL(base);
  const prefix = u.pathname.replace(/\/+$/, "");
  const suffix = path.startsWith("/") ? path : `/${path}`;
  u.pathname = `${prefix}${suffix}`;
  return u.toString();
}

/**
 * Validate and canonicalize a user-entered server URL. A bare `host:port`
 * gets `http://`. Query strings and fragments are dropped; a path prefix stays.
 */
export function normalizeServerUrl(raw: string): string {
  let text = (raw || "").trim();
  if (!text) {
    throw new Error("Enter the server URL, for example http://192.168.0.5:8080.");
  }
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(text)) {
    text = `http://${text}`;
  }
  let u: URL;
  try {
    u = new URL(text);
  } catch {
    throw new Error(`“${raw.trim()}” is not a valid URL.`);
  }
  if (u.protocol !== "http:" && u.protocol !== "https:") {
    throw new Error(`Only http:// and https:// URLs are supported (got ${u.protocol}//).`);
  }
  if (!u.hostname) {
    throw new Error(`“${raw.trim()}” has no host name.`);
  }
  if (u.username || u.password) {
    throw new Error("Put the API key in the API key field, not in the URL.");
  }
  u.search = "";
  u.hash = "";
  const pathPrefix = u.pathname.replace(/\/+$/, "");
  return `${u.protocol}//${u.host}${pathPrefix}`;
}

/** Host part of a URL without IPv6 brackets. */
export function urlHostname(url: string): string {
  try {
    return new URL(url).hostname.replace(/^\[|\]$/g, "");
  } catch {
    return "";
  }
}

export function isLoopbackHost(host: string): boolean {
  const h = host.trim().toLowerCase().replace(/^\[|\]$/g, "");
  return h === "localhost" || h.endsWith(".localhost") || h === "::1" || /^127\./.test(h);
}

/** Bind-only addresses that mean "every interface", not a place to connect to. */
export function isWildcardHost(host: string): boolean {
  const h = host.trim().replace(/^\[|\]$/g, "");
  return h === "" || h === "0.0.0.0" || h === "::";
}

function ipv4Private(ip: string): boolean {
  const [a, b] = ip.split(".").map(Number);
  return (
    a === 10 ||
    a === 127 ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168) ||
    (a === 169 && b === 254) ||
    // CGNAT, used by Tailscale.
    (a === 100 && b >= 64 && b <= 127)
  );
}

/**
 * True when `host` is loopback, a private / link-local / CGNAT address, or a
 * name that only resolves inside a private network. Anything else counts as
 * public for the plain-http warning.
 */
export function isPrivateNetworkHost(host: string): boolean {
  const h = host.trim().toLowerCase().replace(/^\[|\]$/g, "");
  if (!h) {
    return false;
  }
  if (isLoopbackHost(h)) {
    return true;
  }
  const family = net.isIP(h);
  if (family === 4) {
    return ipv4Private(h);
  }
  if (family === 6) {
    const mapped = /^::ffff:(\d+\.\d+\.\d+\.\d+)$/.exec(h);
    if (mapped) {
      return ipv4Private(mapped[1]);
    }
    return /^f[cd]/.test(h) || /^fe[89ab]/.test(h);
  }
  if (!h.includes(".")) {
    return true;
  }
  return /\.(local|lan|home|internal|intranet|corp|home\.arpa|ts\.net)$/.test(h);
}

/** Plain http to a server outside the private network: prompts would cross the internet unencrypted. */
export function isPublicPlainHttp(url: string): boolean {
  try {
    const u = new URL(url);
    return u.protocol === "http:" && !isPrivateNetworkHost(u.hostname);
  } catch {
    return false;
  }
}

export interface ServerRequestOptions {
  method?: string;
  /** JSON-serialized when not a string. */
  body?: unknown;
  headers?: Record<string, string>;
  /** Socket inactivity timeout. 0 disables it (long silent prompt processing). */
  timeoutMs?: number;
  /** Overrides the registered key (connection tests before the key is saved). */
  apiKey?: string;
  signal?: AbortSignal;
}

function classifyNetworkError(err: unknown, url: string): ConnectionError {
  if (err instanceof ConnectionError) {
    return err;
  }
  const e = err as NodeJS.ErrnoException;
  const code = e?.code || "";
  const host = urlHostname(url) || url;
  const msg = e?.message || String(err);
  if (code === "ECONNREFUSED") {
    return new ConnectionError("refused", `Connection refused by ${host}.`, undefined, undefined, code);
  }
  if (code === "ENOTFOUND" || code === "EAI_AGAIN" || code === "EAI_NONAME") {
    return new ConnectionError("dns", `Cannot resolve host name ${host}.`, undefined, undefined, code);
  }
  if (code === "ETIMEDOUT" || code === "ESOCKETTIMEDOUT") {
    return new ConnectionError("timeout", `Connection to ${host} timed out.`, undefined, undefined, code);
  }
  if (
    /CERT|SSL|TLS|SELF_SIGNED|UNABLE_TO_VERIFY|ERR_TLS|EPROTO/i.test(code) ||
    /certificate|ssl|tls/i.test(msg)
  ) {
    return new ConnectionError("tls", `TLS error talking to ${host}: ${msg}`, undefined, undefined, code);
  }
  if (code === "EHOSTUNREACH" || code === "ENETUNREACH") {
    return new ConnectionError("timeout", `No route to ${host}.`, undefined, undefined, code);
  }
  return new ConnectionError("network", `Cannot reach ${host}: ${msg}`, undefined, undefined, code);
}

function httpErrorFor(status: number, body: string): ConnectionError {
  const snippet = body.replace(/\s+/g, " ").trim().slice(0, 400);
  if (status === 401 || status === 403) {
    return new ConnectionError("auth", `HTTP ${status}: the server rejected the API key.`, status, body);
  }
  if (status === 503 && /loading/i.test(snippet)) {
    return new ConnectionError("loading", "The server is still loading the model.", status, body);
  }
  return new ConnectionError("http", snippet ? `HTTP ${status}: ${snippet}` : `HTTP ${status}`, status, body);
}

/**
 * Low-level request. Resolves with the response (any status) so callers that
 * need the raw status (health probes) or a stream (SSE) can use it directly.
 */
export function serverRequest(url: string, options: ServerRequestOptions = {}): Promise<http.IncomingMessage> {
  return new Promise((resolve, reject) => {
    let u: URL;
    try {
      u = new URL(url);
    } catch {
      reject(new ConnectionError("network", `Invalid server URL: ${url}`));
      return;
    }
    const lib = u.protocol === "https:" ? https : http;
    const payload =
      options.body === undefined
        ? undefined
        : typeof options.body === "string"
          ? options.body
          : JSON.stringify(options.body);
    const key = options.apiKey ?? apiKeyForUrl(url);
    const headers: Record<string, string | number> = {
      ...(payload !== undefined
        ? { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(payload) }
        : {}),
      ...(key ? { Authorization: `Bearer ${key}` } : {}),
      ...(options.headers || {}),
    };
    let settled = false;
    const fail = (err: unknown) => {
      if (!settled) {
        settled = true;
        reject(classifyNetworkError(err, url));
      }
    };
    const req = lib.request(
      {
        protocol: u.protocol,
        hostname: u.hostname.replace(/^\[|\]$/g, ""),
        port: u.port || undefined,
        path: `${u.pathname}${u.search}`,
        method: options.method || (payload !== undefined ? "POST" : "GET"),
        headers,
      },
      (res) => {
        settled = true;
        resolve(res);
      }
    );
    const timeoutMs = options.timeoutMs ?? 30_000;
    if (timeoutMs > 0) {
      req.setTimeout(timeoutMs, () => {
        req.destroy(
          new ConnectionError("timeout", `No answer from ${u.host} after ${Math.round(timeoutMs / 1000)} s.`)
        );
      });
    }
    req.on("error", fail);
    const signal = options.signal;
    if (signal) {
      if (signal.aborted) {
        req.destroy();
        fail(new ConnectionError("cancelled", "Cancelled"));
        return;
      }
      signal.addEventListener(
        "abort",
        () => {
          req.destroy();
          fail(new ConnectionError("cancelled", "Cancelled"));
        },
        { once: true }
      );
    }
    req.end(payload);
  });
}

export function readBody(res: http.IncomingMessage, maxBytes = 4 * 1024 * 1024): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    res.on("data", (c: Buffer) => {
      size += c.length;
      if (size <= maxBytes) {
        chunks.push(c);
      }
    });
    res.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    res.on("error", reject);
  });
}

/** GET/POST returning parsed JSON. Non-2xx statuses reject with a {@link ConnectionError}. */
export async function serverJson<T>(url: string, options: ServerRequestOptions = {}): Promise<T> {
  const res = await serverRequest(url, options);
  const text = await readBody(res);
  const status = res.statusCode || 0;
  if (status < 200 || status >= 300) {
    throw httpErrorFor(status, text);
  }
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ConnectionError("http", `Expected JSON from ${new URL(url).pathname}, got: ${text.slice(0, 120)}`, status);
  }
}

/** Open a streaming response; non-2xx statuses reject with the body attached. */
export async function serverStream(url: string, options: ServerRequestOptions = {}): Promise<http.IncomingMessage> {
  const res = await serverRequest(url, options);
  const status = res.statusCode || 0;
  if (status < 200 || status >= 300) {
    const text = await readBody(res);
    throw httpErrorFor(status, text);
  }
  return res;
}

/** One targeted troubleshooting line per error kind, for the sidebar and chat errors. */
export function connectionHint(kind: ConnectionErrorKind | undefined, url: string): string {
  let port = "";
  try {
    const u = new URL(url);
    port = u.port || (u.protocol === "https:" ? "443" : "80");
  } catch {
    // keep generic
  }
  switch (kind) {
    case "refused":
      return `Nothing is listening on port ${port}. Is llama-server running there with --host 0.0.0.0 --port ${port}?`;
    case "timeout":
      return "No answer in time. Check the firewall, VPN, or that the machine is awake.";
    case "dns":
      return "The host name does not resolve. Check the spelling, or use the IP address.";
    case "tls":
      return "The certificate is not trusted. For a self-signed certificate, point NODE_EXTRA_CA_CERTS at its CA file and restart VS Code.";
    case "auth":
      return "The server was started with --api-key. Set the matching API key.";
    case "loading":
      return "The server is still loading the model. Large models can take a few minutes.";
    case "http":
      return "The server answered with an error. Check that the URL points at llama-server (or its proxy).";
    default:
      return "Check that the server is running and reachable from this machine.";
  }
}
